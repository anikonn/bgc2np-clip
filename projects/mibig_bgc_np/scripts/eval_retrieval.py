from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import build_interactions, load_pair_table
from projects.mibig_bgc_np.eval.retrieval_class_metrics import (
    evaluate_bgc_class_retrieval,
    save_bgc_class_retrieval_plots,
)
from projects.mibig_bgc_np.featurization import build_molecule_encoder
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings
from mibig_clip.eval.retrieval import evaluate_global_retrieval_multi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIBiG BGC-compound retrieval metrics.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument("--cv_fold", type=int, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--npatlas_path", type=str, default="data/NPAtlas_download_2024_09.tsv")
    parser.add_argument("--npatlas_n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--no_plots", action="store_true", help="Disable BGC-class retrieval ROC/confusion PNGs.")
    return parser.parse_args()


def _load_model(ckpt_path: str | Path, device: torch.device) -> tuple[DualEncoderCLIP, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = DualEncoderCLIP(
        bgc_input_dim=ckpt["bgc_input_dim"],
        compound_input_dim=ckpt["compound_input_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
        bgc_aggregation=str(cfg["model"].get("bgc_aggregation", "prepooled")),
        bgc_aggregation_config=cfg["model"].get("bgc_aggregation_config", {}),
        projection_head=str(cfg["model"].get("projection_head", "mlp_gelu")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _save_embedding_meta(path: Path, bgc_ids: list[str], compound_ids: list[str], split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "modality", "split"])
        writer.writeheader()
        for bgc_id in bgc_ids:
            writer.writerow({"id": bgc_id, "modality": "bgc", "split": split})
        for compound_id in compound_ids:
            writer.writerow({"id": compound_id, "modality": "compound", "split": split})


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required for NPAtlas retrieval evaluation. Please install rdkit."
        ) from exc
    return Chem


def _safe_canonical_smiles(smiles: str | float | None, chem_module: Any) -> str | None:
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = chem_module.MolFromSmiles(text)
    if mol is None:
        return None
    return str(chem_module.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _safe_inchikey(smiles: str | float | None, chem_module: Any) -> str | None:
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = chem_module.MolFromSmiles(text)
    if mol is None:
        return None
    return str(chem_module.MolToInchiKey(mol))


def _prepare_npatlas_candidates(npatlas_path: str | Path, chem_module: Any) -> pd.DataFrame:
    npatlas_df = pd.read_csv(npatlas_path, sep="\t")
    if "compound_smiles" not in npatlas_df.columns or "compound_inchikey" not in npatlas_df.columns:
        raise ValueError("NPAtlas table must include compound_smiles and compound_inchikey columns.")

    npatlas_df = npatlas_df.copy()
    npatlas_df["compound_inchikey"] = npatlas_df["compound_inchikey"].fillna("").astype(str).str.strip()
    npatlas_df["canonical_smiles"] = [
        _safe_canonical_smiles(smiles, chem_module) for smiles in npatlas_df["compound_smiles"].tolist()
    ]
    npatlas_df = npatlas_df.dropna(subset=["canonical_smiles"]).copy().reset_index(drop=True)
    # candidate_idx is used later as a stable positional identifier into this filtered table.
    npatlas_df["candidate_idx"] = np.arange(len(npatlas_df))
    return npatlas_df


def _prepare_test_query_table(
    data_dir: str | Path,
    interactions: pd.DataFrame,
    split: str,
    chem_module: Any,
) -> pd.DataFrame:
    pair_df = load_pair_table(data_dir).copy()
    split_df = (
        interactions[["bgc_id", "split"]]
        .drop_duplicates(subset=["bgc_id"])
        .copy()
    )
    split_df["bgc_id"] = split_df["bgc_id"].astype(str)
    split_df["split"] = split_df["split"].astype(str).str.lower()
    pair_df["bgc_id"] = pair_df["bgc_id"].astype(str)
    pair_df = pair_df.merge(split_df, on="bgc_id", how="inner")
    pair_df = pair_df[pair_df["split"] == split.lower()].copy().reset_index(drop=True)
    pair_df["mibig_inchikey"] = [_safe_inchikey(smiles, chem_module) for smiles in pair_df["smiles"].tolist()]
    pair_df["mibig_canonical_smiles"] = [
        _safe_canonical_smiles(smiles, chem_module) for smiles in pair_df["smiles"].tolist()
    ]
    return pair_df


def _match_true_products(
    query_df: pd.DataFrame,
    npatlas_df: pd.DataFrame,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    inchikey_index = (
        npatlas_df[npatlas_df["compound_inchikey"] != ""]
        .groupby("compound_inchikey")["candidate_idx"]
        .apply(list)
        .to_dict()
    )
    smiles_index = npatlas_df.groupby("canonical_smiles")["candidate_idx"].apply(list).to_dict()

    matched_by_query: dict[str, set[int]] = {}
    counts = {
        "matches_found_by_inchikey": 0,
        "matches_found_by_smiles": 0,
    }
    for row in query_df.itertuples(index=False):
        candidate_ids: list[int] = []
        match_method: str | None = None
        if getattr(row, "mibig_inchikey") and row.mibig_inchikey in inchikey_index:
            candidate_ids = inchikey_index[row.mibig_inchikey]
            match_method = "inchikey"
        elif getattr(row, "mibig_canonical_smiles") and row.mibig_canonical_smiles in smiles_index:
            candidate_ids = smiles_index[row.mibig_canonical_smiles]
            match_method = "smiles"

        if not candidate_ids:
            continue

        bgc_id = str(row.bgc_id)
        bucket = matched_by_query.setdefault(bgc_id, set())
        bucket.update(int(candidate_id) for candidate_id in candidate_ids)
        if match_method == "inchikey":
            counts["matches_found_by_inchikey"] += 1
        elif match_method == "smiles":
            counts["matches_found_by_smiles"] += 1

    return {bgc_id: sorted(candidate_ids) for bgc_id, candidate_ids in matched_by_query.items()}, counts


def _encode_npatlas_candidates(
    model: DualEncoderCLIP,
    npatlas_df: pd.DataFrame,
    candidate_ids: list[int],
    compound_cache: dict[str, torch.Tensor],
    molecule_encoder,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[list[int], torch.Tensor]:
    feature_rows: list[torch.Tensor] = []
    kept_ids: list[int] = []
    missing_smiles = 0
    local_feature_cache: dict[str, torch.Tensor] = {}

    for candidate_id in candidate_ids:
        row = npatlas_df.iloc[int(candidate_id)]
        cache_key = str(row["canonical_smiles"])
        feature = compound_cache.get(cache_key)
        if feature is None:
            feature = local_feature_cache.get(cache_key)
        if feature is None:
            try:
                feature = molecule_encoder.encode([cache_key])[0]
                local_feature_cache[cache_key] = feature
            except ValueError:
                missing_smiles += 1
                continue
        kept_ids.append(int(candidate_id))
        feature_rows.append(feature.float())

    if missing_smiles:
        pass
    if not kept_ids:
        return [], torch.empty((0, model.compound_proj.net[-1].out_features), dtype=torch.float32)

    encoded_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(feature_rows), batch_size):
            feats = torch.stack(feature_rows[start : start + batch_size]).to(device)
            encoded_chunks.append(model.encode_compound(feats).cpu())
    return kept_ids, torch.cat(encoded_chunks, dim=0)


def _evaluate_npatlas_retrieval(
    model: DualEncoderCLIP,
    interactions: pd.DataFrame,
    data_dir: str | Path,
    split: str,
    bgc_cache: dict[str, torch.Tensor],
    compound_cache: dict[str, torch.Tensor],
    cfg: dict,
    npatlas_path: str | Path,
    npatlas_n: int,
    seed: int,
    device: torch.device,
    logger,
) -> dict[str, Any]:
    if npatlas_n <= 0:
        raise ValueError(f"npatlas_n must be positive, got {npatlas_n}")

    chem = _require_rdkit()
    npatlas_df = _prepare_npatlas_candidates(npatlas_path, chem)
    if len(npatlas_df) < npatlas_n:
        raise ValueError(
            f"NPAtlas candidate library has only {len(npatlas_df)} valid compounds after SMILES cleanup, "
            f"which is smaller than requested npatlas_n={npatlas_n}."
        )

    query_df = _prepare_test_query_table(data_dir, interactions, split, chem)
    matched_by_query, match_counts = _match_true_products(query_df, npatlas_df)
    test_bgc_ids = sorted(interactions[interactions["split"].str.lower() == split.lower()]["bgc_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(int(seed))
    molecule_encoder = build_molecule_encoder(cfg["featurization"], device=device)

    reciprocal_ranks: list[float] = []
    best_ranks: list[int] = []
    hit_at = {1: [], 5: [], 10: []}
    recall_hits = {1: 0, 5: 0, 10: 0}
    precision_at = {1: [], 5: [], 10: []}
    total_ground_truth = 0
    skipped_no_match = 0
    skipped_not_in_sample = 0
    eligible_bgcs = 0
    sampled_candidates = npatlas_df["candidate_idx"].to_numpy(dtype=np.int64)

    progress = tqdm(test_bgc_ids, desc=f"NPAtlas retrieval ({split})")
    for bgc_id in progress:
        true_candidate_ids = matched_by_query.get(bgc_id, [])
        if not true_candidate_ids:
            skipped_no_match += 1
            logger.info("Skipping BGC %s for NPAtlas retrieval: no matched NPAtlas product", bgc_id)
            continue
        if len(true_candidate_ids) > npatlas_n:
            skipped_not_in_sample += 1
            logger.info(
                "Skipping BGC %s for NPAtlas retrieval: %d true candidates exceed sample size %d",
                bgc_id,
                len(true_candidate_ids),
                npatlas_n,
            )
            continue

        true_set = set(int(candidate_id) for candidate_id in true_candidate_ids)
        background = sampled_candidates[~np.isin(sampled_candidates, list(true_set))]
        n_background = npatlas_n - len(true_set)
        sampled_background = rng.choice(background, size=n_background, replace=False)
        candidate_ids = [int(candidate_id) for candidate_id in true_candidate_ids] + [int(x) for x in sampled_background.tolist()]

        kept_ids, candidate_embs = _encode_npatlas_candidates(
            model=model,
            npatlas_df=npatlas_df,
            candidate_ids=candidate_ids,
            compound_cache=compound_cache,
            molecule_encoder=molecule_encoder,
            device=device,
            batch_size=int(cfg["eval"].get("sim_batch_size", 1024)),
        )
        kept_set = set(kept_ids)
        if not true_set.issubset(kept_set):
            skipped_not_in_sample += 1
            logger.info(
                "Skipping BGC %s for NPAtlas retrieval: not all matched truths survived candidate featurization",
                bgc_id,
            )
            continue

        with torch.no_grad():
            bgc_feature = bgc_cache[bgc_id].unsqueeze(0).to(device)
            bgc_emb = model.encode_bgc(bgc_feature).cpu()[0]
        sims = torch.mv(candidate_embs, bgc_emb)
        order = torch.argsort(sims, descending=True)
        ranked_ids = [kept_ids[int(idx)] for idx in order.tolist()]
        rank_lookup = {candidate_id: rank + 1 for rank, candidate_id in enumerate(ranked_ids)}
        true_ranks = sorted(rank_lookup[candidate_id] for candidate_id in true_set)
        best_rank = int(true_ranks[0])

        eligible_bgcs += 1
        total_ground_truth += len(true_set)
        best_ranks.append(best_rank)
        reciprocal_ranks.append(1.0 / float(best_rank))
        ranked_true = [1 if candidate_id in true_set else 0 for candidate_id in ranked_ids]
        for k in (1, 5, 10):
            topk = ranked_true[:k]
            hit_at[k].append(float(any(topk)))
            recall_hits[k] += int(sum(topk))
            precision_at[k].append(float(sum(topk) / float(k)))

    metrics = {
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "median_rank": float(np.median(best_ranks)) if best_ranks else 0.0,
        "hit_at_1": float(np.mean(hit_at[1])) if hit_at[1] else 0.0,
        "hit_at_5": float(np.mean(hit_at[5])) if hit_at[5] else 0.0,
        "hit_at_10": float(np.mean(hit_at[10])) if hit_at[10] else 0.0,
        "recall_at_1": float(recall_hits[1] / total_ground_truth) if total_ground_truth else 0.0,
        "recall_at_5": float(recall_hits[5] / total_ground_truth) if total_ground_truth else 0.0,
        "recall_at_10": float(recall_hits[10] / total_ground_truth) if total_ground_truth else 0.0,
        "precision_at_1": float(np.mean(precision_at[1])) if precision_at[1] else 0.0,
        "precision_at_5": float(np.mean(precision_at[5])) if precision_at[5] else 0.0,
        "precision_at_10": float(np.mean(precision_at[10])) if precision_at[10] else 0.0,
        "npatlas_n": int(npatlas_n),
        "n_test_bgcs": int(len(test_bgc_ids)),
        "n_eligible_bgcs": int(eligible_bgcs),
        "n_skipped_no_match": int(skipped_no_match),
        "n_skipped_not_in_sample": int(skipped_not_in_sample),
        "matches_found_by_inchikey": int(match_counts["matches_found_by_inchikey"]),
        "matches_found_by_smiles": int(match_counts["matches_found_by_smiles"]),
    }
    return metrics


def main() -> None:
    args = parse_args()
    logger = setup_logger("mibig_bgc_np")

    cfg = apply_overrides(load_yaml(args.config), args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_path = args.splits_path if args.splits_path is not None else cfg.get("data", {}).get("splits_path")
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    model, _ = _load_model(args.checkpoint, device)
    interactions = build_interactions(args.data_dir, splits_path=splits_path, cv_fold=args.cv_fold)

    bgc_cache = torch.load(Path(args.cache_dir) / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(Path(args.cache_dir) / "compound_features.pt", map_location="cpu")
    bgc_index, compound_index, bgc_embs, compound_embs, pairs = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split=args.split,
        bgc_cache=bgc_cache,
        compound_cache=compound_cache,
        device=device,
    )

    metrics = evaluate_global_retrieval_multi(
        bgc_embs=bgc_embs,
        compound_embs=compound_embs,
        interaction_pairs=pairs,
        sim_batch_size=cfg["eval"]["sim_batch_size"],
    )

    outdir = Path(args.outdir) if args.outdir is not None else Path(cfg["output"]["dir"]) / "retrieval"
    outdir.mkdir(parents=True, exist_ok=True)
    in_dataset_path = outdir / "in_dataset_retrieval.json"
    save_json(metrics, in_dataset_path)

    sim = model.get_logit_scale().detach().cpu() * (bgc_embs @ compound_embs.t())
    class_retrieval = evaluate_bgc_class_retrieval(
        sim=sim,
        bgc_ids=list(bgc_index.keys()),
        compound_ids=list(compound_index.keys()),
        pairs=pairs,
        interactions=interactions,
        split=args.split,
    )
    class_retrieval["plots"] = [] if bool(args.no_plots) else save_bgc_class_retrieval_plots(
        class_retrieval,
        outdir,
        prefix=args.split,
    )
    class_retrieval_path = outdir / f"bgc_class_retrieval_{args.split}.json"
    save_json(class_retrieval, class_retrieval_path)

    npatlas_metrics = _evaluate_npatlas_retrieval(
        model=model,
        interactions=interactions,
        data_dir=args.data_dir,
        split=args.split,
        bgc_cache=bgc_cache,
        compound_cache=compound_cache,
        cfg=cfg,
        npatlas_path=args.npatlas_path,
        npatlas_n=int(args.npatlas_n),
        seed=seed,
        device=device,
        logger=logger,
    )
    npatlas_path = outdir / f"npatlas_retrieval_n{int(args.npatlas_n)}.json"
    save_json(npatlas_metrics, npatlas_path)

    torch.save(
        {
            "bgc_ids": list(bgc_index.keys()),
            "compound_ids": list(compound_index.keys()),
            "bgc_embeddings": bgc_embs,
            "compound_embeddings": compound_embs,
            "split": args.split,
        },
        outdir / f"embeddings_{args.split}.pt",
    )
    _save_embedding_meta(
        outdir / f"embedding_meta_{args.split}.csv",
        bgc_ids=list(bgc_index.keys()),
        compound_ids=list(compound_index.keys()),
        split=args.split,
    )

    logger.info("Saved in-dataset retrieval metrics to %s", in_dataset_path)
    logger.info("Saved BGC-class retrieval ROC diagnostics to %s", class_retrieval_path)
    logger.info("Saved NPAtlas retrieval metrics to %s", npatlas_path)
    logger.info("In-dataset retrieval: %s", metrics)
    logger.info("NPAtlas retrieval: %s", npatlas_metrics)


if __name__ == "__main__":
    main()
