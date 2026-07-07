from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.eval.baseline_artifacts import save_all_baseline_artifacts
from projects.mibig_bgc_np.eval.retrieval_class_metrics import (
    evaluate_bgc_class_pair_scores,
    save_bgc_class_retrieval_plots,
    save_bgc_map_metrics_table,
)
from projects.mibig_bgc_np.eval.retrieval_baselines import run_retrieval_baseline_suite
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.scripts.run_bgcmac_ensemble import (
    _aggregate_objects,
    _load_member_model,
    _split_counts,
    _train_one_member,
)
from projects.mibig_bgc_np.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BGC-MAP explicit pair retrieval benchmark.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--bgcmap_splits_path", type=str, default="data/MIBIG/splits/MAP_metadata_fold.csv")
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fold", type=int, default=10)
    parser.add_argument("--val_folds", type=int, nargs="*", default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--retrieval_baselines",
        dest="retrieval_baselines",
        action="store_true",
        default=True,
        help="Run random, frozen encoder similarity, kNN transfer, and linear-projection retrieval baselines. Enabled by default.",
    )
    parser.add_argument(
        "--no_retrieval_baselines",
        dest="retrieval_baselines",
        action="store_false",
        help="Disable retrieval baselines.",
    )
    parser.add_argument("--baseline_random_trials", type=int, default=10)
    parser.add_argument("--baseline_k_values", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument(
        "--reuse_existing_members",
        action="store_true",
        help="Reuse existing member checkpoints instead of retraining contrastive models.",
    )
    parser.add_argument(
        "--save_cm_png",
        dest="save_cm_png",
        action="store_true",
        default=True,
        help="Save BGC-MAP ROC, confusion matrix, and metrics-table PNGs. Enabled by default.",
    )
    parser.add_argument(
        "--no_save_cm_png",
        dest="save_cm_png",
        action="store_false",
        help="Disable BGC-MAP plot/table PNG outputs.",
    )
    return parser.parse_args()


def _load_bgcmap_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"BGC_number", "product", "biosyn_class", "is_product", "fold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"BGC-MAP split file {path} is missing required columns: {sorted(missing)}")

    out = df[["BGC_number", "product", "biosyn_class", "is_product", "fold"]].copy()
    out = out.rename(
        columns={
            "BGC_number": "bgc_id",
            "product": "compound_id",
            "biosyn_class": "bgc_classes",
        }
    )
    out["bgc_id"] = out["bgc_id"].astype(str)
    out["compound_id"] = out["compound_id"].astype(str)
    out["bgc_classes"] = out["bgc_classes"].astype(str)
    out["is_product"] = pd.to_numeric(out["is_product"], errors="coerce")
    out["fold"] = pd.to_numeric(out["fold"], errors="coerce")
    if bool(out[["is_product", "fold"]].isna().any().any()):
        raise ValueError(f"BGC-MAP split file {path} contains non-numeric is_product or fold values.")
    out["is_product"] = (out["is_product"].astype(float) > 0.0).astype(int)
    out["fold"] = out["fold"].astype(int)
    return out.dropna(subset=["bgc_id", "compound_id"]).reset_index(drop=True)


def _build_bgcmap_positive_interactions(map_table: pd.DataFrame, val_fold: int, test_fold: int) -> pd.DataFrame:
    positives = map_table[map_table["is_product"] == 1].copy()
    positives["split"] = np.where(
        positives["fold"] == int(test_fold),
        "test",
        np.where(positives["fold"] == int(val_fold), "val", "train"),
    )
    interactions = positives[["bgc_id", "compound_id", "bgc_classes", "split", "fold"]].copy()
    interactions = interactions.drop_duplicates(subset=["bgc_id", "compound_id", "split"]).reset_index(drop=True)
    return interactions


def _write_resolved_pair_split_tsv(interactions: pd.DataFrame, output_path: Path) -> Path:
    split_df = interactions[["bgc_id", "compound_id", "split", "fold"]].drop_duplicates(
        subset=["bgc_id", "compound_id"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.sort_values(["split", "bgc_id", "compound_id"]).to_csv(output_path, sep="\t", index=False)
    return output_path


def _encode_ids(
    model: DualEncoderCLIP,
    ids: list[str],
    cache: dict[str, torch.Tensor],
    *,
    modality: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    missing = sorted(set(ids).difference(cache))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing {modality} features for {len(missing)} BGC-MAP ids. Examples: {preview}")

    output: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            features = torch.stack([cache[item_id].float() for item_id in batch_ids]).to(device)
            if modality == "bgc":
                embeddings = model.encode_bgc(features).cpu()
            elif modality == "compound":
                embeddings = model.encode_compound(features).cpu()
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            for item_id, embedding in zip(batch_ids, embeddings, strict=True):
                output[item_id] = embedding
    return output


def _score_bgcmap_pairs(
    models: list[DualEncoderCLIP],
    map_table: pd.DataFrame,
    cache_dir: str | Path,
    device: torch.device,
    batch_size: int,
    target_fold: int,
) -> pd.DataFrame:
    target_df = map_table[map_table["fold"] == int(target_fold)].copy().reset_index(drop=True)
    if target_df.empty:
        raise ValueError(f"BGC-MAP table has no rows for fold {target_fold}.")

    bgc_ids = sorted(target_df["bgc_id"].astype(str).unique().tolist())
    compound_ids = sorted(target_df["compound_id"].astype(str).unique().tolist())
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(Path(cache_dir) / "compound_features.pt", map_location="cpu")

    score_sum = np.zeros(len(target_df), dtype=np.float64)
    for model in models:
        bgc_embeddings = _encode_ids(
            model,
            bgc_ids,
            bgc_cache,
            modality="bgc",
            device=device,
            batch_size=batch_size,
        )
        compound_embeddings = _encode_ids(
            model,
            compound_ids,
            compound_cache,
            modality="compound",
            device=device,
            batch_size=batch_size,
        )
        scale = float(model.get_logit_scale().detach().cpu().item())
        member_scores = [
            scale * float(torch.dot(bgc_embeddings[str(row.bgc_id)], compound_embeddings[str(row.compound_id)]).item())
            for row in target_df.itertuples(index=False)
        ]
        score_sum += np.asarray(member_scores, dtype=np.float64)

    scored = target_df.copy()
    scored["score"] = score_sum / float(len(models))
    return scored


def _validation_thresholds_from_members(
    models: list[DualEncoderCLIP],
    val_folds: list[int],
    map_table: pd.DataFrame,
    cache_dir: str | Path,
    device: torch.device,
    batch_size: int,
    outdir: Path,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    threshold_values: dict[str, list[float]] = {}
    validation_summaries: list[dict[str, Any]] = []

    for model, val_fold in zip(models, val_folds, strict=True):
        scored_val = _score_bgcmap_pairs(
            models=[model],
            map_table=map_table,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
            target_fold=int(val_fold),
        )
        val_scores_path = outdir / f"val_fold_{int(val_fold)}" / "validation_pair_scores.tsv"
        scored_val.to_csv(val_scores_path, sep="\t", index=False)

        val_report = evaluate_bgc_class_pair_scores(scored_val, split=f"validation_fold_{int(val_fold)}")
        val_report_path = outdir / f"val_fold_{int(val_fold)}" / "validation_pair_retrieval.json"
        save_json(val_report, val_report_path)

        member_thresholds: dict[str, float] = {}
        for class_name, metrics in val_report.get("classes", {}).items():
            threshold = float(metrics["threshold"])
            threshold_values.setdefault(class_name, []).append(threshold)
            member_thresholds[class_name] = threshold

        validation_summaries.append(
            {
                "val_fold": int(val_fold),
                "scored_pairs_path": str(val_scores_path),
                "report_path": str(val_report_path),
                "thresholds": member_thresholds,
                "n_rows": int(len(scored_val)),
                "n_positive": int(scored_val["is_product"].sum()),
                "n_negative": int(len(scored_val) - int(scored_val["is_product"].sum())),
            }
        )

    thresholds_by_class = {
        class_name: float(np.mean(values))
        for class_name, values in sorted(threshold_values.items())
        if values
    }
    return thresholds_by_class, validation_summaries


def main() -> None:
    args = parse_args()
    logger = setup_logger("bgcmap_retrieval")
    cfg = apply_overrides(load_yaml(args.config), args.override)
    cfg["seed"] = int(args.seed)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir) if args.outdir is not None else Path("results") / "bgcmap_retrieval"
    outdir.mkdir(parents=True, exist_ok=True)

    map_table = _load_bgcmap_table(args.bgcmap_splits_path)
    val_folds = args.val_folds if args.val_folds is not None and len(args.val_folds) else sorted(
        int(fold_id) for fold_id in map_table.loc[map_table["fold"] != int(args.test_fold), "fold"].unique().tolist()
    )

    member_summaries: list[dict[str, Any]] = []
    models: list[DualEncoderCLIP] = []
    for val_fold in val_folds:
        member_outdir = outdir / f"val_fold_{int(val_fold)}"
        member_cfg = copy.deepcopy(cfg)
        member_cfg["output"]["dir"] = str(member_outdir)
        set_seed(int(args.seed) + int(val_fold))
        interactions = _build_bgcmap_positive_interactions(
            map_table,
            val_fold=int(val_fold),
            test_fold=int(args.test_fold),
        )
        counts = _split_counts(interactions)
        best_ckpt_path = member_outdir / "contrastive_model_best.pt"
        metrics_path = member_outdir / "contrastive_metrics.json"
        if bool(args.reuse_existing_members) and best_ckpt_path.exists():
            logger.info("Reusing BGC-MAP retrieval member checkpoint for val fold %s", val_fold)
            model = _load_member_model(best_ckpt_path, device=device)
            metrics = load_yaml(metrics_path) if metrics_path.exists() else {}
        else:
            logger.info("Training BGC-MAP retrieval member with val fold %s counts: %s", val_fold, counts)
            model, metrics = _train_one_member(
                interactions=interactions,
                cache_dir=args.cache_dir,
                cfg=member_cfg,
                device=device,
                outdir=member_outdir,
                patience=int(args.patience),
            )
        resolved_splits_path = _write_resolved_pair_split_tsv(
            interactions,
            member_outdir / "bgcmap_positive_resolved_splits.tsv",
        )
        retrieval_baselines_test: dict[str, Any] = {}
        if bool(args.retrieval_baselines):
            retrieval_baselines_test = run_retrieval_baseline_suite(
                interactions=interactions,
                split="test",
                cache_dir=args.cache_dir,
                cfg=member_cfg,
                device=device,
                outdir=member_outdir / "retrieval_baselines",
                seed=int(args.seed) + int(val_fold),
                random_trials=int(args.baseline_random_trials),
                k_values=[int(k) for k in args.baseline_k_values],
                patience=int(args.patience),
            )
        models.append(model)
        member_summary = {
            "val_fold": int(val_fold),
            "output_dir": str(member_outdir),
            "resolved_splits_path": str(resolved_splits_path),
            "counts": counts,
            "metrics": metrics,
            "retrieval_baselines_test": retrieval_baselines_test,
        }
        save_json(member_summary, member_outdir / "member_summary.json")
        member_summaries.append(member_summary)

    validation_thresholds, validation_threshold_summaries = _validation_thresholds_from_members(
        models=models,
        val_folds=[int(fold) for fold in val_folds],
        map_table=map_table,
        cache_dir=args.cache_dir,
        device=device,
        batch_size=int(cfg["eval"]["sim_batch_size"]),
        outdir=outdir,
    )
    validation_thresholds_path = outdir / "validation_thresholds.json"
    save_json(
        {
            "threshold_protocol": "per-member Youden threshold on that member's held-out validation fold, averaged by BGC class",
            "thresholds_by_class": validation_thresholds,
            "members": validation_threshold_summaries,
        },
        validation_thresholds_path,
    )

    scored_pairs = _score_bgcmap_pairs(
        models=models,
        map_table=map_table,
        cache_dir=args.cache_dir,
        device=device,
        batch_size=int(cfg["eval"]["sim_batch_size"]),
        target_fold=int(args.test_fold),
    )
    scored_pairs_path = outdir / "ensemble_test_pair_scores.tsv"
    scored_pairs.to_csv(scored_pairs_path, sep="\t", index=False)

    class_report = evaluate_bgc_class_pair_scores(
        scored_pairs,
        split="test",
        thresholds_by_class=validation_thresholds,
    )
    class_report["threshold_protocol"] = "validation_derived_mean_by_class"
    class_report["validation_thresholds_path"] = str(validation_thresholds_path)
    class_report["plots"] = save_bgc_class_retrieval_plots(
        class_report,
        outdir,
        prefix="ensemble_test",
    ) if bool(args.save_cm_png) else []
    class_report["metrics_table"] = save_bgc_map_metrics_table(
        class_report,
        outdir,
        prefix="ensemble_test",
    ) if bool(args.save_cm_png) else {}

    ensemble = {
        "bgc_class_pair_retrieval": class_report,
        "scored_pairs_path": str(scored_pairs_path),
        "validation_thresholds_path": str(validation_thresholds_path),
        "validation_thresholds": validation_thresholds,
        "n_models": int(len(models)),
        "n_test_rows": int(len(scored_pairs)),
        "n_test_positive": int(scored_pairs["is_product"].sum()),
        "n_test_negative": int(len(scored_pairs) - int(scored_pairs["is_product"].sum())),
        "n_test_bgcs": int(scored_pairs["bgc_id"].nunique()),
        "n_test_compounds": int(scored_pairs["compound_id"].nunique()),
    }
    save_json(ensemble, outdir / "ensemble_test_retrieval.json")

    summary = {
        "protocol": "BGC-MAP fixed fold-10 test with folds 1-9 rotated as validation folds across 9 members",
        "benchmark": "BGC-MAP",
        "benchmark_task": "explicit_pair_retrieval",
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "bgcmap_splits_path": str(args.bgcmap_splits_path),
        "test_fold": int(args.test_fold),
        "val_folds": [int(fold) for fold in val_folds],
        "patience": int(args.patience),
        "members": member_summaries,
        "ensemble_test": ensemble,
        "aggregate": {
            "counts": _aggregate_objects([summary["counts"] for summary in member_summaries]),
            "contrastive_metrics": _aggregate_objects([summary["metrics"] for summary in member_summaries]),
            "retrieval_baselines_test": _aggregate_objects(
                [
                    summary["retrieval_baselines_test"]
                    for summary in member_summaries
                    if "retrieval_baselines_test" in summary
                ]
            ),
        },
    }
    summary_path = outdir / "summary.json"
    save_json(summary, summary_path)
    logger.info("Saved BGC-MAP retrieval summary to %s", summary_path)

    try:
        baseline_artifacts = save_all_baseline_artifacts(outdir)
        summary["baseline_artifacts"] = baseline_artifacts
        save_json(summary, summary_path)
        logger.info("Saved visible baseline artifacts to %s", outdir / "baselines")
    except Exception as exc:
        logger.warning("Could not create visible baseline artifacts: %s", exc)


if __name__ == "__main__":
    main()
