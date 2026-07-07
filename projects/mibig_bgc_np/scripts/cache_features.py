from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.cache import FeatureCache
from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import load_bgc_proteins, load_pair_table
from projects.mibig_bgc_np.featurization import build_bgc_encoder, build_molecule_encoder
from mibig_clip.data.preprocessing import parse_mibig_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache MIBiG BGC features and compound Morgan fingerprints.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument(
        "--map_metadata_path",
        type=str,
        default=None,
        help="Optional BGC-MAP metadata CSV. Adds MAP BGCs and candidate products, including negatives, to the cache.",
    )
    parser.add_argument(
        "--bgcmac_splits_path",
        type=str,
        default=None,
        help="Optional BGC-MAC split CSV. Adds all BGC-MAC BGCs to the BGC cache, including BGCs without SMILES.",
    )
    parser.add_argument(
        "--fasta_path",
        type=str,
        default=None,
        help="Optional MIBiG protein FASTA used to supplement BGCs missing from processed bgc_proteins.jsonl.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def _load_map_cache_rows(path: str | Path | None) -> tuple[set[str], list[dict[str, str]]]:
    if path is None:
        return set(), []
    map_path = Path(path)
    map_df = pd.read_csv(map_path)
    required = {"BGC_number", "product"}
    missing = required.difference(map_df.columns)
    if missing:
        raise ValueError(f"BGC-MAP metadata file {map_path} is missing required columns: {sorted(missing)}")
    map_df = map_df.dropna(subset=["BGC_number", "product"]).copy()
    bgc_ids = set(map_df["BGC_number"].astype(str).tolist())
    compound_rows = (
        map_df[["product"]]
        .drop_duplicates("product")
        .rename(columns={"product": "compound_id"})
        .reset_index(drop=True)
    )
    compound_rows["smiles"] = compound_rows["compound_id"].astype(str)
    return bgc_ids, compound_rows.to_dict("records")


def _load_bgcmac_bgc_ids(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    split_path = Path(path)
    split_df = pd.read_csv(split_path)
    if "BGC_number" not in split_df.columns:
        raise ValueError(f"BGC-MAC split file {split_path} is missing required column: BGC_number")
    return set(split_df["BGC_number"].dropna().astype(str).tolist())


def main() -> None:
    args = parse_args()
    logger = setup_logger("mibig_bgc_np")

    cfg = apply_overrides(load_yaml(args.config), args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pair_df = load_pair_table(args.data_dir)
    proteins_index = load_bgc_proteins(args.data_dir)
    map_bgc_ids, map_compound_rows = _load_map_cache_rows(args.map_metadata_path)
    bgcmac_bgc_ids = _load_bgcmac_bgc_ids(args.bgcmac_splits_path)
    supplemented_bgc_count = 0
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgc_encoder = build_bgc_encoder(cfg["featurization"], device=device)
    molecule_encoder = build_molecule_encoder(cfg["featurization"])
    bgc_batch_size = int(cfg["featurization"].get("bgc_batch_size", 1))

    bgc_ids = sorted(set(pair_df["bgc_id"].astype(str).tolist()) | map_bgc_ids | bgcmac_bgc_ids)
    missing_bgcs = sorted(set(bgc_ids).difference(proteins_index))
    if missing_bgcs and args.fasta_path is not None:
        logger.info("Supplementing %d missing BGC protein records from %s", len(missing_bgcs), args.fasta_path)
        fasta_records = parse_mibig_fasta(Path(args.fasta_path))
        for bgc_id in missing_bgcs:
            fasta_entry = fasta_records.get(bgc_id)
            if fasta_entry is None:
                continue
            proteins_index[bgc_id] = {
                "bgc_id": bgc_id,
                "bgc_version": fasta_entry["bgc_version"],
                "protein_ids": fasta_entry["protein_ids"],
                "protein_seqs": fasta_entry["protein_seqs"],
            }
            supplemented_bgc_count += 1
        missing_bgcs = sorted(set(bgc_ids).difference(proteins_index))
    if missing_bgcs:
        preview = ", ".join(missing_bgcs[:5])
        raise KeyError(
            f"Missing protein sequences for {len(missing_bgcs)} BGCs. Examples: {preview}. "
            "Pass --fasta_path data/MIBIG/mibig_prot_seqs_4.0.fasta if those BGCs exist in the raw FASTA."
        )
    bgc_cache = FeatureCache(outdir / "bgc_features.pt")
    for start in tqdm(range(0, len(bgc_ids), bgc_batch_size), desc="BGC features"):
        batch_ids = bgc_ids[start : start + bgc_batch_size]
        protein_batches = [proteins_index[bgc_id]["protein_seqs"] for bgc_id in batch_ids]
        batch_features = bgc_encoder.encode_bgcs(protein_batches)
        for bgc_id, feature in zip(batch_ids, batch_features, strict=True):
            bgc_cache.add(bgc_id, feature)
    bgc_cache.save()

    compound_df = pair_df[["compound_id", "smiles"]].drop_duplicates("compound_id").reset_index(drop=True)
    if map_compound_rows:
        map_compound_df = pd.DataFrame(map_compound_rows)
        compound_df = (
            pd.concat([compound_df, map_compound_df], ignore_index=True)
            .drop_duplicates("compound_id")
            .reset_index(drop=True)
        )
    compound_cache = FeatureCache(outdir / "compound_features.pt")
    for row in tqdm(compound_df.itertuples(index=False), total=len(compound_df), desc="Compound features"):
        try:
            feature = molecule_encoder.encode([str(row.smiles)])[0]
            compound_cache.add(str(row.compound_id), feature)
        except ValueError:
            logger.warning("Skipping invalid SMILES for compound_id=%s", row.compound_id)
    compound_cache.save()

    cache_index = {
        "bgc_cache": str(outdir / "bgc_features.pt"),
        "compound_cache": str(outdir / "compound_features.pt"),
        "bgc_featurizer": str(cfg["featurization"].get("bgc_encoder", "one_hot")),
        "compound_featurizer": str(cfg["featurization"].get("molecule_encoder", "morgan")),
        "n_bgcs": len(bgc_cache.data),
        "n_compounds": len(compound_cache.data),
        "map_metadata_path": str(args.map_metadata_path) if args.map_metadata_path is not None else None,
        "bgcmac_splits_path": str(args.bgcmac_splits_path) if args.bgcmac_splits_path is not None else None,
        "fasta_path": str(args.fasta_path) if args.fasta_path is not None else None,
        "n_bgc_records_supplemented_from_fasta": int(supplemented_bgc_count),
        "bgc_dim": int(next(iter(bgc_cache.data.values())).numel()),
        "compound_dim": int(next(iter(compound_cache.data.values())).numel()),
    }
    save_json(cache_index, outdir / "cache_index.json")
    logger.info("Saved MIBiG caches to %s", outdir)


if __name__ == "__main__":
    main()
