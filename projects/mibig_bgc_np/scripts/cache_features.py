from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.cache import FeatureCache
from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import load_bgc_proteins, load_pair_table
from projects.mibig_bgc_np.featurization import build_bgc_encoder, build_molecule_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache MIBiG BGC features and compound Morgan fingerprints.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("mibig_bgc_np")

    cfg = apply_overrides(load_yaml(args.config), args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pair_df = load_pair_table(args.data_dir)
    proteins_index = load_bgc_proteins(args.data_dir)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgc_encoder = build_bgc_encoder(cfg["featurization"], device=device)
    molecule_encoder = build_molecule_encoder(cfg["featurization"])
    bgc_batch_size = int(cfg["featurization"].get("bgc_batch_size", 1))

    bgc_ids = sorted(set(pair_df["bgc_id"].astype(str).tolist()))
    bgc_cache = FeatureCache(outdir / "bgc_features.pt")
    for start in tqdm(range(0, len(bgc_ids), bgc_batch_size), desc="BGC features"):
        batch_ids = bgc_ids[start : start + bgc_batch_size]
        protein_batches = [proteins_index[bgc_id]["protein_seqs"] for bgc_id in batch_ids]
        batch_features = bgc_encoder.encode_bgcs(protein_batches)
        for bgc_id, feature in zip(batch_ids, batch_features, strict=True):
            bgc_cache.add(bgc_id, feature)
    bgc_cache.save()

    compound_df = pair_df[["compound_id", "smiles"]].drop_duplicates("compound_id").reset_index(drop=True)
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
        "bgc_dim": int(next(iter(bgc_cache.data.values())).numel()),
        "compound_dim": int(next(iter(compound_cache.data.values())).numel()),
    }
    save_json(cache_index, outdir / "cache_index.json")
    logger.info("Saved MIBiG caches to %s", outdir)


if __name__ == "__main__":
    main()
