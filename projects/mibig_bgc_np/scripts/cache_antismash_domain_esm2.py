from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import load_pair_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean-pool provided BGC-MAC ESM2 antiSMASH-domain embeddings into one feature per BGC."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--embeddings_path", type=Path, required=True)
    parser.add_argument("--compound_cache_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--bgc_ids_csv",
        action="append",
        default=[],
        metavar="PATH:COLUMN",
        help="Optional CSV and ID column to add to the BGC cache; may be repeated.",
    )
    return parser.parse_args()


def _load_extra_bgc_ids(specs: list[str]) -> set[str]:
    bgc_ids: set[str] = set()
    for spec in specs:
        path_text, separator, column = spec.rpartition(":")
        if not separator or not path_text or not column:
            raise ValueError(f"--bgc_ids_csv must be PATH:COLUMN, got {spec!r}")
        path = Path(path_text)
        table = pd.read_csv(path)
        if column not in table.columns:
            raise ValueError(f"{path} does not contain column {column!r}")
        bgc_ids.update(table[column].dropna().astype(str))
    return bgc_ids


def main() -> None:
    args = parse_args()
    logger = setup_logger("cache_antismash_domain_esm2")
    pair_df = load_pair_table(args.data_dir)
    bgc_ids = sorted(
        set(pair_df["bgc_id"].astype(str).unique().tolist())
        | _load_extra_bgc_ids(args.bgc_ids_csv)
    )

    domain_embeddings = torch.load(args.embeddings_path, map_location="cpu", weights_only=True)
    missing = sorted(set(bgc_ids).difference(domain_embeddings))
    if missing:
        raise KeyError(f"Missing domain embeddings for {len(missing)} paired BGCs; examples: {missing[:5]}")

    bgc_cache: dict[str, torch.Tensor] = {}
    domain_counts: dict[str, int] = {}
    for bgc_id in bgc_ids:
        embeddings = domain_embeddings[bgc_id]
        if not embeddings:
            raise ValueError(f"No domain embeddings available for {bgc_id}")
        matrix = torch.stack([embedding.float().reshape(-1) for embedding in embeddings])
        bgc_cache[bgc_id] = matrix.mean(dim=0)
        domain_counts[bgc_id] = int(matrix.size(0))

    compound_cache = torch.load(args.compound_cache_path, map_location="cpu", weights_only=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bgc_path = args.out_dir / "bgc_features.pt"
    compound_path = args.out_dir / "compound_features.pt"
    torch.save(bgc_cache, bgc_path)
    torch.save(compound_cache, compound_path)

    counts = torch.tensor(list(domain_counts.values()), dtype=torch.float32)
    save_json(
        {
            "bgc_cache": str(bgc_path),
            "compound_cache": str(compound_path),
            "bgc_featurizer": "provided_bgcmac_antismash_domain_esm2_mean",
            "pooling": "arithmetic mean across domain/unsplit-enzyme embeddings",
            "source_embeddings_path": str(args.embeddings_path),
            "bgc_ids_csv": list(args.bgc_ids_csv),
            "n_bgcs": len(bgc_cache),
            "n_compounds": len(compound_cache),
            "bgc_dim": int(next(iter(bgc_cache.values())).numel()),
            "compound_dim": int(next(iter(compound_cache.values())).numel()),
            "domain_count_min": int(counts.min().item()),
            "domain_count_mean": float(counts.mean().item()),
            "domain_count_max": int(counts.max().item()),
        },
        args.out_dir / "cache_index.json",
    )
    logger.info("Saved %d mean-pooled ESM2 antiSMASH-domain BGC features to %s", len(bgc_cache), args.out_dir)


if __name__ == "__main__":
    main()
