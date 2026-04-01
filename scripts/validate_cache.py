from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.cache import FeatureCache
from kiba_clip.data.datasets import build_interactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cached protein embeddings and ligand fingerprints.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory with cache_index.json and *.pt cache files.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional interaction split directory to validate cache key coverage.",
    )
    return parser.parse_args()


def _load_index(cache_dir: Path) -> dict:
    idx_path = cache_dir / "cache_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing cache index: {idx_path}")
    with idx_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dim_set(cache: dict[str, torch.Tensor]) -> set[int]:
    return {int(v.numel()) for v in cache.values()}


def _dtype_set(cache: dict[str, torch.Tensor]) -> set[str]:
    return {str(v.dtype) for v in cache.values()}


def _stack(cache: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([v.float() for v in cache.values()], dim=0)


def _summary_stats(name: str, mat: torch.Tensor) -> list[str]:
    norms = torch.linalg.vector_norm(mat, dim=1)
    lines = [
        f"{name} norm min/mean/max: {norms.min().item():.4f} / {norms.mean().item():.4f} / {norms.max().item():.4f}",
        f"{name} abs-value min/max: {mat.abs().min().item():.4f} / {mat.abs().max().item():.4f}",
    ]
    return lines


def _fingerprint_density(name: str, mat: torch.Tensor) -> str:
    # Morgan fingerprints are binary-ish vectors; this reports average on-bit ratio.
    on_ratio = (mat > 0).float().mean(dim=1)
    return (
        f"{name} on-bit ratio min/mean/max: "
        f"{on_ratio.min().item():.4f} / {on_ratio.mean().item():.4f} / {on_ratio.max().item():.4f}"
    )


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    idx = _load_index(cache_dir)
    prot_path = Path(idx["protein_cache"])
    lig_path = Path(idx["ligand_cache"])

    prot_cache = FeatureCache.load(prot_path)
    lig_cache = FeatureCache.load(lig_path)

    issues: list[str] = []
    report: list[str] = []

    report.append(f"Proteins in cache: {len(prot_cache)}")
    report.append(f"Ligands in cache: {len(lig_cache)}")

    if len(prot_cache) == 0:
        issues.append("Protein cache is empty")
    if len(lig_cache) == 0:
        issues.append("Ligand cache is empty")

    if prot_cache:
        prot_dims = _dim_set(prot_cache)
        prot_dtypes = _dtype_set(prot_cache)
        report.append(f"Protein dims seen: {sorted(prot_dims)}")
        report.append(f"Protein dtypes seen: {sorted(prot_dtypes)}")
        if len(prot_dims) != 1:
            issues.append(f"Inconsistent protein dims: {sorted(prot_dims)}")

        prot_mat = _stack(prot_cache)
        if not torch.isfinite(prot_mat).all():
            issues.append("Protein embeddings contain NaN/Inf")
        report.extend(_summary_stats("Protein", prot_mat))

    if lig_cache:
        lig_dims = _dim_set(lig_cache)
        lig_dtypes = _dtype_set(lig_cache)
        report.append(f"Ligand dims seen: {sorted(lig_dims)}")
        report.append(f"Ligand dtypes seen: {sorted(lig_dtypes)}")
        if len(lig_dims) != 1:
            issues.append(f"Inconsistent ligand dims: {sorted(lig_dims)}")

        lig_mat = _stack(lig_cache)
        if not torch.isfinite(lig_mat).all():
            issues.append("Ligand fingerprints contain NaN/Inf")
        report.extend(_summary_stats("Ligand", lig_mat))
        report.append(_fingerprint_density("Ligand", lig_mat))

    if args.data_dir is not None:
        interactions = build_interactions(args.data_dir)
        exp_prot = set(interactions["Target_ID"].drop_duplicates().astype(str).tolist())
        exp_lig = set(interactions["Drug_ID"].drop_duplicates().astype(str).tolist())
        got_prot = set(map(str, prot_cache.keys()))
        got_lig = set(map(str, lig_cache.keys()))

        missing_prot = exp_prot - got_prot
        missing_lig = exp_lig - got_lig

        report.append(f"Expected proteins from data: {len(exp_prot)}")
        report.append(f"Expected ligands from data: {len(exp_lig)}")

        if missing_prot:
            issues.append(f"Missing protein IDs in cache: {len(missing_prot)}")
        if missing_lig:
            issues.append(f"Missing ligand IDs in cache: {len(missing_lig)}")

        # Extra keys are usually not harmful, but report for transparency.
        extra_prot = got_prot - exp_prot
        extra_lig = got_lig - exp_lig
        report.append(f"Extra protein IDs in cache: {len(extra_prot)}")
        report.append(f"Extra ligand IDs in cache: {len(extra_lig)}")

    print("=== Cache Validation Report ===")
    for line in report:
        print(line)

    if issues:
        print("\n=== Issues Found ===")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("\nValidation passed: no blocking issues detected.")


if __name__ == "__main__":
    main()
