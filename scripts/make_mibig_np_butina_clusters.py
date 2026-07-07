from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    raise SystemExit(
        "RDKit is required for NP Butina clustering. Run this script in an environment with RDKit, "
        "for example: conda run -n combi python -m scripts.make_mibig_np_butina_clusters"
    ) from exc


EXPLICIT_PRODUCT_ID_COLUMNS = (
    "product_id",
    "compound_id",
    "np_id",
    "npatlas_id",
    "database_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster unique MiBIG BGC-NP products by NP-side Butina clustering over Morgan/Tanimoto distances. "
            "This prepares chemical leakage groups only; it does not create strict train/validation/test splits."
        )
    )
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=Path("data/MIBIG/processed/mibig_pairs.tsv"),
        help="Final MiBIG BGC-NP paired table.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7.tsv"),
        help="Output TSV with product IDs, SMILES, canonical SMILES, and np_butina_cluster.",
    )
    parser.add_argument(
        "--invalid_output_path",
        type=Path,
        default=None,
        help="Optional invalid SMILES TSV path. Defaults to <output_stem>_invalid_smiles.tsv.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to <output_stem>_report.json.",
    )
    parser.add_argument(
        "--cluster_sizes_path",
        type=Path,
        default=None,
        help="Optional cluster sizes TSV path. Defaults to <output_stem>_cluster_sizes.tsv.",
    )
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan fingerprint bit length.")
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=0.3,
        help="RDKit Butina distance threshold. 0.3 corresponds to Tanimoto similarity >= 0.7.",
    )
    return parser.parse_args()


def _default_sidecar_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}")


def _first_existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in columns:
            return column
    return None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _load_pair_products(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path, sep="\t")
    if "smiles" not in df.columns:
        raise ValueError(f"{path} is missing required column 'smiles'.")

    explicit_id_column = _first_existing_column(list(df.columns), EXPLICIT_PRODUCT_ID_COLUMNS)
    name_column = "compound_name" if "compound_name" in df.columns else None

    rows: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        original_smiles = _normalize_text(row.get("smiles"))
        if original_smiles is None:
            rows.append(
                {
                    "product_id": None,
                    "compound_name": _normalize_text(row.get(name_column)) if name_column else None,
                    "smiles": None,
                    "compound_key_preference": "missing_smiles",
                }
            )
            continue
        product_id = _normalize_text(row.get(explicit_id_column)) if explicit_id_column else None
        rows.append(
            {
                "product_id": product_id,
                "compound_name": _normalize_text(row.get(name_column)) if name_column else None,
                "smiles": original_smiles,
                "compound_key_preference": explicit_id_column or "canonical_smiles",
            }
        )

    products = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    metadata = {
        "pairs_path": str(path),
        "n_pair_rows": int(len(df)),
        "explicit_product_id_column": explicit_id_column,
        "compound_name_column": name_column,
        "compound_key_fallback": "canonical_smiles",
    }
    return products, metadata


def _canonicalize_products(products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    for row in products.itertuples(index=False):
        smiles = _normalize_text(getattr(row, "smiles"))
        product_id = _normalize_text(getattr(row, "product_id"))
        compound_name = _normalize_text(getattr(row, "compound_name"))
        if smiles is None:
            invalid_rows.append(
                {
                    "product_id": product_id,
                    "compound_name": compound_name,
                    "smiles": None,
                    "reason": "missing_smiles",
                }
            )
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_rows.append(
                {
                    "product_id": product_id,
                    "compound_name": compound_name,
                    "smiles": smiles,
                    "reason": "rdkit_parse_failed",
                }
            )
            continue
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        compound_key = product_id or canonical_smiles
        valid_rows.append(
            {
                "product_id": product_id,
                "compound_name": compound_name,
                "compound_key": compound_key,
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "compound_key_source": "product_id" if product_id else "canonical_smiles",
            }
        )

    valid_df = pd.DataFrame(
        valid_rows,
        columns=[
            "product_id",
            "compound_name",
            "compound_key",
            "smiles",
            "canonical_smiles",
            "compound_key_source",
        ],
    )
    invalid_df = pd.DataFrame(
        invalid_rows,
        columns=["product_id", "compound_name", "smiles", "reason"],
    )
    return valid_df, invalid_df


def _deduplicate_compounds(valid_df: pd.DataFrame) -> pd.DataFrame:
    if valid_df.empty:
        return valid_df
    # If a product ID exists, use it as requested. For this MiBIG table product IDs
    # are usually absent, so canonical SMILES becomes the stable unique compound key.
    return (
        valid_df.sort_values(["compound_key_source", "compound_key", "canonical_smiles", "smiles"])
        .drop_duplicates(subset=["compound_key"], keep="first")
        .reset_index(drop=True)
    )


def _compute_butina_clusters(
    valid_unique: pd.DataFrame,
    *,
    radius: int,
    n_bits: int,
    dist_thresh: float,
) -> pd.DataFrame:
    if valid_unique.empty:
        raise ValueError("No valid unique compounds available for Butina clustering.")

    mols = []
    for smiles in valid_unique["canonical_smiles"].astype(str).tolist():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Internal error: canonical SMILES failed to parse: {smiles}")
        mols.append(mol)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    fps = [generator.GetFingerprint(mol) for mol in mols]

    distances: list[float] = []
    for i in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend([1.0 - float(similarity) for similarity in similarities])

    clusters = Butina.ClusterData(distances, len(fps), float(dist_thresh), isDistData=True)
    cluster_by_row_idx: dict[int, str] = {}
    for cluster_idx, members in enumerate(clusters, start=1):
        cluster_id = f"NP_BUTINA_{cluster_idx:06d}"
        for member_idx in members:
            cluster_by_row_idx[int(member_idx)] = cluster_id

    out = valid_unique.copy()
    out["np_butina_cluster"] = [cluster_by_row_idx[idx] for idx in range(len(out))]
    return out.sort_values(["np_butina_cluster", "compound_key", "canonical_smiles"]).reset_index(drop=True)


def _cluster_size_tables(clustered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_sizes = (
        clustered.groupby("np_butina_cluster", dropna=False)
        .size()
        .rename("cluster_size")
        .reset_index()
        .sort_values(["cluster_size", "np_butina_cluster"], ascending=[False, True])
        .reset_index(drop=True)
    )
    size_distribution = (
        cluster_sizes.groupby("cluster_size", dropna=False)
        .size()
        .rename("n_clusters")
        .reset_index()
        .sort_values("cluster_size")
        .reset_index(drop=True)
    )
    return cluster_sizes, size_distribution


def main() -> None:
    args = parse_args()
    output_path = args.output_path
    invalid_output_path = args.invalid_output_path or _default_sidecar_path(output_path, "_invalid_smiles.tsv")
    report_path = args.report_path or _default_sidecar_path(output_path, "_report.json")
    cluster_sizes_path = args.cluster_sizes_path or _default_sidecar_path(output_path, "_cluster_sizes.tsv")

    products, metadata = _load_pair_products(args.pairs_path)
    valid_all, invalid_df = _canonicalize_products(products)
    valid_unique = _deduplicate_compounds(valid_all)
    clustered = _compute_butina_clusters(
        valid_unique,
        radius=args.radius,
        n_bits=args.n_bits,
        dist_thresh=args.dist_thresh,
    )
    cluster_sizes, size_distribution = _cluster_size_tables(clustered)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_sizes_path.parent.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "product_id",
        "compound_name",
        "compound_key",
        "compound_key_source",
        "smiles",
        "canonical_smiles",
        "np_butina_cluster",
    ]
    clustered[output_columns].to_csv(output_path, sep="\t", index=False)
    invalid_df.to_csv(invalid_output_path, sep="\t", index=False)
    cluster_sizes.to_csv(cluster_sizes_path, sep="\t", index=False)

    cluster_size_values = cluster_sizes["cluster_size"].astype(int).tolist()
    report = {
        **metadata,
        "output_path": str(output_path),
        "invalid_smiles_path": str(invalid_output_path),
        "cluster_sizes_path": str(cluster_sizes_path),
        "radius": int(args.radius),
        "n_bits": int(args.n_bits),
        "dist_thresh": float(args.dist_thresh),
        "tanimoto_similarity_threshold": float(1.0 - args.dist_thresh),
        "n_unique_compounds_before_filtering": int(len(products)),
        "n_valid_unique_compounds": int(len(valid_unique)),
        "n_invalid_smiles": int(len(invalid_df)),
        "n_butina_clusters": int(clustered["np_butina_cluster"].nunique()),
        "n_singleton_clusters": int((cluster_sizes["cluster_size"] == 1).sum()),
        "largest_cluster_size": int(max(cluster_size_values)) if cluster_size_values else 0,
        "median_cluster_size": float(np.median(cluster_size_values)) if cluster_size_values else 0.0,
        "cluster_size_distribution": [
            {"cluster_size": int(row.cluster_size), "n_clusters": int(row.n_clusters)}
            for row in size_distribution.itertuples(index=False)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
