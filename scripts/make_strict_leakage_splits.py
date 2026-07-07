from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip_core.logging import setup_logger

try:
    from rdkit import Chem, DataStructs
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "RDKit is required for NP Butina clustering. Run this script in the environment "
        "used for MIBiG feature caching, e.g. `conda run -n combi python ...`."
    ) from exc


LOGGER = setup_logger("make_strict_leakage_splits")
RDLogger.DisableLog("rdApp.warning")


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def components(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in list(self.parent):
            grouped[self.find(item)].append(item)
        return dict(grouped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create strict leakage-aware MIBiG splits using BiG-SCAPE BGC families "
            "and NP-side Butina clusters."
        )
    )
    parser.add_argument("--pairs_path", type=Path, default=Path("data/MIBIG/processed/mibig_pairs.tsv"))
    parser.add_argument(
        "--bigscape_output_dir",
        type=Path,
        default=Path(
            "data/MIBIG/mibig_bigscape_clustered/output_files/"
            "2026-06-24_18-27-02_c0.3"
        ),
        help="BiG-SCAPE output directory containing record_annotations.tsv and *_clustering_c0.3.tsv files.",
    )
    parser.add_argument(
        "--record_annotations_path",
        type=Path,
        default=None,
        help="Optional explicit record_annotations.tsv path. Defaults to <bigscape_output_dir>/record_annotations.tsv.",
    )
    parser.add_argument(
        "--bigscape_clustering_path",
        type=Path,
        default=None,
        help="Deprecated compatibility option. If provided, its parents are used to infer --bigscape_output_dir.",
    )
    parser.add_argument("--splits_dir", type=Path, default=Path("data/MIBIG/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", choices=("random", "cv"), default="random")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=0.3,
        help="Butina distance threshold. 0.3 corresponds to Tanimoto similarity >= 0.7.",
    )
    parser.add_argument(
        "--fail_on_missing_bigscape",
        action="store_true",
        help=(
            "Fail if a paired BGC is absent from record_annotations.tsv. "
            "Processed records absent from clustering TSVs are still assigned singleton families."
        ),
    )
    parser.add_argument("--prefix", type=str, default="strict_bigscape_butina")
    return parser.parse_args()


def _infer_compound_id(row: pd.Series) -> str:
    for column in ("compound_id", "canonical_smiles", "smiles", "compound_name"):
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    raise ValueError("Could not infer product identifier from pair row.")


def _load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"bgc_id", "smiles"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["bgc_id"] = df["bgc_id"].astype(str)
    df["smiles"] = df["smiles"].astype(str)
    df["compound_id"] = df.apply(_infer_compound_id, axis=1)
    return df.dropna(subset=["bgc_id", "compound_id", "smiles"]).reset_index(drop=True)


def _canonicalize_products(pair_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    product_rows = pair_df[["compound_id", "smiles"]].drop_duplicates().reset_index(drop=True)
    valid_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []
    canonical_by_compound: dict[str, str] = {}

    for row in product_rows.itertuples(index=False):
        product_id = str(row.compound_id)
        smiles = str(row.smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_rows.append({"compound_id": product_id, "smiles": smiles, "reason": "rdkit_parse_failed"})
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        valid_rows.append(
            {
                "compound_id": product_id,
                "smiles": smiles,
                "canonical_smiles": canonical,
            }
        )
        canonical_by_compound[product_id] = canonical

    valid_df = pd.DataFrame(valid_rows).drop_duplicates(subset=["compound_id", "smiles", "canonical_smiles"])
    invalid_df = pd.DataFrame(invalid_rows, columns=["compound_id", "smiles", "reason"])
    return valid_df, invalid_df, canonical_by_compound


def _cluster_unique_canonical_smiles(
    valid_products: pd.DataFrame,
    *,
    radius: int,
    n_bits: int,
    dist_thresh: float,
) -> pd.DataFrame:
    unique_smiles = sorted(valid_products["canonical_smiles"].drop_duplicates().astype(str).tolist())
    mols = [Chem.MolFromSmiles(smiles) for smiles in unique_smiles]
    if any(mol is None for mol in mols):
        raise ValueError("Internal error: canonical SMILES failed to parse during clustering.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    fps = [generator.GetFingerprint(mol) for mol in mols]

    distances: list[float] = []
    for i in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend([1.0 - float(similarity) for similarity in similarities])

    clusters = Butina.ClusterData(distances, len(fps), float(dist_thresh), isDistData=True)
    cluster_by_smiles: dict[str, str] = {}
    for cluster_idx, cluster_members in enumerate(clusters, start=1):
        cluster_id = f"NP_BUTINA_{cluster_idx:06d}"
        for member_idx in cluster_members:
            cluster_by_smiles[unique_smiles[int(member_idx)]] = cluster_id

    cluster_df = valid_products.copy()
    cluster_df["np_butina_cluster"] = cluster_df["canonical_smiles"].map(cluster_by_smiles)
    if bool(cluster_df["np_butina_cluster"].isna().any()):
        raise ValueError("Some products did not receive an NP Butina cluster assignment.")
    return cluster_df.sort_values(["np_butina_cluster", "compound_id", "canonical_smiles"]).reset_index(drop=True)


def _extract_bgc_id(value: str) -> str:
    match = re.search(r"(BGC\d+)", str(value))
    if not match:
        raise ValueError(f"Could not extract BGC accession from BiG-SCAPE value: {value}")
    return match.group(1)


def _resolve_bigscape_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.bigscape_clustering_path is not None:
        clustering_path = Path(args.bigscape_clustering_path)
        if clustering_path.name.endswith("_clustering_c0.3.tsv") and clustering_path.parent.parent.exists():
            output_dir = clustering_path.parent.parent
        else:
            output_dir = clustering_path.parent
    else:
        output_dir = Path(args.bigscape_output_dir)
    annotations_path = (
        Path(args.record_annotations_path)
        if args.record_annotations_path is not None
        else output_dir / "record_annotations.tsv"
    )
    return output_dir, annotations_path


def _load_bigscape_processed_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"Record", "GBK"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df[["Record", "GBK"]].copy()
    out["bgc_id"] = out["GBK"].map(_extract_bgc_id)
    out["bigscape_record"] = out["Record"].astype(str)
    out["bigscape_gbk"] = out["GBK"].astype(str)
    return out[["bgc_id", "bigscape_record", "bigscape_gbk"]].drop_duplicates(subset=["bgc_id"]).reset_index(drop=True)


def _load_all_bigscape_cluster_assignments(output_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(output_dir.rglob("*_clustering_c0.3.tsv"))
    rows: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, sep="\t")
        required = {"GBK", "Family"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        out = df[["GBK", "Family"]].copy()
        out["bgc_id"] = out["GBK"].map(_extract_bgc_id)
        rel_parent = str(path.parent.relative_to(output_dir))
        out["bigscape_family"] = rel_parent + "::" + out["Family"].astype(str)
        out["bigscape_clustering_file"] = str(path)
        rows.append(out[["bgc_id", "bigscape_family", "bigscape_clustering_file"]])
    if not rows:
        return pd.DataFrame(columns=["bgc_id", "bigscape_family", "bigscape_clustering_file"]), []
    merged = pd.concat(rows, ignore_index=True)
    return merged.drop_duplicates().reset_index(drop=True), [str(path) for path in paths]


def _build_bigscape_family_table(
    pair_df: pd.DataFrame,
    processed_records: pd.DataFrame,
    clustering_assignments: pd.DataFrame,
    *,
    allow_unprocessed: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paired_bgcs = sorted(pair_df["bgc_id"].astype(str).unique().tolist())
    processed_ids = set(processed_records["bgc_id"].astype(str).tolist())

    if clustering_assignments.empty:
        explicit = pd.DataFrame(columns=["bgc_id", "bigscape_family", "bigscape_clustering_file"])
        duplicate_explicit_bgcs: list[str] = []
    else:
        explicit = clustering_assignments.sort_values(["bgc_id", "bigscape_family"]).copy()
        duplicate_counts = explicit.groupby("bgc_id")["bigscape_family"].nunique()
        duplicate_explicit_bgcs = sorted(duplicate_counts[duplicate_counts > 1].index.astype(str).tolist())
        explicit = explicit.drop_duplicates(subset=["bgc_id"], keep="first")

    table = processed_records.merge(explicit, on="bgc_id", how="left")
    table["bigscape_family_source"] = "explicit_clustering"
    singleton_mask = table["bigscape_family"].isna()
    table.loc[singleton_mask, "bigscape_family"] = table.loc[singleton_mask, "bgc_id"].map(
        lambda bgc_id: f"SINGLETON_{bgc_id}"
    )
    table.loc[singleton_mask, "bigscape_family_source"] = "processed_singleton"

    unprocessed_bgcs = [bgc_id for bgc_id in paired_bgcs if bgc_id not in processed_ids]
    if unprocessed_bgcs and not allow_unprocessed:
        preview = ", ".join(unprocessed_bgcs[:10])
        raise ValueError(
            f"{len(unprocessed_bgcs)} paired BGCs are absent from record_annotations.tsv and appear unprocessed by "
            f"BiG-SCAPE. Examples: {preview}. Rerun without --fail_on_missing_bigscape to assign unprocessed "
            f"singleton families, or rerun BiG-SCAPE with these records."
        )
    if unprocessed_bgcs:
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    {
                        "bgc_id": unprocessed_bgcs,
                        "bigscape_record": [None] * len(unprocessed_bgcs),
                        "bigscape_gbk": [None] * len(unprocessed_bgcs),
                        "bigscape_family": [f"UNPROCESSED_BIGSCAPE_{bgc_id}" for bgc_id in unprocessed_bgcs],
                        "bigscape_clustering_file": [None] * len(unprocessed_bgcs),
                        "bigscape_family_source": ["unprocessed_singleton"] * len(unprocessed_bgcs),
                    }
                ),
            ],
            ignore_index=True,
        )

    paired_table = table[table["bgc_id"].isin(paired_bgcs)].copy()
    explicit_paired = paired_table[paired_table["bigscape_family_source"] == "explicit_clustering"]
    singleton_paired = paired_table[paired_table["bigscape_family_source"] == "processed_singleton"]
    report = {
        "n_paired_bgcs": int(len(paired_bgcs)),
        "n_matched_to_bigscape_processed_records": int(len(set(paired_bgcs).intersection(processed_ids))),
        "n_with_explicit_bigscape_family": int(explicit_paired["bgc_id"].nunique()),
        "n_assigned_singleton_family": int(singleton_paired["bgc_id"].nunique()),
        "n_still_missing_completely": int(len(unprocessed_bgcs)),
        "still_missing_completely_examples": unprocessed_bgcs[:20],
        "n_processed_records_total": int(processed_records["bgc_id"].nunique()),
        "n_explicit_clustered_bgcs_total": int(clustering_assignments["bgc_id"].nunique())
        if not clustering_assignments.empty
        else 0,
        "n_bgc_ids_with_multiple_explicit_families": int(len(duplicate_explicit_bgcs)),
        "multiple_explicit_family_examples": duplicate_explicit_bgcs[:20],
    }
    return table[["bgc_id", "bigscape_family", "bigscape_family_source"]].drop_duplicates(), report


def _assign_group_splits(
    group_sizes: dict[str, int],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> dict[str, str]:
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total_frac}")

    rng = random.Random(int(seed))
    groups = list(group_sizes)
    rng.shuffle(groups)
    groups.sort(key=lambda group: group_sizes[group], reverse=True)

    total = sum(group_sizes.values())
    targets = {
        "train": float(train_frac) * total,
        "val": float(val_frac) * total,
        "test": float(test_frac) * total,
    }
    current = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, str] = {}

    for group in groups:
        split = min(
            ("train", "val", "test"),
            key=lambda name: current[name] / max(targets[name], 1e-9),
        )
        assignments[group] = split
        current[split] += int(group_sizes[group])
    return assignments


def _assign_group_folds(group_sizes: dict[str, int], *, seed: int, n_folds: int) -> dict[str, int]:
    if int(n_folds) < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    rng = random.Random(int(seed))
    groups = list(group_sizes)
    rng.shuffle(groups)
    groups.sort(key=lambda group: group_sizes[group], reverse=True)
    fold_loads = {fold_id: 0 for fold_id in range(1, int(n_folds) + 1)}
    assignments: dict[str, int] = {}
    for group in groups:
        fold_id = min(fold_loads, key=lambda fold: fold_loads[fold])
        assignments[group] = int(fold_id)
        fold_loads[fold_id] += int(group_sizes[group])
    return assignments


def _check_no_leakage(split_df: pd.DataFrame, column: str, assignment_column: str) -> dict[str, Any]:
    split_counts = split_df.groupby(column)[assignment_column].nunique()
    leaked = split_counts[split_counts > 1]
    examples = []
    if not leaked.empty:
        for value in leaked.index[:10]:
            examples.append(
                {
                    column: str(value),
                    assignment_column: sorted(
                        str(item) for item in split_df.loc[split_df[column] == value, assignment_column].unique().tolist()
                    ),
                }
            )
    return {
        "column": column,
        "assignment_column": assignment_column,
        "ok": bool(leaked.empty),
        "n_leaked": int(len(leaked)),
        "examples": examples,
    }


def _build_leakage_split(
    pair_df: pd.DataFrame,
    product_clusters: pd.DataFrame,
    bigscape_families: pd.DataFrame,
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    split_mode: str,
    n_folds: int,
    allow_missing_bigscape: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    product_map = product_clusters[["compound_id", "canonical_smiles", "np_butina_cluster"]].drop_duplicates()
    merged = pair_df.merge(product_map, on="compound_id", how="inner")
    merged = merged.merge(bigscape_families, on="bgc_id", how="left")
    missing_family = sorted(merged.loc[merged["bigscape_family"].isna(), "bgc_id"].astype(str).unique().tolist())
    if missing_family and not allow_missing_bigscape:
        preview = ", ".join(missing_family[:10])
        raise ValueError(
            f"{len(missing_family)} paired BGCs are missing from the prepared BiG-SCAPE family table. "
            f"Examples: {preview}."
        )
    merged["bigscape_family"] = merged.apply(
        lambda row: f"UNPROCESSED_BIGSCAPE_{row.bgc_id}"
        if pd.isna(row.bigscape_family)
        else str(row.bigscape_family),
        axis=1,
    )
    if "bigscape_family_source" not in merged.columns:
        merged["bigscape_family_source"] = "unknown"
    merged["bigscape_family_source"] = merged["bigscape_family_source"].fillna("unprocessed_singleton")

    dsu = DisjointSet()
    for row in merged[["bigscape_family", "np_butina_cluster"]].drop_duplicates().itertuples(index=False):
        family_node = f"BGC_FAMILY::{row.bigscape_family}"
        np_node = f"NP_CLUSTER::{row.np_butina_cluster}"
        dsu.union(family_node, np_node)

    component_by_node: dict[str, str] = {}
    for component_idx, nodes in enumerate(dsu.components().values(), start=1):
        component_id = f"LEAKAGE_GROUP_{component_idx:06d}"
        for node in nodes:
            component_by_node[node] = component_id

    merged["leakage_group"] = merged["bigscape_family"].map(
        lambda family: component_by_node[f"BGC_FAMILY::{family}"]
    )
    group_sizes = merged.groupby("leakage_group").size().astype(int).to_dict()
    if split_mode == "random":
        split_by_group = _assign_group_splits(
            group_sizes,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )
        merged["split"] = merged["leakage_group"].map(split_by_group)
        assignment_column = "split"
        output_columns = [
            "bgc_id",
            "compound_id",
            "smiles",
            "canonical_smiles",
            "split",
            "leakage_group",
            "bigscape_family",
            "bigscape_family_source",
            "np_butina_cluster",
        ]
        sort_columns = ["split", "leakage_group", "bgc_id"]
    elif split_mode == "cv":
        fold_by_group = _assign_group_folds(group_sizes, seed=seed, n_folds=n_folds)
        merged["fold_id"] = merged["leakage_group"].map(fold_by_group).astype(int)
        assignment_column = "fold_id"
        output_columns = [
            "bgc_id",
            "compound_id",
            "smiles",
            "canonical_smiles",
            "fold_id",
            "leakage_group",
            "bigscape_family",
            "bigscape_family_source",
            "np_butina_cluster",
        ]
        sort_columns = ["fold_id", "leakage_group", "bgc_id"]
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    split_df = merged[output_columns].drop_duplicates(subset=["bgc_id", "compound_id"]).sort_values(sort_columns)

    family_check = _check_no_leakage(split_df, "bigscape_family", assignment_column)
    np_check = _check_no_leakage(split_df, "np_butina_cluster", assignment_column)
    if not family_check["ok"] or not np_check["ok"]:
        raise ValueError(
            f"Strict leakage split failed sanity checks: family={family_check}, np_cluster={np_check}"
        )

    report = {
        "split_mode": split_mode,
        "n_pairs": int(len(split_df)),
        "n_bgcs": int(split_df["bgc_id"].nunique()),
        "n_products": int(split_df["compound_id"].nunique()),
        "n_bigscape_families": int(split_df["bigscape_family"].nunique()),
        "n_np_butina_clusters": int(split_df["np_butina_cluster"].nunique()),
        "n_leakage_groups": int(split_df["leakage_group"].nunique()),
        f"{assignment_column}_pair_counts": {
            str(k): int(v) for k, v in split_df[assignment_column].value_counts().sort_index().items()
        },
        f"{assignment_column}_bgc_counts": {
            str(k): int(v) for k, v in split_df.groupby(assignment_column)["bgc_id"].nunique().sort_index().items()
        },
        f"{assignment_column}_product_counts": {
            str(k): int(v) for k, v in split_df.groupby(assignment_column)["compound_id"].nunique().sort_index().items()
        },
        "bigscape_family_source_counts": {
            str(k): int(v) for k, v in split_df.groupby("bigscape_family_source")["bgc_id"].nunique().sort_index().items()
        },
        "unexpected_missing_bigscape_family_table_rows": {
            "allowed": bool(allow_missing_bigscape),
            "n_bgcs": int(len(missing_family)),
            "examples": missing_family[:20],
        },
        "sanity_checks": {
            f"bigscape_family_single_{assignment_column}": family_check,
            f"np_butina_cluster_single_{assignment_column}": np_check,
        },
    }
    return split_df.reset_index(drop=True), report


def main() -> None:
    args = parse_args()
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    pair_df = _load_pairs(args.pairs_path)
    valid_products, invalid_products, _canonical_by_compound = _canonicalize_products(pair_df)
    product_clusters = _cluster_unique_canonical_smiles(
        valid_products,
        radius=int(args.radius),
        n_bits=int(args.n_bits),
        dist_thresh=float(args.dist_thresh),
    )
    bigscape_output_dir, record_annotations_path = _resolve_bigscape_paths(args)
    processed_records = _load_bigscape_processed_records(record_annotations_path)
    clustering_assignments, clustering_paths = _load_all_bigscape_cluster_assignments(bigscape_output_dir)
    bigscape_families, bigscape_report = _build_bigscape_family_table(
        pair_df,
        processed_records,
        clustering_assignments,
        allow_unprocessed=not bool(args.fail_on_missing_bigscape),
    )
    split_df, report = _build_leakage_split(
        pair_df,
        product_clusters,
        bigscape_families,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        split_mode=str(args.split_mode),
        n_folds=int(args.n_folds),
        allow_missing_bigscape=not bool(args.fail_on_missing_bigscape),
    )

    cluster_path = args.splits_dir / f"{args.prefix}_np_butina_clusters.tsv"
    invalid_path = args.splits_dir / f"{args.prefix}_invalid_smiles.tsv"
    if args.split_mode == "random":
        split_path = args.splits_dir / f"{args.prefix}_random_seed{args.seed}.tsv"
        report_path = args.splits_dir / f"{args.prefix}_random_seed{args.seed}_report.json"
    else:
        split_path = args.splits_dir / f"{args.prefix}_cv_seed{args.seed}_n{args.n_folds}.tsv"
        report_path = args.splits_dir / f"{args.prefix}_cv_seed{args.seed}_n{args.n_folds}_report.json"
    product_clusters.to_csv(cluster_path, sep="\t", index=False)
    invalid_products.to_csv(invalid_path, sep="\t", index=False)
    split_df.to_csv(split_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    report.update(
        {
            "pairs_path": str(args.pairs_path),
            "bigscape_output_dir": str(bigscape_output_dir),
            "record_annotations_path": str(record_annotations_path),
            "bigscape_clustering_paths": clustering_paths,
            "bigscape_family_assignment": bigscape_report,
            "np_butina_clusters_path": str(cluster_path),
            "invalid_smiles_path": str(invalid_path),
            "split_path": str(split_path),
            "seed": int(args.seed),
            "butina": {
                "radius": int(args.radius),
                "n_bits": int(args.n_bits),
                "dist_thresh": float(args.dist_thresh),
                "similarity_threshold": float(1.0 - float(args.dist_thresh)),
            },
            "n_invalid_smiles": int(len(invalid_products)),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote NP Butina clusters to %s", cluster_path)
    LOGGER.info("Wrote invalid SMILES report to %s", invalid_path)
    LOGGER.info("Wrote strict leakage split to %s", split_path)
    LOGGER.info("Wrote strict leakage report to %s", report_path)


if __name__ == "__main__":
    main()
