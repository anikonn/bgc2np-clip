from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
except ImportError:  # pragma: no cover
    Chem = None


BGC_RE = re.compile(r"(BGC\d+)")
SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict BGC-family/NP-cluster leakage-aware splits.")
    parser.add_argument("--pairs_path", type=Path, default=Path("data/MIBIG/processed/mibig_pairs.tsv"))
    parser.add_argument("--bigscape_path", type=Path, default=Path("data/MIBIG/processed/bigscape_clustering.tsv"))
    parser.add_argument(
        "--butina_path",
        type=Path,
        default=Path("data/MIBIG/processed/mibig_np_butina_clusters_tanimoto0.7.tsv"),
    )
    parser.add_argument("--out_dir", type=Path, default=Path("data/MIBIG/processed/strict_splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=10)
    return parser.parse_args()


def normalize_bgc_id(value: Any) -> str | None:
    match = BGC_RE.search(str(value))
    return match.group(1) if match else None


def canonicalize_smiles(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if Chem is None:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def first_available(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_pairs(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, sep="\t")
    print(f"Paired-table columns: {list(df.columns)}")
    if "bgc_id" not in df.columns:
        raise ValueError(f"{path} must contain bgc_id.")
    df = df.copy()
    df["bgc_id_original"] = df["bgc_id"].astype(str)
    df["bgc_id"] = df["bgc_id"].map(normalize_bgc_id)
    df = df[df["bgc_id"].notna()].reset_index(drop=True)

    compound_key = first_available(
        list(df.columns),
        ("canonical_smiles", "smiles", "compound_id", "product_id", "np_id", "compound_name", "compound_idx"),
    )
    if compound_key is None:
        raise ValueError("Could not identify a compound/product key in paired table.")
    df["compound_key"] = df[compound_key].astype(str)
    return df, compound_key


def load_bigscape(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"#BGC Name", "Family Number"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df.rename(columns={"#BGC Name": "bgc_id", "Family Number": "bgc_bigscape_family"}).copy()
    out["bgc_id"] = out["bgc_id"].map(normalize_bgc_id)
    out = out[out["bgc_id"].notna()].copy()
    out["bgc_bigscape_family"] = "BIGSCAPE_FAMILY::" + out["bgc_bigscape_family"].astype(str)
    return out[["bgc_id", "bgc_bigscape_family"]].drop_duplicates(subset=["bgc_id"], keep="first")


def choose_butina_merge(
    pairs: pd.DataFrame,
    butina: pd.DataFrame,
    pair_compound_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    print(f"Butina-table columns: {list(butina.columns)}")
    butina_cols = list(butina.columns)
    if "np_butina_cluster" not in butina.columns:
        raise ValueError("Butina table must contain np_butina_cluster.")

    pairs = pairs.copy()
    butina = butina.copy()

    if "canonical_smiles" in butina.columns:
        if "canonical_smiles" in pairs.columns:
            print("Chosen NP merge key: canonical_smiles (present in both tables)")
            return pairs, butina, "canonical_smiles", "canonical_smiles", "canonical_smiles"
        if "smiles" in pairs.columns and Chem is not None:
            pairs["_strict_merge_canonical_smiles"] = pairs["smiles"].map(canonicalize_smiles)
            butina["_strict_merge_canonical_smiles"] = butina["canonical_smiles"].astype(str)
            print("Chosen NP merge key: canonical_smiles (derived from paired-table smiles with RDKit)")
            return (
                pairs,
                butina,
                "_strict_merge_canonical_smiles",
                "_strict_merge_canonical_smiles",
                "_strict_merge_canonical_smiles",
            )

    if "smiles" in pairs.columns and "smiles" in butina.columns:
        print("Chosen NP merge key: smiles")
        return pairs, butina, "smiles", "smiles", "smiles"

    id_candidates = ("compound_id", "product_id", "np_id", "compound_idx", "compound_name", "compound_key")
    pair_key = first_available(list(pairs.columns), id_candidates) or pair_compound_key
    butina_key = first_available(butina_cols, id_candidates)
    if butina_key is None:
        raise ValueError("Could not identify a Butina compound/product ID key.")
    print(f"Chosen NP merge key: paired-table {pair_key} -> Butina-table {butina_key}")
    pairs["_strict_merge_compound_id"] = pairs[pair_key].astype(str)
    butina["_strict_merge_compound_id"] = butina[butina_key].astype(str)
    return pairs, butina, "_strict_merge_compound_id", "_strict_merge_compound_id", "_strict_merge_compound_id"


def fallback_np_key(row: pd.Series) -> str:
    for col in ("canonical_smiles", "_strict_merge_canonical_smiles", "smiles", "compound_key", "compound_name", "compound_idx"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return "UNKNOWN_COMPOUND"


def coverage_summary(df: pd.DataFrame, pair_compound_key: str) -> dict[str, Any]:
    compound_col = "_strict_merge_canonical_smiles" if "_strict_merge_canonical_smiles" in df.columns else pair_compound_key
    return {
        "n_pair_rows": int(len(df)),
        "n_unique_bgcs": int(df["bgc_id"].nunique()),
        "compound_identifier_column": compound_col,
        "n_unique_compounds": int(df[compound_col].nunique(dropna=True)),
        "n_paired_bgcs_with_bigscape_family": int(df.loc[~df["bgc_family_missing"], "bgc_id"].nunique()),
        "n_paired_bgcs_missing_bigscape_family": int(df.loc[df["bgc_family_missing"], "bgc_id"].nunique()),
        "n_paired_compounds_with_butina_cluster": int(df.loc[~df["np_cluster_missing"], compound_col].nunique()),
        "n_paired_compounds_missing_butina_cluster": int(df.loc[df["np_cluster_missing"], compound_col].nunique()),
        "n_bgc_singleton_fallback_rows": int(df["bgc_family_missing"].sum()),
        "n_np_singleton_fallback_rows": int(df["np_cluster_missing"].sum()),
        "n_bgc_singleton_fallback_bgcs": int(df.loc[df["bgc_family_missing"], "bgc_id"].nunique()),
        "n_np_singleton_fallback_compounds": int(df.loc[df["np_cluster_missing"], compound_col].nunique()),
    }


def build_leakage_groups(df: pd.DataFrame) -> pd.DataFrame:
    graph = nx.Graph()
    for row in df[["bgc_bigscape_family", "np_butina_cluster"]].drop_duplicates().itertuples(index=False):
        bgc_node = f"BGC::{row.bgc_bigscape_family}"
        np_node = f"NP::{row.np_butina_cluster}"
        graph.add_edge(bgc_node, np_node)

    component_by_node: dict[str, str] = {}
    components = sorted(nx.connected_components(graph), key=lambda nodes: sorted(nodes)[0])
    for idx, nodes in enumerate(components, start=1):
        leakage_group = f"LEAKAGE_GROUP_{idx:06d}"
        for node in nodes:
            component_by_node[node] = leakage_group

    out = df.copy()
    out["leakage_group"] = out["bgc_bigscape_family"].map(lambda family: component_by_node[f"BGC::{family}"])
    return out


def leakage_group_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats = (
        df.groupby("leakage_group")
        .agg(
            n_pairs=("leakage_group", "size"),
            n_bgcs=("bgc_id", "nunique"),
            n_compounds=("compound_key", "nunique"),
            n_bgc_families=("bgc_bigscape_family", "nunique"),
            n_np_clusters=("np_butina_cluster", "nunique"),
        )
        .reset_index()
        .sort_values(["n_pairs", "n_bgcs", "n_compounds"], ascending=False)
    )
    summary = {
        "n_leakage_groups": int(len(stats)),
        "pair_row_size_distribution": stats["n_pairs"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
        "unique_bgc_size_distribution": stats["n_bgcs"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
        "unique_compound_size_distribution": stats["n_compounds"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
    }
    return stats, summary


def assign_train_val_test(group_stats: pd.DataFrame, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    records = group_stats[["leakage_group", "n_pairs"]].to_dict("records")
    rng.shuffle(records)
    records.sort(key=lambda row: int(row["n_pairs"]), reverse=True)
    total = sum(int(row["n_pairs"]) for row in records)
    targets = {split: ratio * total for split, ratio in TARGET_RATIOS.items()}
    loads = {split: 0 for split in SPLITS}
    assigned: dict[str, str] = {}
    for row in records:
        group = str(row["leakage_group"])
        size = int(row["n_pairs"])
        split = min(SPLITS, key=lambda item: (loads[item] + size - targets[item]) ** 2 - (loads[item] - targets[item]) ** 2)
        assigned[group] = split
        loads[split] += size
    return assigned


def assign_cv10(group_stats: pd.DataFrame, seed: int, n_folds: int) -> dict[str, int]:
    rng = random.Random(seed)
    records = group_stats[["leakage_group", "n_pairs"]].to_dict("records")
    rng.shuffle(records)
    records.sort(key=lambda row: int(row["n_pairs"]), reverse=True)
    loads = {fold: 0 for fold in range(n_folds)}
    assigned: dict[str, int] = {}
    for row in records:
        fold = min(range(n_folds), key=lambda item: (loads[item], item))
        assigned[str(row["leakage_group"])] = fold
        loads[fold] += int(row["n_pairs"])
    return assigned


def split_summary(df: pd.DataFrame, split_col: str, split_name_col: str) -> pd.DataFrame:
    rows = []
    for split, split_df in df.groupby(split_col):
        rows.append(
            {
                split_name_col: split,
                "n_pairs": len(split_df),
                "n_unique_bgcs": split_df["bgc_id"].nunique(),
                "n_unique_compounds": split_df["compound_key"].nunique(),
                "n_bgc_families": split_df["bgc_bigscape_family"].nunique(),
                "n_np_clusters": split_df["np_butina_cluster"].nunique(),
                "n_leakage_groups": split_df["leakage_group"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def cv_summary(df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    rows = []
    for fold in range(n_folds):
        test = df[df["strict_cv10_fold"] == fold]
        train = df[df["strict_cv10_fold"] != fold]
        rows.append(
            {
                "fold": fold,
                "n_test_pairs": len(test),
                "n_train_pairs": len(train),
                "n_test_unique_bgcs": test["bgc_id"].nunique(),
                "n_test_unique_compounds": test["compound_key"].nunique(),
                "n_test_bgc_families": test["bgc_bigscape_family"].nunique(),
                "n_test_np_clusters": test["np_butina_cluster"].nunique(),
                "n_test_leakage_groups": test["leakage_group"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def overlap_values(left: pd.DataFrame, right: pd.DataFrame, column: str) -> set[str]:
    return set(left[column].dropna().astype(str)).intersection(set(right[column].dropna().astype(str)))


def pair_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["bgc_id"].astype(str), df["compound_key"].astype(str), strict=False))


def check_partition_leakage(df: pd.DataFrame, split_col: str) -> list[dict[str, Any]]:
    checks = []
    for column in ("leakage_group", "bgc_bigscape_family", "np_butina_cluster"):
        counts = df.groupby(column)[split_col].nunique()
        leaked = counts[counts > 1]
        checks.append(
            {
                "scope": split_col,
                "column": column,
                "n_overlap": int(len(leaked)),
                "examples": [str(x) for x in leaked.index[:10]],
            }
        )
    return checks


def check_cv_leakage(df: pd.DataFrame, n_folds: int) -> list[dict[str, Any]]:
    checks = []
    for fold in range(n_folds):
        train = df[df["strict_cv10_fold"] != fold]
        test = df[df["strict_cv10_fold"] == fold]
        for column in ("leakage_group", "bgc_bigscape_family", "np_butina_cluster"):
            overlap = overlap_values(train, test, column)
            checks.append(
                {
                    "fold": fold,
                    "column": column,
                    "n_overlap": len(overlap),
                    "examples": sorted(overlap)[:10],
                }
            )
    return checks


def exact_object_diagnostics(df: pd.DataFrame, mode: str, n_folds: int | None = None) -> list[dict[str, Any]]:
    diagnostics = []
    if mode == "split":
        split_pairs = list(combinations(SPLITS, 2))
        for left_name, right_name in split_pairs:
            left = df[df["strict_split"] == left_name]
            right = df[df["strict_split"] == right_name]
            diagnostics.extend(_exact_between(left, right, f"{left_name}_vs_{right_name}"))
    else:
        assert n_folds is not None
        for fold in range(n_folds):
            train = df[df["strict_cv10_fold"] != fold]
            test = df[df["strict_cv10_fold"] == fold]
            diagnostics.extend(_exact_between(train, test, f"cv_fold_{fold}_train_vs_test"))
    return diagnostics


def _exact_between(left: pd.DataFrame, right: pd.DataFrame, scope: str) -> list[dict[str, Any]]:
    bgcs = overlap_values(left, right, "bgc_id")
    compounds = overlap_values(left, right, "compound_key")
    pairs = pair_keys(left).intersection(pair_keys(right))
    return [
        {"scope": scope, "object": "bgc_id", "n_overlap": len(bgcs), "examples": sorted(bgcs)[:10]},
        {"scope": scope, "object": "compound", "n_overlap": len(compounds), "examples": sorted(compounds)[:10]},
        {
            "scope": scope,
            "object": "bgc_np_pair",
            "n_overlap": len(pairs),
            "examples": [f"{bgc}::{compound}" for bgc, compound in sorted(pairs)[:10]],
        },
    ]


def write_sanity(path: Path, coverage: dict[str, Any], group_summary: dict[str, Any], checks: dict[str, Any]) -> None:
    lines = ["COVERAGE SUMMARY", json.dumps(coverage, indent=2), "", "LEAKAGE GROUP SUMMARY", json.dumps(group_summary, indent=2), ""]
    lines.append("SANITY CHECKS")
    for section, values in checks.items():
        lines.append(f"\n[{section}]")
        lines.append(json.dumps(values, indent=2))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_long_cv_table(df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    id_cols = [col for col in ("bgc_id", "compound_key", "compound_name", "smiles") if col in df.columns]
    base_cols = id_cols + ["bgc_bigscape_family", "np_butina_cluster", "leakage_group", "strict_cv10_fold"]
    frames = []
    for fold in range(n_folds):
        part = df[base_cols].copy()
        part["cv_fold"] = fold
        part["cv_split"] = ["test" if value == fold else "train" for value in part["strict_cv10_fold"]]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs, pair_compound_key = load_pairs(args.pairs_path)
    bigscape = load_bigscape(args.bigscape_path)
    butina_raw = pd.read_csv(args.butina_path, sep="\t")
    pairs, butina, pair_merge_key, butina_merge_key, merge_label = choose_butina_merge(pairs, butina_raw, pair_compound_key)
    print(f"Using paired-table merge column: {pair_merge_key}")
    print(f"Using Butina-table merge column: {butina_merge_key}")

    butina_map = butina[[butina_merge_key, "np_butina_cluster"]].dropna().drop_duplicates(subset=[butina_merge_key])
    merged = pairs.merge(bigscape, on="bgc_id", how="left")
    merged = merged.merge(butina_map, left_on=pair_merge_key, right_on=butina_merge_key, how="left")
    if pair_merge_key in merged.columns:
        merged["compound_key"] = merged[pair_merge_key].fillna(merged["compound_key"]).astype(str)

    merged["bgc_family_missing"] = merged["bgc_bigscape_family"].isna()
    merged["np_cluster_missing"] = merged["np_butina_cluster"].isna()
    merged.loc[merged["bgc_family_missing"], "bgc_bigscape_family"] = merged.loc[
        merged["bgc_family_missing"], "bgc_id"
    ].map(lambda value: f"BGC_SINGLETON::{value}")
    merged.loc[merged["np_cluster_missing"], "np_butina_cluster"] = merged.loc[
        merged["np_cluster_missing"]
    ].apply(lambda row: f"NP_SINGLETON::{fallback_np_key(row)}", axis=1)

    coverage = coverage_summary(merged, pair_compound_key)
    coverage["np_merge_key"] = merge_label
    print("Coverage summary:")
    print(json.dumps(coverage, indent=2))

    with_groups = build_leakage_groups(merged)
    group_stats, group_summary = leakage_group_stats(with_groups)
    print("Leakage group summary:")
    print(json.dumps(group_summary, indent=2))

    split_assignments = assign_train_val_test(group_stats, args.seed)
    tvt = with_groups.copy()
    tvt["strict_split"] = tvt["leakage_group"].map(split_assignments)

    fold_assignments = assign_cv10(group_stats, args.seed, args.n_folds)
    cv = with_groups.copy()
    cv["strict_cv10_fold"] = cv["leakage_group"].map(fold_assignments).astype(int)

    tvt_summary = split_summary(tvt, "strict_split", "split")
    cv10_summary = cv_summary(cv, args.n_folds)
    print("Train/validation/test summary:")
    print(tvt_summary.to_string(index=False))
    print("CV10 fold balance summary:")
    print(cv10_summary.to_string(index=False))

    missing_report = pd.DataFrame(
        [
            {"missing_type": "bgc_bigscape_family", "n_rows": int(merged["bgc_family_missing"].sum()), "n_objects": coverage["n_bgc_singleton_fallback_bgcs"]},
            {"missing_type": "np_butina_cluster", "n_rows": int(merged["np_cluster_missing"].sum()), "n_objects": coverage["n_np_singleton_fallback_compounds"]},
        ]
    )
    sanity = {
        "train_val_test_cluster_checks": check_partition_leakage(tvt, "strict_split"),
        "cv10_cluster_checks": check_cv_leakage(cv, args.n_folds),
        "train_val_test_exact_object_diagnostics": exact_object_diagnostics(tvt, "split"),
        "cv10_exact_object_diagnostics": exact_object_diagnostics(cv, "cv", n_folds=args.n_folds),
    }
    cluster_failures = [
        item for section in ("train_val_test_cluster_checks", "cv10_cluster_checks") for item in sanity[section] if item["n_overlap"] != 0
    ]
    if cluster_failures:
        raise RuntimeError(f"Strict leakage sanity checks failed: {cluster_failures[:5]}")

    base_cols = [
        col
        for col in pairs.columns
        if not col.startswith("_strict_merge") and col not in {"bgc_id_original"}
    ]
    extra_cols = ["bgc_bigscape_family", "np_butina_cluster", "leakage_group"]
    with_groups[base_cols + extra_cols].to_csv(args.out_dir / "mibig_pairs_with_leakage_groups.tsv", sep="\t", index=False)
    tvt[base_cols + extra_cols + ["strict_split"]].to_csv(args.out_dir / "mibig_pairs_strict_train_val_test.tsv", sep="\t", index=False)
    cv[base_cols + extra_cols + ["strict_cv10_fold"]].to_csv(args.out_dir / "mibig_pairs_strict_cv10.tsv", sep="\t", index=False)
    make_long_cv_table(cv, args.n_folds).to_csv(args.out_dir / "mibig_pairs_strict_cv10_long.tsv", sep="\t", index=False)
    tvt_summary.to_csv(args.out_dir / "strict_train_val_test_summary.tsv", sep="\t", index=False)
    cv10_summary.to_csv(args.out_dir / "strict_cv10_summary.tsv", sep="\t", index=False)
    group_stats.head(20).to_csv(args.out_dir / "largest_leakage_groups.tsv", sep="\t", index=False)
    missing_report.to_csv(args.out_dir / "missing_cluster_report.tsv", sep="\t", index=False)
    write_sanity(args.out_dir / "strict_split_sanity_checks.txt", coverage, group_summary, sanity)

    print("Strict leakage sanity checks: OK")
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
