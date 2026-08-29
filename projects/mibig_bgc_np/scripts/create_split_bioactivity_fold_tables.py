from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts._bootstrap import ensure_src_path

ensure_src_path()

DEFAULT_CLASSES = ("antibacterial", "cytotoxic", "antifungal", "inhibitor", "siderophore", "antiviral")
DEFAULT_SPLITS = {
    "bgc": Path("data/MIBIG/splits/bgc_cv_seed42_n10.tsv"),
    "np": Path("data/MIBIG/splits/np_cv_seed42_n10.tsv"),
    "combined": Path("data/MIBIG/splits/combined_cv_seed42_n10.tsv"),
    "strict": Path("data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10.tsv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count observed bioactivity classes per CV fold for each split type.")
    parser.add_argument(
        "--bioactivity_table",
        type=Path,
        default=Path("results/EDA/bgc_observed_bioactivities.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/EDA/split_bioactivity_fold_counts"))
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--class_name", action="append", default=None, help="Observed bioactivity class to count.")
    parser.add_argument("--split", action="append", default=None, metavar="NAME=PATH")
    return parser.parse_args()


def _load_split_table(path: Path, n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "fold_id" in df.columns:
        fold_id = pd.to_numeric(df["fold_id"], errors="coerce")
    elif "strict_cv10_fold" in df.columns:
        fold_id = pd.to_numeric(df["strict_cv10_fold"], errors="coerce") + 1
    else:
        raise ValueError(f"Split file {path} must contain fold_id or strict_cv10_fold.")
    if bool(fold_id.isna().any()):
        raise ValueError(f"Split file {path} contains non-numeric fold assignments.")
    out = df.copy()
    out["bgc_id"] = out["bgc_id"].astype(str)
    out["fold_id"] = fold_id.astype(int)
    bad = sorted(set(out["fold_id"].tolist()).difference(range(1, int(n_folds) + 1)))
    if bad:
        raise ValueError(f"Split file {path} contains folds outside 1..{n_folds}: {bad}")
    return out


def _load_bioactivity_table(path: Path, classes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["bgc_id"] = df["bgc_id"].astype(str)
    labels = df["observed_bioactivities"].fillna("").astype(str).map(
        lambda text: {label.strip() for label in text.split(";") if label.strip()}
    )
    for class_name in classes:
        df[class_name] = labels.map(lambda values, name=class_name: name in values)
    return df[["bgc_id", *classes]].drop_duplicates(subset=["bgc_id"])


def _count_by_fold(assignments: pd.DataFrame, bioactivity: pd.DataFrame, classes: list[str], n_folds: int) -> pd.DataFrame:
    unique_assignments = assignments[["fold_id", "bgc_id"]].drop_duplicates().reset_index(drop=True)
    merged = unique_assignments.merge(bioactivity, on="bgc_id", how="left")
    for class_name in classes:
        merged[class_name] = merged[class_name].fillna(False).astype(bool)

    rows = []
    for fold_id in range(1, int(n_folds) + 1):
        fold_df = merged[merged["fold_id"] == fold_id]
        row: dict[str, Any] = {
            "fold_id": int(fold_id),
            "n_bgcs": int(fold_df["bgc_id"].nunique()),
        }
        for class_name in classes:
            row[class_name] = int(fold_df[class_name].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_split_bioactivity_fold_tables(
    bioactivity_table: Path,
    split_paths: dict[str, Path],
    classes: list[str],
    n_folds: int,
) -> dict[str, pd.DataFrame]:
    bioactivity = _load_bioactivity_table(bioactivity_table, classes)
    outputs: dict[str, pd.DataFrame] = {}
    for split_name, split_path in split_paths.items():
        split_df = _load_split_table(split_path, n_folds=n_folds)
        outputs[split_name] = _count_by_fold(split_df, bioactivity, classes, n_folds)
    return outputs


def _parse_split_args(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_SPLITS)
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --split value '{value}'. Expected NAME=PATH.")
        name, raw_path = value.split("=", 1)
        parsed[name.strip()] = Path(raw_path)
    return parsed


def main() -> None:
    args = parse_args()
    classes = list(args.class_name) if args.class_name else list(DEFAULT_CLASSES)
    split_paths = _parse_split_args(args.split)
    args.outdir.mkdir(parents=True, exist_ok=True)

    tables = build_split_bioactivity_fold_tables(
        bioactivity_table=args.bioactivity_table,
        split_paths=split_paths,
        classes=classes,
        n_folds=int(args.n_folds),
    )

    manifest: dict[str, Any] = {
        "bioactivity_table": str(args.bioactivity_table),
        "classes": classes,
        "splits": {name: str(path) for name, path in split_paths.items()},
        "outputs": {},
        "notes": {
            "counting_unit": "Distinct BGCs present in each fold.",
            "pair_level_splits": "For NP and strict pair-level split files, a BGC is counted once in a fold if any of its pair rows is assigned to that fold.",
            "multi_label": "A BGC can contribute to multiple class columns.",
        },
    }

    for split_name, table in tables.items():
        path = args.outdir / f"{split_name}_fold_bioactivity_counts.csv"
        table.to_csv(path, index=False)
        manifest["outputs"][split_name] = str(path)

    (args.outdir / "split_bioactivity_fold_counts_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
