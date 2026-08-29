from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts._bootstrap import ensure_src_path

ensure_src_path()

DEFAULT_SPLITS = {
    "bgc": Path("data/MIBIG/splits/bgc_cv_seed42_n10.tsv"),
    "np": Path("data/MIBIG/splits/np_cv_seed42_n10.tsv"),
    "combined": Path("data/MIBIG/splits/combined_cv_seed42_n10.tsv"),
    "strict": Path("data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10.tsv"),
}
LEVEL_COLUMNS = {
    "class": "npclassifier_class",
    "superclass": "npclassifier_superclass",
    "pathway": "npclassifier_pathway",
}


def _ensure_compound_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "compound_id" not in out.columns:
        if "compound_key" in out.columns:
            out["compound_id"] = out["compound_key"]
        elif "smiles" in out.columns:
            out["compound_id"] = out["smiles"]
        else:
            raise ValueError("Could not infer compound_id; expected compound_id, compound_key, or smiles column.")
    out["compound_id"] = out["compound_id"].astype(str)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count NPClassifier labels per CV fold for each split type.")
    parser.add_argument(
        "--pair_labels_path",
        type=Path,
        default=Path("data/MIBIG/processed/mibig_pairs_npclassifier_labels.tsv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/EDA/split_npclassifier_fold_counts"))
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--split", action="append", default=None, metavar="NAME=PATH")
    return parser.parse_args()


def _split_labels(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [label.strip() for label in str(value).split(";") if label.strip()]


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
    if any(column in out.columns for column in ("compound_id", "compound_key", "smiles")):
        out = _ensure_compound_id(out)
    bad = sorted(set(out["fold_id"].tolist()).difference(range(1, int(n_folds) + 1)))
    if bad:
        raise ValueError(f"Split file {path} contains folds outside 1..{n_folds}: {bad}")
    return out


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


def _assignment_for_split(pair_labels: pd.DataFrame, split_path: Path, n_folds: int) -> pd.DataFrame:
    split_df = _load_split_table(split_path, n_folds=n_folds)
    if "compound_id" in split_df.columns:
        assignments = split_df[["bgc_id", "compound_id", "fold_id"]].drop_duplicates().copy()
        merged = pair_labels.merge(assignments, on=["bgc_id", "compound_id"], how="inner")
    else:
        assignments = split_df[["bgc_id", "fold_id"]].drop_duplicates().copy()
        merged = pair_labels.merge(assignments, on="bgc_id", how="inner")
    return merged


def _count_level_by_fold(assigned: pd.DataFrame, level: str, n_folds: int) -> pd.DataFrame:
    label_column = LEVEL_COLUMNS[level]
    label_set = sorted({label for value in assigned[label_column].tolist() for label in _split_labels(value)})
    rows: list[dict[str, Any]] = []
    for fold_id in range(1, int(n_folds) + 1):
        fold_df = assigned[assigned["fold_id"] == fold_id]
        row: dict[str, Any] = {
            "fold_id": int(fold_id),
            "n_compounds": int(fold_df["canonical_smiles"].nunique()),
        }
        for label in label_set:
            matching = fold_df[fold_df[label_column].map(lambda value, name=label: name in _split_labels(value))]
            row[label] = int(matching["canonical_smiles"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def build_split_npclassifier_fold_tables(
    pair_labels_path: Path,
    split_paths: dict[str, Path],
    n_folds: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    pair_labels = pd.read_csv(pair_labels_path, sep="\t")
    required = {"bgc_id", "canonical_smiles", *LEVEL_COLUMNS.values()}
    missing = required.difference(pair_labels.columns)
    if missing:
        raise ValueError(f"{pair_labels_path} is missing columns: {sorted(missing)}")
    pair_labels = _ensure_compound_id(pair_labels)
    pair_labels["bgc_id"] = pair_labels["bgc_id"].astype(str)
    pair_labels["canonical_smiles"] = pair_labels["canonical_smiles"].astype(str)

    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for split_name, split_path in split_paths.items():
        assigned = _assignment_for_split(pair_labels, split_path, n_folds=n_folds)
        outputs[split_name] = {
            level: _count_level_by_fold(assigned, level=level, n_folds=n_folds)
            for level in ("class", "superclass", "pathway")
        }
    return outputs


def main() -> None:
    args = parse_args()
    split_paths = _parse_split_args(args.split)
    args.outdir.mkdir(parents=True, exist_ok=True)

    tables = build_split_npclassifier_fold_tables(
        pair_labels_path=args.pair_labels_path,
        split_paths=split_paths,
        n_folds=int(args.n_folds),
    )

    manifest: dict[str, Any] = {
        "pair_labels_path": str(args.pair_labels_path),
        "splits": {name: str(path) for name, path in split_paths.items()},
        "outputs": {},
        "notes": {
            "counting_unit": "Distinct canonical compounds present in each fold.",
            "bgc_level_splits": "For BGC and combined BGC-level split files, every compound row for a BGC inherits that BGC fold.",
            "pair_level_splits": "For NP and strict pair-level split files, counts follow the BGC-compound row fold assignment.",
            "multi_label": "A compound can contribute to multiple columns within a level.",
        },
    }
    for split_name, split_tables in tables.items():
        manifest["outputs"][split_name] = {}
        for level, table in split_tables.items():
            path = args.outdir / f"{split_name}_fold_npclassifier_{level}_counts.csv"
            table.to_csv(path, index=False)
            manifest["outputs"][split_name][level] = str(path)

    (args.outdir / "split_npclassifier_fold_counts_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
