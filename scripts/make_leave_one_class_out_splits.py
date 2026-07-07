from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd

from clip_core.logging import save_json, setup_logger


LOGGER = setup_logger("make_leave_one_class_out_splits")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create leave-one-BGC-product-class-out split files. Exp3 holds out "
            "BGCs of a target class; exp4 holds out compounds associated with a target class."
        )
    )
    parser.add_argument("--pairs_path", type=Path, required=True, help="Processed mibig_pairs.tsv.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory where split files will be written.")
    parser.add_argument(
        "--mode",
        choices=("bgc", "np", "both"),
        default="both",
        help="bgc = exp3 BGC-to-chemical; np = exp4 chemical-to-BGC; both = write both sets.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="loco",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--min_test_pairs",
        type=int,
        default=1,
        help="Skip target classes with fewer than this many test pairs.",
    )
    return parser.parse_args()


def _parse_labels(value: Any) -> list[str]:
    raw = "" if value is None else str(value)
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
    candidates = parsed if isinstance(parsed, (list, tuple, set)) else re.split(r"[;,]", raw)
    labels: list[str] = []
    seen: set[str] = set()
    for label in candidates:
        clean = str(label).strip().strip("'\"")
        if clean and clean not in seen:
            labels.append(clean)
            seen.add(clean)
    return labels


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "class"


def _load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"bgc_id", "compound_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    label_col = "bgc_classes" if "bgc_classes" in df.columns else "bgc_class" if "bgc_class" in df.columns else None
    if label_col is None:
        raise ValueError(f"{path} must contain bgc_classes or bgc_class.")
    df["bgc_id"] = df["bgc_id"].astype(str)
    df["compound_id"] = df["compound_id"].astype(str)
    df["bgc_classes"] = df[label_col].fillna("").astype(str)
    return df


def _class_maps(df: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]], list[str]]:
    bgc_to_classes: dict[str, set[str]] = {}
    for row in df[["bgc_id", "bgc_classes"]].drop_duplicates("bgc_id").itertuples(index=False):
        bgc_to_classes[str(row.bgc_id)] = set(_parse_labels(row.bgc_classes))

    compound_to_classes: dict[str, set[str]] = {}
    for row in df[["compound_id", "bgc_id"]].drop_duplicates().itertuples(index=False):
        compound_to_classes.setdefault(str(row.compound_id), set()).update(bgc_to_classes.get(str(row.bgc_id), set()))

    classes = sorted({label for labels in bgc_to_classes.values() for label in labels})
    return bgc_to_classes, compound_to_classes, classes


def _write_bgc_holdout(
    df: pd.DataFrame,
    bgc_to_classes: dict[str, set[str]],
    target_class: str,
    output_path: Path,
) -> dict[str, Any]:
    target_bgcs = {bgc_id for bgc_id, labels in bgc_to_classes.items() if target_class in labels}
    split_df = (
        df[["bgc_id", "bgc_classes"]]
        .drop_duplicates("bgc_id")
        .sort_values("bgc_id")
        .reset_index(drop=True)
    )
    split_df["split"] = split_df["bgc_id"].map(lambda bgc_id: "test" if str(bgc_id) in target_bgcs else "train")
    split_df["target_class"] = target_class
    split_df = split_df[["bgc_id", "split", "target_class", "bgc_classes"]]
    test_pairs = df[df["bgc_id"].isin(target_bgcs)]
    train_pairs = df[~df["bgc_id"].isin(target_bgcs)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, sep="\t", index=False)
    return {
        "target_class": target_class,
        "mode": "bgc",
        "path": str(output_path),
        "n_train_pairs": int(len(train_pairs)),
        "n_test_pairs": int(len(test_pairs)),
        "n_train_bgcs": int(train_pairs["bgc_id"].nunique()),
        "n_test_bgcs": int(test_pairs["bgc_id"].nunique()),
        "n_train_compounds": int(train_pairs["compound_id"].nunique()),
        "n_test_compounds": int(test_pairs["compound_id"].nunique()),
    }


def _write_np_holdout(
    df: pd.DataFrame,
    compound_to_classes: dict[str, set[str]],
    target_class: str,
    output_path: Path,
) -> dict[str, Any]:
    target_compounds = {
        compound_id for compound_id, labels in compound_to_classes.items() if target_class in labels
    }
    split_df = df[["bgc_id", "compound_id", "bgc_classes"]].drop_duplicates(["bgc_id", "compound_id"]).copy()
    split_df = split_df.sort_values(["compound_id", "bgc_id"]).reset_index(drop=True)
    split_df["split"] = split_df["compound_id"].map(
        lambda compound_id: "test" if str(compound_id) in target_compounds else "train"
    )
    split_df["target_class"] = target_class
    split_df = split_df[["bgc_id", "compound_id", "split", "target_class", "bgc_classes"]]
    test_pairs = df[df["compound_id"].isin(target_compounds)]
    train_pairs = df[~df["compound_id"].isin(target_compounds)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, sep="\t", index=False)
    return {
        "target_class": target_class,
        "mode": "np",
        "path": str(output_path),
        "n_train_pairs": int(len(train_pairs)),
        "n_test_pairs": int(len(test_pairs)),
        "n_train_bgcs": int(train_pairs["bgc_id"].nunique()),
        "n_test_bgcs": int(test_pairs["bgc_id"].nunique()),
        "n_train_compounds": int(train_pairs["compound_id"].nunique()),
        "n_test_compounds": int(test_pairs["compound_id"].nunique()),
    }


def main() -> None:
    args = _parse_args()
    df = _load_pairs(args.pairs_path)
    bgc_to_classes, compound_to_classes, classes = _class_maps(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for target_class in classes:
        slug = _slugify(target_class)
        if args.mode in {"bgc", "both"}:
            path = args.out_dir / "exp3_bgc" / f"{args.prefix}_exp3_bgc_{slug}.tsv"
            row = _write_bgc_holdout(df, bgc_to_classes, target_class, path)
            if row["n_test_pairs"] >= int(args.min_test_pairs):
                rows.append(row)
        if args.mode in {"np", "both"}:
            path = args.out_dir / "exp4_np" / f"{args.prefix}_exp4_np_{slug}.tsv"
            row = _write_np_holdout(df, compound_to_classes, target_class, path)
            if row["n_test_pairs"] >= int(args.min_test_pairs):
                rows.append(row)

    summary = {
        "pairs_path": str(args.pairs_path),
        "out_dir": str(args.out_dir),
        "classes": classes,
        "n_classes": int(len(classes)),
        "splits": rows,
    }
    summary_path = args.out_dir / f"{args.prefix}_leave_one_class_out_summary.json"
    save_json(summary, summary_path)
    pd.DataFrame(rows).to_csv(args.out_dir / f"{args.prefix}_leave_one_class_out_summary.tsv", sep="\t", index=False)
    LOGGER.info("Wrote %d leave-one-class-out split files under %s", len(rows), args.out_dir)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
