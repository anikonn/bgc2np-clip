from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip_core.logging import setup_logger
from mibig_clip.data.splits import (
    assign_cv_folds_by_bgc,
    assign_cv_folds_by_np,
    assign_cv_folds_combined,
    random_split_by_bgc,
    random_split_by_np,
    random_split_combined,
    write_split_tsv,
)

LOGGER = setup_logger("make_mibig_splits")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create random or CV split files from processed MIBiG pairs.")
    parser.add_argument("--pairs_path", type=Path, required=True, help="Path to processed mibig_pairs.tsv.")
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("data/MIBIG/splits"),
        help="Directory where split TSVs will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument(
        "--split_mode",
        choices=("random", "cv"),
        default="random",
        help="Whether to create a random train/val/test split or CV fold assignments.",
    )
    parser.add_argument(
        "--split_type",
        choices=("bgc", "combined", "np"),
        default="combined",
        help=(
            "Grouping constraint for split generation: bgc keeps each BGC together; "
            "combined keeps BGCs connected by shared compounds together; np keeps each compound together."
        ),
    )
    parser.add_argument("--train_frac", type=float, default=0.8, help="Training fraction for random splits.")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction for random splits.")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction for random splits.")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds for CV split generation.")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Optional filename prefix. Defaults to the selected split_type.",
    )
    return parser.parse_args()


def _load_bgc_compound_index(path: Path) -> tuple[list[str], dict[str, set[str]]]:
    bgc_ids: list[str] = []
    bgc_to_compound_ids: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            bgc_id = row.get("bgc_id")
            if bgc_id:
                bgc_id = str(bgc_id)
                bgc_ids.append(bgc_id)
                compound_id = row.get("compound_id") or row.get("canonical_smiles") or row.get("smiles")
                if compound_id:
                    bgc_to_compound_ids.setdefault(bgc_id, set()).add(str(compound_id))
    return sorted(set(bgc_ids)), bgc_to_compound_ids


def main() -> None:
    args = _parse_args()
    if not args.pairs_path.exists():
        raise FileNotFoundError(f"Pairs TSV not found: {args.pairs_path}")

    bgc_ids, bgc_to_compound_ids = _load_bgc_compound_index(args.pairs_path)
    if len(bgc_ids) < 3:
        raise ValueError(f"Need at least 3 paired BGCs to create splits, found {len(bgc_ids)}")
    output_prefix = str(args.output_prefix) if args.output_prefix is not None else str(args.split_type)

    if args.split_mode == "random":
        if args.split_type == "bgc":
            random_assignments = random_split_by_bgc(
                bgc_ids=bgc_ids,
                seed=args.seed,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
            )
        elif args.split_type == "combined":
            random_assignments = random_split_combined(
                bgc_ids=bgc_ids,
                bgc_to_compound_ids=bgc_to_compound_ids,
                seed=args.seed,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
            )
        else:
            random_assignments = random_split_by_np(
                bgc_to_compound_ids=bgc_to_compound_ids,
                seed=args.seed,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
            )
        output_path = args.splits_dir / f"{output_prefix}_random_seed{args.seed}.tsv"
        write_split_tsv(random_assignments, output_path)
        LOGGER.info("Wrote random split assignments to %s", output_path)
        return

    if args.split_type == "bgc":
        cv_assignments = assign_cv_folds_by_bgc(
            bgc_ids=bgc_ids,
            seed=args.seed,
            n_folds=args.n_folds,
        )
    elif args.split_type == "combined":
        cv_assignments = assign_cv_folds_combined(
            bgc_ids=bgc_ids,
            bgc_to_compound_ids=bgc_to_compound_ids,
            seed=args.seed,
            n_folds=args.n_folds,
        )
    else:
        cv_assignments = assign_cv_folds_by_np(
            bgc_to_compound_ids=bgc_to_compound_ids,
            seed=args.seed,
            n_folds=args.n_folds,
        )
    output_path = args.splits_dir / f"{output_prefix}_cv_seed{args.seed}_n{args.n_folds}.tsv"
    write_split_tsv(cv_assignments, output_path)
    LOGGER.info("Wrote CV fold assignments to %s", output_path)


if __name__ == "__main__":
    main()
