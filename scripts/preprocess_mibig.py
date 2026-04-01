from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip_core.logging import setup_logger
from mibig_clip.data.preprocessing import build_mibig_dataset
from mibig_clip.data.splits import (
    cold_split_by_minhash_kmers,
    random_split_by_bgc,
    write_clustering_report,
    write_split_tsv,
)

LOGGER = setup_logger("preprocess_mibig_script")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MIBiG data for CLIP-style BGC-NP pairing.")
    parser.add_argument("--fasta_path", type=Path, required=True, help="Path to MIBiG protein FASTA.")
    parser.add_argument("--json_dir", type=Path, required=True, help="Directory with MIBiG JSON files.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for mibig_pairs.tsv, bgc_proteins.jsonl, and summary JSON.",
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for random/cold split TSVs and clustering reports.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument(
        "--make_splits",
        choices=("random", "cold", "both", "none"),
        default="both",
        help="Which split artifacts to write after preprocessing.",
    )
    parser.add_argument("--cold_k", type=int, default=5, help="k-mer size for cold split clustering.")
    parser.add_argument(
        "--cold_threshold",
        type=float,
        default=0.3,
        help="Jaccard threshold for greedy cold split clustering.",
    )
    parser.add_argument("--train_frac", type=float, default=0.8, help="Training fraction.")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction.")
    return parser.parse_args()


def _load_proteins_index(path: Path) -> dict[str, dict[str, Any]]:
    proteins_index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            proteins_index[record["bgc_id"]] = record
    return proteins_index


def _load_bgc_ids_from_pairs(path: Path) -> list[str]:
    bgc_ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            bgc_ids.append(row["bgc_id"])
    return sorted(set(bgc_ids))


def _serialize_moieties(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    moieties = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return ";".join(moieties)


def _build_moieties_index(json_dir: Path) -> dict[tuple[str, int, int], str]:
    moieties_index: dict[tuple[str, int, int], str] = {}
    for json_path in sorted(json_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        bgc_id = payload.get("accession")
        bgc_version = payload.get("version")
        compounds = payload.get("compounds", [])
        if not isinstance(bgc_id, str) or not isinstance(bgc_version, int) or not isinstance(compounds, list):
            continue

        for compound_idx, compound in enumerate(compounds):
            if not isinstance(compound, dict):
                continue
            moieties_index[(bgc_id, bgc_version, compound_idx)] = _serialize_moieties(
                compound.get("moieties")
            )
    return moieties_index


def _ensure_moieties_column(pairs_path: Path, json_dir: Path) -> None:
    with pairs_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"Pairs TSV is missing a header: {pairs_path}")
    if "moieties" in fieldnames:
        LOGGER.info("Pairs TSV already contains moieties column: %s", pairs_path)
        return

    try:
        smiles_idx = fieldnames.index("smiles")
    except ValueError as exc:
        raise ValueError(f"Could not insert moieties column because smiles is missing from {pairs_path}") from exc

    moieties_index = _build_moieties_index(json_dir)
    fieldnames.insert(smiles_idx + 1, "moieties")
    for row in rows:
        bgc_id = row.get("bgc_id", "")
        bgc_version_raw = row.get("bgc_version", "")
        compound_idx_raw = row.get("compound_idx", "")
        try:
            bgc_version = int(bgc_version_raw)
        except (TypeError, ValueError):
            bgc_version = -1
        try:
            compound_idx = int(compound_idx_raw)
        except (TypeError, ValueError):
            compound_idx = -1
        row["moieties"] = moieties_index.get((bgc_id, bgc_version, compound_idx), "")

    with pairs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Backfilled moieties column into %s", pairs_path)


def main() -> None:
    args = _parse_args()
    summary = build_mibig_dataset(
        fasta_path=args.fasta_path,
        json_dir=args.json_dir,
        out_dir=args.out_dir,
    )
    _ensure_moieties_column(args.out_dir / "mibig_pairs.tsv", args.json_dir)
    if args.make_splits == "none":
        return

    pairs_path = args.out_dir / "mibig_pairs.tsv"
    proteins_path = args.out_dir / "bgc_proteins.jsonl"
    bgc_ids = _load_bgc_ids_from_pairs(pairs_path)
    if len(bgc_ids) < 3:
        raise ValueError(
            f"Need at least 3 paired BGCs to create train/val/test splits, found {len(bgc_ids)}"
        )

    if args.make_splits in {"random", "both"}:
        random_assignments = random_split_by_bgc(
            bgc_ids=bgc_ids,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
        random_path = args.splits_dir / f"random_seed{args.seed}.tsv"
        write_split_tsv(random_assignments, random_path)
        LOGGER.info("Wrote random split assignments to %s", random_path)

    if args.make_splits in {"cold", "both"}:
        proteins_index = _load_proteins_index(proteins_path)
        cold_assignments, cold_report = cold_split_by_minhash_kmers(
            proteins_index=proteins_index,
            seed=args.seed,
            k=args.cold_k,
            threshold=args.cold_threshold,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
        threshold_label = f"{args.cold_threshold:g}"
        cold_path = args.splits_dir / f"cold_seed{args.seed}_k{args.cold_k}_thr{threshold_label}.tsv"
        report_path = (
            args.splits_dir / f"cold_seed{args.seed}_k{args.cold_k}_thr{threshold_label}_report.json"
        )
        write_split_tsv(cold_assignments, cold_path)
        write_clustering_report(cold_report, report_path)
        LOGGER.info("Wrote cold split assignments to %s", cold_path)
        LOGGER.info("Wrote clustering report to %s", report_path)

    LOGGER.info("Preprocessing complete with %s pairs written", summary["pairs_written"])


if __name__ == "__main__":
    main()
