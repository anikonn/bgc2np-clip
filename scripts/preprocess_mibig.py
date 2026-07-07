from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip_core.logging import setup_logger
from mibig_clip.data.preprocessing import build_mibig_dataset

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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_mibig_dataset(
        fasta_path=args.fasta_path,
        json_dir=args.json_dir,
        out_dir=args.out_dir,
    )
    LOGGER.info("Preprocessing complete with %s pairs written", summary["pairs_written"])


if __name__ == "__main__":
    main()
