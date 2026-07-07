from __future__ import annotations

import argparse
from pathlib import Path

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.logging import setup_logger
from projects.mibig_bgc_np.eval.baseline_artifacts import save_all_baseline_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect baseline JSONs into visible CSV/PNG artifact folders.")
    parser.add_argument("--run_root", type=Path, required=True, help="Run output root, e.g. results/combined_cv10.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("package_baseline_artifacts")
    manifest = save_all_baseline_artifacts(args.run_root)
    logger.info("Saved baseline artifacts to %s", Path(manifest["run_root"]) / "baselines")


if __name__ == "__main__":
    main()
