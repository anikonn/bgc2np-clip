from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import setup_logger
from kiba_clip.utils.seed import set_seed
from projects.mibig_bgc_np.training.downstream_trainer import train_downstream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train downstream BGC class classifier on frozen MIBiG embeddings.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("mibig_bgc_np")
    cfg = apply_overrides(load_yaml(args.config), args.override)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg["output"]["dir"] = str(Path(cfg["output"]["dir"]))
    splits_path = args.splits_path if args.splits_path is not None else cfg.get("data", {}).get("splits_path")
    metrics = train_downstream(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        contrastive_ckpt=args.checkpoint,
        cfg=cfg,
        device=device,
        splits_path=splits_path,
    )
    logger.info("Downstream val metrics: %s", metrics["val"])
    logger.info("Downstream test metrics: %s", metrics["test"])


if __name__ == "__main__":
    main()
