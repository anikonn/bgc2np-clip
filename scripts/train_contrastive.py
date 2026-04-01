from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import setup_logger
from kiba_clip.training.contrastive_trainer import train_contrastive
from kiba_clip.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive protein-ligand dual encoder.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg["output"]["dir"] = str(Path(cfg["output"]["dir"]))
    _, metrics, _ = train_contrastive(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        cfg=cfg,
        device=device,
    )
    logger.info("Train loss: %.6f", metrics["train"]["loss_last_epoch"])


if __name__ == "__main__":
    main()
