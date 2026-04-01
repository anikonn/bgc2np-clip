from __future__ import annotations

import argparse

import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import setup_logger
from kiba_clip.training.downstream_trainer import train_downstream
from kiba_clip.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train downstream regressor on frozen joint embeddings.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = train_downstream(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        contrastive_ckpt=args.checkpoint,
        cfg=cfg,
        device=device,
    )
    logger.info("Downstream val metrics: %s", metrics["val"])
    logger.info("Downstream test metrics: %s", metrics["test"])


if __name__ == "__main__":
    main()
