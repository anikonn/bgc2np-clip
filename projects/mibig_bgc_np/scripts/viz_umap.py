from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from mibig_clip.viz.umap_plot import save_bgc_class_umap, save_joint_umap
from projects.mibig_bgc_np.data.datasets import build_bgc_class_map, build_interactions
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UMAP visualizations for MIBiG embeddings.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument("--cv_fold", type=int, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def _load_model(ckpt_path: str | Path, device: torch.device) -> DualEncoderCLIP:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = DualEncoderCLIP(
        bgc_input_dim=ckpt["bgc_input_dim"],
        compound_input_dim=ckpt["compound_input_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
        projection_head=str(cfg["model"].get("projection_head", "mlp_gelu")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    logger = setup_logger("mibig_bgc_np")
    cfg = apply_overrides(load_yaml(args.config), args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_path = args.splits_path if args.splits_path is not None else cfg.get("data", {}).get("splits_path")
    model = _load_model(args.checkpoint, device)

    interactions = build_interactions(args.data_dir, splits_path=splits_path, cv_fold=args.cv_fold)
    bgc_class_map = build_bgc_class_map(args.data_dir, splits_path=splits_path, cv_fold=args.cv_fold)
    bgc_cache = torch.load(Path(args.cache_dir) / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(Path(args.cache_dir) / "compound_features.pt", map_location="cpu")

    bgc_index, compound_index, bgc_embs, compound_embs, _ = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split=args.split,
        bgc_cache=bgc_cache,
        compound_cache=compound_cache,
        device=device,
    )

    bgc_ids = list(bgc_index.keys())
    compound_ids = list(compound_index.keys())
    outdir = Path(cfg["output"]["dir"]) / "viz"

    joint_paths = save_joint_umap(
        bgc_embs=bgc_embs.numpy(),
        compound_embs=compound_embs.numpy(),
        bgc_ids=bgc_ids,
        compound_ids=compound_ids,
        bgc_classes=bgc_class_map,
        outdir=outdir,
        prefix=f"{args.split}",
    )
    bgc_class_paths = save_bgc_class_umap(
        bgc_embs=bgc_embs.numpy(),
        bgc_ids=bgc_ids,
        bgc_classes=bgc_class_map,
        outdir=outdir,
        prefix=f"{args.split}_bgc_class",
    )

    payload = {
        "joint_umap": joint_paths,
        "bgc_class_umap": bgc_class_paths,
    }
    save_json(payload, outdir / f"{args.split}_umap_files.json")
    logger.info("Saved UMAP outputs: %s", payload)


if __name__ == "__main__":
    main()
