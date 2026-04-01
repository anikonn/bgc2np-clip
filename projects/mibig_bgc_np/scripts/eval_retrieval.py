from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings
from mibig_clip.eval.retrieval import evaluate_global_retrieval_multi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIBiG BGC-compound retrieval metrics.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def _load_model(ckpt_path: str | Path, device: torch.device) -> tuple[DualEncoderCLIP, dict]:
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _save_embedding_meta(path: Path, bgc_ids: list[str], compound_ids: list[str], split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "modality", "split"])
        writer.writeheader()
        for bgc_id in bgc_ids:
            writer.writerow({"id": bgc_id, "modality": "bgc", "split": split})
        for compound_id in compound_ids:
            writer.writerow({"id": compound_id, "modality": "compound", "split": split})


def main() -> None:
    args = parse_args()
    logger = setup_logger("mibig_bgc_np")

    cfg = apply_overrides(load_yaml(args.config), args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_path = args.splits_path if args.splits_path is not None else cfg.get("data", {}).get("splits_path")

    model, _ = _load_model(args.checkpoint, device)
    interactions = build_interactions(args.data_dir, splits_path=splits_path)

    bgc_cache = torch.load(Path(args.cache_dir) / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(Path(args.cache_dir) / "compound_features.pt", map_location="cpu")
    bgc_index, compound_index, bgc_embs, compound_embs, pairs = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split=args.split,
        bgc_cache=bgc_cache,
        compound_cache=compound_cache,
        device=device,
    )

    metrics = evaluate_global_retrieval_multi(
        bgc_embs=bgc_embs,
        compound_embs=compound_embs,
        interaction_pairs=pairs,
        sim_batch_size=cfg["eval"]["sim_batch_size"],
    )

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    retrieval_path = outdir / f"retrieval_{args.split}.json"
    save_json(metrics, retrieval_path)

    torch.save(
        {
            "bgc_ids": list(bgc_index.keys()),
            "compound_ids": list(compound_index.keys()),
            "bgc_embeddings": bgc_embs,
            "compound_embeddings": compound_embs,
            "split": args.split,
        },
        outdir / f"embeddings_{args.split}.pt",
    )
    _save_embedding_meta(
        outdir / f"embedding_meta_{args.split}.csv",
        bgc_ids=list(bgc_index.keys()),
        compound_ids=list(compound_index.keys()),
        split=args.split,
    )

    logger.info("Saved retrieval metrics to %s", retrieval_path)
    logger.info("%s", metrics)


if __name__ == "__main__":
    main()
