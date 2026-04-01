from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from clip_core.retrieval import evaluate_global_retrieval_multi
from kiba_clip.data.datasets import build_interactions
from kiba_clip.models.clip_dual import DualEncoderCLIP
from kiba_clip.training.contrastive_trainer import build_unique_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate global retrieval metrics.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def _load_model(ckpt_path: str | Path, device: torch.device) -> tuple[DualEncoderCLIP, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = DualEncoderCLIP(
        protein_input_dim=ckpt["protein_input_dim"],
        ligand_input_dim=ckpt["ligand_input_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    cfg = apply_overrides(load_yaml(args.config), args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = _load_model(args.checkpoint, device)
    interactions = build_interactions(args.data_dir)

    prot_cache = torch.load(Path(args.cache_dir) / "protein_embeddings.pt", map_location="cpu")
    lig_cache = torch.load(Path(args.cache_dir) / "ligand_fingerprints.pt", map_location="cpu")

    p_index, l_index, p_embs, l_embs, pairs = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split=args.split,
        protein_cache=prot_cache,
        ligand_cache=lig_cache,
        device=device,
    )

    metrics = evaluate_global_retrieval_multi(
        protein_embs=p_embs,
        ligand_embs=l_embs,
        interaction_pairs=pairs,
        sim_batch_size=cfg["eval"]["sim_batch_size"],
    )

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"retrieval_{args.split}.json"
    save_json(metrics, out_json)

    torch.save(
        {
            "protein_ids": list(p_index.keys()),
            "ligand_ids": list(l_index.keys()),
            "protein_embeddings": p_embs,
            "ligand_embeddings": l_embs,
            "split": args.split,
        },
        outdir / f"embeddings_{args.split}.pt",
    )

    meta = pd.DataFrame(
        {
            "id": list(p_index.keys()) + list(l_index.keys()),
            "modality": ["protein"] * len(p_index) + ["ligand"] * len(l_index),
            "split": [args.split] * (len(p_index) + len(l_index)),
        }
    )
    meta.to_csv(outdir / f"embedding_meta_{args.split}.csv", index=False)

    logger.info("Saved retrieval metrics to %s", out_json)
    logger.info("%s", metrics)


if __name__ == "__main__":
    main()
