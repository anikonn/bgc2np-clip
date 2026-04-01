from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_core.logging import save_json
from mibig_clip.eval.retrieval import evaluate_global_retrieval_multi
from projects.mibig_bgc_np.data.datasets import CachedInteractionDataset, build_interactions, collate_interactions
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP


def _get_cached_paths(cache_dir: str | Path) -> tuple[Path, Path]:
    cache_root = Path(cache_dir)
    return cache_root / "bgc_features.pt", cache_root / "compound_features.pt"


def _infer_input_dims(bgc_cache_path: Path, compound_cache_path: Path) -> tuple[int, int]:
    bgc_cache = torch.load(bgc_cache_path, map_location="cpu")
    compound_cache = torch.load(compound_cache_path, map_location="cpu")
    return int(next(iter(bgc_cache.values())).numel()), int(next(iter(compound_cache.values())).numel())


def _build_loader(
    interactions: pd.DataFrame,
    bgc_cache_path: Path,
    compound_cache_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CachedInteractionDataset(
        interactions=interactions,
        bgc_cache_path=bgc_cache_path,
        compound_cache_path=compound_cache_path,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_interactions,
        pin_memory=True,
    )


def build_unique_embeddings(
    model: torch.nn.Module,
    interactions: pd.DataFrame,
    split: str,
    bgc_cache: dict[str, torch.Tensor],
    compound_cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[dict[str, int], dict[str, int], torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Build projected normalized embeddings for all unique entities in a split."""
    df = interactions[interactions["split"].str.lower() == split.lower()].copy()
    bgc_ids = sorted(df["bgc_id"].unique().tolist())
    compound_ids = sorted(df["compound_id"].unique().tolist())
    bgc_index = {bgc_id: idx for idx, bgc_id in enumerate(bgc_ids)}
    compound_index = {compound_id: idx for idx, compound_id in enumerate(compound_ids)}

    model.eval()

    bgc_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(bgc_ids), batch_size):
            batch_ids = bgc_ids[start : start + batch_size]
            feats = torch.stack([bgc_cache[item_id] for item_id in batch_ids]).to(device)
            bgc_chunks.append(model.encode_bgc(feats).cpu())
    bgc_embs = torch.cat(bgc_chunks, dim=0)

    compound_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(compound_ids), batch_size):
            batch_ids = compound_ids[start : start + batch_size]
            feats = torch.stack([compound_cache[item_id] for item_id in batch_ids]).to(device)
            compound_chunks.append(model.encode_compound(feats).cpu())
    compound_embs = torch.cat(compound_chunks, dim=0)

    pairs = [(bgc_index[row.bgc_id], compound_index[row.compound_id]) for row in df.itertuples(index=False)]
    return bgc_index, compound_index, bgc_embs, compound_embs, pairs


def evaluate_split_retrieval(
    model: torch.nn.Module,
    interactions: pd.DataFrame,
    split: str,
    bgc_cache_path: Path,
    compound_cache_path: Path,
    device: torch.device,
    sim_batch_size: int,
) -> dict[str, dict[str, float]]:
    bgc_cache = torch.load(bgc_cache_path, map_location="cpu")
    compound_cache = torch.load(compound_cache_path, map_location="cpu")
    _, _, bgc_embs, compound_embs, pairs = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split=split,
        bgc_cache=bgc_cache,
        compound_cache=compound_cache,
        device=device,
    )
    return evaluate_global_retrieval_multi(
        bgc_embs=bgc_embs,
        compound_embs=compound_embs,
        interaction_pairs=pairs,
        sim_batch_size=sim_batch_size,
    )


def _val_selection_score(retrieval_metrics: dict[str, dict[str, float]]) -> float:
    bgc_to_compound = retrieval_metrics.get("bgc_to_compound", {})
    compound_to_bgc = retrieval_metrics.get("compound_to_bgc", {})
    return float(0.5 * (bgc_to_compound.get("mrr", 0.0) + compound_to_bgc.get("mrr", 0.0)))


def train_contrastive(
    data_dir: str | Path,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    splits_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], pd.DataFrame]:
    """Train CLIP-style projection heads on cached BGC and compound features."""
    bgc_cache_path, compound_cache_path = _get_cached_paths(cache_dir)
    interactions = build_interactions(data_dir, splits_path=splits_path)

    bgc_dim, compound_dim = _infer_input_dims(bgc_cache_path, compound_cache_path)
    model = DualEncoderCLIP(
        bgc_input_dim=bgc_dim,
        compound_input_dim=compound_dim,
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
    ).to(device)

    train_loader = _build_loader(
        interactions=interactions,
        bgc_cache_path=bgc_cache_path,
        compound_cache_path=compound_cache_path,
        split="train",
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=True,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[float]] = {"train_loss": []}
    epochs = int(cfg["train"]["epochs"])
    best_score = -1e9
    best_epoch = 0
    best_ckpt_path = outdir / "contrastive_model_best.pt"
    selection_split = str(cfg.get("eval", {}).get("selection_split", "val"))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            bgc_features = batch["bgc_feature"].to(device)
            compound_features = batch["compound_feature"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(bgc_features, compound_features)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * bgc_features.size(0)
            count += bgc_features.size(0)
            progress.set_postfix(loss=float(loss.item()))

        epoch_loss = running / max(count, 1)
        history["train_loss"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_retrieval = evaluate_split_retrieval(
                model=model,
                interactions=interactions,
                split=selection_split,
                bgc_cache_path=bgc_cache_path,
                compound_cache_path=compound_cache_path,
                device=device,
                sim_batch_size=cfg["eval"]["sim_batch_size"],
            )
        score = _val_selection_score(val_retrieval)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "bgc_input_dim": bgc_dim,
                    "compound_input_dim": compound_dim,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "selection_split": selection_split,
                    "train_loss": epoch_loss,
                    "retrieval_selection": val_retrieval,
                },
                best_ckpt_path,
            )

    best = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])

    last_ckpt_path = outdir / "contrastive_model_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "bgc_input_dim": bgc_dim,
            "compound_input_dim": compound_dim,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_split": selection_split,
        },
        last_ckpt_path,
    )

    metrics: dict[str, Any] = {
        "train": {"loss_last_epoch": history["train_loss"][-1]},
        "model_selection": {
            "selection_split": selection_split,
            "best_epoch": int(best_epoch),
            "best_score_mean_mrr": float(best_score),
            "best_checkpoint": str(best_ckpt_path),
        },
    }
    for split in ("val", "test"):
        metrics[f"retrieval_{split}"] = evaluate_split_retrieval(
            model=model,
            interactions=interactions,
            split=split,
            bgc_cache_path=bgc_cache_path,
            compound_cache_path=compound_cache_path,
            device=device,
            sim_batch_size=cfg["eval"]["sim_batch_size"],
        )

    save_json(metrics, outdir / "contrastive_metrics.json")
    return model, metrics, interactions
