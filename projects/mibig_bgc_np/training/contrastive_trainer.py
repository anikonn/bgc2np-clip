from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_core.logging import save_json
from mibig_clip.eval.retrieval import evaluate_global_retrieval_multi
from projects.mibig_bgc_np.eval.retrieval_class_metrics import evaluate_bgc_class_retrieval
from projects.mibig_bgc_np.data.datasets import CachedInteractionDataset, build_interactions, collate_interactions
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP


def _get_cached_paths(cache_dir: str | Path) -> tuple[Path, Path]:
    cache_root = Path(cache_dir)
    return cache_root / "bgc_features.pt", cache_root / "compound_features.pt"


def _infer_input_dims(bgc_cache_path: Path, compound_cache_path: Path) -> tuple[int, int]:
    bgc_cache = torch.load(bgc_cache_path, map_location="cpu")
    compound_cache = torch.load(compound_cache_path, map_location="cpu")
    cache_root = bgc_cache_path.parent
    protein_path = cache_root / "protein_positions.pt"
    domain_path = cache_root / "domain_positions.pt"
    protein_positions = torch.load(protein_path, map_location="cpu", weights_only=True) if protein_path.exists() else None
    domain_positions = torch.load(domain_path, map_location="cpu", weights_only=True) if domain_path.exists() else None
    bgc_example = next(iter(bgc_cache.values()))
    if bgc_example.ndim not in (1, 2):
        raise ValueError(f"BGC cache entries must be vectors or domain matrices, got {tuple(bgc_example.shape)}")
    return int(bgc_example.shape[-1]), int(next(iter(compound_cache.values())).numel())


def _pad_bgc_features(features: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    if features[0].ndim == 1:
        return torch.stack(features).to(device), None
    if any(feature.ndim != 2 or feature.shape[1] != features[0].shape[1] for feature in features):
        raise ValueError("Variable-length BGC caches must contain [domains, common_dim] tensors")
    max_domains = max(feature.shape[0] for feature in features)
    padded = torch.zeros((len(features), max_domains, features[0].shape[1]), dtype=torch.float32)
    mask = torch.ones((len(features), max_domains), dtype=torch.bool)
    for index, feature in enumerate(features):
        padded[index, : feature.shape[0]] = feature.float()
        mask[index, : feature.shape[0]] = False
    return padded.to(device), mask.to(device)


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


def _build_positive_pair_set(interactions: pd.DataFrame, split: str) -> set[tuple[str, str]]:
    split_df = interactions[interactions["split"].str.lower() == split.lower()]
    return {
        (str(row.bgc_id), str(row.compound_id))
        for row in split_df[["bgc_id", "compound_id"]].drop_duplicates().itertuples(index=False)
    }


def _build_batch_positive_mask(
    bgc_ids: list[str],
    compound_ids: list[str],
    positive_pairs: set[tuple[str, str]],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            [(str(bgc_id), str(compound_id)) in positive_pairs for compound_id in compound_ids]
            for bgc_id in bgc_ids
        ],
        dtype=torch.bool,
        device=device,
    )


def build_unique_embeddings(
    model: torch.nn.Module,
    interactions: pd.DataFrame,
    split: str,
    bgc_cache: dict[str, torch.Tensor],
    compound_cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 1024,
    protein_position_cache: dict[str, torch.Tensor] | None = None,
    domain_position_cache: dict[str, torch.Tensor] | None = None,
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
            feats, padding_mask = _pad_bgc_features([bgc_cache[item_id] for item_id in batch_ids], device)
            position_args: dict[str, torch.Tensor] = {}
            for name, cache in (
                ("protein_positions", protein_position_cache),
                ("domain_positions", domain_position_cache),
            ):
                if cache is not None:
                    values = [cache[item_id].long() for item_id in batch_ids]
                    padded = torch.zeros(feats.shape[:2], dtype=torch.long, device=device)
                    for index, value in enumerate(values):
                        padded[index, : value.numel()] = value.to(device)
                    position_args[name] = padded
            bgc_chunks.append(model.encode_bgc(feats, padding_mask=padding_mask, **position_args).cpu())
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
    cache_root = bgc_cache_path.parent
    protein_path = cache_root / "protein_positions.pt"
    domain_path = cache_root / "domain_positions.pt"
    protein_positions = torch.load(protein_path, map_location="cpu", weights_only=True) if protein_path.exists() else None
    domain_positions = torch.load(domain_path, map_location="cpu", weights_only=True) if domain_path.exists() else None
    _, _, bgc_embs, compound_embs, pairs = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split=split,
        bgc_cache=bgc_cache,
        compound_cache=compound_cache,
        device=device,
        protein_position_cache=protein_positions,
        domain_position_cache=domain_positions,
    )
    return evaluate_global_retrieval_multi(
        bgc_embs=bgc_embs,
        compound_embs=compound_embs,
        interaction_pairs=pairs,
        sim_batch_size=sim_batch_size,
    )


def evaluate_split_bgc_class_retrieval(
    model: torch.nn.Module,
    interactions: pd.DataFrame,
    split: str,
    bgc_cache_path: Path,
    compound_cache_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    bgc_cache = torch.load(bgc_cache_path, map_location="cpu")
    compound_cache = torch.load(compound_cache_path, map_location="cpu")
    cache_root = bgc_cache_path.parent
    protein_path = cache_root / "protein_positions.pt"
    domain_path = cache_root / "domain_positions.pt"
    protein_positions = torch.load(protein_path, map_location="cpu", weights_only=True) if protein_path.exists() else None
    domain_positions = torch.load(domain_path, map_location="cpu", weights_only=True) if domain_path.exists() else None
    bgc_index, compound_index, bgc_embs, compound_embs, pairs = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split=split,
        bgc_cache=bgc_cache,
        compound_cache=compound_cache,
        device=device,
        protein_position_cache=protein_positions,
        domain_position_cache=domain_positions,
    )
    sim = model.get_logit_scale().detach().cpu() * (bgc_embs @ compound_embs.t())
    return evaluate_bgc_class_retrieval(
        sim=sim,
        bgc_ids=list(bgc_index.keys()),
        compound_ids=list(compound_index.keys()),
        pairs=pairs,
        interactions=interactions,
        split=split,
    )


def _val_selection_score(retrieval_metrics: dict[str, dict[str, float]], metric: str = "mean_mrr") -> float:
    bgc_to_compound = retrieval_metrics.get("bgc_to_compound", {})
    compound_to_bgc = retrieval_metrics.get("compound_to_bgc", {})
    metric = str(metric).lower()
    if metric == "mean_mrr":
        return float(0.5 * (bgc_to_compound.get("mrr", 0.0) + compound_to_bgc.get("mrr", 0.0)))
    if metric == "bgc_to_np_recall_at_10":
        return float(bgc_to_compound.get("recall_at_10", 0.0))
    if metric == "bidirectional_recall_at_10":
        return float(0.5 * (bgc_to_compound.get("recall_at_10", 0.0) + compound_to_bgc.get("recall_at_10", 0.0)))
    raise ValueError(
        f"Unknown eval.selection_metric={metric!r}; expected mean_mrr, "
        "bgc_to_np_recall_at_10, or bidirectional_recall_at_10"
    )


def _build_lr_scheduler(optimizer: AdamW, cfg: dict[str, Any], steps_per_epoch: int, epochs: int) -> LambdaLR | None:
    scheduler_name = str(cfg["train"].get("scheduler", "none")).lower()
    if scheduler_name in ("none", ""):
        return None
    total_steps = max(1, int(steps_per_epoch) * int(epochs))
    warmup_steps = int(total_steps * float(cfg["train"].get("warmup_fraction", 0.0)))

    def warmup_scale(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    if scheduler_name == "linear_warmup_decay":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return warmup_scale(step)
            denom = max(1, total_steps - warmup_steps)
            return max(0.0, float(total_steps - step) / float(denom))
        return LambdaLR(optimizer, lr_lambda)

    if scheduler_name == "cosine_warmup":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return warmup_scale(step)
            denom = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
            return 0.5 * (1.0 + math.cos(progress * math.pi))
        return LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown train.scheduler={scheduler_name!r}; expected none, linear_warmup_decay, or cosine_warmup")


def train_contrastive(
    data_dir: str | Path,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    splits_path: str | Path | None = None,
    cv_fold: int | None = None,
    val_fold: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], pd.DataFrame]:
    """Train CLIP-style projection heads on cached BGC and compound features."""
    bgc_cache_path, compound_cache_path = _get_cached_paths(cache_dir)
    interactions = build_interactions(data_dir, splits_path=splits_path, cv_fold=cv_fold, val_fold=val_fold)
    available_splits = set(interactions["split"].astype(str).str.lower().unique().tolist())

    bgc_dim, compound_dim = _infer_input_dims(bgc_cache_path, compound_cache_path)
    model = DualEncoderCLIP(
        bgc_input_dim=bgc_dim,
        compound_input_dim=compound_dim,
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
        bgc_aggregation=str(cfg["model"].get("bgc_aggregation", "prepooled")),
        bgc_aggregation_config=dict(cfg["model"].get("bgc_aggregation_config", {})),
        projection_head=str(cfg["model"].get("projection_head", "mlp_gelu")),
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
    train_positive_pairs = _build_positive_pair_set(interactions, split="train")
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = _build_lr_scheduler(
        optimizer,
        cfg,
        steps_per_epoch=len(train_loader),
        epochs=int(cfg["train"]["epochs"]),
    )

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[float]] = {"train_loss": []}
    epochs = int(cfg["train"]["epochs"])
    best_score = -1e9
    best_epoch = 0
    best_ckpt_path = outdir / "contrastive_model_best.pt"
    selection_split = str(cfg.get("eval", {}).get("selection_split", "val")).lower()
    selection_metric = str(cfg.get("eval", {}).get("selection_metric", "mean_mrr")).lower()
    if selection_split not in available_splits:
        if "val" in available_splits:
            selection_split = "val"
        elif "train" in available_splits:
            selection_split = "train"
        else:
            selection_split = sorted(available_splits)[0]

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            bgc_features = batch["bgc_feature"].to(device)
            bgc_padding_mask = batch.get("bgc_padding_mask")
            if bgc_padding_mask is not None:
                bgc_padding_mask = bgc_padding_mask.to(device)
            protein_positions = batch.get("protein_positions")
            domain_positions = batch.get("domain_positions")
            if protein_positions is not None:
                protein_positions = protein_positions.to(device)
            if domain_positions is not None:
                domain_positions = domain_positions.to(device)
            compound_features = batch["compound_feature"].to(device)
            positive_mask = _build_batch_positive_mask(
                bgc_ids=batch["bgc_id"],
                compound_ids=batch["compound_id"],
                positive_pairs=train_positive_pairs,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(
                bgc_features,
                compound_features,
                positive_mask=positive_mask,
                bgc_padding_mask=bgc_padding_mask,
                protein_positions=protein_positions,
                domain_positions=domain_positions,
            )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
        score = _val_selection_score(val_retrieval, metric=selection_metric)

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
                    "selection_metric": selection_metric,
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
            "selection_metric": selection_metric,
        },
        last_ckpt_path,
    )

    metrics: dict[str, Any] = {
        "train": {"loss_last_epoch": history["train_loss"][-1]},
        "model_selection": {
            "selection_split": selection_split,
            "selection_metric": selection_metric,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "best_score_mean_mrr": float(best_score) if selection_metric == "mean_mrr" else None,
            "best_checkpoint": str(best_ckpt_path),
        },
    }
    for split in ("train", "val", "test"):
        if split not in available_splits:
            continue
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
