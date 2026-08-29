from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from projects.mibig_bgc_np.training.contrastive_trainer import (
    _build_batch_positive_mask,
    _build_positive_pair_set,
    _get_cached_paths,
    _infer_input_dims,
)


def _parse_labels(value: Any) -> list[str]:
    raw = "" if value is None else str(value)
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
    candidates = parsed if isinstance(parsed, (list, tuple, set)) else re.split(r"[;,]", raw)
    labels: list[str] = []
    seen: set[str] = set()
    for label in candidates:
        clean = str(label).strip().strip("'\"")
        if clean and clean not in seen:
            labels.append(clean)
            seen.add(clean)
    return labels


def split_entities_and_pairs(
    interactions: pd.DataFrame,
    split: str,
) -> tuple[list[str], list[str], list[tuple[int, int]], pd.DataFrame]:
    df = interactions[interactions["split"].astype(str).str.lower() == split.lower()].copy().reset_index(drop=True)
    bgc_ids = sorted(df["bgc_id"].astype(str).unique().tolist())
    compound_ids = sorted(df["compound_id"].astype(str).unique().tolist())
    bgc_index = {bgc_id: idx for idx, bgc_id in enumerate(bgc_ids)}
    compound_index = {compound_id: idx for idx, compound_id in enumerate(compound_ids)}
    pairs = [
        (bgc_index[str(row.bgc_id)], compound_index[str(row.compound_id)])
        for row in df[["bgc_id", "compound_id"]].itertuples(index=False)
    ]
    return bgc_ids, compound_ids, pairs, df


def _metrics_from_sorted_positive_mask(sorted_pos: torch.Tensor) -> dict[str, float]:
    has_pos = sorted_pos.any(dim=1)
    first_idx = sorted_pos.float().argmax(dim=1)
    ranks = torch.where(has_pos, first_idx + 1, torch.full_like(first_idx, fill_value=sorted_pos.size(1) + 1))
    metrics = {"mrr": float((1.0 / ranks.float()).mean().item())}
    positives = float(sorted_pos.float().sum().item())
    for k in (1, 5, 10, 20, 50, 100, 200, 500):
        cutoff = min(int(k), int(sorted_pos.size(1)))
        hits = float(sorted_pos[:, :cutoff].float().sum().item())
        metrics[f"hit_at_{k}"] = float(sorted_pos[:, :cutoff].any(dim=1).float().mean().item())
        metrics[f"recall_at_{k}"] = hits / positives if positives > 0.0 else 0.0
        metrics[f"precision_at_{k}"] = float((sorted_pos[:, :cutoff].float().sum(dim=1) / float(k)).mean().item())
    return metrics


def evaluate_similarity_retrieval(
    bgc_to_compound_scores: torch.Tensor,
    pairs: list[tuple[int, int]],
    *,
    compound_to_bgc_scores: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    if compound_to_bgc_scores is None:
        compound_to_bgc_scores = bgc_to_compound_scores.t()
    n_bgcs, n_compounds = bgc_to_compound_scores.shape
    pos_bgc_to_compound = torch.zeros((n_bgcs, n_compounds), dtype=torch.bool)
    pos_compound_to_bgc = torch.zeros((n_compounds, n_bgcs), dtype=torch.bool)
    if pairs:
        pair_array = np.asarray(pairs, dtype=np.int64)
        bgc_idx = torch.tensor(pair_array[:, 0], dtype=torch.long)
        compound_idx = torch.tensor(pair_array[:, 1], dtype=torch.long)
        pos_bgc_to_compound[bgc_idx, compound_idx] = True
        pos_compound_to_bgc[compound_idx, bgc_idx] = True

    sorted_compounds = torch.argsort(bgc_to_compound_scores.cpu(), dim=1, descending=True)
    sorted_pos_bgc = pos_bgc_to_compound.gather(1, sorted_compounds)
    sorted_bgcs = torch.argsort(compound_to_bgc_scores.cpu(), dim=1, descending=True)
    sorted_pos_compound = pos_compound_to_bgc.gather(1, sorted_bgcs)
    return {
        "bgc_to_compound": _metrics_from_sorted_positive_mask(sorted_pos_bgc),
        "compound_to_bgc": _metrics_from_sorted_positive_mask(sorted_pos_compound),
    }


def _aggregate_metric_dicts(values: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    if not values:
        return {}
    out: dict[str, Any] = {}
    for direction in sorted(values[0]):
        out[direction] = {}
        for metric in sorted(values[0][direction]):
            arr = np.asarray([value[direction][metric] for value in values], dtype=np.float64)
            out[direction][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
    return out


def random_retrieval_baseline(
    n_bgcs: int,
    n_compounds: int,
    pairs: list[tuple[int, int]],
    *,
    seed: int,
    n_trials: int = 10,
) -> dict[str, Any]:
    trials: list[dict[str, dict[str, float]]] = []
    for offset in range(int(n_trials)):
        rng = np.random.default_rng(int(seed) + offset)
        bgc_scores = torch.tensor(rng.random((n_bgcs, n_compounds)), dtype=torch.float32)
        compound_scores = torch.tensor(rng.random((n_compounds, n_bgcs)), dtype=torch.float32)
        trials.append(evaluate_similarity_retrieval(bgc_scores, pairs, compound_to_bgc_scores=compound_scores))
    return {
        "name": "random",
        "seed": int(seed),
        "n_trials": int(n_trials),
        "metrics": _aggregate_metric_dicts(trials),
        "trials": trials,
    }


def _labels_by_bgc(df: pd.DataFrame) -> dict[str, set[str]]:
    label_col = "bgc_classes" if "bgc_classes" in df.columns else "bgc_class"
    if label_col not in df.columns:
        return {}
    labels: dict[str, set[str]] = {}
    for row in df[["bgc_id", label_col]].drop_duplicates().itertuples(index=False):
        labels.setdefault(str(row.bgc_id), set()).update(_parse_labels(getattr(row, label_col)))
    return labels


def _labels_by_compound(df: pd.DataFrame) -> dict[str, set[str]]:
    label_col = "bgc_classes" if "bgc_classes" in df.columns else "bgc_class"
    if label_col not in df.columns:
        return {}
    labels: dict[str, set[str]] = {}
    for row in df[["compound_id", label_col]].drop_duplicates().itertuples(index=False):
        labels.setdefault(str(row.compound_id), set()).update(_parse_labels(getattr(row, label_col)))
    return labels


def class_matching_retrieval_baseline(
    interactions: pd.DataFrame,
    split: str,
    *,
    binary: bool = False,
) -> dict[str, Any]:
    bgc_ids, compound_ids, pairs, split_df = split_entities_and_pairs(interactions, split)
    bgc_labels = _labels_by_bgc(split_df)
    compound_labels = _labels_by_compound(split_df)
    scores = torch.zeros((len(bgc_ids), len(compound_ids)), dtype=torch.float32)
    for i, bgc_id in enumerate(bgc_ids):
        left = bgc_labels.get(str(bgc_id), set())
        for j, compound_id in enumerate(compound_ids):
            shared = len(left.intersection(compound_labels.get(str(compound_id), set())))
            scores[i, j] = float(shared > 0) if binary else float(shared)
    return {
        "name": "class_matching_binary" if binary else "class_matching_shared_count",
        "metrics": evaluate_similarity_retrieval(scores, pairs),
        "n_bgcs": int(len(bgc_ids)),
        "n_compounds": int(len(compound_ids)),
        "n_pairs": int(len(pairs)),
    }


def _stack_cached_features(ids: list[str], cache: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([_as_baseline_feature(cache[item]) for item in ids])


def _as_baseline_feature(feature: torch.Tensor) -> torch.Tensor:
    feature = feature.float()
    if feature.ndim == 1:
        return feature
    if feature.ndim == 2:
        return feature.mean(dim=0)
    raise ValueError(f"Cached feature must be 1D or 2D, got {tuple(feature.shape)}")


def _fixed_random_projection(matrix: torch.Tensor, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(
        (int(matrix.size(1)), int(output_dim)),
        generator=generator,
        dtype=matrix.dtype,
    ) / math.sqrt(float(output_dim))
    return matrix @ projection


def frozen_encoder_similarity_baseline(
    interactions: pd.DataFrame,
    split: str,
    cache_dir: str | Path,
    *,
    seed: int,
    projection_dim: int | None = None,
) -> dict[str, Any]:
    bgc_ids, compound_ids, pairs, _ = split_entities_and_pairs(interactions, split)
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(Path(cache_dir) / "compound_features.pt", map_location="cpu")
    bgc_features = _stack_cached_features(bgc_ids, bgc_cache)
    compound_features = _stack_cached_features(compound_ids, compound_cache)
    bgc_dim = int(bgc_features.size(1)) if bgc_features.ndim == 2 else 0
    compound_dim = int(compound_features.size(1)) if compound_features.ndim == 2 else 0

    projection: dict[str, Any]
    if bgc_dim == compound_dim:
        bgc_projected = bgc_features
        compound_projected = compound_features
        projection = {"type": "identity", "output_dim": int(bgc_dim)}
    else:
        output_dim = int(projection_dim or min(512, bgc_dim, compound_dim))
        if output_dim <= 0:
            raise ValueError(
                f"Cannot build frozen encoder similarity baseline with dimensions "
                f"BGC={bgc_dim}, compound={compound_dim}."
            )
        bgc_projected = _fixed_random_projection(bgc_features, output_dim=output_dim, seed=int(seed))
        compound_projected = _fixed_random_projection(compound_features, output_dim=output_dim, seed=int(seed) + 100_003)
        projection = {
            "type": "fixed_gaussian_random_projection",
            "output_dim": int(output_dim),
            "seed_bgc": int(seed),
            "seed_compound": int(seed) + 100_003,
        }

    bgc_projected = F.normalize(bgc_projected, dim=-1)
    compound_projected = F.normalize(compound_projected, dim=-1)
    scores = bgc_projected @ compound_projected.t()
    return {
        "name": "frozen_encoder_similarity",
        "metrics": evaluate_similarity_retrieval(scores, pairs),
        "n_bgcs": int(len(bgc_ids)),
        "n_compounds": int(len(compound_ids)),
        "n_pairs": int(len(pairs)),
        "input": "cached frozen BGC and NP features",
        "scoring": "cosine_similarity",
        "training": "none",
        "bgc_input_dim": int(bgc_dim),
        "compound_input_dim": int(compound_dim),
        "projection": projection,
    }


class LinearDualEncoderCLIP(nn.Module):
    def __init__(
        self,
        bgc_input_dim: int,
        compound_input_dim: int,
        emb_dim: int,
        init_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.bgc_proj = nn.Linear(bgc_input_dim, emb_dim)
        self.compound_proj = nn.Linear(compound_input_dim, emb_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32))
        self.max_logit_scale = float(max_logit_scale)

    def encode_bgc(self, bgc_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.bgc_proj(bgc_features.float()), dim=-1)

    def encode_compound(self, compound_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.compound_proj(compound_features.float()), dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_logit_scale)

    def forward(
        self,
        bgc_features: torch.Tensor,
        compound_features: torch.Tensor,
        positive_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from clip_core.losses import multi_positive_infonce_loss, symmetric_infonce_loss

        z_bgc = self.encode_bgc(bgc_features)
        z_cmp = self.encode_compound(compound_features)
        logits = self.get_logit_scale() * (z_bgc @ z_cmp.t())
        loss = multi_positive_infonce_loss(logits, positive_mask) if positive_mask is not None else symmetric_infonce_loss(logits)
        return loss, logits


class _MeanPooledCachedInteractionDataset(Dataset):
    """Cached interaction dataset for baselines that require fixed 1D feature vectors."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        bgc_cache_path: str | Path,
        compound_cache_path: str | Path,
        split: str,
    ) -> None:
        self.frame = interactions[
            interactions["split"].astype(str).str.lower() == str(split).lower()
        ].copy().reset_index(drop=True)
        self.bgc_cache: dict[str, torch.Tensor] = torch.load(bgc_cache_path, map_location="cpu")
        self.compound_cache: dict[str, torch.Tensor] = torch.load(compound_cache_path, map_location="cpu")

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        bgc_id = str(row["bgc_id"])
        compound_id = str(row["compound_id"])
        return {
            "bgc_id": bgc_id,
            "compound_id": compound_id,
            "bgc_feature": _as_baseline_feature(self.bgc_cache[bgc_id]),
            "compound_feature": _as_baseline_feature(self.compound_cache[compound_id]),
        }


def _collate_mean_pooled_interactions(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bgc_id": [str(item["bgc_id"]) for item in batch],
        "compound_id": [str(item["compound_id"]) for item in batch],
        "bgc_feature": torch.stack([item["bgc_feature"] for item in batch]),
        "compound_feature": torch.stack([item["compound_feature"] for item in batch]),
    }


def _build_loader(
    interactions: pd.DataFrame,
    cache_dir: str | Path,
    split: str,
    cfg: dict[str, Any],
    *,
    shuffle: bool,
) -> DataLoader:
    bgc_cache_path, compound_cache_path = _get_cached_paths(cache_dir)
    dataset = _MeanPooledCachedInteractionDataset(interactions, bgc_cache_path, compound_cache_path, split)
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=_collate_mean_pooled_interactions,
        pin_memory=True,
    )


def train_linear_projection_baseline(
    interactions: pd.DataFrame,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    outdir: str | Path,
    *,
    patience: int | None = None,
) -> tuple[LinearDualEncoderCLIP, dict[str, Any]]:
    bgc_cache_path, compound_cache_path = _get_cached_paths(cache_dir)
    bgc_dim, compound_dim = _infer_input_dims(bgc_cache_path, compound_cache_path)
    model = LinearDualEncoderCLIP(
        bgc_input_dim=bgc_dim,
        compound_input_dim=compound_dim,
        emb_dim=int(cfg["model"]["emb_dim"]),
        init_temperature=float(cfg["model"].get("init_temperature", 0.07)),
        max_logit_scale=float(cfg["model"].get("max_logit_scale", 100.0)),
    ).to(device)
    train_loader = _build_loader(interactions, cache_dir, "train", cfg, shuffle=True)
    positive_pairs = _build_positive_pair_set(interactions, "train")
    optimizer = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "linear_projection_baseline_best.pt"
    available_splits = set(interactions["split"].astype(str).str.lower())
    val_loader = _build_loader(interactions, cache_dir, "val", cfg, shuffle=False) if "val" in available_splits else None
    val_positive_pairs = _build_positive_pair_set(interactions, "val") if val_loader is not None else set()
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    patience_value = int(patience if patience is not None else cfg["train"].get("patience", 0))
    stale = 0
    for epoch in tqdm(range(1, int(cfg["train"]["epochs"]) + 1), desc=f"{out.name} linear baseline", leave=False):
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            bgc_features = batch["bgc_feature"].to(device)
            compound_features = batch["compound_feature"].to(device)
            positive_mask = _build_batch_positive_mask(batch["bgc_id"], batch["compound_id"], positive_pairs, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(bgc_features, compound_features, positive_mask=positive_mask)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * bgc_features.size(0)
            count += bgc_features.size(0)
        train_loss = running / max(count, 1)
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    bgc_features = batch["bgc_feature"].to(device)
                    compound_features = batch["compound_feature"].to(device)
                    positive_mask = _build_batch_positive_mask(
                        batch["bgc_id"],
                        batch["compound_id"],
                        val_positive_pairs,
                        device,
                    )
                    loss, _ = model(bgc_features, compound_features, positive_mask=positive_mask)
                    val_running += float(loss.item()) * bgc_features.size(0)
                    val_count += bgc_features.size(0)
            val_loss = val_running / max(val_count, 1)
        selection_loss = val_loss if val_loader is not None else train_loss
        history.append({"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})
        if selection_loss < best_loss:
            best_loss = float(selection_loss)
            best_epoch = int(epoch)
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "bgc_input_dim": bgc_dim,
                    "compound_input_dim": compound_dim,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "selection_split": "val" if val_loader is not None else "train",
                },
                best_path,
            )
        else:
            stale += 1
            if patience_value > 0 and stale >= patience_value:
                break
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    metrics = {
        "name": "linear_projection",
        "model_selection": {
            "selection_split": "val" if val_loader is not None else "train",
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss),
            "best_checkpoint": str(best_path),
        },
        "history": history,
    }
    for split in ("train", "val", "test"):
        if split not in set(interactions["split"].astype(str).str.lower()):
            continue
        metrics[f"retrieval_{split}"] = evaluate_model_retrieval_baseline(
            model=model,
            interactions=interactions,
            split=split,
            cache_dir=cache_dir,
            device=device,
            batch_size=int(cfg["eval"]["sim_batch_size"]),
        )
    return model, metrics


def evaluate_model_retrieval_baseline(
    model: nn.Module,
    interactions: pd.DataFrame,
    split: str,
    cache_dir: str | Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    bgc_ids, compound_ids, pairs, _ = split_entities_and_pairs(interactions, split)
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(Path(cache_dir) / "compound_features.pt", map_location="cpu")
    bgc_chunks: list[torch.Tensor] = []
    compound_chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bgc_ids), batch_size):
            ids = bgc_ids[start : start + batch_size]
            bgc_chunks.append(model.encode_bgc(_stack_cached_features(ids, bgc_cache).to(device)).cpu())
        for start in range(0, len(compound_ids), batch_size):
            ids = compound_ids[start : start + batch_size]
            compound_chunks.append(
                model.encode_compound(_stack_cached_features(ids, compound_cache).to(device)).cpu()
            )
    bgc_embs = torch.cat(bgc_chunks, dim=0) if bgc_chunks else torch.empty((0, 0))
    compound_embs = torch.cat(compound_chunks, dim=0) if compound_chunks else torch.empty((0, 0))
    return evaluate_similarity_retrieval(bgc_embs @ compound_embs.t(), pairs)


def _normalize_feature_matrix(ids: list[str], cache: dict[str, torch.Tensor]) -> torch.Tensor:
    matrix = _stack_cached_features(ids, cache)
    return F.normalize(matrix, dim=-1)


def _tanimoto_feature_matrix(query_ids: list[str], reference_ids: list[str], cache: dict[str, torch.Tensor]) -> torch.Tensor:
    query = torch.stack([cache[item].float() for item in query_ids]) > 0
    reference = torch.stack([cache[item].float() for item in reference_ids]) > 0
    intersection = query.float() @ reference.float().t()
    query_sum = query.float().sum(dim=1, keepdim=True)
    reference_sum = reference.float().sum(dim=1, keepdim=True).t()
    union = query_sum + reference_sum - intersection
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(intersection))


def knn_transfer_retrieval_baseline(
    interactions: pd.DataFrame,
    split: str,
    cache_dir: str | Path,
    *,
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
    train_split: str = "train",
) -> dict[str, Any]:
    bgc_ids, compound_ids, pairs, _ = split_entities_and_pairs(interactions, split)
    train_df = interactions[interactions["split"].astype(str).str.lower() == train_split.lower()].copy()
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(Path(cache_dir) / "compound_features.pt", map_location="cpu")
    train_bgc_ids = sorted(train_df["bgc_id"].astype(str).unique().tolist())
    train_compound_ids = sorted(train_df["compound_id"].astype(str).unique().tolist())
    test_bgc_features = _normalize_feature_matrix(bgc_ids, bgc_cache)
    train_bgc_features = _normalize_feature_matrix(train_bgc_ids, bgc_cache)
    test_compound_features = _normalize_feature_matrix(compound_ids, compound_cache)
    train_compound_features = _normalize_feature_matrix(train_compound_ids, compound_cache)
    bgc_train_sim = test_bgc_features @ train_bgc_features.t()
    compound_train_sim = test_compound_features @ train_compound_features.t()

    train_bgcs_by_compound: dict[str, list[int]] = {}
    train_bgc_index = {bgc_id: idx for idx, bgc_id in enumerate(train_bgc_ids)}
    for row in train_df[["bgc_id", "compound_id"]].drop_duplicates().itertuples(index=False):
        train_bgcs_by_compound.setdefault(str(row.compound_id), []).append(train_bgc_index[str(row.bgc_id)])

    train_compounds_by_bgc: dict[str, list[int]] = {}
    train_compound_index = {compound_id: idx for idx, compound_id in enumerate(train_compound_ids)}
    for row in train_df[["bgc_id", "compound_id"]].drop_duplicates().itertuples(index=False):
        train_compounds_by_bgc.setdefault(str(row.bgc_id), []).append(train_compound_index[str(row.compound_id)])

    results: dict[str, Any] = {
        "name": "knn_transfer",
        "route_scoring": "maximum similarity over associated training routes",
        "bgc_similarity": "cosine",
        "compound_similarity": "cosine",
        "k_values": [int(k) for k in k_values],
        "metrics_by_k": {},
    }
    for k in k_values:
        bgc_to_compound = torch.zeros((len(bgc_ids), len(compound_ids)), dtype=torch.float32)
        for j, compound_id in enumerate(compound_ids):
            train_indices = train_bgcs_by_compound.get(str(compound_id), [])
            if train_indices:
                values = bgc_train_sim[:, train_indices]
                bgc_to_compound[:, j] = values.max(dim=1).values

        compound_to_bgc = torch.zeros((len(compound_ids), len(bgc_ids)), dtype=torch.float32)
        for i, bgc_id in enumerate(bgc_ids):
            train_indices = train_compounds_by_bgc.get(str(bgc_id), [])
            if train_indices:
                values = compound_train_sim[:, train_indices]
                compound_to_bgc[:, i] = values.max(dim=1).values

        results["metrics_by_k"][str(int(k))] = evaluate_similarity_retrieval(
            bgc_to_compound,
            pairs,
            compound_to_bgc_scores=compound_to_bgc,
        )
    return results


def evaluate_score_only_retrieval_baselines(
    interactions: pd.DataFrame,
    split: str,
    cache_dir: str | Path,
    *,
    seed: int,
    random_trials: int = 10,
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
    projection_dim: int | None = None,
) -> dict[str, Any]:
    bgc_ids, compound_ids, pairs, _ = split_entities_and_pairs(interactions, split)
    return {
        "split": split,
        "random": random_retrieval_baseline(
            len(bgc_ids),
            len(compound_ids),
            pairs,
            seed=int(seed),
            n_trials=int(random_trials),
        ),
        "frozen_encoder_similarity": frozen_encoder_similarity_baseline(
            interactions,
            split,
            cache_dir,
            seed=int(seed),
            projection_dim=projection_dim,
        ),
        "knn_transfer": knn_transfer_retrieval_baseline(interactions, split, cache_dir, k_values=k_values),
    }


def run_retrieval_baseline_suite(
    interactions: pd.DataFrame,
    split: str,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    outdir: str | Path,
    *,
    seed: int,
    random_trials: int = 10,
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
    patience: int | None = None,
    include_linear: bool = True,
) -> dict[str, Any]:
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    baselines = evaluate_score_only_retrieval_baselines(
        interactions=interactions,
        split=split,
        cache_dir=cache_dir,
        seed=int(seed),
        random_trials=int(random_trials),
        k_values=k_values,
        projection_dim=int(cfg["model"]["emb_dim"]),
    )
    if include_linear:
        linear_model, linear_metrics = train_linear_projection_baseline(
            interactions=interactions,
            cache_dir=cache_dir,
            cfg=cfg,
            device=device,
            outdir=output / "linear_projection",
            patience=patience,
        )
        del linear_model
        baselines["linear_projection"] = {
            "name": "linear_projection",
            "metrics": linear_metrics.get(f"retrieval_{split}", {}),
            "all_metrics": linear_metrics,
        }
    save_json_path = output / f"retrieval_baselines_{split}.json"
    baselines["path"] = str(save_json_path)
    from clip_core.logging import save_json

    save_json(baselines, save_json_path)
    return baselines
