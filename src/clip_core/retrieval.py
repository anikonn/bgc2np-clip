from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RetrievalMetrics:
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    hit_at_20: float
    hit_at_50: float
    hit_at_100: float
    hit_at_200: float
    hit_at_500: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    recall_at_50: float
    recall_at_100: float
    recall_at_200: float
    recall_at_500: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    precision_at_50: float
    precision_at_100: float
    precision_at_200: float
    precision_at_500: float
    mrr: float


def batched_similarity(a: torch.Tensor, b: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """Compute A @ B^T using row-wise batching."""
    chunks: list[torch.Tensor] = []
    for i in range(0, a.size(0), batch_size):
        chunks.append(a[i : i + batch_size] @ b.t())
    return torch.cat(chunks, dim=0)


def _metrics_from_sorted_positive_mask(sorted_pos: torch.Tensor) -> RetrievalMetrics:
    positives = float(sorted_pos.float().sum().item())

    def hit_at(k: int) -> float:
        cutoff = min(int(k), int(sorted_pos.size(1)))
        return float(sorted_pos[:, :cutoff].any(dim=1).float().mean().item())

    def recall_at(k: int) -> float:
        cutoff = min(int(k), int(sorted_pos.size(1)))
        if positives <= 0.0:
            return 0.0
        hits = float(sorted_pos[:, :cutoff].float().sum().item())
        return hits / positives

    def precision_at(k: int) -> float:
        cutoff = min(int(k), int(sorted_pos.size(1)))
        return float((sorted_pos[:, :cutoff].float().sum(dim=1) / float(k)).mean().item())

    has_pos = sorted_pos.any(dim=1)
    first_idx = sorted_pos.float().argmax(dim=1)
    ranks = torch.where(has_pos, first_idx + 1, torch.full_like(first_idx, fill_value=sorted_pos.size(1) + 1))
    mrr = float((1.0 / ranks.float()).mean().item())

    return RetrievalMetrics(
        hit_at_1=hit_at(1),
        hit_at_5=hit_at(5),
        hit_at_10=hit_at(10),
        hit_at_20=hit_at(20),
        hit_at_50=hit_at(50),
        hit_at_100=hit_at(100),
        hit_at_200=hit_at(200),
        hit_at_500=hit_at(500),
        recall_at_1=recall_at(1),
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        recall_at_20=recall_at(20),
        recall_at_50=recall_at(50),
        recall_at_100=recall_at(100),
        recall_at_200=recall_at(200),
        recall_at_500=recall_at(500),
        precision_at_1=precision_at(1),
        precision_at_5=precision_at(5),
        precision_at_10=precision_at(10),
        precision_at_20=precision_at(20),
        precision_at_50=precision_at(50),
        precision_at_100=precision_at(100),
        precision_at_200=precision_at(200),
        precision_at_500=precision_at(500),
        mrr=mrr,
    )


def evaluate_global_retrieval(
    left_embs: torch.Tensor,
    right_embs: torch.Tensor,
    pair_indices: list[tuple[int, int]],
    sim_batch_size: int = 1024,
) -> dict[str, dict[str, float]]:
    """Single-positive retrieval wrapper for paired datasets."""
    return evaluate_global_retrieval_multi(
        left_embs=left_embs,
        right_embs=right_embs,
        pair_indices=pair_indices,
        sim_batch_size=sim_batch_size,
        left_label="left",
        right_label="right",
    )


def evaluate_global_retrieval_multi(
    protein_embs: torch.Tensor | None = None,
    ligand_embs: torch.Tensor | None = None,
    interaction_pairs: list[tuple[int, int]] | None = None,
    sim_batch_size: int = 1024,
    *,
    left_embs: torch.Tensor | None = None,
    right_embs: torch.Tensor | None = None,
    pair_indices: list[tuple[int, int]] | None = None,
    left_label: str = "protein",
    right_label: str = "ligand",
) -> dict[str, dict[str, float]]:
    """
    Multi-positive retrieval evaluation over two embedding sets.

    The legacy protein/ligand argument names are kept for backward compatibility.
    """
    left = protein_embs if protein_embs is not None else left_embs
    right = ligand_embs if ligand_embs is not None else right_embs
    pairs = interaction_pairs if interaction_pairs is not None else pair_indices
    if left is None or right is None or pairs is None:
        raise ValueError("Embeddings and positive pair indices are required for retrieval evaluation.")

    device = left.device
    sim = batched_similarity(left, right, batch_size=sim_batch_size)
    n_left, n_right = sim.shape

    pos_left_to_right = torch.zeros((n_left, n_right), dtype=torch.bool, device=device)
    pos_right_to_left = torch.zeros((n_right, n_left), dtype=torch.bool, device=device)

    pair_array = np.asarray(pairs, dtype=np.int64)
    left_idx = torch.tensor(pair_array[:, 0], dtype=torch.long, device=device)
    right_idx = torch.tensor(pair_array[:, 1], dtype=torch.long, device=device)

    pos_left_to_right[left_idx, right_idx] = True
    pos_right_to_left[right_idx, left_idx] = True

    sorted_right = torch.argsort(sim, dim=1, descending=True)
    sorted_pos_left_to_right = pos_left_to_right.gather(1, sorted_right)
    left_to_right_metrics = _metrics_from_sorted_positive_mask(sorted_pos_left_to_right)

    sim_t = sim.t()
    sorted_left = torch.argsort(sim_t, dim=1, descending=True)
    sorted_pos_right_to_left = pos_right_to_left.gather(1, sorted_left)
    right_to_left_metrics = _metrics_from_sorted_positive_mask(sorted_pos_right_to_left)

    return {
        f"{left_label}_to_{right_label}": left_to_right_metrics.__dict__,
        f"{right_label}_to_{left_label}": right_to_left_metrics.__dict__,
    }
