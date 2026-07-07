from __future__ import annotations

from clip_core.retrieval import RetrievalMetrics, batched_similarity
from clip_core.retrieval import evaluate_global_retrieval_multi as _core_evaluate_global_retrieval_multi


def evaluate_global_retrieval(
    bgc_embs: object | None = None,
    compound_embs: object | None = None,
    interaction_pairs: list[tuple[int, int]] | None = None,
    sim_batch_size: int = 1024,
    *,
    left_embs: object | None = None,
    right_embs: object | None = None,
    pair_indices: list[tuple[int, int]] | None = None,
) -> dict[str, dict[str, float]]:
    """Single-positive wrapper kept for API parity with the shared retrieval helpers."""
    left = bgc_embs if bgc_embs is not None else left_embs
    right = compound_embs if compound_embs is not None else right_embs
    pairs = interaction_pairs if interaction_pairs is not None else pair_indices
    return _core_evaluate_global_retrieval_multi(
        left_embs=left,
        right_embs=right,
        pair_indices=pairs,
        sim_batch_size=sim_batch_size,
        left_label="bgc",
        right_label="compound",
    )


def evaluate_global_retrieval_multi(
    bgc_embs: object | None = None,
    compound_embs: object | None = None,
    interaction_pairs: list[tuple[int, int]] | None = None,
    sim_batch_size: int = 1024,
    *,
    left_embs: object | None = None,
    right_embs: object | None = None,
    pair_indices: list[tuple[int, int]] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Multi-positive retrieval evaluation for MIBiG BGC-compound embeddings.

    The legacy left/right keyword aliases are accepted so this mirrors the
    flexibility of the shared retrieval API.
    """
    left = bgc_embs if bgc_embs is not None else left_embs
    right = compound_embs if compound_embs is not None else right_embs
    pairs = interaction_pairs if interaction_pairs is not None else pair_indices
    return _core_evaluate_global_retrieval_multi(
        left_embs=left,
        right_embs=right,
        pair_indices=pairs,
        sim_batch_size=sim_batch_size,
        left_label="bgc",
        right_label="compound",
    )


__all__ = [
    "RetrievalMetrics",
    "batched_similarity",
    "evaluate_global_retrieval",
    "evaluate_global_retrieval_multi",
]
