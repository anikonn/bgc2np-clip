"""Shared CLIP utilities for multiple retrieval projects."""

from .cache import FeatureCache
from .config import apply_overrides, load_yaml
from .logging import save_json, setup_logger
from .losses import symmetric_infonce_loss
from .retrieval import (
    RetrievalMetrics,
    batched_similarity,
    evaluate_global_retrieval,
    evaluate_global_retrieval_multi,
)

__all__ = [
    "FeatureCache",
    "RetrievalMetrics",
    "apply_overrides",
    "batched_similarity",
    "evaluate_global_retrieval",
    "evaluate_global_retrieval_multi",
    "load_yaml",
    "save_json",
    "setup_logger",
    "symmetric_infonce_loss",
]
