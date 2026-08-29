"""Shared CLIP utilities for multiple retrieval projects."""

from .cache import FeatureCache
from .logging import save_json, setup_logger
from .losses import multi_positive_infonce_loss, symmetric_infonce_loss
from .retrieval import (
    RetrievalMetrics,
    batched_similarity,
    evaluate_global_retrieval,
    evaluate_global_retrieval_multi,
)


def apply_overrides(cfg, overrides):
    from .config import apply_overrides as _apply_overrides

    return _apply_overrides(cfg, overrides)


def load_yaml(path):
    from .config import load_yaml as _load_yaml

    return _load_yaml(path)


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
    "multi_positive_infonce_loss",
]
