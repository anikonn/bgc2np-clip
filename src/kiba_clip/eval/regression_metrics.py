from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    coef, _ = spearmanr(y_true, y_pred)
    if np.isnan(coef):
        return 0.0
    return float(coef)
