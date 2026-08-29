from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return 0.0
    coef = float(np.corrcoef(y_true, y_pred)[0, 1])
    if np.isnan(coef):
        return 0.0
    return coef


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    coef, _ = spearmanr(y_true, y_pred)
    if np.isnan(coef):
        return 0.0
    return float(coef)
