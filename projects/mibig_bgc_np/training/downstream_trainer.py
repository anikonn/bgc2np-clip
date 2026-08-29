from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from clip_core.logging import save_json
from projects.mibig_bgc_np.eval.classification_metrics import (
    compute_confusion_matrix,
    confusion_matrix_normalized,
    macro_micro_f1_from_cm,
    per_class_prf,
    random_baselines,
    wrong_class_ratios,
)
from projects.mibig_bgc_np.eval.regression_metrics import pearson, rmse, spearman
from projects.mibig_bgc_np.data.datasets import build_bgc_class_table, build_interactions
from projects.mibig_bgc_np.models.classification import BGCClassifier
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.training.contrastive_trainer import _pad_bgc_features
from projects.mibig_bgc_np.models.regression import EmbeddingRegressor

LOGGER = logging.getLogger("mibig_bgc_np")
BIOACTIVITY_CLASS_NAMES = ["antibacterial", "cytotoxic", "antifungal", "inhibitor", "siderophore", "antiviral"]
NPCLASSIFIER_TASKS = {
    "npclassifier_pathway": {
        "level": "pathway",
        "label_column": "npclassifier_pathway",
        "counts_path": None,
        "min_count": None,
        "display_name": "NPClassifier pathway",
    },
    "npclassifier_superclass": {
        "level": "superclass",
        "label_column": "npclassifier_superclass",
        "counts_path": "results/downstream_distributions/npclassifier_superclass_counts.csv",
        "min_count": 100,
        "display_name": "NPClassifier superclass",
    },
    "npclassifier_class": {
        "level": "class",
        "label_column": "npclassifier_class",
        "counts_path": "results/downstream_distributions/npclassifier_class_counts.csv",
        "min_count": 50,
        "display_name": "NPClassifier class",
    },
}
DEFAULT_TASKS = (
    "bgc_class",
    "bioactivity_class",
    "npclassifier_pathway",
    "npclassifier_superclass",
    "npclassifier_class",
    "compound_mw",
    "compound_logp",
    "compound_tpsa",
    "origin_type",
)
COMPOUND_REGRESSION_TASKS = {
    "compound_mw": {
        "target": "compound_molecular_weight",
        "display_name": "Molecular weight",
        "short_name": "mw",
        "hist_title": "Compound molecular weight distribution",
        "x_label": "Molecular weight",
        "boxplot_ylabel": "Molecular weight",
    },
    "compound_logp": {
        "target": "compound_logp",
        "display_name": "Wildman-Crippen logP",
        "short_name": "logp",
        "hist_title": "Compound logP distribution",
        "x_label": "logP",
        "boxplot_ylabel": "logP",
    },
    "compound_tpsa": {
        "target": "compound_tpsa",
        "display_name": "Topological polar surface area",
        "short_name": "tpsa",
        "hist_title": "Compound TPSA distribution",
        "x_label": "TPSA",
        "boxplot_ylabel": "TPSA",
    },
}
COMPOUND_TASKS = set(COMPOUND_REGRESSION_TASKS) | {"origin_type"}
ORIGIN_LABEL_TO_IDX = {"Bacterium": 0, "Fungus": 1}
ORIGIN_CLASS_NAMES = ["Bacterium", "Fungus"]


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def _load_contrastive_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DualEncoderCLIP, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = DualEncoderCLIP(
        bgc_input_dim=ckpt["bgc_input_dim"],
        compound_input_dim=ckpt["compound_input_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
        bgc_aggregation=str(cfg["model"].get("bgc_aggregation", "prepooled")),
        bgc_aggregation_config=dict(cfg["model"].get("bgc_aggregation_config", {})),
        projection_head=str(cfg["model"].get("projection_head", "mlp_gelu")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _predict_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    classifier.eval()
    if len(loader.dataset) == 0:
        empty = torch.empty(0, dtype=torch.long)
        return float("nan"), empty, empty, torch.empty((0, 0), dtype=torch.float32)

    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            running_loss += float(loss.item()) * x.size(0)
            count += x.size(0)
            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())

    y_true = torch.cat(targets_all)
    logits = torch.cat(logits_all)
    y_pred = logits.argmax(dim=-1)
    return running_loss / max(count, 1), y_true, y_pred, logits


def _predict_multilabel_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    classifier.eval()
    if len(loader.dataset) == 0:
        empty = torch.empty((0, 0), dtype=torch.float32)
        return float("nan"), empty, empty, empty

    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            running_loss += float(loss.item()) * x.size(0)
            count += x.size(0)
            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())

    y_true = torch.cat(targets_all, dim=0)
    logits = torch.cat(logits_all, dim=0)
    probs = torch.sigmoid(logits)
    y_pred = (probs >= 0.5).to(dtype=torch.float32)
    return running_loss / max(count, 1), y_true, y_pred, logits


def _predict_regressor(
    regressor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    regressor.eval()
    if len(loader.dataset) == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return float("nan"), empty, empty

    preds_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = regressor(x)
            loss = loss_fn(preds, y)
            running_loss += float(loss.item()) * x.size(0)
            count += x.size(0)
            preds_all.append(preds.cpu())
            targets_all.append(y.cpu())

    y_true = torch.cat(targets_all)
    y_pred = torch.cat(preds_all)
    return running_loss / max(count, 1), y_true, y_pred


def _per_class_with_names(cm: torch.Tensor, class_names: list[str]) -> dict[str, Any]:
    per_class = per_class_prf(cm)
    return {
        class_name: {
            "precision": float(per_class["precision"][idx]),
            "recall": float(per_class["recall"][idx]),
            "f1": float(per_class["f1"][idx]),
            "support": float(per_class["support"][idx]),
        }
        for idx, class_name in enumerate(class_names)
    }


def _matrix_with_class_names(matrix: torch.Tensor, class_names: list[str]) -> dict[str, dict[str, int | float]]:
    values = matrix.detach().cpu().tolist()
    named: dict[str, dict[str, int | float]] = {}
    for true_idx, true_name in enumerate(class_names):
        named[true_name] = {}
        for pred_idx, pred_name in enumerate(class_names):
            value = values[true_idx][pred_idx]
            named[true_name][pred_name] = int(value) if isinstance(value, int) else float(value)
    return named


def _matrix_from_named(matrix: dict[str, dict[str, int | float]], class_names: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[matrix[true_name][pred_name] for pred_name in class_names] for true_name in class_names],
        dtype=torch.float32,
    )


def _binary_roc_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    true_np = y_true.detach().to(dtype=torch.long, device="cpu").numpy()
    score_np = y_score.detach().to(dtype=torch.float64, device="cpu").numpy()
    pos_mask = true_np == 1
    neg_mask = true_np == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(score_np, kind="mergesort")
    sorted_scores = score_np[order]
    ranks = np.arange(1, len(score_np) + 1, dtype=np.float64)

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = float(ranks[start:end].mean())
        ranks[start:end] = avg_rank
        start = end

    inv_ranks = np.empty_like(ranks)
    inv_ranks[order] = ranks
    rank_sum_pos = float(inv_ranks[pos_mask].sum())
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _binary_roc_curve(y_true: torch.Tensor, y_score: torch.Tensor) -> tuple[list[float], list[float], float]:
    true_np = y_true.detach().to(dtype=torch.long, device="cpu").numpy()
    score_np = y_score.detach().to(dtype=torch.float64, device="cpu").numpy()
    pos_mask = true_np == 1
    neg_mask = true_np == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0], 0.0

    order = np.argsort(-score_np, kind="mergesort")
    sorted_true = true_np[order]
    sorted_scores = score_np[order]

    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_idxs = np.r_[distinct, len(sorted_scores) - 1]

    tps = np.r_[0, tp[threshold_idxs]]
    fps = np.r_[0, fp[threshold_idxs]]
    tpr = tps / float(n_pos)
    fpr = fps / float(n_neg)
    auc = _trapezoid_integral(tpr, fpr)
    return fpr.tolist(), tpr.tolist(), float(max(0.0, min(1.0, auc)))


def _classification_report(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_train: torch.Tensor,
    num_classes: int,
    class_names: list[str],
    baseline_trials: int,
    baseline_seed: int,
) -> dict[str, Any]:
    loss_fn = nn.CrossEntropyLoss()
    loss, y_true, y_pred, logits = _predict_classifier(classifier, loader, device, loss_fn)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    overall = macro_micro_f1_from_cm(cm)
    overall["loss"] = float(loss)

    report: dict[str, Any] = {
        "loss": float(loss),
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "micro_f1": overall["micro_f1"],
        "overall": overall,
        "per_class": _per_class_with_names(cm, class_names),
        "confusion_matrix": {
            "labels": class_names,
            "raw": _matrix_with_class_names(cm, class_names),
            "normalized_true": _matrix_with_class_names(confusion_matrix_normalized(cm, mode="true"), class_names),
        },
        "wrong_ratios": wrong_class_ratios(y_true, y_pred, num_classes, class_names=class_names),
        "random_baselines": random_baselines(
            y_train=y_train,
            y_true=y_true,
            num_classes=num_classes,
            trials=baseline_trials,
            seed=baseline_seed,
        ),
    }

    if num_classes == 2 and logits.numel() > 0:
        probs = torch.softmax(logits, dim=-1)[:, 1]
        positive_name = class_names[1]
        positive_metrics = report["per_class"][positive_name]
        report["positive_class"] = {
            "label": positive_name,
            "precision": float(positive_metrics["precision"]),
            "recall": float(positive_metrics["recall"]),
            "f1": float(positive_metrics["f1"]),
        }
        report["roc_auc"] = _binary_roc_auc(y_true, probs)
        report["overall"]["roc_auc"] = report["roc_auc"]

    return report


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return slug.strip("_") or "unknown"


def _multilabel_per_class_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for idx, class_name in enumerate(class_names):
        true_col = y_true[:, idx].to(dtype=torch.bool)
        pred_col = y_pred[:, idx].to(dtype=torch.bool)
        tp = int((true_col & pred_col).sum().item())
        fp = int((~true_col & pred_col).sum().item())
        fn = int((true_col & ~pred_col).sum().item())
        support = int(true_col.sum().item())
        precision = 0.0 if (tp + fp) == 0 else tp / float(tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / float(tp + fn)
        f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
        metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(support),
            "prevalence": float(support / max(len(y_true), 1)),
        }
    return metrics


def _multilabel_overall_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    true_bool = y_true.to(dtype=torch.bool)
    pred_bool = y_pred.to(dtype=torch.bool)
    tp = int((true_bool & pred_bool).sum().item())
    fp = int((~true_bool & pred_bool).sum().item())
    fn = int((true_bool & ~pred_bool).sum().item())

    micro_precision = 0.0 if (tp + fp) == 0 else tp / float(tp + fp)
    micro_recall = 0.0 if (tp + fn) == 0 else tp / float(tp + fn)
    micro_f1 = 0.0 if (micro_precision + micro_recall) == 0.0 else (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )

    per_class = _multilabel_per_class_metrics(y_true, y_pred, class_names)
    macro_precision = float(np.mean([item["precision"] for item in per_class.values()])) if per_class else 0.0
    macro_recall = float(np.mean([item["recall"] for item in per_class.values()])) if per_class else 0.0
    macro_f1 = float(np.mean([item["f1"] for item in per_class.values()])) if per_class else 0.0

    exact_match = float((true_bool == pred_bool).all(dim=1).to(dtype=torch.float32).mean().item()) if len(y_true) else 0.0
    hamming_accuracy = float(1.0 - (true_bool != pred_bool).to(dtype=torch.float32).mean().item()) if len(y_true) else 0.0
    mean_labels_true = float(y_true.sum(dim=1).to(dtype=torch.float32).mean().item()) if len(y_true) else 0.0
    mean_labels_pred = float(y_pred.sum(dim=1).to(dtype=torch.float32).mean().item()) if len(y_pred) else 0.0

    return {
        "accuracy": exact_match,
        "subset_accuracy": exact_match,
        "hamming_accuracy": hamming_accuracy,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "mean_labels_true": mean_labels_true,
        "mean_labels_pred": mean_labels_pred,
    }


def _binary_confusion_named(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, Any]:
    labels = ["negative", "positive"]
    cm = compute_confusion_matrix(y_true.to(dtype=torch.long), y_pred.to(dtype=torch.long), 2)
    return {
        "labels": labels,
        "raw": _matrix_with_class_names(cm, labels),
        "normalized_true": _matrix_with_class_names(confusion_matrix_normalized(cm, mode="true"), labels),
    }


def _expanded_multilabel_confusion(
    y_true: torch.Tensor,
    top1_pred: torch.Tensor,
    class_names: list[str],
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    num_classes = len(class_names)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    if y_true.numel() == 0:
        return {
            "labels": class_names,
            "raw": _matrix_with_class_names(cm, class_names),
            "normalized_true": _matrix_with_class_names(confusion_matrix_normalized(cm, mode="true"), class_names),
            "n_rows": 0,
        }

    active_mask = torch.ones(y_true.size(0), dtype=torch.bool) if mask is None else mask.to(dtype=torch.bool)
    for row_idx in torch.nonzero(active_mask, as_tuple=False).flatten().tolist():
        true_indices = torch.nonzero(y_true[row_idx] > 0.0, as_tuple=False).flatten().tolist()
        pred_idx = int(top1_pred[row_idx].item())
        for true_idx in true_indices:
            cm[true_idx, pred_idx] += 1

    return {
        "labels": class_names,
        "raw": _matrix_with_class_names(cm, class_names),
        "normalized_true": _matrix_with_class_names(confusion_matrix_normalized(cm, mode="true"), class_names),
        "n_rows": int(active_mask.sum().item()),
    }


def _multilabel_classification_report(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    loss_fn = nn.BCEWithLogitsLoss()
    loss, y_true, y_pred, logits = _predict_multilabel_classifier(classifier, loader, device, loss_fn)
    probs = torch.sigmoid(logits) if logits.numel() else torch.empty_like(logits)
    overall = _multilabel_overall_metrics(y_true, y_pred, class_names)
    overall["loss"] = float(loss)

    top1_pred = logits.argmax(dim=-1) if logits.numel() else torch.empty(0, dtype=torch.long)
    single_class_mask = (y_true.sum(dim=1) == 1) if y_true.numel() else torch.empty(0, dtype=torch.bool)
    per_class = _multilabel_per_class_metrics(y_true, y_pred, class_names)

    per_class_binary: dict[str, Any] = {}
    roc_curves: dict[str, Any] = {}
    for idx, class_name in enumerate(class_names):
        y_true_col = y_true[:, idx] if y_true.numel() else torch.empty(0, dtype=torch.float32)
        y_pred_col = y_pred[:, idx] if y_pred.numel() else torch.empty(0, dtype=torch.float32)
        y_score_col = probs[:, idx] if probs.numel() else torch.empty(0, dtype=torch.float32)
        fpr, tpr, auc = _binary_roc_curve(y_true_col, y_score_col)
        per_class[class_name]["auroc"] = float(auc)
        per_class_binary[class_name] = {
            **per_class[class_name],
            "confusion_matrix": _binary_confusion_named(y_true_col, y_pred_col),
        }
        roc_curves[class_name] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "auroc": float(auc),
        }

    if probs.numel():
        micro_fpr, micro_tpr, micro_auroc = _binary_roc_curve(y_true.reshape(-1), probs.reshape(-1))
    else:
        micro_fpr, micro_tpr, micro_auroc = [0.0, 1.0], [0.0, 1.0], 0.0
    macro_auroc = float(np.mean([curve["auroc"] for curve in roc_curves.values()])) if roc_curves else 0.0
    overall["micro_auroc"] = float(micro_auroc)
    overall["macro_auroc"] = float(macro_auroc)

    report: dict[str, Any] = {
        "loss": float(loss),
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "micro_f1": overall["micro_f1"],
        "macro_auroc": overall["macro_auroc"],
        "micro_auroc": overall["micro_auroc"],
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": _expanded_multilabel_confusion(y_true, top1_pred, class_names),
        "confusion_matrix_single_class_only": _expanded_multilabel_confusion(
            y_true, top1_pred, class_names, mask=single_class_mask
        ),
        "per_class_binary": per_class_binary,
        "roc_curves": {
            "per_class": roc_curves,
            "micro": {
                "fpr": [float(x) for x in micro_fpr],
                "tpr": [float(x) for x in micro_tpr],
                "auroc": float(micro_auroc),
            },
        },
        "n_single_class_bgcs": int(single_class_mask.sum().item()) if single_class_mask.numel() else 0,
        "n_multilabel_bgcs": int((y_true.sum(dim=1) > 1).sum().item()) if y_true.numel() else 0,
        "random_baselines": {},
    }
    return report


def _regression_metrics_dict(y_true: torch.Tensor, y_pred: torch.Tensor, loss: float) -> dict[str, float]:
    true_np = y_true.detach().to(dtype=torch.float64, device="cpu").numpy()
    pred_np = y_pred.detach().to(dtype=torch.float64, device="cpu").numpy()
    mse = float(np.mean((true_np - pred_np) ** 2)) if true_np.size else 0.0
    ss_res = float(np.sum((true_np - pred_np) ** 2))
    true_mean = float(true_np.mean()) if true_np.size else 0.0
    ss_tot = float(np.sum((true_np - true_mean) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else 1.0 - (ss_res / ss_tot)
    return {
        "loss": float(loss),
        "mse": mse,
        "rmse": rmse(true_np, pred_np) if true_np.size else 0.0,
        "r2": float(r2),
        "pearson": pearson(true_np, pred_np) if true_np.size else 0.0,
        "spearman": spearman(true_np, pred_np) if true_np.size else 0.0,
    }


def _summarize_regression_trials(trial_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in ("mse", "rmse", "r2", "pearson", "spearman"):
        values = np.asarray([metrics[metric_name] for metrics in trial_metrics], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean()) if values.size else 0.0
        summary[f"{metric_name}_std"] = float(values.std(ddof=0)) if values.size else 0.0
    return summary


def _regression_baselines(
    y_train: torch.Tensor,
    y_true: torch.Tensor,
    trials: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    train_np = y_train.detach().to(dtype=torch.float64, device="cpu").numpy()
    true_np = y_true.detach().to(dtype=torch.float64, device="cpu").numpy()
    if train_np.size == 0:
        raise ValueError("y_train must contain at least one target.")

    train_mean = float(train_np.mean())
    mean_pred = torch.full_like(y_true, fill_value=train_mean, dtype=torch.float32)
    mean_metrics = _regression_metrics_dict(y_true, mean_pred, loss=float("nan"))
    for metric_name in ("mse", "rmse", "r2", "pearson", "spearman"):
        mean_metrics[f"{metric_name}_mean"] = float(mean_metrics[metric_name])
        mean_metrics[f"{metric_name}_std"] = 0.0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm_trials: list[dict[str, float]] = []
    if y_true.numel() == 0:
        perm_summary = {
            f"{name}_{suffix}": 0.0
            for name in ("mse", "rmse", "r2", "pearson", "spearman")
            for suffix in ("mean", "std")
        }
    else:
        for _ in range(int(trials)):
            perm = torch.randperm(y_true.numel(), generator=generator)
            shuffled = y_true[perm]
            perm_trials.append(_regression_metrics_dict(y_true, shuffled, loss=float("nan")))
        perm_summary = _summarize_regression_trials(perm_trials)

    return {
        "train_mean": mean_metrics,
        "permutation": perm_summary,
    }


def _regression_report(
    regressor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_train: torch.Tensor,
    baseline_trials: int,
    baseline_seed: int,
) -> dict[str, Any]:
    loss_fn = nn.MSELoss()
    loss, y_true, y_pred = _predict_regressor(regressor, loader, device, loss_fn)
    report = _regression_metrics_dict(y_true, y_pred, loss)
    report["overall"] = dict(report)
    report["random_baselines"] = _regression_baselines(
        y_train=y_train,
        y_true=y_true,
        trials=baseline_trials,
        seed=baseline_seed,
    )
    return report


def _save_confusion_matrix_png(
    report: dict[str, Any],
    class_names: list[str],
    path: Path,
    *,
    title: str = "Row-normalized confusion matrix",
) -> None:
    import matplotlib.pyplot as plt

    cm_data = report["confusion_matrix"]
    cm_norm = _matrix_from_named(cm_data["normalized_true"], class_names)
    cm_raw = _matrix_from_named(cm_data["raw"], class_names).to(dtype=torch.long)
    fig_width = max(7.0, min(22.0, 0.75 * len(class_names)))
    fig_height = max(6.0, min(20.0, 0.65 * len(class_names)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    text_size = 8 if len(class_names) <= 12 else 6
    for true_idx in range(cm_norm.size(0)):
        for pred_idx in range(cm_norm.size(1)):
            norm_value = float(cm_norm[true_idx, pred_idx].item())
            raw_value = int(cm_raw[true_idx, pred_idx].item())
            if raw_value == 0:
                label = "0"
            else:
                label = f"{norm_value:.2f}\n({raw_value})"
            text_color = "white" if norm_value >= 0.5 else "black"
            ax.text(
                pred_idx,
                true_idx,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=text_size,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_named_confusion_matrix_png(
    confusion_matrix: dict[str, Any],
    class_names: list[str],
    path: Path,
    *,
    title: str,
) -> None:
    _save_confusion_matrix_png({"confusion_matrix": confusion_matrix}, class_names, path, title=title)


def _save_multilabel_roc_curve_png(report: dict[str, Any], path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    curves = report["roc_curves"]["per_class"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for class_name, curve in sorted(curves.items()):
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=1.6,
            label=f"{class_name} (AUC = {curve['auroc']:.3f})",
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.6)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_wrong_ratio_png(report: dict[str, Any], class_names: list[str], split: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    true_ratios = report["wrong_ratios"]["ratio_true_among_wrongs"]
    pred_ratios = report["wrong_ratios"]["ratio_pred_among_wrongs"]
    true_values = [float(true_ratios[class_name]) for class_name in class_names]
    pred_values = [float(pred_ratios[class_name]) for class_name in class_names]
    x = np.arange(len(class_names))
    width = 0.38
    fig_width = max(7.0, min(16.0, 0.8 * len(class_names)))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    true_bars = ax.bar(x - (width / 2.0), true_values, width, label="True class", color="#4C78A8")
    pred_bars = ax.bar(x + (width / 2.0), pred_values, width, label="Predicted class", color="#F58518")
    ax.set_title(f"Class ratios among wrong predictions ({split})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Ratio among wrong predictions")
    max_value = max(true_values + pred_values, default=0.0)
    ax.set_ylim(0.0, max(1.0, max_value * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False)

    for bars, values in ((true_bars, true_values), (pred_bars, pred_values)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_histogram(
    values: pd.Series,
    bins: int,
    path: Path,
    *,
    title: str = "Compound molecular weight distribution",
    xlabel: str = "Molecular weight",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(values.to_numpy(dtype=np.float64), bins=int(bins), color="#4C78A8", edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_grouped_property_boxplot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    path: Path,
    *,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    grouped = []
    labels = []
    for label, sub_df in sorted(df.groupby(group_col), key=lambda item: str(item[0])):
        values = pd.to_numeric(sub_df[value_col], errors="coerce").dropna()
        if values.empty:
            continue
        grouped.append(values.to_numpy(dtype=np.float64))
        labels.append(str(label))

    if not grouped:
        return

    fig_width = max(8.0, min(18.0, 0.9 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    ax.boxplot(grouped, labels=labels, patch_artist=True)
    ax.set_title(title)
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _explode_bgc_classes(df: pd.DataFrame) -> pd.DataFrame:
    if "bgc_classes" not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["bgc_class_single"])
    out = df.copy()
    out["bgc_class_single"] = out["bgc_classes"].fillna("").astype(str).map(
        lambda text: [label.strip() for label in text.split(";") if label.strip()]
    )
    out = out.explode("bgc_class_single")
    out = out.dropna(subset=["bgc_class_single"])
    out = out[out["bgc_class_single"].astype(str).str.strip() != ""].copy()
    return out


def _origin_type_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats = {
        "row_counts_by_origin_type": {},
        "bgc_counts_by_origin_type": {},
        "compound_counts_by_origin_type": {},
    }
    if df.empty:
        return stats

    row_counts = df["origin_type"].value_counts(dropna=False).to_dict()
    stats["row_counts_by_origin_type"] = {str(key): int(value) for key, value in row_counts.items()}

    bgc_counts = (
        df.drop_duplicates(subset=["bgc_id", "origin_type"])
        .groupby("origin_type")["bgc_id"]
        .nunique()
        .to_dict()
    )
    stats["bgc_counts_by_origin_type"] = {str(key): int(value) for key, value in bgc_counts.items()}

    compound_counts = (
        df.drop_duplicates(subset=["compound_id", "origin_type"])
        .groupby("origin_type")["compound_id"]
        .nunique()
        .to_dict()
    )
    stats["compound_counts_by_origin_type"] = {str(key): int(value) for key, value in compound_counts.items()}
    return stats


def _build_bgc_multilabel_features(
    bgc_df: pd.DataFrame,
    model: DualEncoderCLIP,
    bgc_cache: dict[str, torch.Tensor],
    label_to_idx: dict[str, int],
    device: torch.device,
    batch_size: int,
    protein_position_cache: dict[str, torch.Tensor] | None = None,
    domain_position_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    if bgc_df.empty:
        emb_dim = int(model.bgc_proj.net[-1].out_features)
        return (
            torch.empty((0, emb_dim), dtype=torch.float32),
            torch.empty((0, len(label_to_idx)), dtype=torch.float32),
        )

    with torch.no_grad():
        for start in range(0, len(bgc_df), batch_size):
            chunk = bgc_df.iloc[start : start + batch_size]
            bgc_features, padding_mask = _pad_bgc_features(
                [bgc_cache[str(bgc_id)] for bgc_id in chunk["bgc_id"].tolist()], device
            )
            position_args: dict[str, torch.Tensor] = {}
            for name, cache in (
                ("protein_positions", protein_position_cache),
                ("domain_positions", domain_position_cache),
            ):
                if cache is not None:
                    values = [cache[str(bgc_id)].long() for bgc_id in chunk["bgc_id"].tolist()]
                    if padding_mask is None:
                        position_args[name] = torch.stack(values).to(device)
                    else:
                        position_args[name] = torch.nn.utils.rnn.pad_sequence(
                            values, batch_first=True, padding_value=0
                        ).to(device)
                        if position_args[name].shape != padding_mask.shape:
                            raise ValueError(
                                f"{name} shape {tuple(position_args[name].shape)} does not match "
                                f"BGC padding mask {tuple(padding_mask.shape)}"
                            )
            z_bgc = model.encode_bgc(bgc_features, padding_mask=padding_mask, **position_args)
            y = torch.zeros((len(chunk), len(label_to_idx)), dtype=torch.float32)
            for row_idx, labels_for_bgc in enumerate(chunk["bgc_class_list"].tolist()):
                for label in labels_for_bgc:
                    y[row_idx, label_to_idx[str(label)]] = 1.0
            features.append(z_bgc.cpu())
            labels.append(y)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _attach_split_column(
    df: pd.DataFrame,
    splits_path: str | Path | None,
    cv_fold: int | None,
    val_fold: int | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "split" in out.columns:
        out["split"] = out["split"].astype(str).str.lower()
        return out
    if splits_path is None:
        raise ValueError("A split column or a split assignment TSV is required for compound downstream tasks.")
    split_df = pd.read_csv(splits_path, sep="\t")
    if "split" not in split_df.columns and "strict_split" in split_df.columns:
        split_df["split"] = split_df["strict_split"]
    if "fold_id" not in split_df.columns and "strict_cv10_fold" in split_df.columns:
        split_df["fold_id"] = pd.to_numeric(split_df["strict_cv10_fold"], errors="coerce") + 1
    split_df["bgc_id"] = split_df["bgc_id"].astype(str)
    merge_cols = ["bgc_id"]
    if "compound_id" in split_df.columns:
        split_df["compound_id"] = split_df["compound_id"].astype(str)
        merge_cols.append("compound_id")
    if "split" in split_df.columns:
        split_df["split"] = split_df["split"].astype(str).str.lower()
    elif "fold_id" in split_df.columns:
        if cv_fold is None:
            raise ValueError("cv_fold is required when the split file contains fold_id instead of split.")
        split_df["fold_id"] = pd.to_numeric(split_df["fold_id"], errors="coerce")
        if bool(split_df["fold_id"].isna().any()):
            raise ValueError(f"Invalid fold_id values found in {splits_path}")
        fold_ids = split_df["fold_id"].astype(int)
        split_df["split"] = np.where(
            fold_ids == int(cv_fold),
            "test",
            np.where(fold_ids == int(val_fold), "val", "train") if val_fold is not None else "train",
        )
    else:
        raise ValueError(f"Split file must contain either split or fold_id columns: {splits_path}")
    out["bgc_id"] = out["bgc_id"].astype(str)
    if "compound_id" in merge_cols and "compound_id" not in out.columns:
        compound_id_col = _infer_compound_id_column(out)
        out["compound_id"] = out[compound_id_col].astype(str)
    if "compound_id" in merge_cols:
        out["compound_id"] = out["compound_id"].astype(str)
    keep_cols = merge_cols + ["split"]
    split_df = split_df[keep_cols].drop_duplicates(subset=merge_cols)
    out = out.merge(split_df, on=merge_cols, how="left")
    return out


def _infer_compound_id_column(df: pd.DataFrame) -> str:
    if "compound_id" in df.columns:
        return "compound_id"
    if "canonical_smiles" in df.columns:
        return "canonical_smiles"
    if "smiles" in df.columns:
        return "smiles"
    raise ValueError("Could not infer compound identifier column from the MIBiG pairs table.")


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required for compound_mw, compound_logp, compound_tpsa, and origin_type downstream tasks. "
            "Please install rdkit, for example with `conda install -c conda-forge rdkit`."
        ) from exc
    return Chem


def _safe_canonical_smiles(smiles: str | float | None, chem_module: Any) -> str | None:
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = chem_module.MolFromSmiles(text)
    if mol is None:
        return None
    return str(chem_module.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _safe_inchikey(smiles: str | float | None, chem_module: Any) -> str | None:
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = chem_module.MolFromSmiles(text)
    if mol is None:
        return None
    return str(chem_module.MolToInchiKey(mol))


def _safe_molecular_properties(smiles: str | float | None, chem_module: Any) -> dict[str, float | None]:
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return {"compound_rdkit_mw": None, "compound_logp": None, "compound_tpsa": None}
    text = str(smiles).strip()
    if not text:
        return {"compound_rdkit_mw": None, "compound_logp": None, "compound_tpsa": None}
    mol = chem_module.MolFromSmiles(text)
    if mol is None:
        return {"compound_rdkit_mw": None, "compound_logp": None, "compound_tpsa": None}
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    return {
        "compound_rdkit_mw": float(Descriptors.MolWt(mol)),
        "compound_logp": float(Crippen.MolLogP(mol)),
        "compound_tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
    }


def _first_unique_rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    valid = df.dropna(subset=[key]).copy()
    valid[key] = valid[key].astype(str).str.strip()
    valid = valid[valid[key] != ""].copy()
    return valid.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)


def _prepare_compound_match_table(
    mibig_pairs_path: str | Path,
    npatlas_path: str | Path,
    splits_path: str | Path | None,
    cv_fold: int | None,
    val_fold: int | None,
    output_path: Path,
    force_rebuild: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if output_path.exists() and not force_rebuild:
        matched_df = pd.read_csv(output_path, sep="\t")
        required_property_cols = {"compound_rdkit_mw", "compound_logp", "compound_tpsa"}
        if not required_property_cols.issubset(matched_df.columns) and "smiles" in matched_df.columns:
            chem = _require_rdkit()
            properties = [_safe_molecular_properties(smiles, chem) for smiles in matched_df["smiles"].tolist()]
            for col in sorted(required_property_cols):
                if col not in matched_df.columns:
                    matched_df[col] = [item[col] for item in properties]
            matched_df.to_csv(output_path, sep="\t", index=False)
        total_mibig_rows = int(pd.read_csv(mibig_pairs_path, sep="\t", usecols=["bgc_id"]).shape[0])
        stats = {
            "total_mibig_rows": total_mibig_rows,
            "matched_by_inchikey": int((matched_df["match_method"] == "inchikey").sum()) if "match_method" in matched_df else 0,
            "matched_by_smiles": int((matched_df["match_method"] == "smiles").sum()) if "match_method" in matched_df else 0,
            "total_matched_rows": int(len(matched_df)),
        }
        return matched_df, stats

    chem = _require_rdkit()
    mibig_df = pd.read_csv(mibig_pairs_path, sep="\t")
    mibig_df = _attach_split_column(mibig_df, splits_path, cv_fold=cv_fold, val_fold=val_fold)
    compound_id_col = _infer_compound_id_column(mibig_df)
    mibig_df["bgc_id"] = mibig_df["bgc_id"].astype(str)
    mibig_df["compound_id"] = mibig_df[compound_id_col].astype(str)
    if "smiles" not in mibig_df.columns:
        mibig_df["smiles"] = mibig_df["compound_id"]
    mibig_df["smiles"] = mibig_df["smiles"].astype(str)
    mibig_df = mibig_df.dropna(subset=["bgc_id", "compound_id", "smiles", "split"]).reset_index(drop=True)

    npatlas_df = pd.read_csv(npatlas_path, sep="\t")
    if "compound_smiles" not in npatlas_df.columns or "compound_inchikey" not in npatlas_df.columns:
        raise ValueError("NPAtlas table must include compound_smiles and compound_inchikey columns.")

    npatlas_df = npatlas_df.copy()
    npatlas_df["compound_inchikey"] = npatlas_df["compound_inchikey"].astype(str).str.strip()
    npatlas_df["canonical_smiles"] = [
        _safe_canonical_smiles(smiles, chem) for smiles in npatlas_df["compound_smiles"].tolist()
    ]

    npatlas_inchikey = _first_unique_rows(npatlas_df, "compound_inchikey").set_index("compound_inchikey")
    npatlas_smiles = _first_unique_rows(npatlas_df, "canonical_smiles").set_index("canonical_smiles")

    mibig_df["mibig_inchikey"] = [_safe_inchikey(smiles, chem) for smiles in mibig_df["smiles"].tolist()]
    mibig_df["mibig_canonical_smiles"] = [_safe_canonical_smiles(smiles, chem) for smiles in mibig_df["smiles"].tolist()]

    matched_records: list[dict[str, Any]] = []
    matched_by_inchikey = 0
    matched_by_smiles = 0

    for row in mibig_df.itertuples(index=False):
        npatlas_row = None
        match_method = None
        match_key = None

        mibig_inchikey = getattr(row, "mibig_inchikey")
        if mibig_inchikey and mibig_inchikey in npatlas_inchikey.index:
            npatlas_row = npatlas_inchikey.loc[mibig_inchikey]
            match_method = "inchikey"
            match_key = mibig_inchikey
            matched_by_inchikey += 1
        else:
            mibig_smiles = getattr(row, "mibig_canonical_smiles")
            if mibig_smiles and mibig_smiles in npatlas_smiles.index:
                npatlas_row = npatlas_smiles.loc[mibig_smiles]
                match_method = "smiles"
                match_key = mibig_smiles
                matched_by_smiles += 1

        if npatlas_row is None:
            continue
        properties = _safe_molecular_properties(str(row.smiles), chem)

        matched_records.append(
            {
                "bgc_id": str(row.bgc_id),
                "compound_id": str(row.compound_id),
                "split": str(row.split).lower(),
                "compound_name": getattr(row, "compound_name", None),
                "bgc_classes": getattr(row, "bgc_classes", None),
                "n_bgc_classes": getattr(row, "n_bgc_classes", None),
                "smiles": str(row.smiles),
                "mibig_inchikey": mibig_inchikey,
                "mibig_canonical_smiles": getattr(row, "mibig_canonical_smiles"),
                "match_method": match_method,
                "match_key": match_key,
                "npatlas_compound_name": npatlas_row.get("compound_name"),
                "npatlas_compound_smiles": npatlas_row.get("compound_smiles"),
                "npatlas_compound_inchikey": npatlas_row.get("compound_inchikey"),
                "compound_molecular_weight": npatlas_row.get("compound_molecular_weight"),
                "compound_rdkit_mw": properties["compound_rdkit_mw"],
                "compound_logp": properties["compound_logp"],
                "compound_tpsa": properties["compound_tpsa"],
                "origin_type": npatlas_row.get("origin_type"),
            }
        )

    matched_df = pd.DataFrame(matched_records)
    if not matched_df.empty:
        matched_df.to_csv(output_path, sep="\t", index=False)

    stats = {
        "total_mibig_rows": int(len(mibig_df)),
        "matched_by_inchikey": int(matched_by_inchikey),
        "matched_by_smiles": int(matched_by_smiles),
        "total_matched_rows": int(len(matched_df)),
    }
    return matched_df, stats


def _build_compound_embedding_map(
    df: pd.DataFrame,
    model: DualEncoderCLIP,
    compound_cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    unique_ids = sorted({str(compound_id) for compound_id in df["compound_id"].tolist()})
    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(unique_ids), batch_size):
            batch_ids = unique_ids[start : start + batch_size]
            compound_features = torch.stack([compound_cache[compound_id] for compound_id in batch_ids]).to(device)
            z_compound = model.encode_compound(compound_features).cpu()
            for compound_id, embedding in zip(batch_ids, z_compound, strict=True):
                embeddings[compound_id] = embedding
    return embeddings


def _frame_to_tensor_dataset(
    df: pd.DataFrame,
    embedding_map: dict[str, torch.Tensor],
    label_column: str,
    label_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if df.empty:
        if not embedding_map:
            raise ValueError("Cannot infer embedding dimension from an empty embedding map.")
        emb_dim = int(next(iter(embedding_map.values())).numel())
        return torch.empty((0, emb_dim), dtype=torch.float32), torch.empty(0, dtype=label_dtype)

    features = torch.stack([embedding_map[str(compound_id)] for compound_id in df["compound_id"].tolist()])
    if label_dtype == torch.long:
        labels = torch.tensor(df[label_column].tolist(), dtype=label_dtype)
    else:
        labels = torch.tensor(df[label_column].tolist(), dtype=torch.float32)
    return features, labels


def _log_split_sizes(task_name: str, split_frames: dict[str, pd.DataFrame]) -> None:
    LOGGER.info(
        "%s dataset sizes: train=%d val=%d test=%d",
        task_name,
        len(split_frames["train"]),
        len(split_frames["val"]),
        len(split_frames["test"]),
    )


def _build_bgc_split_assignments(
    data_dir: str | Path,
    splits_path: str | Path | None,
    cv_fold: int | None,
    val_fold: int | None,
) -> pd.DataFrame:
    interactions = build_interactions(data_dir, splits_path=splits_path, cv_fold=cv_fold, val_fold=val_fold)
    assignments = interactions[["bgc_id", "split"]].drop_duplicates().copy()
    assignments["bgc_id"] = assignments["bgc_id"].astype(str)
    assignments["split"] = assignments["split"].astype(str).str.lower()
    return assignments


def _load_bioactivity_class_table(
    data_dir: str | Path,
    bioactivity_table_path: str | Path,
    splits_path: str | Path | None,
    cv_fold: int | None,
    val_fold: int | None,
) -> pd.DataFrame:
    table_path = Path(bioactivity_table_path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Bioactivity label table not found: {table_path}. "
            "Create it with `python -m projects.mibig_bgc_np.scripts.create_eda_artifacts`."
        )

    bio_df = pd.read_csv(table_path)
    required = {"bgc_id", "observed_bioactivities"}
    missing = required.difference(bio_df.columns)
    if missing:
        raise ValueError(f"Bioactivity table {table_path} is missing columns: {sorted(missing)}")

    bio_df = bio_df[["bgc_id", "observed_bioactivities"]].copy()
    bio_df["bgc_id"] = bio_df["bgc_id"].astype(str)
    bio_df["bgc_class_list"] = bio_df["observed_bioactivities"].fillna("").astype(str).map(
        lambda text: [
            label
            for label in BIOACTIVITY_CLASS_NAMES
            if label in {item.strip() for item in text.split(";") if item.strip()}
        ]
    )
    bio_df = bio_df[bio_df["bgc_class_list"].map(len) > 0].copy()

    assignments = _build_bgc_split_assignments(data_dir, splits_path=splits_path, cv_fold=cv_fold, val_fold=val_fold)
    bio_df = bio_df.merge(assignments, on="bgc_id", how="inner")
    return bio_df[["bgc_id", "split", "bgc_class_list"]].drop_duplicates(subset=["bgc_id", "split"]).reset_index(drop=True)


def _bioactivity_dataset_stats(bio_df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "classes": BIOACTIVITY_CLASS_NAMES,
        "n_rows": int(len(bio_df)),
        "n_bgcs": int(bio_df["bgc_id"].nunique()) if not bio_df.empty else 0,
        "split_rows": {},
        "split_bgcs": {},
        "class_bgcs": {},
    }
    for split, split_df in bio_df.groupby("split"):
        stats["split_rows"][str(split)] = int(len(split_df))
        stats["split_bgcs"][str(split)] = int(split_df["bgc_id"].nunique())
    for class_name in BIOACTIVITY_CLASS_NAMES:
        stats["class_bgcs"][class_name] = int(
            bio_df[bio_df["bgc_class_list"].map(lambda labels, name=class_name: name in labels)]["bgc_id"].nunique()
        )
    return stats


def _labels_from_semicolon_text(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [label.strip() for label in str(value).split(";") if label.strip()]


def _load_npclassifier_allowed_labels(
    labels_df: pd.DataFrame,
    *,
    label_column: str,
    counts_path: str | Path | None,
    min_count: int | None,
) -> list[str]:
    if counts_path is None:
        return sorted({label for value in labels_df[label_column].tolist() for label in _labels_from_semicolon_text(value)})

    path = Path(counts_path)
    if not path.exists():
        raise FileNotFoundError(
            f"NPClassifier count table not found: {path}. "
            "Create it with `python -m projects.mibig_bgc_np.scripts.create_downstream_distributions`."
        )
    counts_df = pd.read_csv(path)
    required = {"label", "n_compounds"}
    missing = required.difference(counts_df.columns)
    if missing:
        raise ValueError(f"NPClassifier count table {path} is missing columns: {sorted(missing)}")
    threshold = int(min_count) if min_count is not None else 0
    filtered = counts_df[pd.to_numeric(counts_df["n_compounds"], errors="coerce") > threshold].copy()
    return [str(label) for label in filtered["label"].tolist()]


def _load_npclassifier_bgc_label_table(
    data_dir: str | Path,
    npclassifier_pair_labels_path: str | Path,
    *,
    task_name: str,
    splits_path: str | Path | None,
    cv_fold: int | None,
    val_fold: int | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if task_name not in NPCLASSIFIER_TASKS:
        raise ValueError(f"Unsupported NPClassifier downstream task: {task_name}")
    spec = NPCLASSIFIER_TASKS[task_name]
    label_column = str(spec["label_column"])
    table_path = Path(npclassifier_pair_labels_path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"NPClassifier pair label table not found: {table_path}. "
            "Create it with `python -m projects.mibig_bgc_np.scripts.run_npclassifier`."
        )

    labels_df = pd.read_csv(table_path, sep="\t")
    required = {"bgc_id", label_column}
    missing = required.difference(labels_df.columns)
    if missing:
        raise ValueError(f"NPClassifier table {table_path} is missing columns: {sorted(missing)}")
    labels_df = labels_df.copy()
    labels_df["bgc_id"] = labels_df["bgc_id"].astype(str)

    allowed_labels = _load_npclassifier_allowed_labels(
        labels_df,
        label_column=label_column,
        counts_path=spec["counts_path"],
        min_count=spec["min_count"],
    )
    if not allowed_labels:
        raise ValueError(f"No labels selected for {task_name}. Check the NPClassifier count threshold.")
    allowed_set = set(allowed_labels)

    if splits_path is None and "split" not in labels_df.columns:
        assignments = build_interactions(data_dir)[["bgc_id", "compound_id", "split"]].drop_duplicates().copy()
        assignments["bgc_id"] = assignments["bgc_id"].astype(str)
        assignments["compound_id"] = assignments["compound_id"].astype(str)
        assignments["split"] = assignments["split"].astype(str).str.lower()
        if "compound_id" not in labels_df.columns:
            compound_id_col = _infer_compound_id_column(labels_df)
            labels_df["compound_id"] = labels_df[compound_id_col].astype(str)
        labels_df["compound_id"] = labels_df["compound_id"].astype(str)
        labels_df = labels_df.merge(assignments, on=["bgc_id", "compound_id"], how="left")
    else:
        labels_df = _attach_split_column(labels_df, splits_path=splits_path, cv_fold=cv_fold, val_fold=val_fold)
    labels_df = labels_df.dropna(subset=["split"]).copy()
    labels_df["split"] = labels_df["split"].astype(str).str.lower()
    labels_df["label_list"] = labels_df[label_column].map(
        lambda value: [label for label in _labels_from_semicolon_text(value) if label in allowed_set]
    )
    labels_df = labels_df[labels_df["label_list"].map(len) > 0].copy()

    rows: list[dict[str, Any]] = []
    for (bgc_id, split), group in labels_df.groupby(["bgc_id", "split"], sort=True):
        labels = sorted(
            {label for labels_for_row in group["label_list"].tolist() for label in labels_for_row},
            key=lambda label: allowed_labels.index(label),
        )
        if labels:
            rows.append({"bgc_id": str(bgc_id), "split": str(split), "bgc_class_list": labels})
    out = pd.DataFrame(rows, columns=["bgc_id", "split", "bgc_class_list"])

    stats = {
        "task": task_name,
        "level": spec["level"],
        "label_column": label_column,
        "npclassifier_pair_labels_path": str(table_path),
        "counts_path": str(spec["counts_path"]) if spec["counts_path"] is not None else None,
        "min_count_exclusive": spec["min_count"],
        "selected_labels": allowed_labels,
        "n_selected_labels": int(len(allowed_labels)),
        "n_pair_label_rows": int(len(labels_df)),
        "n_rows": int(len(out)),
        "n_bgcs": int(out["bgc_id"].nunique()) if not out.empty else 0,
        "split_rows": {},
        "split_bgcs": {},
        "label_bgcs": {},
    }
    for split, split_df in out.groupby("split"):
        stats["split_rows"][str(split)] = int(len(split_df))
        stats["split_bgcs"][str(split)] = int(split_df["bgc_id"].nunique())
    for label in allowed_labels:
        stats["label_bgcs"][label] = int(
            out[out["bgc_class_list"].map(lambda labels, name=label: name in labels)]["bgc_id"].nunique()
        )

    return out.reset_index(drop=True), allowed_labels, stats


def _train_bgc_class_task(
    data_dir: str | Path,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    contrastive_model: DualEncoderCLIP,
    *,
    splits_path: str | Path | None,
    cv_fold: int | None,
    val_fold: int | None,
    baseline_trials: int,
    class_names: list[str] | None,
    save_cm_png: bool,
    output_dir: Path,
) -> dict[str, Any]:
    bgc_df = build_bgc_class_table(data_dir, splits_path=splits_path, cv_fold=cv_fold, val_fold=val_fold)
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    protein_path = Path(cache_dir) / "protein_positions.pt"
    domain_path = Path(cache_dir) / "domain_positions.pt"
    protein_positions = torch.load(protein_path, map_location="cpu", weights_only=True) if protein_path.exists() else None
    domain_positions = torch.load(domain_path, map_location="cpu", weights_only=True) if domain_path.exists() else None

    split_frames = {
        split: bgc_df[bgc_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    _log_split_sizes("bgc_class", split_frames)

    train_labels = sorted(
        {
            str(label)
            for labels_for_bgc in split_frames["train"]["bgc_class_list"].tolist()
            for label in labels_for_bgc
        }
    )
    label_vocab = train_labels
    if not label_vocab:
        raise ValueError("Training split does not contain any BGC classes for downstream training.")
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
    output_class_names = class_names if class_names is not None else [str(label) for label in label_vocab]
    if len(output_class_names) != len(label_vocab):
        raise ValueError("class_names length must match the number of training classes.")
    for split in ("val", "test"):
        unknown = sorted(
            {
                str(label)
                for labels_for_bgc in split_frames[split]["bgc_class_list"].tolist()
                for label in labels_for_bgc
                if str(label) not in label_vocab
            }
        )
        if unknown:
            missing = ", ".join(unknown)
            raise ValueError(f"Split '{split}' contains labels absent from train: {missing}")

    x_train, y_train = _build_bgc_multilabel_features(
        split_frames["train"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )
    x_val, y_val = _build_bgc_multilabel_features(
        split_frames["val"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )
    x_test, y_test = _build_bgc_multilabel_features(
        split_frames["test"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )

    classifier = BGCClassifier(
        emb_dim=int(cfg["model"]["emb_dim"]),
        num_classes=len(label_vocab),
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        classifier.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    pos_counts = y_train.sum(dim=0)
    neg_counts = y_train.size(0) - pos_counts
    pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts.clamp_min(1.0), torch.ones_like(pos_counts))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)

    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc="Training bgc_class", leave=False):
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    metrics: dict[str, Any] = {
        "label_vocab": label_vocab,
        "class_names": output_class_names,
        "loss": {
            "name": "BCEWithLogitsLoss",
            "pos_weight": [float(value) for value in pos_weight.cpu().tolist()],
        },
    }
    eval_loaders = {"val": val_loader, "test": test_loader}
    for split_name, loader in eval_loaders.items():
        if len(loader.dataset) == 0:
            continue
        metrics[split_name] = _multilabel_classification_report(
            classifier=classifier,
            loader=loader,
            device=device,
            class_names=output_class_names,
        )

    if save_cm_png:
        for split_name in ("val", "test"):
            if split_name not in metrics:
                continue
            split_report = metrics[split_name]
            _save_named_confusion_matrix_png(
                split_report["confusion_matrix"],
                output_class_names,
                output_dir / f"downstream_confusion_matrix_{split_name}_all_bgcs.png",
                title=f"Expanded-label confusion matrix ({split_name}, all BGCs)",
            )
            _save_named_confusion_matrix_png(
                split_report["confusion_matrix_single_class_only"],
                output_class_names,
                output_dir / f"downstream_confusion_matrix_{split_name}_single_class_bgcs.png",
                title=f"Expanded-label confusion matrix ({split_name}, single-class BGCs only)",
            )
            _save_multilabel_roc_curve_png(
                split_report,
                output_dir / f"downstream_roc_curve_{split_name}.png",
                title=f"ROC Curve for BGC Class Prediction ({split_name})",
            )
            for class_name in output_class_names:
                _save_named_confusion_matrix_png(
                    split_report["per_class_binary"][class_name]["confusion_matrix"],
                    ["negative", "positive"],
                    output_dir / f"downstream_confusion_matrix_{split_name}_class_{_slugify_label(class_name)}.png",
                    title=f"One-vs-rest confusion matrix ({split_name}, class={class_name})",
                )

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "metrics": metrics,
            "label_vocab": label_vocab,
        },
        output_dir / "downstream_classifier.pt",
    )
    save_json(metrics, output_dir / "downstream_metrics.json")
    return metrics


def _train_bioactivity_class_task(
    data_dir: str | Path,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    contrastive_model: DualEncoderCLIP,
    *,
    splits_path: str | Path | None,
    cv_fold: int | None,
    val_fold: int | None,
    save_cm_png: bool,
    output_dir: Path,
    bioactivity_table_path: str | Path,
) -> dict[str, Any]:
    bio_df = _load_bioactivity_class_table(
        data_dir=data_dir,
        bioactivity_table_path=bioactivity_table_path,
        splits_path=splits_path,
        cv_fold=cv_fold,
        val_fold=val_fold,
    )
    if bio_df.empty:
        raise ValueError("No BGCs with the selected observed bioactivity classes were found in the selected splits.")

    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    protein_path = Path(cache_dir) / "protein_positions.pt"
    domain_path = Path(cache_dir) / "domain_positions.pt"
    protein_positions = torch.load(protein_path, map_location="cpu", weights_only=True) if protein_path.exists() else None
    domain_positions = torch.load(domain_path, map_location="cpu", weights_only=True) if domain_path.exists() else None
    split_frames = {
        split: bio_df[bio_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    _log_split_sizes("bioactivity_class", split_frames)
    if split_frames["train"].empty:
        raise ValueError("Training split is empty for bioactivity_class.")

    label_to_idx = {label: idx for idx, label in enumerate(BIOACTIVITY_CLASS_NAMES)}
    x_train, y_train = _build_bgc_multilabel_features(
        split_frames["train"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )
    x_val, y_val = _build_bgc_multilabel_features(
        split_frames["val"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )
    x_test, y_test = _build_bgc_multilabel_features(
        split_frames["test"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )

    classifier = BGCClassifier(
        emb_dim=int(cfg["model"]["emb_dim"]),
        num_classes=len(BIOACTIVITY_CLASS_NAMES),
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        classifier.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    pos_counts = y_train.sum(dim=0)
    neg_counts = y_train.size(0) - pos_counts
    pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts.clamp_min(1.0), torch.ones_like(pos_counts))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)

    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc="Training bioactivity_class", leave=False):
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    metrics: dict[str, Any] = {
        "label_vocab": BIOACTIVITY_CLASS_NAMES,
        "class_names": BIOACTIVITY_CLASS_NAMES,
        "target_protocol": "observed_true_only_selected_classes",
        "bioactivity_table_path": str(bioactivity_table_path),
        "dataset_stats": _bioactivity_dataset_stats(bio_df),
        "loss": {
            "name": "BCEWithLogitsLoss",
            "pos_weight": [float(value) for value in pos_weight.cpu().tolist()],
        },
    }
    for split_name, loader in {"val": val_loader, "test": test_loader}.items():
        if len(loader.dataset) == 0:
            continue
        metrics[split_name] = _multilabel_classification_report(
            classifier=classifier,
            loader=loader,
            device=device,
            class_names=BIOACTIVITY_CLASS_NAMES,
        )

    if save_cm_png:
        for split_name in ("val", "test"):
            if split_name not in metrics:
                continue
            split_report = metrics[split_name]
            _save_named_confusion_matrix_png(
                split_report["confusion_matrix"],
                BIOACTIVITY_CLASS_NAMES,
                output_dir / f"downstream_bioactivity_confusion_matrix_{split_name}_all_bgcs.png",
                title=f"Expanded-label confusion matrix ({split_name}, observed bioactivity)",
            )
            _save_named_confusion_matrix_png(
                split_report["confusion_matrix_single_class_only"],
                BIOACTIVITY_CLASS_NAMES,
                output_dir / f"downstream_bioactivity_confusion_matrix_{split_name}_single_class_bgcs.png",
                title=f"Expanded-label confusion matrix ({split_name}, single-bioactivity BGCs only)",
            )
            _save_multilabel_roc_curve_png(
                split_report,
                output_dir / f"downstream_bioactivity_roc_curve_{split_name}.png",
                title=f"ROC Curve for Bioactivity Prediction ({split_name})",
            )
            for class_name in BIOACTIVITY_CLASS_NAMES:
                _save_named_confusion_matrix_png(
                    split_report["per_class_binary"][class_name]["confusion_matrix"],
                    ["negative", "positive"],
                    output_dir / f"downstream_bioactivity_confusion_matrix_{split_name}_class_{_slugify_label(class_name)}.png",
                    title=f"One-vs-rest confusion matrix ({split_name}, bioactivity={class_name})",
                )

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "metrics": metrics,
            "label_vocab": BIOACTIVITY_CLASS_NAMES,
        },
        output_dir / "downstream_bioactivity_classifier.pt",
    )
    save_json(metrics, output_dir / "downstream_bioactivity_metrics.json")
    save_json(metrics["dataset_stats"], output_dir / "downstream_bioactivity_dataset_stats.json")
    return metrics


def _train_npclassifier_multilabel_task(
    task_name: str,
    data_dir: str | Path,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    contrastive_model: DualEncoderCLIP,
    *,
    splits_path: str | Path | None,
    cv_fold: int | None,
    val_fold: int | None,
    save_cm_png: bool,
    output_dir: Path,
    npclassifier_pair_labels_path: str | Path,
) -> dict[str, Any]:
    if task_name not in NPCLASSIFIER_TASKS:
        raise ValueError(f"Unsupported NPClassifier downstream task: {task_name}")
    spec = NPCLASSIFIER_TASKS[task_name]
    level = str(spec["level"])
    display_name = str(spec["display_name"])
    target_df, label_vocab, dataset_stats = _load_npclassifier_bgc_label_table(
        data_dir=data_dir,
        npclassifier_pair_labels_path=npclassifier_pair_labels_path,
        task_name=task_name,
        splits_path=splits_path,
        cv_fold=cv_fold,
        val_fold=val_fold,
    )
    if target_df.empty:
        raise ValueError(f"No BGCs with selected {display_name} labels were found in the selected splits.")

    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    protein_path = Path(cache_dir) / "protein_positions.pt"
    domain_path = Path(cache_dir) / "domain_positions.pt"
    protein_positions = torch.load(protein_path, map_location="cpu", weights_only=True) if protein_path.exists() else None
    domain_positions = torch.load(domain_path, map_location="cpu", weights_only=True) if domain_path.exists() else None
    split_frames = {
        split: target_df[target_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    _log_split_sizes(task_name, split_frames)
    if split_frames["train"].empty:
        raise ValueError(f"Training split is empty for {task_name}.")

    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
    x_train, y_train = _build_bgc_multilabel_features(
        split_frames["train"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )
    x_val, y_val = _build_bgc_multilabel_features(
        split_frames["val"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )
    x_test, y_test = _build_bgc_multilabel_features(
        split_frames["test"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
        protein_positions,
        domain_positions,
    )

    classifier = BGCClassifier(
        emb_dim=int(cfg["model"]["emb_dim"]),
        num_classes=len(label_vocab),
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        classifier.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    pos_counts = y_train.sum(dim=0)
    neg_counts = y_train.size(0) - pos_counts
    pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts.clamp_min(1.0), torch.ones_like(pos_counts))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)

    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc=f"Training {task_name}", leave=False):
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    metrics: dict[str, Any] = {
        "label_vocab": label_vocab,
        "class_names": label_vocab,
        "target_protocol": f"npclassifier_{level}_selected_labels",
        "dataset_stats": dataset_stats,
        "loss": {
            "name": "BCEWithLogitsLoss",
            "pos_weight": [float(value) for value in pos_weight.cpu().tolist()],
        },
    }
    for split_name, loader in {"val": val_loader, "test": test_loader}.items():
        if len(loader.dataset) == 0:
            continue
        metrics[split_name] = _multilabel_classification_report(
            classifier=classifier,
            loader=loader,
            device=device,
            class_names=label_vocab,
        )

    file_prefix = f"downstream_npclassifier_{level}"
    if save_cm_png:
        for split_name in ("val", "test"):
            if split_name not in metrics:
                continue
            split_report = metrics[split_name]
            _save_named_confusion_matrix_png(
                split_report["confusion_matrix"],
                label_vocab,
                output_dir / f"{file_prefix}_confusion_matrix_{split_name}_all_bgcs.png",
                title=f"Expanded-label confusion matrix ({split_name}, {display_name})",
            )
            _save_named_confusion_matrix_png(
                split_report["confusion_matrix_single_class_only"],
                label_vocab,
                output_dir / f"{file_prefix}_confusion_matrix_{split_name}_single_class_bgcs.png",
                title=f"Expanded-label confusion matrix ({split_name}, single-label BGCs only)",
            )
            _save_multilabel_roc_curve_png(
                split_report,
                output_dir / f"{file_prefix}_roc_curve_{split_name}.png",
                title=f"ROC Curve for {display_name} Prediction ({split_name})",
            )
            for class_name in label_vocab:
                _save_named_confusion_matrix_png(
                    split_report["per_class_binary"][class_name]["confusion_matrix"],
                    ["negative", "positive"],
                    output_dir / f"{file_prefix}_confusion_matrix_{split_name}_class_{_slugify_label(class_name)}.png",
                    title=f"One-vs-rest confusion matrix ({split_name}, {level}={class_name})",
                )

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "metrics": metrics,
            "label_vocab": label_vocab,
        },
        output_dir / f"{file_prefix}_classifier.pt",
    )
    save_json(metrics, output_dir / f"{file_prefix}_metrics.json")
    save_json(metrics["dataset_stats"], output_dir / f"{file_prefix}_dataset_stats.json")
    return metrics


def _train_compound_property_regression_task(
    task_name: str,
    matched_df: pd.DataFrame,
    embedding_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    baseline_trials: int,
    mw_bins: int,
) -> dict[str, Any]:
    if task_name not in COMPOUND_REGRESSION_TASKS:
        raise ValueError(f"Unsupported compound regression task: {task_name}")
    spec = COMPOUND_REGRESSION_TASKS[task_name]
    target_col = str(spec["target"])
    short_name = str(spec["short_name"])

    task_df = matched_df.dropna(subset=[target_col]).copy()
    task_df[target_col] = pd.to_numeric(task_df[target_col], errors="coerce")
    task_df = task_df.dropna(subset=[target_col]).reset_index(drop=True)
    if task_df.empty:
        raise ValueError(f"No matched compounds with non-null {target_col} were found.")

    _save_histogram(
        task_df[target_col],
        bins=mw_bins,
        path=output_dir / f"downstream_{short_name}_hist.png",
        title=str(spec["hist_title"]),
        xlabel=str(spec["x_label"]),
    )
    exploded_classes = _explode_bgc_classes(task_df)
    _save_grouped_property_boxplot(
        exploded_classes,
        group_col="bgc_class_single",
        value_col=target_col,
        title=f"{spec['display_name']} distribution by BGC class",
        path=output_dir / f"downstream_{short_name}_by_bgc_class.png",
        ylabel=str(spec["boxplot_ylabel"]),
    )
    _save_grouped_property_boxplot(
        task_df.dropna(subset=["origin_type"]).copy(),
        group_col="origin_type",
        value_col=target_col,
        title=f"{spec['display_name']} distribution by origin type",
        path=output_dir / f"downstream_{short_name}_by_origin_type.png",
        ylabel=str(spec["boxplot_ylabel"]),
    )

    split_frames = {
        split: task_df[task_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    _log_split_sizes(task_name, split_frames)
    if split_frames["train"].empty:
        raise ValueError(f"Training split is empty for {task_name}.")

    x_train, y_train = _frame_to_tensor_dataset(split_frames["train"], embedding_map, target_col, torch.float32)
    x_val, y_val = _frame_to_tensor_dataset(split_frames["val"], embedding_map, target_col, torch.float32)
    x_test, y_test = _frame_to_tensor_dataset(split_frames["test"], embedding_map, target_col, torch.float32)

    regressor = EmbeddingRegressor(
        emb_dim=int(cfg["model"]["emb_dim"]),
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        regressor.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)

    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc=f"Training {task_name}", leave=False):
        regressor.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = regressor(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

    baseline_seed = int(cfg.get("seed", 42))
    metrics: dict[str, Any] = {
        "target": target_col,
        "display_name": str(spec["display_name"]),
        "histogram_bins": int(mw_bins),
        "match_counts": {
            "final_dataset_size": int(len(task_df)),
        },
    }
    eval_specs = {
        "val": (val_loader, baseline_seed),
        "test": (test_loader, baseline_seed + 1),
    }
    for split_name, (loader, seed) in eval_specs.items():
        if len(loader.dataset) == 0:
            continue
        metrics[split_name] = _regression_report(
            regressor=regressor,
            loader=loader,
            device=device,
            y_train=y_train,
            baseline_trials=baseline_trials,
            baseline_seed=seed,
        )

    torch.save(
        {
            "regressor_state_dict": regressor.state_dict(),
            "metrics": metrics,
        },
        output_dir / f"downstream_{task_name}_regressor.pt",
    )
    save_json(metrics, output_dir / f"downstream_{task_name}_metrics.json")
    return metrics


def _classification_metrics_from_predictions(y_true: torch.Tensor, y_pred: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    per_class = per_class_prf(cm)
    overall = macro_micro_f1_from_cm(cm)
    fungus_idx = ORIGIN_LABEL_TO_IDX["Fungus"]
    return {
        "accuracy": float(overall["accuracy"]),
        "macro_f1": float(overall["macro_f1"]),
        "precision_positive": float(per_class["precision"][fungus_idx]),
        "recall_positive": float(per_class["recall"][fungus_idx]),
        "f1_positive": float(per_class["f1"][fungus_idx]),
    }


def _summarize_classification_trials(trial_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in ("accuracy", "macro_f1", "precision_positive", "recall_positive", "f1_positive"):
        values = np.asarray([metrics[metric_name] for metrics in trial_metrics], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean()) if values.size else 0.0
        summary[f"{metric_name}_std"] = float(values.std(ddof=0)) if values.size else 0.0
    return summary


def _origin_baselines(
    y_train: torch.Tensor,
    y_true: torch.Tensor,
    trials: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    train_cpu = y_train.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    true_cpu = y_true.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    if train_cpu.numel() == 0:
        raise ValueError("y_train must contain at least one label.")

    train_pos_rate = float((train_cpu == ORIGIN_LABEL_TO_IDX["Fungus"]).to(dtype=torch.float32).mean().item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    majority_class = int(torch.mode(train_cpu).values.item())
    majority_pred = torch.full_like(true_cpu, fill_value=majority_class)
    majority = _classification_metrics_from_predictions(true_cpu, majority_pred, ORIGIN_CLASS_NAMES)
    for metric_name, value in list(majority.items()):
        majority[f"{metric_name}_mean"] = float(value)
        majority[f"{metric_name}_std"] = 0.0

    uniform_trials: list[dict[str, float]] = []
    prior_trials: list[dict[str, float]] = []
    for _ in range(int(trials)):
        uniform_pred = torch.randint(0, 2, true_cpu.shape, generator=generator, dtype=torch.long)
        prior_draws = torch.rand(true_cpu.shape, generator=generator)
        prior_pred = (prior_draws < train_pos_rate).to(dtype=torch.long)
        uniform_trials.append(_classification_metrics_from_predictions(true_cpu, uniform_pred, ORIGIN_CLASS_NAMES))
        prior_trials.append(_classification_metrics_from_predictions(true_cpu, prior_pred, ORIGIN_CLASS_NAMES))

    return {
        "majority": majority,
        "uniform": _summarize_classification_trials(uniform_trials),
        "prior": {
            **_summarize_classification_trials(prior_trials),
            "positive_rate_train": train_pos_rate,
        },
    }


def _train_origin_type_task(
    matched_df: pd.DataFrame,
    embedding_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    baseline_trials: int,
    save_cm_png: bool,
) -> dict[str, Any]:
    task_df = matched_df[matched_df["origin_type"].isin(ORIGIN_LABEL_TO_IDX.keys())].copy()
    if task_df.empty:
        raise ValueError("No matched compounds with origin_type in {'Fungus', 'Bacterium'} were found.")
    task_df["origin_label"] = task_df["origin_type"].map(ORIGIN_LABEL_TO_IDX)
    dataset_stats = _origin_type_dataset_stats(task_df)

    split_frames = {
        split: task_df[task_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    _log_split_sizes("origin_type", split_frames)
    if split_frames["train"].empty:
        raise ValueError("Training split is empty for origin_type.")

    x_train, y_train = _frame_to_tensor_dataset(split_frames["train"], embedding_map, "origin_label", torch.long)
    x_val, y_val = _frame_to_tensor_dataset(split_frames["val"], embedding_map, "origin_label", torch.long)
    x_test, y_test = _frame_to_tensor_dataset(split_frames["test"], embedding_map, "origin_label", torch.long)

    classifier = BGCClassifier(
        emb_dim=int(cfg["model"]["emb_dim"]),
        num_classes=2,
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        classifier.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)

    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc="Training origin_type", leave=False):
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    baseline_seed = int(cfg.get("seed", 42))
    metrics: dict[str, Any] = {
        "target": "origin_type",
        "label_mapping": ORIGIN_LABEL_TO_IDX,
        "positive_label": "Fungus",
        "class_names": ORIGIN_CLASS_NAMES,
        "dataset_stats": dataset_stats,
        "match_counts": {
            "final_dataset_size": int(len(task_df)),
        },
    }
    eval_specs = {
        "val": (val_loader, y_val, baseline_seed),
        "test": (test_loader, y_test, baseline_seed + 1),
    }
    for split_name, (loader, y_split, seed) in eval_specs.items():
        if len(loader.dataset) == 0:
            continue
        metrics[split_name] = _classification_report(
            classifier=classifier,
            loader=loader,
            device=device,
            y_train=y_train,
            num_classes=2,
            class_names=ORIGIN_CLASS_NAMES,
            baseline_trials=baseline_trials,
            baseline_seed=seed,
        )
        metrics[split_name]["random_baselines"] = _origin_baselines(y_train, y_split, baseline_trials, seed)

    if save_cm_png:
        for split_name in ("val", "test"):
            if split_name not in metrics:
                continue
            _save_confusion_matrix_png(
                metrics[split_name],
                ORIGIN_CLASS_NAMES,
                output_dir / f"downstream_origin_type_confusion_matrix_{split_name}.png",
            )

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "metrics": metrics,
            "label_mapping": ORIGIN_LABEL_TO_IDX,
        },
        output_dir / "downstream_origin_type_classifier.pt",
    )
    save_json(dataset_stats, output_dir / "downstream_origin_type_dataset_stats.json")
    save_json(metrics, output_dir / "downstream_origin_type_metrics.json")
    return metrics


def train_downstream(
    data_dir: str | Path,
    cache_dir: str | Path,
    contrastive_ckpt: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    splits_path: str | Path | None = None,
    cv_fold: int | None = None,
    val_fold: int | None = None,
    baseline_trials: int = 100,
    class_names: list[str] | None = None,
    save_cm_png: bool = False,
    tasks: list[str] | tuple[str, ...] | None = None,
    npatlas_path: str | Path = "data/NPAtlas_download_2024_09.tsv",
    mibig_pairs_path: str | Path = "data/MIBIG/processed/mibig_pairs.tsv",
    bioactivity_table_path: str | Path = "results/EDA/bgc_observed_bioactivities.csv",
    npclassifier_pair_labels_path: str | Path = "data/MIBIG/processed/mibig_pairs_npclassifier_labels.tsv",
    mw_bins: int = 50,
    force_rebuild_match: bool = False,
) -> dict[str, Any]:
    """Train one or more downstream models on frozen CLIP embeddings."""
    selected_tasks = list(tasks) if tasks else list(DEFAULT_TASKS)
    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    contrastive_model, _ = _load_contrastive_model(contrastive_ckpt, device)
    for param in contrastive_model.parameters():
        param.requires_grad = False

    results: dict[str, Any] = {"tasks": selected_tasks}
    compound_task_requested = any(task in COMPOUND_TASKS for task in selected_tasks)
    matched_df: pd.DataFrame | None = None
    compound_embeddings: dict[str, torch.Tensor] | None = None
    compound_match_stats: dict[str, int] | None = None

    if compound_task_requested:
        matched_path = outdir / "matched_compounds.tsv"
        matched_df, compound_match_stats = _prepare_compound_match_table(
            mibig_pairs_path=mibig_pairs_path,
            npatlas_path=npatlas_path,
            splits_path=splits_path,
            cv_fold=cv_fold,
            val_fold=val_fold,
            output_path=matched_path,
            force_rebuild=force_rebuild_match,
        )
        if matched_df.empty:
            raise ValueError("No MIBiG compounds could be matched to NPAtlas using InChIKey or canonical SMILES.")
        LOGGER.info(
            "Compound matching counts: total_mibig_rows=%d matched_by_inchikey=%d matched_by_smiles=%d total_matched_rows=%d",
            compound_match_stats["total_mibig_rows"],
            compound_match_stats["matched_by_inchikey"],
            compound_match_stats["matched_by_smiles"],
            compound_match_stats["total_matched_rows"],
        )

        interactions = build_interactions(data_dir, splits_path=splits_path, cv_fold=cv_fold, val_fold=val_fold)
        valid_pairs = interactions[["bgc_id", "compound_id", "split"]].drop_duplicates().copy()
        valid_pairs["bgc_id"] = valid_pairs["bgc_id"].astype(str)
        valid_pairs["compound_id"] = valid_pairs["compound_id"].astype(str)
        valid_pairs["split"] = valid_pairs["split"].astype(str).str.lower()
        matched_df = matched_df.merge(valid_pairs, on=["bgc_id", "compound_id", "split"], how="inner")
        if matched_df.empty:
            raise ValueError("Matched NPAtlas compounds do not overlap with cached compound features in the selected splits.")

        compound_cache = torch.load(Path(cache_dir) / "compound_features.pt", map_location="cpu")
        missing_ids = sorted(set(matched_df["compound_id"].astype(str).tolist()) - set(compound_cache))
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise KeyError(f"Missing compound features for {len(missing_ids)} matched compounds. Examples: {preview}")
        compound_embeddings = _build_compound_embedding_map(
            matched_df,
            contrastive_model,
            compound_cache,
            device,
            int(cfg["downstream"]["feature_batch_size"]),
        )
        matched_df.to_csv(matched_path, sep="\t", index=False)
        results["compound_matching"] = compound_match_stats
        results["matched_compounds_path"] = str(matched_path)

    for task in selected_tasks:
        if task == "bgc_class":
            results[task] = _train_bgc_class_task(
                data_dir=data_dir,
                cache_dir=cache_dir,
                cfg=cfg,
                device=device,
                contrastive_model=contrastive_model,
                splits_path=splits_path,
                cv_fold=cv_fold,
                val_fold=val_fold,
                baseline_trials=baseline_trials,
                class_names=class_names,
                save_cm_png=save_cm_png,
                output_dir=outdir,
            )
        elif task == "bioactivity_class":
            results[task] = _train_bioactivity_class_task(
                data_dir=data_dir,
                cache_dir=cache_dir,
                cfg=cfg,
                device=device,
                contrastive_model=contrastive_model,
                splits_path=splits_path,
                cv_fold=cv_fold,
                val_fold=val_fold,
                save_cm_png=save_cm_png,
                output_dir=outdir,
                bioactivity_table_path=bioactivity_table_path,
            )
        elif task in NPCLASSIFIER_TASKS:
            results[task] = _train_npclassifier_multilabel_task(
                task_name=task,
                data_dir=data_dir,
                cache_dir=cache_dir,
                cfg=cfg,
                device=device,
                contrastive_model=contrastive_model,
                splits_path=splits_path,
                cv_fold=cv_fold,
                val_fold=val_fold,
                save_cm_png=save_cm_png,
                output_dir=outdir,
                npclassifier_pair_labels_path=npclassifier_pair_labels_path,
            )
        elif task in COMPOUND_REGRESSION_TASKS:
            if matched_df is None or compound_embeddings is None or compound_match_stats is None:
                raise RuntimeError(f"Compound match state is missing for {task}.")
            results[task] = _train_compound_property_regression_task(
                task_name=task,
                matched_df=matched_df,
                embedding_map=compound_embeddings,
                cfg=cfg,
                device=device,
                output_dir=outdir,
                baseline_trials=baseline_trials,
                mw_bins=mw_bins,
            )
            results[task]["match_counts"].update(compound_match_stats)
        elif task == "origin_type":
            if matched_df is None or compound_embeddings is None or compound_match_stats is None:
                raise RuntimeError("Compound match state is missing for origin_type.")
            results[task] = _train_origin_type_task(
                matched_df=matched_df,
                embedding_map=compound_embeddings,
                cfg=cfg,
                device=device,
                output_dir=outdir,
                baseline_trials=baseline_trials,
                save_cm_png=save_cm_png,
            )
            results[task]["match_counts"].update(compound_match_stats)
        else:
            raise ValueError(f"Unsupported downstream task: {task}")

    return results
