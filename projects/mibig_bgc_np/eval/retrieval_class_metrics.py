from __future__ import annotations

import math
import os
import re
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def _parse_label_text(label_text: object) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    raw = "" if label_text is None else str(label_text)
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        candidates = [str(item) for item in parsed]
    else:
        candidates = re.split(r"[;,]", raw)

    for label in candidates:
        clean = str(label).strip().strip("'\"")
        if clean and clean not in seen:
            labels.append(clean)
            seen.add(clean)
    return labels


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(label).strip().lower()).strip("_")
    return slug or "class"


def _ordered_class_items(classes: dict[str, Any]) -> list[tuple[str, Any]]:
    preferred = ["NRPS", "other", "PKS", "ribosomal", "saccharide", "terpene"]
    emitted: set[str] = set()
    ordered: list[tuple[str, Any]] = []
    for class_name in preferred:
        if class_name in classes:
            ordered.append((class_name, classes[class_name]))
            emitted.add(class_name)
    ordered.extend((class_name, classes[class_name]) for class_name in sorted(classes) if class_name not in emitted)
    return ordered


def _class_map(interactions: pd.DataFrame, split: str) -> dict[str, list[str]]:
    split_df = interactions[interactions["split"].astype(str).str.lower() == split.lower()].copy()
    label_col = "bgc_classes" if "bgc_classes" in split_df.columns else "bgc_class"
    if label_col not in split_df.columns:
        return {}

    labels_by_bgc: dict[str, list[str]] = {}
    for row in split_df[["bgc_id", label_col]].drop_duplicates().itertuples(index=False):
        bgc_id = str(row.bgc_id)
        labels = _parse_label_text(getattr(row, label_col))
        if labels:
            labels_by_bgc[bgc_id] = labels
    return labels_by_bgc


def _binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[list[float], list[float], list[float], float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0 or y_true.size == 0:
        return [0.0, 1.0], [0.0, 1.0], [math.inf, -math.inf], 0.0

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    sorted_score = y_score[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    distinct = np.where(np.diff(sorted_score))[0]
    threshold_idxs = np.r_[distinct, len(sorted_score) - 1]

    tps = np.r_[0, tp[threshold_idxs]]
    fps = np.r_[0, fp[threshold_idxs]]
    thresholds = np.r_[math.inf, sorted_score[threshold_idxs]]
    tpr = tps / float(pos)
    fpr = fps / float(neg)
    auc = _trapezoid_integral(tpr, fpr)
    return (
        [float(x) for x in fpr.tolist()],
        [float(x) for x in tpr.tolist()],
        [float(x) for x in thresholds.tolist()],
        float(max(0.0, min(1.0, auc))),
    )


def _best_threshold(fpr: list[float], tpr: list[float], thresholds: list[float]) -> float:
    if not thresholds:
        return 0.0
    scores = np.asarray(tpr, dtype=np.float64) - np.asarray(fpr, dtype=np.float64)
    idx = int(np.argmax(scores))
    threshold = float(thresholds[idx])
    if math.isinf(threshold):
        finite = [value for value in thresholds if not math.isinf(float(value))]
        return float(finite[0]) if finite else 0.0
    return threshold


def _confusion(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = np.asarray(y_score, dtype=np.float64) >= float(threshold)
    true = np.asarray(y_true, dtype=np.int64).astype(bool)
    tn = int((~true & ~pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tp = int((true & pred).sum())

    raw = {
        "Negative": {"Negative": tn, "Positive": fp},
        "Positive": {"Negative": fn, "Positive": tp},
    }
    normalized_true: dict[str, dict[str, float]] = {}
    for row_name, row in raw.items():
        denom = float(sum(row.values()))
        normalized_true[row_name] = {
            col_name: (float(value) / denom if denom else 0.0)
            for col_name, value in row.items()
        }
    return {"labels": ["Negative", "Positive"], "raw": raw, "normalized_true": normalized_true}


def _precision_recall_f1(confusion: dict[str, Any]) -> dict[str, float]:
    raw = confusion["raw"]
    tn = float(raw["Negative"]["Negative"])
    fp = float(raw["Negative"]["Positive"])
    fn = float(raw["Positive"]["Negative"])
    tp = float(raw["Positive"]["Positive"])
    del tn
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recall": float(recall), "precision": float(precision), "f1": float(f1)}


def evaluate_bgc_class_retrieval(
    sim: torch.Tensor,
    bgc_ids: list[str],
    compound_ids: list[str],
    pairs: list[tuple[int, int]],
    interactions: pd.DataFrame,
    split: str,
) -> dict[str, Any]:
    """Compute BGC-product binary matching ROC diagnostics stratified by BGC class."""
    del compound_ids
    if sim.numel() == 0:
        return {"split": split, "classes": {}, "macro_auc": 0.0, "micro_auc": 0.0}

    device = sim.device
    pos_mask = torch.zeros(sim.shape, dtype=torch.bool, device=device)
    if pairs:
        pair_array = np.asarray(pairs, dtype=np.int64)
        left_idx = torch.tensor(pair_array[:, 0], dtype=torch.long, device=device)
        right_idx = torch.tensor(pair_array[:, 1], dtype=torch.long, device=device)
        pos_mask[left_idx, right_idx] = True

    labels_by_bgc = _class_map(interactions, split)
    class_to_rows: dict[str, list[int]] = {}
    for row_idx, bgc_id in enumerate(bgc_ids):
        for label in labels_by_bgc.get(str(bgc_id), []):
            class_to_rows.setdefault(label, []).append(row_idx)

    class_metrics: dict[str, Any] = {}
    micro_true_parts: list[np.ndarray] = []
    micro_score_parts: list[np.ndarray] = []
    sim_cpu = sim.detach().cpu()
    pos_cpu = pos_mask.detach().cpu()

    for class_name in sorted(class_to_rows):
        rows = class_to_rows[class_name]
        y_true = pos_cpu[rows, :].reshape(-1).to(dtype=torch.int64).numpy()
        y_score = sim_cpu[rows, :].reshape(-1).to(dtype=torch.float32).numpy()
        positives = int(y_true.sum())
        negatives = int(y_true.size - positives)
        if positives == 0 or negatives == 0:
            continue

        fpr, tpr, thresholds, auc = _binary_roc_curve(y_true, y_score)
        threshold = _best_threshold(fpr, tpr, thresholds)
        class_metrics[class_name] = {
            "auroc": float(auc),
            "threshold": float(threshold),
            "n_bgcs": int(len(rows)),
            "n_pairs": int(y_true.size),
            "n_positive": positives,
            "n_negative": negatives,
            "roc_curve": {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
            },
            "confusion_matrix": _confusion(y_true, y_score, threshold),
        }
        micro_true_parts.append(y_true)
        micro_score_parts.append(y_score)

    if micro_true_parts:
        micro_true = np.concatenate(micro_true_parts)
        micro_score = np.concatenate(micro_score_parts)
        micro_fpr, micro_tpr, micro_thresholds, micro_auc = _binary_roc_curve(micro_true, micro_score)
    else:
        micro_fpr, micro_tpr, micro_thresholds, micro_auc = [0.0, 1.0], [0.0, 1.0], [math.inf, -math.inf], 0.0

    return {
        "split": split,
        "classes": class_metrics,
        "macro_auc": float(np.mean([item["auroc"] for item in class_metrics.values()])) if class_metrics else 0.0,
        "micro_auc": float(micro_auc),
        "micro_roc_curve": {
            "fpr": micro_fpr,
            "tpr": micro_tpr,
            "thresholds": micro_thresholds,
        },
    }


def evaluate_bgc_class_pair_scores(
    pair_scores: pd.DataFrame,
    *,
    split: str,
    bgc_col: str = "bgc_id",
    score_col: str = "score",
    label_col: str = "is_product",
    class_col: str = "bgc_classes",
    thresholds_by_class: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute paper-style BGC-product binary matching metrics over explicit pair rows."""
    required = {bgc_col, score_col, label_col, class_col}
    missing = required.difference(pair_scores.columns)
    if missing:
        raise ValueError(f"Pair score table is missing required columns: {sorted(missing)}")

    df = pair_scores.dropna(subset=[bgc_col, score_col, label_col, class_col]).copy()
    df[bgc_col] = df[bgc_col].astype(str)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[score_col, label_col]).copy()
    df[label_col] = (df[label_col].astype(float) > 0.0).astype(int)
    if df.empty:
        return {"split": split, "classes": {}, "macro_auc": 0.0, "micro_auc": 0.0}

    class_to_indices: dict[str, list[int]] = {}
    for idx, labels in enumerate(df[class_col].tolist()):
        for label in _parse_label_text(labels):
            class_to_indices.setdefault(label, []).append(idx)

    class_metrics: dict[str, Any] = {}
    micro_true_parts: list[np.ndarray] = []
    micro_score_parts: list[np.ndarray] = []
    y_all = df[label_col].to_numpy(dtype=np.int64)
    score_all = df[score_col].to_numpy(dtype=np.float64)
    bgc_all = df[bgc_col].to_numpy(dtype=str)

    for class_name in sorted(class_to_indices):
        indices = np.asarray(class_to_indices[class_name], dtype=np.int64)
        y_true = y_all[indices]
        y_score = score_all[indices]
        positives = int(y_true.sum())
        negatives = int(y_true.size - positives)
        if positives == 0 or negatives == 0:
            continue

        fpr, tpr, thresholds, auc = _binary_roc_curve(y_true, y_score)
        threshold = (
            float(thresholds_by_class[class_name])
            if thresholds_by_class is not None and class_name in thresholds_by_class
            else _best_threshold(fpr, tpr, thresholds)
        )
        confusion = _confusion(y_true, y_score, threshold)
        prf = _precision_recall_f1(confusion)
        class_metrics[class_name] = {
            "auroc": float(auc),
            "threshold": float(threshold),
            "threshold_source": "external" if thresholds_by_class is not None and class_name in thresholds_by_class else split,
            "recall": prf["recall"],
            "precision": prf["precision"],
            "f1": prf["f1"],
            "n_bgcs": int(len(set(bgc_all[indices].tolist()))),
            "n_rows": int(y_true.size),
            "n_positive": positives,
            "n_negative": negatives,
            "roc_curve": {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
            },
            "confusion_matrix": confusion,
        }
        micro_true_parts.append(y_true)
        micro_score_parts.append(y_score)

    if micro_true_parts:
        micro_true = np.concatenate(micro_true_parts)
        micro_score = np.concatenate(micro_score_parts)
        micro_fpr, micro_tpr, micro_thresholds, micro_auc = _binary_roc_curve(micro_true, micro_score)
    else:
        micro_fpr, micro_tpr, micro_thresholds, micro_auc = [0.0, 1.0], [0.0, 1.0], [math.inf, -math.inf], 0.0

    return {
        "split": split,
        "classes": class_metrics,
        "macro_auc": float(np.mean([item["auroc"] for item in class_metrics.values()])) if class_metrics else 0.0,
        "micro_auc": float(micro_auc),
        "micro_roc_curve": {
            "fpr": micro_fpr,
            "tpr": micro_tpr,
            "thresholds": micro_thresholds,
        },
        "n_rows": int(len(df)),
        "n_positive": int(df[label_col].sum()),
        "n_negative": int(len(df) - int(df[label_col].sum())),
    }


def save_bgc_map_metrics_table(report: dict[str, Any], output_dir: str | Path, prefix: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for class_name, metrics in _ordered_class_items(report.get("classes", {})):
        rows.extend(
            [
                {
                    "class": class_name,
                    "BGC count": int(metrics.get("n_bgcs", 0)),
                    "model": "AUROC",
                    "BGC-MAP": float(metrics.get("auroc", 0.0)),
                },
                {
                    "class": "",
                    "BGC count": "",
                    "model": "recall",
                    "BGC-MAP": float(metrics.get("recall", 0.0)),
                },
                {
                    "class": "",
                    "BGC count": "",
                    "model": "precision",
                    "BGC-MAP": float(metrics.get("precision", 0.0)),
                },
                {
                    "class": "",
                    "BGC count": "",
                    "model": "F1",
                    "BGC-MAP": float(metrics.get("f1", 0.0)),
                },
            ]
        )

    table_df = pd.DataFrame(rows, columns=["class", "BGC count", "model", "BGC-MAP"])
    csv_path = output / f"{prefix}_bgc_map_metrics_table.csv"
    table_df.to_csv(csv_path, index=False)

    png_path = output / f"{prefix}_bgc_map_metrics_table.png"
    if not table_df.empty:
        display = table_df.copy()
        display["BGC-MAP"] = display["BGC-MAP"].map(lambda value: f"{float(value):.3f}")
        fig_height = max(3.0, 0.34 * (len(display) + 1))
        fig, ax = plt.subplots(figsize=(7.2, fig_height))
        ax.axis("off")
        table = ax.table(
            cellText=display.values.tolist(),
            colLabels=display.columns.tolist(),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.25)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#e6e6e6")
                cell.set_text_props(weight="bold")
            elif display.iloc[row - 1]["class"]:
                cell.set_text_props(weight="bold")
        fig.tight_layout()
        fig.savefig(png_path, dpi=220)
        plt.close(fig)
    else:
        png_path.write_text("No BGC-MAP class metrics available.\n", encoding="utf-8")

    return {"csv": str(csv_path), "png": str(png_path)}


def save_bgc_class_retrieval_plots(report: dict[str, Any], output_dir: str | Path, prefix: str) -> list[str]:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    classes = report.get("classes", {})
    paths: list[str] = []
    if not classes:
        return paths
    ordered_classes = _ordered_class_items(classes)

    roc_path = output / f"{prefix}_bgc_class_retrieval_roc.png"
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for class_name, metrics in ordered_classes:
        curve = metrics["roc_curve"]
        ax.plot(curve["fpr"], curve["tpr"], linewidth=1.5, label=f"{class_name} (AUC = {metrics['auroc']:.3f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve for BGC-product Matching")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    paths.append(str(roc_path))

    n_classes = len(ordered_classes)
    n_cols = min(3, n_classes)
    n_rows = int(math.ceil(n_classes / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, (class_name, metrics) in zip(axes.flat, ordered_classes, strict=False):
        ax.axis("on")
        raw = metrics["confusion_matrix"]["raw"]
        matrix = np.asarray(
            [
                [raw["Negative"]["Negative"], raw["Negative"]["Positive"]],
                [raw["Positive"]["Negative"], raw["Positive"]["Positive"]],
            ],
            dtype=float,
        )
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(class_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"], rotation=90, va="center")
        max_value = float(matrix.max()) if matrix.size else 0.0
        for i in range(2):
            for j in range(2):
                value = int(matrix[i, j])
                color = "white" if max_value and matrix[i, j] >= 0.5 * max_value else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    cm_path = output / f"{prefix}_bgc_class_retrieval_confusion_matrices.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    paths.append(str(cm_path))
    return paths
