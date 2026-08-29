from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt


TOP_K_DEFAULT = (5, 10, 20, 50, 100, 200, 500)
DIRECTIONS = ("bgc_to_compound", "compound_to_bgc")
DIRECTION_LABELS = {"bgc_to_compound": "BGC to NP", "compound_to_bgc": "NP to BGC"}
DIRECTION_SUFFIXES = {"bgc_to_compound": "bgc_to_np", "compound_to_bgc": "np_to_bgc"}
METHOD_ORDER = ("random", "frozen_encoder_similarity", "knn", "model")
METHOD_LABELS = {
    "random": "Random",
    "frozen_encoder_similarity": "Frozen",
    "knn": "KNN",
    "model": "BGC2NP-CLIP",
}
COLORS = {
    "random": "#b85b61",
    "frozen_encoder_similarity": "#8c8c8c",
    "knn": "#64b27b",
    "model": "#4c72b0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create aggregate retrieval plots from a CV summary.json.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="retrieval")
    parser.add_argument("--top_k", type=int, nargs="*", default=list(TOP_K_DEFAULT))
    parser.add_argument("--model_label", type=str, default="BGC2NP-CLIP")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("mean", "value"):
            if key in value:
                return _as_float(value[key])
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _baseline_blocks(baselines: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    blocks: list[tuple[str, dict[str, Any]]] = []
    for name in ("random", "frozen_encoder_similarity"):
        payload = baselines.get(name)
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            blocks.append((name, payload["metrics"]))
    knn = baselines.get("knn_transfer")
    if isinstance(knn, dict):
        metrics = knn.get("metrics_by_k", {}).get("1")
        if isinstance(metrics, dict):
            blocks.append(("knn", metrics))
    return blocks


def build_retrieval_long(summary: dict[str, Any], top_k_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in summary.get("folds", []):
        fold_id = int(fold.get("fold_id", 0))
        method_payloads = [("model", fold.get("retrieval_test", {}))]
        method_payloads.extend(_baseline_blocks(fold.get("retrieval_baselines_test", {})))
        for method, payload in method_payloads:
            for direction in DIRECTIONS:
                metrics = payload.get(direction, {}) if isinstance(payload, dict) else {}
                if not isinstance(metrics, dict):
                    continue
                mrr = _as_float(metrics.get("mrr"))
                if mrr is not None:
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "method": method,
                            "direction": direction,
                            "metric": "mrr",
                            "top_k": np.nan,
                            "value": float(mrr),
                        }
                    )
                for top_k in top_k_values:
                    hit = _as_float(metrics.get(f"hit_at_{int(top_k)}"))
                    if hit is not None:
                        rows.append(
                            {
                                "fold_id": fold_id,
                                "method": method,
                                "direction": direction,
                                "metric": "hit",
                                "top_k": int(top_k),
                                "value": float(hit),
                            }
                        )
                    recall = _as_float(metrics.get(f"recall_at_{int(top_k)}"))
                    if recall is None:
                        continue
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "method": method,
                            "direction": direction,
                            "metric": "recall",
                            "top_k": int(top_k),
                            "value": float(recall),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_retrieval(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    return (
        long_df.groupby(["method", "direction", "metric", "top_k"], dropna=False)
        .agg(value_mean=("value", "mean"), value_std=("value", lambda x: float(np.std(x, ddof=0))), n=("value", "count"))
        .reset_index()
    )


def _available_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].dropna().astype(str).tolist())
    return [method for method in METHOD_ORDER if method in present]


def plot_topk_recall(summary_df: pd.DataFrame, outdir: Path, prefix: str, top_k_values: list[int]) -> list[str]:
    paths: list[str] = []
    recall_df = summary_df[summary_df["metric"] == "recall"].copy()
    methods = _available_methods(recall_df)
    for direction in DIRECTIONS:
        direction_df = recall_df[recall_df["direction"] == direction]
        if direction_df.empty:
            continue
        x = np.arange(len(top_k_values), dtype=float)
        width = min(0.12, 0.78 / max(len(methods), 1))
        offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width
        fig, ax = plt.subplots(figsize=(8.8, 4.4))
        for offset, method in zip(offsets, methods, strict=True):
            values = []
            for top_k in top_k_values:
                row = direction_df[(direction_df["method"] == method) & (direction_df["top_k"] == int(top_k))]
                values.append(float(row.iloc[0]["value_mean"]) if not row.empty else np.nan)
            ax.bar(
                x + offset,
                values,
                width=width,
                label=METHOD_LABELS.get(method, method),
                color=COLORS.get(method, "#999999"),
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in top_k_values])
        ax.set_xlabel("top-K")
        ax.set_ylabel("Recall")
        ax.set_title(DIRECTION_LABELS.get(direction, direction))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0.0)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", frameon=True)
        fig.subplots_adjust(right=0.78)
        path = outdir / f"{prefix}_{DIRECTION_SUFFIXES[direction]}_topk_recall.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_topk_hit(summary_df: pd.DataFrame, outdir: Path, prefix: str, top_k_values: list[int]) -> list[str]:
    paths: list[str] = []
    hit_df = summary_df[summary_df["metric"] == "hit"].copy()
    methods = _available_methods(hit_df)
    for direction in DIRECTIONS:
        direction_df = hit_df[hit_df["direction"] == direction]
        if direction_df.empty:
            continue
        x = np.arange(len(top_k_values), dtype=float)
        width = min(0.12, 0.78 / max(len(methods), 1))
        offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width
        fig, ax = plt.subplots(figsize=(8.8, 4.4))
        for offset, method in zip(offsets, methods, strict=True):
            values = []
            for top_k in top_k_values:
                row = direction_df[(direction_df["method"] == method) & (direction_df["top_k"] == int(top_k))]
                values.append(float(row.iloc[0]["value_mean"]) if not row.empty else np.nan)
            ax.bar(
                x + offset,
                values,
                width=width,
                label=METHOD_LABELS.get(method, method),
                color=COLORS.get(method, "#999999"),
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in top_k_values])
        ax.set_xlabel("top-K")
        ax.set_ylabel("Hit@K")
        ax.set_title(DIRECTION_LABELS.get(direction, direction))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0.0)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", frameon=True)
        fig.subplots_adjust(right=0.78)
        path = outdir / f"{prefix}_{DIRECTION_SUFFIXES[direction]}_topk_hit.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_mrr(summary_df: pd.DataFrame, outdir: Path, prefix: str) -> str | None:
    mrr_df = summary_df[summary_df["metric"] == "mrr"].copy()
    if mrr_df.empty:
        return None
    methods = _available_methods(mrr_df)
    x = np.arange(len(DIRECTIONS), dtype=float)
    width = min(0.16, 0.78 / max(len(methods), 1))
    offsets = (np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for offset, method in zip(offsets, methods, strict=True):
        values = []
        errors = []
        for direction in DIRECTIONS:
            row = mrr_df[(mrr_df["method"] == method) & (mrr_df["direction"] == direction)]
            values.append(float(row.iloc[0]["value_mean"]) if not row.empty else np.nan)
            errors.append(float(row.iloc[0]["value_std"]) if not row.empty else 0.0)
        ax.bar(
            x + offset,
            values,
            yerr=errors,
            capsize=3,
            width=width,
            label=METHOD_LABELS.get(method, method),
            color=COLORS.get(method, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DIRECTION_LABELS[d] for d in DIRECTIONS])
    ax.set_ylabel("MRR")
    ax.set_title("Mean Reciprocal Rank")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0.0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True)
    fig.subplots_adjust(right=0.76)
    path = outdir / f"{prefix}_mrr.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def _class_order(classes: list[str]) -> list[str]:
    preferred = ["NRPS", "other", "PKS", "ribosomal", "saccharide", "terpene"]
    return [c for c in preferred if c in classes] + sorted(c for c in classes if c not in preferred)


def _raw_matrix(confusion: dict[str, Any]) -> np.ndarray:
    raw = confusion["raw"]
    return np.asarray(
        [
            [raw["Negative"]["Negative"], raw["Negative"]["Positive"]],
            [raw["Positive"]["Negative"], raw["Positive"]["Positive"]],
        ],
        dtype=float,
    )


def build_class_summary(summary: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_class: dict[str, list[dict[str, Any]]] = {}
    micro_curves: list[dict[str, Any]] = []
    for fold in summary.get("folds", []):
        fold_id = int(fold.get("fold_id", 0))
        report = fold.get("retrieval_class_test", {})
        if not isinstance(report, dict):
            continue
        micro = report.get("micro_roc_curve")
        if isinstance(micro, dict) and report.get("micro_auc", 0.0):
            micro_curves.append({"fold_id": fold_id, "curve": micro, "auc": float(report.get("micro_auc", 0.0))})
        for class_name, metrics in report.get("classes", {}).items():
            by_class.setdefault(class_name, []).append(metrics)
            rows.append(
                {
                    "fold_id": fold_id,
                    "class": class_name,
                    "auroc": float(metrics.get("auroc", 0.0)),
                    "n_positive": int(metrics.get("n_positive", 0)),
                    "n_negative": int(metrics.get("n_negative", 0)),
                }
            )
    return pd.DataFrame(rows), by_class, micro_curves


def _plot_mean_roc(curves: list[tuple[list[float], list[float], float]], label: str, ax: plt.Axes) -> None:
    grid = np.linspace(0.0, 1.0, 201)
    tprs = []
    aucs = []
    for fpr, tpr, auc in curves:
        if len(fpr) < 2 or len(tpr) < 2:
            continue
        tprs.append(np.interp(grid, np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)))
        aucs.append(float(auc))
    if not tprs:
        return
    mean_tpr = np.mean(np.vstack(tprs), axis=0)
    ax.plot(grid, mean_tpr, linewidth=1.7, label=f"{label} (AUC = {np.mean(aucs):.3f})")


def plot_class_retrieval(summary: dict[str, Any], outdir: Path, prefix: str) -> dict[str, str]:
    class_df, by_class, micro_curves = build_class_summary(summary)
    outputs: dict[str, str] = {}
    class_csv = outdir / f"{prefix}_bgc_class_retrieval_summary.csv"
    class_df.to_csv(class_csv, index=False)
    outputs["class_summary_csv"] = str(class_csv)
    if class_df.empty:
        note = outdir / f"{prefix}_bgc_class_retrieval_plots_not_created.txt"
        note.write_text(
            "No retrieval_class_test.classes were found in the summary. "
            "For strict CV runs made before the bgc_classes merge fix, rerun run_cv10 to regenerate these metrics.\n",
            encoding="utf-8",
        )
        outputs["note"] = str(note)
        return outputs

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for class_name in _class_order(list(by_class)):
        curves = []
        for metrics in by_class[class_name]:
            curve = metrics.get("roc_curve", {})
            curves.append((curve.get("fpr", []), curve.get("tpr", []), float(metrics.get("auroc", 0.0))))
        _plot_mean_roc(curves, class_name, ax)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Retrieval BGC-Class ROC")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    roc_path = outdir / f"{prefix}_bgc_class_retrieval_roc.png"
    fig.savefig(roc_path, dpi=220)
    plt.close(fig)
    outputs["class_roc"] = str(roc_path)

    if micro_curves:
        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        curves = [
            (item["curve"].get("fpr", []), item["curve"].get("tpr", []), float(item["auc"]))
            for item in micro_curves
        ]
        _plot_mean_roc(curves, "micro", ax)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.7)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Retrieval Micro ROC")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        micro_path = outdir / f"{prefix}_retrieval_micro_roc.png"
        fig.savefig(micro_path, dpi=220)
        plt.close(fig)
        outputs["micro_roc"] = str(micro_path)

    classes = _class_order(list(by_class))
    n_cols = min(3, len(classes))
    n_rows = int(math.ceil(len(classes) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, class_name in zip(axes.flat, classes, strict=False):
        ax.axis("on")
        matrix = np.zeros((2, 2), dtype=float)
        for metrics in by_class[class_name]:
            matrix += _raw_matrix(metrics["confusion_matrix"])
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
                color = "white" if max_value and matrix[i, j] >= 0.5 * max_value else "black"
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color=color, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = outdir / f"{prefix}_bgc_class_retrieval_confusion_matrices.png"
    fig.savefig(cm_path, dpi=220)
    plt.close(fig)
    outputs["class_confusion_matrices"] = str(cm_path)
    return outputs


def main() -> None:
    args = parse_args()
    METHOD_LABELS["model"] = args.model_label
    summary = _load_json(args.summary)
    outdir = args.outdir if args.outdir is not None else args.summary.parent / "retrieval_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    top_k_values = [int(k) for k in args.top_k]

    long_df = build_retrieval_long(summary, top_k_values)
    long_path = outdir / f"{args.prefix}_long.csv"
    long_df.to_csv(long_path, index=False)
    summary_df = summarize_retrieval(long_df)
    summary_path = outdir / f"{args.prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plots = {
        "topk_hit": plot_topk_hit(summary_df, outdir, args.prefix, top_k_values),
        "topk_recall": plot_topk_recall(summary_df, outdir, args.prefix, top_k_values),
        "mrr": plot_mrr(summary_df, outdir, args.prefix),
        "class_retrieval": plot_class_retrieval(summary, outdir, args.prefix),
    }
    manifest = {
        "summary": str(args.summary),
        "long_csv": str(long_path),
        "summary_csv": str(summary_path),
        "plots": plots,
        "top_k": top_k_values,
    }
    (outdir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
