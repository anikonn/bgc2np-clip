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


DEFAULT_TOP_K = (5, 10, 20, 50, 100, 200, 500)
DEFAULT_DIRECTIONS = ("bgc_to_compound", "compound_to_bgc")
METHOD_ORDER = ("random", "frozen_encoder_similarity", "knn5", "model")
METHOD_LABELS = {
    "random": "Random",
    "frozen_encoder_similarity": "Frozen",
    "knn5": "KNN-5",
    "model": "Combi",
}
DIRECTION_LABELS = {
    "bgc_to_compound": "BGC to NP",
    "compound_to_bgc": "NP to BGC",
}
DIRECTION_SUFFIXES = {
    "bgc_to_compound": "bgc_to_np",
    "compound_to_bgc": "np_to_bgc",
}
COLORS = {
    "random": "#b85b61",
    "frozen_encoder_similarity": "#8c8c8c",
    "knn5": "#64b27b",
    "model": "#4c72b0",
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BCCoE-style top-K recall panels from a run summary.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to a run_cv10 summary.json file.")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory. Defaults beside the summary.")
    parser.add_argument("--prefix", type=str, default="bccoe_retrieval", help="Output filename prefix.")
    parser.add_argument("--model_label", type=str, default="Combi", help="Legend label for the trained model.")
    parser.add_argument("--top_k", type=int, nargs="*", default=list(DEFAULT_TOP_K), help="Top-K values to plot.")
    parser.add_argument(
        "--directions",
        choices=list(DEFAULT_DIRECTIONS),
        nargs="*",
        default=list(DEFAULT_DIRECTIONS),
        help="Retrieval direction(s) to plot. Defaults to both, saved as separate files.",
    )
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
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _candidate_count(fold: dict[str, Any], direction: str) -> int | None:
    test_counts = fold.get("counts", {}).get("test", {})
    key = "n_compounds" if direction == "bgc_to_compound" else "n_bgcs"
    value = test_counts.get(key)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _append_metrics(
    rows: list[dict[str, Any]],
    *,
    fold_id: int,
    method: str,
    direction: str,
    metrics: dict[str, Any],
    candidate_count: int | None,
    top_k_values: list[int],
) -> None:
    for top_k in top_k_values:
        recall = _as_float(metrics.get(f"recall_at_{int(top_k)}"))
        if recall is None:
            continue
        rows.append(
            {
                "fold_id": int(fold_id),
                "method": method,
                "direction": direction,
                "top_k": int(top_k),
                "recall": float(recall),
                "candidate_count": int(candidate_count) if candidate_count is not None else np.nan,
            }
        )


def _baseline_metric_blocks(baselines: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    blocks: list[tuple[str, dict[str, Any]]] = []
    for name in ("random", "frozen_encoder_similarity"):
        payload = baselines.get(name)
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            blocks.append((name, payload["metrics"]))
    knn = baselines.get("knn_transfer")
    if isinstance(knn, dict):
        metrics = knn.get("metrics_by_k", {}).get("5")
        if isinstance(metrics, dict):
            blocks.append(("knn5", metrics))
    return blocks


def build_long_table(summary: dict[str, Any], top_k_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in summary.get("folds", []):
        fold_id = int(fold.get("fold_id", 0))
        for direction in ("bgc_to_compound", "compound_to_bgc"):
            candidate_count = _candidate_count(fold, direction)
            model_metrics = fold.get("retrieval_test", {}).get(direction, {})
            if isinstance(model_metrics, dict):
                _append_metrics(
                    rows,
                    fold_id=fold_id,
                    method="model",
                    direction=direction,
                    metrics=model_metrics,
                    candidate_count=candidate_count,
                    top_k_values=top_k_values,
                )
            baselines = fold.get("retrieval_baselines_test", {})
            if isinstance(baselines, dict):
                for method, metrics in _baseline_metric_blocks(baselines):
                    direction_metrics = metrics.get(direction, {}) if isinstance(metrics, dict) else {}
                    if isinstance(direction_metrics, dict):
                        _append_metrics(
                            rows,
                            fold_id=fold_id,
                            method=method,
                            direction=direction,
                            metrics=direction_metrics,
                            candidate_count=candidate_count,
                            top_k_values=top_k_values,
                        )
    return pd.DataFrame(rows)


def _summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    return (
        long_df.groupby(["method", "direction", "top_k"], dropna=False)
        .agg(
            recall_mean=("recall", "mean"),
            recall_std=("recall", lambda values: float(np.std(values, ddof=0))),
            n=("recall", "count"),
        )
        .reset_index()
    )


def _available_methods(summary_df: pd.DataFrame) -> list[str]:
    present = set(summary_df["method"].dropna().astype(str).tolist())
    ordered = [method for method in METHOD_ORDER if method in present]
    return ordered


def _plot_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    *,
    direction: str,
    metric: str,
    top_k_values: list[int],
    methods: list[str],
) -> None:
    x = np.arange(len(top_k_values), dtype=np.float64)
    width = min(0.12, 0.78 / max(len(methods), 1))
    offsets = (np.arange(len(methods), dtype=np.float64) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods, strict=True):
        values = []
        for top_k in top_k_values:
            row = summary_df[
                (summary_df["direction"] == direction)
                & (summary_df["method"] == method)
                & (summary_df["top_k"] == int(top_k))
            ]
            values.append(float(row.iloc[0][f"{metric}_mean"]) if not row.empty else np.nan)
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


def save_plot(
    summary_df: pd.DataFrame,
    output_path: Path,
    *,
    direction: str,
    top_k_values: list[int],
    model_label: str,
) -> None:
    METHOD_LABELS["model"] = model_label
    methods = _available_methods(summary_df)
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.4))
    _plot_panel(ax, summary_df, direction=direction, metric="recall", top_k_values=top_k_values, methods=methods)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True)
    fig.subplots_adjust(right=0.78)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = _load_json(args.summary)
    top_k_values = [int(k) for k in args.top_k]
    outdir = args.outdir if args.outdir is not None else args.summary.parent / "bccoe_retrieval_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    long_df = build_long_table(summary, top_k_values)
    long_path = outdir / f"{args.prefix}_long.csv"
    long_df.to_csv(long_path, index=False)
    summary_df = _summarize(long_df)
    summary_path = outdir / f"{args.prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    plot_paths: dict[str, str] = {}
    if not summary_df.empty:
        for direction in args.directions:
            direction_df = summary_df[summary_df["direction"] == direction]
            if direction_df.empty:
                continue
            suffix = DIRECTION_SUFFIXES[direction]
            plot_path = outdir / f"{args.prefix}_{suffix}_topk_recall.png"
            save_plot(direction_df, plot_path, direction=direction, top_k_values=top_k_values, model_label=str(args.model_label))
            plot_paths[direction] = str(plot_path)

    manifest = {
        "summary": str(args.summary),
        "long_csv": str(long_path),
        "summary_csv": str(summary_path),
        "plots": plot_paths,
        "top_k": top_k_values,
        "directions": list(args.directions),
    }
    (outdir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
