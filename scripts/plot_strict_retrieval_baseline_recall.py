from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOP_K_VALUES = (1, 5, 10)
DIRECTIONS = {
    "bgc_to_compound": ("BGC to NP", "bgc_to_np"),
    "compound_to_bgc": ("NP to BGC", "np_to_bgc"),
}
METHODS = (
    ("random", "Random"),
    ("frozen_encoder_similarity", "Frozen"),
    ("knn_transfer_k5", "KNN-5"),
    ("model", "BGC2NP-CLIP"),
)


@dataclass(frozen=True)
class MetricValue:
    mean: float
    std: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot strict-split retrieval recall with baselines and BGC2NP-CLIP.")
    parser.add_argument("--run_root", type=Path, default=Path("results/ohe_strict_cv10_val_selected"))
    parser.add_argument("--outdir", type=Path, default=Path("results/paper_plots"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=("png", "pdf", "svg"))
    return parser.parse_args()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.6, 4.2),
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#d7d7d7",
            "grid.linewidth": 0.8,
            "grid.linestyle": ":",
            "axes.axisbelow": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_model_recall(run_root: Path) -> dict[tuple[str, int], MetricValue]:
    summary_path = run_root / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    retrieval = payload["aggregate"]["contrastive_metrics"]["retrieval_test"]
    values: dict[tuple[str, int], MetricValue] = {}
    for direction in DIRECTIONS:
        for top_k in TOP_K_VALUES:
            item = retrieval[direction][f"recall_at_{top_k}"]
            values[(direction, top_k)] = MetricValue(
                mean=float(item["mean"]),
                std=float(item.get("std", 0.0)),
                n=int(item.get("n", 0)),
            )
    return values


def _load_baseline_recall(run_root: Path) -> dict[tuple[str, str, int], MetricValue]:
    summary_path = run_root / "baselines" / "retrieval" / "retrieval_baselines_summary.csv"
    df = pd.read_csv(summary_path)
    values: dict[tuple[str, str, int], MetricValue] = {}
    for method, _ in METHODS:
        if method == "model":
            continue
        for direction in DIRECTIONS:
            for top_k in TOP_K_VALUES:
                row = df[
                    (df["baseline"] == method)
                    & (df["direction"] == direction)
                    & (df["metric"] == f"recall_at_{top_k}")
                ]
                if row.empty:
                    raise ValueError(f"Missing {method} {direction} recall_at_{top_k} in {summary_path}")
                record = row.iloc[0]
                values[(method, direction, top_k)] = MetricValue(
                    mean=float(record["mean"]),
                    std=float(record["std"]) if pd.notna(record["std"]) else 0.0,
                    n=int(record["n"]) if pd.notna(record["n"]) else 0,
                )
    return values


def load_strict_retrieval_recall(run_root: Path) -> dict[tuple[str, str, int], MetricValue]:
    """Load strict-split recall@K values for Random, Frozen, KNN-5, and BGC2NP-CLIP."""
    baseline_values = _load_baseline_recall(run_root)
    model_values = _load_model_recall(run_root)
    values = dict(baseline_values)
    for direction in DIRECTIONS:
        for top_k in TOP_K_VALUES:
            values[("model", direction, top_k)] = model_values[(direction, top_k)]
    return values


def plot_strict_retrieval_baseline_recall(
    values: dict[tuple[str, str, int], MetricValue],
    *,
    direction: str,
    outdir: Path,
    formats: list[str],
    dpi: int,
    ymax: float | None = None,
) -> list[Path]:
    title, direction_slug = DIRECTIONS[direction]
    method_keys = [key for key, _ in METHODS]
    method_labels = [label for _, label in METHODS]
    set3 = plt.get_cmap("Set3").colors
    colors = [set3[0], set3[4], set3[2], set3[3]]

    x = np.arange(len(TOP_K_VALUES), dtype=float)
    width = min(0.18, 0.78 / len(method_keys))
    offsets = (np.arange(len(method_keys), dtype=float) - (len(method_keys) - 1) / 2.0) * width

    fig, ax = plt.subplots()
    for idx, method in enumerate(method_keys):
        means = [values[(method, direction, top_k)].mean for top_k in TOP_K_VALUES]
        stds = [values[(method, direction, top_k)].std for top_k in TOP_K_VALUES]
        ax.bar(
            x + offsets[idx],
            means,
            yerr=stds,
            capsize=2.5,
            width=width,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.6,
            label=method_labels[idx],
        )

    ax.set_title(title)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Top-K")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in TOP_K_VALUES])
    ax.set_ylim(bottom=0.0, top=ymax)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), ncols=1)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

    paths: list[Path] = []
    for fmt in formats:
        path = outdir / f"strict_retrieval_{direction_slug}_recall_baselines.{fmt}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def _write_values_csv(values: dict[tuple[str, str, int], MetricValue], outdir: Path) -> Path:
    path = outdir / "strict_retrieval_baseline_recall_values.csv"
    rows = []
    for method, label in METHODS:
        for direction in DIRECTIONS:
            for top_k in TOP_K_VALUES:
                item = values[(method, direction, top_k)]
                rows.append(
                    {
                        "method": method,
                        "label": label,
                        "direction": direction,
                        "top_k": top_k,
                        "recall_mean": item.mean,
                        "recall_std": item.std,
                        "n": item.n,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _shared_ymax(values: dict[tuple[str, str, int], MetricValue]) -> float:
    return 0.35


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _set_style()
    values = load_strict_retrieval_recall(args.run_root)
    values_csv = _write_values_csv(values, args.outdir)
    ymax = _shared_ymax(values)
    print(f"Wrote values CSV: {values_csv}")
    for direction in DIRECTIONS:
        for path in plot_strict_retrieval_baseline_recall(
            values,
            direction=direction,
            outdir=args.outdir,
            formats=list(args.formats),
            dpi=int(args.dpi),
            ymax=ymax,
        ):
            print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
