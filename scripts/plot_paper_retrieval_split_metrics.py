from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


TOP_K_VALUES = (1, 5, 10)
DIRECTIONS = {
    "bgc_to_compound": "BGC to NP",
    "compound_to_bgc": "NP to BGC",
}
METRICS = {
    "hit": "Hit",
    "recall": "Recall",
    "mrr": "MRR",
}
DEFAULT_SPLITS = (
    ("BGC", Path("results/ohe_bgc_cv10_val_selected/retrieval_plots/retrieval_summary.csv")),
    ("NP", Path("results/ohe_np_cv10_val_selected/retrieval_plots/retrieval_summary.csv")),
    ("Combined", Path("results/ohe_combined_cv10_val_selected/retrieval_plots/retrieval_summary.csv")),
    ("Strict", Path("results/ohe_strict_cv10_val_selected/retrieval_plots/retrieval_summary.csv")),
)


@dataclass(frozen=True)
class MetricValue:
    mean: float
    std: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper retrieval split-comparison plots for Hit, Recall, and MRR."
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/paper_plots"))
    parser.add_argument("--method", type=str, default="model")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Retrieval summary CSV to compare; repeat for each split.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=("png", "pdf", "svg"))
    return parser.parse_args()


def _parse_splits(values: list[str] | None) -> tuple[tuple[str, Path], ...]:
    if not values:
        return DEFAULT_SPLITS
    splits: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --split value {value!r}; expected LABEL=PATH")
        label, path = value.split("=", 1)
        splits.append((label.strip(), Path(path)))
    return tuple(splits)


def _parse_top_k(value: str) -> int | None:
    if value == "":
        return None
    return int(float(value))


def _load_summary(path: Path, method: str) -> dict[tuple[str, str, int | None], MetricValue]:
    rows: dict[tuple[str, str, int | None], MetricValue] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["method"] != method:
                continue
            metric = row["metric"]
            if metric not in METRICS:
                continue
            direction = row["direction"]
            if direction not in DIRECTIONS:
                continue
            top_k = _parse_top_k(row["top_k"])
            rows[(direction, metric, top_k)] = MetricValue(
                mean=float(row["value_mean"]),
                std=float(row["value_std"]) if row["value_std"] else 0.0,
                n=int(float(row["n"])) if row["n"] else 0,
            )
    if method == "model":
        _backfill_model_top1_from_summary_json(rows, path)
    return rows


def _backfill_model_top1_from_summary_json(
    rows: dict[tuple[str, str, int | None], MetricValue],
    retrieval_summary_path: Path,
) -> None:
    summary_path = retrieval_summary_path.parent / "summary.json"
    if not summary_path.exists():
        summary_path = retrieval_summary_path.parents[1] / "summary.json"
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    retrieval = aggregate.get("retrieval_test", {})
    if not retrieval:
        retrieval = aggregate.get("contrastive_metrics", {}).get("retrieval_test", {})
    for direction in DIRECTIONS:
        direction_metrics = retrieval.get(direction, {})
        for metric in ("hit", "recall"):
            key = (direction, metric, 1)
            if key in rows:
                continue
            source = direction_metrics.get(f"{metric}_at_1")
            if not isinstance(source, dict):
                continue
            rows[key] = MetricValue(
                mean=float(source.get("mean", 0.0)),
                std=float(source.get("std", 0.0)),
                n=int(source.get("n", 0)),
            )


def _load_all(
    method: str,
    splits: tuple[tuple[str, Path], ...],
) -> dict[str, dict[tuple[str, str, int | None], MetricValue]]:
    loaded: dict[str, dict[tuple[str, str, int | None], MetricValue]] = {}
    missing = [str(path) for _, path in splits if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing retrieval summaries: " + ", ".join(missing))
    for split_name, path in splits:
        loaded[split_name] = _load_summary(path, method)
    return loaded


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


def _plot_metric(
    data: dict[str, dict[tuple[str, str, int | None], MetricValue]],
    *,
    direction: str,
    metric: str,
    splits: tuple[tuple[str, Path], ...],
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    split_names = [name for name, _ in splits]
    colors = plt.get_cmap("Set2").colors[: len(split_names)]

    if metric == "mrr":
        x_labels = ["MRR"]
        top_k_keys: list[int | None] = [None]
    else:
        x_labels = [str(k) for k in TOP_K_VALUES]
        top_k_keys = list(TOP_K_VALUES)

    x = np.arange(len(x_labels), dtype=float)
    width = min(0.18, 0.78 / len(split_names))
    offsets = (np.arange(len(split_names), dtype=float) - (len(split_names) - 1) / 2.0) * width

    fig, ax = plt.subplots()
    for split_idx, split_name in enumerate(split_names):
        values: list[float] = []
        errors: list[float] = []
        for top_k in top_k_keys:
            metric_value = data[split_name].get((direction, metric, top_k))
            if metric_value is None:
                values.append(np.nan)
                errors.append(0.0)
            else:
                values.append(metric_value.mean)
                errors.append(metric_value.std)
        ax.bar(
            x + offsets[split_idx],
            values,
            yerr=errors,
            capsize=2.5,
            width=width,
            color=colors[split_idx],
            edgecolor="white",
            linewidth=0.6,
            label=split_name,
        )

    ax.set_title(DIRECTIONS[direction])
    ax.set_ylabel(METRICS[metric])
    ax.set_xlabel("Top-K" if metric != "mrr" else "")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(bottom=0.0)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
    ax.legend(title="Splits", loc="upper left", bbox_to_anchor=(1.01, 1.0), ncols=1)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

    metric_slug = metric.replace("_", "-")
    direction_slug = "bgc_to_np" if direction == "bgc_to_compound" else "np_to_bgc"
    paths: list[Path] = []
    for fmt in formats:
        path = outdir / f"retrieval_{direction_slug}_{metric_slug}.{fmt}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def _write_values_csv(
    data: dict[str, dict[tuple[str, str, int | None], MetricValue]],
    outdir: Path,
    splits: tuple[tuple[str, Path], ...],
) -> Path:
    path = outdir / "retrieval_split_metric_values.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "direction", "metric", "top_k", "value_mean", "value_std", "n"])
        for split_name in [name for name, _ in splits]:
            for direction in DIRECTIONS:
                for metric in METRICS:
                    top_k_values: tuple[int | None, ...] = (None,) if metric == "mrr" else TOP_K_VALUES
                    for top_k in top_k_values:
                        metric_value = data[split_name].get((direction, metric, top_k))
                        if metric_value is None:
                            continue
                        writer.writerow(
                            [
                                split_name,
                                direction,
                                metric,
                                "" if top_k is None else top_k,
                                metric_value.mean,
                                metric_value.std,
                                metric_value.n,
                            ]
                        )
    return path


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _set_style()

    splits = _parse_splits(args.split)
    data = _load_all(args.method, splits)
    values_csv = _write_values_csv(data, args.outdir, splits)
    plot_paths: list[Path] = []
    for direction in DIRECTIONS:
        for metric in METRICS:
            plot_paths.extend(
                _plot_metric(
                    data,
                    direction=direction,
                    metric=metric,
                    splits=splits,
                    outdir=args.outdir,
                    formats=list(args.formats),
                    dpi=int(args.dpi),
                )
            )

    print(f"Wrote values CSV: {values_csv}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
