from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._bootstrap import ensure_src_path

ensure_src_path()

PROPERTY_TASKS = [
    ("compound_mw", "MW"),
    ("compound_logp", "logP"),
    ("compound_tpsa", "TPSA"),
]
SPLIT_LABELS = {
    "bgc": "BGC",
    "np": "NP",
    "combined": "Combined",
    "strict": "Strict",
}
DEFAULT_SUMMARIES = {
    "bgc": Path("results/ohe_bgc_cv10_val_selected/summary.json"),
    "np": Path("results/ohe_np_cv10_val_selected/summary.json"),
    "combined": Path("results/ohe_combined_cv10_val_selected/summary.json"),
    "strict": Path("results/ohe_strict_cv10_val_selected/summary.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot molecular property prediction metrics across CV split protocols.")
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help="Summary JSON for one split. Repeatable. Defaults to the OHE all-split rerun outputs.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/molecular_property_prediction"))
    parser.add_argument("--prefix", type=str, default="molecular_property_prediction")
    parser.add_argument("--split", choices=list(SPLIT_LABELS), action="append", default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_summary(summary: dict[str, Any], task: str, metric: str) -> tuple[float, float, int]:
    payload = summary.get("aggregate", {}).get("downstream", {}).get(task, {}).get("test", {}).get(metric, {})
    if isinstance(payload, dict):
        return float(payload.get("mean", 0.0)), float(payload.get("std", 0.0)), int(payload.get("n", 0))
    if isinstance(payload, (int, float)):
        return float(payload), 0.0, 1
    return 0.0, 0.0, 0


def _parse_summary_args(values: list[str], splits: list[str] | None) -> dict[str, Path]:
    if values:
        parsed: dict[str, Path] = {}
        for value in values:
            if "=" not in value:
                raise ValueError(f"Invalid --summary value '{value}'. Expected SPLIT=PATH.")
            split, raw_path = value.split("=", 1)
            split = split.strip().lower()
            if split not in SPLIT_LABELS:
                raise ValueError(f"Unknown split label '{split}'. Expected one of {sorted(SPLIT_LABELS)}.")
            parsed[split] = Path(raw_path)
    else:
        parsed = dict(DEFAULT_SUMMARIES)

    if splits is not None:
        split_set = {split.lower() for split in splits}
        parsed = {split: path for split, path in parsed.items() if split in split_set}
    return parsed


def build_molecular_property_metric_table(summary_paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, path in summary_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing summary for {split}: {path}")
        summary = _load_json(path)
        for task, property_label in PROPERTY_TASKS:
            for metric in ("pearson", "spearman"):
                mean, std, n = _metric_summary(summary, task, metric)
                rows.append(
                    {
                        "split": split,
                        "split_label": SPLIT_LABELS[split],
                        "task": task,
                        "property": property_label,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "n": n,
                        "summary_path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def plot_molecular_property_prediction(metric_df: pd.DataFrame, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    split_order = [split for split in ("bgc", "np", "combined", "strict") if split in set(metric_df["split"])]
    property_order = [label for _, label in PROPERTY_TASKS]
    colors = {"MW": "#4C78A8", "logP": "#F58518", "TPSA": "#54A24B"}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    fig.suptitle("Molecular property prediction", fontsize=14)

    x = np.arange(len(split_order), dtype=np.float64)
    width = 0.24
    offsets = np.linspace(-width, width, num=len(property_order))

    for ax, metric in zip(axes, ("pearson", "spearman"), strict=True):
        metric_rows = metric_df[metric_df["metric"] == metric]
        for offset, property_label in zip(offsets, property_order, strict=True):
            values = []
            errors = []
            for split in split_order:
                row = metric_rows[(metric_rows["split"] == split) & (metric_rows["property"] == property_label)]
                if row.empty:
                    values.append(0.0)
                    errors.append(0.0)
                    continue
                values.append(float(row.iloc[0]["mean"]))
                errors.append(float(row.iloc[0]["std"]))
            ax.bar(
                x + offset,
                values,
                width=width,
                yerr=errors,
                capsize=3,
                label=property_label,
                color=colors[property_label],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_title(metric.title())
        ax.set_xlabel("Split")
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABELS[split] for split in split_order])
        ax.set_ylim(-1.0, 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axes[0].set_ylabel("Correlation")
    axes[1].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    summary_paths = _parse_summary_args(args.summary, args.split)
    metric_df = build_molecular_property_metric_table(summary_paths)
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / f"{args.prefix}.csv"
    metric_df.to_csv(csv_path, index=False)
    png_path = plot_molecular_property_prediction(metric_df, args.outdir / f"{args.prefix}.png")
    manifest = {
        "metric_table": str(csv_path),
        "plot": str(png_path),
        "summaries": {split: str(path) for split, path in summary_paths.items()},
    }
    (args.outdir / f"{args.prefix}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
