from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-combi-poster-panel")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/combi-cache-poster-panel")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLIT_ORDER = ("BGC", "NP", "Combined", "Strict")
SPLIT_NORMALIZE = {"BGC": "BGC", "NP": "NP", "combined": "Combined", "strict": "Strict"}
LEAKAGE_TYPES = (
    ("BGC", "BGC"),
    ("NP", "NP"),
    ("BGC_family", "BGC family"),
    ("NP_cluster", "NP cluster"),
)
DIRECTIONS = (
    ("bgc_to_compound", "BGC → NP"),
    ("compound_to_bgc", "NP → BGC"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poster panel: split leakage and Recall@10 comparison.")
    parser.add_argument(
        "--leakage_csv",
        type=Path,
        default=Path("results/split_leakage/cv_split_leakage_comparison.csv"),
    )
    parser.add_argument(
        "--retrieval_csv",
        type=Path,
        default=Path("results/paper_plots/final_results_t33/retrieval/split_comparison/retrieval_split_metric_values.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/paper_plots/final_results_t33/poster_panels"),
    )
    parser.add_argument("--prefix", default="leakage_and_recall10_panel")
    return parser.parse_args()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _load_leakage(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["split_label"] = df["split"].map(SPLIT_NORMALIZE).fillna(df["split"].astype(str))
    df = df[df["split_label"].isin(SPLIT_ORDER) & df["entity"].isin([key for key, _ in LEAKAGE_TYPES])]
    if df.empty:
        raise ValueError(f"No usable leakage rows in {path}")
    return df


def _load_recall10(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df = df[
        (df["split"].isin(SPLIT_ORDER))
        & (df["metric"] == "recall")
        & (df["top_k"].astype(float) == 10.0)
        & (df["direction"].isin([key for key, _ in DIRECTIONS]))
    ].copy()
    if df.empty:
        raise ValueError(f"No Recall@10 rows in {path}")
    return df


def _plot_grouped_bars(
    ax: plt.Axes,
    *,
    x_labels: list[str],
    values_by_split: dict[str, list[float]],
    errors_by_split: dict[str, list[float]],
    colors,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    width_scale: float = 1.0,
) -> None:
    x = np.arange(len(x_labels), dtype=float)
    width = min(0.18, 0.76 / len(SPLIT_ORDER)) * float(width_scale)
    offsets = (np.arange(len(SPLIT_ORDER), dtype=float) - (len(SPLIT_ORDER) - 1) / 2.0) * width
    for idx, split in enumerate(SPLIT_ORDER):
        ax.bar(
            x + offsets[idx],
            values_by_split[split],
            yerr=errors_by_split.get(split),
            capsize=2.4,
            width=width,
            color=colors[idx],
            edgecolor="white",
            linewidth=0.7,
            label=split,
        )
    ax.set_title(title, fontsize=15, pad=9)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylim(ylim if ylim is not None else (0, None))
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")


def main() -> None:
    args = parse_args()
    _set_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    leakage = _load_leakage(args.leakage_csv)
    retrieval = _load_recall10(args.retrieval_csv)
    colors = plt.get_cmap("Set2").colors[: len(SPLIT_ORDER)]

    leakage_values: dict[str, list[float]] = {split: [] for split in SPLIT_ORDER}
    leakage_errors: dict[str, list[float]] = {split: [] for split in SPLIT_ORDER}
    for entity, _label in LEAKAGE_TYPES:
        for split in SPLIT_ORDER:
            row = leakage[(leakage["split_label"] == split) & (leakage["entity"] == entity)]
            if row.empty:
                leakage_values[split].append(np.nan)
                leakage_errors[split].append(0.0)
            else:
                leakage_values[split].append(float(row.iloc[0]["mean_leakage_percent"]))
                leakage_errors[split].append(float(row.iloc[0]["std_leakage_percent"]))

    recall_values: dict[str, list[float]] = {split: [] for split in SPLIT_ORDER}
    recall_errors: dict[str, list[float]] = {split: [] for split in SPLIT_ORDER}
    for direction, _label in DIRECTIONS:
        for split in SPLIT_ORDER:
            row = retrieval[(retrieval["split"] == split) & (retrieval["direction"] == direction)]
            if row.empty:
                recall_values[split].append(np.nan)
                recall_errors[split].append(0.0)
            else:
                recall_values[split].append(100.0 * float(row.iloc[0]["value_mean"]))
                recall_errors[split].append(100.0 * float(row.iloc[0]["value_std"]))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.9, 4.25),
        gridspec_kw={"width_ratios": (0.82, 0.70), "wspace": 0.18},
    )
    _plot_grouped_bars(
        axes[0],
        x_labels=[label for _key, label in LEAKAGE_TYPES],
        values_by_split=leakage_values,
        errors_by_split=leakage_errors,
        colors=colors,
        ylabel="Train/test leakage (%)",
        title="Split leakage",
        ylim=(0, 90),
        width_scale=1.08,
    )
    _plot_grouped_bars(
        axes[1],
        x_labels=[label for _key, label in DIRECTIONS],
        values_by_split=recall_values,
        errors_by_split=recall_errors,
        colors=colors,
        ylabel="Recall@10 (%)",
        title="Retrieval results",
        ylim=(0, 90),
        width_scale=0.64,
    )
    for ax in axes:
        ax.set_yticks(np.arange(0, 91, 20))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Split",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        fontsize=12,
        title_fontsize=12,
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.075, right=0.99)

    leakage_out = pd.DataFrame(
        [
            {
                "panel": "leakage",
                "split": split,
                "leakage_type": label,
                "mean_percent": leakage_values[split][idx],
                "std_percent": leakage_errors[split][idx],
            }
            for idx, (_entity, label) in enumerate(LEAKAGE_TYPES)
            for split in SPLIT_ORDER
        ]
    )
    recall_out = pd.DataFrame(
        [
            {
                "panel": "recall10",
                "split": split,
                "direction": label,
                "mean_percent": recall_values[split][idx],
                "std_percent": recall_errors[split][idx],
            }
            for idx, (_direction, label) in enumerate(DIRECTIONS)
            for split in SPLIT_ORDER
        ]
    )
    pd.concat([leakage_out, recall_out], ignore_index=True).to_csv(
        args.outdir / f"{args.prefix}_values.csv",
        index=False,
    )
    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.outdir / f"{args.prefix}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.outdir / (args.prefix + '.pdf')}")
    print(f"Saved {args.outdir / (args.prefix + '.png')}")
    print(f"Saved {args.outdir / (args.prefix + '.svg')}")


if __name__ == "__main__":
    main()
