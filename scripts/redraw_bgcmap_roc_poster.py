#!/usr/bin/env python3
"""Redraw BGC-MAP benchmark ROC curves in a compact poster/paper style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


CLASS_ORDER = ["NRPS", "other", "PKS", "ribosomal", "saccharide", "terpene"]
DISPLAY = {
    "NRPS": "NRP",
    "other": "other",
    "PKS": "Polyketide",
    "ribosomal": "RiPP",
    "saccharide": "Saccharide",
    "terpene": "Terpene",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval_json",
        type=Path,
        default=Path("results/paper_plots/final_results_t33/benchmarks/bgcmap/ensemble_test_retrieval.json"),
    )
    parser.add_argument(
        "--out_png",
        type=Path,
        default=Path("results/paper_plots/final_results_t33/benchmarks/bgcmap/ensemble_test_bgc_class_retrieval_roc.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.retrieval_json.read_text(encoding="utf-8"))
    classes = report["bgc_class_pair_retrieval"]["classes"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 24,
            "axes.titlesize": 31,
            "axes.labelsize": 31,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 22,
            "axes.linewidth": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(12.2, 9.4))
    legend_rows: list[tuple[str, str, str]] = []
    for key in CLASS_ORDER:
        metrics = classes[key]
        curve = metrics["roc_curve"]
        auc = float(metrics["auroc"])
        line, = ax.plot(curve["fpr"], curve["tpr"], linewidth=3.0)
        legend_rows.append((DISPLAY[key], f"{auc:.3f}", line.get_color()))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#9E9E9E", linewidth=3.0, alpha=0.85)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", labelpad=8)
    ax.set_ylabel("True Positive Rate", labelpad=8)
    ax.grid(True, color="#BDBDBD", linewidth=1.5, alpha=0.42)

    # Plain SVG-safe title: no mathtext/LaTeX, so vector editors do not choke on it.
    fig.text(
        0.44,
        0.972,
        "BGC2NP-CLIP:",
        ha="right",
        va="top",
        fontsize=31,
        fontweight="bold",
    )
    fig.text(
        0.44,
        0.972,
        " ROC Curve for BGC-product Matching",
        ha="left",
        va="top",
        fontsize=31,
    )

    # Manual legend, also SVG-safe: AUC numbers are separate bold text objects.
    box = Rectangle(
        (0.42, 0.065),
        0.56,
        0.405,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        alpha=0.84,
        zorder=2,
    )
    ax.add_patch(box)
    y0 = 0.43
    dy = 0.069
    for row_idx, (display, auc_text, color) in enumerate(legend_rows):
        y = y0 - row_idx * dy
        ax.plot([0.455, 0.525], [y, y], transform=ax.transAxes, color=color, linewidth=3.0, clip_on=False, zorder=3)
        ax.text(0.545, y, f"{display} (AUC =", transform=ax.transAxes, ha="left", va="center", fontsize=20.5, zorder=3)
        ax.text(
            0.88,
            y,
            f"{auc_text})",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=20.5,
            fontweight="bold",
            zorder=3,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(2.4)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = args.out_png.with_suffix(".pdf")
    out_svg = args.out_png.with_suffix(".svg")
    for path in (args.out_png, out_pdf, out_svg):
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {args.out_png}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
