from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path("results/best_esm_domains_molformer_bgcmap/ensemble_test_retrieval.json")
OUTDIR = Path("results/paper_plots/best_esm_domains_molformer/benchmarks/bgcmap/comparison")
CLASSES = (
    ("NRP", "NRPS"),
    ("Other", "other"),
    ("Polyketide", "PKS"),
    ("RiPP", "ribosomal"),
    ("Saccharide", "saccharide"),
    ("Terpene", "terpene"),
)
PUBLISHED_BGCMAP = (0.873, 0.825, 0.839, 0.923, 0.784, 0.826)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BGC-MAP per-class AUROC comparison table.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = json.loads(args.results.read_text(encoding="utf-8"))["bgc_class_pair_retrieval"]["classes"]
    ours = tuple(float(classes[source]["auroc"]) for _, source in CLASSES)
    columns = [display for display, _ in CLASSES]
    table = pd.DataFrame(
        [
            {"model": "BGC-MAP", **dict(zip(columns, PUBLISHED_BGCMAP, strict=True))},
            {"model": "BGC2NP-CLIP", **dict(zip(columns, ours, strict=True))},
        ]
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "bgcmap_per_class_auroc_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.4, 2.25))
    ax.axis("off")
    rendered = ax.table(
        cellText=[
            ["BGC-MAP", *[f"{value:.3f}" for value in PUBLISHED_BGCMAP]],
            ["BGC2NP-CLIP", *[f"{value:.3f}" for value in ours]],
        ],
        colLabels=["Model", *columns],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.20, 0.11, 0.11, 0.16, 0.11, 0.16, 0.12],
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(11)
    rendered.scale(1, 1.55)
    rendered[(1, 0)].set_text_props(ha="left")
    rendered[(2, 0)].set_text_props(ha="left")
    for column_index, (published, model) in enumerate(zip(PUBLISHED_BGCMAP, ours, strict=True), start=1):
        winner_row = 1 if published >= model else 2
        rendered[(winner_row, column_index)].set_text_props(weight="bold")
    for key, cell in rendered.get_celld().items():
        cell.set_edgecolor("white")
        if key[0] == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("BGC class", fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "bgcmap_per_class_auroc_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.outdir / "bgcmap_per_class_auroc_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
