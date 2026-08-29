from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS = [
    (
        "OHE proteins + Morgan",
        "results/ohe_strict_cv10_val_selected/summary.json",
        21504,
        2048,
        "positional OHE; mean across proteins",
    ),
    (
        "ESM2 proteins + Morgan",
        "results/esm2_clipped_strict_cv10/summary.json",
        640,
        2048,
        "esm2_t30_150M CLS; mean across proteins",
    ),
    (
        "OHE domains + Morgan",
        "results/antismash_domain_ohe_strict_cv10/summary.json",
        21504,
        2048,
        "positional OHE; mean across domains/unsplit CDS",
    ),
    (
        "ESM2 domains + Morgan",
        "results/antismash_domain_esm2_strict_cv10/summary.json",
        1280,
        2048,
        "provided BGC-MAC ESM2; mean across domains/unsplit enzymes",
    ),
    (
        "OHE domains + MolFormer",
        "results/antismash_domain_ohe_molformer_strict_cv10/summary.json",
        21504,
        768,
        "antiSMASH-domain OHE + frozen MoLFormer pooled output",
    ),
    (
        "ESM2 domains + MolFormer",
        "results/antismash_domain_esm2_molformer_strict_cv10/summary.json",
        1280,
        768,
        "provided BGC-MAC ESM2 + frozen MoLFormer pooled output",
    ),
]
METRICS = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10")
DIRECTIONS = ("bgc_to_compound", "compound_to_bgc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the strict-CV10 four-encoder retrieval comparison table.")
    parser.add_argument("--outdir", type=Path, default=Path("results/strict_encoder_retrieval_comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for encoder, summary_path, input_dim, compound_dim, details in RUNS:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        retrieval = payload["aggregate"]["retrieval_test"]
        row: dict[str, object] = {
            "encoder": encoder,
            "input_dim": input_dim,
            "compound_input_dim": compound_dim,
            "details": details,
            "n_folds": 10,
        }
        for direction in DIRECTIONS:
            prefix = "bgc_to_np" if direction == "bgc_to_compound" else "np_to_bgc"
            for metric in METRICS:
                values = retrieval[direction][metric]
                row[f"{prefix}_{metric}_mean"] = float(values["mean"])
                row[f"{prefix}_{metric}_std"] = float(values["std"])
        rows.append(row)

    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "strict_cv10_encoder_retrieval_comparison.csv", index=False)

    display_columns = [
        ("encoder", "Encoder"),
        ("bgc_to_np_mrr", "BGC→NP\nMRR"),
        ("bgc_to_np_recall_at_1", "BGC→NP\nR@1"),
        ("bgc_to_np_recall_at_5", "BGC→NP\nR@5"),
        ("bgc_to_np_recall_at_10", "BGC→NP\nR@10"),
        ("np_to_bgc_mrr", "NP→BGC\nMRR"),
        ("np_to_bgc_recall_at_1", "NP→BGC\nR@1"),
        ("np_to_bgc_recall_at_5", "NP→BGC\nR@5"),
        ("np_to_bgc_recall_at_10", "NP→BGC\nR@10"),
    ]
    display_rows: list[list[str]] = []
    for row in rows:
        values = [str(row["encoder"])]
        for stem, _ in display_columns[1:]:
            values.append(f"{float(row[f'{stem}_mean']):.3f} ± {float(row[f'{stem}_std']):.3f}")
        display_rows.append(values)

    fig, ax = plt.subplots(figsize=(16, 3.4))
    ax.axis("off")
    rendered = ax.table(
        cellText=display_rows,
        colLabels=[label for _, label in display_columns],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.18] + [0.1025] * 8,
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.7)
    for row_index in range(1, len(display_rows) + 1):
        rendered[(row_index, 0)].set_text_props(ha="left")
    for col_index in range(len(display_columns)):
        rendered[(0, col_index)].set_facecolor("#DCE6F1")
        rendered[(0, col_index)].set_text_props(weight="bold")
    ax.set_title("Strict CV10 retrieval comparison (mean ± SD across folds)", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(args.outdir / "strict_cv10_encoder_retrieval_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
