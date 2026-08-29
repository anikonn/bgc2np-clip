from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY = Path("results/best_esm_domains_molformer_strict_cv10/summary.json")
BASELINE = Path(
    "results/best_esm_domains_molformer_strict_cv10/baselines/downstream_encoder/"
    "downstream_encoder_baselines_summary.csv"
)
OUTDIR = Path("results/paper_plots/best_esm_domains_molformer/downstream/model_vs_baseline")
TASKS = (
    ("bgc_class", "BGC class", "classification", "BGC", "macro_auroc", "AUROC"),
    ("bioactivity_class", "Bioactivity", "classification", "BGC", "macro_auroc", "AUROC"),
    ("compound_logp", "logP", "regression", "NP", "pearson", "Pearson"),
    ("compound_mw", "Molecular weight", "regression", "NP", "pearson", "Pearson"),
    ("compound_tpsa", "TPSA", "regression", "NP", "pearson", "Pearson"),
    ("npclassifier_class", "NPClassifier class", "classification", "NP", "macro_auroc", "AUROC"),
    ("npclassifier_pathway", "NPClassifier pathway", "classification", "NP", "macro_auroc", "AUROC"),
    ("npclassifier_superclass", "NPClassifier superclass", "classification", "NP", "macro_auroc", "AUROC"),
    ("origin_type", "Origin type", "classification", "NP", "macro_f1", "F1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict CV10 downstream model-vs-baseline table.")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downstream = json.loads(args.summary.read_text(encoding="utf-8"))["aggregate"]["downstream"]
    baseline = pd.read_csv(args.baseline)
    rows: list[dict[str, object]] = []
    display_rows: list[list[str]] = []
    for task_key, task, task_type, modality, metric_key, metric in TASKS:
        model_node = downstream[task_key]["test"]
        model_node = model_node.get("overall", model_node)[metric_key]
        baseline_row = baseline[
            (baseline["eval_split"] == "test")
            & (baseline["task_key"] == task_key)
            & (baseline["metric"] == metric_key)
        ].iloc[0]
        model_mean, model_std = float(model_node["mean"]), float(model_node["std"])
        baseline_mean, baseline_std = float(baseline_row["value"]), float(baseline_row["std"])
        improvement = model_mean - baseline_mean
        rows.append(
            {
                "task": task,
                "type": task_type,
                "modality": modality,
                "metric": metric,
                "model_mean": model_mean,
                "model_std": model_std,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "improvement": improvement,
            }
        )
        display_rows.append(
            [
                task,
                task_type,
                modality,
                metric,
                f"{model_mean:.2f} ± {model_std:.2f}",
                f"{baseline_mean:.2f} ± {baseline_std:.2f}",
                f"{improvement:+.2f}",
            ]
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.outdir / "strict_cv10_downstream_model_vs_baseline.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=display_rows,
        colLabels=["Task", "Type", "Modality", "Metric", "Model score", "Baseline score", "Improvement"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.25, 0.15, 0.09, 0.10, 0.15, 0.15, 0.11],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    for column in range(7):
        table[(0, column)].set_text_props(weight="bold")
        table[(0, column)].set_facecolor("#E8EEF5")
    fig.tight_layout()
    fig.savefig(args.outdir / "strict_cv10_downstream_model_vs_baseline.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.outdir / "strict_cv10_downstream_model_vs_baseline.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
