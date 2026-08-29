from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("results/intermediate/projection_head_ablation")
OUT = Path("results/paper_plots/projection_head_ablation")


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(ROOT.glob("*/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = data["summary"]
        row: dict[str, object] = {
            "run_name": data["run_name"],
            "head": data["head"],
            "lr": float(data["lr"]),
            "n_folds": int(summary["n_folds"]),
            "mean_bidirectional_mrr_mean": float(summary["mean_bidirectional_mrr"]["mean"]),
            "mean_bidirectional_mrr_std": float(summary["mean_bidirectional_mrr"]["std"]),
        }
        for key in (
            "bgc_to_np_mrr",
            "bgc_to_np_recall_at_1",
            "bgc_to_np_recall_at_5",
            "bgc_to_np_recall_at_10",
            "np_to_bgc_mrr",
            "np_to_bgc_recall_at_1",
            "np_to_bgc_recall_at_5",
            "np_to_bgc_recall_at_10",
        ):
            row[f"{key}_mean"] = float(summary[key]["mean"])
            row[f"{key}_std"] = float(summary[key]["std"])
            row[key] = _fmt(float(summary[key]["mean"]), float(summary[key]["std"]))
        row["mean_bidirectional_mrr"] = _fmt(
            float(summary["mean_bidirectional_mrr"]["mean"]),
            float(summary["mean_bidirectional_mrr"]["std"]),
        )
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No summary.json files found under {ROOT}")

    table = pd.DataFrame(rows).sort_values(["mean_bidirectional_mrr_mean", "bgc_to_np_mrr_mean"], ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "projection_head_ablation_strict_folds_1_2_3.csv", index=False)

    short_cols = [
        "head",
        "lr",
        "n_folds",
        "mean_bidirectional_mrr",
        "bgc_to_np_mrr",
        "bgc_to_np_recall_at_1",
        "bgc_to_np_recall_at_10",
        "np_to_bgc_mrr",
        "np_to_bgc_recall_at_1",
        "np_to_bgc_recall_at_10",
    ]
    table[short_cols].to_csv(OUT / "projection_head_ablation_strict_folds_1_2_3_short.csv", index=False)
    print(table[short_cols].to_string(index=False))


if __name__ == "__main__":
    main()
