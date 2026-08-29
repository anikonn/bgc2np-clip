from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("results/intermediate/hparam_search_stage1")
OUT = Path("results/paper_plots/hparam_search")


def _cfg(data: dict[str, Any], key: str) -> Any:
    cfg = data["config"]
    if key in {"lr", "weight_decay", "batch_size", "scheduler", "warmup_fraction"}:
        return cfg["train"][key]
    if key == "temperature":
        return cfg["model"]["init_temperature"]
    return cfg["model"][key]


def main() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(ROOT.glob("trial_*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data["summary"]
        row = {
            "trial_id": data["trial_id"],
            "selection_score_val_bgc_to_np_recall_at_10": float(summary["selection_score"]),
            "n_folds": int(summary["n_folds"]),
        }
        for key in (
            "lr",
            "weight_decay",
            "dropout",
            "batch_size",
            "hidden_dim",
            "emb_dim",
            "temperature",
            "scheduler",
            "warmup_fraction",
        ):
            row[key] = _cfg(data, key)
        for key, value in summary.items():
            if key != "n_folds":
                row[key] = value
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No trial summaries found under {ROOT}")

    table = pd.DataFrame(rows).sort_values(
        ["selection_score_val_bgc_to_np_recall_at_10", "val_bidirectional_recall_at_10_mean"],
        ascending=False,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "hparam_search_stage1_all_trials.csv", index=False)
    short_cols = [
        "trial_id",
        "selection_score_val_bgc_to_np_recall_at_10",
        "val_bgc_to_np_recall_at_10_mean",
        "val_bgc_to_np_mrr_mean",
        "test_bgc_to_np_recall_at_10_mean",
        "test_bgc_to_np_mrr_mean",
        "lr",
        "weight_decay",
        "dropout",
        "batch_size",
        "hidden_dim",
        "emb_dim",
        "temperature",
        "scheduler",
        "warmup_fraction",
    ]
    table[short_cols].to_csv(OUT / "hparam_search_stage1_ranked_short.csv", index=False)
    best = table.iloc[0].to_dict()
    Path(OUT / "best_hparams_stage1.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(table[short_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
