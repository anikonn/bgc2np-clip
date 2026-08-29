from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("results/esm30_vs_esm33_ablation")
OUTDIR = Path("results/paper_plots/esm30_vs_esm33_ablation")


def _metric(summary: dict, path: str) -> tuple[float, float]:
    cur = summary
    for part in path.split("."):
        cur = cur[part]
    return float(cur["mean"]), float(cur["std"])


def _to_markdown_simple(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    specs = [
        ("ESM2-t30 / 150M / 640d", Path("results/final_results/cv/strict_cv10/summary.json")),
        ("ESM2-t33 / 650M / 1280d", ROOT / "esm2_t33_strict_cv10_retrieval_only" / "summary.json"),
    ]
    rows = []
    metric_paths = {
        "bgc_to_np_recall_at_10": "contrastive_metrics.retrieval_test.bgc_to_compound.recall_at_10",
        "bgc_to_np_mrr": "contrastive_metrics.retrieval_test.bgc_to_compound.mrr",
        "np_to_bgc_recall_at_10": "contrastive_metrics.retrieval_test.compound_to_bgc.recall_at_10",
        "np_to_bgc_mrr": "contrastive_metrics.retrieval_test.compound_to_bgc.mrr",
    }
    for model, path in specs:
        if not path.exists():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "model": model,
            "split": "strict",
            "folds": ",".join(str(x) for x in summary.get("fold_ids", list(range(1, int(summary.get("n_folds", 10)) + 1)))),
            "n_run_folds": int(summary.get("n_run_folds", summary.get("n_folds", 10))),
            "cache_dir": summary.get("cache_dir"),
        }
        for name, metric_path in metric_paths.items():
            mean, std = _metric(summary["aggregate"], metric_path)
            row[name] = mean
            row[f"{name}_std"] = std
            row[f"{name}_pretty"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 2:
        base = df.iloc[0]
        for idx in range(1, len(df)):
            for name in metric_paths:
                df.loc[idx, f"{name}_delta_vs_t30"] = float(df.loc[idx, name]) - float(base[name])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_csv = ROOT / "esm33_vs_t30_strict_cv10.csv"
    paper_csv = OUTDIR / "esm33_vs_t30_strict_cv10.csv"
    df.to_csv(out_csv, index=False)
    df.to_csv(paper_csv, index=False)
    display_cols = [
        "model",
        "split",
        "folds",
        "bgc_to_np_recall_at_10_pretty",
        "bgc_to_np_mrr_pretty",
        "np_to_bgc_recall_at_10_pretty",
        "np_to_bgc_mrr_pretty",
    ]
    if "bgc_to_np_recall_at_10_delta_vs_t30" in df.columns:
        display_cols.append("bgc_to_np_recall_at_10_delta_vs_t30")
    (OUTDIR / "esm33_vs_t30_strict_cv10.md").write_text(
        _to_markdown_simple(df[display_cols].fillna("")),
        encoding="utf-8",
    )
    print(df.to_string(index=False))
    print(f"Saved: {out_csv}")
    print(f"Saved: {paper_csv}")


if __name__ == "__main__":
    main()
