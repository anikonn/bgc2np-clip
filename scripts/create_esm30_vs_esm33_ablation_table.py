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
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = []
    specs = [
        ("ESM2-t30 / 150M / 640d", ROOT / "esm2_t30_strict_folds_1_3" / "summary.json"),
        ("ESM2-t33 / 650M / 1280d", ROOT / "esm2_t33_strict_folds_1_3" / "summary.json"),
    ]
    for model, path in specs:
        if not path.exists():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "model": model,
            "split": "strict",
            "folds": ",".join(str(x) for x in summary.get("fold_ids", [])),
            "cache_dir": summary.get("cache_dir"),
        }
        metric_paths = {
            "bgc_to_np_recall_at_10": "contrastive_metrics.retrieval_test.bgc_to_compound.recall_at_10",
            "bgc_to_np_mrr": "contrastive_metrics.retrieval_test.bgc_to_compound.mrr",
            "np_to_bgc_recall_at_10": "contrastive_metrics.retrieval_test.compound_to_bgc.recall_at_10",
            "np_to_bgc_mrr": "contrastive_metrics.retrieval_test.compound_to_bgc.mrr",
        }
        for name, metric_path in metric_paths.items():
            mean, std = _metric(summary["aggregate"], metric_path)
            row[name] = mean
            row[f"{name}_std"] = std
            row[f"{name}_pretty"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "esm30_vs_esm33_strict_folds_1_3.csv", index=False)
    df.to_csv(OUTDIR / "esm30_vs_esm33_strict_folds_1_3.csv", index=False)
    markdown_df = df[
            [
                "model",
                "split",
                "folds",
                "bgc_to_np_recall_at_10_pretty",
                "bgc_to_np_mrr_pretty",
                "np_to_bgc_recall_at_10_pretty",
                "np_to_bgc_mrr_pretty",
            ]
        ]
    (OUTDIR / "esm30_vs_esm33_strict_folds_1_3.md").write_text(
        _to_markdown_simple(markdown_df),
        encoding="utf-8",
    )
    print(df.to_string(index=False))
    print(f"Saved: {ROOT / 'esm30_vs_esm33_strict_folds_1_3.csv'}")
    print(f"Saved: {OUTDIR / 'esm30_vs_esm33_strict_folds_1_3.csv'}")


if __name__ == "__main__":
    main()
