from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

TOP_K = (10, 100)
METHOD_ORDER = (
    "random",
    "frozen_encoder_similarity",
    "knn5",
    "bccoe_paper",
    "model",
)
METHOD_LABELS = {
    "random": "Random",
    "frozen_encoder_similarity": "Frozen",
    "knn5": "KNN-5",
    "bccoe_paper": "BCCoE (paper)",
    "model": "BGC2NP-CLIP",
}


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    summary_path: Path
    direction: str


DEFAULT_EXPERIMENTS = (
    Experiment("exp1_bgc_cv", "Exp 1 CV BGC->NP", Path("results/bccoe_bgc_cv10/summary.json"), "bgc_to_compound"),
    Experiment("exp2_np_cv", "Exp 2 CV NP->BGC", Path("results/bccoe_np_cv10/summary.json"), "compound_to_bgc"),
    Experiment(
        "exp3_loco_bgc",
        "Exp 3 LOCO BGC->NP",
        Path("results/bccoe_loco_exp3_bgc/summary.json"),
        "bgc_to_compound",
    ),
    Experiment(
        "exp4_loco_np",
        "Exp 4 LOCO NP->BGC",
        Path("results/bccoe_loco_exp4_np/summary.json"),
        "compound_to_bgc",
    ),
)


PAPER_BCCOE = {
    ("exp1_bgc_cv", 10): (1125.0, 0.329),
    ("exp1_bgc_cv", 100): (2243.0, 0.655),
    ("exp2_np_cv", 10): (2235.0, 0.653),
    ("exp2_np_cv", 100): (2864.0, 0.837),
    ("exp3_loco_bgc", 10): (253.0, 0.059),
    ("exp3_loco_bgc", 100): (1063.0, 0.246),
    ("exp4_loco_np", 10): (333.0, 0.075),
    ("exp4_loco_np", 100): (1333.0, 0.301),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BCCoE-style recall comparison tables.")
    parser.add_argument("--outdir", type=Path, default=Path("results/bccoe_comparison_table"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_block(fold: dict[str, Any], method: str, direction: str) -> dict[str, Any] | None:
    if method == "model":
        payload = fold.get("retrieval_test", {}).get(direction)
        return payload if isinstance(payload, dict) else None

    baselines = fold.get("retrieval_baselines_test", {})
    if not isinstance(baselines, dict):
        return None
    if method in {"random", "frozen_encoder_similarity"}:
        payload = baselines.get(method)
        metrics = payload.get("metrics", {}).get(direction) if isinstance(payload, dict) else None
        return metrics if isinstance(metrics, dict) else None
    if method == "knn5":
        payload = baselines.get("knn_transfer")
        metrics = payload.get("metrics_by_k", {}).get("5", {}).get(direction) if isinstance(payload, dict) else None
        return metrics if isinstance(metrics, dict) else None
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("mean", "value"):
            if key in value:
                return _as_float(value[key])
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mean_recall(summary: dict[str, Any], method: str, direction: str, top_k: int) -> float:
    recalls: list[float] = []
    for fold in summary.get("folds", []):
        metrics = _metric_block(fold, method, direction)
        if not metrics:
            continue
        recall = _as_float(metrics.get(f"recall_at_{top_k}"))
        if recall is None:
            continue
        recalls.append(recall)
    if not recalls:
        return float("nan")
    return sum(recalls) / len(recalls)


def _format_recall(value: float) -> str:
    if value != value:
        return ""
    return f"{100.0 * value:.1f}%"


def _make_long_rows(experiments: tuple[Experiment, ...]) -> list[dict[str, str]]:
    summaries = {exp.key: _load_json(exp.summary_path) for exp in experiments}
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        for exp in experiments:
            for top_k in TOP_K:
                if method == "bccoe_paper":
                    _paper_count, recall = PAPER_BCCOE[(exp.key, top_k)]
                else:
                    recall = _mean_recall(summaries[exp.key], method, exp.direction, top_k)
                rows.append(
                    {
                        "method": METHOD_LABELS[method],
                        "experiment": exp.label,
                        "top_k": str(top_k),
                        "recall": _format_recall(recall),
                    }
                )
    return rows


def _make_wide_rows(long_rows: list[dict[str, str]], experiments: tuple[Experiment, ...]) -> list[dict[str, str]]:
    by_key = {
        (row["method"], row["experiment"], row["top_k"]): row
        for row in long_rows
    }
    rows: list[dict[str, str]] = []
    for method in (METHOD_LABELS[key] for key in METHOD_ORDER):
        row = {"method": method}
        for exp in experiments:
            for top_k in TOP_K:
                source = by_key[(method, exp.label, str(top_k))]
                prefix = f"{exp.label} top-{top_k}"
                row[f"{prefix} Recall"] = source["recall"]
        rows.append(row)
    return rows


def _make_compact_rows(long_rows: list[dict[str, str]], experiments: tuple[Experiment, ...]) -> list[dict[str, str]]:
    by_key = {
        (row["method"], row["experiment"], row["top_k"]): row
        for row in long_rows
    }
    rows: list[dict[str, str]] = []
    for method in (METHOD_LABELS[key] for key in METHOD_ORDER):
        row = {"method": method}
        for exp in experiments:
            for top_k in TOP_K:
                source = by_key[(method, exp.label, str(top_k))]
                row[f"{exp.label} top-{top_k}"] = source["recall"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    return "\n".join(lines) + "\n"


def _write_png_table(path: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    columns = list(rows[0].keys())
    cell_text = [[row[column] for column in columns] for row in rows]
    fig_width = max(12.0, 1.55 * len(columns))
    fig_height = 1.3 + 0.42 * len(rows)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    missing = [str(exp.summary_path) for exp in DEFAULT_EXPERIMENTS if not exp.summary_path.exists()]
    if missing:
        raise FileNotFoundError("Missing summary file(s): " + ", ".join(missing))

    long_rows = _make_long_rows(DEFAULT_EXPERIMENTS)
    wide_rows = _make_wide_rows(long_rows, DEFAULT_EXPERIMENTS)
    compact_rows = _make_compact_rows(long_rows, DEFAULT_EXPERIMENTS)
    _write_csv(args.outdir / "bccoe_recall_long.csv", long_rows)
    _write_csv(args.outdir / "bccoe_recall_by_method.csv", wide_rows)
    _write_csv(args.outdir / "bccoe_recall_compact.csv", compact_rows)
    (args.outdir / "bccoe_recall_by_method.md").write_text(_markdown_table(wide_rows), encoding="utf-8")
    (args.outdir / "bccoe_recall_long.md").write_text(_markdown_table(long_rows), encoding="utf-8")
    (args.outdir / "bccoe_recall_compact.md").write_text(_markdown_table(compact_rows), encoding="utf-8")
    _write_png_table(args.outdir / "bccoe_recall_compact.png", compact_rows)


if __name__ == "__main__":
    main()
