from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SUMMARIES = {
    "BGC": Path("results/ohe_bgc_cv10_val_selected/summary.json"),
    "NP": Path("results/ohe_np_cv10_val_selected/summary.json"),
    "combined": Path("results/ohe_combined_cv10_val_selected/summary.json"),
    "strict": Path("results/ohe_strict_cv10_val_selected/summary.json"),
}

TASK_INFO = {
    "bgc_class": {"task": "BGC class", "type": "classification", "modality": "BGC"},
    "bioactivity_class": {"task": "Bioactivity", "type": "classification", "modality": "BGC"},
    "npclassifier_pathway": {"task": "NPClassifier pathway", "type": "classification", "modality": "BGC"},
    "npclassifier_superclass": {"task": "NPClassifier superclass", "type": "classification", "modality": "BGC"},
    "npclassifier_class": {"task": "NPClassifier class", "type": "classification", "modality": "BGC"},
    "compound_mw": {"task": "Molecular weight", "type": "regression", "modality": "NP"},
    "compound_logp": {"task": "logP", "type": "regression", "modality": "NP"},
    "compound_tpsa": {"task": "TPSA", "type": "regression", "modality": "NP"},
    "origin_type": {"task": "Origin type", "type": "classification", "modality": "NP"},
}

METRICS_BY_TYPE = {
    "classification": (
        "macro_auroc",
        "micro_auroc",
        "macro_f1",
        "micro_f1",
        "accuracy",
        "hamming_accuracy",
        "subset_accuracy",
        "loss",
    ),
    "regression": ("pearson", "spearman", "rmse", "r2", "mse", "loss"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create flat downstream metric tables from CV summary.json files.")
    parser.add_argument("--outdir", type=Path, default=Path("results/downstream_metric_tables"))
    parser.add_argument(
        "--summary",
        action="append",
        default=None,
        metavar="SPLIT=PATH",
        help="Summary JSON to include. Repeat for multiple splits. Defaults to the four OHE CV summaries.",
    )
    parser.add_argument("--eval_split", choices=("test", "val"), default="test")
    return parser.parse_args()


def _parse_summary_args(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_SUMMARIES)
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --summary value '{value}'. Expected SPLIT=PATH.")
        split_name, raw_path = value.split("=", 1)
        parsed[split_name.strip()] = Path(raw_path)
    return parsed


def _metric_value(node: Any) -> tuple[float | None, float | None, int | None]:
    if isinstance(node, dict) and "mean" in node:
        value = node.get("mean")
        std = node.get("std")
        n = node.get("n")
        return (
            None if value is None else float(value),
            None if std is None else float(std),
            None if n is None else int(n),
        )
    if isinstance(node, int | float):
        return float(node), None, None
    return None, None, None


def _load_downstream_metrics(summary_path: Path, eval_split: str) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    downstream = summary.get("aggregate", {}).get("downstream", {})
    if not isinstance(downstream, dict):
        raise ValueError(f"{summary_path} does not contain aggregate.downstream metrics.")
    metrics: dict[str, Any] = {}
    for task_name, task_node in downstream.items():
        if not isinstance(task_node, dict) or task_name not in TASK_INFO:
            continue
        split_node = task_node.get(eval_split)
        if isinstance(split_node, dict):
            metrics[task_name] = split_node.get("overall", split_node)
    return metrics


def build_downstream_metric_long_table(
    summary_paths: dict[str, Path],
    *,
    eval_split: str = "test",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, summary_path in summary_paths.items():
        if not summary_path.exists():
            continue
        downstream = _load_downstream_metrics(summary_path, eval_split=eval_split)
        for task_key, task_metrics in downstream.items():
            info = TASK_INFO[task_key]
            for metric in METRICS_BY_TYPE[str(info["type"])]:
                if metric not in task_metrics:
                    continue
                value, std, n = _metric_value(task_metrics[metric])
                if value is None:
                    continue
                rows.append(
                    {
                        "split": split_name,
                        "eval_split": eval_split,
                        "task_key": task_key,
                        "task": info["task"],
                        "type": info["type"],
                        "modality": info["modality"],
                        "metric": metric,
                        "value": value,
                        "std": std,
                        "n_folds": n,
                        "summary_path": str(summary_path),
                    }
                )
    return pd.DataFrame(rows)


def build_downstream_per_class_table(
    summary_paths: dict[str, Path], *, eval_split: str = "test"
) -> pd.DataFrame:
    """Flatten already-computed classification per-class metrics from CV summaries."""
    rows: list[dict[str, Any]] = []
    for split_name, summary_path in summary_paths.items():
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        downstream = summary.get("aggregate", {}).get("downstream", {})
        for task_key, task_node in downstream.items():
            if task_key not in TASK_INFO or not isinstance(task_node, dict):
                continue
            split_node = task_node.get(eval_split, {})
            per_class = split_node.get("per_class", {}) if isinstance(split_node, dict) else {}
            if not isinstance(per_class, dict):
                continue
            for class_name, metrics in per_class.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, node in metrics.items():
                    value, std, n = _metric_value(node)
                    if value is None:
                        continue
                    rows.append({
                        "split": split_name, "eval_split": eval_split, "task_key": task_key,
                        "task": TASK_INFO[task_key]["task"], "class": class_name, "metric": metric,
                        "value": value, "std": std, "n_folds": n, "summary_path": str(summary_path),
                    })
    return pd.DataFrame(rows)


def _wide_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if long_df.empty:
        return pd.DataFrame()
    group_cols = ["split", "eval_split", "task_key", "task", "type", "modality", "summary_path"]
    for keys, group in long_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys, strict=True))
        for idx, item in enumerate(group.itertuples(index=False), start=1):
            row[f"metric_{idx}"] = item.metric
            row[f"value_{idx}"] = item.value
            row[f"std_{idx}"] = item.std
            row[f"n_folds_{idx}"] = item.n_folds
        rows.append(row)
    return pd.DataFrame(rows)


def _short_wide_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if long_df.empty:
        return pd.DataFrame()
    group_cols = ["split", "task", "type", "modality"]
    for keys, group in long_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys, strict=True))
        for idx, item in enumerate(group.itertuples(index=False), start=1):
            row[f"metric_{idx}"] = item.metric
            row[f"mean_{idx}"] = item.value
            row[f"std_{idx}"] = item.std
        rows.append(row)
    return pd.DataFrame(rows)


def write_downstream_metric_tables(
    long_df: pd.DataFrame, outdir: Path, per_class_df: pd.DataFrame | None = None
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    long_path = outdir / "downstream_metrics_long.csv"
    long_df.to_csv(long_path, index=False)
    outputs["long_all_splits"] = str(long_path)

    wide_df = _wide_rows(long_df)
    wide_path = outdir / "downstream_metrics_wide.csv"
    wide_df.to_csv(wide_path, index=False)
    outputs["wide_all_splits"] = str(wide_path)

    short_wide_df = _short_wide_rows(long_df)
    short_wide_path = outdir / "downstream_metrics_short_wide.csv"
    short_wide_df.to_csv(short_wide_path, index=False)
    outputs["short_wide_all_splits"] = str(short_wide_path)

    per_class_path = outdir / "downstream_metrics_per_class.csv"
    (per_class_df if per_class_df is not None else pd.DataFrame()).to_csv(per_class_path, index=False)
    outputs["per_class_all_splits"] = str(per_class_path)

    for split_name, split_long in long_df.groupby("split", sort=False):
        safe_split = str(split_name).lower().replace(" ", "_")
        split_long_path = outdir / f"{safe_split}_downstream_metrics_long.csv"
        split_wide_path = outdir / f"{safe_split}_downstream_metrics_wide.csv"
        split_short_wide_path = outdir / f"{safe_split}_downstream_metrics_short_wide.csv"
        split_long.to_csv(split_long_path, index=False)
        _wide_rows(split_long).to_csv(split_wide_path, index=False)
        _short_wide_rows(split_long).to_csv(split_short_wide_path, index=False)
        outputs[f"{safe_split}_long"] = str(split_long_path)
        outputs[f"{safe_split}_wide"] = str(split_wide_path)
        outputs[f"{safe_split}_short_wide"] = str(split_short_wide_path)
    return outputs


def main() -> None:
    args = parse_args()
    summary_paths = _parse_summary_args(args.summary)
    long_df = build_downstream_metric_long_table(summary_paths, eval_split=str(args.eval_split))
    per_class_df = build_downstream_per_class_table(summary_paths, eval_split=str(args.eval_split))
    outputs = write_downstream_metric_tables(long_df, args.outdir, per_class_df)
    manifest = {
        "eval_split": str(args.eval_split),
        "summaries": {name: str(path) for name, path in summary_paths.items()},
        "outputs": outputs,
        "notes": {
            "long": "One row per split, downstream task, and metric.",
            "wide": "One row per split and downstream task, with metric/value/std/n_folds groups.",
            "short_wide": "One row per split and downstream task, without paths or internal task keys.",
            "value": "CV aggregate mean from summary.json when available.",
            "std": "CV aggregate standard deviation from summary.json when available.",
        },
    }
    (args.outdir / "downstream_metric_tables_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
