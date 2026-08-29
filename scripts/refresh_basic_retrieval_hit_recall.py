from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from projects.mibig_bgc_np.scripts.plot_retrieval_summary import (
    build_retrieval_long,
    plot_class_retrieval,
    plot_mrr,
    plot_topk_hit,
    plot_topk_recall,
    summarize_retrieval,
)
from projects.mibig_bgc_np.scripts.run_cv10 import _build_summary


DEFAULT_RUNS = (
    Path("results/bgc_cv10"),
    Path("results/combined_cv10"),
    Path("results/cv10"),
    Path("results/np_cv10"),
    Path("results/strict_modal_cv10"),
)
TOP_K_VALUES = (1, 5, 10, 20, 50, 100, 200, 500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill basic retrieval summaries with Hit@K and pair-level Recall@K."
    )
    parser.add_argument("--runs", type=Path, nargs="*", default=list(DEFAULT_RUNS))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _update_direction_metrics(metrics: dict[str, Any], counts: dict[str, Any], direction: str) -> bool:
    n_pairs = _as_float(counts.get("n_pairs"))
    n_queries_key = "n_bgcs" if direction == "bgc_to_compound" else "n_compounds"
    n_queries = _as_float(counts.get(n_queries_key))
    if n_pairs is None or n_queries is None or n_pairs <= 0:
        return False

    changed = False
    for top_k in TOP_K_VALUES:
        recall_key = f"recall_at_{top_k}"
        hit_key = f"hit_at_{top_k}"
        precision_key = f"precision_at_{top_k}"

        old_hit = _as_float(metrics.get(hit_key))
        old_recall = _as_float(metrics.get(recall_key))
        if old_hit is None and old_recall is not None:
            metrics[hit_key] = old_recall
            changed = True

        precision = _as_float(metrics.get(precision_key))
        if precision is not None:
            hits = precision * float(top_k) * n_queries
            metrics[recall_key] = max(0.0, min(1.0, hits / n_pairs))
            changed = True
    return changed


def _update_metric_payload(payload: dict[str, Any], counts: dict[str, Any]) -> bool:
    changed = False
    for direction in ("bgc_to_compound", "compound_to_bgc"):
        direction_metrics = payload.get(direction)
        if isinstance(direction_metrics, dict):
            changed = _update_direction_metrics(direction_metrics, counts, direction) or changed
    return changed


def _update_baselines(baselines: dict[str, Any], counts: dict[str, Any]) -> bool:
    changed = False
    for method in ("random", "frozen_encoder_similarity"):
        payload = baselines.get(method)
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if isinstance(metrics, dict):
            changed = _update_metric_payload(metrics, counts) or changed

    knn = baselines.get("knn_transfer")
    metrics_by_k = knn.get("metrics_by_k") if isinstance(knn, dict) else None
    if isinstance(metrics_by_k, dict):
        for metrics in metrics_by_k.values():
            if isinstance(metrics, dict):
                changed = _update_metric_payload(metrics, counts) or changed

    linear = baselines.get("linear_projection")
    if isinstance(linear, dict):
        for key in ("metrics", "all_metrics"):
            metrics = linear.get(key)
            if isinstance(metrics, dict):
                changed = _update_metric_payload(metrics, counts) or changed
    return changed


def _refresh_retrieval_plots(summary: dict[str, Any], run_root: Path) -> dict[str, Any]:
    retrieval_plot_dir = run_root / "retrieval_plots"
    retrieval_plot_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = list(TOP_K_VALUES)
    retrieval_long = build_retrieval_long(summary, top_k_values)
    long_path = retrieval_plot_dir / "retrieval_long.csv"
    retrieval_long.to_csv(long_path, index=False)
    retrieval_summary = summarize_retrieval(retrieval_long)
    summary_path = retrieval_plot_dir / "retrieval_summary.csv"
    retrieval_summary.to_csv(summary_path, index=False)
    return {
        "long_csv": str(long_path),
        "summary_csv": str(summary_path),
        "topk_hit": plot_topk_hit(retrieval_summary, retrieval_plot_dir, "retrieval", top_k_values),
        "topk_recall": plot_topk_recall(retrieval_summary, retrieval_plot_dir, "retrieval", top_k_values),
        "mrr": plot_mrr(retrieval_summary, retrieval_plot_dir, "retrieval"),
        "class_retrieval": plot_class_retrieval(summary, retrieval_plot_dir, "retrieval"),
    }


def refresh_run(run_root: Path) -> dict[str, Any]:
    summary_path = run_root / "summary.json"
    summary = _load_json(summary_path)
    changed_folds = 0
    for fold in summary.get("folds", []):
        counts = fold.get("counts", {}).get("test", {})
        if not isinstance(counts, dict):
            continue
        changed = False
        retrieval_test = fold.get("retrieval_test")
        if isinstance(retrieval_test, dict):
            changed = _update_metric_payload(retrieval_test, counts) or changed
        baselines = fold.get("retrieval_baselines_test")
        if isinstance(baselines, dict):
            changed = _update_baselines(baselines, counts) or changed
        if changed:
            changed_folds += 1
            fold_summary_path = Path(str(fold.get("output_dir", ""))) / "fold_summary.json"
            if fold_summary_path.exists():
                fold_summary = _load_json(fold_summary_path)
                fold_summary["retrieval_test"] = copy.deepcopy(fold.get("retrieval_test", {}))
                fold_summary["retrieval_baselines_test"] = copy.deepcopy(fold.get("retrieval_baselines_test", {}))
                _save_json(fold_summary, fold_summary_path)

    try:
        summary["aggregate"] = _build_summary(summary.get("folds", []))
    except KeyError:
        pass
    summary["retrieval_plot_artifacts"] = _refresh_retrieval_plots(summary, run_root)
    _save_json(summary, summary_path)
    return {
        "summary": str(summary_path),
        "changed_folds": int(changed_folds),
        "retrieval_plot_artifacts": summary["retrieval_plot_artifacts"],
    }


def main() -> None:
    args = parse_args()
    manifest = {
        str(run): refresh_run(run)
        for run in args.runs
        if (run / "summary.json").exists()
    }
    _save_json(manifest, Path("results") / "basic_retrieval_hit_recall_refresh.json")


if __name__ == "__main__":
    main()
