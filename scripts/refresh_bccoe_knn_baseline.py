from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.eval.retrieval_baselines import knn_transfer_retrieval_baseline
from projects.mibig_bgc_np.scripts.plot_bccoe_retrieval import (
    build_long_table,
    save_plot,
    _summarize,
)
from projects.mibig_bgc_np.scripts.run_cv10 import _build_summary as _build_cv_summary
from projects.mibig_bgc_np.scripts.run_leave_one_class_out import _build_summary as _build_loco_summary


DEFAULT_SUMMARIES = (
    Path("results/bccoe_bgc_cv10/summary.json"),
    Path("results/bccoe_np_cv10/summary.json"),
    Path("results/bccoe_loco_exp3_bgc/summary.json"),
    Path("results/bccoe_loco_exp4_np/summary.json"),
)
TOP_K_VALUES = (5, 10, 20, 50, 100, 200, 500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh BCCoE summaries with paper-matched direct KNN baseline.")
    parser.add_argument("--summaries", type=Path, nargs="*", default=list(DEFAULT_SUMMARIES))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _refresh_fold_knn(summary: dict[str, Any], fold: dict[str, Any]) -> dict[str, Any]:
    data_dir = summary["data_dir"]
    cache_dir = summary["cache_dir"]
    splits_path = fold.get("splits_path", summary.get("splits_path"))
    cv_fold = fold.get("fold_id") if summary.get("splits_path") else None
    val_fold = fold.get("val_fold")
    interactions = build_interactions(data_dir, splits_path=splits_path, cv_fold=cv_fold, val_fold=val_fold)
    knn = knn_transfer_retrieval_baseline(
        interactions=interactions,
        split="test",
        cache_dir=cache_dir,
        k_values=(1,),
    )
    baselines = copy.deepcopy(fold.get("retrieval_baselines_test", {}))
    baselines["knn_transfer"] = knn
    fold["retrieval_baselines_test"] = baselines
    output_dir = Path(str(fold.get("output_dir", "")))
    if output_dir.exists():
        _save_json(baselines, output_dir / "retrieval_baselines_test.json")
        fold_summary_path = output_dir / "fold_summary.json"
        if fold_summary_path.exists():
            fold_summary = _load_json(fold_summary_path)
            fold_summary["retrieval_baselines_test"] = baselines
            _save_json(fold_summary, fold_summary_path)
        experiment_summary_path = output_dir / "experiment_summary.json"
        if experiment_summary_path.exists():
            experiment_summary = _load_json(experiment_summary_path)
            experiment_summary["retrieval_baselines_test"] = baselines
            _save_json(experiment_summary, experiment_summary_path)
    return knn


def _refresh_bccoe_plots(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    outdir = summary_path.parent / "bccoe_retrieval_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    long_df = build_long_table(summary, list(TOP_K_VALUES))
    prefix = {
        "bccoe_bgc_cv10": "bccoe_bgc",
        "bccoe_np_cv10": "bccoe_np",
    }.get(summary_path.parent.name, "bccoe_retrieval")
    long_path = outdir / f"{prefix}_long.csv"
    long_df.to_csv(long_path, index=False)
    summary_df = _summarize(long_df)
    summary_csv = outdir / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    plots: dict[str, str] = {}
    for direction in ("bgc_to_compound", "compound_to_bgc"):
        direction_df = summary_df[summary_df["direction"] == direction]
        if direction_df.empty:
            continue
        suffix = "bgc_to_np" if direction == "bgc_to_compound" else "np_to_bgc"
        plot_path = outdir / f"{prefix}_{suffix}_topk_recall.png"
        save_plot(direction_df, plot_path, direction=direction, top_k_values=list(TOP_K_VALUES), model_label="Combi")
        plots[direction] = str(plot_path)
    manifest = {
        "summary": str(summary_path),
        "long_csv": str(long_path),
        "summary_csv": str(summary_csv),
        "plots": plots,
        "top_k": list(TOP_K_VALUES),
        "directions": ["bgc_to_compound", "compound_to_bgc"],
    }
    manifest_path = outdir / f"{prefix}_manifest.json"
    _save_json(manifest, manifest_path)
    return manifest


def refresh_summary(summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    refreshed = []
    for fold in summary.get("folds", []):
        knn = _refresh_fold_knn(summary, fold)
        refreshed.append(
            {
                "fold_id": fold.get("fold_id"),
                "output_dir": fold.get("output_dir"),
                "route_scoring": knn.get("route_scoring"),
                "compound_similarity": knn.get("compound_similarity"),
            }
        )

    if "n_experiments" in summary:
        summary["aggregate"] = _build_loco_summary(summary.get("folds", []))
    else:
        summary["aggregate"] = _build_cv_summary(summary.get("folds", []))
    _save_json(summary, summary_path)
    plots = _refresh_bccoe_plots(summary, summary_path)
    return {
        "summary": str(summary_path),
        "n_refreshed": len(refreshed),
        "folds": refreshed,
        "plots": plots,
    }


def main() -> None:
    args = parse_args()
    manifest = {
        str(path): refresh_summary(path)
        for path in args.summaries
        if path.exists()
    }
    _save_json(manifest, Path("results") / "bccoe_knn_refresh.json")


if __name__ == "__main__":
    main()
