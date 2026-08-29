from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._bootstrap import ensure_src_path
ensure_src_path()

from projects.mibig_bgc_np.eval.retrieval_class_metrics import (
    evaluate_bgc_class_pair_scores,
    save_bgc_class_retrieval_plots,
    save_bgc_map_metrics_table,
)
from projects.mibig_bgc_np.scripts.run_bgcmac_ensemble import save_bgcmac_benchmark_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh BGC-MAC/BGC-MAP saved outputs with TP/TN/FP/FN paper-style metrics."
    )
    parser.add_argument("--bgcmac_dir", type=Path, default=Path("results/bgcmac_ensemble"))
    parser.add_argument("--bgcmap_dir", type=Path, default=Path("results/bgcmap_retrieval"))
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _accuracy_from_binary_confusion(confusion: dict[str, Any]) -> float:
    raw = confusion["raw"]
    negative = "Negative" if "Negative" in raw else "negative"
    positive = "Positive" if "Positive" in raw else "positive"
    tn = float(raw[negative][negative])
    fp = float(raw[negative][positive])
    fn = float(raw[positive][negative])
    tp = float(raw[positive][positive])
    total = tp + tn + fp + fn
    return float((tp + tn) / total) if total else 0.0


def _add_accuracy_to_bgcmac_report(report: dict[str, Any]) -> dict[str, Any]:
    test = report.get("bgc_class", {}).get("test", {})
    per_class = test.get("per_class", {})
    per_class_binary = test.get("per_class_binary", {})
    if not isinstance(per_class, dict) or not isinstance(per_class_binary, dict):
        return report
    for class_name, binary_metrics in per_class_binary.items():
        if not isinstance(binary_metrics, dict) or "confusion_matrix" not in binary_metrics:
            continue
        accuracy = _accuracy_from_binary_confusion(binary_metrics["confusion_matrix"])
        binary_metrics["accuracy"] = accuracy
        if isinstance(per_class.get(class_name), dict):
            per_class[class_name]["accuracy"] = accuracy
    return report


def refresh_bgcmac(root: Path, *, save_plots: bool) -> dict[str, Any]:
    paths = {
        "ensemble_downstream": root / "ensemble_downstream_metrics.json",
        "ensemble_downstream_full_bgcmac": root / "ensemble_downstream_full_bgcmac_metrics.json",
        "raw_bgc_baseline": root / "raw_bgc_baseline" / "raw_bgc_baseline_summary.json",
    }
    reports = {key: _add_accuracy_to_bgcmac_report(_load_json(path)) for key, path in paths.items()}
    for key, path in paths.items():
        _save_json(reports[key], path)

    artifacts: dict[str, Any] = {}
    if save_plots:
        artifacts = save_bgcmac_benchmark_artifacts(
            strict_report=reports["ensemble_downstream"],
            full_report=reports["ensemble_downstream_full_bgcmac"],
            baseline_report=reports["raw_bgc_baseline"],
            output_dir=root / "bgcmac_benchmark_artifacts",
        )

    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        summary["ensemble_downstream"] = reports["ensemble_downstream"]
        summary["ensemble_downstream_full_bgcmac"] = reports["ensemble_downstream_full_bgcmac"]
        summary["raw_bgc_baseline"] = reports["raw_bgc_baseline"]
        if artifacts:
            summary["bgcmac_artifacts"] = artifacts
        _save_json(summary, summary_path)

    return {
        "updated": [str(path) for path in paths.values()],
        "artifacts": artifacts,
    }


def refresh_bgcmap(root: Path, *, save_plots: bool) -> dict[str, Any]:
    scored_pairs_path = root / "ensemble_test_pair_scores.tsv"
    thresholds_path = root / "validation_thresholds.json"
    scored_pairs = pd.read_csv(scored_pairs_path, sep="\t")
    threshold_payload = _load_json(thresholds_path)
    thresholds_by_class = {
        str(key): float(value)
        for key, value in threshold_payload.get("thresholds_by_class", {}).items()
    }

    class_report = evaluate_bgc_class_pair_scores(
        scored_pairs,
        split="test",
        thresholds_by_class=thresholds_by_class,
    )
    class_report["threshold_protocol"] = "validation_derived_mean_by_class"
    class_report["validation_thresholds_path"] = str(thresholds_path)
    if save_plots:
        class_report["plots"] = save_bgc_class_retrieval_plots(class_report, root, prefix="ensemble_test")
        class_report["metrics_table"] = save_bgc_map_metrics_table(class_report, root, prefix="ensemble_test")

    ensemble_path = root / "ensemble_test_retrieval.json"
    ensemble = _load_json(ensemble_path)
    ensemble["bgc_class_pair_retrieval"] = class_report
    _save_json(ensemble, ensemble_path)

    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        summary.setdefault("ensemble_test", {})
        summary["ensemble_test"]["bgc_class_pair_retrieval"] = class_report
        _save_json(summary, summary_path)

    return {
        "updated": [str(ensemble_path), str(summary_path)],
        "metrics_table": class_report.get("metrics_table", {}),
    }


def main() -> None:
    args = parse_args()
    save_plots = not bool(args.no_plots)
    manifest = {
        "bgcmac": refresh_bgcmac(args.bgcmac_dir, save_plots=save_plots),
        "bgcmap": refresh_bgcmap(args.bgcmap_dir, save_plots=save_plots),
    }
    _save_json(manifest, Path("results") / "bgcmac_bgcmap_paper_metrics_refresh.json")


if __name__ == "__main__":
    main()
