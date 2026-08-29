from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clip_core.logging import save_json


RETRIEVAL_METRICS = (
    "mrr",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "hit_at_20",
    "hit_at_50",
    "hit_at_100",
    "hit_at_200",
    "hit_at_500",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "recall_at_100",
    "recall_at_200",
    "recall_at_500",
)
CLASSIFICATION_METRICS = ("accuracy", "macro_f1", "micro_f1", "auroc")
PER_CLASS_METRICS = ("auroc", "recall", "precision", "f1")
BGC_CLASS_ORDER = ["NRPS", "other", "PKS", "ribosomal", "saccharide", "terpene"]
BGC_CLASS_DISPLAY_NAMES = {
    "NRPS": "NRP",
    "other": "Other",
    "PKS": "Polyketide",
    "ribosomal": "RiPP",
    "saccharide": "Saccharide",
    "terpene": "Terpene",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("mean", "value"):
            if key in value:
                return _as_float(value[key])
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _std_value(value: Any) -> float | None:
    if isinstance(value, dict) and "std" in value:
        return _as_float(value["std"])
    return None


def _direction_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction in ("bgc_to_compound", "compound_to_bgc"):
        direction_metrics = metrics.get(direction)
        if not isinstance(direction_metrics, dict):
            continue
        for metric in RETRIEVAL_METRICS:
            value = _as_float(direction_metrics.get(metric))
            if value is None:
                continue
            rows.append(
                {
                    "direction": direction,
                    "metric": metric,
                    "value": value,
                    "std_in_source": _std_value(direction_metrics.get(metric)),
                }
            )
    return rows


def _extract_retrieval_rows(path: Path, run_root: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows: list[dict[str, Any]] = []
    source = str(path.parent.relative_to(run_root))
    for baseline_name, baseline_payload in payload.items():
        if baseline_name == "path" or not isinstance(baseline_payload, dict):
            continue
        if baseline_name == "knn_transfer":
            for k, metrics in baseline_payload.get("metrics_by_k", {}).items():
                for row in _direction_metrics(metrics):
                    rows.append(
                        {
                            "source": source,
                            "baseline": f"knn_transfer_k{k}",
                            **row,
                        }
                    )
            continue
        if baseline_name == "linear_projection":
            metrics = baseline_payload.get("all_metrics", {}).get("retrieval_test")
            if not isinstance(metrics, dict):
                metrics = baseline_payload.get("metrics")
        else:
            metrics = baseline_payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for row in _direction_metrics(metrics):
            rows.append({"source": source, "baseline": baseline_name, **row})
    return rows


def _plot_retrieval_metric(summary_df: pd.DataFrame, metric: str, output: Path) -> None:
    metric_df = summary_df[summary_df["metric"] == metric].copy()
    if metric_df.empty:
        return
    baselines = metric_df["baseline"].drop_duplicates().tolist()
    directions = ["bgc_to_compound", "compound_to_bgc"]
    x = np.arange(len(baselines), dtype=np.float64)
    width = 0.38
    fig_width = max(8.0, 0.9 * len(baselines))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    for offset, direction in enumerate(directions):
        values = []
        errors = []
        for baseline in baselines:
            row = metric_df[(metric_df["baseline"] == baseline) & (metric_df["direction"] == direction)]
            if row.empty:
                values.append(np.nan)
                errors.append(0.0)
            else:
                values.append(float(row.iloc[0]["mean"]))
                errors.append(float(row.iloc[0]["std"]))
        xpos = x + (offset - 0.5) * width
        ax.bar(xpos, values, width=width, yerr=errors, capsize=3, label=direction)
    ax.set_title(f"Retrieval baseline {metric}")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=35, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_retrieval_baseline_artifacts(run_root: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_root)
    output = Path(output_dir) if output_dir is not None else root / "baselines" / "retrieval"
    output.mkdir(parents=True, exist_ok=True)
    json_paths = sorted(
        path
        for path in root.rglob("retrieval_baselines_test.json")
        if "baselines" not in path.relative_to(root).parts
    )
    rows: list[dict[str, Any]] = []
    for path in json_paths:
        rows.extend(_extract_retrieval_rows(path, root))

    manifest: dict[str, Any] = {
        "run_root": str(root),
        "output_dir": str(output),
        "n_source_files": int(len(json_paths)),
        "source_files": [str(path) for path in json_paths],
    }
    if not rows:
        manifest["status"] = "no_retrieval_baseline_files_found"
        save_json(manifest, output / "retrieval_baseline_artifacts.json")
        return manifest

    long_df = pd.DataFrame(rows)
    long_path = output / "retrieval_baselines_long.csv"
    long_df.to_csv(long_path, index=False)

    summary_df = (
        long_df.groupby(["baseline", "direction", "metric"], dropna=False)["value"]
        .agg(mean="mean", std=lambda x: float(np.std(x, ddof=0)), n="count")
        .reset_index()
        .sort_values(["metric", "direction", "baseline"])
    )
    summary_path = output / "retrieval_baselines_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_json_path = output / "retrieval_baselines_summary.json"
    save_json(summary_df.to_dict("records"), summary_json_path)

    plots: list[str] = []
    for metric in RETRIEVAL_METRICS:
        plot_path = output / f"retrieval_baselines_{metric}.png"
        _plot_retrieval_metric(summary_df, metric, plot_path)
        if plot_path.exists():
            plots.append(str(plot_path))

    manifest.update(
        {
            "status": "ok",
            "long_csv": str(long_path),
            "summary_csv": str(summary_path),
            "summary_json": str(summary_json_path),
            "plots": plots,
        }
    )
    save_json(manifest, output / "retrieval_baseline_artifacts.json")
    return manifest


def _classification_metric_rows(report: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    if "bgc_class" in report and isinstance(report["bgc_class"], dict):
        test_metrics = report["bgc_class"].get("test", {})
    else:
        test_metrics = report.get("aggregate", {}).get("test", report.get("test", {}))
    if not isinstance(test_metrics, dict):
        return []

    rows: list[dict[str, Any]] = []
    for metric in CLASSIFICATION_METRICS:
        value = _as_float(test_metrics.get(metric))
        if value is not None:
            rows.append(
                {
                    "scenario": scenario,
                    "split": "test",
                    "metric": metric,
                    "value": value,
                    "std": _std_value(test_metrics.get(metric)),
                }
            )
    per_class = test_metrics.get("per_class")
    if isinstance(per_class, dict):
        for class_name, class_metrics in per_class.items():
            if not isinstance(class_metrics, dict):
                continue
            for metric in PER_CLASS_METRICS:
                value = _as_float(class_metrics.get(metric))
                if value is not None:
                    rows.append(
                        {
                            "scenario": scenario,
                            "split": "test",
                            "class": str(class_name),
                            "metric": metric,
                            "value": value,
                            "std": _std_value(class_metrics.get(metric)),
                        }
                    )
    return rows


def _extract_test_report(report: dict[str, Any]) -> dict[str, Any] | None:
    if "bgc_class" in report and isinstance(report["bgc_class"], dict):
        test_metrics = report["bgc_class"].get("test")
    elif "aggregate" in report and isinstance(report["aggregate"], dict):
        test_metrics = report["aggregate"].get("test")
    else:
        test_metrics = report.get("test")
    return test_metrics if isinstance(test_metrics, dict) else None


def _extract_curve_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    if "bgc_class" in report and isinstance(report["bgc_class"], dict):
        test_metrics = report["bgc_class"].get("test")
        return [test_metrics] if isinstance(test_metrics, dict) else []
    folds = report.get("folds")
    if isinstance(folds, list):
        reports = []
        for fold in folds:
            if isinstance(fold, dict):
                test_metrics = fold.get("metrics", {}).get("test")
                if isinstance(test_metrics, dict):
                    reports.append(test_metrics)
        if reports:
            return reports
    test_metrics = report.get("test")
    return [test_metrics] if isinstance(test_metrics, dict) else []


def _ordered_classes(class_names: list[str]) -> list[str]:
    return [name for name in BGC_CLASS_ORDER if name in class_names] + sorted(
        name for name in class_names if name not in BGC_CLASS_ORDER
    )


def _display_class_name(class_name: str) -> str:
    return BGC_CLASS_DISPLAY_NAMES.get(class_name, class_name)


def _display_scenario_name(scenario: str) -> str:
    display = scenario.replace("_", " ").replace("-", " ").strip()
    return display.title() if display else scenario


def _save_classification_roc_plot(
    curve_reports: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> None:
    if not curve_reports:
        return
    class_names = sorted(
        {
            class_name
            for report in curve_reports
            for class_name in report.get("roc_curves", {}).get("per_class", {})
        }
    )
    class_names = _ordered_classes(class_names)
    if not class_names:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fpr_grid = np.linspace(0.0, 1.0, 301)
    for idx, class_name in enumerate(class_names):
        aucs = []
        tpr_parts: list[np.ndarray] = []
        for report in curve_reports:
            curve = report.get("roc_curves", {}).get("per_class", {}).get(class_name, {})
            fpr = curve.get("fpr")
            tpr = curve.get("tpr")
            auc = _as_float(curve.get("auroc", report.get("per_class", {}).get(class_name, {}).get("auroc")))
            if auc is not None:
                aucs.append(auc)
            if isinstance(fpr, list) and isinstance(tpr, list) and len(fpr) == len(tpr) and len(fpr) > 1:
                fpr_arr = np.asarray(fpr, dtype=float)
                tpr_arr = np.asarray(tpr, dtype=float)
                order = np.argsort(fpr_arr)
                fpr_sorted = fpr_arr[order]
                tpr_sorted = tpr_arr[order]
                unique_fpr, unique_idx = np.unique(fpr_sorted, return_index=True)
                unique_tpr = tpr_sorted[unique_idx]
                if unique_fpr[0] > 0.0:
                    unique_fpr = np.r_[0.0, unique_fpr]
                    unique_tpr = np.r_[0.0, unique_tpr]
                if unique_fpr[-1] < 1.0:
                    unique_fpr = np.r_[unique_fpr, 1.0]
                    unique_tpr = np.r_[unique_tpr, 1.0]
                tpr_parts.append(np.interp(fpr_grid, unique_fpr, unique_tpr))
        if not tpr_parts:
            continue
        if len(tpr_parts) == 1:
            mean_tpr = tpr_parts[0]
        else:
            mean_tpr = np.mean(np.stack(tpr_parts, axis=0), axis=0)
        label = _display_class_name(class_name)
        if aucs:
            label = f"{_display_class_name(class_name)} (AUC = {np.mean(aucs):.3f})"
        ax.plot(fpr_grid, mean_tpr, linewidth=1.6, label=label)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _named_matrix_to_array(matrix_report: dict[str, Any], *, normalized: bool = False) -> tuple[list[str], np.ndarray] | None:
    labels = [str(label) for label in matrix_report.get("labels", [])]
    key = "normalized_true" if normalized else "raw"
    raw = matrix_report.get(key)
    if not labels or not isinstance(raw, dict):
        if isinstance(raw, dict) and raw:
            labels = _ordered_classes([str(label) for label in raw.keys()])
        else:
            return None
    arr = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, row_label in enumerate(labels):
        row = raw.get(row_label, {})
        if not isinstance(row, dict):
            continue
        for j, col_label in enumerate(labels):
            arr[i, j] = _as_float(row.get(col_label)) or 0.0
    return labels, arr


def _plot_matrix(ax: Any, arr: np.ndarray, labels: list[str], title: str, *, fmt: str) -> None:
    im = ax.imshow(arr, cmap="Blues", vmin=0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    threshold = float(np.nanmax(arr)) / 2.0 if arr.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            text = f"{value:.2f}" if fmt == ".2f" else f"{value:.0f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black", fontsize=8)
    return im


def _save_expanded_confusion_plot(test_report: dict[str, Any], output_path: Path, *, title: str) -> None:
    matrix_report = test_report.get("confusion_matrix")
    if not isinstance(matrix_report, dict):
        return
    extracted = _named_matrix_to_array(matrix_report, normalized=True)
    if extracted is None:
        extracted = _named_matrix_to_array(matrix_report, normalized=False)
        fmt = ".0f"
    else:
        fmt = ".2f"
    if extracted is None:
        return
    labels, arr = extracted
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = _plot_matrix(ax, arr, labels, title, fmt=fmt)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _binary_confusion_to_array(confusion: dict[str, Any]) -> tuple[list[str], np.ndarray] | None:
    matrix = confusion.get("confusion_matrix", confusion)
    if not isinstance(matrix, dict):
        return None
    extracted = _named_matrix_to_array(matrix, normalized=False)
    if extracted is None:
        return None
    return extracted


def _save_binary_confusion_grid(test_report: dict[str, Any], output_path: Path, *, title: str) -> None:
    per_class = test_report.get("per_class_binary")
    if not isinstance(per_class, dict) or not per_class:
        return
    classes = _ordered_classes(list(per_class))
    n_cols = min(3, len(classes))
    n_rows = int(math.ceil(len(classes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.5 * n_rows))
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for ax in axes_arr.flat:
        ax.axis("off")
    for idx, class_name in enumerate(classes):
        ax = axes_arr.flat[idx]
        ax.axis("on")
        extracted = _binary_confusion_to_array(per_class[class_name])
        if extracted is None:
            ax.set_title(class_name)
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            continue
        labels, arr = extracted
        _plot_matrix(ax, arr, labels, class_name, fmt=".0f")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _copy_if_exists(source: Path, output: Path) -> str | None:
    if not source.exists():
        return None
    output.mkdir(parents=True, exist_ok=True)
    dest = output / source.name
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return str(dest)


def _discover_classification_reports(root: Path) -> list[tuple[str, Path]]:
    aggregate_candidates: list[tuple[str, Path]] = []
    direct = root / "raw_bgc_classifier_baseline_summary.json"
    if direct.exists():
        aggregate_candidates.append(("raw_bgc_classifier_baseline", direct))
    bgcmac = root / "raw_bgc_baseline" / "raw_bgc_baseline_summary.json"
    if bgcmac.exists():
        aggregate_candidates.append(("raw_bgc_baseline", bgcmac))
    if aggregate_candidates:
        return aggregate_candidates

    candidates: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("raw_bgc_metrics.json")):
        if "baselines" not in path.relative_to(root).parts:
            candidates.append((str(path.parent.parent.relative_to(root)), path))
    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for scenario, path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((scenario, path))
    return unique


def _plot_classification_overall(df: pd.DataFrame, output: Path) -> None:
    overall = df[df["class"].isna() if "class" in df.columns else [True] * len(df)].copy()
    overall = overall[overall["metric"].isin(CLASSIFICATION_METRICS)]
    if overall.empty:
        return
    scenarios = overall["scenario"].drop_duplicates().tolist()
    metrics = [metric for metric in CLASSIFICATION_METRICS if metric in set(overall["metric"])]
    x = np.arange(len(metrics), dtype=np.float64)
    width = 0.8 / max(len(scenarios), 1)
    fig, ax = plt.subplots(figsize=(max(7.5, len(metrics) * 1.3), 4.8))
    for idx, scenario in enumerate(scenarios):
        values = []
        errors = []
        for metric in metrics:
            row = overall[(overall["scenario"] == scenario) & (overall["metric"] == metric)]
            values.append(float(row.iloc[0]["value"]) if not row.empty else np.nan)
            errors.append(float(row.iloc[0]["std"]) if not row.empty and pd.notna(row.iloc[0]["std"]) else 0.0)
        ax.bar(x + (idx - (len(scenarios) - 1) / 2) * width, values, width=width, yerr=errors, capsize=3, label=scenario)
    ax.set_title("Classification baseline overall metrics")
    ax.set_ylabel("score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_classification_per_class(df: pd.DataFrame, metric: str, output: Path) -> None:
    if "class" not in df.columns:
        return
    per_class = df[df["class"].notna() & (df["metric"] == metric)].copy()
    if per_class.empty:
        return
    scenarios = per_class["scenario"].drop_duplicates().tolist()
    classes = per_class["class"].drop_duplicates().tolist()
    x = np.arange(len(classes), dtype=np.float64)
    width = 0.8 / max(len(scenarios), 1)
    fig, ax = plt.subplots(figsize=(max(8.0, len(classes) * 1.1), 4.8))
    for idx, scenario in enumerate(scenarios):
        values = []
        errors = []
        for class_name in classes:
            row = per_class[(per_class["scenario"] == scenario) & (per_class["class"] == class_name)]
            values.append(float(row.iloc[0]["value"]) if not row.empty else np.nan)
            errors.append(float(row.iloc[0]["std"]) if not row.empty and pd.notna(row.iloc[0]["std"]) else 0.0)
        ax.bar(x + (idx - (len(scenarios) - 1) / 2) * width, values, width=width, yerr=errors, capsize=3, label=scenario)
    ax.set_title(f"Classification baseline per-class {metric}")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_classification_baseline_artifacts(run_root: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_root)
    output = Path(output_dir) if output_dir is not None else root / "baselines" / "classification"
    output.mkdir(parents=True, exist_ok=True)
    source_json_dir = output / "source_json"
    if source_json_dir.exists():
        shutil.rmtree(source_json_dir)
    reports = _discover_classification_reports(root)
    manifest: dict[str, Any] = {
        "run_root": str(root),
        "output_dir": str(output),
        "n_source_files": int(len(reports)),
        "source_files": [str(path) for _, path in reports],
    }
    rows: list[dict[str, Any]] = []
    copied: list[str] = []
    for scenario, path in reports:
        copied_path = _copy_if_exists(path, output / "source_json")
        if copied_path is not None:
            copied.append(copied_path)
        rows.extend(_classification_metric_rows(_load_json(path), scenario))

    if not rows:
        manifest["status"] = "no_classification_baseline_files_found"
        save_json(manifest, output / "classification_baseline_artifacts.json")
        return manifest

    df = pd.DataFrame(rows)
    if "class" not in df.columns:
        df["class"] = np.nan
    table_path = output / "classification_baselines_summary.csv"
    df.sort_values(["scenario", "class", "metric"], na_position="first").to_csv(table_path, index=False)
    json_path = output / "classification_baselines_summary.json"
    save_json(df.where(pd.notna(df), None).to_dict("records"), json_path)

    plots: list[str] = []
    overall_path = output / "classification_baselines_overall.png"
    _plot_classification_overall(df, overall_path)
    if overall_path.exists():
        plots.append(str(overall_path))
    for metric in PER_CLASS_METRICS:
        plot_path = output / f"classification_baselines_per_class_{metric}.png"
        _plot_classification_per_class(df, metric, plot_path)
        if plot_path.exists():
            plots.append(str(plot_path))

    for scenario, path in reports:
        report = _load_json(path)
        test_report = _extract_test_report(report)
        if test_report is None:
            continue
        safe_scenario = scenario.replace("/", "_")
        roc_path = output / f"{safe_scenario}_roc.png"
        _save_classification_roc_plot(
            _extract_curve_reports(report),
            roc_path,
            title=f"ROC Curve for {_display_scenario_name(scenario)}",
        )
        if roc_path.exists():
            plots.append(str(roc_path))
        binary_cm_path = output / f"{safe_scenario}_one_vs_rest_confusion_matrices.png"
        _save_binary_confusion_grid(
            test_report,
            binary_cm_path,
            title=f"One-vs-rest confusion matrices ({scenario})",
        )
        if binary_cm_path.exists():
            plots.append(str(binary_cm_path))
        expanded_cm_path = output / f"{safe_scenario}_expanded_confusion_matrix.png"
        _save_expanded_confusion_plot(
            test_report,
            expanded_cm_path,
            title=f"Expanded confusion matrix ({scenario})",
        )
        if expanded_cm_path.exists():
            plots.append(str(expanded_cm_path))

    manifest.update(
        {
            "status": "ok",
            "summary_csv": str(table_path),
            "summary_json": str(json_path),
            "copied_source_json": copied,
            "plots": plots,
        }
    )
    save_json(manifest, output / "classification_baseline_artifacts.json")
    return manifest


def save_all_baseline_artifacts(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    output = root / "baselines"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_root": str(root),
        "retrieval": save_retrieval_baseline_artifacts(root, output / "retrieval"),
        "classification": save_classification_baseline_artifacts(root, output / "classification"),
    }
    save_json(manifest, output / "baseline_artifacts.json")
    return manifest
