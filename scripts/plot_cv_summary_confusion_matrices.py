"""Plot aggregate confusion matrices from a CV summary JSON file."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

PREFERRED_CLASS_ORDER = ["NRPS", "other", "PKS", "ribosomal", "saccharide", "terpene"]
DISPLAY_CLASS_NAMES = {
    "NRPS": "NRP",
    "other": "Other",
    "PKS": "Polyketide",
    "ribosomal": "RiPP",
    "saccharide": "Saccharide",
    "terpene": "Terpene",
}


def _mean_value(value: Any) -> float:
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)


def _labels(matrix: dict[str, Any]) -> list[str]:
    return list(matrix.keys())


def _ordered_class_labels(labels: list[str]) -> list[str]:
    emitted: set[str] = set()
    ordered: list[str] = []
    for label in PREFERRED_CLASS_ORDER:
        if label in labels:
            ordered.append(label)
            emitted.add(label)
    ordered.extend(label for label in labels if label not in emitted)
    return ordered


def _display_label(label: str) -> str:
    return DISPLAY_CLASS_NAMES.get(label, label)


def _matrix(matrix: dict[str, Any], row_labels: list[str], col_labels: list[str]) -> np.ndarray:
    values = np.zeros((len(row_labels), len(col_labels)), dtype=float)
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            values[row_idx, col_idx] = _mean_value(matrix[row_label][col_label])
    return values


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "matrix"


def _plot_confusion_matrix(
    confusion_matrix: dict[str, Any],
    output_path: Path,
    *,
    title: str,
    reorder_bgc_classes: bool = False,
) -> None:
    normalized = confusion_matrix["normalized_true"]
    raw = confusion_matrix["raw"]
    row_labels = _labels(normalized)
    col_labels = list(next(iter(normalized.values())).keys())
    if reorder_bgc_classes:
        row_labels = _ordered_class_labels(row_labels)
        col_labels = _ordered_class_labels(col_labels)
    cm_norm = _matrix(normalized, row_labels, col_labels)
    cm_raw = _matrix(raw, row_labels, col_labels)
    row_display = [_display_label(label) for label in row_labels]
    col_display = [_display_label(label) for label in col_labels]

    fig_width = max(6.0, min(22.0, 0.9 * len(col_labels) + 2.0))
    fig_height = max(5.0, min(20.0, 0.8 * len(row_labels) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_display, rotation=45, ha="right")
    ax.set_yticklabels(row_display)

    text_size = 9 if max(len(row_labels), len(col_labels)) <= 8 else 7
    for true_idx in range(cm_norm.shape[0]):
        for pred_idx in range(cm_norm.shape[1]):
            norm_value = cm_norm[true_idx, pred_idx]
            raw_value = cm_raw[true_idx, pred_idx]
            if raw_value == 0:
                label = "0"
            else:
                label = f"{norm_value:.2f}\n({raw_value:.1f})"
            text_color = "white" if norm_value >= 0.5 else "black"
            ax.text(
                pred_idx,
                true_idx,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=text_size,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _extract_member_bgc_reports(summary: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for member in [*summary.get("members", []), *summary.get("folds", [])]:
        try:
            reports.append(member["downstream"]["bgc_class"]["test"])
        except KeyError:
            continue
    return reports


def _aggregate_auc(bgc_test: dict[str, Any], class_name: str) -> float | None:
    try:
        return _mean_value(bgc_test["roc_curves"]["per_class"][class_name]["auroc"])
    except KeyError:
        pass
    try:
        return _mean_value(bgc_test["per_class"][class_name]["auroc"])
    except KeyError:
        return None


def _plot_bgc_class_roc(
    summary: dict[str, Any],
    bgc_test: dict[str, Any],
    output_path: Path,
    *,
    title: str,
) -> None:
    member_reports = _extract_member_bgc_reports(summary)
    class_names = _ordered_class_labels(list(bgc_test.get("per_class", {}).keys()))
    if not class_names:
        class_names = _ordered_class_labels(list(bgc_test.get("roc_curves", {}).get("per_class", {}).keys()))
    if not class_names:
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    direct_curves = bgc_test.get("roc_curves", {}).get("per_class", {})
    has_direct_curves = any(isinstance(direct_curves.get(class_name, {}).get("fpr"), list) for class_name in class_names)
    if has_direct_curves:
        for class_name in class_names:
            curve = direct_curves.get(class_name)
            if not curve:
                continue
            fpr = curve.get("fpr", [])
            tpr = curve.get("tpr", [])
            if len(fpr) < 2 or len(tpr) < 2:
                continue
            auc = _aggregate_auc(bgc_test, class_name)
            label = _display_label(class_name)
            if auc is not None:
                label = f"{label} (AUC = {auc:.3f})"
            ax.plot(fpr, tpr, linewidth=1.5, label=label)
    elif member_reports:
        fpr_grid = np.linspace(0.0, 1.0, 301)
        for class_name in class_names:
            tpr_parts: list[np.ndarray] = []
            for report in member_reports:
                curve = report.get("roc_curves", {}).get("per_class", {}).get(class_name)
                if not curve:
                    continue
                fpr = np.asarray(curve.get("fpr", []), dtype=float)
                tpr = np.asarray(curve.get("tpr", []), dtype=float)
                if fpr.size < 2 or tpr.size < 2:
                    continue
                order = np.argsort(fpr)
                fpr_sorted = fpr[order]
                tpr_sorted = tpr[order]
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
            mean_tpr = np.mean(np.stack(tpr_parts, axis=0), axis=0)
            auc = _aggregate_auc(bgc_test, class_name)
            label = _display_label(class_name)
            if auc is not None:
                label = f"{label} (AUC = {auc:.3f})"
            ax.plot(fpr_grid, mean_tpr, linewidth=1.5, label=label)
    else:
        plt.close(fig)
        return

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.65)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_binary_confusion_grid(
    per_class_binary: dict[str, Any],
    output_path: Path,
    *,
    title: str,
) -> None:
    class_names = _ordered_class_labels(list(per_class_binary.keys()))
    if not class_names:
        return
    n_cols = min(3, len(class_names))
    n_rows = int(np.ceil(len(class_names) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.0 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    matrices: list[np.ndarray] = []
    for class_name in class_names:
        raw = per_class_binary[class_name]["confusion_matrix"]["raw"]
        matrices.append(
            np.asarray(
                [
                    [_mean_value(raw["negative"]["negative"]), _mean_value(raw["negative"]["positive"])],
                    [_mean_value(raw["positive"]["negative"]), _mean_value(raw["positive"]["positive"])],
                ],
                dtype=float,
            )
        )
    vmax = max(float(matrix.max()) for matrix in matrices) if matrices else 1.0

    for ax, class_name, matrix in zip(axes.flat, class_names, matrices, strict=False):
        ax.axis("on")
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)
        del image
        ax.set_title(_display_label(class_name))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"], rotation=90, va="center")
        local_max = float(matrix.max()) if matrix.size else 0.0
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                color = "white" if local_max and value >= 0.5 * local_max else "black"
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=10)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_summary(summary_path: Path, output_dir: Path, suffix: str = "cv10_mean") -> list[Path]:
    summary = json.loads(summary_path.read_text())
    downstream = summary.get("ensemble_downstream")
    if not isinstance(downstream, dict) or "bgc_class" not in downstream:
        downstream = summary["aggregate"]["downstream"]
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    bgc_test = downstream["bgc_class"]["test"]
    bgc_matrices = [
        (
            f"downstream_confusion_matrix_test_all_bgcs_{suffix}.png",
            bgc_test["confusion_matrix"],
            f"BGC class confusion matrix ({suffix}, test, all BGCs)",
        ),
        (
            f"downstream_confusion_matrix_test_single_class_bgcs_{suffix}.png",
            bgc_test["confusion_matrix_single_class_only"],
            f"BGC class confusion matrix ({suffix}, test, single-class BGCs)",
        ),
    ]
    for filename, matrix, title in bgc_matrices:
        path = output_dir / filename
        _plot_confusion_matrix(matrix, path, title=title, reorder_bgc_classes=True)
        outputs.append(path)

    roc_title = "ROC Curve for BGC-MAC" if "bgcmac" in suffix.lower() else f"ROC Curve for BGC Class Prediction ({suffix}, test)"
    roc_path = output_dir / f"downstream_roc_curve_test_{suffix}.png"
    _plot_bgc_class_roc(summary, bgc_test, roc_path, title=roc_title)
    if roc_path.exists():
        outputs.append(roc_path)

    grid_path = output_dir / f"downstream_confusion_matrices_test_classes_{suffix}.png"
    _plot_binary_confusion_grid(
        bgc_test["per_class_binary"],
        grid_path,
        title=f"One-vs-rest BGC class confusion matrices ({suffix}, test)",
    )
    if grid_path.exists():
        outputs.append(grid_path)

    for class_name in _ordered_class_labels(list(bgc_test["per_class_binary"].keys())):
        class_report = bgc_test["per_class_binary"][class_name]
        path = output_dir / f"downstream_confusion_matrix_test_class_{_slugify(class_name)}_{suffix}.png"
        _plot_confusion_matrix(
            class_report["confusion_matrix"],
            path,
            title=f"One-vs-rest confusion matrix ({suffix}, test, class={class_name})",
        )
        outputs.append(path)

    if "origin_type" in downstream:
        origin_path = output_dir / f"downstream_origin_type_confusion_matrix_test_{suffix}.png"
        _plot_confusion_matrix(
            downstream["origin_type"]["test"]["confusion_matrix"],
            origin_path,
            title=f"Origin type confusion matrix ({suffix}, test)",
        )
        outputs.append(origin_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/combined_cv10/summary.json"),
        help="Path to the CV summary JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/combined_cv10/summary_confusion_matrices"),
        help="Directory where PNG files will be written.",
    )
    args = parser.parse_args()

    outputs = plot_summary(args.summary, args.output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
