from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import build_bgc_class_table, build_interactions, load_pair_table
from projects.mibig_bgc_np.eval.baseline_artifacts import save_all_baseline_artifacts
from projects.mibig_bgc_np.eval.retrieval_class_metrics import (
    evaluate_bgc_class_retrieval,
    save_bgc_class_retrieval_plots,
)
from projects.mibig_bgc_np.eval.retrieval_baselines import run_retrieval_baseline_suite
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.models.classification import BGCClassifier
from projects.mibig_bgc_np.training.contrastive_trainer import (
    _build_batch_positive_mask,
    _build_loader,
    _build_positive_pair_set,
    _get_cached_paths,
    _infer_input_dims,
    build_unique_embeddings,
    evaluate_split_retrieval,
)
from projects.mibig_bgc_np.training.downstream_trainer import (
    _binary_confusion_named,
    _binary_roc_curve,
    _expanded_multilabel_confusion,
    _multilabel_overall_metrics,
    train_downstream,
)
from projects.mibig_bgc_np.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BGC-MAC fixed-test NP class classification benchmark.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--bgcmac_splits_path", type=str, default="data/MIBIG/splits/bgcmac_fold.csv")
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fold", type=int, default=10)
    parser.add_argument("--val_folds", type=int, nargs="*", default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--retrieval_baselines",
        dest="retrieval_baselines",
        action="store_true",
        default=True,
        help="Run random, frozen encoder similarity, kNN transfer, and linear-projection retrieval baselines. Enabled by default.",
    )
    parser.add_argument(
        "--no_retrieval_baselines",
        dest="retrieval_baselines",
        action="store_false",
        help="Disable retrieval baselines.",
    )
    parser.add_argument("--baseline_random_trials", type=int, default=10)
    parser.add_argument("--baseline_k_values", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument(
        "--reuse_existing_members",
        action="store_true",
        help="Reuse existing member checkpoints instead of retraining contrastive models.",
    )
    parser.add_argument(
        "--save_cm_png",
        dest="save_cm_png",
        action="store_true",
        default=True,
        help="Save downstream class ROC plots, confusion matrices, and aggregate summary plots. Enabled by default.",
    )
    parser.add_argument(
        "--no_save_cm_png",
        dest="save_cm_png",
        action="store_false",
        help="Disable downstream class ROC plots, confusion matrices, and aggregate summary plots.",
    )
    return parser.parse_args()


def _load_bgcmac_fold_table(path: str | Path, test_fold: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"BGC_number", "biosyn_class", "fold", "is_test"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"BGC-MAC split file {path} is missing required columns: {sorted(missing)}")

    out = df[["BGC_number", "biosyn_class", "fold", "is_test"]].copy()
    out["bgc_id"] = out["BGC_number"].astype(str)
    out["bgc_classes"] = out["biosyn_class"].fillna("").astype(str)
    out["fold"] = pd.to_numeric(out["fold"], errors="coerce")
    if bool(out["fold"].isna().any()):
        raise ValueError(f"BGC-MAC split file {path} contains non-numeric fold values.")
    out["fold"] = out["fold"].astype(int)
    out["is_test"] = out["is_test"].astype(str).str.lower().isin({"true", "1", "yes", "y"})

    test_folds = sorted(out.loc[out["is_test"], "fold"].unique().tolist())
    if test_folds and test_folds != [int(test_fold)]:
        raise ValueError(f"Expected is_test rows only in fold {test_fold}, found test folds {test_folds}")
    if not test_folds:
        out["is_test"] = out["fold"] == int(test_fold)
    return out[["bgc_id", "bgc_classes", "fold", "is_test"]]


def _build_bgcmac_interactions(data_dir: str | Path, fold_table: pd.DataFrame, val_fold: int) -> pd.DataFrame:
    pair_df = load_pair_table(data_dir).copy()
    fold_df = fold_table.copy()
    fold_df["bgc_id"] = fold_df["bgc_id"].astype(str)
    pair_df["bgc_id"] = pair_df["bgc_id"].astype(str)
    interactions = pair_df.merge(fold_df, on="bgc_id", how="inner")
    if "bgc_classes_y" in interactions.columns:
        interactions["bgc_classes"] = interactions["bgc_classes_y"].fillna("").astype(str)
    elif "bgc_classes" in fold_df.columns:
        interactions["bgc_classes"] = interactions["bgc_classes"].fillna("").astype(str)
    interactions["split"] = np.where(
        interactions["is_test"],
        "test",
        np.where(interactions["fold"] == int(val_fold), "val", "train"),
    )
    interactions = interactions.dropna(subset=["bgc_id", "compound_id", "split"]).copy()
    interactions["split"] = interactions["split"].astype(str).str.lower()
    interactions = interactions.drop_duplicates(subset=["bgc_id", "compound_id", "split"]).reset_index(drop=True)
    return interactions


def _split_counts(interactions: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_df = interactions[interactions["split"] == split]
        counts[split] = {
            "n_bgcs": int(split_df["bgc_id"].nunique()),
            "n_compounds": int(split_df["compound_id"].nunique()),
            "n_pairs": int(len(split_df)),
        }
    return counts


def _write_resolved_split_tsv(interactions: pd.DataFrame, output_path: Path) -> Path:
    keep_cols = ["bgc_id", "split"]
    if "bgc_classes" in interactions.columns:
        keep_cols.append("bgc_classes")
    split_df = interactions[keep_cols].drop_duplicates(subset=["bgc_id"]).copy()
    split_df["bgc_id"] = split_df["bgc_id"].astype(str)
    split_df["split"] = split_df["split"].astype(str).str.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.sort_values("bgc_id").to_csv(output_path, sep="\t", index=False)
    return output_path


def _write_full_bgcmac_split_tsv(fold_table: pd.DataFrame, val_fold: int, output_path: Path) -> Path:
    split_df = fold_table[["bgc_id", "bgc_classes", "fold", "is_test"]].drop_duplicates(subset=["bgc_id"]).copy()
    split_df["split"] = np.where(
        split_df["is_test"],
        "test",
        np.where(split_df["fold"] == int(val_fold), "val", "train"),
    )
    split_df = split_df.rename(columns={"fold": "fold_id"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df[["bgc_id", "bgc_classes", "split", "fold_id"]].sort_values("bgc_id").to_csv(
        output_path, sep="\t", index=False
    )
    return output_path


def _load_downstream_classifier(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[BGCClassifier, list[str]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    label_vocab = [str(label) for label in ckpt["label_vocab"]]
    classifier = BGCClassifier(
        emb_dim=int(cfg["model"]["emb_dim"]),
        num_classes=len(label_vocab),
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    classifier.load_state_dict(ckpt["classifier_state_dict"])
    classifier.eval()
    return classifier, label_vocab


def _build_label_matrix(bgc_df: pd.DataFrame, label_to_idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.zeros((len(bgc_df), len(label_to_idx)), dtype=torch.float32)
    for row_idx, labels in enumerate(bgc_df["bgc_class_list"].tolist()):
        for label in labels:
            label_text = str(label)
            if label_text not in label_to_idx:
                raise ValueError(f"BGC-MAC label absent from ensemble label vocabulary: {label_text}")
            y[row_idx, label_to_idx[label_text]] = 1.0
    single_class_mask = y.sum(dim=1) == 1
    return y, single_class_mask


def _parse_label_text(label_text: Any) -> list[str]:
    raw = "" if label_text is None else str(label_text)
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
    candidates = parsed if isinstance(parsed, (list, tuple, set)) else raw.replace(";", ",").split(",")
    labels: list[str] = []
    seen: set[str] = set()
    for label in candidates:
        clean = str(label).strip().strip("'\"")
        if clean and clean not in seen:
            labels.append(clean)
            seen.add(clean)
    return labels


def _build_raw_bgc_features(
    bgc_df: pd.DataFrame,
    bgc_cache: dict[str, torch.Tensor],
    label_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    missing = sorted(set(bgc_df["bgc_id"].astype(str).tolist()).difference(bgc_cache))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"Missing cached BGC features for {len(missing)} BGC-MAC ids. Examples: {preview}. "
            "Rebuild the cache with --bgcmac_splits_path and --fasta_path."
        )
    if bgc_df.empty:
        dim = int(next(iter(bgc_cache.values())).numel())
        return torch.empty((0, dim), dtype=torch.float32), torch.empty((0, len(label_to_idx)), dtype=torch.float32)
    x = torch.stack([bgc_cache[str(bgc_id)].float() for bgc_id in bgc_df["bgc_id"].tolist()])
    y, _ = _build_label_matrix(bgc_df, label_to_idx)
    return x, y


def _predict_raw_classifier(
    classifier: BGCClassifier,
    bgc_df: pd.DataFrame,
    bgc_cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    missing = sorted(set(bgc_df["bgc_id"].astype(str).tolist()).difference(bgc_cache))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing cached BGC features for {len(missing)} BGC-MAC ids. Examples: {preview}")
    probs: list[torch.Tensor] = []
    classifier.eval()
    with torch.no_grad():
        for start in range(0, len(bgc_df), batch_size):
            chunk = bgc_df.iloc[start : start + batch_size]
            features = torch.stack([bgc_cache[str(bgc_id)].float() for bgc_id in chunk["bgc_id"].tolist()]).to(device)
            probs.append(torch.sigmoid(classifier(features)).cpu())
    return torch.cat(probs, dim=0) if probs else torch.empty((0, 0), dtype=torch.float32)


def _predict_bgc_probabilities(
    model: DualEncoderCLIP,
    classifier: BGCClassifier,
    bgc_ids: list[str],
    bgc_cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    missing = sorted(set(bgc_ids).difference(bgc_cache))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing BGC features for {len(missing)} BGC-MAC ids. Examples: {preview}")

    probs: list[torch.Tensor] = []
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for start in range(0, len(bgc_ids), batch_size):
            batch_ids = bgc_ids[start : start + batch_size]
            features = torch.stack([bgc_cache[bgc_id].float() for bgc_id in batch_ids]).to(device)
            embeddings = model.encode_bgc(features)
            logits = classifier(embeddings)
            probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs, dim=0) if probs else torch.empty((0, 0), dtype=torch.float32)


def _best_threshold(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true_np = y_true.detach().cpu().numpy().astype(int)
    y_score_np = y_score.detach().cpu().numpy().astype(float)
    if y_true_np.size == 0 or y_true_np.sum() == 0 or y_true_np.sum() == y_true_np.size:
        return 0.5
    thresholds = np.unique(y_score_np)
    best = 0.5
    best_score = -float("inf")
    for threshold in thresholds:
        pred = y_score_np >= threshold
        tp = float(((y_true_np == 1) & pred).sum())
        fp = float(((y_true_np == 0) & pred).sum())
        fn = float(((y_true_np == 1) & ~pred).sum())
        tn = float(((y_true_np == 0) & ~pred).sum())
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best = float(threshold)
    return best


def _ensemble_multilabel_report(
    y_true: torch.Tensor,
    probs: torch.Tensor,
    thresholds: torch.Tensor,
    class_names: list[str],
) -> dict[str, Any]:
    y_pred = (probs >= thresholds.reshape(1, -1)).to(dtype=torch.float32)
    top1_pred = probs.argmax(dim=-1) if probs.numel() else torch.empty(0, dtype=torch.long)
    single_class_mask = y_true.sum(dim=1) == 1 if y_true.numel() else torch.empty(0, dtype=torch.bool)
    overall = _multilabel_overall_metrics(y_true, y_pred, class_names)

    per_class: dict[str, Any] = {}
    per_class_binary: dict[str, Any] = {}
    roc_curves: dict[str, Any] = {}
    for idx, class_name in enumerate(class_names):
        y_true_col = y_true[:, idx]
        y_pred_col = y_pred[:, idx]
        y_score_col = probs[:, idx]
        tp = float(((y_true_col == 1) & (y_pred_col == 1)).sum().item())
        fp = float(((y_true_col == 0) & (y_pred_col == 1)).sum().item())
        fn = float(((y_true_col == 1) & (y_pred_col == 0)).sum().item())
        support = int((y_true_col == 1).sum().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr, tpr, auc = _binary_roc_curve(y_true_col, y_score_col)
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "auroc": float(auc),
            "threshold": float(thresholds[idx].item()),
        }
        per_class[class_name] = metrics
        per_class_binary[class_name] = {
            **metrics,
            "confusion_matrix": _binary_confusion_named(y_true_col, y_pred_col),
        }
        roc_curves[class_name] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "auroc": float(auc),
        }

    micro_fpr, micro_tpr, micro_auroc = _binary_roc_curve(y_true.reshape(-1), probs.reshape(-1))
    macro_auroc = float(np.mean([curve["auroc"] for curve in roc_curves.values()])) if roc_curves else 0.0
    overall["micro_auroc"] = float(micro_auroc)
    overall["macro_auroc"] = float(macro_auroc)
    return {
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "micro_f1": overall["micro_f1"],
        "macro_auroc": overall["macro_auroc"],
        "micro_auroc": overall["micro_auroc"],
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": _expanded_multilabel_confusion(y_true, top1_pred, class_names),
        "confusion_matrix_single_class_only": _expanded_multilabel_confusion(
            y_true, top1_pred, class_names, mask=single_class_mask
        ),
        "per_class_binary": per_class_binary,
        "roc_curves": {
            "per_class": roc_curves,
            "micro": {
                "fpr": [float(x) for x in micro_fpr],
                "tpr": [float(x) for x in micro_tpr],
                "auroc": float(micro_auroc),
            },
        },
        "n_single_class_bgcs": int(single_class_mask.sum().item()) if single_class_mask.numel() else 0,
        "n_multilabel_bgcs": int((y_true.sum(dim=1) > 1).sum().item()) if y_true.numel() else 0,
    }


_BGC_MAC_CLASS_ORDER = ["NRPS", "other", "PKS", "ribosomal", "saccharide", "terpene"]
_BGC_MAC_DISPLAY_NAMES = {
    "NRPS": "NRP",
    "PKS": "polyketide",
    "other": "other",
    "ribosomal": "RiPP",
    "saccharide": "saccharide",
    "terpene": "terpene",
}


def _ordered_bgcmac_classes(class_names: list[str]) -> list[str]:
    emitted: set[str] = set()
    ordered: list[str] = []
    for class_name in _BGC_MAC_CLASS_ORDER:
        if class_name in class_names:
            ordered.append(class_name)
            emitted.add(class_name)
    ordered.extend(class_name for class_name in class_names if class_name not in emitted)
    return ordered


def _display_bgcmac_class(class_name: str) -> str:
    return _BGC_MAC_DISPLAY_NAMES.get(class_name, class_name)


def _metric_value(report: dict[str, Any], class_name: str, metric_name: str) -> float:
    class_metrics = report["bgc_class"]["test"]["per_class"].get(class_name, {})
    if metric_name == "AUROC":
        return float(class_metrics.get("auroc", 0.0))
    if metric_name == "F1":
        return float(class_metrics.get("f1", 0.0))
    return float(class_metrics.get(metric_name.lower(), 0.0))


def _support_value(report: dict[str, Any], class_name: str) -> int:
    return int(report["bgc_class"]["test"]["per_class"].get(class_name, {}).get("support", 0))


def _save_bgcmac_metrics_table(
    strict_report: dict[str, Any],
    full_report: dict[str, Any],
    baseline_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = _ordered_bgcmac_classes(
        sorted(
            set(strict_report["bgc_class"]["test"]["per_class"])
            | set(full_report["bgc_class"]["test"]["per_class"])
            | set(baseline_report["bgc_class"]["test"]["per_class"])
        )
    )
    rows: list[dict[str, Any]] = []
    for class_name in classes:
        support = _support_value(full_report, class_name) or _support_value(strict_report, class_name)
        for metric_name in ("AUROC", "recall", "precision", "F1"):
            rows.append(
                {
                    "class": _display_bgcmac_class(class_name) if metric_name == "AUROC" else "",
                    "BGC count": support if metric_name == "AUROC" else "",
                    "model": metric_name,
                    "BGC-MAC": _metric_value(strict_report, class_name, metric_name),
                    "BGC-MAC-full": _metric_value(full_report, class_name, metric_name),
                    "baseline": _metric_value(baseline_report, class_name, metric_name),
                }
            )
    table_df = pd.DataFrame(rows, columns=["class", "BGC count", "model", "BGC-MAC", "BGC-MAC-full", "baseline"])
    csv_path = output_dir / "bgcmac_metrics_table.csv"
    table_df.to_csv(csv_path, index=False)

    png_path = output_dir / "bgcmac_metrics_table.png"
    try:
        import matplotlib.pyplot as plt

        display = table_df.copy()
        for col in ("BGC-MAC", "BGC-MAC-full", "baseline"):
            display[col] = display[col].map(lambda value: f"{float(value):.3f}")
        fig_height = max(4.0, 0.34 * (len(display) + 1))
        fig, ax = plt.subplots(figsize=(9.4, fig_height))
        ax.axis("off")
        table = ax.table(
            cellText=display.values.tolist(),
            colLabels=display.columns.tolist(),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.25)
        for (row_idx, _col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#e6e6e6")
                cell.set_text_props(weight="bold")
            elif str(display.iloc[row_idx - 1]["class"]).strip():
                cell.set_text_props(weight="bold")
        fig.tight_layout()
        fig.savefig(png_path, dpi=220)
        plt.close(fig)
    except Exception as exc:
        png_path.write_text(f"Could not render table PNG: {exc}\n", encoding="utf-8")
    return {"csv": str(csv_path), "png": str(png_path)}


def _save_bgcmac_roc_plot(report: dict[str, Any], output_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    test = report["bgc_class"]["test"]
    curves = test.get("roc_curves", {}).get("per_class", {})
    classes = _ordered_bgcmac_classes(list(curves.keys()))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for class_name in classes:
        curve = curves.get(class_name, {})
        fpr = curve.get("fpr", [])
        tpr = curve.get("tpr", [])
        if len(fpr) < 2 or len(tpr) < 2:
            continue
        auc = float(curve.get("auroc", test["per_class"].get(class_name, {}).get("auroc", 0.0)))
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{_display_bgcmac_class(class_name)} (AUC = {auc:.3f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", alpha=0.65)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _extract_binary_matrix(confusion_matrix: dict[str, Any]) -> np.ndarray:
    raw = confusion_matrix["raw"]
    return np.asarray(
        [
            [float(raw["negative"]["negative"]), float(raw["negative"]["positive"])],
            [float(raw["positive"]["negative"]), float(raw["positive"]["positive"])],
        ],
        dtype=float,
    )


def _save_bgcmac_binary_confusion_grid(report: dict[str, Any], output_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_class = report["bgc_class"]["test"]["per_class_binary"]
    classes = _ordered_bgcmac_classes(list(per_class.keys()))
    if not classes:
        return
    n_cols = min(3, len(classes))
    n_rows = int(np.ceil(len(classes) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    matrices = [_extract_binary_matrix(per_class[class_name]["confusion_matrix"]) for class_name in classes]
    vmax = max(float(matrix.max()) for matrix in matrices) if matrices else 1.0
    for ax, class_name, matrix in zip(axes.flat, classes, matrices, strict=False):
        ax.axis("on")
        ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)
        ax.set_title(_display_bgcmac_class(class_name))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"], rotation=90, va="center")
        local_max = float(matrix.max()) if matrix.size else 0.0
        for i in range(2):
            for j in range(2):
                color = "white" if local_max and matrix[i, j] >= 0.5 * local_max else "black"
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color=color, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_bgcmac_expanded_confusion(report: dict[str, Any], output_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = report["bgc_class"]["test"]["confusion_matrix"]
    raw = cm["raw"]
    normalized = cm["normalized_true"]
    row_labels = _ordered_bgcmac_classes(list(normalized.keys()))
    col_labels = _ordered_bgcmac_classes(list(next(iter(normalized.values())).keys()))
    values = np.asarray(
        [[float(normalized[row][col]) for col in col_labels] for row in row_labels],
        dtype=float,
    )
    counts = np.asarray([[float(raw[row][col]) for col in col_labels] for row in row_labels], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([_display_bgcmac_class(label) for label in col_labels], rotation=45, ha="right")
    ax.set_yticklabels([_display_bgcmac_class(label) for label in row_labels])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = "0" if counts[i, j] == 0 else f"{values[i, j]:.2f}\n({counts[i, j]:.0f})"
            color = "white" if values[i, j] >= 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_bgcmac_scenario_artifacts(report: dict[str, Any], output_dir: Path, prefix: str, title: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / f"{prefix}_roc.png",
        output_dir / f"{prefix}_one_vs_rest_confusion_matrices.png",
        output_dir / f"{prefix}_expanded_confusion_matrix.png",
    ]
    _save_bgcmac_roc_plot(report, paths[0], title=f"ROC Curve for {title}")
    _save_bgcmac_binary_confusion_grid(
        report,
        paths[1],
        title=f"One-vs-rest BGC class confusion matrices ({title})",
    )
    _save_bgcmac_expanded_confusion(
        report,
        paths[2],
        title=f"Expanded BGC class confusion matrix ({title})",
    )
    return [str(path) for path in paths]


def save_bgcmac_benchmark_artifacts(
    strict_report: dict[str, Any],
    full_report: dict[str, Any],
    baseline_report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "strict_clip_covered": _save_bgcmac_scenario_artifacts(
            strict_report,
            output,
            prefix="bgcmac_strict",
            title="BGC-MAC strict CLIP-covered",
        ),
        "full_bgcmac": _save_bgcmac_scenario_artifacts(
            full_report,
            output,
            prefix="bgcmac_full",
            title="BGC-MAC full",
        ),
        "raw_bgc_baseline": _save_bgcmac_scenario_artifacts(
            baseline_report,
            output,
            prefix="bgcmac_baseline",
            title="raw BGC baseline",
        ),
        "metrics_table": _save_bgcmac_metrics_table(strict_report, full_report, baseline_report, output),
    }
    save_json(artifacts, output / "bgcmac_artifacts.json")
    return artifacts


def _evaluate_downstream_prediction_ensemble(
    member_summaries: list[dict[str, Any]],
    models: list[DualEncoderCLIP],
    cfg: dict[str, Any],
    data_dir: str | Path,
    cache_dir: str | Path,
    device: torch.device,
    *,
    split_path_key: str = "resolved_splits_path",
    classifier_dir_key: str = "output_dir",
) -> dict[str, Any]:
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    classifiers: list[BGCClassifier] = []
    label_vocab: list[str] | None = None
    member_thresholds: list[torch.Tensor] = []
    test_probs_parts: list[torch.Tensor] = []
    test_bgc_ids: list[str] | None = None
    y_test: torch.Tensor | None = None
    single_test_mask: torch.Tensor | None = None

    for member_summary, model in zip(member_summaries, models, strict=True):
        member_outdir = Path(member_summary[classifier_dir_key])
        classifier, member_vocab = _load_downstream_classifier(
            member_outdir / "downstream_classifier.pt",
            cfg=cfg,
            device=device,
        )
        classifiers.append(classifier)
        if label_vocab is None:
            label_vocab = member_vocab
        elif label_vocab != member_vocab:
            raise ValueError(f"Incompatible downstream label vocabulary in {member_outdir}")
        label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}

        bgc_df = build_bgc_class_table(data_dir, splits_path=member_summary[split_path_key])
        val_df = bgc_df[bgc_df["split"] == "val"].drop_duplicates(subset=["bgc_id"]).sort_values("bgc_id")
        test_df = bgc_df[bgc_df["split"] == "test"].drop_duplicates(subset=["bgc_id"]).sort_values("bgc_id")

        val_ids = val_df["bgc_id"].astype(str).tolist()
        val_probs = _predict_bgc_probabilities(
            model,
            classifier,
            val_ids,
            bgc_cache,
            device,
            int(cfg["downstream"]["feature_batch_size"]),
        )
        y_val, _ = _build_label_matrix(val_df.reset_index(drop=True), label_to_idx)
        thresholds = torch.tensor(
            [_best_threshold(y_val[:, idx], val_probs[:, idx]) for idx in range(len(label_vocab))],
            dtype=torch.float32,
        )
        member_thresholds.append(thresholds)

        current_test_ids = test_df["bgc_id"].astype(str).tolist()
        if test_bgc_ids is None:
            test_bgc_ids = current_test_ids
            y_test, single_test_mask = _build_label_matrix(test_df.reset_index(drop=True), label_to_idx)
        elif test_bgc_ids != current_test_ids:
            raise ValueError(f"Incompatible downstream test BGC ordering in {member_outdir}")
        test_probs_parts.append(
            _predict_bgc_probabilities(
                model,
                classifier,
                current_test_ids,
                bgc_cache,
                device,
                int(cfg["downstream"]["feature_batch_size"]),
            )
        )

    if label_vocab is None or y_test is None or single_test_mask is None or test_bgc_ids is None:
        raise ValueError("No downstream ensemble members were available.")
    ensemble_probs = torch.stack(test_probs_parts, dim=0).mean(dim=0)
    thresholds = torch.stack(member_thresholds, dim=0).mean(dim=0)
    test_report = _ensemble_multilabel_report(y_test, ensemble_probs, thresholds, label_vocab)
    return {
        "tasks": ["bgc_class"],
        "bgc_class": {
            "label_vocab": label_vocab,
            "class_names": label_vocab,
            "test": test_report,
        },
        "n_models": int(len(models)),
        "n_test_bgcs": int(len(test_bgc_ids)),
        "test_bgc_ids": test_bgc_ids,
        "threshold_source": "mean validation Youden threshold across ensemble members",
    }


def _train_raw_bgc_baseline_member(
    bgc_df: pd.DataFrame,
    bgc_cache: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[BGCClassifier, dict[str, Any]]:
    split_frames = {
        split: bgc_df[bgc_df["split"] == split].drop_duplicates(subset=["bgc_id"]).sort_values("bgc_id").reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    label_vocab = sorted(
        {
            str(label)
            for labels in split_frames["train"]["bgc_class_list"].tolist()
            for label in labels
        }
    )
    if not label_vocab:
        raise ValueError("Raw BGC baseline training split does not contain any labels.")
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
    for split in ("val", "test"):
        unknown = sorted(
            {
                str(label)
                for labels in split_frames[split]["bgc_class_list"].tolist()
                for label in labels
                if str(label) not in label_to_idx
            }
        )
        if unknown:
            raise ValueError(f"Raw BGC baseline split '{split}' contains labels absent from train: {', '.join(unknown)}")

    x_train, y_train = _build_raw_bgc_features(split_frames["train"], bgc_cache, label_to_idx)
    x_val, y_val = _build_raw_bgc_features(split_frames["val"], bgc_cache, label_to_idx)
    x_test, y_test = _build_raw_bgc_features(split_frames["test"], bgc_cache, label_to_idx)
    input_dim = int(x_train.size(1))
    classifier = BGCClassifier(
        emb_dim=input_dim,
        num_classes=len(label_vocab),
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        classifier.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    pos_counts = y_train.sum(dim=0)
    neg_counts = y_train.size(0) - pos_counts
    pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts.clamp_min(1.0), torch.ones_like(pos_counts))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc=f"{output_dir.name} raw baseline", leave=False):
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(classifier(x), y)
            loss.backward()
            optimizer.step()

    val_probs = _predict_raw_classifier(
        classifier, split_frames["val"], bgc_cache, device, int(cfg["downstream"]["feature_batch_size"])
    )
    test_probs = _predict_raw_classifier(
        classifier, split_frames["test"], bgc_cache, device, int(cfg["downstream"]["feature_batch_size"])
    )
    thresholds = torch.tensor(
        [_best_threshold(y_val[:, idx], val_probs[:, idx]) for idx in range(len(label_vocab))],
        dtype=torch.float32,
    )
    metrics = {
        "label_vocab": label_vocab,
        "class_names": label_vocab,
        "input": "cached raw BGC encoder features",
        "loss": {
            "name": "BCEWithLogitsLoss",
            "pos_weight": [float(value) for value in pos_weight.cpu().tolist()],
        },
        "counts": {
            split: int(len(split_frames[split]))
            for split in ("train", "val", "test")
        },
        "val": _ensemble_multilabel_report(y_val, val_probs, thresholds, label_vocab),
        "test": _ensemble_multilabel_report(y_test, test_probs, thresholds, label_vocab),
        "thresholds": [float(value) for value in thresholds.tolist()],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "metrics": metrics,
            "label_vocab": label_vocab,
            "input_dim": input_dim,
        },
        output_dir / "raw_bgc_classifier.pt",
    )
    save_json(metrics, output_dir / "raw_bgc_metrics.json")
    return classifier, metrics


def _evaluate_raw_bgc_baseline_ensemble(
    fold_table: pd.DataFrame,
    val_folds: list[int],
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    outdir: Path,
) -> dict[str, Any]:
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")
    test_probs_parts: list[torch.Tensor] = []
    thresholds_parts: list[torch.Tensor] = []
    member_summaries: list[dict[str, Any]] = []
    test_bgc_ids: list[str] | None = None
    y_test: torch.Tensor | None = None
    label_vocab: list[str] | None = None

    for val_fold in val_folds:
        member_df = fold_table.copy()
        member_df["split"] = np.where(
            member_df["is_test"],
            "test",
            np.where(member_df["fold"] == int(val_fold), "val", "train"),
        )
        member_df["bgc_class_list"] = member_df["bgc_classes"].map(_parse_label_text)
        member_df = member_df[member_df["bgc_class_list"].map(len) > 0].copy()
        member_outdir = outdir / f"val_fold_{int(val_fold)}"
        classifier, metrics = _train_raw_bgc_baseline_member(
            bgc_df=member_df,
            bgc_cache=bgc_cache,
            cfg=cfg,
            device=device,
            output_dir=member_outdir,
        )
        current_vocab = [str(label) for label in metrics["label_vocab"]]
        if label_vocab is None:
            label_vocab = current_vocab
        elif label_vocab != current_vocab:
            raise ValueError(f"Raw BGC baseline label vocabulary changed for val fold {val_fold}")
        label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
        test_df = (
            member_df[member_df["split"] == "test"]
            .drop_duplicates(subset=["bgc_id"])
            .sort_values("bgc_id")
            .reset_index(drop=True)
        )
        current_test_ids = test_df["bgc_id"].astype(str).tolist()
        if test_bgc_ids is None:
            test_bgc_ids = current_test_ids
            y_test, _ = _build_label_matrix(test_df, label_to_idx)
        elif test_bgc_ids != current_test_ids:
            raise ValueError(f"Raw BGC baseline test BGC ordering changed for val fold {val_fold}")
        test_probs_parts.append(
            _predict_raw_classifier(
                classifier,
                test_df,
                bgc_cache,
                device,
                int(cfg["downstream"]["feature_batch_size"]),
            )
        )
        thresholds_parts.append(torch.tensor(metrics["thresholds"], dtype=torch.float32))
        member_summaries.append(
            {
                "val_fold": int(val_fold),
                "output_dir": str(member_outdir),
                "metrics": metrics,
            }
        )

    if label_vocab is None or y_test is None or test_bgc_ids is None:
        raise ValueError("No raw BGC baseline members were trained.")
    ensemble_probs = torch.stack(test_probs_parts, dim=0).mean(dim=0)
    thresholds = torch.stack(thresholds_parts, dim=0).mean(dim=0)
    ensemble_report = _ensemble_multilabel_report(y_test, ensemble_probs, thresholds, label_vocab)
    summary = {
        "tasks": ["bgc_class"],
        "bgc_class": {
            "label_vocab": label_vocab,
            "class_names": label_vocab,
            "test": ensemble_report,
        },
        "input": "cached raw BGC encoder features",
        "n_models": int(len(member_summaries)),
        "n_test_bgcs": int(len(test_bgc_ids)),
        "test_bgc_ids": test_bgc_ids,
        "threshold_source": "mean validation Youden threshold across raw BGC baseline members",
        "members": member_summaries,
    }
    save_json(summary, outdir / "raw_bgc_baseline_summary.json")
    return summary


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _aggregate_objects(values: list[Any]) -> Any:
    if not values:
        return None
    if all(_is_number(value) for value in values):
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "n": int(arr.size),
        }
    if all(isinstance(value, dict) for value in values):
        keys = sorted({key for value in values for key in value})
        aggregated: dict[str, Any] = {}
        for key in keys:
            child_values = [value[key] for value in values if key in value]
            child = _aggregate_objects(child_values)
            if child is not None:
                aggregated[key] = child
        return aggregated
    return None


def _make_model(cfg: dict[str, Any], bgc_dim: int, compound_dim: int, device: torch.device) -> DualEncoderCLIP:
    return DualEncoderCLIP(
        bgc_input_dim=bgc_dim,
        compound_input_dim=compound_dim,
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
    ).to(device)


def _load_member_model(checkpoint_path: Path, device: torch.device) -> DualEncoderCLIP:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = _make_model(
        ckpt["config"],
        bgc_dim=int(ckpt["bgc_input_dim"]),
        compound_dim=int(ckpt["compound_input_dim"]),
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _run_epoch(
    model: DualEncoderCLIP,
    loader,
    positive_pairs: set[tuple[str, str]],
    device: torch.device,
    optimizer: AdamW | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    running = 0.0
    count = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            bgc_features = batch["bgc_feature"].to(device)
            compound_features = batch["compound_feature"].to(device)
            positive_mask = _build_batch_positive_mask(
                bgc_ids=batch["bgc_id"],
                compound_ids=batch["compound_id"],
                positive_pairs=positive_pairs,
                device=device,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, _ = model(bgc_features, compound_features, positive_mask=positive_mask)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            running += float(loss.item()) * bgc_features.size(0)
            count += bgc_features.size(0)
    return running / max(count, 1)


def _train_one_member(
    interactions: pd.DataFrame,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    outdir: Path,
    patience: int,
) -> tuple[DualEncoderCLIP, dict[str, Any]]:
    bgc_cache_path, compound_cache_path = _get_cached_paths(cache_dir)
    bgc_dim, compound_dim = _infer_input_dims(bgc_cache_path, compound_cache_path)
    model = _make_model(cfg, bgc_dim=bgc_dim, compound_dim=compound_dim, device=device)

    train_loader = _build_loader(
        interactions=interactions,
        bgc_cache_path=bgc_cache_path,
        compound_cache_path=compound_cache_path,
        split="train",
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=True,
    )
    val_loader = _build_loader(
        interactions=interactions,
        bgc_cache_path=bgc_cache_path,
        compound_cache_path=compound_cache_path,
        split="val",
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
        shuffle=False,
    )
    train_positive_pairs = _build_positive_pair_set(interactions, split="train")
    val_positive_pairs = _build_positive_pair_set(interactions, split="val")
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "contrastive_model_best.pt"
    last_path = outdir / "contrastive_model_last.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in tqdm(range(1, int(cfg["train"]["epochs"]) + 1), desc=f"{outdir.name} epochs"):
        train_loss = _run_epoch(model, train_loader, train_positive_pairs, device, optimizer)
        val_loss = _run_epoch(model, val_loader, val_positive_pairs, device, optimizer=None)
        history.append({"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "bgc_input_dim": bgc_dim,
                    "compound_input_dim": compound_dim,
                    "best_epoch": best_epoch,
                    "best_score": -best_val_loss,
                    "best_val_loss": best_val_loss,
                    "selection_split": "val",
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "early_stopping_patience": int(patience),
                },
                best_path,
            )
            continue

        epochs_without_improvement += 1
        if epochs_without_improvement >= int(patience):
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "bgc_input_dim": bgc_dim,
            "compound_input_dim": compound_dim,
            "best_epoch": best_epoch,
            "best_score": -best_val_loss,
            "best_val_loss": best_val_loss,
            "selection_split": "val",
            "early_stopping_patience": int(patience),
        },
        last_path,
    )
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])

    metrics = {
        "train": {"loss_last_epoch": float(history[-1]["train_loss"]) if history else float("nan")},
        "model_selection": {
            "selection_split": "val",
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_checkpoint": str(best_path),
            "early_stopping_patience": int(patience),
            "epochs_ran": int(len(history)),
        },
        "history": history,
    }
    for split in ("train", "val", "test"):
        metrics[f"retrieval_{split}"] = evaluate_split_retrieval(
            model=model,
            interactions=interactions,
            split=split,
            bgc_cache_path=bgc_cache_path,
            compound_cache_path=compound_cache_path,
            device=device,
            sim_batch_size=int(cfg["eval"]["sim_batch_size"]),
        )
    save_json(metrics, outdir / "contrastive_metrics.json")
    return model, metrics


def _metrics_from_similarity(sim: torch.Tensor, pairs: list[tuple[int, int]]) -> dict[str, dict[str, float]]:
    n_left, n_right = sim.shape
    pos_left_to_right = torch.zeros((n_left, n_right), dtype=torch.bool)
    pos_right_to_left = torch.zeros((n_right, n_left), dtype=torch.bool)
    if pairs:
        pair_array = np.asarray(pairs, dtype=np.int64)
        left_idx = torch.tensor(pair_array[:, 0], dtype=torch.long)
        right_idx = torch.tensor(pair_array[:, 1], dtype=torch.long)
        pos_left_to_right[left_idx, right_idx] = True
        pos_right_to_left[right_idx, left_idx] = True

    def calculate(sorted_pos: torch.Tensor) -> dict[str, float]:
        has_pos = sorted_pos.any(dim=1)
        first_idx = sorted_pos.float().argmax(dim=1)
        ranks = torch.where(has_pos, first_idx + 1, torch.full_like(first_idx, fill_value=sorted_pos.size(1) + 1))
        return {
            "recall_at_1": float(sorted_pos[:, :1].any(dim=1).float().mean().item()),
            "recall_at_5": float(sorted_pos[:, :5].any(dim=1).float().mean().item()),
            "recall_at_10": float(sorted_pos[:, :10].any(dim=1).float().mean().item()),
            "precision_at_1": float((sorted_pos[:, :1].float().sum(dim=1) / 1.0).mean().item()),
            "precision_at_5": float((sorted_pos[:, :5].float().sum(dim=1) / 5.0).mean().item()),
            "precision_at_10": float((sorted_pos[:, :10].float().sum(dim=1) / 10.0).mean().item()),
            "mrr": float((1.0 / ranks.float()).mean().item()),
        }

    sorted_right = torch.argsort(sim, dim=1, descending=True)
    sorted_pos_left_to_right = pos_left_to_right.gather(1, sorted_right)
    sorted_left = torch.argsort(sim.t(), dim=1, descending=True)
    sorted_pos_right_to_left = pos_right_to_left.gather(1, sorted_left)
    return {
        "bgc_to_compound": calculate(sorted_pos_left_to_right),
        "compound_to_bgc": calculate(sorted_pos_right_to_left),
    }


def _evaluate_ensemble(
    models: list[DualEncoderCLIP],
    interactions: pd.DataFrame,
    cache_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    bgc_cache_path, compound_cache_path = _get_cached_paths(cache_dir)
    bgc_cache = torch.load(bgc_cache_path, map_location="cpu")
    compound_cache = torch.load(compound_cache_path, map_location="cpu")
    averaged_sim: torch.Tensor | None = None
    bgc_ids: list[str] = []
    compound_ids: list[str] = []
    pairs: list[tuple[int, int]] = []

    for member_idx, model in enumerate(models):
        bgc_index, compound_index, bgc_embs, compound_embs, model_pairs = build_unique_embeddings(
            model=model,
            interactions=interactions,
            split="test",
            bgc_cache=bgc_cache,
            compound_cache=compound_cache,
            device=device,
        )
        sim = model.get_logit_scale().detach().cpu() * (bgc_embs @ compound_embs.t())
        if averaged_sim is None:
            averaged_sim = sim
            bgc_ids = list(bgc_index.keys())
            compound_ids = list(compound_index.keys())
            pairs = model_pairs
        else:
            if bgc_ids != list(bgc_index.keys()) or compound_ids != list(compound_index.keys()):
                raise ValueError(f"Ensemble member {member_idx} produced incompatible test entity ordering.")
            averaged_sim += sim

    if averaged_sim is None:
        raise ValueError("No ensemble models were provided.")
    averaged_sim = averaged_sim / float(len(models))
    return {
        "metrics": _metrics_from_similarity(averaged_sim, pairs),
        "bgc_class_retrieval": evaluate_bgc_class_retrieval(
            sim=averaged_sim,
            bgc_ids=bgc_ids,
            compound_ids=compound_ids,
            pairs=pairs,
            interactions=interactions,
            split="test",
        ),
        "n_models": int(len(models)),
        "n_test_bgcs": int(len(bgc_ids)),
        "n_test_compounds": int(len(compound_ids)),
        "n_test_pairs": int(len(pairs)),
    }


def main() -> None:
    args = parse_args()
    logger = setup_logger("bgcmac_ensemble")
    cfg = apply_overrides(load_yaml(args.config), args.override)
    cfg["seed"] = int(args.seed)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir) if args.outdir is not None else Path("results") / "bgcmac_ensemble"
    outdir.mkdir(parents=True, exist_ok=True)

    fold_table = _load_bgcmac_fold_table(args.bgcmac_splits_path, test_fold=int(args.test_fold))
    val_folds = args.val_folds if args.val_folds is not None and len(args.val_folds) else sorted(
        int(fold_id) for fold_id in fold_table.loc[~fold_table["is_test"], "fold"].unique().tolist()
    )

    member_summaries: list[dict[str, Any]] = []
    models: list[DualEncoderCLIP] = []
    for val_fold in val_folds:
        member_outdir = outdir / f"val_fold_{int(val_fold)}"
        member_cfg = copy.deepcopy(cfg)
        member_cfg["output"]["dir"] = str(member_outdir)
        set_seed(int(args.seed) + int(val_fold))
        interactions = _build_bgcmac_interactions(args.data_dir, fold_table, val_fold=int(val_fold))
        counts = _split_counts(interactions)
        best_ckpt_path = member_outdir / "contrastive_model_best.pt"
        metrics_path = member_outdir / "contrastive_metrics.json"
        if bool(args.reuse_existing_members) and best_ckpt_path.exists():
            logger.info("Reusing BGC-MAC ensemble member checkpoint for val fold %s", val_fold)
            model = _load_member_model(best_ckpt_path, device=device)
            metrics = load_yaml(metrics_path) if metrics_path.exists() else {}
        else:
            logger.info("Training BGC-MAC ensemble member with val fold %s counts: %s", val_fold, counts)
            model, metrics = _train_one_member(
                interactions=interactions,
                cache_dir=args.cache_dir,
                cfg=member_cfg,
                device=device,
                outdir=member_outdir,
                patience=int(args.patience),
            )
        resolved_splits_path = _write_resolved_split_tsv(interactions, member_outdir / "bgcmac_resolved_splits.tsv")
        full_resolved_splits_path = _write_full_bgcmac_split_tsv(
            fold_table,
            val_fold=int(val_fold),
            output_path=member_outdir / "bgcmac_full_resolved_splits.tsv",
        )
        retrieval_baselines_test: dict[str, Any] = {}
        if bool(args.retrieval_baselines):
            retrieval_baselines_test = run_retrieval_baseline_suite(
                interactions=interactions,
                split="test",
                cache_dir=args.cache_dir,
                cfg=member_cfg,
                device=device,
                outdir=member_outdir / "retrieval_baselines",
                seed=int(args.seed) + int(val_fold),
                random_trials=int(args.baseline_random_trials),
                k_values=[int(k) for k in args.baseline_k_values],
                patience=int(args.patience),
            )
        downstream_metrics = train_downstream(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            contrastive_ckpt=member_outdir / "contrastive_model_best.pt",
            cfg=member_cfg,
            device=device,
            splits_path=resolved_splits_path,
            save_cm_png=bool(args.save_cm_png),
            tasks=("bgc_class",),
        )
        full_member_cfg = copy.deepcopy(member_cfg)
        full_member_cfg["output"]["dir"] = str(member_outdir / "downstream_full_bgcmac")
        downstream_full_metrics = train_downstream(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            contrastive_ckpt=member_outdir / "contrastive_model_best.pt",
            cfg=full_member_cfg,
            device=device,
            splits_path=full_resolved_splits_path,
            save_cm_png=bool(args.save_cm_png),
            tasks=("bgc_class",),
        )
        models.append(model)
        member_summary = {
            "val_fold": int(val_fold),
            "output_dir": str(member_outdir),
            "resolved_splits_path": str(resolved_splits_path),
            "full_bgcmac_output_dir": str(full_member_cfg["output"]["dir"]),
            "full_resolved_splits_path": str(full_resolved_splits_path),
            "counts": counts,
            "metrics": metrics,
            "retrieval_baselines_test": retrieval_baselines_test,
            "downstream": downstream_metrics,
            "downstream_full_bgcmac": downstream_full_metrics,
        }
        save_json(member_summary, member_outdir / "member_summary.json")
        member_summaries.append(member_summary)

    ensemble_downstream = _evaluate_downstream_prediction_ensemble(
        member_summaries=member_summaries,
        models=models,
        cfg=cfg,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        device=device,
    )
    save_json(ensemble_downstream, outdir / "ensemble_downstream_metrics.json")

    ensemble_downstream_full = _evaluate_downstream_prediction_ensemble(
        member_summaries=member_summaries,
        models=models,
        cfg=cfg,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        device=device,
        split_path_key="full_resolved_splits_path",
        classifier_dir_key="full_bgcmac_output_dir",
    )
    save_json(ensemble_downstream_full, outdir / "ensemble_downstream_full_bgcmac_metrics.json")

    raw_bgc_baseline = _evaluate_raw_bgc_baseline_ensemble(
        fold_table=fold_table,
        val_folds=[int(fold) for fold in val_folds],
        cache_dir=args.cache_dir,
        cfg=cfg,
        device=device,
        outdir=outdir / "raw_bgc_baseline",
    )
    bgcmac_artifacts: dict[str, Any] = {}
    if bool(args.save_cm_png):
        bgcmac_artifacts = save_bgcmac_benchmark_artifacts(
            strict_report=ensemble_downstream,
            full_report=ensemble_downstream_full,
            baseline_report=raw_bgc_baseline,
            output_dir=outdir / "bgcmac_benchmark_artifacts",
        )

    summary = {
        "protocol": "BGC-MAC fixed fold-10 test with folds 1-9 rotated as validation folds across 9 members",
        "benchmark": "BGC-MAC",
        "benchmark_task": "natural_product_classification",
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "bgcmac_splits_path": str(args.bgcmac_splits_path),
        "test_fold": int(args.test_fold),
        "val_folds": [int(fold) for fold in val_folds],
        "patience": int(args.patience),
        "members": member_summaries,
        "strict_clip_covered_note": "Uses only BGCs present in processed BGC-NP pairs; BGCs without usable SMILES are excluded.",
        "ensemble_downstream": ensemble_downstream,
        "ensemble_downstream_full_bgcmac": ensemble_downstream_full,
        "raw_bgc_baseline": raw_bgc_baseline,
        "bgcmac_artifacts": bgcmac_artifacts,
        "aggregate": {
            "counts": _aggregate_objects([summary["counts"] for summary in member_summaries]),
            "contrastive_metrics": _aggregate_objects([summary["metrics"] for summary in member_summaries]),
            "retrieval_baselines_test": _aggregate_objects(
                [
                    summary["retrieval_baselines_test"]
                    for summary in member_summaries
                    if "retrieval_baselines_test" in summary
                ]
            ),
            "downstream": _aggregate_objects([summary["downstream"] for summary in member_summaries]),
            "downstream_full_bgcmac": _aggregate_objects(
                [summary["downstream_full_bgcmac"] for summary in member_summaries]
            ),
        },
    }
    summary_path = outdir / "summary.json"
    save_json(summary, summary_path)
    logger.info("Saved BGC-MAC ensemble summary to %s", summary_path)

    try:
        baseline_artifacts = save_all_baseline_artifacts(outdir)
        summary["baseline_artifacts"] = baseline_artifacts
        save_json(summary, summary_path)
        logger.info("Saved visible baseline artifacts to %s", outdir / "baselines")
    except Exception as exc:
        logger.warning("Could not create visible baseline artifacts: %s", exc)

    if bool(args.save_cm_png):
        try:
            from scripts.plot_cv_summary_confusion_matrices import plot_summary

            outputs = plot_summary(summary_path, outdir / "summary_confusion_matrices", suffix="bgcmac_mean")
            logger.info(
                "Saved %d aggregate BGC-MAC downstream summary plots to %s",
                len(outputs),
                outdir / "summary_confusion_matrices",
            )
        except Exception as exc:
            logger.warning("Could not create aggregate BGC-MAC downstream summary plots: %s", exc)


if __name__ == "__main__":
    main()
