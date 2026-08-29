from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts._bootstrap import ensure_src_path
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.logging import save_json
from projects.mibig_bgc_np.data.datasets import build_bgc_class_table
from projects.mibig_bgc_np.eval.classification_metrics import (
    compute_confusion_matrix,
    confusion_matrix_normalized,
    macro_micro_f1_from_cm,
    per_class_prf,
)
from projects.mibig_bgc_np.eval.regression_metrics import pearson, rmse, spearman
from projects.mibig_bgc_np.training.downstream_trainer import (
    BIOACTIVITY_CLASS_NAMES,
    COMPOUND_REGRESSION_TASKS,
    NPCLASSIFIER_TASKS,
    ORIGIN_CLASS_NAMES,
    ORIGIN_LABEL_TO_IDX,
    _binary_roc_auc,
    _binary_roc_curve,
    _build_bgc_split_assignments,
    _load_bioactivity_class_table,
    _load_npclassifier_bgc_label_table,
    _matrix_with_class_names,
    _multilabel_overall_metrics,
    _multilabel_per_class_metrics,
)


TASK_DISPLAY = {
    "bgc_class": "BGC class",
    "bioactivity_class": "Bioactivity",
    "npclassifier_pathway": "NPClassifier pathway",
    "npclassifier_superclass": "NPClassifier superclass",
    "npclassifier_class": "NPClassifier class",
    "compound_mw": "Molecular weight",
    "compound_logp": "logP",
    "compound_tpsa": "TPSA",
    "origin_type": "Origin type",
}
SPLIT_PATHS = {
    "bgc": Path("data/MIBIG/splits/bgc_cv_seed42_n10.tsv"),
    "np": Path("data/MIBIG/splits/np_cv_seed42_n10.tsv"),
    "combined": Path("data/MIBIG/splits/combined_cv_seed42_n10.tsv"),
    "strict": Path("data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10.tsv"),
}
BGC_TASKS = {
    "bgc_class",
    "bioactivity_class",
    "npclassifier_pathway",
    "npclassifier_superclass",
    "npclassifier_class",
}
ALL_TASKS = tuple(TASK_DISPLAY)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_split_name(run_root: Path) -> str:
    name = run_root.name
    for split in ("combined", "strict", "bgc", "np"):
        if f"_{split}_" in name:
            return split
    raise ValueError(f"Could not infer split name from run root: {run_root}")


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy()


def _stack_features(ids: list[str], cache: dict[str, torch.Tensor]) -> torch.Tensor:
    if not ids:
        dim = int(next(iter(cache.values())).numel())
        return torch.empty((0, dim), dtype=torch.float32)
    missing = [item for item in ids if item not in cache]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing cached features for {len(missing)} ids. Examples: {preview}")
    return torch.tensor(np.stack([_tensor_to_numpy(cache[item]) for item in ids]), dtype=torch.float32)


def _standardize(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std


def _make_multilabel_dataset(
    frames: dict[str, pd.DataFrame],
    cache: dict[str, torch.Tensor],
    label_vocab: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
    xs: dict[str, torch.Tensor] = {}
    ys: dict[str, torch.Tensor] = {}
    for split, frame in frames.items():
        frame = frame.reset_index(drop=True)
        xs[split] = _stack_features(frame["bgc_id"].astype(str).tolist(), cache)
        y = torch.zeros((len(frame), len(label_vocab)), dtype=torch.float32)
        for row_idx, labels_for_bgc in enumerate(frame["bgc_class_list"].tolist()):
            for label in labels_for_bgc:
                if str(label) in label_to_idx:
                    y[row_idx, label_to_idx[str(label)]] = 1.0
        ys[split] = y
    return xs, ys


def _make_compound_dataset(
    frames: dict[str, pd.DataFrame],
    cache: dict[str, torch.Tensor],
    target: str,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    xs: dict[str, torch.Tensor] = {}
    ys: dict[str, torch.Tensor] = {}
    for split, frame in frames.items():
        frame = frame.reset_index(drop=True)
        xs[split] = _stack_features(frame["compound_id"].astype(str).tolist(), cache)
        ys[split] = torch.tensor(frame[target].tolist(), dtype=dtype)
    return xs, ys


def _fit_linear_multilabel(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> LinearProbe:
    torch.manual_seed(seed)
    model = LinearProbe(x_train.size(1), y_train.size(1))
    pos_counts = y_train.sum(dim=0)
    neg_counts = y_train.size(0) - pos_counts
    pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts.clamp_min(1.0), torch.ones_like(pos_counts))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    return model.eval()


def _fit_linear_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> LinearProbe:
    torch.manual_seed(seed)
    model = LinearProbe(x_train.size(1), int(y_train.max().item()) + 1)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    return model.eval()


def _ridge_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    x_np = x_train.detach().cpu().numpy().astype(np.float64)
    y_np = y_train.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
    eval_np = x_eval.detach().cpu().numpy().astype(np.float64)
    x_aug = np.concatenate([x_np, np.ones((x_np.shape[0], 1), dtype=np.float64)], axis=1)
    eval_aug = np.concatenate([eval_np, np.ones((eval_np.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_np)
    return torch.tensor((eval_aug @ coef).reshape(-1), dtype=torch.float32)


def _multilabel_report(model: nn.Module, x: torch.Tensor, y_true: torch.Tensor, class_names: list[str]) -> dict[str, Any]:
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        y_pred = (probs >= 0.5).to(dtype=torch.float32)
    overall = _multilabel_overall_metrics(y_true, y_pred, class_names)
    per_class = _multilabel_per_class_metrics(y_true, y_pred, class_names)
    roc_curves: dict[str, Any] = {}
    for idx, class_name in enumerate(class_names):
        fpr, tpr, auc = _binary_roc_curve(y_true[:, idx], probs[:, idx])
        per_class[class_name]["auroc"] = float(auc)
        roc_curves[class_name] = {"fpr": fpr, "tpr": tpr, "auroc": float(auc)}
    micro_fpr, micro_tpr, micro_auroc = _binary_roc_curve(y_true.reshape(-1), probs.reshape(-1))
    macro_auroc = float(np.mean([curve["auroc"] for curve in roc_curves.values()])) if roc_curves else 0.0
    top1_pred = logits.argmax(dim=-1) if logits.numel() else torch.empty(0, dtype=torch.long)
    cm = torch.zeros((len(class_names), len(class_names)), dtype=torch.long)
    for row_idx in range(y_true.size(0)):
        pred_idx = int(top1_pred[row_idx].item())
        for true_idx in torch.nonzero(y_true[row_idx] > 0, as_tuple=False).flatten().tolist():
            cm[true_idx, pred_idx] += 1
    overall["micro_auroc"] = float(micro_auroc)
    overall["macro_auroc"] = macro_auroc
    return {
        "accuracy": float(overall["accuracy"]),
        "macro_f1": float(overall["macro_f1"]),
        "micro_f1": float(overall["micro_f1"]),
        "macro_auroc": macro_auroc,
        "micro_auroc": float(micro_auroc),
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "raw": _matrix_with_class_names(cm, class_names),
            "normalized_true": _matrix_with_class_names(confusion_matrix_normalized(cm, mode="true"), class_names),
        },
        "roc_curves": {"per_class": roc_curves, "micro": {"fpr": micro_fpr, "tpr": micro_tpr, "auroc": micro_auroc}},
    }


def _classification_report(model: nn.Module, x: torch.Tensor, y_true: torch.Tensor, class_names: list[str]) -> dict[str, Any]:
    with torch.no_grad():
        logits = model(x)
        y_pred = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
    cm = compute_confusion_matrix(y_true, y_pred, len(class_names))
    overall = macro_micro_f1_from_cm(cm)
    per_class_raw = per_class_prf(cm)
    per_class = {
        class_name: {
            "precision": float(per_class_raw["precision"][idx]),
            "recall": float(per_class_raw["recall"][idx]),
            "f1": float(per_class_raw["f1"][idx]),
            "support": float(per_class_raw["support"][idx]),
        }
        for idx, class_name in enumerate(class_names)
    }
    out: dict[str, Any] = {
        "accuracy": float(overall["accuracy"]),
        "macro_f1": float(overall["macro_f1"]),
        "micro_f1": float(overall["micro_f1"]),
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "raw": _matrix_with_class_names(cm, class_names),
            "normalized_true": _matrix_with_class_names(confusion_matrix_normalized(cm, mode="true"), class_names),
        },
    }
    if len(class_names) == 2 and probs.numel():
        out["roc_auc"] = _binary_roc_auc(y_true, probs[:, 1])
        out["overall"]["roc_auc"] = out["roc_auc"]
    return out


def _regression_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, Any]:
    true_np = y_true.detach().to(dtype=torch.float64, device="cpu").numpy()
    pred_np = y_pred.detach().to(dtype=torch.float64, device="cpu").numpy()
    mse = float(np.mean((true_np - pred_np) ** 2)) if true_np.size else 0.0
    ss_res = float(np.sum((true_np - pred_np) ** 2))
    true_mean = float(true_np.mean()) if true_np.size else 0.0
    ss_tot = float(np.sum((true_np - true_mean) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else 1.0 - (ss_res / ss_tot)
    report = {
        "mse": mse,
        "rmse": rmse(true_np, pred_np) if true_np.size else 0.0,
        "r2": float(r2),
        "pearson": pearson(true_np, pred_np) if true_np.size else 0.0,
        "spearman": spearman(true_np, pred_np) if true_np.size else 0.0,
    }
    report["overall"] = dict(report)
    return report


def _split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {split: df[df["split"] == split].reset_index(drop=True) for split in ("train", "val", "test")}


def _metric_rows(
    *,
    split_name: str,
    fold: int,
    eval_split: str,
    task_key: str,
    task_type: str,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = (
        ("macro_auroc", "micro_auroc", "macro_f1", "micro_f1", "accuracy")
        if task_type == "classification"
        else ("rmse", "mse", "r2", "pearson", "spearman")
    )
    for metric in metric_names:
        value = report.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            rows.append(
                {
                    "split": split_name,
                    "fold": fold,
                    "eval_split": eval_split,
                    "task_key": task_key,
                    "task": TASK_DISPLAY[task_key],
                    "type": task_type,
                    "baseline": "linear_encoder_probe",
                    "metric": metric,
                    "value": float(value),
                }
            )
    per_class = report.get("per_class")
    if isinstance(per_class, dict):
        for class_name, metrics in per_class.items():
            if not isinstance(metrics, dict):
                continue
            for metric in ("auroc", "f1", "precision", "recall"):
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    rows.append(
                        {
                            "split": split_name,
                            "fold": fold,
                            "eval_split": eval_split,
                            "task_key": task_key,
                            "task": TASK_DISPLAY[task_key],
                            "type": task_type,
                            "baseline": "linear_encoder_probe",
                            "metric": metric,
                            "class": str(class_name),
                            "value": float(value),
                        }
                    )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    group_cols = ["split", "eval_split", "task_key", "task", "type", "baseline", "metric"]
    if "class" in df.columns:
        df["class"] = df["class"].fillna("")
        group_cols.append("class")
    return (
        df.groupby(group_cols, dropna=False)["value"]
        .agg(value="mean", std=lambda x: float(np.std(x, ddof=0)), n_folds="count")
        .reset_index()
        .sort_values(group_cols)
    )


def _run_bgc_task(
    task: str,
    *,
    data_dir: Path,
    cache: dict[str, torch.Tensor],
    splits_path: Path,
    fold: int,
    val_fold: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if task == "bgc_class":
        target_df = build_bgc_class_table(data_dir, splits_path=splits_path, cv_fold=fold, val_fold=val_fold)
        label_vocab = sorted(
            {
                str(label)
                for labels in target_df[target_df["split"] == "train"]["bgc_class_list"].tolist()
                for label in labels
            }
        )
    elif task == "bioactivity_class":
        target_df = _load_bioactivity_class_table(
            data_dir=data_dir,
            bioactivity_table_path=args.bioactivity_table_path,
            splits_path=splits_path,
            cv_fold=fold,
            val_fold=val_fold,
        )
        label_vocab = BIOACTIVITY_CLASS_NAMES
    elif task in NPCLASSIFIER_TASKS:
        target_df, label_vocab, _ = _load_npclassifier_bgc_label_table(
            data_dir=data_dir,
            npclassifier_pair_labels_path=args.npclassifier_pair_labels_path,
            task_name=task,
            splits_path=splits_path,
            cv_fold=fold,
            val_fold=val_fold,
        )
    else:
        raise ValueError(f"Unsupported BGC task: {task}")

    frames = _split_frames(target_df)
    xs, ys = _make_multilabel_dataset(frames, cache, label_vocab)
    xs["train"], xs["val"], xs["test"] = _standardize(xs["train"], xs["val"], xs["test"])
    model = _fit_linear_multilabel(
        xs["train"],
        ys["train"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed + fold,
    )
    report: dict[str, Any] = {
        "task": task,
        "input": "cached raw BGC encoder features",
        "model": "linear_probe",
        "label_vocab": label_vocab,
        "dataset_sizes": {split: int(len(frame)) for split, frame in frames.items()},
    }
    rows: list[dict[str, Any]] = []
    for eval_split in ("val", "test"):
        if xs[eval_split].numel() == 0:
            continue
        report[eval_split] = _multilabel_report(model, xs[eval_split], ys[eval_split], label_vocab)
        rows.extend(
            _metric_rows(
                split_name=args.split_name,
                fold=fold,
                eval_split=eval_split,
                task_key=task,
                task_type="classification",
                report=report[eval_split],
            )
        )
    return report, rows


def _run_compound_task(
    task: str,
    *,
    compound_cache: dict[str, torch.Tensor],
    matched_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not matched_path.exists():
        raise FileNotFoundError(f"Missing matched compounds table: {matched_path}")
    matched = pd.read_csv(matched_path, sep="\t")
    if task == "origin_type":
        task_df = matched[matched["origin_type"].isin(ORIGIN_LABEL_TO_IDX.keys())].copy()
        task_df["origin_label"] = task_df["origin_type"].map(ORIGIN_LABEL_TO_IDX)
        target_col = "origin_label"
        task_type = "classification"
        dtype = torch.long
    else:
        target_col = str(COMPOUND_REGRESSION_TASKS[task]["target"])
        task_df = matched.dropna(subset=[target_col]).copy()
        task_df[target_col] = pd.to_numeric(task_df[target_col], errors="coerce")
        task_df = task_df.dropna(subset=[target_col]).reset_index(drop=True)
        task_type = "regression"
        dtype = torch.float32

    frames = _split_frames(task_df)
    xs, ys = _make_compound_dataset(frames, compound_cache, target_col, dtype)
    xs["train"], xs["val"], xs["test"] = _standardize(xs["train"], xs["val"], xs["test"])
    report: dict[str, Any] = {
        "task": task,
        "input": "cached raw compound encoder features",
        "model": "linear_probe" if task_type == "classification" else "ridge_regression",
        "dataset_sizes": {split: int(len(frame)) for split, frame in frames.items()},
    }
    rows: list[dict[str, Any]] = []
    if task_type == "classification":
        model = _fit_linear_classifier(
            xs["train"],
            ys["train"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + args.fold,
        )
        report["class_names"] = ORIGIN_CLASS_NAMES
        for eval_split in ("val", "test"):
            if xs[eval_split].numel() == 0:
                continue
            report[eval_split] = _classification_report(model, xs[eval_split], ys[eval_split], ORIGIN_CLASS_NAMES)
            rows.extend(
                _metric_rows(
                    split_name=args.split_name,
                    fold=args.fold,
                    eval_split=eval_split,
                    task_key=task,
                    task_type="classification",
                    report=report[eval_split],
                )
            )
    else:
        for eval_split in ("val", "test"):
            if xs[eval_split].numel() == 0:
                continue
            y_pred = _ridge_predict(xs["train"], ys["train"], xs[eval_split], alpha=args.ridge_alpha)
            report[eval_split] = _regression_report(ys[eval_split], y_pred)
            rows.extend(
                _metric_rows(
                    split_name=args.split_name,
                    fold=args.fold,
                    eval_split=eval_split,
                    task_key=task,
                    task_type="regression",
                    report=report[eval_split],
                )
            )
    return report, rows


def run_for_root(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = _load_json(run_root / "summary.json")
    split_name = args.split_name or _infer_split_name(run_root)
    splits_path = args.splits_path or Path(summary.get("splits_path") or SPLIT_PATHS[split_name])
    cache_dir = args.cache_dir or Path(summary.get("cache_dir") or "cache/ohe")
    data_dir = args.data_dir or Path(summary.get("data_dir") or "data/MIBIG/processed")
    outdir = run_root / "baselines" / "downstream_encoder"
    outdir.mkdir(parents=True, exist_ok=True)

    bgc_cache = torch.load(cache_dir / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(cache_dir / "compound_features.pt", map_location="cpu")
    n_folds = int(args.n_folds)
    tasks = list(args.task or ALL_TASKS)
    all_rows: list[dict[str, Any]] = []
    fold_reports: dict[str, Any] = {}

    for fold in range(1, n_folds + 1):
        val_fold = (fold % n_folds) + 1
        fold_dir = run_root / f"fold_{fold}"
        fold_out = outdir / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        fold_report: dict[str, Any] = {"fold": fold, "val_fold": val_fold, "tasks": {}}
        for task in tasks:
            run_args = argparse.Namespace(**vars(args))
            run_args.split_name = split_name
            run_args.fold = fold
            if task in BGC_TASKS:
                report, rows = _run_bgc_task(
                    task,
                    data_dir=data_dir,
                    cache=bgc_cache,
                    splits_path=splits_path,
                    fold=fold,
                    val_fold=val_fold,
                    args=run_args,
                )
            else:
                report, rows = _run_compound_task(
                    task,
                    compound_cache=compound_cache,
                    matched_path=fold_dir / "matched_compounds.tsv",
                    args=run_args,
                )
            fold_report["tasks"][task] = report
            all_rows.extend(rows)
        save_json(fold_report, fold_out / "downstream_encoder_baselines.json")
        fold_reports[f"fold_{fold}"] = fold_report

    long_df = pd.DataFrame(all_rows)
    long_path = outdir / "downstream_encoder_baselines_long.csv"
    long_df.to_csv(long_path, index=False)
    summary_df = _aggregate_rows(all_rows)
    summary_path = outdir / "downstream_encoder_baselines_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    manifest = {
        "run_root": str(run_root),
        "split": split_name,
        "splits_path": str(splits_path),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "tasks": tasks,
        "n_folds": n_folds,
        "baseline": {
            "classification": "linear probe on cached encoder features",
            "regression": "ridge regression on cached encoder features",
            "uses_clip_checkpoint": False,
        },
        "long_csv": str(long_path),
        "summary_csv": str(summary_path),
        "fold_reports": fold_reports,
    }
    save_json(manifest, outdir / "downstream_encoder_baseline_artifacts.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple encoder-feature downstream baselines for CV runs.")
    parser.add_argument("--run_root", action="append", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--splits_path", type=Path, default=None)
    parser.add_argument("--split_name", type=str, default=None)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--task", action="append", choices=ALL_TASKS, default=None)
    parser.add_argument("--bioactivity_table_path", type=Path, default=Path("results/EDA/bgc_observed_bioactivities.csv"))
    parser.add_argument(
        "--npclassifier_pair_labels_path",
        type=Path,
        default=Path("data/MIBIG/processed/mibig_pairs_npclassifier_labels.tsv"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = [run_for_root(run_root, args) for run_root in args.run_root]
    if len(manifests) > 1:
        combined_dir = Path("results/downstream_metric_tables")
        combined_dir.mkdir(parents=True, exist_ok=True)
        long_parts = [pd.read_csv(manifest["long_csv"]) for manifest in manifests]
        combined_long = pd.concat(long_parts, ignore_index=True)
        combined_long.to_csv(combined_dir / "encoder_baseline_downstream_metrics_long.csv", index=False)
        combined_summary = _aggregate_rows(combined_long.to_dict("records"))
        combined_summary.to_csv(combined_dir / "encoder_baseline_downstream_metrics_summary.csv", index=False)
        save_json(
            {"runs": manifests, "long_csv": str(combined_dir / "encoder_baseline_downstream_metrics_long.csv")},
            combined_dir / "encoder_baseline_downstream_metrics_manifest.json",
        )


if __name__ == "__main__":
    main()
