from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from clip_core.logging import save_json
from kiba_clip.eval.classification_metrics import (
    compute_confusion_matrix,
    confusion_matrix_normalized,
    macro_micro_f1_from_cm,
    per_class_prf,
    random_baselines,
    wrong_class_ratios,
)
from kiba_clip.eval.regression_metrics import rmse, spearman
from projects.mibig_bgc_np.data.datasets import build_bgc_class_table, build_interactions
from projects.mibig_bgc_np.models.classification import BGCClassifier
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.models.regression import EmbeddingRegressor

LOGGER = logging.getLogger("mibig_bgc_np")
DEFAULT_TASKS = ("bgc_class",)
COMPOUND_TASKS = {"compound_mw", "origin_type"}
ORIGIN_LABEL_TO_IDX = {"Bacterium": 0, "Fungus": 1}
ORIGIN_CLASS_NAMES = ["Bacterium", "Fungus"]


def _load_contrastive_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DualEncoderCLIP, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = DualEncoderCLIP(
        bgc_input_dim=ckpt["bgc_input_dim"],
        compound_input_dim=ckpt["compound_input_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _predict_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    classifier.eval()
    if len(loader.dataset) == 0:
        empty = torch.empty(0, dtype=torch.long)
        return float("nan"), empty, empty, torch.empty((0, 0), dtype=torch.float32)

    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            running_loss += float(loss.item()) * x.size(0)
            count += x.size(0)
            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())

    y_true = torch.cat(targets_all)
    logits = torch.cat(logits_all)
    y_pred = logits.argmax(dim=-1)
    return running_loss / max(count, 1), y_true, y_pred, logits


def _predict_regressor(
    regressor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    regressor.eval()
    if len(loader.dataset) == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return float("nan"), empty, empty

    preds_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = regressor(x)
            loss = loss_fn(preds, y)
            running_loss += float(loss.item()) * x.size(0)
            count += x.size(0)
            preds_all.append(preds.cpu())
            targets_all.append(y.cpu())

    y_true = torch.cat(targets_all)
    y_pred = torch.cat(preds_all)
    return running_loss / max(count, 1), y_true, y_pred


def _per_class_with_names(cm: torch.Tensor, class_names: list[str]) -> dict[str, Any]:
    per_class = per_class_prf(cm)
    return {
        class_name: {
            "precision": float(per_class["precision"][idx]),
            "recall": float(per_class["recall"][idx]),
            "f1": float(per_class["f1"][idx]),
            "support": float(per_class["support"][idx]),
        }
        for idx, class_name in enumerate(class_names)
    }


def _matrix_with_class_names(matrix: torch.Tensor, class_names: list[str]) -> dict[str, dict[str, int | float]]:
    values = matrix.detach().cpu().tolist()
    named: dict[str, dict[str, int | float]] = {}
    for true_idx, true_name in enumerate(class_names):
        named[true_name] = {}
        for pred_idx, pred_name in enumerate(class_names):
            value = values[true_idx][pred_idx]
            named[true_name][pred_name] = int(value) if isinstance(value, int) else float(value)
    return named


def _matrix_from_named(matrix: dict[str, dict[str, int | float]], class_names: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[matrix[true_name][pred_name] for pred_name in class_names] for true_name in class_names],
        dtype=torch.float32,
    )


def _binary_roc_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    true_np = y_true.detach().to(dtype=torch.long, device="cpu").numpy()
    score_np = y_score.detach().to(dtype=torch.float64, device="cpu").numpy()
    pos_mask = true_np == 1
    neg_mask = true_np == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(score_np, kind="mergesort")
    sorted_scores = score_np[order]
    ranks = np.arange(1, len(score_np) + 1, dtype=np.float64)

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = float(ranks[start:end].mean())
        ranks[start:end] = avg_rank
        start = end

    inv_ranks = np.empty_like(ranks)
    inv_ranks[order] = ranks
    rank_sum_pos = float(inv_ranks[pos_mask].sum())
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _classification_report(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_train: torch.Tensor,
    num_classes: int,
    class_names: list[str],
    baseline_trials: int,
    baseline_seed: int,
) -> dict[str, Any]:
    loss_fn = nn.CrossEntropyLoss()
    loss, y_true, y_pred, logits = _predict_classifier(classifier, loader, device, loss_fn)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    overall = macro_micro_f1_from_cm(cm)
    overall["loss"] = float(loss)

    report: dict[str, Any] = {
        "loss": float(loss),
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "micro_f1": overall["micro_f1"],
        "overall": overall,
        "per_class": _per_class_with_names(cm, class_names),
        "confusion_matrix": {
            "labels": class_names,
            "raw": _matrix_with_class_names(cm, class_names),
            "normalized_true": _matrix_with_class_names(confusion_matrix_normalized(cm, mode="true"), class_names),
        },
        "wrong_ratios": wrong_class_ratios(y_true, y_pred, num_classes, class_names=class_names),
        "random_baselines": random_baselines(
            y_train=y_train,
            y_true=y_true,
            num_classes=num_classes,
            trials=baseline_trials,
            seed=baseline_seed,
        ),
    }

    if num_classes == 2 and logits.numel() > 0:
        probs = torch.softmax(logits, dim=-1)[:, 1]
        positive_name = class_names[1]
        positive_metrics = report["per_class"][positive_name]
        report["positive_class"] = {
            "label": positive_name,
            "precision": float(positive_metrics["precision"]),
            "recall": float(positive_metrics["recall"]),
            "f1": float(positive_metrics["f1"]),
        }
        report["roc_auc"] = _binary_roc_auc(y_true, probs)
        report["overall"]["roc_auc"] = report["roc_auc"]

    return report


def _regression_metrics_dict(y_true: torch.Tensor, y_pred: torch.Tensor, loss: float) -> dict[str, float]:
    true_np = y_true.detach().to(dtype=torch.float64, device="cpu").numpy()
    pred_np = y_pred.detach().to(dtype=torch.float64, device="cpu").numpy()
    mse = float(np.mean((true_np - pred_np) ** 2)) if true_np.size else 0.0
    ss_res = float(np.sum((true_np - pred_np) ** 2))
    true_mean = float(true_np.mean()) if true_np.size else 0.0
    ss_tot = float(np.sum((true_np - true_mean) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else 1.0 - (ss_res / ss_tot)
    return {
        "loss": float(loss),
        "mse": mse,
        "rmse": rmse(true_np, pred_np) if true_np.size else 0.0,
        "r2": float(r2),
        "spearman": spearman(true_np, pred_np) if true_np.size else 0.0,
    }


def _summarize_regression_trials(trial_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in ("mse", "rmse", "r2", "spearman"):
        values = np.asarray([metrics[metric_name] for metrics in trial_metrics], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean()) if values.size else 0.0
        summary[f"{metric_name}_std"] = float(values.std(ddof=0)) if values.size else 0.0
    return summary


def _regression_baselines(
    y_train: torch.Tensor,
    y_true: torch.Tensor,
    trials: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    train_np = y_train.detach().to(dtype=torch.float64, device="cpu").numpy()
    true_np = y_true.detach().to(dtype=torch.float64, device="cpu").numpy()
    if train_np.size == 0:
        raise ValueError("y_train must contain at least one target.")

    train_mean = float(train_np.mean())
    mean_pred = torch.full_like(y_true, fill_value=train_mean, dtype=torch.float32)
    mean_metrics = _regression_metrics_dict(y_true, mean_pred, loss=float("nan"))
    for metric_name in ("mse", "rmse", "r2", "spearman"):
        mean_metrics[f"{metric_name}_mean"] = float(mean_metrics[metric_name])
        mean_metrics[f"{metric_name}_std"] = 0.0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm_trials: list[dict[str, float]] = []
    if y_true.numel() == 0:
        perm_summary = {f"{name}_{suffix}": 0.0 for name in ("mse", "rmse", "r2", "spearman") for suffix in ("mean", "std")}
    else:
        for _ in range(int(trials)):
            perm = torch.randperm(y_true.numel(), generator=generator)
            shuffled = y_true[perm]
            perm_trials.append(_regression_metrics_dict(y_true, shuffled, loss=float("nan")))
        perm_summary = _summarize_regression_trials(perm_trials)

    return {
        "train_mean": mean_metrics,
        "permutation": perm_summary,
    }


def _regression_report(
    regressor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_train: torch.Tensor,
    baseline_trials: int,
    baseline_seed: int,
) -> dict[str, Any]:
    loss_fn = nn.MSELoss()
    loss, y_true, y_pred = _predict_regressor(regressor, loader, device, loss_fn)
    report = _regression_metrics_dict(y_true, y_pred, loss)
    report["overall"] = dict(report)
    report["random_baselines"] = _regression_baselines(
        y_train=y_train,
        y_true=y_true,
        trials=baseline_trials,
        seed=baseline_seed,
    )
    return report


def _save_confusion_matrix_png(report: dict[str, Any], class_names: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    cm_data = report["confusion_matrix"]
    cm_norm = _matrix_from_named(cm_data["normalized_true"], class_names)
    cm_raw = _matrix_from_named(cm_data["raw"], class_names).to(dtype=torch.long)
    fig_width = max(7.0, min(22.0, 0.75 * len(class_names)))
    fig_height = max(6.0, min(20.0, 0.65 * len(class_names)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Row-normalized confusion matrix")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    text_size = 8 if len(class_names) <= 12 else 6
    for true_idx in range(cm_norm.size(0)):
        for pred_idx in range(cm_norm.size(1)):
            norm_value = float(cm_norm[true_idx, pred_idx].item())
            raw_value = int(cm_raw[true_idx, pred_idx].item())
            if raw_value == 0:
                label = "0"
            else:
                label = f"{norm_value:.2f}\n({raw_value})"
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
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_wrong_ratio_png(report: dict[str, Any], class_names: list[str], split: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    true_ratios = report["wrong_ratios"]["ratio_true_among_wrongs"]
    pred_ratios = report["wrong_ratios"]["ratio_pred_among_wrongs"]
    true_values = [float(true_ratios[class_name]) for class_name in class_names]
    pred_values = [float(pred_ratios[class_name]) for class_name in class_names]
    x = np.arange(len(class_names))
    width = 0.38
    fig_width = max(7.0, min(16.0, 0.8 * len(class_names)))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    true_bars = ax.bar(x - (width / 2.0), true_values, width, label="True class", color="#4C78A8")
    pred_bars = ax.bar(x + (width / 2.0), pred_values, width, label="Predicted class", color="#F58518")
    ax.set_title(f"Class ratios among wrong predictions ({split})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Ratio among wrong predictions")
    max_value = max(true_values + pred_values, default=0.0)
    ax.set_ylim(0.0, max(1.0, max_value * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False)

    for bars, values in ((true_bars, true_values), (pred_bars, pred_values)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_histogram(values: pd.Series, bins: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(values.to_numpy(dtype=np.float64), bins=int(bins), color="#4C78A8", edgecolor="black", linewidth=0.6)
    ax.set_title("Compound molecular weight distribution")
    ax.set_xlabel("Molecular weight")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _build_bgc_features(
    bgc_df: pd.DataFrame,
    model: DualEncoderCLIP,
    bgc_cache: dict[str, torch.Tensor],
    label_to_idx: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(bgc_df), batch_size):
            chunk = bgc_df.iloc[start : start + batch_size]
            bgc_features = torch.stack([bgc_cache[str(bgc_id)] for bgc_id in chunk["bgc_id"].tolist()]).to(device)
            z_bgc = model.encode_bgc(bgc_features)
            y = torch.tensor([label_to_idx[str(label)] for label in chunk["bgc_class"].tolist()], dtype=torch.long)
            features.append(z_bgc.cpu())
            labels.append(y)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _attach_split_column(df: pd.DataFrame, splits_path: str | Path | None) -> pd.DataFrame:
    out = df.copy()
    if "split" in out.columns:
        out["split"] = out["split"].astype(str).str.lower()
        return out
    if splits_path is None:
        raise ValueError("A split column or a split assignment TSV is required for compound downstream tasks.")
    split_df = pd.read_csv(splits_path, sep="\t")
    split_df["bgc_id"] = split_df["bgc_id"].astype(str)
    split_df["split"] = split_df["split"].astype(str).str.lower()
    out["bgc_id"] = out["bgc_id"].astype(str)
    out = out.merge(split_df[["bgc_id", "split"]], on="bgc_id", how="left")
    return out


def _infer_compound_id_column(df: pd.DataFrame) -> str:
    if "compound_id" in df.columns:
        return "compound_id"
    if "canonical_smiles" in df.columns:
        return "canonical_smiles"
    if "smiles" in df.columns:
        return "smiles"
    raise ValueError("Could not infer compound identifier column from the MIBiG pairs table.")


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required for compound_mw and origin_type downstream tasks. "
            "Please install rdkit, for example with `conda install -c conda-forge rdkit`."
        ) from exc
    return Chem


def _safe_canonical_smiles(smiles: str | float | None, chem_module: Any) -> str | None:
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = chem_module.MolFromSmiles(text)
    if mol is None:
        return None
    return str(chem_module.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _safe_inchikey(smiles: str | float | None, chem_module: Any) -> str | None:
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = chem_module.MolFromSmiles(text)
    if mol is None:
        return None
    return str(chem_module.MolToInchiKey(mol))


def _first_unique_rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    valid = df.dropna(subset=[key]).copy()
    valid[key] = valid[key].astype(str).str.strip()
    valid = valid[valid[key] != ""].copy()
    return valid.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)


def _prepare_compound_match_table(
    mibig_pairs_path: str | Path,
    npatlas_path: str | Path,
    splits_path: str | Path | None,
    output_path: Path,
    force_rebuild: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if output_path.exists() and not force_rebuild:
        matched_df = pd.read_csv(output_path, sep="\t")
        total_mibig_rows = int(pd.read_csv(mibig_pairs_path, sep="\t", usecols=["bgc_id"]).shape[0])
        stats = {
            "total_mibig_rows": total_mibig_rows,
            "matched_by_inchikey": int((matched_df["match_method"] == "inchikey").sum()) if "match_method" in matched_df else 0,
            "matched_by_smiles": int((matched_df["match_method"] == "smiles").sum()) if "match_method" in matched_df else 0,
            "total_matched_rows": int(len(matched_df)),
        }
        return matched_df, stats

    chem = _require_rdkit()
    mibig_df = pd.read_csv(mibig_pairs_path, sep="\t")
    mibig_df = _attach_split_column(mibig_df, splits_path)
    compound_id_col = _infer_compound_id_column(mibig_df)
    mibig_df["bgc_id"] = mibig_df["bgc_id"].astype(str)
    mibig_df["compound_id"] = mibig_df[compound_id_col].astype(str)
    if "smiles" not in mibig_df.columns:
        mibig_df["smiles"] = mibig_df["compound_id"]
    mibig_df["smiles"] = mibig_df["smiles"].astype(str)
    mibig_df = mibig_df.dropna(subset=["bgc_id", "compound_id", "smiles", "split"]).reset_index(drop=True)

    npatlas_df = pd.read_csv(npatlas_path, sep="\t")
    if "compound_smiles" not in npatlas_df.columns or "compound_inchikey" not in npatlas_df.columns:
        raise ValueError("NPAtlas table must include compound_smiles and compound_inchikey columns.")

    npatlas_df = npatlas_df.copy()
    npatlas_df["compound_inchikey"] = npatlas_df["compound_inchikey"].astype(str).str.strip()
    npatlas_df["canonical_smiles"] = [
        _safe_canonical_smiles(smiles, chem) for smiles in npatlas_df["compound_smiles"].tolist()
    ]

    npatlas_inchikey = _first_unique_rows(npatlas_df, "compound_inchikey").set_index("compound_inchikey")
    npatlas_smiles = _first_unique_rows(npatlas_df, "canonical_smiles").set_index("canonical_smiles")

    mibig_df["mibig_inchikey"] = [_safe_inchikey(smiles, chem) for smiles in mibig_df["smiles"].tolist()]
    mibig_df["mibig_canonical_smiles"] = [_safe_canonical_smiles(smiles, chem) for smiles in mibig_df["smiles"].tolist()]

    matched_records: list[dict[str, Any]] = []
    matched_by_inchikey = 0
    matched_by_smiles = 0

    for row in mibig_df.itertuples(index=False):
        npatlas_row = None
        match_method = None
        match_key = None

        mibig_inchikey = getattr(row, "mibig_inchikey")
        if mibig_inchikey and mibig_inchikey in npatlas_inchikey.index:
            npatlas_row = npatlas_inchikey.loc[mibig_inchikey]
            match_method = "inchikey"
            match_key = mibig_inchikey
            matched_by_inchikey += 1
        else:
            mibig_smiles = getattr(row, "mibig_canonical_smiles")
            if mibig_smiles and mibig_smiles in npatlas_smiles.index:
                npatlas_row = npatlas_smiles.loc[mibig_smiles]
                match_method = "smiles"
                match_key = mibig_smiles
                matched_by_smiles += 1

        if npatlas_row is None:
            continue

        matched_records.append(
            {
                "bgc_id": str(row.bgc_id),
                "compound_id": str(row.compound_id),
                "split": str(row.split).lower(),
                "compound_name": getattr(row, "compound_name", None),
                "smiles": str(row.smiles),
                "mibig_inchikey": mibig_inchikey,
                "mibig_canonical_smiles": getattr(row, "mibig_canonical_smiles"),
                "match_method": match_method,
                "match_key": match_key,
                "npatlas_compound_name": npatlas_row.get("compound_name"),
                "npatlas_compound_smiles": npatlas_row.get("compound_smiles"),
                "npatlas_compound_inchikey": npatlas_row.get("compound_inchikey"),
                "compound_molecular_weight": npatlas_row.get("compound_molecular_weight"),
                "origin_type": npatlas_row.get("origin_type"),
            }
        )

    matched_df = pd.DataFrame(matched_records)
    if not matched_df.empty:
        matched_df.to_csv(output_path, sep="\t", index=False)

    stats = {
        "total_mibig_rows": int(len(mibig_df)),
        "matched_by_inchikey": int(matched_by_inchikey),
        "matched_by_smiles": int(matched_by_smiles),
        "total_matched_rows": int(len(matched_df)),
    }
    return matched_df, stats


def _build_compound_embedding_map(
    df: pd.DataFrame,
    model: DualEncoderCLIP,
    compound_cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    unique_ids = sorted({str(compound_id) for compound_id in df["compound_id"].tolist()})
    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(unique_ids), batch_size):
            batch_ids = unique_ids[start : start + batch_size]
            compound_features = torch.stack([compound_cache[compound_id] for compound_id in batch_ids]).to(device)
            z_compound = model.encode_compound(compound_features).cpu()
            for compound_id, embedding in zip(batch_ids, z_compound, strict=True):
                embeddings[compound_id] = embedding
    return embeddings


def _frame_to_tensor_dataset(
    df: pd.DataFrame,
    embedding_map: dict[str, torch.Tensor],
    label_column: str,
    label_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack([embedding_map[str(compound_id)] for compound_id in df["compound_id"].tolist()])
    if label_dtype == torch.long:
        labels = torch.tensor(df[label_column].tolist(), dtype=label_dtype)
    else:
        labels = torch.tensor(df[label_column].tolist(), dtype=torch.float32)
    return features, labels


def _log_split_sizes(task_name: str, split_frames: dict[str, pd.DataFrame]) -> None:
    LOGGER.info(
        "%s dataset sizes: train=%d val=%d test=%d",
        task_name,
        len(split_frames["train"]),
        len(split_frames["val"]),
        len(split_frames["test"]),
    )


def _train_bgc_class_task(
    data_dir: str | Path,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    contrastive_model: DualEncoderCLIP,
    *,
    splits_path: str | Path | None,
    baseline_trials: int,
    class_names: list[str] | None,
    save_cm_png: bool,
    output_dir: Path,
) -> dict[str, Any]:
    bgc_df = build_bgc_class_table(data_dir, splits_path=splits_path)
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")

    split_frames = {
        split: bgc_df[bgc_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    _log_split_sizes("bgc_class", split_frames)

    label_vocab = sorted(split_frames["train"]["bgc_class"].unique().tolist())
    if not label_vocab:
        raise ValueError("Training split does not contain any BGC classes for downstream training.")
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
    output_class_names = class_names if class_names is not None else [str(label) for label in label_vocab]
    if len(output_class_names) != len(label_vocab):
        raise ValueError("class_names length must match the number of training classes.")
    for split in ("val", "test"):
        unknown = sorted(set(split_frames[split]["bgc_class"].tolist()) - set(label_vocab))
        if unknown:
            missing = ", ".join(unknown)
            raise ValueError(f"Split '{split}' contains labels absent from train: {missing}")

    x_train, y_train = _build_bgc_features(
        split_frames["train"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
    )
    x_val, y_val = _build_bgc_features(
        split_frames["val"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
    )
    x_test, y_test = _build_bgc_features(
        split_frames["test"],
        contrastive_model,
        bgc_cache,
        label_to_idx,
        device,
        int(cfg["downstream"]["feature_batch_size"]),
    )

    classifier = BGCClassifier(
        emb_dim=int(cfg["model"]["emb_dim"]),
        num_classes=len(label_vocab),
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        classifier.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)

    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc="Training bgc_class", leave=False):
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    baseline_seed = int(cfg.get("seed", 42))
    metrics: dict[str, Any] = {
        "val": _classification_report(
            classifier,
            val_loader,
            device,
            y_train,
            len(label_vocab),
            output_class_names,
            baseline_trials,
            baseline_seed,
        ),
        "test": _classification_report(
            classifier,
            test_loader,
            device,
            y_train,
            len(label_vocab),
            output_class_names,
            baseline_trials,
            baseline_seed + 1,
        ),
        "label_vocab": label_vocab,
        "class_names": output_class_names,
    }

    if save_cm_png:
        _save_confusion_matrix_png(metrics["val"], output_class_names, output_dir / "downstream_confusion_matrix_val.png")
        _save_confusion_matrix_png(metrics["test"], output_class_names, output_dir / "downstream_confusion_matrix_test.png")
        _save_wrong_ratio_png(metrics["val"], output_class_names, "val", output_dir / "downstream_wrong_ratios_val.png")
        _save_wrong_ratio_png(metrics["test"], output_class_names, "test", output_dir / "downstream_wrong_ratios_test.png")

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "metrics": metrics,
            "label_vocab": label_vocab,
        },
        output_dir / "downstream_classifier.pt",
    )
    save_json(metrics, output_dir / "downstream_metrics.json")
    return metrics


def _train_compound_mw_task(
    matched_df: pd.DataFrame,
    embedding_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    baseline_trials: int,
    mw_bins: int,
) -> dict[str, Any]:
    task_df = matched_df.dropna(subset=["compound_molecular_weight"]).copy()
    task_df["compound_molecular_weight"] = pd.to_numeric(task_df["compound_molecular_weight"], errors="coerce")
    task_df = task_df.dropna(subset=["compound_molecular_weight"]).reset_index(drop=True)
    if task_df.empty:
        raise ValueError("No matched compounds with non-null compound_molecular_weight were found.")

    _save_histogram(task_df["compound_molecular_weight"], bins=mw_bins, path=output_dir / "downstream_mw_hist.png")

    split_frames = {
        split: task_df[task_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    _log_split_sizes("compound_mw", split_frames)
    if split_frames["train"].empty:
        raise ValueError("Training split is empty for compound_mw.")
    if split_frames["val"].empty or split_frames["test"].empty:
        raise ValueError("Validation and test splits must be non-empty for compound_mw.")

    x_train, y_train = _frame_to_tensor_dataset(split_frames["train"], embedding_map, "compound_molecular_weight", torch.float32)
    x_val, y_val = _frame_to_tensor_dataset(split_frames["val"], embedding_map, "compound_molecular_weight", torch.float32)
    x_test, y_test = _frame_to_tensor_dataset(split_frames["test"], embedding_map, "compound_molecular_weight", torch.float32)

    regressor = EmbeddingRegressor(
        emb_dim=int(cfg["model"]["emb_dim"]),
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        regressor.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)

    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc="Training compound_mw", leave=False):
        regressor.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = regressor(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

    baseline_seed = int(cfg.get("seed", 42))
    metrics = {
        "target": "compound_molecular_weight",
        "histogram_bins": int(mw_bins),
        "match_counts": {
            "final_dataset_size": int(len(task_df)),
        },
        "val": _regression_report(
            regressor=regressor,
            loader=val_loader,
            device=device,
            y_train=y_train,
            baseline_trials=baseline_trials,
            baseline_seed=baseline_seed,
        ),
        "test": _regression_report(
            regressor=regressor,
            loader=test_loader,
            device=device,
            y_train=y_train,
            baseline_trials=baseline_trials,
            baseline_seed=baseline_seed + 1,
        ),
    }

    torch.save(
        {
            "regressor_state_dict": regressor.state_dict(),
            "metrics": metrics,
        },
        output_dir / "downstream_compound_mw_regressor.pt",
    )
    save_json(metrics, output_dir / "downstream_compound_mw_metrics.json")
    return metrics


def _classification_metrics_from_predictions(y_true: torch.Tensor, y_pred: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    per_class = per_class_prf(cm)
    overall = macro_micro_f1_from_cm(cm)
    fungus_idx = ORIGIN_LABEL_TO_IDX["Fungus"]
    return {
        "accuracy": float(overall["accuracy"]),
        "macro_f1": float(overall["macro_f1"]),
        "precision_positive": float(per_class["precision"][fungus_idx]),
        "recall_positive": float(per_class["recall"][fungus_idx]),
        "f1_positive": float(per_class["f1"][fungus_idx]),
    }


def _summarize_classification_trials(trial_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in ("accuracy", "macro_f1", "precision_positive", "recall_positive", "f1_positive"):
        values = np.asarray([metrics[metric_name] for metrics in trial_metrics], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean()) if values.size else 0.0
        summary[f"{metric_name}_std"] = float(values.std(ddof=0)) if values.size else 0.0
    return summary


def _origin_baselines(
    y_train: torch.Tensor,
    y_true: torch.Tensor,
    trials: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    train_cpu = y_train.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    true_cpu = y_true.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    if train_cpu.numel() == 0:
        raise ValueError("y_train must contain at least one label.")

    train_pos_rate = float((train_cpu == ORIGIN_LABEL_TO_IDX["Fungus"]).to(dtype=torch.float32).mean().item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    majority_class = int(torch.mode(train_cpu).values.item())
    majority_pred = torch.full_like(true_cpu, fill_value=majority_class)
    majority = _classification_metrics_from_predictions(true_cpu, majority_pred, ORIGIN_CLASS_NAMES)
    for metric_name, value in list(majority.items()):
        majority[f"{metric_name}_mean"] = float(value)
        majority[f"{metric_name}_std"] = 0.0

    uniform_trials: list[dict[str, float]] = []
    prior_trials: list[dict[str, float]] = []
    for _ in range(int(trials)):
        uniform_pred = torch.randint(0, 2, true_cpu.shape, generator=generator, dtype=torch.long)
        prior_draws = torch.rand(true_cpu.shape, generator=generator)
        prior_pred = (prior_draws < train_pos_rate).to(dtype=torch.long)
        uniform_trials.append(_classification_metrics_from_predictions(true_cpu, uniform_pred, ORIGIN_CLASS_NAMES))
        prior_trials.append(_classification_metrics_from_predictions(true_cpu, prior_pred, ORIGIN_CLASS_NAMES))

    return {
        "majority": majority,
        "uniform": _summarize_classification_trials(uniform_trials),
        "prior": {
            **_summarize_classification_trials(prior_trials),
            "positive_rate_train": train_pos_rate,
        },
    }


def _train_origin_type_task(
    matched_df: pd.DataFrame,
    embedding_map: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    baseline_trials: int,
    save_cm_png: bool,
) -> dict[str, Any]:
    task_df = matched_df[matched_df["origin_type"].isin(ORIGIN_LABEL_TO_IDX.keys())].copy()
    if task_df.empty:
        raise ValueError("No matched compounds with origin_type in {'Fungus', 'Bacterium'} were found.")
    task_df["origin_label"] = task_df["origin_type"].map(ORIGIN_LABEL_TO_IDX)

    split_frames = {
        split: task_df[task_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    _log_split_sizes("origin_type", split_frames)
    if split_frames["train"].empty:
        raise ValueError("Training split is empty for origin_type.")
    if split_frames["val"].empty or split_frames["test"].empty:
        raise ValueError("Validation and test splits must be non-empty for origin_type.")

    x_train, y_train = _frame_to_tensor_dataset(split_frames["train"], embedding_map, "origin_label", torch.long)
    x_val, y_val = _frame_to_tensor_dataset(split_frames["val"], embedding_map, "origin_label", torch.long)
    x_test, y_test = _frame_to_tensor_dataset(split_frames["test"], embedding_map, "origin_label", torch.long)

    classifier = BGCClassifier(
        emb_dim=int(cfg["model"]["emb_dim"]),
        num_classes=2,
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        classifier.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(cfg["downstream"]["batch_size"]), shuffle=False)

    for _ in tqdm(range(int(cfg["downstream"]["epochs"])), desc="Training origin_type", leave=False):
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    baseline_seed = int(cfg.get("seed", 42))
    metrics = {
        "target": "origin_type",
        "label_mapping": ORIGIN_LABEL_TO_IDX,
        "positive_label": "Fungus",
        "class_names": ORIGIN_CLASS_NAMES,
        "match_counts": {
            "final_dataset_size": int(len(task_df)),
        },
        "val": _classification_report(
            classifier=classifier,
            loader=val_loader,
            device=device,
            y_train=y_train,
            num_classes=2,
            class_names=ORIGIN_CLASS_NAMES,
            baseline_trials=baseline_trials,
            baseline_seed=baseline_seed,
        ),
        "test": _classification_report(
            classifier=classifier,
            loader=test_loader,
            device=device,
            y_train=y_train,
            num_classes=2,
            class_names=ORIGIN_CLASS_NAMES,
            baseline_trials=baseline_trials,
            baseline_seed=baseline_seed + 1,
        ),
    }
    metrics["val"]["random_baselines"] = _origin_baselines(y_train, y_val, baseline_trials, baseline_seed)
    metrics["test"]["random_baselines"] = _origin_baselines(y_train, y_test, baseline_trials, baseline_seed + 1)

    if save_cm_png:
        _save_confusion_matrix_png(metrics["val"], ORIGIN_CLASS_NAMES, output_dir / "downstream_origin_type_confusion_matrix_val.png")
        _save_confusion_matrix_png(metrics["test"], ORIGIN_CLASS_NAMES, output_dir / "downstream_origin_type_confusion_matrix_test.png")

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "metrics": metrics,
            "label_mapping": ORIGIN_LABEL_TO_IDX,
        },
        output_dir / "downstream_origin_type_classifier.pt",
    )
    save_json(metrics, output_dir / "downstream_origin_type_metrics.json")
    return metrics


def train_downstream(
    data_dir: str | Path,
    cache_dir: str | Path,
    contrastive_ckpt: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    splits_path: str | Path | None = None,
    baseline_trials: int = 100,
    class_names: list[str] | None = None,
    save_cm_png: bool = False,
    tasks: list[str] | tuple[str, ...] | None = None,
    npatlas_path: str | Path = "data/NPAtlas_download_2024_09.tsv",
    mibig_pairs_path: str | Path = "data/MIBIG/processed/mibig_pairs.tsv",
    mw_bins: int = 50,
    force_rebuild_match: bool = False,
) -> dict[str, Any]:
    """Train one or more downstream models on frozen CLIP embeddings."""
    selected_tasks = list(tasks) if tasks else list(DEFAULT_TASKS)
    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    contrastive_model, _ = _load_contrastive_model(contrastive_ckpt, device)
    for param in contrastive_model.parameters():
        param.requires_grad = False

    results: dict[str, Any] = {"tasks": selected_tasks}
    compound_task_requested = any(task in COMPOUND_TASKS for task in selected_tasks)
    matched_df: pd.DataFrame | None = None
    compound_embeddings: dict[str, torch.Tensor] | None = None
    compound_match_stats: dict[str, int] | None = None

    if compound_task_requested:
        matched_path = outdir / "matched_compounds.tsv"
        matched_df, compound_match_stats = _prepare_compound_match_table(
            mibig_pairs_path=mibig_pairs_path,
            npatlas_path=npatlas_path,
            splits_path=splits_path,
            output_path=matched_path,
            force_rebuild=force_rebuild_match,
        )
        if matched_df.empty:
            raise ValueError("No MIBiG compounds could be matched to NPAtlas using InChIKey or canonical SMILES.")
        LOGGER.info(
            "Compound matching counts: total_mibig_rows=%d matched_by_inchikey=%d matched_by_smiles=%d total_matched_rows=%d",
            compound_match_stats["total_mibig_rows"],
            compound_match_stats["matched_by_inchikey"],
            compound_match_stats["matched_by_smiles"],
            compound_match_stats["total_matched_rows"],
        )

        interactions = build_interactions(data_dir, splits_path=splits_path)
        valid_pairs = interactions[["bgc_id", "compound_id", "split"]].drop_duplicates().copy()
        valid_pairs["bgc_id"] = valid_pairs["bgc_id"].astype(str)
        valid_pairs["compound_id"] = valid_pairs["compound_id"].astype(str)
        valid_pairs["split"] = valid_pairs["split"].astype(str).str.lower()
        matched_df = matched_df.merge(valid_pairs, on=["bgc_id", "compound_id", "split"], how="inner")
        if matched_df.empty:
            raise ValueError("Matched NPAtlas compounds do not overlap with cached compound features in the selected splits.")

        compound_cache = torch.load(Path(cache_dir) / "compound_features.pt", map_location="cpu")
        missing_ids = sorted(set(matched_df["compound_id"].astype(str).tolist()) - set(compound_cache))
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise KeyError(f"Missing compound features for {len(missing_ids)} matched compounds. Examples: {preview}")
        compound_embeddings = _build_compound_embedding_map(
            matched_df,
            contrastive_model,
            compound_cache,
            device,
            int(cfg["downstream"]["feature_batch_size"]),
        )
        matched_df.to_csv(matched_path, sep="\t", index=False)
        results["compound_matching"] = compound_match_stats
        results["matched_compounds_path"] = str(matched_path)

    for task in selected_tasks:
        if task == "bgc_class":
            results[task] = _train_bgc_class_task(
                data_dir=data_dir,
                cache_dir=cache_dir,
                cfg=cfg,
                device=device,
                contrastive_model=contrastive_model,
                splits_path=splits_path,
                baseline_trials=baseline_trials,
                class_names=class_names,
                save_cm_png=save_cm_png,
                output_dir=outdir,
            )
        elif task == "compound_mw":
            if matched_df is None or compound_embeddings is None or compound_match_stats is None:
                raise RuntimeError("Compound match state is missing for compound_mw.")
            results[task] = _train_compound_mw_task(
                matched_df=matched_df,
                embedding_map=compound_embeddings,
                cfg=cfg,
                device=device,
                output_dir=outdir,
                baseline_trials=baseline_trials,
                mw_bins=mw_bins,
            )
            results[task]["match_counts"].update(compound_match_stats)
        elif task == "origin_type":
            if matched_df is None or compound_embeddings is None or compound_match_stats is None:
                raise RuntimeError("Compound match state is missing for origin_type.")
            results[task] = _train_origin_type_task(
                matched_df=matched_df,
                embedding_map=compound_embeddings,
                cfg=cfg,
                device=device,
                output_dir=outdir,
                baseline_trials=baseline_trials,
                save_cm_png=save_cm_png,
            )
            results[task]["match_counts"].update(compound_match_stats)
        else:
            raise ValueError(f"Unsupported downstream task: {task}")

    return results
