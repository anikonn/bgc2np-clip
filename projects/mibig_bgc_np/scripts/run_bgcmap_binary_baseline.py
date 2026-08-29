from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from projects.mibig_bgc_np.utils.seed import set_seed


def save_json(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return {}
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parsed = _parse_scalar(value)
        parent[key] = parsed
        if isinstance(parsed, dict):
            stack.append((indent, parsed))
    return root


def apply_simple_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            continue
        dotted, raw_value = override.split("=", 1)
        target = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = _parse_scalar(raw_value)
    return cfg


class PairBinaryClassifier(nn.Module):
    """MLP over frozen BGC/product features for explicit match prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mid_dim = max(1, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenDualEncoder(nn.Module):
    def __init__(
        self,
        bgc_input_dim: int,
        compound_input_dim: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        init_temperature: float,
        max_logit_scale: float,
    ) -> None:
        super().__init__()
        self.bgc_proj = ProjectionHead(bgc_input_dim, emb_dim, hidden_dim, dropout)
        self.compound_proj = ProjectionHead(compound_input_dim, emb_dim, hidden_dim, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32))
        self.max_logit_scale = max_logit_scale

    def encode_bgc(self, bgc_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.bgc_proj(bgc_features), dim=-1)

    def encode_compound(self, compound_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.compound_proj(compound_features), dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_logit_scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-feature binary classifier baseline for BGC-MAP candidate pairs."
    )
    parser.add_argument("--cache_dir", type=str, default="cache/mibig_map")
    parser.add_argument("--bgcmap_splits_path", type=str, default="data/MIBIG/splits/MAP_metadata_fold.csv")
    parser.add_argument("--fallback_results_dir", type=str, default="results/bgcmap_retrieval")
    parser.add_argument(
        "--feature_source",
        choices=["raw", "clip"],
        default="raw",
        help="Use raw cached BGC/product features or frozen embeddings from trained BGC2NP-CLIP members.",
    )
    parser.add_argument(
        "--clip_members_dir",
        type=str,
        default="results/bgcmap_retrieval",
        help="Directory containing val_fold_*/contrastive_model_best.pt checkpoints for --feature_source clip.",
    )
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fold", type=int, default=10)
    parser.add_argument("--val_folds", type=int, nargs="*", default=None)
    parser.add_argument("--outdir", type=str, default="results/bgcmap_retrieval/baselines/classification")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--reuse_existing_members", action="store_true")
    return parser.parse_args()


def _load_map_rows(path: str | Path, fallback_results_dir: str | Path) -> list[dict[str, Any]]:
    split_path = Path(path)
    if split_path.exists():
        with split_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"BGC_number", "product", "biosyn_class", "is_product", "fold"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{split_path} is missing columns: {sorted(missing)}")
            rows = [
                {
                    "bgc_id": row["BGC_number"],
                    "compound_id": row["product"],
                    "bgc_classes": row["biosyn_class"],
                    "is_product": int(float(row["is_product"]) > 0.0),
                    "fold": int(float(row["fold"])),
                }
                for row in reader
                if row.get("BGC_number") and row.get("product")
            ]
            return rows

    root = Path(fallback_results_dir)
    paths = sorted(root.glob("val_fold_*/validation_pair_scores.tsv"))
    test_path = root / "ensemble_test_pair_scores.tsv"
    if test_path.exists():
        paths.append(test_path)
    if not paths:
        raise FileNotFoundError(
            f"Could not find {split_path} or fallback validation/test pair score TSVs under {root}."
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for pair_path in paths:
        with pair_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"bgc_id", "compound_id", "bgc_classes", "is_product", "fold"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{pair_path} is missing columns: {sorted(missing)}")
            for row in reader:
                key = (row["bgc_id"], row["compound_id"], int(float(row["fold"])))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "bgc_id": row["bgc_id"],
                        "compound_id": row["compound_id"],
                        "bgc_classes": row["bgc_classes"],
                        "is_product": int(float(row["is_product"]) > 0.0),
                        "fold": int(float(row["fold"])),
                    }
                )
    return rows


def _parse_classes(text: object) -> list[str]:
    raw = "" if text is None else str(text)
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        candidates = [str(item) for item in parsed]
    else:
        candidates = raw.replace(";", ",").split(",")
    labels: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        label = str(item).strip().strip("'\"")
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _split_rows(rows: list[dict[str, Any]], val_fold: int, test_fold: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [row for row in rows if int(row["fold"]) not in {int(val_fold), int(test_fold)}],
        "val": [row for row in rows if int(row["fold"]) == int(val_fold)],
        "test": [row for row in rows if int(row["fold"]) == int(test_fold)],
    }


def _pair_features(
    rows: list[dict[str, Any]],
    bgc_cache: dict[str, torch.Tensor],
    compound_cache: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    missing_bgc = sorted({str(row["bgc_id"]) for row in rows}.difference(bgc_cache))
    missing_compound = sorted({str(row["compound_id"]) for row in rows}.difference(compound_cache))
    if missing_bgc:
        raise KeyError(f"Missing BGC features for {len(missing_bgc)} ids. Examples: {missing_bgc[:5]}")
    if missing_compound:
        raise KeyError(f"Missing compound features for {len(missing_compound)} ids. Examples: {missing_compound[:5]}")

    x = torch.stack(
        [
            torch.cat(
                [
                    bgc_cache[str(row["bgc_id"])].float(),
                    compound_cache[str(row["compound_id"])].float(),
                ]
            )
            for row in rows
        ]
    )
    y = torch.tensor([float(row["is_product"]) for row in rows], dtype=torch.float32)
    return x, y


def _make_clip_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> FrozenDualEncoder:
    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    cfg = ckpt["config"]
    model = FrozenDualEncoder(
        bgc_input_dim=int(ckpt["bgc_input_dim"]),
        compound_input_dim=int(ckpt["compound_input_dim"]),
        emb_dim=int(cfg["model"]["emb_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        init_temperature=float(cfg["model"]["init_temperature"]),
        max_logit_scale=float(cfg["model"]["max_logit_scale"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _encode_clip_ids(
    model: FrozenDualEncoder,
    ids: list[str],
    cache: dict[str, torch.Tensor],
    *,
    modality: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    missing = sorted(set(ids).difference(cache))
    if missing:
        raise KeyError(f"Missing {modality} features for {len(missing)} ids. Examples: {missing[:5]}")

    encoded: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            features = torch.stack([cache[item_id].float() for item_id in batch_ids]).to(device)
            if modality == "bgc":
                embeddings = model.encode_bgc(features).cpu()
            elif modality == "compound":
                embeddings = model.encode_compound(features).cpu()
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            for item_id, embedding in zip(batch_ids, embeddings, strict=True):
                encoded[item_id] = embedding
    return encoded


def _member_feature_caches(
    *,
    feature_source: str,
    rows: list[dict[str, Any]],
    val_fold: int,
    raw_bgc_cache: dict[str, torch.Tensor],
    raw_compound_cache: dict[str, torch.Tensor],
    clip_members_dir: str | Path,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    if feature_source == "raw":
        return raw_bgc_cache, raw_compound_cache, {"feature_source": "raw_cached_features"}

    checkpoint_path = Path(clip_members_dir) / f"val_fold_{int(val_fold)}" / "contrastive_model_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing CLIP member checkpoint: {checkpoint_path}")
    model = _make_clip_model_from_checkpoint(checkpoint_path, device=device)
    bgc_ids = sorted({str(row["bgc_id"]) for row in rows})
    compound_ids = sorted({str(row["compound_id"]) for row in rows})
    bgc_cache = _encode_clip_ids(
        model,
        bgc_ids,
        raw_bgc_cache,
        modality="bgc",
        device=device,
        batch_size=batch_size,
    )
    compound_cache = _encode_clip_ids(
        model,
        compound_ids,
        raw_compound_cache,
        modality="compound",
        device=device,
        batch_size=batch_size,
    )
    return (
        bgc_cache,
        compound_cache,
        {
            "feature_source": "frozen_bgc2np_clip_embeddings",
            "clip_checkpoint": str(checkpoint_path),
            "embedding_dim": int(next(iter(bgc_cache.values())).numel()) if bgc_cache else 0,
        },
    )


def _binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[list[float], list[float], list[float], float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0 or y_true.size == 0:
        return [0.0, 1.0], [0.0, 1.0], [math.inf, -math.inf], 0.0

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    sorted_score = y_score[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    distinct = np.where(np.diff(sorted_score))[0]
    threshold_idxs = np.r_[distinct, len(sorted_score) - 1]
    tps = np.r_[0, tp[threshold_idxs]]
    fps = np.r_[0, fp[threshold_idxs]]
    thresholds = np.r_[math.inf, sorted_score[threshold_idxs]]
    tpr = tps / float(pos)
    fpr = fps / float(neg)
    trapezoid = getattr(np, "trapezoid", None)
    auc = float(trapezoid(tpr, fpr) if trapezoid is not None else _trapz(tpr, fpr))
    return (
        [float(x) for x in fpr.tolist()],
        [float(x) for x in tpr.tolist()],
        [float(x) for x in thresholds.tolist()],
        float(max(0.0, min(1.0, auc))),
    )


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _best_threshold(fpr: list[float], tpr: list[float], thresholds: list[float]) -> float:
    scores = np.asarray(tpr, dtype=np.float64) - np.asarray(fpr, dtype=np.float64)
    idx = int(np.argmax(scores))
    threshold = float(thresholds[idx])
    if math.isinf(threshold):
        finite = [float(value) for value in thresholds if not math.isinf(float(value))]
        return finite[0] if finite else 0.0
    return threshold


def _confusion_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = np.asarray(y_score, dtype=np.float64) >= float(threshold)
    true = np.asarray(y_true, dtype=np.int64).astype(bool)
    tn = int((~true & ~pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tp = int((true & pred).sum())
    total = tp + tn + fp + fn
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": {
            "labels": ["Negative", "Positive"],
            "raw": {
                "Negative": {"Negative": tn, "Positive": fp},
                "Positive": {"Negative": fn, "Positive": tp},
            },
        },
    }


def _evaluate_pair_scores(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    thresholds_by_class: dict[str, float] | None = None,
) -> dict[str, Any]:
    class_to_indices: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        for label in _parse_classes(row["bgc_classes"]):
            class_to_indices.setdefault(label, []).append(idx)

    y_all = np.asarray([int(row["is_product"]) for row in rows], dtype=np.int64)
    bgc_all = [str(row["bgc_id"]) for row in rows]
    class_metrics: dict[str, Any] = {}
    micro_true_parts: list[np.ndarray] = []
    micro_score_parts: list[np.ndarray] = []
    for class_name in sorted(class_to_indices):
        indices = np.asarray(class_to_indices[class_name], dtype=np.int64)
        y_true = y_all[indices]
        y_score = scores[indices]
        positives = int(y_true.sum())
        negatives = int(y_true.size - positives)
        if positives == 0 or negatives == 0:
            continue
        fpr, tpr, thresholds, auc = _binary_roc_curve(y_true, y_score)
        threshold = (
            float(thresholds_by_class[class_name])
            if thresholds_by_class is not None and class_name in thresholds_by_class
            else _best_threshold(fpr, tpr, thresholds)
        )
        metrics = _confusion_metrics(y_true, y_score, threshold)
        class_metrics[class_name] = {
            "auroc": auc,
            "threshold": threshold,
            "threshold_source": "external" if thresholds_by_class and class_name in thresholds_by_class else "evaluation",
            "n_bgcs": int(len({bgc_all[i] for i in indices.tolist()})),
            "n_rows": int(y_true.size),
            "n_positive": positives,
            "n_negative": negatives,
            "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
            **metrics,
        }
        micro_true_parts.append(y_true)
        micro_score_parts.append(y_score)

    if micro_true_parts:
        micro_true = np.concatenate(micro_true_parts)
        micro_score = np.concatenate(micro_score_parts)
        micro_fpr, micro_tpr, micro_thresholds, micro_auc = _binary_roc_curve(micro_true, micro_score)
    else:
        micro_fpr, micro_tpr, micro_thresholds, micro_auc = [0.0, 1.0], [0.0, 1.0], [math.inf, -math.inf], 0.0
    return {
        "classes": class_metrics,
        "macro_auc": float(np.mean([item["auroc"] for item in class_metrics.values()])) if class_metrics else 0.0,
        "micro_auc": float(micro_auc),
        "micro_roc_curve": {"fpr": micro_fpr, "tpr": micro_tpr, "thresholds": micro_thresholds},
        "n_rows": int(len(rows)),
        "n_positive": int(y_all.sum()),
        "n_negative": int(len(y_all) - int(y_all.sum())),
    }


def _predict(model: PairBinaryClassifier, x: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            logits = model(x[start : start + batch_size].to(device))
            parts.append(logits.cpu())
    return torch.cat(parts, dim=0)


def _train_member(
    rows_by_split: dict[str, list[dict[str, Any]]],
    bgc_cache: dict[str, torch.Tensor],
    compound_cache: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    outdir: Path,
    patience: int,
) -> tuple[PairBinaryClassifier, dict[str, Any]]:
    x_train, y_train = _pair_features(rows_by_split["train"], bgc_cache, compound_cache)
    x_val, y_val = _pair_features(rows_by_split["val"], bgc_cache, compound_cache)
    input_dim = int(x_train.size(1))
    model = PairBinaryClassifier(
        input_dim=input_dim,
        hidden_dim=int(cfg["downstream"]["hidden_dim"]),
        dropout=float(cfg["downstream"]["dropout"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["downstream"]["lr"]),
        weight_decay=float(cfg["downstream"]["weight_decay"]),
    )
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=True,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(cfg["downstream"]["epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch.to(device))
            loss = loss_fn(logits, y_batch.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_logits = _predict(model, x_val, device, int(cfg["downstream"]["feature_batch_size"]))
            val_loss = float(loss_fn(val_logits.to(device), y_val.to(device)).detach().cpu().item())
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": int(cfg["downstream"]["hidden_dim"]),
            "dropout": float(cfg["downstream"]["dropout"]),
            "pos_weight": float(pos_weight.cpu().item()),
            "history": history,
        },
        outdir / "frozen_pair_binary_classifier.pt",
    )
    return model, {
        "input_dim": input_dim,
        "pos_weight": float(pos_weight.cpu().item()),
        "best_val_loss": float(best_val),
        "epochs_ran": int(len(history)),
        "history": history,
        "counts": {
            split: {
                "n_rows": int(len(split_rows)),
                "n_positive": int(sum(int(row["is_product"]) for row in split_rows)),
                "n_negative": int(len(split_rows) - sum(int(row["is_product"]) for row in split_rows)),
            }
            for split, split_rows in rows_by_split.items()
        },
    }


def _write_scores(path: Path, rows: list[dict[str, Any]], scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bgc_id", "compound_id", "bgc_classes", "is_product", "fold", "score"],
            delimiter="\t",
        )
        writer.writeheader()
        for row, score in zip(rows, scores.tolist(), strict=True):
            writer.writerow({**row, "score": float(score)})


def _write_metrics_table(path: Path, report: dict[str, Any]) -> None:
    preferred = ["NRPS", "other", "PKS", "ribosomal", "saccharide", "terpene"]
    classes = report.get("classes", {})
    ordered = [name for name in preferred if name in classes] + sorted(name for name in classes if name not in preferred)
    baseline = str(report.get("baseline", "frozen_pair_binary_classifier"))
    value_col = (
        "CLIP embedding binary classifier"
        if baseline == "clip_embedding_pair_binary_classifier"
        else "Frozen binary classifier"
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "BGC count", "model", value_col])
        writer.writeheader()
        for class_name in ordered:
            metrics = classes[class_name]
            for idx, metric in enumerate(["auroc", "accuracy", "precision", "recall", "f1"]):
                writer.writerow(
                    {
                        "class": class_name if idx == 0 else "",
                        "BGC count": int(metrics.get("n_bgcs", 0)) if idx == 0 else "",
                        "model": "AUROC" if metric == "auroc" else metric.capitalize(),
                        value_col: float(metrics.get(metric, 0.0)),
                    }
                )


def main() -> None:
    args = parse_args()
    logger = setup_logger("bgcmap_binary_baseline")
    cfg = apply_simple_overrides(load_simple_yaml(args.config), args.override)
    cfg["seed"] = int(args.seed)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_outdir = "results/bgcmap_retrieval/baselines/classification"
    if args.feature_source == "clip" and args.outdir == default_outdir:
        outdir = Path("results/bgcmap_retrieval/baselines/clip_embedding_classification")
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_name = (
        "clip_embedding_pair_binary_classifier"
        if args.feature_source == "clip"
        else "frozen_pair_binary_classifier"
    )
    file_prefix = (
        "clip_embedding_pair_binary_baseline"
        if args.feature_source == "clip"
        else "frozen_pair_binary_baseline"
    )

    rows = _load_map_rows(args.bgcmap_splits_path, args.fallback_results_dir)
    val_folds = args.val_folds if args.val_folds is not None and len(args.val_folds) else sorted(
        {int(row["fold"]) for row in rows if int(row["fold"]) != int(args.test_fold)}
    )
    raw_bgc_cache = torch.load(Path(args.cache_dir) / "bgc_features.pt", map_location="cpu")
    raw_compound_cache = torch.load(Path(args.cache_dir) / "compound_features.pt", map_location="cpu")
    test_rows = [row for row in rows if int(row["fold"]) == int(args.test_fold)]

    test_score_parts: list[np.ndarray] = []
    member_summaries: list[dict[str, Any]] = []
    threshold_values: dict[str, list[float]] = {}
    for val_fold in val_folds:
        member_outdir = outdir / f"val_fold_{int(val_fold)}"
        set_seed(int(args.seed) + int(val_fold))
        rows_by_split = _split_rows(rows, int(val_fold), int(args.test_fold))
        member_rows = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
        member_bgc_cache, member_compound_cache, feature_meta = _member_feature_caches(
            feature_source=str(args.feature_source),
            rows=member_rows,
            val_fold=int(val_fold),
            raw_bgc_cache=raw_bgc_cache,
            raw_compound_cache=raw_compound_cache,
            clip_members_dir=args.clip_members_dir,
            device=device,
            batch_size=int(cfg["downstream"]["feature_batch_size"]),
        )
        ckpt_path = member_outdir / f"{file_prefix}_classifier.pt"
        if bool(args.reuse_existing_members) and ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model = PairBinaryClassifier(
                input_dim=int(checkpoint["input_dim"]),
                hidden_dim=int(checkpoint.get("hidden_dim", cfg["downstream"]["hidden_dim"])),
                dropout=float(checkpoint.get("dropout", cfg["downstream"]["dropout"])),
            ).to(device)
            model.load_state_dict(checkpoint["classifier_state_dict"])
            metrics = {
                "input_dim": int(checkpoint["input_dim"]),
                "pos_weight": float(checkpoint.get("pos_weight", 1.0)),
                "history": checkpoint.get("history", []),
                "counts": {
                    split: {
                        "n_rows": int(len(split_rows)),
                        "n_positive": int(sum(int(row["is_product"]) for row in split_rows)),
                        "n_negative": int(len(split_rows) - sum(int(row["is_product"]) for row in split_rows)),
                    }
                    for split, split_rows in rows_by_split.items()
                },
                **checkpoint.get("feature_meta", {}),
            }
        else:
            logger.info("Training %s member with val fold %s", baseline_name, val_fold)
            model, metrics = _train_member(
                rows_by_split,
                member_bgc_cache,
                member_compound_cache,
                cfg,
                device,
                member_outdir,
                patience=int(args.patience),
            )
            original_ckpt = member_outdir / "frozen_pair_binary_classifier.pt"
            desired_ckpt = member_outdir / f"{file_prefix}_classifier.pt"
            if original_ckpt.exists() and original_ckpt != desired_ckpt:
                checkpoint = torch.load(original_ckpt, map_location="cpu")
                checkpoint["feature_meta"] = feature_meta
                torch.save(checkpoint, desired_ckpt)
                original_ckpt.unlink()
            metrics.update(feature_meta)

        x_val, _ = _pair_features(rows_by_split["val"], member_bgc_cache, member_compound_cache)
        val_scores = torch.sigmoid(_predict(model, x_val, device, int(cfg["downstream"]["feature_batch_size"]))).numpy()
        val_report = _evaluate_pair_scores(rows_by_split["val"], val_scores)
        member_thresholds: dict[str, float] = {}
        for class_name, class_metrics in val_report.get("classes", {}).items():
            threshold = float(class_metrics["threshold"])
            threshold_values.setdefault(class_name, []).append(threshold)
            member_thresholds[class_name] = threshold
        _write_scores(member_outdir / f"{file_prefix}_validation_pair_scores.tsv", rows_by_split["val"], val_scores)
        save_json(val_report, member_outdir / f"{file_prefix}_validation_report.json")
        metrics["validation_thresholds"] = member_thresholds
        metrics["validation_report_path"] = str(member_outdir / f"{file_prefix}_validation_report.json")
        save_json(metrics, member_outdir / f"{file_prefix}_metrics.json")
        member_summaries.append({"val_fold": int(val_fold), "output_dir": str(member_outdir), **metrics})
        x_test_member, _ = _pair_features(test_rows, member_bgc_cache, member_compound_cache)
        test_scores = torch.sigmoid(
            _predict(model, x_test_member, device, int(cfg["downstream"]["feature_batch_size"]))
        ).numpy()
        test_score_parts.append(test_scores)

    thresholds_by_class = {name: float(np.mean(values)) for name, values in sorted(threshold_values.items())}
    save_json(
        {
            "threshold_protocol": "per-member Youden threshold on held-out validation fold, averaged by BGC class",
            "thresholds_by_class": thresholds_by_class,
        },
        outdir / f"{file_prefix}_thresholds.json",
    )

    ensemble_scores = np.mean(np.stack(test_score_parts, axis=0), axis=0)
    scores_path = outdir / f"{file_prefix}_test_pair_scores.tsv"
    _write_scores(scores_path, test_rows, ensemble_scores)

    report = _evaluate_pair_scores(test_rows, ensemble_scores, thresholds_by_class=thresholds_by_class)
    report["threshold_protocol"] = "validation_derived_mean_by_class"
    report["scored_pairs_path"] = str(scores_path)
    report["members"] = member_summaries
    report["n_models"] = int(len(test_score_parts))
    report["baseline"] = baseline_name
    report["feature_source"] = str(args.feature_source)
    report_path = outdir / f"{file_prefix}_test_retrieval.json"
    save_json(report, report_path)
    table_path = outdir / f"{file_prefix}_metrics_table.csv"
    _write_metrics_table(table_path, report)

    manifest = {
        "status": "ok",
        "baseline": baseline_name,
        "feature_source": str(args.feature_source),
        "run_root": str(Path(args.fallback_results_dir)),
        "output_dir": str(outdir),
        "test_report": str(report_path),
        "test_pair_scores": str(scores_path),
        "metrics_table": str(table_path),
        "n_models": int(len(test_score_parts)),
        "n_test_rows": int(len(test_rows)),
        "n_test_positive": int(sum(int(row["is_product"]) for row in test_rows)),
        "n_test_negative": int(len(test_rows) - sum(int(row["is_product"]) for row in test_rows)),
    }
    save_json(manifest, outdir / "classification_baseline_artifacts.json")
    logger.info("Saved %s to %s", baseline_name, outdir)


if __name__ == "__main__":
    main()
