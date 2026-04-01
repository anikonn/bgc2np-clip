from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from clip_core.logging import save_json
from projects.mibig_bgc_np.data.datasets import build_bgc_class_table
from projects.mibig_bgc_np.models.classification import BGCClassifier
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP


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


def _macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1_scores: list[float] = []
    for class_idx in range(num_classes):
        true_pos = int(((y_true == class_idx) & (y_pred == class_idx)).sum().item())
        false_pos = int(((y_true != class_idx) & (y_pred == class_idx)).sum().item())
        false_neg = int(((y_true == class_idx) & (y_pred != class_idx)).sum().item())
        denom = (2 * true_pos) + false_pos + false_neg
        f1_scores.append(0.0 if denom == 0 else (2.0 * true_pos) / denom)
    return float(sum(f1_scores) / max(len(f1_scores), 1))


def _evaluate_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    classifier.eval()
    if len(loader.dataset) == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "macro_f1": float("nan")}

    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    loss_fn = nn.CrossEntropyLoss()
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
    y_pred = torch.cat(logits_all).argmax(dim=-1)
    accuracy = (y_true == y_pred).float().mean().item()
    return {
        "loss": running_loss / max(count, 1),
        "accuracy": float(accuracy),
        "macro_f1": _macro_f1_score(y_true, y_pred, num_classes),
    }


def _build_bgc_features(
    bgc_df,
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


def train_downstream(
    data_dir: str | Path,
    cache_dir: str | Path,
    contrastive_ckpt: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    splits_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train downstream classifier on frozen projected BGC embeddings."""
    bgc_df = build_bgc_class_table(data_dir, splits_path=splits_path)
    bgc_cache = torch.load(Path(cache_dir) / "bgc_features.pt", map_location="cpu")

    contrastive_model, _ = _load_contrastive_model(contrastive_ckpt, device)
    for param in contrastive_model.parameters():
        param.requires_grad = False

    split_frames = {
        split: bgc_df[bgc_df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    label_vocab = sorted(split_frames["train"]["bgc_class"].unique().tolist())
    if not label_vocab:
        raise ValueError("Training split does not contain any BGC classes for downstream training.")
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
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
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=int(cfg["downstream"]["batch_size"]),
        shuffle=False,
    )

    for _ in range(int(cfg["downstream"]["epochs"])):
        classifier.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    metrics = {
        "val": _evaluate_classifier(classifier, val_loader, device, len(label_vocab)),
        "test": _evaluate_classifier(classifier, test_loader, device, len(label_vocab)),
        "label_vocab": label_vocab,
    }

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "metrics": metrics,
            "label_vocab": label_vocab,
        },
        outdir / "downstream_classifier.pt",
    )
    save_json(metrics, outdir / "downstream_metrics.json")
    return metrics
