from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from clip_core.logging import save_json
from kiba_clip.data.datasets import build_interactions
from kiba_clip.eval.regression_metrics import rmse, spearman
from kiba_clip.models.clip_dual import DualEncoderCLIP
from kiba_clip.models.regression import RegressionHead


def _load_contrastive_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DualEncoderCLIP, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = DualEncoderCLIP(
        protein_input_dim=ckpt["protein_input_dim"],
        ligand_input_dim=ckpt["ligand_input_dim"],
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def _build_interaction_features(
    df: pd.DataFrame,
    model: DualEncoderCLIP,
    protein_cache: dict[str, torch.Tensor],
    ligand_cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    feats: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            chunk = df.iloc[i : i + batch_size]
            p = torch.stack([protein_cache[x] for x in chunk["Target_ID"].tolist()]).to(device)
            l = torch.stack([ligand_cache[x] for x in chunk["Drug_ID"].tolist()]).to(device)
            y = torch.tensor(chunk["Y"].to_numpy(dtype=np.float32), device=device)

            zp = model.encode_protein(p)
            zl = model.encode_ligand(l)
            x = torch.cat([zp, zl, zp * zl, torch.abs(zp - zl)], dim=-1)
            feats.append(x.cpu())
            ys.append(y.cpu())

    return torch.cat(feats, dim=0), torch.cat(ys, dim=0)


def _evaluate_regression(
    reg: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    reg.eval()
    preds: list[torch.Tensor] = []
    trues: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            pred = reg(x.to(device)).cpu()
            preds.append(pred)
            trues.append(y)

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()
    return {
        "rmse": rmse(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


def train_downstream(
    data_dir: str | Path,
    cache_dir: str | Path,
    contrastive_ckpt: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Train downstream regressor on frozen projected embeddings."""
    interactions = build_interactions(data_dir)
    prot_cache = torch.load(Path(cache_dir) / "protein_embeddings.pt", map_location="cpu")
    lig_cache = torch.load(Path(cache_dir) / "ligand_fingerprints.pt", map_location="cpu")

    contrastive_model, _ = _load_contrastive_model(contrastive_ckpt, device)
    for p in contrastive_model.parameters():
        p.requires_grad = False

    split_frames = {
        split: interactions[interactions["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }

    x_train, y_train = _build_interaction_features(
        split_frames["train"], contrastive_model, prot_cache, lig_cache, device, cfg["downstream"]["feature_batch_size"]
    )
    x_val, y_val = _build_interaction_features(
        split_frames["val"], contrastive_model, prot_cache, lig_cache, device, cfg["downstream"]["feature_batch_size"]
    )
    x_test, y_test = _build_interaction_features(
        split_frames["test"], contrastive_model, prot_cache, lig_cache, device, cfg["downstream"]["feature_batch_size"]
    )

    reg = RegressionHead(
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["downstream"]["hidden_dim"],
        dropout=cfg["downstream"]["dropout"],
    ).to(device)

    optimizer = AdamW(
        reg.parameters(),
        lr=cfg["downstream"]["lr"],
        weight_decay=cfg["downstream"]["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg["downstream"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=cfg["downstream"]["batch_size"], shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=cfg["downstream"]["batch_size"], shuffle=False)

    for _ in range(int(cfg["downstream"]["epochs"])):
        reg.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = reg(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    metrics = {
        "val": _evaluate_regression(reg, val_loader, device),
        "test": _evaluate_regression(reg, test_loader, device),
    }

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"regressor_state_dict": reg.state_dict(), "metrics": metrics}, outdir / "downstream_regressor.pt")
    save_json(metrics, outdir / "downstream_metrics.json")
    return metrics
