from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_core.logging import save_json
from clip_core.retrieval import evaluate_global_retrieval_multi
from kiba_clip.data.datasets import CachedInteractionDataset, build_interactions, collate_interactions
from kiba_clip.models.registry import MODEL_REGISTRY


def _get_cached_paths(cache_dir: str | Path) -> tuple[Path, Path]:
    c = Path(cache_dir)
    return c / "protein_embeddings.pt", c / "ligand_fingerprints.pt"


def _infer_input_dims(protein_cache_path: Path, ligand_cache_path: Path) -> tuple[int, int]:
    prot = torch.load(protein_cache_path, map_location="cpu")
    lig = torch.load(ligand_cache_path, map_location="cpu")
    p_dim = next(iter(prot.values())).numel()
    l_dim = next(iter(lig.values())).numel()
    return p_dim, l_dim


def _build_loader(
    interactions: pd.DataFrame,
    protein_cache_path: Path,
    ligand_cache_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    ds = CachedInteractionDataset(
        interactions=interactions,
        protein_cache_path=protein_cache_path,
        ligand_cache_path=ligand_cache_path,
        split=split,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_interactions,
        pin_memory=True,
    )


def build_unique_embeddings(
    model: torch.nn.Module,
    interactions: pd.DataFrame,
    split: str,
    protein_cache: dict[str, torch.Tensor],
    ligand_cache: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[dict[str, int], dict[str, int], torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Build projected normalized embeddings for all unique entities in a split."""
    df = interactions[interactions["split"].str.lower() == split.lower()].copy()

    protein_ids = sorted(df["Target_ID"].unique().tolist())
    ligand_ids = sorted(df["Drug_ID"].unique().tolist())
    p_index = {pid: i for i, pid in enumerate(protein_ids)}
    l_index = {lid: i for i, lid in enumerate(ligand_ids)}

    model.eval()

    p_emb_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(protein_ids), batch_size):
            ids = protein_ids[i : i + batch_size]
            feats = torch.stack([protein_cache[x] for x in ids]).to(device)
            p_emb_chunks.append(model.encode_protein(feats).cpu())
    p_embs = torch.cat(p_emb_chunks, dim=0)

    l_emb_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(ligand_ids), batch_size):
            ids = ligand_ids[i : i + batch_size]
            feats = torch.stack([ligand_cache[x] for x in ids]).to(device)
            l_emb_chunks.append(model.encode_ligand(feats).cpu())
    l_embs = torch.cat(l_emb_chunks, dim=0)

    pairs = [(p_index[r.Target_ID], l_index[r.Drug_ID]) for r in df.itertuples(index=False)]
    return p_index, l_index, p_embs, l_embs, pairs


def evaluate_split_retrieval(
    model: torch.nn.Module,
    interactions: pd.DataFrame,
    split: str,
    protein_cache_path: Path,
    ligand_cache_path: Path,
    device: torch.device,
    sim_batch_size: int,
) -> dict[str, dict[str, float]]:
    prot_cache = torch.load(protein_cache_path, map_location="cpu")
    lig_cache = torch.load(ligand_cache_path, map_location="cpu")
    _, _, p_embs, l_embs, pairs = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split=split,
        protein_cache=prot_cache,
        ligand_cache=lig_cache,
        device=device,
    )
    return evaluate_global_retrieval_multi(
        protein_embs=p_embs,
        ligand_embs=l_embs,
        interaction_pairs=pairs,
        sim_batch_size=sim_batch_size,
    )


def _val_selection_score(retrieval_metrics: dict[str, dict[str, float]]) -> float:
    """
    Convert retrieval dict into a single scalar for model selection.

    Assumes evaluate_global_retrieval_multi returns something like:
      {
        "protein_to_ligand": {"recall@1":..., "recall@5":..., "recall@10":..., "mrr":...},
        "ligand_to_protein": {"recall@1":..., "recall@5":..., "recall@10":..., "mrr":...},
      }
    We select by mean MRR across directions.
    """
    p2l = retrieval_metrics.get("protein_to_ligand", {})
    l2p = retrieval_metrics.get("ligand_to_protein", {})
    return float(0.5 * (p2l.get("mrr", 0.0) + l2p.get("mrr", 0.0)))


def train_contrastive(
    data_dir: str | Path,
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], pd.DataFrame]:
    """Train CLIP-style dual encoder projection heads on cached features and select best epoch on validation."""
    protein_cache_path, ligand_cache_path = _get_cached_paths(cache_dir)
    interactions = build_interactions(data_dir)

    p_dim, l_dim = _infer_input_dims(protein_cache_path, ligand_cache_path)
    model = MODEL_REGISTRY.build(
        cfg["model"]["name"],
        protein_input_dim=p_dim,
        ligand_input_dim=l_dim,
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        init_temperature=cfg["model"]["init_temperature"],
        max_logit_scale=cfg["model"]["max_logit_scale"],
    )
    assert isinstance(model, torch.nn.Module)
    model.to(device)

    train_loader = _build_loader(
        interactions,
        protein_cache_path,
        ligand_cache_path,
        split="train",
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    outdir = Path(cfg["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {"train_loss": []}
    epochs = int(cfg["train"]["epochs"])

    best_score = -1e9
    best_epoch = 0
    best_ckpt_path = outdir / "contrastive_model_best.pt"

    # Optional: choose which split to use for selection (default "val")
    selection_split = str(cfg.get("eval", {}).get("selection_split", "val"))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            prot = batch["protein_feature"].to(device)
            lig = batch["ligand_feature"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(prot, lig)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * prot.size(0)
            count += prot.size(0)
            pbar.set_postfix(loss=float(loss.item()))

        epoch_loss = running / max(count, 1)
        history["train_loss"].append(epoch_loss)

        # --- Validation retrieval for model selection ---
        model.eval()
        with torch.no_grad():
            val_retrieval = evaluate_split_retrieval(
                model=model,
                interactions=interactions,
                split=selection_split,
                protein_cache_path=protein_cache_path,
                ligand_cache_path=ligand_cache_path,
                device=device,
                sim_batch_size=cfg["eval"]["sim_batch_size"],
            )
        score = _val_selection_score(val_retrieval)

        # Save best checkpoint by validation score
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "protein_input_dim": p_dim,
                    "ligand_input_dim": l_dim,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "selection_split": selection_split,
                    "train_loss": epoch_loss,
                    "retrieval_selection": val_retrieval,
                },
                best_ckpt_path,
            )

    # Load best model weights for final reporting
    best = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])

    # Save a "last" checkpoint too (optional, keeps previous behavior)
    last_ckpt_path = outdir / "contrastive_model_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "protein_input_dim": p_dim,
            "ligand_input_dim": l_dim,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_split": selection_split,
        },
        last_ckpt_path,
    )

    # Final metrics: report best epoch and evaluate on val + test with best checkpoint
    metrics: dict[str, Any] = {
        "train": {"loss_last_epoch": history["train_loss"][-1]},
        "model_selection": {
            "selection_split": selection_split,
            "best_epoch": int(best_epoch),
            "best_score_mean_mrr": float(best_score),
            "best_checkpoint": str(best_ckpt_path),
        },
    }

    for split in ("val", "test"):
        metrics[f"retrieval_{split}"] = evaluate_split_retrieval(
            model=model,
            interactions=interactions,
            split=split,
            protein_cache_path=protein_cache_path,
            ligand_cache_path=ligand_cache_path,
            device=device,
            sim_batch_size=cfg["eval"]["sim_batch_size"],
        )

    save_json(metrics, outdir / "contrastive_metrics.json")
    return model, metrics, interactions
