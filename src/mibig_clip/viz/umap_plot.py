from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import umap


def _maybe_to_parquet(df: pd.DataFrame, parquet_path: Path) -> bool:
    try:
        df.to_parquet(parquet_path, index=False)
        return True
    except Exception:
        return False


def save_joint_umap(
    bgc_embs: np.ndarray,
    compound_embs: np.ndarray,
    bgc_ids: list[str],
    compound_ids: list[str],
    bgc_classes: dict[str, str],
    outdir: str | Path,
    prefix: str,
) -> dict[str, str]:
    """Fit UMAP on combined BGC and compound embeddings and save figure/data."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    combined = np.concatenate([bgc_embs, compound_embs], axis=0)
    modalities = ["bgc"] * len(bgc_embs) + ["compound"] * len(compound_embs)
    ids = bgc_ids + compound_ids
    classes = [bgc_classes.get(item_id, "") for item_id in bgc_ids] + [""] * len(compound_ids)

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(combined)

    df = pd.DataFrame(
        {
            "id": ids,
            "modality": modalities,
            "bgc_class": classes,
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
        }
    )
    df.to_csv(out / f"{prefix}_umap_coords.csv", index=False)

    emb_cols = [f"emb_{i}" for i in range(combined.shape[1])]
    emb_df = pd.DataFrame(combined, columns=emb_cols)
    emb_df.insert(0, "bgc_class", classes)
    emb_df.insert(0, "modality", modalities)
    emb_df.insert(0, "id", ids)

    csv_path = out / f"{prefix}_embeddings.csv"
    png_path = out / f"{prefix}_umap.png"
    parquet_path = out / f"{prefix}_embeddings.parquet"

    emb_df.to_csv(csv_path, index=False)
    _maybe_to_parquet(emb_df, parquet_path)

    plt.figure(figsize=(8, 6))
    for modality, color in [("bgc", "tab:blue"), ("compound", "tab:orange")]:
        sub = df[df["modality"] == modality]
        plt.scatter(sub["umap_x"], sub["umap_y"], s=10, alpha=0.7, c=color, label=modality)
    plt.title("Joint BGC-Compound Embedding UMAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return {
        "png": str(png_path),
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "coords_csv": str(out / f"{prefix}_umap_coords.csv"),
    }


def save_bgc_class_umap(
    bgc_embs: np.ndarray,
    bgc_ids: list[str],
    bgc_classes: dict[str, str],
    outdir: str | Path,
    prefix: str,
) -> dict[str, str]:
    """Fit UMAP on BGC embeddings only and color points by BGC class."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    class_labels = [bgc_classes.get(item_id, "unknown") for item_id in bgc_ids]
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(bgc_embs)

    df = pd.DataFrame(
        {
            "id": bgc_ids,
            "bgc_class": class_labels,
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
        }
    )
    coords_path = out / f"{prefix}_umap_coords.csv"
    csv_path = out / f"{prefix}_embeddings.csv"
    png_path = out / f"{prefix}_umap.png"
    parquet_path = out / f"{prefix}_embeddings.parquet"

    df.to_csv(coords_path, index=False)

    emb_cols = [f"emb_{i}" for i in range(bgc_embs.shape[1])]
    emb_df = pd.DataFrame(bgc_embs, columns=emb_cols)
    emb_df.insert(0, "bgc_class", class_labels)
    emb_df.insert(0, "id", bgc_ids)
    emb_df.to_csv(csv_path, index=False)
    _maybe_to_parquet(emb_df, parquet_path)

    plt.figure(figsize=(10, 8))
    unique_classes = sorted(set(class_labels))
    cmap = plt.get_cmap("tab20", max(len(unique_classes), 1))
    for idx, bgc_class in enumerate(unique_classes):
        sub = df[df["bgc_class"] == bgc_class]
        plt.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=14,
            alpha=0.75,
            color=cmap(idx),
            label=bgc_class,
        )
    plt.title("BGC Embedding UMAP by BGC Class")
    if len(unique_classes) <= 20:
        plt.legend(fontsize=8, markerscale=1.2, frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return {
        "png": str(png_path),
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "coords_csv": str(coords_path),
    }
