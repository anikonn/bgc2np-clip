from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from kiba_clip.utils.io import maybe_to_parquet


def save_umap(
    protein_embs: np.ndarray,
    ligand_embs: np.ndarray,
    protein_ids: list[str],
    ligand_ids: list[str],
    outdir: str | Path,
    prefix: str,
) -> dict[str, str]:
    """Fit UMAP on combined embeddings and save figure/metadata."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    combined = np.concatenate([protein_embs, ligand_embs], axis=0)
    modalities = ["protein"] * len(protein_embs) + ["ligand"] * len(ligand_embs)
    ids = protein_ids + ligand_ids

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(combined)

    df = pd.DataFrame(
        {
            "id": ids,
            "modality": modalities,
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
        }
    )
    
    df.to_csv(out / f"{prefix}_umap_coords.csv", index=False)

    emb_cols = [f"emb_{i}" for i in range(combined.shape[1])]
    emb_df = pd.DataFrame(combined, columns=emb_cols)
    emb_df.insert(0, "modality", modalities)
    emb_df.insert(0, "id", ids)

    csv_path = out / f"{prefix}_embeddings.csv"
    png_path = out / f"{prefix}_umap.png"
    parquet_path = out / f"{prefix}_embeddings.parquet"

    emb_df.to_csv(csv_path, index=False)
    maybe_to_parquet(emb_df, parquet_path)

    plt.figure(figsize=(8, 6))
    for modality, color in [("protein", "tab:blue"), ("ligand", "tab:orange")]:
        sub = df[df["modality"] == modality]
        plt.scatter(sub["umap_x"], sub["umap_y"], s=10, alpha=0.7, c=color, label=modality)
    plt.title("Joint Embedding UMAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return {
        "png": str(png_path),
        "csv": str(csv_path),
        "parquet": str(parquet_path),
    }
