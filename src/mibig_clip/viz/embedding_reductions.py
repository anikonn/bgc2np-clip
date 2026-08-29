from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import umap

ReductionMethod = Literal["umap", "pca", "tsne"]
Normalization = Literal["none", "l2"]


METHOD_DISPLAY = {
    "umap": "UMAP",
    "pca": "PCA",
    "tsne": "t-SNE",
}


def _maybe_to_parquet(df: pd.DataFrame, parquet_path: Path) -> bool:
    try:
        df.to_parquet(parquet_path, index=False)
        return True
    except Exception:
        return False


def prepare_embeddings(embeddings: np.ndarray, normalization: Normalization) -> np.ndarray:
    """Return finite float32 embeddings with optional deterministic preprocessing."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Embedding array contains NaN or infinite values")
    if normalization == "l2":
        return normalize(arr, norm="l2", axis=1).astype(np.float32)
    if normalization == "none":
        return arr
    raise ValueError(f"Unsupported normalization: {normalization}")


def reduce_embeddings(
    embeddings: np.ndarray,
    method: ReductionMethod,
    *,
    random_state: int,
    normalization: Normalization,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    tsne_perplexity: float,
    tsne_learning_rate: str | float,
    tsne_max_iter: int,
    tsne_metric: str,
    tsne_init: str,
) -> np.ndarray:
    """Fit a 2D reducer and return coordinates."""
    x = prepare_embeddings(embeddings, normalization)
    if x.shape[0] < 2:
        raise ValueError("At least two embeddings are required for a 2D reduction")

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(x)

    if method == "umap":
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(int(umap_n_neighbors), x.shape[0] - 1),
            min_dist=float(umap_min_dist),
            metric=umap_metric,
            random_state=random_state,
        )
        return reducer.fit_transform(x)

    if method == "tsne":
        perplexity = min(float(tsne_perplexity), max(1.0, (x.shape[0] - 1) / 3.0))
        kwargs = {
            "n_components": 2,
            "perplexity": perplexity,
            "learning_rate": tsne_learning_rate,
            "metric": tsne_metric,
            "init": tsne_init,
            "random_state": random_state,
        }
        if "max_iter" in inspect.signature(TSNE).parameters:
            kwargs["max_iter"] = int(tsne_max_iter)
        else:
            kwargs["n_iter"] = int(tsne_max_iter)
        reducer = TSNE(**kwargs)
        return reducer.fit_transform(x)

    raise ValueError(f"Unsupported reduction method: {method}")


def save_embedding_table(
    embeddings: np.ndarray,
    ids: list[str],
    modalities: list[str],
    outdir: str | Path,
    prefix: str,
    labels: list[str] | None = None,
    write_tables: bool = True,
) -> dict[str, str]:
    """Persist the high-dimensional embeddings used to make the plots."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if len(ids) != len(embeddings) or len(modalities) != len(embeddings):
        raise ValueError("ids, modalities, and embeddings must have the same length")
    if not write_tables:
        return {"csv": "", "parquet": ""}
    label_values = labels if labels is not None else [""] * len(ids)
    emb_cols = [f"emb_{idx}" for idx in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(np.asarray(embeddings), columns=emb_cols)
    emb_df.insert(0, "label", label_values)
    emb_df.insert(0, "modality", modalities)
    emb_df.insert(0, "id", ids)
    csv_path = out / f"{prefix}_embeddings.csv"
    parquet_path = out / f"{prefix}_embeddings.parquet"
    emb_df.to_csv(csv_path, index=False)
    parquet_written = _maybe_to_parquet(emb_df, parquet_path)
    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path) if parquet_written else "",
    }


def save_single_modality_reduction(
    embeddings: np.ndarray,
    ids: list[str],
    *,
    modality_name: str,
    embedding_space_name: str,
    method: ReductionMethod,
    outdir: str | Path,
    prefix: str,
    labels: list[str] | None = None,
    color_by_label: bool = False,
    random_state: int = 42,
    normalization: Normalization = "none",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
    tsne_max_iter: int = 1000,
    tsne_metric: str = "cosine",
    tsne_init: str = "random",
    label_name: str | None = None,
    max_legend_labels: int | None = None,
    label_order: list[str] | None = None,
    write_tables: bool = True,
) -> dict[str, str]:
    """Save a 2D reduction plot and coordinates for one modality."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    label_values = labels if labels is not None else [""] * len(ids)
    coords = reduce_embeddings(
        embeddings,
        method,
        random_state=random_state,
        normalization=normalization,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        tsne_perplexity=tsne_perplexity,
        tsne_learning_rate=tsne_learning_rate,
        tsne_max_iter=tsne_max_iter,
        tsne_metric=tsne_metric,
        tsne_init=tsne_init,
    )

    method_label = METHOD_DISPLAY[method]
    coords_path = out / f"{prefix}_{method}_coords.csv"
    png_path = out / f"{prefix}_{method}.png"
    df = pd.DataFrame(
        {
            "id": ids,
            "modality": modality_name,
            "label": label_values,
            f"{method}_x": coords[:, 0],
            f"{method}_y": coords[:, 1],
        }
    )
    if write_tables:
        df.to_csv(coords_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    if color_by_label and any(label_values):
        if label_order is None:
            unique_labels = sorted(set(label_values))
        else:
            seen_labels = set(label_values)
            unique_labels = [label for label in label_order if label in seen_labels]
            unique_labels.extend(sorted(seen_labels.difference(unique_labels)))
        cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
        for idx, label in enumerate(unique_labels):
            sub = df[df["label"] == label]
            ax.scatter(
                sub[f"{method}_x"],
                sub[f"{method}_y"],
                s=12,
                alpha=0.75,
                color=cmap(idx),
                label=label,
                linewidths=0,
            )
        if max_legend_labels is None or len(unique_labels) <= int(max_legend_labels):
            ax.legend(
                title=label_name or "Class",
                fontsize=7,
                title_fontsize=8,
                markerscale=1.3,
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
            )
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.75, color="tab:blue", linewidths=0)

    ax.set_title(f"{method_label} of {modality_name} embeddings ({embedding_space_name})")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return {"png": str(png_path), "coords_csv": str(coords_path) if write_tables else ""}


def save_single_modality_continuous_reduction(
    embeddings: np.ndarray,
    ids: list[str],
    *,
    values: list[float | int | None],
    value_name: str,
    modality_name: str,
    embedding_space_name: str,
    method: ReductionMethod,
    outdir: str | Path,
    prefix: str,
    random_state: int = 42,
    normalization: Normalization = "none",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
    tsne_max_iter: int = 1000,
    tsne_metric: str = "cosine",
    tsne_init: str = "random",
    write_tables: bool = True,
    cmap: str = "viridis",
) -> dict[str, str]:
    """Save one-modality 2D reduction colored by a continuous value."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if len(values) != len(ids):
        raise ValueError("values must have the same length as ids")

    coords = reduce_embeddings(
        embeddings,
        method,
        random_state=random_state,
        normalization=normalization,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        tsne_perplexity=tsne_perplexity,
        tsne_learning_rate=tsne_learning_rate,
        tsne_max_iter=tsne_max_iter,
        tsne_metric=tsne_metric,
        tsne_init=tsne_init,
    )

    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(numeric_values)
    if not finite_mask.any():
        raise ValueError(f"No finite values available for {value_name}")

    method_label = METHOD_DISPLAY[method]
    coords_path = out / f"{prefix}_{method}_coords.csv"
    png_path = out / f"{prefix}_{method}.png"
    df = pd.DataFrame(
        {
            "id": ids,
            "modality": modality_name,
            "value_name": value_name,
            "value": numeric_values,
            f"{method}_x": coords[:, 0],
            f"{method}_y": coords[:, 1],
        }
    )
    if write_tables:
        df.to_csv(coords_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    if (~finite_mask).any():
        ax.scatter(
            coords[~finite_mask, 0],
            coords[~finite_mask, 1],
            s=10,
            alpha=0.35,
            color="0.75",
            linewidths=0,
            label="missing",
        )
    points = ax.scatter(
        coords[finite_mask, 0],
        coords[finite_mask, 1],
        c=numeric_values[finite_mask],
        s=12,
        alpha=0.8,
        cmap=cmap,
        linewidths=0,
    )
    cbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_name)
    if (~finite_mask).any():
        ax.legend(frameon=False, loc="best", fontsize=7)

    ax.set_title(f"{method_label} of {modality_name} embeddings colored by {value_name}\n({embedding_space_name})")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return {"png": str(png_path), "coords_csv": str(coords_path) if write_tables else ""}


def save_joint_modality_reduction(
    bgc_embeddings: np.ndarray,
    np_embeddings: np.ndarray,
    bgc_ids: list[str],
    np_ids: list[str],
    *,
    embedding_space_name: str,
    method: ReductionMethod,
    outdir: str | Path,
    prefix: str,
    random_state: int = 42,
    normalization: Normalization = "none",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
    tsne_max_iter: int = 1000,
    tsne_metric: str = "cosine",
    tsne_init: str = "random",
    color_labels: list[str] | None = None,
    color_label_name: str | None = None,
    max_legend_labels: int | None = None,
    pair_edges: list[tuple[str, str]] | None = None,
    edge_alpha: float = 0.12,
    edge_linewidth: float = 0.9,
    write_tables: bool = True,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> dict[str, str]:
    """Fit one reducer on BGC and NP embeddings together, using shape to mark modality."""
    if bgc_embeddings.shape[1] != np_embeddings.shape[1]:
        raise ValueError(
            "Joint modality plots require BGC and NP embeddings with the same dimensionality; "
            f"got {bgc_embeddings.shape[1]} and {np_embeddings.shape[1]}"
        )
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    combined = np.concatenate([bgc_embeddings, np_embeddings], axis=0)
    ids = bgc_ids + np_ids
    modalities = ["BGC"] * len(bgc_ids) + ["NP"] * len(np_ids)
    if color_labels is not None and len(color_labels) != len(ids):
        raise ValueError("color_labels must match the combined BGC+NP row count")
    color_values = color_labels if color_labels is not None else modalities
    coords = reduce_embeddings(
        combined,
        method,
        random_state=random_state,
        normalization=normalization,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        tsne_perplexity=tsne_perplexity,
        tsne_learning_rate=tsne_learning_rate,
        tsne_max_iter=tsne_max_iter,
        tsne_metric=tsne_metric,
        tsne_init=tsne_init,
    )

    method_label = METHOD_DISPLAY[method]
    coords_path = out / f"{prefix}_{method}_coords.csv"
    png_path = out / f"{prefix}_{method}.png"
    df = pd.DataFrame(
        {
            "id": ids,
            "modality": modalities,
            "color_label": color_values,
            f"{method}_x": coords[:, 0],
            f"{method}_y": coords[:, 1],
        }
    )
    if write_tables:
        df.to_csv(coords_path, index=False)

    fig, ax = plt.subplots(figsize=figsize)
    edge_path = out / f"{prefix}_{method}_edges.csv"
    edge_written = False
    if pair_edges:
        bgc_coords = {
            str(row.id): (float(getattr(row, f"{method}_x")), float(getattr(row, f"{method}_y")))
            for row in df[df["modality"] == "BGC"].itertuples(index=False)
        }
        np_coords = {
            str(row.id): (float(getattr(row, f"{method}_x")), float(getattr(row, f"{method}_y")))
            for row in df[df["modality"] == "NP"].itertuples(index=False)
        }
        edge_rows = []
        for bgc_id, np_id in pair_edges:
            if bgc_id not in bgc_coords or np_id not in np_coords:
                continue
            x0, y0 = bgc_coords[bgc_id]
            x1, y1 = np_coords[np_id]
            edge_rows.append(
                {
                    "bgc_id": bgc_id,
                    "np_id": np_id,
                    f"{method}_bgc_x": x0,
                    f"{method}_bgc_y": y0,
                    f"{method}_np_x": x1,
                    f"{method}_np_y": y1,
                }
            )
            ax.plot([x0, x1], [y0, y1], color="0.25", alpha=float(edge_alpha), linewidth=float(edge_linewidth), zorder=1)
        if write_tables:
            pd.DataFrame(edge_rows).to_csv(edge_path, index=False)
            edge_written = True

    style = {
        "BGC": {"marker": "o", "label": "BGCs"},
        "NP": {"marker": "^", "label": "NPs"},
    }
    if color_labels is None:
        color_map = {"BGC": "tab:blue", "NP": "tab:orange"}
    else:
        unique_labels = sorted(label for label in set(color_values) if label != "unknown")
        if "unknown" in set(color_values):
            unique_labels.append("unknown")
        cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
        color_map = {
            label: ("0.7" if label == "unknown" else cmap(idx % 20))
            for idx, label in enumerate(unique_labels)
        }

    for modality, marker_style in style.items():
        modality_df = df[df["modality"] == modality]
        for label, sub in modality_df.groupby("color_label", sort=True):
            ax.scatter(
                sub[f"{method}_x"],
                sub[f"{method}_y"],
                s=14,
                alpha=0.7,
                linewidths=0,
                color=color_map.get(str(label), "0.7"),
                marker=marker_style["marker"],
                zorder=2,
            )
    color_suffix = f"\ncolored by {color_label_name}" if color_label_name else ""
    ax.set_title(f"Joint {method_label} of BGC and NP embeddings{color_suffix}\n({embedding_space_name})")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.grid(alpha=0.2, linewidth=0.6)
    shape_handles = [
        Line2D([0], [0], marker=style["BGC"]["marker"], color="black", linestyle="", label="BGCs", markersize=6),
        Line2D([0], [0], marker=style["NP"]["marker"], color="black", linestyle="", label="NPs", markersize=6),
    ]
    if pair_edges:
        shape_handles.append(Line2D([0], [0], color="0.25", alpha=0.7, linestyle="-", label="Known pair", linewidth=1.0))
    shape_legend = ax.legend(handles=shape_handles, title="Modality", frameon=False, loc="best")
    ax.add_artist(shape_legend)
    if color_labels is not None and (max_legend_labels is None or len(color_map) <= int(max_legend_labels)):
        color_handles = [
            Line2D([0], [0], marker="o", color=color, linestyle="", label=label, markersize=6)
            for label, color in color_map.items()
        ]
        ax.legend(
            handles=color_handles,
            title=color_label_name or "Label",
            frameon=False,
            fontsize=7,
            title_fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.08, top=0.88)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return {
        "png": str(png_path),
        "coords_csv": str(coords_path) if write_tables else "",
        "edges_csv": str(edge_path) if edge_written else "",
    }
