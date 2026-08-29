from __future__ import annotations

import argparse
import ast
import inspect
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from mibig_clip.viz.embedding_reductions import (
    ReductionMethod,
    save_embedding_table,
    save_joint_modality_reduction,
    save_single_modality_continuous_reduction,
    save_single_modality_reduction,
)
from projects.mibig_bgc_np.data.datasets import load_pair_table
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.training.contrastive_trainer import _pad_bgc_features


DEFAULT_METHODS: tuple[ReductionMethod, ...] = ("umap", "pca", "tsne")
NPCLASSIFIER_LEVELS = ("pathway", "superclass", "class")
NPCLASSIFIER_DOWNSTREAM_SPECS = {
    "pathway": {"counts_path": None, "min_count": None},
    "superclass": {"counts_path": Path("results/downstream_distributions/npclassifier_superclass_counts.csv"), "min_count": 100},
    "class": {"counts_path": Path("results/downstream_distributions/npclassifier_class_counts.csv"), "min_count": 50},
}
MOLECULAR_PROPERTY_SPECS = {
    "logp": {"display_name": "logP", "min_value": -10.0, "max_value": 10.0},
    "molecular_weight": {"display_name": "Molecular weight", "min_value": 0.0, "max_value": 2000.0},
    "tpsa": {"display_name": "TPSA", "min_value": 0.0, "max_value": 600.0},
}


def _setup_logger(name: str = "mibig_bgc_np") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def _save_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate UMAP, PCA, and t-SNE visualizations for frozen and BGC2NP-CLIP "
            "BGC/NP embeddings."
        )
    )
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory with bgc_features.pt and compound_features.pt.")
    parser.add_argument("--out_dir", type=str, default="results/embedding_reductions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional BGC2NP-CLIP checkpoint. Enables projected and joint embedding plots.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/MIBIG/processed",
        help="Processed data directory. Used to color BGC-only and joint plots by BGC class.",
    )
    parser.add_argument("--splits_path", type=str, default=None, help="Optional split TSV for BGC class labels.")
    parser.add_argument("--cv_fold", type=int, default=None, help="Optional CV fold for split-derived BGC class labels.")
    parser.add_argument("--val_fold", type=int, default=None, help="Optional validation fold for split-derived BGC class labels.")
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--normalization",
        choices=["none", "l2"],
        default="none",
        help="Preprocessing applied before dimensionality reduction. CLIP embeddings are already L2-normalized.",
    )
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_learning_rate", default="auto")
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    parser.add_argument("--tsne_metric", type=str, default="cosine")
    parser.add_argument("--tsne_init", type=str, default="random")
    parser.add_argument(
        "--npclassifier_pair_labels_path",
        type=str,
        default="data/MIBIG/processed/mibig_pairs_npclassifier_labels.tsv",
        help="Optional pair-level NPClassifier labels. Used for pathway/superclass/class-colored joint plots when present.",
    )
    parser.add_argument(
        "--molecular_property_values_path",
        type=str,
        default="results/downstream_distributions/molecular_property_values.csv",
        help="CSV with canonical_smiles, logp, molecular_weight, and tpsa values for NP-colored plots.",
    )
    parser.add_argument(
        "--max_points_per_modality",
        type=int,
        default=None,
        help="Optional deterministic cap for quick drafts. Uses sorted IDs and keeps the first N per modality.",
    )
    parser.add_argument(
        "--single_class_only",
        action="store_true",
        help=(
            "Keep only BGCs and NPs with exactly one parent BGC class. "
            "Use a separate out_dir to keep the full-data plots."
        ),
    )
    parser.add_argument("--skip_frozen", action="store_true", help="Skip raw frozen embedding plots.")
    parser.add_argument("--skip_clip", action="store_true", help="Skip checkpoint-projected embedding plots.")
    parser.add_argument("--skip_joint", action="store_true", help="Skip joint BGC/NP plots in the CLIP space.")
    parser.add_argument(
        "--only_molecular_properties",
        action="store_true",
        help="Only save NP plots colored by logP, molecular weight, and TPSA.",
    )
    parser.add_argument("--no_tables", action="store_true", help="Save PNG figures and manifest only; skip CSV/parquet tables.")
    parser.add_argument(
        "--skip_pair_edges",
        action="store_true",
        help="Skip additional joint plots with known BGC-NP pair edges.",
    )
    parser.add_argument("--pair_edge_linewidth", type=float, default=0.9, help="Line width for BGC-NP pair edges.")
    parser.add_argument("--pair_edge_alpha", type=float, default=0.12, help="Opacity for BGC-NP pair edges.")
    return parser.parse_args()


def _load_cache(cache_dir: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    root = Path(cache_dir)
    bgc_path = root / "bgc_features.pt"
    compound_path = root / "compound_features.pt"
    if not bgc_path.exists():
        raise FileNotFoundError(f"Missing BGC feature cache: {bgc_path}")
    if not compound_path.exists():
        raise FileNotFoundError(f"Missing compound/NP feature cache: {compound_path}")
    return _torch_load(bgc_path, map_location="cpu"), _torch_load(compound_path, map_location="cpu")


def _torch_load(path: str | Path, *, map_location: str | torch.device):
    if "weights_only" in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location)


def _stack_cache(
    cache: dict[str, torch.Tensor],
    *,
    max_points: int | None,
) -> tuple[list[str], np.ndarray]:
    ids = sorted(str(item_id) for item_id in cache.keys())
    if max_points is not None:
        ids = ids[: int(max_points)]
    features: list[torch.Tensor] = []
    for item_id in ids:
        feature = cache[item_id].float()
        if feature.ndim == 1:
            features.append(feature)
        elif feature.ndim == 2:
            features.append(feature.mean(dim=0))
        else:
            raise ValueError(f"Cached feature for {item_id} must be 1D or 2D, got {tuple(feature.shape)}")
    embeddings = torch.stack(features).cpu().numpy()
    return ids, embeddings


def _load_model(ckpt_path: str | Path, device: torch.device) -> DualEncoderCLIP:
    ckpt = _torch_load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = DualEncoderCLIP(
        bgc_input_dim=int(ckpt["bgc_input_dim"]),
        compound_input_dim=int(ckpt["compound_input_dim"]),
        emb_dim=int(cfg["model"]["emb_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        init_temperature=float(cfg["model"]["init_temperature"]),
        max_logit_scale=float(cfg["model"]["max_logit_scale"]),
        bgc_aggregation=str(cfg["model"].get("bgc_aggregation", "prepooled")),
        bgc_aggregation_config=cfg["model"].get("bgc_aggregation_config", {}),
        projection_head=str(cfg["model"].get("projection_head", "mlp_gelu")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _project_cache(
    model: DualEncoderCLIP,
    cache: dict[str, torch.Tensor],
    ids: list[str],
    *,
    modality: str,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            if modality == "bgc":
                features, padding_mask = _pad_bgc_features(
                    [cache[item_id].float() for item_id in batch_ids],
                    device,
                )
                chunks.append(model.encode_bgc(features, padding_mask=padding_mask).cpu())
            elif modality in {"compound", "np"}:
                features = torch.stack([cache[item_id].float().reshape(-1) for item_id in batch_ids]).to(device)
                chunks.append(model.encode_compound(features).cpu())
            else:
                raise ValueError(f"Unsupported modality: {modality}")
    return torch.cat(chunks, dim=0).numpy()


def _parse_label_text(label_text: object) -> list[str]:
    raw = "" if label_text is None else str(label_text)
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        candidates = [str(label) for label in parsed]
    else:
        candidates = re.split(r"[;,]", raw)
    labels: list[str] = []
    seen: set[str] = set()
    for label in candidates:
        clean = str(label).strip().strip("'\"")
        if clean and clean.lower() != "nan" and clean not in seen:
            labels.append(clean)
            seen.add(clean)
    return labels


def _collapse_labels(values: list[object]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        for label in _parse_label_text(value):
            if label not in seen:
                labels.append(label)
                seen.add(label)
    return "; ".join(sorted(labels)) if labels else "unknown"


def _infer_compound_id_column(df: pd.DataFrame) -> str:
    for column in ("compound_id", "canonical_smiles", "smiles", "product", "compound_name"):
        if column in df.columns:
            return column
    raise ValueError("Could not infer compound/NP identifier column")


def _load_bgc_class_label_maps(data_dir: str | None) -> tuple[dict[str, str], dict[str, str]]:
    if data_dir is None:
        return {}, {}
    pair_df = load_pair_table(data_dir)
    if "bgc_classes" in pair_df.columns:
        label_col = "bgc_classes"
    elif "bgc_class" in pair_df.columns:
        label_col = "bgc_class"
    else:
        return {}, {}
    bgc_labels = {
        str(bgc_id): _collapse_labels(group[label_col].tolist())
        for bgc_id, group in pair_df.groupby("bgc_id", sort=True)
    }
    np_labels = {
        str(compound_id): _collapse_labels(group[label_col].tolist())
        for compound_id, group in pair_df.groupby("compound_id", sort=True)
    }
    return bgc_labels, np_labels


def _load_pair_edges(data_dir: str | None, bgc_ids: list[str], np_ids: list[str]) -> list[tuple[str, str]]:
    if data_dir is None:
        return []
    pair_df = load_pair_table(data_dir)
    required = {"bgc_id", "compound_id"}
    missing = required.difference(pair_df.columns)
    if missing:
        raise ValueError(f"Pair table is missing columns needed for edge plots: {sorted(missing)}")
    bgc_set = set(bgc_ids)
    np_set = set(np_ids)
    edges = (
        pair_df[["bgc_id", "compound_id"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["bgc_id", "compound_id"])
    )
    return [
        (str(row.bgc_id), str(row.compound_id))
        for row in edges.itertuples(index=False)
        if str(row.bgc_id) in bgc_set and str(row.compound_id) in np_set
    ]


def _load_npclassifier_label_maps(
    path: str | Path | None,
) -> tuple[dict[str, tuple[dict[str, str], dict[str, str]]], dict[str, list[str]], list[str]]:
    empty = {level: ({}, {}) for level in NPCLASSIFIER_LEVELS}
    if path is None:
        return empty, {}, ["No NPClassifier pair-label path was provided."]
    table_path = Path(path)
    if not table_path.exists():
        return empty, {}, [f"NPClassifier pair-label table not found: {table_path}"]
    labels_df = pd.read_csv(table_path, sep="\t")
    required = {"bgc_id"} | {f"npclassifier_{level}" for level in NPCLASSIFIER_LEVELS}
    missing = required.difference(labels_df.columns)
    if missing:
        return empty, {}, [f"NPClassifier pair-label table is missing columns: {sorted(missing)}"]
    compound_col = _infer_compound_id_column(labels_df)
    labels_df = labels_df.copy()
    labels_df["bgc_id"] = labels_df["bgc_id"].astype(str)
    labels_df["_compound_id"] = labels_df[compound_col].astype(str)
    label_maps: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    allowed_labels_by_level: dict[str, list[str]] = {}
    for level in NPCLASSIFIER_LEVELS:
        label_col = f"npclassifier_{level}"
        allowed_labels = _load_downstream_npclassifier_labels(labels_df, level=level, label_col=label_col)
        allowed_set = set(allowed_labels)
        allowed_labels_by_level[level] = allowed_labels
        bgc_labels = {
            str(bgc_id): _collapse_allowed_labels(group[label_col].tolist(), allowed_set)
            for bgc_id, group in labels_df.groupby("bgc_id", sort=True)
        }
        np_labels = {
            str(compound_id): _collapse_allowed_labels(group[label_col].tolist(), allowed_set)
            for compound_id, group in labels_df.groupby("_compound_id", sort=True)
        }
        label_maps[level] = (bgc_labels, np_labels)
    return label_maps, allowed_labels_by_level, []


def _load_molecular_property_maps(path: str | Path | None) -> tuple[dict[str, dict[str, float]], list[str]]:
    empty = {key: {} for key in MOLECULAR_PROPERTY_SPECS}
    if path is None:
        return empty, ["No molecular property value path was provided."]
    table_path = Path(path)
    if not table_path.exists():
        return empty, [f"Molecular property value table not found: {table_path}"]
    values_df = pd.read_csv(table_path)
    required = {"canonical_smiles"} | set(MOLECULAR_PROPERTY_SPECS)
    missing = required.difference(values_df.columns)
    if missing:
        return empty, [f"Molecular property value table is missing columns: {sorted(missing)}"]
    values_df = values_df.copy()
    values_df["canonical_smiles"] = values_df["canonical_smiles"].astype(str)
    property_maps: dict[str, dict[str, float]] = {}
    for key in MOLECULAR_PROPERTY_SPECS:
        numeric_values = pd.to_numeric(values_df[key], errors="coerce")
        valid = values_df.loc[numeric_values.notna(), ["canonical_smiles"]].copy()
        valid[key] = numeric_values.loc[numeric_values.notna()].astype(float).to_numpy()
        property_maps[key] = dict(zip(valid["canonical_smiles"], valid[key], strict=False))
    return property_maps, []


def _load_downstream_npclassifier_labels(labels_df: pd.DataFrame, *, level: str, label_col: str) -> list[str]:
    spec = NPCLASSIFIER_DOWNSTREAM_SPECS[level]
    counts_path = spec["counts_path"]
    min_count = spec["min_count"]
    if counts_path is None:
        return sorted({label for value in labels_df[label_col].tolist() for label in _parse_label_text(value)})
    counts_df = pd.read_csv(counts_path)
    required = {"label", "n_compounds"}
    missing = required.difference(counts_df.columns)
    if missing:
        raise ValueError(f"NPClassifier count table {counts_path} is missing columns: {sorted(missing)}")
    threshold = int(min_count) if min_count is not None else 0
    filtered = counts_df[pd.to_numeric(counts_df["n_compounds"], errors="coerce") > threshold].copy()
    return [str(label) for label in filtered["label"].tolist()]


def _collapse_allowed_labels(values: list[object], allowed_labels: set[str]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        for label in _parse_label_text(value):
            if label in allowed_labels and label not in seen:
                labels.append(label)
                seen.add(label)
    return "; ".join(sorted(labels)) if labels else "unknown"


def _label_bgc_ids(
    bgc_ids: list[str],
    *,
    bgc_class_map: dict[str, str],
) -> list[str] | None:
    if not bgc_class_map:
        return None
    return [bgc_class_map.get(item_id, "unknown") for item_id in bgc_ids]


def _label_np_ids_by_parent_bgc_class(np_ids: list[str], *, np_class_map: dict[str, str]) -> list[str] | None:
    if not np_class_map:
        return None
    return [np_class_map.get(item_id, "unknown") for item_id in np_ids]


def _label_order(*label_maps: dict[str, str]) -> list[str]:
    labels = sorted({label for label_map in label_maps for label in label_map.values() if label != "unknown"})
    if any("unknown" in label_map.values() for label_map in label_maps):
        labels.append("unknown")
    return labels


def _is_single_class_label(label: str | None) -> bool:
    if label is None or label == "unknown":
        return False
    return len(_parse_label_text(label)) == 1


def _filter_embeddings_by_single_class(
    ids: list[str],
    embeddings: np.ndarray,
    label_map: dict[str, str],
) -> tuple[list[str], np.ndarray]:
    keep_indices = [
        idx
        for idx, item_id in enumerate(ids)
        if _is_single_class_label(label_map.get(item_id, "unknown"))
    ]
    filtered_ids = [ids[idx] for idx in keep_indices]
    filtered_embeddings = embeddings[np.asarray(keep_indices, dtype=int)]
    return filtered_ids, filtered_embeddings


def _filter_pair_edges(pair_edges: list[tuple[str, str]], bgc_ids: list[str], np_ids: list[str]) -> list[tuple[str, str]]:
    bgc_set = set(bgc_ids)
    np_set = set(np_ids)
    return [(bgc_id, np_id) for bgc_id, np_id in pair_edges if bgc_id in bgc_set and np_id in np_set]


def _joint_labels(
    bgc_ids: list[str],
    np_ids: list[str],
    *,
    bgc_label_map: dict[str, str],
    np_label_map: dict[str, str],
) -> list[str] | None:
    if not bgc_label_map and not np_label_map:
        return None
    return [bgc_label_map.get(item_id, "unknown") for item_id in bgc_ids] + [
        np_label_map.get(item_id, "unknown") for item_id in np_ids
    ]


def _label_bgc_ids_from_data_dir(
    bgc_ids: list[str],
    *,
    data_dir: str | None,
) -> list[str] | None:
    if data_dir is None:
        return None
    bgc_class_map, _ = _load_bgc_class_label_maps(data_dir)
    return _label_bgc_ids(bgc_ids, bgc_class_map=bgc_class_map)


def _common_reducer_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "random_state": int(args.random_state),
        "normalization": args.normalization,
        "umap_n_neighbors": int(args.umap_n_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "umap_metric": str(args.umap_metric),
        "tsne_perplexity": float(args.tsne_perplexity),
        "tsne_learning_rate": args.tsne_learning_rate,
        "tsne_max_iter": int(args.tsne_max_iter),
        "tsne_metric": str(args.tsne_metric),
        "tsne_init": str(args.tsne_init),
    }


def _save_single_set(
    *,
    bgc_embeddings: np.ndarray,
    np_embeddings: np.ndarray,
    bgc_ids: list[str],
    np_ids: list[str],
    bgc_labels: list[str] | None,
    np_labels: list[str] | None = None,
    label_name: str | None = None,
    label_order: list[str] | None = None,
    space_key: str,
    space_name: str,
    outdir: Path,
    methods: list[ReductionMethod],
    reducer_kwargs: dict[str, Any],
    write_tables: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"embeddings": {}, "figures": {}}
    space_dir = outdir / space_key

    payload["embeddings"]["bgc"] = save_embedding_table(
        bgc_embeddings,
        bgc_ids,
        ["BGC"] * len(bgc_ids),
        space_dir,
        "bgc",
        labels=bgc_labels,
        write_tables=write_tables,
    )
    payload["embeddings"]["np"] = save_embedding_table(
        np_embeddings,
        np_ids,
        ["NP"] * len(np_ids),
        space_dir,
        "np",
        labels=np_labels,
        write_tables=write_tables,
    )

    for method in methods:
        payload["figures"][f"bgc_{method}"] = save_single_modality_reduction(
            bgc_embeddings,
            bgc_ids,
            modality_name="BGC",
            embedding_space_name=space_name,
            method=method,
            outdir=space_dir,
            prefix="bgc",
            labels=bgc_labels,
            color_by_label=bgc_labels is not None,
            label_name=label_name,
            label_order=label_order,
            write_tables=write_tables,
            **reducer_kwargs,
        )
        payload["figures"][f"np_{method}"] = save_single_modality_reduction(
            np_embeddings,
            np_ids,
            modality_name="NP",
            embedding_space_name=space_name,
            method=method,
            outdir=space_dir,
            prefix="np",
            labels=np_labels,
            color_by_label=np_labels is not None,
            label_name=label_name,
            label_order=label_order,
            write_tables=write_tables,
            **reducer_kwargs,
        )
    return payload


def _save_npclassifier_single_modality_sets(
    *,
    bgc_embeddings: np.ndarray,
    np_embeddings: np.ndarray,
    bgc_ids: list[str],
    np_ids: list[str],
    npclassifier_label_maps: dict[str, tuple[dict[str, str], dict[str, str]]],
    npclassifier_allowed_labels: dict[str, list[str]],
    space_key: str,
    space_name: str,
    outdir: Path,
    methods: list[ReductionMethod],
    reducer_kwargs: dict[str, Any],
    write_tables: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    space_dir = outdir / space_key
    for level, (bgc_label_map, np_label_map) in npclassifier_label_maps.items():
        level_payload: dict[str, Any] = {"bgc": {}, "np": {}}
        level_bgc_ids, level_bgc_embeddings = _filter_embeddings_by_single_class(
            bgc_ids,
            bgc_embeddings,
            bgc_label_map,
        )
        level_np_ids, level_np_embeddings = _filter_embeddings_by_single_class(
            np_ids,
            np_embeddings,
            np_label_map,
        )
        bgc_labels = _label_bgc_ids(level_bgc_ids, bgc_class_map=bgc_label_map)
        np_labels = _label_np_ids_by_parent_bgc_class(level_np_ids, np_class_map=np_label_map)
        label_order = npclassifier_allowed_labels.get(level)
        display_level = level.capitalize()
        for method in methods:
            if len(level_bgc_ids) >= 2:
                level_payload["bgc"][method] = save_single_modality_reduction(
                    level_bgc_embeddings,
                    level_bgc_ids,
                    modality_name="BGC",
                    embedding_space_name=space_name,
                    method=method,
                    outdir=space_dir,
                    prefix=f"bgc_by_npclassifier_{level}",
                    labels=bgc_labels,
                    color_by_label=True,
                    label_name=f"NPClassifier {display_level}",
                    label_order=label_order,
                    write_tables=write_tables,
                    **reducer_kwargs,
                )
            if len(level_np_ids) >= 2:
                level_payload["np"][method] = save_single_modality_reduction(
                    level_np_embeddings,
                    level_np_ids,
                    modality_name="NP",
                    embedding_space_name=space_name,
                    method=method,
                    outdir=space_dir,
                    prefix=f"np_by_npclassifier_{level}",
                    labels=np_labels,
                    color_by_label=True,
                    label_name=f"NPClassifier {display_level}",
                    label_order=label_order,
                    write_tables=write_tables,
                    **reducer_kwargs,
                )
        level_payload["n_bgcs"] = len(level_bgc_ids)
        level_payload["n_nps"] = len(level_np_ids)
        payload[level] = level_payload
    return payload


def _save_molecular_property_np_sets(
    *,
    np_embeddings: np.ndarray,
    np_ids: list[str],
    molecular_property_maps: dict[str, dict[str, float]],
    space_key: str,
    space_name: str,
    outdir: Path,
    methods: list[ReductionMethod],
    reducer_kwargs: dict[str, Any],
    write_tables: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    space_dir = outdir / space_key
    for property_key, property_spec in MOLECULAR_PROPERTY_SPECS.items():
        property_name = str(property_spec["display_name"])
        min_value = float(property_spec["min_value"])
        max_value = float(property_spec["max_value"])
        value_map = molecular_property_maps.get(property_key, {})
        values = np.asarray([value_map.get(item_id, np.nan) for item_id in np_ids], dtype=float)
        finite_mask = np.isfinite(values)
        range_mask = finite_mask & (values >= min_value) & (values <= max_value)
        keep_indices = np.flatnonzero(range_mask)
        filtered_np_ids = [np_ids[idx] for idx in keep_indices]
        filtered_np_embeddings = np_embeddings[keep_indices]
        filtered_values = values[keep_indices].tolist()
        property_payload: dict[str, Any] = {
            "n_values": int(finite_mask.sum()),
            "n_values_in_range": int(range_mask.sum()),
            "min_value": min_value,
            "max_value": max_value,
            "figures": {},
        }
        if len(filtered_np_ids) < 2:
            payload[property_key] = property_payload
            continue
        for method in methods:
            property_payload["figures"][method] = save_single_modality_continuous_reduction(
                filtered_np_embeddings,
                filtered_np_ids,
                values=filtered_values,
                value_name=f"{property_name} ({min_value:g} to {max_value:g})",
                modality_name="NP",
                embedding_space_name=space_name,
                method=method,
                outdir=space_dir,
                prefix=f"np_by_{property_key}",
                write_tables=write_tables,
                **reducer_kwargs,
            )
        payload[property_key] = property_payload
    return payload


def main() -> None:
    args = parse_args()
    logger = _setup_logger("mibig_bgc_np")
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = [str(method) for method in args.methods]
    reducer_kwargs = _common_reducer_kwargs(args)
    write_tables = not bool(args.no_tables)

    bgc_cache, np_cache = _load_cache(args.cache_dir)
    bgc_ids, frozen_bgc_embeddings = _stack_cache(bgc_cache, max_points=args.max_points_per_modality)
    np_ids, frozen_np_embeddings = _stack_cache(np_cache, max_points=args.max_points_per_modality)
    bgc_class_bgc_labels, bgc_class_np_labels = _load_bgc_class_label_maps(args.data_dir)
    npclassifier_label_maps, npclassifier_allowed_labels, npclassifier_warnings = _load_npclassifier_label_maps(
        args.npclassifier_pair_labels_path
    )
    molecular_property_maps, molecular_property_warnings = _load_molecular_property_maps(args.molecular_property_values_path)
    n_bgcs_before_filter = len(bgc_ids)
    n_nps_before_filter = len(np_ids)
    if args.single_class_only:
        bgc_ids, frozen_bgc_embeddings = _filter_embeddings_by_single_class(
            bgc_ids,
            frozen_bgc_embeddings,
            bgc_class_bgc_labels,
        )
        np_ids, frozen_np_embeddings = _filter_embeddings_by_single_class(
            np_ids,
            frozen_np_embeddings,
            bgc_class_np_labels,
        )
        logger.info(
            "Single-class filter kept %d/%d BGCs and %d/%d NPs",
            len(bgc_ids),
            n_bgcs_before_filter,
            len(np_ids),
            n_nps_before_filter,
        )
    bgc_labels = _label_bgc_ids(bgc_ids, bgc_class_map=bgc_class_bgc_labels)
    np_parent_bgc_labels = _label_np_ids_by_parent_bgc_class(np_ids, np_class_map=bgc_class_np_labels)
    bgc_class_label_order = _label_order(bgc_class_bgc_labels, bgc_class_np_labels)
    pair_edges = _load_pair_edges(args.data_dir, bgc_ids, np_ids)

    manifest: dict[str, Any] = {
        "cache_dir": str(args.cache_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "data_dir": str(args.data_dir) if args.data_dir is not None else None,
        "splits_path": str(args.splits_path) if args.splits_path is not None else None,
        "npclassifier_pair_labels_path": str(args.npclassifier_pair_labels_path)
        if args.npclassifier_pair_labels_path is not None
        else None,
        "npclassifier_warnings": npclassifier_warnings,
        "molecular_property_values_path": str(args.molecular_property_values_path)
        if args.molecular_property_values_path is not None
        else None,
        "molecular_property_warnings": molecular_property_warnings,
        "cv_fold": args.cv_fold,
        "val_fold": args.val_fold,
        "methods": methods,
        "random_state": int(args.random_state),
        "normalization": args.normalization,
        "umap": {
            "n_neighbors": int(args.umap_n_neighbors),
            "min_dist": float(args.umap_min_dist),
            "metric": str(args.umap_metric),
        },
        "tsne": {
            "perplexity": float(args.tsne_perplexity),
            "learning_rate": args.tsne_learning_rate,
            "max_iter": int(args.tsne_max_iter),
            "metric": str(args.tsne_metric),
            "init": str(args.tsne_init),
        },
        "n_bgcs": len(bgc_ids),
        "n_nps": len(np_ids),
        "n_pair_edges": len(pair_edges),
        "pair_edge_linewidth": float(args.pair_edge_linewidth),
        "pair_edge_alpha": float(args.pair_edge_alpha),
        "write_tables": write_tables,
        "single_class_only": bool(args.single_class_only),
        "npclassifier_allowed_labels": npclassifier_allowed_labels,
        "molecular_property_names": MOLECULAR_PROPERTY_SPECS,
        "n_bgcs_before_single_class_filter": n_bgcs_before_filter,
        "n_nps_before_single_class_filter": n_nps_before_filter,
        "outputs": {},
    }

    if not args.skip_frozen:
        logger.info("Saving frozen embedding reductions for %d BGCs and %d NPs", len(bgc_ids), len(np_ids))
        manifest["outputs"]["frozen"] = {}
        if not args.only_molecular_properties:
            manifest["outputs"]["frozen"] = _save_single_set(
                bgc_embeddings=frozen_bgc_embeddings,
                np_embeddings=frozen_np_embeddings,
                bgc_ids=bgc_ids,
                np_ids=np_ids,
                bgc_labels=bgc_labels,
                np_labels=np_parent_bgc_labels,
                label_name="BGC class",
                label_order=bgc_class_label_order,
                space_key="frozen",
                space_name="frozen encoder space",
                outdir=outdir,
                methods=methods,
                reducer_kwargs=reducer_kwargs,
                write_tables=write_tables,
            )
            manifest["outputs"]["frozen"]["npclassifier_single_modality"] = _save_npclassifier_single_modality_sets(
                bgc_embeddings=frozen_bgc_embeddings,
                np_embeddings=frozen_np_embeddings,
                bgc_ids=bgc_ids,
                np_ids=np_ids,
                npclassifier_label_maps=npclassifier_label_maps,
                npclassifier_allowed_labels=npclassifier_allowed_labels,
                space_key="frozen",
                space_name="frozen encoder space",
                outdir=outdir,
                methods=methods,
                reducer_kwargs=reducer_kwargs,
                write_tables=write_tables,
            )
        manifest["outputs"]["frozen"]["molecular_property_np"] = _save_molecular_property_np_sets(
            np_embeddings=frozen_np_embeddings,
            np_ids=np_ids,
            molecular_property_maps=molecular_property_maps,
            space_key="frozen",
            space_name="frozen encoder space",
            outdir=outdir,
            methods=methods,
            reducer_kwargs=reducer_kwargs,
            write_tables=write_tables,
        )
        for warning in molecular_property_warnings:
            logger.warning("%s; skipping molecular-property-colored frozen NP plots.", warning)

    if not args.skip_clip:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --skip_clip is set")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Projecting all embeddings through BGC2NP-CLIP on %s", device)
        model = _load_model(args.checkpoint, device=device)
        clip_bgc_embeddings = _project_cache(model, bgc_cache, bgc_ids, modality="bgc", device=device)
        clip_np_embeddings = _project_cache(model, np_cache, np_ids, modality="np", device=device)
        manifest["outputs"]["clip"] = {}
        if not args.only_molecular_properties:
            manifest["outputs"]["clip"] = _save_single_set(
                bgc_embeddings=clip_bgc_embeddings,
                np_embeddings=clip_np_embeddings,
                bgc_ids=bgc_ids,
                np_ids=np_ids,
                bgc_labels=bgc_labels,
                np_labels=np_parent_bgc_labels,
                label_name="Parent BGC class",
                label_order=bgc_class_label_order,
                space_key="bgc2np_clip",
                space_name="BGC2NP-CLIP space",
                outdir=outdir,
                methods=methods,
                reducer_kwargs=reducer_kwargs,
                write_tables=write_tables,
            )
            manifest["outputs"]["clip"]["npclassifier_single_modality"] = _save_npclassifier_single_modality_sets(
                bgc_embeddings=clip_bgc_embeddings,
                np_embeddings=clip_np_embeddings,
                bgc_ids=bgc_ids,
                np_ids=np_ids,
                npclassifier_label_maps=npclassifier_label_maps,
                npclassifier_allowed_labels=npclassifier_allowed_labels,
                space_key="bgc2np_clip",
                space_name="BGC2NP-CLIP space",
                outdir=outdir,
                methods=methods,
                reducer_kwargs=reducer_kwargs,
                write_tables=write_tables,
            )
        manifest["outputs"]["clip"]["molecular_property_np"] = _save_molecular_property_np_sets(
            np_embeddings=clip_np_embeddings,
            np_ids=np_ids,
            molecular_property_maps=molecular_property_maps,
            space_key="bgc2np_clip",
            space_name="BGC2NP-CLIP space",
            outdir=outdir,
            methods=methods,
            reducer_kwargs=reducer_kwargs,
            write_tables=write_tables,
        )
        for warning in molecular_property_warnings:
            logger.warning("%s; skipping molecular-property-colored CLIP NP plots.", warning)

        if not args.skip_joint and not args.only_molecular_properties:
            joint_outputs: dict[str, Any] = {
                "by_modality": {},
                "by_modality_with_pairs": {},
                "by_bgc_class": {},
                "by_bgc_class_with_pairs": {},
            }
            for level in NPCLASSIFIER_LEVELS:
                joint_outputs[f"by_npclassifier_{level}"] = {}
                joint_outputs[f"by_npclassifier_{level}_with_pairs"] = {}
            bgc_class_joint_labels = _joint_labels(
                bgc_ids,
                np_ids,
                bgc_label_map=bgc_class_bgc_labels,
                np_label_map=bgc_class_np_labels,
            )
            for method in methods:
                joint_outputs["by_modality"][method] = save_joint_modality_reduction(
                    clip_bgc_embeddings,
                    clip_np_embeddings,
                    bgc_ids,
                    np_ids,
                    embedding_space_name="BGC2NP-CLIP space",
                    method=method,
                    outdir=outdir / "bgc2np_clip",
                    prefix="joint_bgc_np",
                    write_tables=write_tables,
                    **reducer_kwargs,
                )
                if pair_edges and not args.skip_pair_edges:
                    joint_outputs["by_modality_with_pairs"][method] = save_joint_modality_reduction(
                        clip_bgc_embeddings,
                        clip_np_embeddings,
                        bgc_ids,
                        np_ids,
                        embedding_space_name="BGC2NP-CLIP space",
                        method=method,
                        outdir=outdir / "bgc2np_clip",
                        prefix="joint_bgc_np_with_pairs",
                        pair_edges=pair_edges,
                        edge_linewidth=float(args.pair_edge_linewidth),
                        edge_alpha=float(args.pair_edge_alpha),
                        write_tables=write_tables,
                        **reducer_kwargs,
                    )
                if bgc_class_joint_labels is not None:
                    joint_outputs["by_bgc_class"][method] = save_joint_modality_reduction(
                        clip_bgc_embeddings,
                        clip_np_embeddings,
                        bgc_ids,
                        np_ids,
                        embedding_space_name="BGC2NP-CLIP space",
                        method=method,
                        outdir=outdir / "bgc2np_clip",
                        prefix="joint_bgc_np_by_bgc_class",
                        color_labels=bgc_class_joint_labels,
                        color_label_name="BGC class",
                        write_tables=write_tables,
                        **reducer_kwargs,
                    )
                    if pair_edges and not args.skip_pair_edges:
                        joint_outputs["by_bgc_class_with_pairs"][method] = save_joint_modality_reduction(
                            clip_bgc_embeddings,
                            clip_np_embeddings,
                            bgc_ids,
                            np_ids,
                            embedding_space_name="BGC2NP-CLIP space",
                            method=method,
                            outdir=outdir / "bgc2np_clip",
                            prefix="joint_bgc_np_by_bgc_class_with_pairs",
                            color_labels=bgc_class_joint_labels,
                            color_label_name="BGC class",
                            pair_edges=pair_edges,
                            edge_linewidth=float(args.pair_edge_linewidth),
                            edge_alpha=float(args.pair_edge_alpha),
                            write_tables=write_tables,
                            **reducer_kwargs,
                        )
                for level, label_maps in npclassifier_label_maps.items():
                    np_bgc_label_map, np_label_map = label_maps
                    level_bgc_ids, level_bgc_embeddings = _filter_embeddings_by_single_class(
                        bgc_ids,
                        clip_bgc_embeddings,
                        np_bgc_label_map,
                    )
                    level_np_ids, level_np_embeddings = _filter_embeddings_by_single_class(
                        np_ids,
                        clip_np_embeddings,
                        np_label_map,
                    )
                    labels = _joint_labels(
                        level_bgc_ids,
                        level_np_ids,
                        bgc_label_map=np_bgc_label_map,
                        np_label_map=np_label_map,
                    )
                    if labels is None or len(level_bgc_ids) + len(level_np_ids) < 2:
                        continue
                    level_pair_edges = _filter_pair_edges(pair_edges, level_bgc_ids, level_np_ids)
                    display_level = level.capitalize()
                    joint_outputs[f"by_npclassifier_{level}"][method] = save_joint_modality_reduction(
                        level_bgc_embeddings,
                        level_np_embeddings,
                        level_bgc_ids,
                        level_np_ids,
                        embedding_space_name="BGC2NP-CLIP space",
                        method=method,
                        outdir=outdir / "bgc2np_clip",
                        prefix=f"joint_bgc_np_by_npclassifier_{level}",
                        color_labels=labels,
                        color_label_name=f"NPClassifier {display_level}",
                        write_tables=write_tables,
                        **reducer_kwargs,
                    )
                    if level_pair_edges and not args.skip_pair_edges:
                        joint_outputs[f"by_npclassifier_{level}_with_pairs"][method] = save_joint_modality_reduction(
                            level_bgc_embeddings,
                            level_np_embeddings,
                            level_bgc_ids,
                            level_np_ids,
                            embedding_space_name="BGC2NP-CLIP space",
                            method=method,
                            outdir=outdir / "bgc2np_clip",
                            prefix=f"joint_bgc_np_by_npclassifier_{level}_with_pairs",
                            color_labels=labels,
                            color_label_name=f"NPClassifier {display_level}",
                            pair_edges=level_pair_edges,
                            edge_linewidth=float(args.pair_edge_linewidth),
                            edge_alpha=float(args.pair_edge_alpha),
                            write_tables=write_tables,
                            **reducer_kwargs,
                        )
            for warning in npclassifier_warnings:
                logger.warning("%s; skipping NPClassifier-colored joint plots.", warning)
            manifest["outputs"]["clip"]["figures"]["joint"] = joint_outputs

    manifest_path = outdir / "embedding_reductions_manifest.json"
    if args.only_molecular_properties and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        previous_manifest.update({key: value for key, value in manifest.items() if key != "outputs"})
        previous_outputs = previous_manifest.setdefault("outputs", {})
        for space_key, space_payload in manifest["outputs"].items():
            previous_outputs.setdefault(space_key, {}).update(space_payload)
        manifest = previous_manifest

    _save_json(manifest, manifest_path)
    logger.info("Saved embedding reduction manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
