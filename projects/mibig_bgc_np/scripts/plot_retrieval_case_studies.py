from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, MACCSkeys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.scripts.eval_retrieval import _load_model
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings

RDLogger.DisableLog("rdApp.*")


def _maccs(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.bool_)


def _tanimoto(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(candidates, query).sum(axis=1)
    union = np.logical_or(candidates, query).sum(axis=1)
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)


def _reduce(features: np.ndarray, method: str, seed: int) -> np.ndarray:
    n = len(features)
    if features.ndim != 2 or n < 3:
        raise ValueError(f"Expected [n_candidates, n_features] with n>=3, got {features.shape}")
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(features)
    if method == "umap":
        return umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        ).fit_transform(features)
    if method == "tsne":
        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
        return TSNE(
            n_components=2,
            metric="cosine",
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=seed,
            max_iter=1500,
        ).fit_transform(features)
    raise ValueError(method)


def _draw_manifold(ax, xy, scores, true_idx, top10, title, norm):
    points = ax.scatter(
        xy[:, 0], xy[:, 1], c=scores, cmap="viridis", norm=norm,
        s=24, alpha=0.78, linewidths=0,
    )
    ax.scatter(
        xy[top10, 0], xy[top10, 1], facecolors="none", edgecolors="red",
        s=76, linewidths=1.25, label="CLIP top-10",
    )
    ax.scatter(
        xy[true_idx, 0], xy[true_idx, 1], marker="*", c="red", edgecolors="black",
        s=220, linewidths=0.7, zorder=5, label="True NP",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    return points


def _save_structure_grid(path: Path, mols: list[Chem.Mol], ids: list[str], similarities: np.ndarray, true_id: str) -> None:
    legends = []
    for rank, (compound_id, similarity) in enumerate(zip(ids, similarities), start=1):
        true_label = "  [TRUE]" if compound_id == true_id else ""
        legends.append(f"#{rank}{true_label}\nTanimoto={similarity:.3f}")
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(320, 260),
        legends=legends,
        useSVG=False,
    )
    image.save(path)


def _render_case(case: dict, outdir: Path, seed: int) -> dict:
    case_dir = outdir / f"{case['case_key']}_{case['bgc_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    features = case["molformer"]
    embeddings = {method: _reduce(features, method, seed) for method in ("pca", "umap", "tsne")}
    scores = case["scores"]
    norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()))

    manifold_path = case_dir / "molformer_pca_umap_tsne.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    last = None
    for ax, method in zip(axes, ("pca", "umap", "tsne")):
        last = _draw_manifold(
            ax, embeddings[method], scores, case["true_idx"], case["top10_indices"],
            f"{method.upper()} — MolFormer", norm,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.colorbar(last, ax=axes.tolist(), label="CLIP score", fraction=0.018, pad=0.02)
    fig.subplots_adjust(top=0.85, wspace=0.12)
    fig.savefig(manifold_path, dpi=300, bbox_inches="tight")
    fig.savefig(manifold_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    tanimoto_path = case_dir / "top10_rank_vs_tanimoto.png"
    ranks = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(ranks, case["top10_tanimoto"], marker="o", linewidth=2, color="#3567a8")
    ax.fill_between(ranks, 0, case["top10_tanimoto"], color="#3567a8", alpha=0.12)
    ax.set(xlabel="CLIP retrieval rank", ylabel="MACCS Tanimoto to true NP", xticks=ranks, ylim=(0, 1.03))
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"Top-10 structural similarity (max={case['max_tanimoto_at_10']:.3f})")
    fig.tight_layout()
    fig.savefig(tanimoto_path, dpi=300, bbox_inches="tight")
    fig.savefig(tanimoto_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    structures_path = case_dir / "top10_structures.png"
    _save_structure_grid(
        structures_path,
        case["top10_mols"],
        case["top10_compound_ids"],
        case["top10_tanimoto"],
        case["true_compound_id"],
    )

    composite_path = case_dir / "case_study_composite.png"
    manifold_img = mpimg.imread(manifold_path)
    tanimoto_img = mpimg.imread(tanimoto_path)
    structures_img = mpimg.imread(structures_path)
    fig = plt.figure(figsize=(17, 15))
    grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 0.62, 1.15), hspace=0.08)
    for row, image in enumerate((manifold_img, tanimoto_img, structures_img)):
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(image)
        ax.axis("off")
    fig.suptitle(
        f"{case['case_title']} — {case['bgc_id']} (strict fold {case['fold']})\n"
        f"true rank={case['true_rank']}, max MACCS Tanimoto@10={case['max_tanimoto_at_10']:.3f}",
        fontsize=17,
        y=0.995,
    )
    fig.savefig(composite_path, dpi=220, bbox_inches="tight")
    fig.savefig(composite_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        key: value for key, value in case.items()
        if key not in {"molformer", "scores", "top10_indices", "top10_mols", "top10_tanimoto", "true_idx"}
    }
    metadata["top10_tanimoto"] = [float(value) for value in case["top10_tanimoto"]]
    metadata["selection"] = {
        "top1_success": "single-answer query with true NP at rank 1; deterministic first after sorting",
        "high_similarity_miss": "highest max MACCS Tanimoto@10 among single-answer queries with true rank > 10",
        "failure": "lowest max MACCS Tanimoto@10 among single-answer queries with true rank > 10",
    }[case["case_key"]]
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"case_dir": str(case_dir), "composite": str(composite_path), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Three strict-CV retrieval case studies")
    parser.add_argument("--run_root", type=Path, default=Path("results/best_esm_domains_molformer_strict_cv10"))
    parser.add_argument("--outdir", type=Path, default=Path("results/paper_plots/retrieval_case_studies"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.run_root / "summary.json").read_text(encoding="utf-8"))
    cache_dir = Path(summary["cache_dir"])
    bgc_cache = torch.load(cache_dir / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(cache_dir / "compound_features.pt", map_location="cpu")
    candidates: list[dict] = []

    for fold in range(1, 11):
        fold_dir = args.run_root / f"fold_{fold}"
        fold_summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        interactions = build_interactions(
            summary["data_dir"], splits_path=summary["splits_path"], cv_fold=fold,
            val_fold=fold_summary.get("val_fold"),
        )
        model, _ = _load_model(fold_dir / "contrastive_model_best.pt", torch.device("cpu"))
        bgc_index, compound_index, z_bgc, z_compound, pairs = build_unique_embeddings(
            model, interactions, "test", bgc_cache, compound_cache, torch.device("cpu"),
        )
        bgcs, compounds = list(bgc_index), list(compound_index)
        truth = {bgc: [] for bgc in bgcs}
        for bgc_i, compound_i in pairs:
            truth[bgcs[bgc_i]].append(compounds[compound_i])
        score_matrix = (model.get_logit_scale().detach().cpu() * (z_bgc @ z_compound.T)).numpy()
        test = interactions[interactions.split.astype(str).str.lower() == "test"]
        smiles_by_id = (
            test[["compound_id", "smiles"]].drop_duplicates("compound_id")
            .set_index("compound_id").smiles.astype(str).to_dict()
        )
        mols = []
        for compound_id in compounds:
            mol = Chem.MolFromSmiles(smiles_by_id.get(compound_id, compound_id))
            if mol is None:
                raise ValueError(f"Invalid SMILES for {compound_id}")
            mols.append(mol)
        fingerprints = np.stack([_maccs(mol) for mol in mols])
        molformer = torch.stack([compound_cache[compound_id].float() for compound_id in compounds]).numpy()

        for bgc in bgcs:
            true_ids = sorted(set(truth[bgc]))
            if len(true_ids) != 1:
                continue
            true_id = true_ids[0]
            true_idx = compound_index[true_id]
            scores = score_matrix[bgc_index[bgc]]
            ordering = np.argsort(-scores)
            true_rank = int(np.flatnonzero(ordering == true_idx)[0]) + 1
            top10 = ordering[:10]
            top10_tanimoto = _tanimoto(fingerprints[true_idx], fingerprints[top10])
            candidates.append({
                "fold": fold,
                "bgc_id": bgc,
                "true_compound_id": true_id,
                "true_idx": true_idx,
                "true_rank": true_rank,
                "n_candidates": len(compounds),
                "max_tanimoto_at_10": float(top10_tanimoto.max()),
                "top10_tanimoto": top10_tanimoto,
                "top10_indices": top10,
                "top10_compound_ids": [compounds[index] for index in top10],
                "top10_mols": [mols[index] for index in top10],
                "scores": scores,
                "molformer": molformer,
            })

    successes = sorted((item for item in candidates if item["true_rank"] == 1), key=lambda x: (x["fold"], x["bgc_id"]))
    misses = [item for item in candidates if item["true_rank"] > 10]
    if not successes or len(misses) < 2:
        raise RuntimeError(f"Insufficient cases: top1={len(successes)}, misses={len(misses)}")
    selected = [
        ("top1_success", "Case 1: exact top-1 success", successes[0]),
        ("high_similarity_miss", "Case 2: structurally close retrieval miss", max(misses, key=lambda x: x["max_tanimoto_at_10"])),
        ("failure", "Case 3: retrieval failure", min(misses, key=lambda x: x["max_tanimoto_at_10"])),
    ]
    reports = []
    for case_key, case_title, case in selected:
        case["case_key"] = case_key
        case["case_title"] = case_title
        reports.append(_render_case(case, args.outdir, args.seed))
    report = {
        "run_root": str(args.run_root),
        "seed": args.seed,
        "n_single_answer_queries": len(candidates),
        "n_top1_successes": len(successes),
        "n_rank_gt_10_misses": len(misses),
        "cases": reports,
    }
    (args.outdir / "case_studies_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
