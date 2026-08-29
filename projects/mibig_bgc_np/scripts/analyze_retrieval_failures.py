from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from scipy.stats import spearmanr

from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.scripts.eval_retrieval import _load_model
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings

RDLogger.DisableLog("rdApp.*")

GROUP_COLORS = {
    "success": "#2a9d8f",
    "ordinary_miss": "#e9c46a",
    "intermediate_miss": "#f4a261",
    "hard_failure": "#d1495b",
    "top2_10": "#9aa0a6",
}
GROUP_LABELS = {
    "success": "Success (true rank = 1)",
    "ordinary_miss": "Ordinary miss (rank > 10, max Tanimoto@10 ≥ 0.8)",
    "intermediate_miss": "Intermediate miss (rank > 10, 0.5 ≤ max Tanimoto@10 < 0.8)",
    "hard_failure": "Hard failure (rank > 10, max Tanimoto@10 < 0.5)",
    "top2_10": "True rank 2–10",
}


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return np.asarray(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.bool_)


def _max_tanimoto(query: np.ndarray, references: np.ndarray) -> float:
    intersections = np.logical_and(references, query).sum(axis=1)
    unions = np.logical_or(references, query).sum(axis=1)
    similarities = np.divide(
        intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0,
    )
    return float(similarities.max())


def _group(true_rank: int, max_tanimoto_at_10: float) -> str:
    if true_rank == 1:
        return "success"
    if true_rank <= 10:
        return "top2_10"
    if max_tanimoto_at_10 >= 0.8:
        return "ordinary_miss"
    if max_tanimoto_at_10 < 0.5:
        return "hard_failure"
    return "intermediate_miss"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _distribution_plot(frame: pd.DataFrame, outdir: Path) -> None:
    groups = ("success", "ordinary_miss", "hard_failure")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(0, 1, 21)
    for group in groups:
        values = frame.loc[frame.failure_group == group, "max_maccs_to_train"].to_numpy()
        label = f"{GROUP_LABELS[group]} (n={len(values)})"
        axes[0].hist(values, bins=bins, density=True, histtype="step", linewidth=2.2,
                     color=GROUP_COLORS[group], label=label)
        ordered = np.sort(values)
        axes[1].step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post",
                     linewidth=2.2, color=GROUP_COLORS[group], label=label)
    axes[0].set(xlabel="Maximum MACCS Tanimoto to train NPs", ylabel="Density", title="Distribution")
    axes[1].set(xlabel="Maximum MACCS Tanimoto to train NPs", ylabel="ECDF", title="Cumulative distribution")
    for ax in axes:
        ax.set_xlim(0, 1.01)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, outdir / "np_train_similarity_failure_groups.png")


def _rank_scatter(frame: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    scatter = ax.scatter(
        frame.max_maccs_to_train, frame.true_rank,
        c=frame.max_tanimoto_at_10, cmap="viridis", vmin=0, vmax=1,
        s=19, alpha=0.65, linewidths=0,
    )
    hard = frame.failure_group == "hard_failure"
    ax.scatter(frame.loc[hard, "max_maccs_to_train"], frame.loc[hard, "true_rank"],
               facecolors="none", edgecolors=GROUP_COLORS["hard_failure"], s=45, linewidths=0.8,
               label="Hard failure")
    ax.set(xlabel="Maximum MACCS Tanimoto of true NP to train NPs", ylabel="True NP rank",
           title="Retrieval rank versus chemical novelty")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="Maximum MACCS Tanimoto@10")
    fig.tight_layout()
    _save(fig, outdir / "true_rank_vs_np_train_similarity.png")


def _failure_rate_plot(frame: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    edges = np.linspace(0, 1, 11)
    working = frame.copy()
    working["similarity_bin"] = pd.cut(working.max_maccs_to_train, bins=edges, include_lowest=True)
    rows = []
    for bin_index, (interval, part) in enumerate(working.groupby("similarity_bin", observed=False)):
        n = len(part)
        rows.append({
            "similarity_bin": str(interval),
            "similarity_bin_label": f"{edges[bin_index]:.1f}–{edges[bin_index + 1]:.1f}",
            "bin_midpoint": float(interval.mid),
            "n": n,
            "rank_gt_10_rate": float((part.true_rank > 10).mean()) if n else np.nan,
            "hard_failure_rate": float((part.failure_group == "hard_failure").mean()) if n else np.nan,
        })
    rates = pd.DataFrame(rows)
    plotted = rates[rates.n > 0].reset_index(drop=True)
    positions = np.arange(len(plotted))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(positions, plotted.rank_gt_10_rate, marker="o", linewidth=2,
            color="#457b9d", label="True rank > 10")
    ax.plot(positions, plotted.hard_failure_rate, marker="o", linewidth=2,
            color=GROUP_COLORS["hard_failure"], label="Hard failure")
    for position, row in zip(positions, plotted.itertuples()):
        alignment = "left" if position == 0 else "right" if position == positions[-1] else "center"
        ax.annotate(f"n={row.n} queries", (position, row.rank_gt_10_rate),
                    xytext=(0, 7), textcoords="offset points", ha=alignment, fontsize=7)
    ax.set_xticks(positions, plotted.similarity_bin_label)
    ax.set(xlabel="Maximum MACCS Tanimoto of true NP to train NPs (non-overlapping bins)",
           ylabel="Fraction of queries within similarity bin",
           title="Failure rate versus chemical novelty", ylim=(0, 1.05))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, outdir / "failure_rate_by_np_train_similarity.png")
    return rates


def _bimodal_plot(frame: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    order = ("top2_10", "success", "intermediate_miss", "ordinary_miss", "hard_failure")
    for group in order:
        part = frame[frame.failure_group == group]
        ax.scatter(part.max_maccs_to_train, part.max_bgc_cosine_to_train, s=24, alpha=0.68,
                   color=GROUP_COLORS[group], linewidths=0, label=f"{GROUP_LABELS[group]} (n={len(part)})")
    ax.set(xlabel="NP novelty: max MACCS Tanimoto to train NPs",
           ylabel="BGC novelty: max centered frozen-ESM cosine to train BGCs",
           title="Chemical and BGC novelty of strict-CV queries")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    _save(fig, outdir / "bimodal_train_similarity_by_failure_group.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose strict-CV BGC-to-NP retrieval failures")
    parser.add_argument("--run_root", type=Path, default=Path("results/best_esm_domains_molformer_strict_cv10"))
    parser.add_argument("--outdir", type=Path, default=Path("results/paper_plots/retrieval_failure_analysis"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.run_root / "summary.json").read_text(encoding="utf-8"))
    cache_dir = Path(summary["cache_dir"])
    bgc_cache = torch.load(cache_dir / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(cache_dir / "compound_features.pt", map_location="cpu")
    fingerprint_cache: dict[str, np.ndarray] = {}
    records = []

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
        train_bgc_index, _, z_train_bgc, _, _ = build_unique_embeddings(
            model, interactions, "train", bgc_cache, compound_cache, torch.device("cpu"),
        )
        truths = {bgc: [] for bgc in bgcs}
        for bgc_i, compound_i in pairs:
            truths[bgcs[bgc_i]].append(compounds[compound_i])
        score_matrix = (model.get_logit_scale().detach().cpu() * (z_bgc @ z_compound.T)).numpy()

        train = interactions[interactions.split.astype(str).str.lower() == "train"]
        train_compounds = sorted(train.compound_id.astype(str).unique())
        train_bgcs = sorted(train.bgc_id.astype(str).unique())
        for compound_id in set(train_compounds) | set(compounds):
            if compound_id not in fingerprint_cache:
                fingerprint_cache[compound_id] = _fingerprint(compound_id)
        train_fingerprints = np.stack([fingerprint_cache[item] for item in train_compounds])
        train_molformer = _normalise(np.stack([compound_cache[item].float().numpy() for item in train_compounds]))
        train_bgc_raw = np.stack([bgc_cache[item].float().numpy() for item in train_bgcs])
        bgc_train_mean = train_bgc_raw.mean(axis=0, keepdims=True)
        train_bgc_centered = _normalise(train_bgc_raw - bgc_train_mean)
        train_bgc_clip = z_train_bgc.numpy()

        for bgc in bgcs:
            true_ids = sorted(set(truths[bgc]))
            if len(true_ids) != 1:
                continue
            true_id = true_ids[0]
            true_idx = compound_index[true_id]
            scores = score_matrix[bgc_index[bgc]]
            ordering = np.argsort(-scores)
            true_rank = int(np.flatnonzero(ordering == true_idx)[0]) + 1
            top10_ids = [compounds[index] for index in ordering[:10]]
            top10_fingerprints = np.stack([fingerprint_cache[item] for item in top10_ids])
            max_at_10 = _max_tanimoto(fingerprint_cache[true_id], top10_fingerprints)
            true_molformer = _normalise(compound_cache[true_id].float().numpy()[None, :])[0]
            query_bgc_centered = _normalise(
                bgc_cache[bgc].float().numpy()[None, :] - bgc_train_mean
            )[0]
            query_bgc_clip = z_bgc[bgc_index[bgc]].numpy()
            records.append({
                "fold": fold,
                "bgc_id": bgc,
                "true_compound_id": true_id,
                "true_rank": true_rank,
                "n_candidates": len(compounds),
                "max_tanimoto_at_10": max_at_10,
                "failure_group": _group(true_rank, max_at_10),
                "max_maccs_to_train": _max_tanimoto(fingerprint_cache[true_id], train_fingerprints),
                "max_molformer_cosine_to_train": float((train_molformer @ true_molformer).max()),
                "max_bgc_cosine_to_train": float((train_bgc_centered @ query_bgc_centered).max()),
                "max_bgc_clip_cosine_to_train": float((train_bgc_clip @ query_bgc_clip).max()),
            })

    frame = pd.DataFrame(records).sort_values(["fold", "bgc_id"]).reset_index(drop=True)
    frame.to_csv(args.outdir / "strict_cv10_single_answer_failure_queries.csv", index=False)
    _distribution_plot(frame, args.outdir)
    _rank_scatter(frame, args.outdir)
    rates = _failure_rate_plot(frame, args.outdir)
    rates.to_csv(args.outdir / "failure_rate_by_np_train_similarity.csv", index=False)
    _bimodal_plot(frame, args.outdir)

    selected_groups = ("success", "ordinary_miss", "hard_failure")
    group_summary = (
        frame[frame.failure_group.isin(selected_groups)]
        .groupby("failure_group")
        .agg(
            n=("bgc_id", "size"),
            true_rank_median=("true_rank", "median"),
            max_tanimoto_at_10_median=("max_tanimoto_at_10", "median"),
            max_maccs_to_train_median=("max_maccs_to_train", "median"),
            max_molformer_cosine_to_train_median=("max_molformer_cosine_to_train", "median"),
            max_bgc_cosine_to_train_median=("max_bgc_cosine_to_train", "median"),
            max_bgc_clip_cosine_to_train_median=("max_bgc_clip_cosine_to_train", "median"),
        )
        .reset_index()
    )
    group_summary.to_csv(args.outdir / "failure_group_summary.csv", index=False)
    rank_corr = spearmanr(frame.max_maccs_to_train, frame.true_rank)
    report = {
        "definitions": {
            "success": "true_rank == 1",
            "ordinary_miss": "true_rank > 10 and max_tanimoto_at_10 >= 0.8",
            "intermediate_miss": "true_rank > 10 and 0.5 <= max_tanimoto_at_10 < 0.8",
            "hard_failure": "true_rank > 10 and max_tanimoto_at_10 < 0.5",
        },
        "n_queries": len(frame),
        "counts": frame.failure_group.value_counts().to_dict(),
        "spearman_max_maccs_to_train_vs_true_rank": {
            "rho": float(rank_corr.statistic), "pvalue": float(rank_corr.pvalue),
        },
        "group_summary": group_summary.to_dict(orient="records"),
    }
    (args.outdir / "analysis_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
