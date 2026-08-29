from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-combi-pair-alignment")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/combi-cache-pair-alignment")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.scripts.eval_retrieval import _load_model
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize matched BGC-NP pairs before and after CLIP training using real embeddings "
            "projected into the shared 256D space, then reduced to 2D."
        )
    )
    parser.add_argument("--run_root", type=Path, default=Path("results/final_results_t33/cv/strict_cv10"))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/paper_plots/final_results_t33/clip_alignment_before_after"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_pairs", type=int, default=28)
    parser.add_argument(
        "--selection",
        choices=("top_improved", "random"),
        default="top_improved",
        help=(
            "Which positive pairs to visualize. top_improved is poster-friendly and deterministic; "
            "random is less cherry-picked but often visually messier."
        ),
    )
    return parser.parse_args()


def _make_untrained_like_checkpoint(ckpt_path: Path, device: torch.device, seed: int) -> DualEncoderCLIP:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    torch.manual_seed(seed)
    model = DualEncoderCLIP(
        bgc_input_dim=int(ckpt["bgc_input_dim"]),
        compound_input_dim=int(ckpt["compound_input_dim"]),
        emb_dim=int(cfg["model"]["emb_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        init_temperature=float(cfg["model"]["init_temperature"]),
        max_logit_scale=float(cfg["model"]["max_logit_scale"]),
        bgc_aggregation=str(cfg["model"].get("bgc_aggregation", "prepooled")),
        bgc_aggregation_config=dict(cfg["model"].get("bgc_aggregation_config", {})),
        projection_head=str(cfg["model"].get("projection_head", "mlp_gelu")),
    ).to(device)
    model.eval()
    return model


def _embed(
    model: DualEncoderCLIP,
    interactions: pd.DataFrame,
    bgc_cache: dict[str, torch.Tensor],
    compound_cache: dict[str, torch.Tensor],
    device: torch.device,
):
    return build_unique_embeddings(
        model,
        interactions,
        "test",
        bgc_cache,
        compound_cache,
        device,
        batch_size=512,
    )


def _select_pairs(
    pairs: list[tuple[int, int]],
    z_bgc_before: torch.Tensor,
    z_cmp_before: torch.Tensor,
    z_bgc_after: torch.Tensor,
    z_cmp_after: torch.Tensor,
    n_pairs: int,
    seed: int,
    selection: str,
) -> tuple[list[int], pd.DataFrame]:
    before_sim = np.asarray([float((z_bgc_before[i] * z_cmp_before[j]).sum()) for i, j in pairs])
    after_sim = np.asarray([float((z_bgc_after[i] * z_cmp_after[j]).sum()) for i, j in pairs])
    improvement = after_sim - before_sim
    if selection == "top_improved":
        selected = np.argsort(-improvement)[:n_pairs]
    else:
        rng = np.random.default_rng(seed)
        selected = rng.choice(np.arange(len(pairs)), size=min(n_pairs, len(pairs)), replace=False)
    selected = [int(x) for x in selected]
    table = pd.DataFrame(
        {
            "pair_index": np.arange(len(pairs)),
            "bgc_index": [i for i, _ in pairs],
            "compound_index": [j for _, j in pairs],
            "before_cosine_similarity": before_sim,
            "after_cosine_similarity": after_sim,
            "before_cosine_distance": 1.0 - before_sim,
            "after_cosine_distance": 1.0 - after_sim,
            "similarity_improvement": improvement,
            "selected": [idx in set(selected) for idx in range(len(pairs))],
        }
    )
    return selected, table


def _reduce(points: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "PCA":
        return PCA(n_components=2, random_state=seed).fit_transform(points)
    if method == "UMAP":
        return umap.UMAP(
            n_components=2,
            n_neighbors=min(10, max(2, len(points) - 1)),
            min_dist=0.25,
            metric="cosine",
            random_state=seed,
        ).fit_transform(points)
    raise ValueError(method)


def _panel(
    ax: plt.Axes,
    bgc_xy: np.ndarray,
    cmp_xy: np.ndarray,
    *,
    title: str,
    median_distance: float,
) -> None:
    for idx in range(len(bgc_xy)):
        ax.plot(
            [bgc_xy[idx, 0], cmp_xy[idx, 0]],
            [bgc_xy[idx, 1], cmp_xy[idx, 1]],
            color="#8A8A8A",
            alpha=0.42,
            lw=1.05,
            zorder=1,
        )
    ax.scatter(
        bgc_xy[:, 0],
        bgc_xy[:, 1],
        s=50,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
        label="BGC",
        zorder=3,
    )
    ax.scatter(
        cmp_xy[:, 0],
        cmp_xy[:, 1],
        s=62,
        marker="*",
        color="#F58518",
        edgecolor="white",
        linewidth=0.45,
        label="NP",
        zorder=4,
    )
    ax.set_title(title, fontsize=14, pad=7)
    ax.text(
        0.03,
        0.04,
        f"median cosine distance={median_distance:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.25"),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    summary = json.loads((args.run_root / "summary.json").read_text(encoding="utf-8"))
    fold_dir = args.run_root / f"fold_{args.fold}"
    fold_summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
    ckpt_path = fold_dir / "contrastive_model_best.pt"
    cache_dir = Path(summary["cache_dir"])
    bgc_cache = torch.load(cache_dir / "bgc_features.pt", map_location="cpu")
    compound_cache = torch.load(cache_dir / "compound_features.pt", map_location="cpu")
    interactions = build_interactions(
        summary["data_dir"],
        splits_path=summary["splits_path"],
        cv_fold=int(args.fold),
        val_fold=fold_summary.get("val_fold"),
    )

    trained_model, _ = _load_model(ckpt_path, device)
    untrained_model = _make_untrained_like_checkpoint(ckpt_path, device, seed=int(args.seed))
    bgc_index, compound_index, z_bgc_after, z_cmp_after, pairs = _embed(
        trained_model, interactions, bgc_cache, compound_cache, device
    )
    bgc_index_before, compound_index_before, z_bgc_before, z_cmp_before, pairs_before = _embed(
        untrained_model, interactions, bgc_cache, compound_cache, device
    )
    if bgc_index != bgc_index_before or compound_index != compound_index_before or pairs != pairs_before:
        raise RuntimeError("Untrained and trained embedding indices/pairs do not match.")

    selected_pair_indices, pair_table = _select_pairs(
        pairs,
        z_bgc_before,
        z_cmp_before,
        z_bgc_after,
        z_cmp_after,
        n_pairs=int(args.n_pairs),
        seed=int(args.seed),
        selection=str(args.selection),
    )
    pair_table.to_csv(args.outdir / "clip_pair_alignment_2d_all_pair_stats.csv", index=False)

    selected_pairs = [pairs[idx] for idx in selected_pair_indices]
    bgc_ids = list(bgc_index)
    cmp_ids = list(compound_index)
    selected_meta = []
    before_bgc = []
    before_cmp = []
    after_bgc = []
    after_cmp = []
    for selected_rank, pair_idx in enumerate(selected_pair_indices, start=1):
        bgc_i, cmp_i = pairs[pair_idx]
        before_bgc.append(z_bgc_before[bgc_i].numpy())
        before_cmp.append(z_cmp_before[cmp_i].numpy())
        after_bgc.append(z_bgc_after[bgc_i].numpy())
        after_cmp.append(z_cmp_after[cmp_i].numpy())
        row = pair_table.iloc[pair_idx].to_dict()
        selected_meta.append(
            {
                "selected_rank": int(selected_rank),
                "pair_index": int(pair_idx),
                "bgc_id": str(bgc_ids[bgc_i]),
                "compound_id": str(cmp_ids[cmp_i]),
                **{key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in row.items()},
            }
        )

    before_bgc_arr = np.vstack(before_bgc)
    before_cmp_arr = np.vstack(before_cmp)
    after_bgc_arr = np.vstack(after_bgc)
    after_cmp_arr = np.vstack(after_cmp)
    before_dist = 1.0 - np.sum(before_bgc_arr * before_cmp_arr, axis=1)
    after_dist = 1.0 - np.sum(after_bgc_arr * after_cmp_arr, axis=1)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), gridspec_kw={"wspace": 0.08, "hspace": 0.22})
    for row, method in enumerate(("PCA", "UMAP")):
        before_xy = _reduce(np.vstack([before_bgc_arr, before_cmp_arr]), method, seed=int(args.seed))
        after_xy = _reduce(np.vstack([after_bgc_arr, after_cmp_arr]), method, seed=int(args.seed))
        n = len(selected_pairs)
        _panel(
            axes[row, 0],
            before_xy[:n],
            before_xy[n:],
            title=f"{method}: untrained projections",
            median_distance=float(np.median(before_dist)),
        )
        _panel(
            axes[row, 1],
            after_xy[:n],
            after_xy[n:],
            title=f"{method}: trained BGC2NP-CLIP",
            median_distance=float(np.median(after_dist)),
        )
        axes[row, 0].text(-0.03, 0.5, method, transform=axes[row, 0].transAxes, rotation=90, ha="right", va="center", fontsize=15, fontweight="bold")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4C78A8", markeredgecolor="white", markersize=8, label="BGC"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#F58518", markeredgecolor="white", markersize=11, label="NP"),
        Line2D([0], [0], color="#8A8A8A", lw=1.4, label="Known BGC–NP pair"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    fig.suptitle(
        "Matched BGC–NP pairs move closer after CLIP training",
        fontsize=17,
        y=1.035,
    )
    fig.subplots_adjust(top=0.90, bottom=0.04, left=0.055, right=0.995)

    prefix = f"clip_pair_alignment_2d_{args.selection}_n{len(selected_pairs)}"
    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.outdir / f"{prefix}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "run_root": str(args.run_root),
        "fold": int(args.fold),
        "selection": str(args.selection),
        "n_pairs": int(len(selected_pairs)),
        "checkpoint": str(ckpt_path),
        "cache_dir": str(cache_dir),
        "median_cosine_distance": {
            "untrained": float(np.median(before_dist)),
            "trained": float(np.median(after_dist)),
        },
        "mean_cosine_distance": {
            "untrained": float(np.mean(before_dist)),
            "trained": float(np.mean(after_dist)),
        },
        "note": (
            "Real BGC/NP embeddings are first mapped into the shared 256D space using either untrained "
            "or trained projection heads. The selected subset is used for visual clarity; see CSV for all pairs."
        ),
        "selected_pairs": selected_meta,
    }
    (args.outdir / f"{prefix}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "selected_pairs"}, indent=2))
    print(f"Saved {args.outdir / (prefix + '.pdf')}")


if __name__ == "__main__":
    main()
