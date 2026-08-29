from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-combi-clip-alignment")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/combi-cache-clip-alignment")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.scripts.eval_retrieval import _load_model
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poster-ready comparison of true/mismatched BGC-NP cosine similarities "
            "before and after CLIP training. 'Before' uses the same randomly "
            "initialized projection-head architecture, making cross-modal comparison well-defined."
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
    parser.add_argument("--scatter_max_per_group", type=int, default=450)
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


def _embeddings_for_model(
    model: DualEncoderCLIP,
    interactions: pd.DataFrame,
    bgc_cache: dict[str, torch.Tensor],
    compound_cache: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, int], dict[str, int], torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    return build_unique_embeddings(
        model,
        interactions,
        "test",
        bgc_cache,
        compound_cache,
        device,
        batch_size=512,
    )


def _sample_negatives(
    pairs: list[tuple[int, int]],
    n_bgcs: int,
    n_compounds: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    positives_by_bgc: dict[int, set[int]] = {}
    positive_set = set(pairs)
    for bgc_i, compound_i in pairs:
        positives_by_bgc.setdefault(int(bgc_i), set()).add(int(compound_i))

    negatives: list[tuple[int, int]] = []
    compound_ids = np.arange(n_compounds)
    for bgc_i, _ in pairs:
        true_for_bgc = positives_by_bgc.get(int(bgc_i), set())
        allowed = compound_ids[[int(c) not in true_for_bgc for c in compound_ids]]
        if len(allowed) == 0:
            continue
        compound_i = int(rng.choice(allowed))
        # Extremely defensive: avoid any global positive pair if duplicates/multi-positive edge cases exist.
        tries = 0
        while (int(bgc_i), compound_i) in positive_set and tries < 100:
            compound_i = int(rng.choice(allowed))
            tries += 1
        if (int(bgc_i), compound_i) not in positive_set:
            negatives.append((int(bgc_i), compound_i))
    return negatives


def _similarities(
    z_bgc: torch.Tensor,
    z_compound: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    values = [float((z_bgc[bgc_i] * z_compound[compound_i]).sum()) for bgc_i, compound_i in pairs]
    return np.asarray(values, dtype=float)


def _make_long_table(
    before_pos: np.ndarray,
    before_neg: np.ndarray,
    after_pos: np.ndarray,
    after_neg: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for stage, pair_type, values in (
        ("Untrained projections", "Positive pairs", before_pos),
        ("Untrained projections", "Negative pairs", before_neg),
        ("Trained BGC2NP-CLIP", "Positive pairs", after_pos),
        ("Trained BGC2NP-CLIP", "Negative pairs", after_neg),
    ):
        rows.extend({"space": stage, "pair_type": pair_type, "cosine_similarity": float(value)} for value in values)
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, outdir: Path, seed: int, scatter_max_per_group: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    groups = [
        ("Untrained projections", "Positive pairs"),
        ("Untrained projections", "Negative pairs"),
        ("Trained BGC2NP-CLIP", "Positive pairs"),
        ("Trained BGC2NP-CLIP", "Negative pairs"),
    ]
    labels = ["Positive", "Negative", "Positive", "Negative"]
    colors = ["#3B7DDD", "#B8B8B8", "#3B7DDD", "#B8B8B8"]
    positions = [1.0, 1.8, 3.2, 4.0]
    arrays = [
        df[(df["space"] == space) & (df["pair_type"] == pair_type)]["cosine_similarity"].to_numpy()
        for space, pair_type in groups
    ]

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    parts = ax.violinplot(arrays, positions=positions, widths=0.62, showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.36)

    rng = np.random.default_rng(seed)
    for x, values, color in zip(positions, arrays, colors, strict=True):
        show = values
        if len(show) > scatter_max_per_group:
            show = rng.choice(show, size=scatter_max_per_group, replace=False)
        jitter = rng.normal(0, 0.055, size=len(show))
        ax.scatter(
            np.full(len(show), x) + jitter,
            show,
            s=11,
            alpha=0.32,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        mean = float(np.mean(values))
        median = float(np.median(values))
        ax.scatter([x], [mean], marker="D", s=64, color="black", zorder=5)
        ax.plot([x - 0.24, x + 0.24], [median, median], color="black", lw=2.0, zorder=4)

    before_delta = float(np.mean(arrays[0]) - np.mean(arrays[1]))
    after_delta = float(np.mean(arrays[2]) - np.mean(arrays[3]))
    ax.text(
        1.4,
        0.96,
        f"Δ mean={before_delta:.2f}",
        ha="center",
        va="top",
        fontsize=12,
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        3.6,
        0.96,
        f"Δ mean={after_delta:.2f}",
        ha="center",
        va="top",
        fontsize=12,
        transform=ax.get_xaxis_transform(),
    )

    ax.axvline(2.5, color="#D0D0D0", lw=1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Cosine similarity in shared 256-d space", fontsize=13)
    ax.set_title("CLIP training separates positive and negative BGC–NP pairs", fontsize=16, pad=12)
    ax.text(1.4, -0.095, "Before training", ha="center", va="top", fontsize=13, transform=ax.get_xaxis_transform())
    ax.text(3.6, -0.095, "After training", ha="center", va="top", fontsize=13, transform=ax.get_xaxis_transform())
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(outdir / "clip_alignment_before_after_violin.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "clip_alignment_before_after_violin.pdf", bbox_inches="tight")
    fig.savefig(outdir / "clip_alignment_before_after_violin.svg", bbox_inches="tight")
    plt.close(fig)


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

    bgc_index, compound_index, z_bgc_after, z_compound_after, positive_pairs = _embeddings_for_model(
        trained_model,
        interactions,
        bgc_cache,
        compound_cache,
        device,
    )
    bgc_index_before, compound_index_before, z_bgc_before, z_compound_before, positive_pairs_before = _embeddings_for_model(
        untrained_model,
        interactions,
        bgc_cache,
        compound_cache,
        device,
    )
    if bgc_index != bgc_index_before or compound_index != compound_index_before or positive_pairs != positive_pairs_before:
        raise RuntimeError("Untrained and trained embedding indices/pairs do not match.")

    negative_pairs = _sample_negatives(
        positive_pairs,
        n_bgcs=len(bgc_index),
        n_compounds=len(compound_index),
        seed=int(args.seed) + int(args.fold) * 1009,
    )
    before_pos = _similarities(z_bgc_before, z_compound_before, positive_pairs)
    before_neg = _similarities(z_bgc_before, z_compound_before, negative_pairs)
    after_pos = _similarities(z_bgc_after, z_compound_after, positive_pairs)
    after_neg = _similarities(z_bgc_after, z_compound_after, negative_pairs)
    df = _make_long_table(before_pos, before_neg, after_pos, after_neg)
    df.to_csv(args.outdir / "clip_alignment_before_after_pair_similarities.csv", index=False)
    stats = {
        "run_root": str(args.run_root),
        "fold": int(args.fold),
        "checkpoint": str(ckpt_path),
        "cache_dir": str(cache_dir),
        "n_test_bgcs": int(len(bgc_index)),
        "n_test_compounds": int(len(compound_index)),
        "n_positive_pairs": int(len(positive_pairs)),
        "n_negative_pairs": int(len(negative_pairs)),
        "means": {
            "untrained_positive": float(before_pos.mean()),
            "untrained_negative": float(before_neg.mean()),
            "trained_positive": float(after_pos.mean()),
            "trained_negative": float(after_neg.mean()),
        },
        "medians": {
            "untrained_positive": float(np.median(before_pos)),
            "untrained_negative": float(np.median(before_neg)),
            "trained_positive": float(np.median(after_pos)),
            "trained_negative": float(np.median(after_neg)),
        },
        "delta_mean_positive_minus_negative": {
            "untrained": float(before_pos.mean() - before_neg.mean()),
            "trained": float(after_pos.mean() - after_neg.mean()),
        },
        "note": (
            "Before-training space uses the same randomly initialized projection heads as the trained model. "
            "Raw ESM2 and MolFormer embeddings are not directly comparable because they have different dimensions. "
            "Positive pairs are known BGC-NP associations; negative pairs are randomly sampled non-associated "
            "BGC-NP pairs from the same test fold."
        ),
    }
    (args.outdir / "clip_alignment_before_after_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot(df, args.outdir, seed=int(args.seed), scatter_max_per_group=int(args.scatter_max_per_group))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
