from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.scripts.eval_retrieval import _load_model
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
RDLogger.DisableLog("rdApp.*")


DEFAULT_RUNS = (
    ("BGC", Path("results/ohe_bgc_cv10_val_selected")),
    ("NP", Path("results/ohe_np_cv10_val_selected")),
    ("Combined", Path("results/ohe_combined_cv10_val_selected")),
    ("Strict", Path("results/ohe_strict_cv10_val_selected")),
)


@dataclass(frozen=True)
class SplitRun:
    label: str
    root: Path
    data_dir: Path
    cache_dir: Path
    splits_path: Path
    n_folds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot BGC-to-NP Top-k Tanimoto metrics across CV split strategies: "
            "max Tanimoto among top-k retrieved NPs, fraction of top-k NPs above a Tanimoto "
            "threshold, and whether top-k contains at least one threshold-matching NP."
        )
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/paper_plots"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=("png", "pdf", "svg"))
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument(
        "--fingerprint", choices=("maccs", "morgan"), default="maccs",
        help="Fingerprint used only for Tanimoto diagnostics (default: MACCS keys).",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--tanimoto_threshold", type=float, default=0.8)
    return parser.parse_args()


def _load_run(label: str, root: Path) -> SplitRun:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return SplitRun(
        label=label,
        root=root,
        data_dir=Path(summary["data_dir"]),
        cache_dir=Path(summary["cache_dir"]),
        splits_path=Path(summary["splits_path"]),
        n_folds=int(summary.get("n_folds", 10)),
    )


def _fingerprint(smiles: str, *, fingerprint: str, radius: int, n_bits: int) -> Any | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    if fingerprint == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _fingerprints_by_compound(
    interactions: Any,
    compound_ids: list[str],
    *,
    fingerprint: str,
    radius: int,
    n_bits: int,
) -> dict[str, Any]:
    smiles_by_compound = (
        interactions[["compound_id", "smiles"]]
        .drop_duplicates(subset=["compound_id"])
        .set_index("compound_id")["smiles"]
        .astype(str)
        .to_dict()
    )
    fingerprints: dict[str, Any] = {}
    for compound_id in compound_ids:
        smiles = smiles_by_compound.get(str(compound_id), str(compound_id))
        fp = _fingerprint(smiles, fingerprint=fingerprint, radius=radius, n_bits=n_bits)
        if fp is not None:
            fingerprints[str(compound_id)] = fp
    return fingerprints


def _true_compounds_by_bgc(interactions: Any, split: str) -> dict[str, list[str]]:
    split_df = interactions[interactions["split"].astype(str).str.lower() == split.lower()]
    grouped = split_df.groupby("bgc_id")["compound_id"].apply(lambda values: sorted(set(map(str, values))))
    return {str(bgc_id): compound_ids for bgc_id, compound_ids in grouped.items()}


def _topk_tanimoto_for_fold(
    run: SplitRun,
    fold_id: int,
    *,
    device: torch.device,
    bgc_cache: dict[str, torch.Tensor],
    compound_cache: dict[str, torch.Tensor],
    fingerprint: str,
    radius: int,
    n_bits: int,
    top_k: int,
    tanimoto_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    fold_dir = run.root / f"fold_{fold_id}"
    checkpoint_path = fold_dir / "contrastive_model_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model, _cfg = _load_model(checkpoint_path, device)
    fold_summary_path = fold_dir / "fold_summary.json"
    if fold_summary_path.exists():
        fold_summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
        val_fold = fold_summary.get("val_fold")
    else:
        val_fold = fold_id + 1 if fold_id < run.n_folds else 1
    interactions = build_interactions(
        run.data_dir,
        splits_path=run.splits_path,
        cv_fold=fold_id,
        val_fold=int(val_fold) if val_fold is not None else None,
    )
    bgc_index, compound_index, bgc_embs, compound_embs, _pairs = build_unique_embeddings(
        model=model,
        interactions=interactions,
        split="test",
        bgc_cache=bgc_cache,
        compound_cache=compound_cache,
        device=device,
    )

    bgc_ids = list(bgc_index.keys())
    compound_ids = list(compound_index.keys())
    true_by_bgc = _true_compounds_by_bgc(interactions, "test")
    fingerprints = _fingerprints_by_compound(
        interactions, compound_ids, fingerprint=fingerprint, radius=radius, n_bits=n_bits
    )
    label_col = "bgc_classes" if "bgc_classes" in interactions.columns else "bgc_class"
    class_by_bgc = (
        interactions[["bgc_id", label_col]].drop_duplicates("bgc_id").set_index("bgc_id")[label_col].to_dict()
        if label_col in interactions.columns else {}
    )

    sim = model.get_logit_scale().detach().cpu() * (bgc_embs @ compound_embs.t())
    k = min(int(top_k), len(compound_ids))
    top_indices = torch.topk(sim, k=k, dim=1).indices.cpu().numpy().tolist()

    records: list[dict[str, Any]] = []
    skipped = {"missing_candidate_fp": 0, "missing_true_fp": 0, "no_true_compounds": 0, "no_scored_candidates": 0}
    for row_idx, candidate_indices in enumerate(top_indices):
        bgc_id = str(bgc_ids[row_idx])
        true_compound_ids = true_by_bgc.get(bgc_id, [])
        if not true_compound_ids:
            skipped["no_true_compounds"] += 1
            continue
        true_fps = [fingerprints[compound_id] for compound_id in true_compound_ids if compound_id in fingerprints]
        if not true_fps:
            skipped["missing_true_fp"] += 1
            continue

        candidate_ids: list[str] = []
        candidate_best_tanimotos: list[float] = []
        for candidate_idx in candidate_indices:
            candidate_id = str(compound_ids[int(candidate_idx)])
            candidate_fp = fingerprints.get(candidate_id)
            if candidate_fp is None:
                skipped["missing_candidate_fp"] += 1
                continue
            similarities = [float(DataStructs.TanimotoSimilarity(candidate_fp, true_fp)) for true_fp in true_fps]
            candidate_ids.append(candidate_id)
            candidate_best_tanimotos.append(float(max(similarities)))
        if not candidate_best_tanimotos:
            skipped["no_scored_candidates"] += 1
            continue

        threshold_count = sum(value > float(tanimoto_threshold) for value in candidate_best_tanimotos)
        threshold_fraction = threshold_count / float(top_k)
        records.append(
            {
                "split": run.label,
                "fold_id": int(fold_id),
                "bgc_id": bgc_id,
                "bgc_classes": str(class_by_bgc.get(bgc_id, "")),
                "fingerprint_type": fingerprint,
                "topk_candidate_ids": ";".join(candidate_ids),
                "topk_candidate_best_tanimotos": ";".join(f"{value:.6g}" for value in candidate_best_tanimotos),
                "true_compound_ids": ";".join(true_compound_ids),
                "n_true_compounds": int(len(true_compound_ids)),
                "top_k": int(k),
                "tanimoto_threshold": float(tanimoto_threshold),
                "topk_max_tanimoto": float(max(candidate_best_tanimotos)),
                "topk_n_tanimoto_gt_threshold": int(threshold_count),
                "topk_fraction_tanimoto_gt_threshold": float(threshold_fraction),
                "topk_any_tanimoto_gt_threshold": int(threshold_count > 0),
            }
        )
    del model, bgc_embs, compound_embs, sim
    return records, skipped


def _compute_all(
    runs: list[SplitRun],
    *,
    device: torch.device,
    fingerprint: str,
    radius: int,
    n_bits: int,
    top_k: int,
    tanimoto_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    all_records: list[dict[str, Any]] = []
    skipped_by_split: dict[str, dict[str, int]] = {}
    for run in runs:
        print(f"Loading caches for {run.label} from {run.cache_dir}", flush=True)
        bgc_cache = torch.load(run.cache_dir / "bgc_features.pt", map_location="cpu")
        compound_cache = torch.load(run.cache_dir / "compound_features.pt", map_location="cpu")
        skipped_total = {
            "missing_candidate_fp": 0,
            "missing_true_fp": 0,
            "no_true_compounds": 0,
            "no_scored_candidates": 0,
        }
        for fold_id in range(1, run.n_folds + 1):
            records, skipped = _topk_tanimoto_for_fold(
                run,
                fold_id,
                device=device,
                bgc_cache=bgc_cache,
                compound_cache=compound_cache,
                fingerprint=fingerprint,
                radius=radius,
                n_bits=n_bits,
                top_k=top_k,
                tanimoto_threshold=tanimoto_threshold,
            )
            all_records.extend(records)
            for key, value in skipped.items():
                skipped_total[key] += int(value)
            print(f"{run.label} fold {fold_id}: {len(records)} scored BGCs", flush=True)
        skipped_by_split[run.label] = skipped_total
    return all_records, skipped_by_split


def _write_distribution(records: list[dict[str, Any]], outdir: Path, fingerprint: str) -> Path:
    path = outdir / f"top10_tanimoto_distribution_{fingerprint}.csv"
    columns = [
        "split",
        "fold_id",
        "bgc_id",
        "bgc_classes",
        "fingerprint_type",
        "topk_candidate_ids",
        "topk_candidate_best_tanimotos",
        "true_compound_ids",
        "n_true_compounds",
        "top_k",
        "tanimoto_threshold",
        "topk_max_tanimoto",
        "topk_n_tanimoto_gt_threshold",
        "topk_fraction_tanimoto_gt_threshold",
        "topk_any_tanimoto_gt_threshold",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return path


def _write_summary(
    records: list[dict[str, Any]], skipped: dict[str, dict[str, int]], outdir: Path, fingerprint: str
) -> Path:
    path = outdir / f"top10_tanimoto_summary_{fingerprint}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "fingerprint_type",
                "topk_max_tanimoto_mean",
                "topk_max_tanimoto_std",
                "topk_max_tanimoto_median",
                "topk_fraction_tanimoto_gt_threshold_mean",
                "topk_fraction_tanimoto_gt_threshold_std",
                "topk_fraction_tanimoto_gt_threshold_median",
                "topk_any_tanimoto_gt_threshold_mean",
                "topk_any_tanimoto_gt_threshold_std",
                "topk_any_tanimoto_gt_threshold_median",
                "topk_n_tanimoto_gt_threshold_mean",
                "topk_n_tanimoto_gt_threshold_std",
                "topk_n_tanimoto_gt_threshold_median",
                "n",
                "missing_candidate_fp",
                "missing_true_fp",
                "no_true_compounds",
                "no_scored_candidates",
            ]
        )
        for split_name, _root in DEFAULT_RUNS:
            topk_max = np.asarray(
                [float(record["topk_max_tanimoto"]) for record in records if record["split"] == split_name],
                dtype=np.float64,
            )
            topk_count = np.asarray(
                [float(record["topk_n_tanimoto_gt_threshold"]) for record in records if record["split"] == split_name],
                dtype=np.float64,
            )
            topk_fraction = np.asarray(
                [float(record["topk_fraction_tanimoto_gt_threshold"]) for record in records if record["split"] == split_name],
                dtype=np.float64,
            )
            topk_any = np.asarray(
                [float(record["topk_any_tanimoto_gt_threshold"]) for record in records if record["split"] == split_name],
                dtype=np.float64,
            )
            skip = skipped.get(split_name, {})
            writer.writerow(
                [
                    split_name,
                    fingerprint,
                    float(topk_max.mean()) if topk_max.size else np.nan,
                    float(topk_max.std(ddof=0)) if topk_max.size else np.nan,
                    float(np.median(topk_max)) if topk_max.size else np.nan,
                    float(topk_fraction.mean()) if topk_fraction.size else np.nan,
                    float(topk_fraction.std(ddof=0)) if topk_fraction.size else np.nan,
                    float(np.median(topk_fraction)) if topk_fraction.size else np.nan,
                    float(topk_any.mean()) if topk_any.size else np.nan,
                    float(topk_any.std(ddof=0)) if topk_any.size else np.nan,
                    float(np.median(topk_any)) if topk_any.size else np.nan,
                    float(topk_count.mean()) if topk_count.size else np.nan,
                    float(topk_count.std(ddof=0)) if topk_count.size else np.nan,
                    float(np.median(topk_count)) if topk_count.size else np.nan,
                    int(topk_max.size),
                    int(skip.get("missing_candidate_fp", 0)),
                    int(skip.get("missing_true_fp", 0)),
                    int(skip.get("no_true_compounds", 0)),
                    int(skip.get("no_scored_candidates", 0)),
                ]
            )
    return path


def _parse_classes(value: Any) -> list[str]:
    raw = str(value).strip().strip("[]")
    return [item.strip().strip("'\"") for item in raw.replace(",", ";").split(";") if item.strip().strip("'\"")]


def _write_per_class_summary(records: list[dict[str, Any]], outdir: Path, fingerprint: str) -> Path:
    path = outdir / f"top10_tanimoto_per_class_summary_{fingerprint}.csv"
    expanded: list[dict[str, Any]] = []
    for record in records:
        for class_name in _parse_classes(record.get("bgc_classes", "")):
            expanded.append({**record, "bgc_class": class_name})
    columns = [
        "split", "bgc_class", "fingerprint_type", "n",
        "topk_max_tanimoto_mean", "topk_max_tanimoto_std",
        "topk_fraction_tanimoto_gt_threshold_mean", "topk_any_tanimoto_gt_threshold_mean",
    ]
    if not expanded:
        import pandas as pd
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return path
    import pandas as pd
    frame = pd.DataFrame(expanded)
    rows = []
    for (split_name, class_name), group in frame.groupby(["split", "bgc_class"], sort=True):
        rows.append({
            "split": split_name, "bgc_class": class_name, "fingerprint_type": fingerprint,
            "n": int(len(group)),
            "topk_max_tanimoto_mean": float(group["topk_max_tanimoto"].mean()),
            "topk_max_tanimoto_std": float(group["topk_max_tanimoto"].std(ddof=0)),
            "topk_fraction_tanimoto_gt_threshold_mean": float(group["topk_fraction_tanimoto_gt_threshold"].mean()),
            "topk_any_tanimoto_gt_threshold_mean": float(group["topk_any_tanimoto_gt_threshold"].mean()),
        })
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.6, 4.2),
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#d7d7d7",
            "grid.linewidth": 0.8,
            "grid.linestyle": ":",
            "axes.axisbelow": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_metric(
    records: list[dict[str, Any]],
    outdir: Path,
    formats: list[str],
    dpi: int,
    *,
    metric: str,
    ylabel: str,
    filename_stem: str,
    ylim: tuple[float, float] | None = None,
) -> list[Path]:
    split_names = [name for name, _root in DEFAULT_RUNS]
    values_by_split = [
        np.asarray(
            [float(record[metric]) for record in records if record["split"] == split_name],
            dtype=np.float64,
        )
        for split_name in split_names
    ]
    means = [float(values.mean()) if values.size else np.nan for values in values_by_split]
    stds = [float(values.std(ddof=0)) if values.size else 0.0 for values in values_by_split]
    colors = plt.get_cmap("Set2").colors[: len(split_names)]

    rng = np.random.default_rng(42)
    x = np.arange(len(split_names), dtype=float)
    fig, ax = plt.subplots()
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=2.5,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    for idx, values in enumerate(values_by_split):
        if not values.size:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=int(values.size))
        ax.scatter(
            np.full(int(values.size), x[idx]) + jitter,
            values,
            s=7,
            color="#2f2f2f",
            alpha=0.14,
            linewidths=0,
        )

    ax.set_title("BGC to NP")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
    fig.tight_layout()

    paths: list[Path] = []
    for fmt in formats:
        path = outdir / f"{filename_stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_metric_distribution(
    records: list[dict[str, Any]],
    outdir: Path,
    formats: list[str],
    dpi: int,
    *,
    metric: str,
    xlabel: str,
    filename_stem: str,
    bins: np.ndarray,
    xlim: tuple[float, float] = (0.0, 1.0),
) -> list[Path]:
    split_names = [name for name, _root in DEFAULT_RUNS]
    colors = plt.get_cmap("Set2").colors[: len(split_names)]
    fig, axes = plt.subplots(1, len(split_names), figsize=(11.5, 3.2), sharex=True, sharey=True)
    if len(split_names) == 1:
        axes = [axes]

    for ax, split_name, color in zip(axes, split_names, colors, strict=False):
        values = np.asarray(
            [float(record[metric]) for record in records if record["split"] == split_name],
            dtype=np.float64,
        )
        if values.size:
            ax.hist(values, bins=bins, density=True, color=color, alpha=0.78, edgecolor="white", linewidth=0.4)
            ax.axvline(float(values.mean()), color="#222222", linewidth=1.4, linestyle="--", label="Mean")
            ax.text(
                0.04,
                0.94,
                f"mean = {values.mean():.3f}\nn = {values.size}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
        ax.set_title(split_name)
        ax.set_xlim(*xlim)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

    axes[0].set_ylabel("Density")
    for ax in axes:
        ax.set_xlabel(xlabel)
    fig.suptitle("Per-query BGC-to-NP retrieval distribution", y=1.03)
    fig.tight_layout()

    paths: list[Path] = []
    for fmt in formats:
        path = outdir / f"{filename_stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _plot(
    records: list[dict[str, Any]], outdir: Path, formats: list[str], dpi: int, top_k: int, fingerprint: str
) -> list[Path]:
    paths = _plot_metric(
        records,
        outdir,
        formats,
        dpi,
        metric="topk_max_tanimoto",
        ylabel=f"Top-{top_k} max Tanimoto",
        filename_stem=f"retrieval_bgc_to_np_top{top_k}_max_tanimoto_{fingerprint}",
        ylim=(0.0, 1.02),
    )
    paths.extend(
        _plot_metric(
            records,
            outdir,
            formats,
            dpi,
            metric="topk_fraction_tanimoto_gt_threshold",
            ylabel=f"Fraction of top-{top_k} NPs with Tanimoto > 0.8",
            filename_stem=f"retrieval_bgc_to_np_top{top_k}_fraction_tanimoto_gt_0p8_{fingerprint}",
            ylim=(-0.02, 1.02),
        )
    )
    paths.extend(
        _plot_metric(
            records,
            outdir,
            formats,
            dpi,
            metric="topk_any_tanimoto_gt_threshold",
            ylabel=f"Queries with any top-{top_k} NP Tanimoto > 0.8",
            filename_stem=f"retrieval_bgc_to_np_top{top_k}_any_tanimoto_gt_0p8_{fingerprint}",
            ylim=(-0.02, 1.02),
        )
    )
    paths.extend(
        _plot_metric_distribution(
            records,
            outdir,
            formats,
            dpi,
            metric="topk_max_tanimoto",
            xlabel=f"Top-{top_k} max Tanimoto",
            filename_stem=f"retrieval_bgc_to_np_top{top_k}_max_tanimoto_distribution_{fingerprint}",
            bins=np.linspace(0.0, 1.0, 21),
        )
    )
    paths.extend(
        _plot_metric_distribution(
            records,
            outdir,
            formats,
            dpi,
            metric="topk_fraction_tanimoto_gt_threshold",
            xlabel=f"Fraction of top-{top_k} NPs with Tanimoto > 0.8",
            filename_stem=f"retrieval_bgc_to_np_top{top_k}_fraction_tanimoto_gt_0p8_distribution_{fingerprint}",
            bins=np.linspace(0.0, 1.0, top_k + 2),
        )
    )
    return paths


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    runs = [_load_run(label, root) for label, root in DEFAULT_RUNS]

    records, skipped = _compute_all(
        runs,
        device=device,
        fingerprint=str(args.fingerprint),
        radius=int(args.radius),
        n_bits=int(args.n_bits),
        top_k=int(args.top_k),
        tanimoto_threshold=float(args.tanimoto_threshold),
    )
    distribution_path = _write_distribution(records, args.outdir, str(args.fingerprint))
    summary_path = _write_summary(records, skipped, args.outdir, str(args.fingerprint))
    per_class_path = _write_per_class_summary(records, args.outdir, str(args.fingerprint))
    _set_style()
    plot_paths = _plot(
        records, args.outdir, list(args.formats), int(args.dpi), int(args.top_k), str(args.fingerprint)
    )

    print(f"Wrote distribution CSV: {distribution_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote per-class summary CSV: {per_class_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
