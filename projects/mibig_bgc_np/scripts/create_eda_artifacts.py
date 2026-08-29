from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._bootstrap import ensure_src_path

ensure_src_path()

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MIBiG EDA tables and plots.")
    parser.add_argument("--json_dir", type=Path, default=Path("data/MIBIG/mibig_json_4.0"))
    parser.add_argument("--proteins_path", type=Path, default=Path("data/MIBIG/processed/bgc_proteins.jsonl"))
    parser.add_argument("--outdir", type=Path, default=Path("results/EDA"))
    parser.add_argument("--top_bioactivities", type=int, default=12)
    parser.add_argument("--top_intersections", type=int, default=15)
    parser.add_argument("--bins", type=int, default=60)
    return parser.parse_args()


def _activity_label(raw_name: Any) -> str:
    if isinstance(raw_name, dict):
        raw_name = raw_name.get("activity") or raw_name.get("name") or raw_name.get("label") or str(raw_name)
    label = str(raw_name or "").strip()
    return label or "unknown"


def extract_observed_bioactivities(json_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    compound_rows: list[dict[str, Any]] = []
    bgc_to_activities: dict[str, set[str]] = {}
    total_compounds = 0
    compounds_with_any_bioactivity = 0
    compounds_with_observed_bioactivity = 0

    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        bgc_id = str(data.get("accession") or path.stem)
        bgc_to_activities.setdefault(bgc_id, set())
        for compound_idx, compound in enumerate(data.get("compounds") or []):
            if not isinstance(compound, dict):
                continue
            total_compounds += 1
            bioactivities = compound.get("bioactivities") or []
            if bioactivities:
                compounds_with_any_bioactivity += 1
            observed_labels = sorted(
                {
                    _activity_label(item.get("name"))
                    for item in bioactivities
                    if isinstance(item, dict) and bool(item.get("observed")) is True
                }
            )
            if observed_labels:
                compounds_with_observed_bioactivity += 1
                bgc_to_activities[bgc_id].update(observed_labels)
            compound_rows.append(
                {
                    "bgc_id": bgc_id,
                    "compound_idx": int(compound_idx),
                    "compound_name": compound.get("name"),
                    "n_bioactivities": int(len(bioactivities)),
                    "n_observed_bioactivities": int(len(observed_labels)),
                    "observed_bioactivities": ";".join(observed_labels),
                }
            )

    compound_table = pd.DataFrame(compound_rows)
    bgc_table = pd.DataFrame(
        [
            {
                "bgc_id": bgc_id,
                "n_observed_bioactivities": int(len(labels)),
                "observed_bioactivities": ";".join(sorted(labels)),
            }
            for bgc_id, labels in sorted(bgc_to_activities.items())
        ]
    )

    label_counts: Counter[str] = Counter()
    for labels in bgc_to_activities.values():
        label_counts.update(labels)
    activity_table = pd.DataFrame(
        [{"bioactivity": label, "n_bgcs": int(count)} for label, count in label_counts.most_common()]
    )
    stats = {
        "json_dir": str(json_dir),
        "n_json_files": int(len(list(json_dir.glob("*.json")))),
        "n_compound_entries": int(total_compounds),
        "n_compound_entries_with_any_bioactivity": int(compounds_with_any_bioactivity),
        "n_compound_entries_with_observed_bioactivity": int(compounds_with_observed_bioactivity),
        "n_bgcs": int(len(bgc_to_activities)),
        "n_bgcs_with_observed_bioactivity": int(sum(1 for labels in bgc_to_activities.values() if labels)),
        "n_observed_bioactivity_classes": int(len(label_counts)),
    }
    return compound_table, bgc_table, activity_table, stats


def build_bioactivity_intersections(
    bgc_table: pd.DataFrame,
    activity_table: pd.DataFrame,
    *,
    top_bioactivities: int,
    top_intersections: int,
) -> tuple[pd.DataFrame, list[str]]:
    top_labels = activity_table.head(int(top_bioactivities))["bioactivity"].astype(str).tolist()
    combo_counts: Counter[tuple[str, ...]] = Counter()
    for text in bgc_table["observed_bioactivities"].fillna("").astype(str).tolist():
        labels = sorted(label for label in text.split(";") if label in top_labels)
        if labels:
            combo_counts[tuple(labels)] += 1
    rows = []
    for combo, count in combo_counts.most_common(int(top_intersections)):
        row = {"combination": "&".join(combo), "n_bgcs": int(count)}
        for label in top_labels:
            row[label] = label in combo
        rows.append(row)
    return pd.DataFrame(rows), top_labels


def plot_bioactivity_upset(intersections: pd.DataFrame, activity_table: pd.DataFrame, labels: list[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if intersections.empty or not labels:
        return

    labels = [label for label in labels if label in activity_table["bioactivity"].astype(str).tolist()]
    labels = list(reversed(labels))
    n_cols = len(intersections)
    n_rows = len(labels)
    fig_width = max(10.0, 0.62 * n_cols + 4.2)
    fig_height = max(6.0, 0.36 * n_rows + 3.4)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.1, max(4.0, n_cols)], height_ratios=[2.4, max(3.5, n_rows)])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1])

    x = np.arange(n_cols)
    counts = intersections["n_bgcs"].to_numpy(dtype=np.int64)
    ax_top.bar(x, counts, color="#2B2528", width=0.58)
    ax_top.set_ylabel("BGC")
    ax_top.set_title("Observed bioactivity combinations")
    ax_top.set_xticks([])
    ax_top.grid(axis="y", color="#BFC2C4", linewidth=0.8)
    for idx, count in enumerate(counts):
        ax_top.text(idx, count, str(int(count)), ha="center", va="bottom", fontsize=9)

    activity_counts = activity_table.set_index("bioactivity")["n_bgcs"].to_dict()
    y = np.arange(n_rows)
    left_counts = [int(activity_counts.get(label, 0)) for label in labels]
    ax_left.barh(y, left_counts, color="#2B2528", height=0.72)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels)
    ax_left.invert_xaxis()
    ax_left.set_xlabel("BGC")
    ax_left.set_ylim(-0.5, n_rows - 0.5)
    for spine in ("top", "right", "left"):
        ax_left.spines[spine].set_visible(False)

    active_by_col: dict[int, list[int]] = {col_idx: [] for col_idx in range(n_cols)}
    for row_idx, label in enumerate(labels):
        if row_idx % 2 == 1:
            ax_matrix.axhspan(row_idx - 0.5, row_idx + 0.5, color="#EDEDED", zorder=0)
        for col_idx in range(n_cols):
            active = bool(intersections.iloc[col_idx].get(label, False))
            ax_matrix.scatter(
                col_idx,
                row_idx,
                s=125,
                color="#2B2528" if active else "#D0D2D3",
                zorder=3,
            )
            if active:
                active_by_col[col_idx].append(row_idx)
    for col_idx, active_rows in active_by_col.items():
        if len(active_rows) > 1:
            ax_matrix.plot(
                [col_idx, col_idx],
                [min(active_rows), max(active_rows)],
                color="#2B2528",
                linewidth=1.2,
                zorder=2,
            )
    ax_matrix.set_xlim(-0.5, n_cols - 0.5)
    ax_matrix.set_ylim(-0.5, n_rows - 0.5)
    ax_matrix.set_yticks(y)
    ax_matrix.set_yticklabels([])
    ax_matrix.set_xticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax_matrix.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_bioactivity_class_counts(activity_table: pd.DataFrame, output_path: Path, *, top_n: int) -> None:
    import matplotlib.pyplot as plt

    plot_df = activity_table.head(int(top_n)).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.35 * len(plot_df))))
    ax.barh(plot_df["bioactivity"], plot_df["n_bgcs"], color="#4C78A8")
    ax.set_title("Observed bioactivity class distribution")
    ax.set_xlabel("BGC count")
    ax.set_ylabel("Bioactivity")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    for idx, value in enumerate(plot_df["n_bgcs"].tolist()):
        ax.text(value, idx, str(int(value)), ha="left", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_protein_length_table(proteins_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    protein_rows: list[dict[str, Any]] = []
    bgc_rows: list[dict[str, Any]] = []
    with proteins_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            bgc_id = str(record.get("bgc_id"))
            seqs = [str(seq) for seq in record.get("protein_seqs") or [] if str(seq)]
            ids = [str(item) for item in record.get("protein_ids") or []]
            for idx, seq in enumerate(seqs):
                protein_rows.append(
                    {
                        "bgc_id": bgc_id,
                        "protein_id": ids[idx] if idx < len(ids) else "",
                        "protein_length": int(len(seq)),
                    }
                )
            bgc_rows.append(
                {
                    "bgc_id": bgc_id,
                    "n_proteins": int(len(seqs)),
                    "total_protein_length": int(sum(len(seq) for seq in seqs)),
                    "mean_protein_length": float(np.mean([len(seq) for seq in seqs])) if seqs else 0.0,
                }
            )
    protein_table = pd.DataFrame(protein_rows)
    bgc_table = pd.DataFrame(bgc_rows)
    stats = {
        "proteins_path": str(proteins_path),
        "n_bgcs": int(len(bgc_table)),
        "n_proteins": int(len(protein_table)),
        "protein_length_median": float(protein_table["protein_length"].median()) if not protein_table.empty else 0.0,
        "n_proteins_per_bgc_median": float(bgc_table["n_proteins"].median()) if not bgc_table.empty else 0.0,
    }
    return protein_table, bgc_table, stats


def _plot_hist(values: pd.Series, output_path: Path, *, title: str, xlabel: str, bins: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(values.to_numpy(dtype=np.float64), bins=int(bins), color="#4C78A8", edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    compound_bio, bgc_bio, activity_counts, bio_stats = extract_observed_bioactivities(args.json_dir)
    compound_bio.to_csv(args.outdir / "compound_observed_bioactivities.csv", index=False)
    bgc_bio.to_csv(args.outdir / "bgc_observed_bioactivities.csv", index=False)
    activity_counts.to_csv(args.outdir / "observed_bioactivity_class_counts.csv", index=False)
    intersections, labels = build_bioactivity_intersections(
        bgc_bio,
        activity_counts,
        top_bioactivities=int(args.top_bioactivities),
        top_intersections=int(args.top_intersections),
    )
    intersections.to_csv(args.outdir / "observed_bioactivity_intersections.csv", index=False)
    plot_bioactivity_upset(
        intersections,
        activity_counts,
        labels,
        args.outdir / "observed_bioactivity_upset.png",
    )
    plot_bioactivity_class_counts(
        activity_counts,
        args.outdir / "observed_bioactivity_class_counts.png",
        top_n=int(args.top_bioactivities),
    )

    protein_table, bgc_protein_table, protein_stats = build_protein_length_table(args.proteins_path)
    protein_table.to_csv(args.outdir / "protein_lengths.csv", index=False)
    bgc_protein_table.to_csv(args.outdir / "bgc_protein_length_summary.csv", index=False)
    _plot_hist(
        protein_table["protein_length"],
        args.outdir / "protein_length_distribution.png",
        title="Protein length distribution",
        xlabel="Protein length (aa)",
        bins=int(args.bins),
    )
    _plot_hist(
        bgc_protein_table["n_proteins"],
        args.outdir / "genes_per_bgc_distribution.png",
        title="Genes per BGC distribution",
        xlabel="Proteins per BGC",
        bins=int(args.bins),
    )

    manifest = {
        "bioactivity": bio_stats,
        "protein_lengths": protein_stats,
        "outputs": {
            "compound_observed_bioactivities": str(args.outdir / "compound_observed_bioactivities.csv"),
            "bgc_observed_bioactivities": str(args.outdir / "bgc_observed_bioactivities.csv"),
            "observed_bioactivity_class_counts": str(args.outdir / "observed_bioactivity_class_counts.csv"),
            "observed_bioactivity_intersections": str(args.outdir / "observed_bioactivity_intersections.csv"),
            "observed_bioactivity_upset": str(args.outdir / "observed_bioactivity_upset.png"),
            "protein_length_distribution": str(args.outdir / "protein_length_distribution.png"),
            "genes_per_bgc_distribution": str(args.outdir / "genes_per_bgc_distribution.png"),
        },
    }
    (args.outdir / "eda_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
