from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._bootstrap import ensure_src_path

ensure_src_path()

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create downstream target-value distribution plots.")
    parser.add_argument("--pairs_path", type=Path, default=Path("data/MIBIG/processed/mibig_pairs.tsv"))
    parser.add_argument(
        "--npclassifier_labels_path",
        type=Path,
        default=Path("data/MIBIG/processed/mibig_npclassifier_labels.tsv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/downstream_distributions"))
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--top_n_npclassifier", type=int, default=30)
    return parser.parse_args()


def _require_rdkit() -> tuple[Any, Any, Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RDKit is required to compute molecular property distributions.") from exc
    return Chem, Crippen, Descriptors, rdMolDescriptors


def _descriptor_row(smiles: str, modules: tuple[Any, Any, Any, Any]) -> dict[str, Any] | None:
    Chem, Crippen, Descriptors, rdMolDescriptors = modules
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return {
        "canonical_smiles": canonical_smiles,
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
    }


def build_molecular_property_table(pairs_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs = pd.read_csv(pairs_path, sep="\t")
    if "smiles" not in pairs.columns:
        raise ValueError(f"{pairs_path} must contain a smiles column.")

    modules = _require_rdkit()
    records_by_smiles: dict[str, dict[str, Any]] = {}
    invalid = 0
    for smiles in pairs["smiles"].dropna().astype(str).tolist():
        row = _descriptor_row(smiles, modules)
        if row is None:
            invalid += 1
            continue
        records_by_smiles.setdefault(row["canonical_smiles"], row)

    table = pd.DataFrame(sorted(records_by_smiles.values(), key=lambda item: item["canonical_smiles"]))
    stats = {
        "source": str(pairs_path),
        "n_pair_rows": int(len(pairs)),
        "n_unique_valid_compounds": int(len(table)),
        "n_invalid_smiles_rows": int(invalid),
        "properties": {
            name: {
                "min": float(table[name].min()) if not table.empty else 0.0,
                "median": float(table[name].median()) if not table.empty else 0.0,
                "mean": float(table[name].mean()) if not table.empty else 0.0,
                "max": float(table[name].max()) if not table.empty else 0.0,
            }
            for name in ("molecular_weight", "logp", "tpsa")
        },
    }
    return table, stats


def _plot_single_distribution(values: pd.Series, output_path: Path, *, title: str, xlabel: str, bins: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(values.to_numpy(dtype=np.float64), bins=int(bins), color="#4C78A8", edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Compound count")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_molecular_property_distributions(table: pd.DataFrame, outdir: Path, bins: int) -> dict[str, str]:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("molecular_weight", "Molecular weight distribution", "Molecular weight", "molecular_weight_distribution.png"),
        ("logp", "logP distribution", "logP", "logp_distribution.png"),
        ("tpsa", "TPSA distribution", "TPSA", "tpsa_distribution.png"),
    ]
    paths: dict[str, str] = {}
    for column, title, xlabel, filename in specs:
        path = outdir / filename
        _plot_single_distribution(table[column], path, title=title, xlabel=xlabel, bins=bins)
        paths[column] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for ax, (column, title, xlabel, _) in zip(axes, specs, strict=True):
        ax.hist(table[column].to_numpy(dtype=np.float64), bins=int(bins), color="#4C78A8", edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Compound count")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()
    combined_path = outdir / "molecular_property_distributions.png"
    fig.savefig(combined_path, dpi=220)
    plt.close(fig)
    paths["combined"] = str(combined_path)
    return paths


def _split_labels(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [label.strip() for label in str(value).split(";") if label.strip()]


def build_npclassifier_count_tables(labels_path: Path) -> dict[str, pd.DataFrame]:
    labels = pd.read_csv(labels_path, sep="\t")
    specs = {
        "class": "npclassifier_class",
        "superclass": "npclassifier_superclass",
        "pathway": "npclassifier_pathway",
    }
    tables: dict[str, pd.DataFrame] = {}
    for level, column in specs.items():
        if column not in labels.columns:
            raise ValueError(f"{labels_path} is missing required column {column}.")
        counts: dict[str, int] = {}
        for value in labels[column].tolist():
            for label in _split_labels(value):
                counts[label] = counts.get(label, 0) + 1
        table = pd.DataFrame(
            [{"label": label, "n_compounds": int(count)} for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
        )
        tables[level] = table
    return tables


def plot_npclassifier_label_distribution(table: pd.DataFrame, output_path: Path, *, title: str, top_n: int) -> None:
    import matplotlib.pyplot as plt

    plot_df = table.head(int(top_n)).iloc[::-1].copy()
    fig_height = max(4.8, 0.34 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    ax.barh(plot_df["label"], plot_df["n_compounds"], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Compound count")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)
    for idx, value in enumerate(plot_df["n_compounds"].tolist()):
        ax.text(value, idx, str(int(value)), ha="left", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_npclassifier_distributions(labels_path: Path, outdir: Path, top_n: int) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    tables = build_npclassifier_count_tables(labels_path)
    paths: dict[str, Any] = {"tables": {}, "plots": {}}
    title_by_level = {
        "class": "NPClassifier class distribution",
        "superclass": "NPClassifier superclass distribution",
        "pathway": "NPClassifier pathway distribution",
    }
    for level, table in tables.items():
        table_path = outdir / f"npclassifier_{level}_counts.csv"
        table.to_csv(table_path, index=False)
        plot_path = outdir / f"npclassifier_{level}_distribution.png"
        plot_npclassifier_label_distribution(table, plot_path, title=title_by_level[level], top_n=top_n)
        paths["tables"][level] = str(table_path)
        paths["plots"][level] = str(plot_path)
    return paths


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    table, stats = build_molecular_property_table(args.pairs_path)
    table_path = args.outdir / "molecular_property_values.csv"
    table.to_csv(table_path, index=False)
    plot_paths = plot_molecular_property_distributions(table, args.outdir, bins=int(args.bins))
    npclassifier_paths: dict[str, Any] = {}
    if args.npclassifier_labels_path.exists():
        npclassifier_paths = save_npclassifier_distributions(
            args.npclassifier_labels_path,
            args.outdir,
            top_n=int(args.top_n_npclassifier),
        )
    manifest = {
        "table": str(table_path),
        "plots": plot_paths,
        "npclassifier": npclassifier_paths,
        "stats": stats,
    }
    (args.outdir / "downstream_distribution_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
