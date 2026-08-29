from __future__ import annotations

import argparse
import ast
import csv
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FIXED_TEST_FILENAMES = {"MAP_metadata_fold.csv", "bgcmac_fold.csv"}
MISSING = "NA"
ENTITIES = ("BGC", "NP", "BGC_family", "NP_cluster")
ENTITY_LABELS = {
    "BGC": "BGC",
    "NP": "NP",
    "BGC_family": "BGC family",
    "NP_cluster": "NP cluster",
}
ENTITY_COLORS = {
    # Pastel/Set3-like qualitative colors.
    "BGC": "#8DD3C7",
    "NP": "#FDB462",
    "BGC_family": "#BEBADA",
    "NP_cluster": "#FB8072",
}
COMPARISON_SPLITS = {
    "BGC": "bgc_cv_seed42_n10.tsv",
    "NP": "np_cv_seed42_n10.tsv",
    "combined": "combined_cv_seed42_n10.tsv",
    "strict": "mibig_pairs_strict_cv10.tsv",
}


@dataclass(frozen=True)
class SplitRecord:
    bgc_id: str | None
    compound_id: str | None = None
    bgc_family: str | None = None
    np_cluster: str | None = None
    fold: int | None = None
    split: str | None = None
    is_product: bool | None = None
    is_test: bool | None = None


@dataclass(frozen=True)
class Evaluation:
    split_file: str
    evaluation_strategy: str
    heldout_label: str
    train: list[SplitRecord]
    heldout: list[SplitRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure BGC/NP leakage in MIBiG split files as train-heldout shared "
            "unique identifiers divided by heldout unique identifiers."
        )
    )
    parser.add_argument("--splits_dir", type=Path, default=Path("data/MIBIG/splits"))
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=Path("data/MIBIG/processed/mibig_pairs.tsv"),
        help="Pair table used to add NP IDs to BGC-only split files.",
    )
    parser.add_argument(
        "--strict_cv_path",
        type=Path,
        default=Path("data/MIBIG/processed/strict_splits/mibig_pairs_strict_cv10.tsv"),
        help="Strict pair-level CV split file to include in leakage summaries and comparison artifacts.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("results/split_leakage"))
    parser.add_argument(
        "--fixed_eval_folds",
        type=int,
        nargs="+",
        default=list(range(1, 10)),
        help="Held-out folds to evaluate for MAP/BGC-MAC fixed-test files. Fold 10 is excluded by default.",
    )
    parser.add_argument(
        "--map_train_positive_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For MAP_metadata_fold.csv, use only is_product==1 rows on the training side.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Write only CSV tables.",
    )
    return parser.parse_args()


def read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_bool(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def parse_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(str(value).strip()))


def clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def load_pairs(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    rows = read_table(path)
    mapping: dict[str, set[str]] = {}
    for row in rows:
        bgc_id = clean_text(row.get("bgc_id") or row.get("BGC_number"))
        compound_id = clean_text(
            row.get("compound_id")
            or row.get("canonical_smiles")
            or row.get("smiles")
            or row.get("product")
            or row.get("compound_name")
        )
        if bgc_id and compound_id:
            mapping.setdefault(bgc_id, set()).add(compound_id)
    return mapping


def load_strict_annotations(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.exists():
        return {}, {}
    rows = read_table(path)
    bgc_family_by_bgc: dict[str, str] = {}
    np_cluster_by_compound: dict[str, str] = {}
    for row in rows:
        bgc_id = clean_text(row.get("bgc_id"))
        bgc_family = clean_text(row.get("bgc_bigscape_family"))
        if bgc_id and bgc_family:
            bgc_family_by_bgc[bgc_id] = bgc_family

        np_cluster = clean_text(row.get("np_butina_cluster"))
        if not np_cluster:
            continue
        for column in ("compound_key", "smiles", "compound_id", "canonical_smiles"):
            compound = clean_text(row.get(column))
            if compound:
                np_cluster_by_compound[compound] = np_cluster
    return bgc_family_by_bgc, np_cluster_by_compound


def annotate_records(
    records: list[SplitRecord],
    bgc_family_by_bgc: dict[str, str],
    np_cluster_by_compound: dict[str, str],
) -> list[SplitRecord]:
    annotated: list[SplitRecord] = []
    for record in records:
        annotated.append(
            SplitRecord(
                bgc_id=record.bgc_id,
                compound_id=record.compound_id,
                bgc_family=record.bgc_family or (bgc_family_by_bgc.get(record.bgc_id) if record.bgc_id else None),
                np_cluster=record.np_cluster
                or (np_cluster_by_compound.get(record.compound_id) if record.compound_id else None),
                fold=record.fold,
                split=record.split,
                is_product=record.is_product,
                is_test=record.is_test,
            )
        )
    return annotated


def parse_bgcmac_compounds(value: object) -> set[str]:
    text = clean_text(value)
    if text is None:
        return set()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {text}
    compounds: set[str] = set()
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                compound = clean_text(item[1])
            else:
                compound = clean_text(item)
            if compound:
                compounds.add(compound)
    return compounds


def expand_bgc_only_records(
    base: SplitRecord,
    bgc_to_compounds: dict[str, set[str]],
) -> list[SplitRecord]:
    if not base.bgc_id:
        return []
    compounds = sorted(bgc_to_compounds.get(base.bgc_id, set()))
    if not compounds:
        return [base]
    return [
        SplitRecord(
            bgc_id=base.bgc_id,
            compound_id=compound,
            bgc_family=base.bgc_family,
            np_cluster=base.np_cluster,
            fold=base.fold,
            split=base.split,
            is_product=base.is_product,
            is_test=base.is_test,
        )
        for compound in compounds
    ]


def load_split_records(path: Path, bgc_to_compounds: dict[str, set[str]]) -> tuple[list[SplitRecord], str]:
    rows = read_table(path)
    if not rows:
        return [], "empty"
    columns = set(rows[0])
    records: list[SplitRecord] = []

    if {"BGC_number", "product", "fold"}.issubset(columns):
        for row in rows:
            records.append(
                SplitRecord(
                    bgc_id=clean_text(row.get("BGC_number")),
                    compound_id=clean_text(row.get("product")),
                    fold=parse_int(row.get("fold")),
                    is_product=parse_bool(row.get("is_product")),
                )
            )
        return records, "fixed_fold_csv"

    if {"BGC_number", "compounds", "fold"}.issubset(columns):
        for row in rows:
            compounds = parse_bgcmac_compounds(row.get("compounds"))
            base = SplitRecord(
                bgc_id=clean_text(row.get("BGC_number")),
                compound_id=None,
                fold=parse_int(row.get("fold")),
                is_test=parse_bool(row.get("is_test")),
            )
            if compounds:
                records.extend(
                    SplitRecord(
                        bgc_id=base.bgc_id,
                        compound_id=compound,
                        bgc_family=base.bgc_family,
                        np_cluster=base.np_cluster,
                        fold=base.fold,
                        is_test=base.is_test,
                    )
                    for compound in sorted(compounds)
                )
            else:
                records.append(base)
        return records, "fixed_fold_csv"

    if {"bgc_id", "compound_id", "fold_id"}.issubset(columns):
        for row in rows:
            records.append(
                SplitRecord(
                    bgc_id=clean_text(row.get("bgc_id")),
                    compound_id=clean_text(row.get("compound_id")),
                    fold=parse_int(row.get("fold_id")),
                )
            )
        return records, "normal_cv"

    if {"bgc_id", "strict_cv10_fold"}.issubset(columns):
        for row in rows:
            records.append(
                SplitRecord(
                    bgc_id=clean_text(row.get("bgc_id")),
                    compound_id=clean_text(
                        row.get("compound_key")
                        or row.get("compound_id")
                        or row.get("canonical_smiles")
                        or row.get("smiles")
                        or row.get("compound_name")
                    ),
                    bgc_family=clean_text(row.get("bgc_bigscape_family")),
                    np_cluster=clean_text(row.get("np_butina_cluster")),
                    fold=parse_int(row.get("strict_cv10_fold")),
                )
            )
        return records, "normal_cv"

    if {"bgc_id", "fold_id"}.issubset(columns):
        for row in rows:
            base = SplitRecord(bgc_id=clean_text(row.get("bgc_id")), fold=parse_int(row.get("fold_id")))
            records.extend(expand_bgc_only_records(base, bgc_to_compounds))
        return records, "normal_cv"

    if {"bgc_id", "split"}.issubset(columns):
        for row in rows:
            base = SplitRecord(bgc_id=clean_text(row.get("bgc_id")), split=clean_text(row.get("split")))
            records.extend(expand_bgc_only_records(base, bgc_to_compounds))
        return records, "random_train_val_test"

    return records, "unknown"


def ids(records: Iterable[SplitRecord], entity: str) -> set[str]:
    if entity == "BGC":
        return {record.bgc_id for record in records if record.bgc_id}
    if entity == "NP":
        return {record.compound_id for record in records if record.compound_id}
    if entity == "BGC_family":
        return {record.bgc_family for record in records if record.bgc_family}
    if entity == "NP_cluster":
        return {record.np_cluster for record in records if record.np_cluster}
    raise ValueError(f"Unknown entity: {entity}")


def entity_value(record: SplitRecord, entity: str) -> str | None:
    if entity == "BGC":
        return record.bgc_id
    if entity == "NP":
        return record.compound_id
    if entity == "BGC_family":
        return record.bgc_family
    if entity == "NP_cluster":
        return record.np_cluster
    raise ValueError(f"Unknown entity: {entity}")


def build_evaluations(
    split_file: str,
    records: list[SplitRecord],
    strategy: str,
    fixed_eval_folds: list[int],
    *,
    map_train_positive_only: bool,
) -> list[Evaluation]:
    if strategy == "normal_cv":
        folds = sorted({record.fold for record in records if record.fold is not None})
        return [
            Evaluation(
                split_file=split_file,
                evaluation_strategy=strategy,
                heldout_label=f"fold_{fold}",
                train=[record for record in records if record.fold != fold],
                heldout=[record for record in records if record.fold == fold],
            )
            for fold in folds
        ]

    if strategy == "fixed_fold_csv":
        fixed_fold_set = set(fixed_eval_folds)
        usable = [record for record in records if record.fold in fixed_fold_set]
        evaluations: list[Evaluation] = []
        for fold in sorted(fixed_fold_set):
            heldout = [record for record in usable if record.fold == fold]
            train = [record for record in usable if record.fold != fold]
            if split_file == "MAP_metadata_fold.csv" and map_train_positive_only:
                train = [record for record in train if record.is_product is True]
            if heldout:
                evaluations.append(
                    Evaluation(
                        split_file=split_file,
                        evaluation_strategy="fixed_test_validation_folds_1_9",
                        heldout_label=f"validation_fold_{fold}",
                        train=train,
                        heldout=heldout,
                    )
                )
        return evaluations

    if strategy == "random_train_val_test":
        train = [record for record in records if (record.split or "").lower() == "train"]
        heldout = [record for record in records if (record.split or "").lower() == "test"]
        if train and heldout:
            return [
                Evaluation(
                    split_file=split_file,
                    evaluation_strategy=strategy,
                    heldout_label="test",
                    train=train,
                    heldout=heldout,
                )
            ]
        return []

    return []


def leakage_rows(evaluations: list[Evaluation]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation in evaluations:
        for entity in ENTITIES:
            train_ids = ids(evaluation.train, entity)
            heldout_values = [entity_value(record, entity) for record in evaluation.heldout]
            heldout_values = [value for value in heldout_values if value]
            heldout_ids = set(heldout_values)
            shared = train_ids.intersection(heldout_ids)
            leaked_sample_count = sum(1 for value in heldout_values if value in train_ids)
            sample_denominator = len(heldout_values)
            unique_denominator = len(heldout_ids)
            leakage_fraction = (leaked_sample_count / sample_denominator) if sample_denominator else 0.0
            unique_leakage_fraction = (len(shared) / unique_denominator) if unique_denominator else 0.0
            rows.append(
                {
                    "split_file": evaluation.split_file,
                    "evaluation_strategy": evaluation.evaluation_strategy,
                    "heldout_label": evaluation.heldout_label,
                    "entity": entity,
                    "train_unique_count": len(train_ids),
                    "heldout_sample_count": sample_denominator,
                    "leaked_sample_count": leaked_sample_count,
                    "heldout_unique_count": unique_denominator,
                    "leaked_unique_count": len(shared),
                    "leakage_fraction": f"{leakage_fraction:.8f}",
                    "leakage_percent": f"{100.0 * leakage_fraction:.4f}",
                    "unique_leakage_fraction": f"{unique_leakage_fraction:.8f}",
                    "unique_leakage_percent": f"{100.0 * unique_leakage_fraction:.4f}",
                    "leaked_ids": ";".join(sorted(shared)) if shared else MISSING,
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["split_file"]), str(row["evaluation_strategy"]), str(row["entity"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for (split_file, strategy, entity), group in sorted(grouped.items()):
        n_evaluations = len(group)
        sum_heldout = sum(int(row["heldout_sample_count"]) for row in group)
        sum_leaked = sum(int(row["leaked_sample_count"]) for row in group)
        sum_heldout_unique = sum(int(row["heldout_unique_count"]) for row in group)
        sum_leaked_unique = sum(int(row["leaked_unique_count"]) for row in group)
        fractions = [float(row["leakage_fraction"]) for row in group]
        unique_fractions = [float(row["unique_leakage_fraction"]) for row in group]
        nonzero_evaluations = sum(1 for row in group if int(row["leaked_sample_count"]) > 0)
        aggregate_fraction = (sum_leaked / sum_heldout) if sum_heldout else 0.0
        unique_aggregate_fraction = (
            sum_leaked_unique / sum_heldout_unique if sum_heldout_unique else 0.0
        )
        out.append(
            {
                "split_file": split_file,
                "evaluation_strategy": strategy,
                "entity": entity,
                "n_evaluations": n_evaluations,
                "nonzero_evaluations": nonzero_evaluations,
                "sum_heldout_sample_count": sum_heldout,
                "sum_leaked_sample_count": sum_leaked,
                "aggregate_leakage_fraction": f"{aggregate_fraction:.8f}",
                "aggregate_leakage_percent": f"{100.0 * aggregate_fraction:.4f}",
                "mean_fold_leakage_fraction": f"{(sum(fractions) / n_evaluations) if n_evaluations else 0.0:.8f}",
                "mean_fold_leakage_percent": f"{(100.0 * sum(fractions) / n_evaluations) if n_evaluations else 0.0:.4f}",
                "sum_heldout_unique_count": sum_heldout_unique,
                "sum_leaked_unique_count": sum_leaked_unique,
                "unique_aggregate_leakage_fraction": f"{unique_aggregate_fraction:.8f}",
                "unique_aggregate_leakage_percent": f"{100.0 * unique_aggregate_fraction:.4f}",
                "unique_mean_fold_leakage_fraction": (
                    f"{(sum(unique_fractions) / n_evaluations) if n_evaluations else 0.0:.8f}"
                ),
                "unique_mean_fold_leakage_percent": (
                    f"{(100.0 * sum(unique_fractions) / n_evaluations) if n_evaluations else 0.0:.4f}"
                ),
            }
        )
    return out


def heldout_sort_key(label: object) -> tuple[int, str]:
    text = str(label)
    digits = "".join(character for character in text if character.isdigit())
    if digits:
        return int(digits), text
    return 0, text


def comparison_rows(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    split_name_by_file = {filename: split_name for split_name, filename in COMPARISON_SPLITS.items()}
    for row in detail_rows:
        split_name = split_name_by_file.get(str(row["split_file"]))
        if split_name is None or row["evaluation_strategy"] != "normal_cv":
            continue
        rows_by_key.setdefault((split_name, str(row["entity"])), []).append(row)

    out: list[dict[str, object]] = []
    for split_name in COMPARISON_SPLITS:
        for entity in ENTITIES:
            group = rows_by_key.get((split_name, entity), [])
            group = sorted(group, key=lambda row: heldout_sort_key(row["heldout_label"]))
            values = [float(row["leakage_percent"]) for row in group]
            mean_value = statistics.fmean(values) if values else 0.0
            std_value = statistics.stdev(values) if len(values) > 1 else 0.0
            out_row: dict[str, object] = {
                "split": split_name,
                "entity": entity,
                "mean_leakage_percent": f"{mean_value:.4f}",
                "std_leakage_percent": f"{std_value:.4f}",
            }
            for idx in range(10):
                out_row[f"fold_{idx + 1}"] = f"{values[idx]:.4f}" if idx < len(values) else ""
            out.append(out_row)
    return out


def save_cv_nonzero_barplot(rows: list[dict[str, object]], out_paths: list[Path]) -> str:
    plot_rows = [
        row
        for row in rows
        if row["evaluation_strategy"] == "normal_cv" and int(row["leaked_sample_count"]) > 0
    ]
    if not plot_rows:
        return "empty"

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "missing_matplotlib"

    labels = [
        f"{Path(str(row['split_file'])).stem}\n{row['heldout_label']} {row['entity']}"
        for row in plot_rows
    ]
    values = [float(row["leakage_percent"]) for row in plot_rows]
    colors = [ENTITY_COLORS.get(str(row["entity"]), "#BBBBBB") for row in plot_rows]

    fig_width = max(8.0, min(24.0, 0.38 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_ylabel("Leakage (% of held-out unique IDs)")
    ax.set_title("Nonzero CV Leakage by Fold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return "created"


def save_comparison_barplot(rows: list[dict[str, object]], out_paths: list[Path]) -> str:
    if not rows:
        return "empty"

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "missing_matplotlib"

    split_order = list(COMPARISON_SPLITS)
    entity_order = list(ENTITIES)
    by_key = {(str(row["split"]), str(row["entity"])): row for row in rows}
    width = 0.18
    x_positions = list(range(len(split_order)))

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    start_offset = -width * (len(entity_order) - 1) / 2
    for entity_idx, entity in enumerate(entity_order):
        offset = start_offset + width * entity_idx
        means = [float(by_key[(split, entity)]["mean_leakage_percent"]) for split in split_order]
        stds = [float(by_key[(split, entity)]["std_leakage_percent"]) for split in split_order]
        ax.bar(
            [x + offset for x in x_positions],
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=ENTITY_COLORS[entity],
            edgecolor="white",
            linewidth=0.7,
            label=f"{ENTITY_LABELS[entity]} leakage",
        )

    ax.set_ylabel("Leakage (% of held-out unique IDs)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_order)
    ax.set_title("Mean CV Leakage Across Folds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return "created"


def compact_comparison_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(str(row["split"]), str(row["entity"])): row for row in rows}
    compact_rows: list[dict[str, object]] = []
    for split_name in COMPARISON_SPLITS:
        entity_rows = {entity: by_key.get((split_name, entity), {}) for entity in ENTITIES}
        compact_rows.append(
            {
                "split": split_name,
                **{
                    f"{entity.lower()}_leakage_mean_percent": entity_rows[entity].get(
                        "mean_leakage_percent", "0.0000"
                    )
                    for entity in ENTITIES
                },
                **{
                    f"{entity.lower()}_leakage_std_percent": entity_rows[entity].get(
                        "std_leakage_percent", "0.0000"
                    )
                    for entity in ENTITIES
                },
                "summary": "; ".join(
                    (
                        f"{ENTITY_LABELS[entity]} "
                        f"{entity_rows[entity].get('mean_leakage_percent', '0.0000')}% +/- "
                        f"{entity_rows[entity].get('std_leakage_percent', '0.0000')}"
                    )
                    for entity in ENTITIES
                ),
            }
        )
    return compact_rows


def main() -> None:
    args = parse_args()
    if not args.splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {args.splits_dir}")

    bgc_to_compounds = load_pairs(args.pairs_path)
    bgc_family_by_bgc, np_cluster_by_compound = load_strict_annotations(args.strict_cv_path)
    detail_rows: list[dict[str, object]] = []
    split_files = sorted(path for path in args.splits_dir.iterdir() if path.suffix in {".csv", ".tsv"})
    if args.strict_cv_path.exists():
        split_files.append(args.strict_cv_path)
    for path in split_files:
        records, strategy = load_split_records(path, bgc_to_compounds)
        records = annotate_records(records, bgc_family_by_bgc, np_cluster_by_compound)
        if path.name in FIXED_TEST_FILENAMES:
            strategy = "fixed_fold_csv"
        evaluations = build_evaluations(
            path.name,
            records,
            strategy,
            args.fixed_eval_folds,
            map_train_positive_only=bool(args.map_train_positive_only),
        )
        detail_rows.extend(leakage_rows(evaluations))

    summary_rows = summarize(detail_rows)
    cv_comparison_rows = comparison_rows(detail_rows)
    compact_rows = compact_comparison_rows(cv_comparison_rows)
    detail_fields = [
        "split_file",
        "evaluation_strategy",
        "heldout_label",
        "entity",
        "train_unique_count",
        "heldout_sample_count",
        "leaked_sample_count",
        "leakage_fraction",
        "leakage_percent",
        "heldout_unique_count",
        "leaked_unique_count",
        "unique_leakage_fraction",
        "unique_leakage_percent",
        "leaked_ids",
    ]
    summary_fields = [
        "split_file",
        "evaluation_strategy",
        "entity",
        "n_evaluations",
        "nonzero_evaluations",
        "sum_heldout_sample_count",
        "sum_leaked_sample_count",
        "aggregate_leakage_fraction",
        "aggregate_leakage_percent",
        "mean_fold_leakage_fraction",
        "mean_fold_leakage_percent",
        "sum_heldout_unique_count",
        "sum_leaked_unique_count",
        "unique_aggregate_leakage_fraction",
        "unique_aggregate_leakage_percent",
        "unique_mean_fold_leakage_fraction",
        "unique_mean_fold_leakage_percent",
    ]
    comparison_fields = [
        "split",
        "entity",
        *[f"fold_{idx}" for idx in range(1, 11)],
        "mean_leakage_percent",
        "std_leakage_percent",
    ]
    compact_fields = [
        "split",
        "bgc_leakage_mean_percent",
        "bgc_leakage_std_percent",
        "np_leakage_mean_percent",
        "np_leakage_std_percent",
        "bgc_family_leakage_mean_percent",
        "bgc_family_leakage_std_percent",
        "np_cluster_leakage_mean_percent",
        "np_cluster_leakage_std_percent",
        "summary",
    ]

    detail_path = args.out_dir / "split_leakage_detail.csv"
    summary_path = args.out_dir / "split_leakage_summary.csv"
    comparison_path = args.out_dir / "cv_split_leakage_comparison.csv"
    compact_path = args.out_dir / "cv_split_leakage_compact.csv"
    write_csv(detail_path, detail_rows, detail_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(comparison_path, cv_comparison_rows, comparison_fields)
    write_csv(compact_path, compact_rows, compact_fields)

    plot_path = args.out_dir / "cv_nonzero_leakage_by_fold.png"
    plot_pdf_path = args.out_dir / "cv_nonzero_leakage_by_fold.pdf"
    comparison_plot_path = args.out_dir / "cv_split_leakage_comparison.png"
    comparison_plot_pdf_path = args.out_dir / "cv_split_leakage_comparison.pdf"
    plot_status = "disabled"
    comparison_plot_status = "disabled"
    if not args.no_plots:
        plot_status = save_cv_nonzero_barplot(detail_rows, [plot_path, plot_pdf_path])
        comparison_plot_status = save_comparison_barplot(
            cv_comparison_rows,
            [comparison_plot_path, comparison_plot_pdf_path],
        )

    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {compact_path}")
    if not args.no_plots:
        if plot_status == "created":
            print(f"Wrote {plot_path}")
            print(f"Wrote {plot_pdf_path}")
        elif plot_status == "missing_matplotlib":
            print("matplotlib is not installed in this Python environment; skipped plot.")
        else:
            print("No nonzero CV leakage rows to plot.")
        if comparison_plot_status == "created":
            print(f"Wrote {comparison_plot_path}")
            print(f"Wrote {comparison_plot_pdf_path}")
        elif comparison_plot_status == "missing_matplotlib":
            print("matplotlib is not installed in this Python environment; skipped comparison plot.")
        else:
            print("No comparison rows to plot.")


if __name__ == "__main__":
    main()
