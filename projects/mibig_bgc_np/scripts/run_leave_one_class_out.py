from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.eval.baseline_artifacts import save_all_baseline_artifacts
from projects.mibig_bgc_np.eval.retrieval_baselines import run_retrieval_baseline_suite
from projects.mibig_bgc_np.training.contrastive_trainer import evaluate_split_retrieval, train_contrastive
from projects.mibig_bgc_np.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leave-one-BGC-product-class-out retrieval experiments.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/ohe.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--split_glob",
        type=str,
        default="*.tsv",
        help="Glob relative to splits_dir. Example: '*.tsv' or 'loco_exp3_bgc_*.tsv'.",
    )
    parser.add_argument("--retrieval_baselines", dest="retrieval_baselines", action="store_true", default=True)
    parser.add_argument("--no_retrieval_baselines", dest="retrieval_baselines", action="store_false")
    parser.add_argument("--baseline_random_trials", type=int, default=10)
    parser.add_argument("--baseline_k_values", type=int, nargs="*", default=[1, 5, 10, 20, 50, 100, 200, 500])
    parser.add_argument("--include_linear_baseline", dest="include_linear_baseline", action="store_true", default=True)
    parser.add_argument("--no_linear_baseline", dest="include_linear_baseline", action="store_false")
    return parser.parse_args()


def _as_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _aggregate_objects(values: list[Any]) -> Any:
    if not values:
        return None
    if all(_as_number(value) for value in values):
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}
    if all(isinstance(value, dict) for value in values):
        keys = sorted({key for value in values for key in value})
        out: dict[str, Any] = {}
        for key in keys:
            child = _aggregate_objects([value[key] for value in values if key in value])
            if child is not None:
                out[key] = child
        return out
    return None


def _split_counts(interactions: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in sorted(interactions["split"].astype(str).str.lower().unique().tolist()):
        split_df = interactions[interactions["split"].astype(str).str.lower() == split].copy()
        counts[split] = {
            "n_bgcs": int(split_df["bgc_id"].astype(str).nunique()),
            "n_compounds": int(split_df["compound_id"].astype(str).nunique()),
            "n_pairs": int(len(split_df)),
        }
    return counts


def _target_class_from_split(path: Path) -> str:
    split_df = pd.read_csv(path, sep="\t", usecols=lambda col: col in {"target_class"})
    if "target_class" in split_df.columns:
        values = split_df["target_class"].dropna().astype(str).drop_duplicates().tolist()
        if values:
            return values[0]
    stem = path.stem
    match = re.search(r"(?:exp3_bgc|exp4_np)_(.+)$", stem)
    return match.group(1) if match else stem


def _find_split_files(splits_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(path for path in splits_dir.glob(pattern) if path.is_file() and path.suffix == ".tsv")
    if not paths:
        raise FileNotFoundError(f"No split TSV files found under {splits_dir} matching {pattern!r}")
    return paths


def _build_summary(experiment_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "counts": _aggregate_objects([summary["counts"] for summary in experiment_summaries]),
        "contrastive_metrics": _aggregate_objects([summary["contrastive_metrics"] for summary in experiment_summaries]),
        "retrieval_test": _aggregate_objects([summary["retrieval_test"] for summary in experiment_summaries]),
        "retrieval_baselines_test": _aggregate_objects(
            [
                summary["retrieval_baselines_test"]
                for summary in experiment_summaries
                if "retrieval_baselines_test" in summary
            ]
        ),
    }


def main() -> None:
    args = parse_args()
    logger = setup_logger("mibig_loco")
    cfg = apply_overrides(load_yaml(args.config), args.override)
    cfg["seed"] = int(args.seed)

    splits_dir = Path(args.splits_dir)
    split_files = _find_split_files(splits_dir, str(args.split_glob))
    run_name = args.run_name or f"leave_one_class_out_{splits_dir.name}"
    root_outdir = Path("results") / run_name
    root_outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_summaries: list[dict[str, Any]] = []
    for experiment_idx, splits_path in enumerate(split_files, start=1):
        target_class = _target_class_from_split(splits_path)
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", target_class.strip().lower()).strip("_") or f"class_{experiment_idx}"
        experiment_outdir = root_outdir / f"{experiment_idx:02d}_{safe_name}"
        experiment_outdir.mkdir(parents=True, exist_ok=True)

        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["output"]["dir"] = str(experiment_outdir)
        set_seed(int(args.seed) + int(experiment_idx))

        logger.info("Starting LOCO experiment %d/%d target_class=%s", experiment_idx, len(split_files), target_class)
        interactions = build_interactions(args.data_dir, splits_path=splits_path)
        counts = _split_counts(interactions)
        if "train" not in counts or "test" not in counts:
            raise ValueError(f"{splits_path} must create train and test rows; got splits {sorted(counts)}")
        logger.info("Counts for %s: %s", target_class, counts)

        model, contrastive_metrics, _ = train_contrastive(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            cfg=fold_cfg,
            device=device,
            splits_path=splits_path,
        )

        retrieval_test = evaluate_split_retrieval(
            model=model,
            interactions=interactions,
            split="test",
            bgc_cache_path=Path(args.cache_dir) / "bgc_features.pt",
            compound_cache_path=Path(args.cache_dir) / "compound_features.pt",
            device=device,
            sim_batch_size=int(fold_cfg["eval"]["sim_batch_size"]),
        )
        save_json(retrieval_test, experiment_outdir / "retrieval_test.json")

        retrieval_baselines_test: dict[str, Any] = {}
        if bool(args.retrieval_baselines):
            retrieval_baselines_test = run_retrieval_baseline_suite(
                interactions=interactions,
                split="test",
                cache_dir=args.cache_dir,
                cfg=fold_cfg,
                device=device,
                outdir=experiment_outdir / "retrieval_baselines",
                seed=int(args.seed) + int(experiment_idx),
                random_trials=int(args.baseline_random_trials),
                k_values=[int(k) for k in args.baseline_k_values],
                include_linear=bool(args.include_linear_baseline),
            )
            save_json(retrieval_baselines_test, experiment_outdir / "retrieval_baselines_test.json")

        experiment_summary = {
            "fold_id": int(experiment_idx),
            "experiment_id": int(experiment_idx),
            "target_class": target_class,
            "splits_path": str(splits_path),
            "output_dir": str(experiment_outdir),
            "counts": counts,
            "contrastive_metrics": contrastive_metrics,
            "retrieval_test": retrieval_test,
            "retrieval_baselines_test": retrieval_baselines_test,
        }
        save_json(experiment_summary, experiment_outdir / "experiment_summary.json")
        experiment_summaries.append(experiment_summary)

    summary = {
        "seed": int(args.seed),
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "splits_dir": str(splits_dir),
        "split_glob": str(args.split_glob),
        "n_experiments": int(len(experiment_summaries)),
        "folds": experiment_summaries,
        "aggregate": _build_summary(experiment_summaries),
    }
    summary_path = root_outdir / "summary.json"
    save_json(summary, summary_path)
    logger.info("Saved LOCO summary to %s", summary_path)

    try:
        artifacts = save_all_baseline_artifacts(root_outdir)
        summary["baseline_artifacts"] = artifacts
        save_json(summary, summary_path)
    except Exception as exc:
        logger.warning("Could not create baseline artifacts: %s", exc)


if __name__ == "__main__":
    main()
