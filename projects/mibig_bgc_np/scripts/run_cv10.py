from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import build_bgc_class_table, build_interactions
from projects.mibig_bgc_np.eval.baseline_artifacts import save_all_baseline_artifacts
from projects.mibig_bgc_np.eval.retrieval_class_metrics import save_bgc_class_retrieval_plots
from projects.mibig_bgc_np.eval.retrieval_baselines import run_retrieval_baseline_suite
from projects.mibig_bgc_np.scripts.run_bgcmac_ensemble import (
    _parse_label_text,
    _train_raw_bgc_baseline_member,
)
from projects.mibig_bgc_np.training.contrastive_trainer import (
    evaluate_split_bgc_class_retrieval,
    evaluate_split_retrieval,
    train_contrastive,
)
from projects.mibig_bgc_np.training.downstream_trainer import train_downstream
from projects.mibig_bgc_np.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N-fold CV for the MIBiG BGC-NP CLIP pipeline.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--split_type", choices=("bgc", "combined", "np", "strict"), default="combined")
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional results directory name under results/. Defaults to <split_type>_cv<n_folds>.",
    )
    parser.add_argument(
        "--retrieval_baselines",
        dest="retrieval_baselines",
        action="store_true",
        default=True,
        help="Run random, frozen encoder similarity, kNN transfer, and linear-projection retrieval baselines. Enabled by default.",
    )
    parser.add_argument(
        "--no_retrieval_baselines",
        dest="retrieval_baselines",
        action="store_false",
        help="Disable retrieval baselines.",
    )
    parser.add_argument("--baseline_random_trials", type=int, default=10)
    parser.add_argument("--baseline_k_values", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument(
        "--raw_bgc_classifier_baseline",
        dest="raw_bgc_classifier_baseline",
        action="store_true",
        default=True,
        help="Run cached-BGC-feature MLP classifier baseline for downstream BGC class prediction. Enabled by default.",
    )
    parser.add_argument(
        "--no_raw_bgc_classifier_baseline",
        dest="raw_bgc_classifier_baseline",
        action="store_false",
        help="Disable cached-BGC-feature MLP classifier baseline.",
    )
    parser.add_argument(
        "--save_cm_png",
        dest="save_cm_png",
        action="store_true",
        default=True,
        help="Save per-fold downstream confusion matrices and ROC plots. Enabled by default.",
    )
    parser.add_argument(
        "--no_save_cm_png",
        dest="save_cm_png",
        action="store_false",
        help="Disable per-fold downstream confusion matrices, ROC plots, and aggregate CV confusion plots.",
    )
    return parser.parse_args()


def _resolve_cv_splits_path(
    cfg: dict[str, Any],
    seed: int,
    n_folds: int,
    split_type: str,
    splits_path: str | None,
) -> Path:
    if splits_path is not None:
        path = Path(splits_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"CV split file does not exist: {path}")

    candidates = []
    if split_type == "strict":
        candidates.append(Path(f"data/MIBIG/splits/strict_bigscape_butina_cv_seed{seed}_n{n_folds}.tsv"))
    candidates.extend([
        Path(f"data/MIBIG/splits/{split_type}_cv_seed{seed}_n{n_folds}.tsv"),
        Path(f"data/MIBIG/splits/cv_seed{seed}_n{n_folds}.tsv"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate

    cfg_path = cfg.get("data", {}).get("splits_path")
    if cfg_path is not None:
        path = Path(str(cfg_path))
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find a CV split file. Expected one of {candidates} or an existing data.splits_path in the config."
    )


def _normalize_split_columns(split_df: pd.DataFrame) -> pd.DataFrame:
    split_df = split_df.copy()
    if "split" not in split_df.columns and "strict_split" in split_df.columns:
        split_df["split"] = split_df["strict_split"]
    if "fold_id" not in split_df.columns and "strict_cv10_fold" in split_df.columns:
        split_df["fold_id"] = pd.to_numeric(split_df["strict_cv10_fold"], errors="coerce") + 1
    return split_df


def _load_cv_assignments(path: Path, n_folds: int) -> pd.DataFrame:
    split_df = _normalize_split_columns(pd.read_csv(path, sep="\t"))
    required = {"bgc_id", "fold_id"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"CV split file {path} is missing required columns: {sorted(missing)}")

    keep_cols = ["bgc_id", "fold_id"]
    if "compound_id" in split_df.columns:
        keep_cols.insert(1, "compound_id")
    split_df = split_df[keep_cols].copy()
    split_df["bgc_id"] = split_df["bgc_id"].astype(str)
    if "compound_id" in split_df.columns:
        split_df["compound_id"] = split_df["compound_id"].astype(str)
    split_df["fold_id"] = pd.to_numeric(split_df["fold_id"], errors="coerce")
    if bool(split_df["fold_id"].isna().any()):
        raise ValueError(f"CV split file {path} contains non-numeric fold_id values.")
    split_df["fold_id"] = split_df["fold_id"].astype(int)

    key_cols = ["bgc_id", "compound_id"] if "compound_id" in split_df.columns else ["bgc_id"]
    if split_df.duplicated(subset=key_cols).any():
        conflicting = split_df.groupby(key_cols)["fold_id"].nunique()
        conflicting = conflicting[conflicting > 1]
        if not conflicting.empty:
            dupes = conflicting.reset_index().head(5)[key_cols].to_dict("records")
            raise ValueError(f"CV split file {path} contains conflicting assignments for {key_cols}, e.g. {dupes}")
        split_df = split_df.drop_duplicates(subset=key_cols).copy()

    bad_folds = sorted(set(split_df["fold_id"].tolist()).difference(range(1, n_folds + 1)))
    if bad_folds:
        raise ValueError(f"CV split file {path} contains fold_id values outside 1..{n_folds}: {bad_folds}")
    return split_df


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _aggregate_objects(values: list[Any]) -> Any:
    if not values:
        return None
    if all(_is_number(value) for value in values):
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "n": int(arr.size),
        }
    if all(isinstance(value, dict) for value in values):
        keys = sorted({key for value in values for key in value})
        aggregated: dict[str, Any] = {}
        for key in keys:
            child_values = [value[key] for value in values if key in value]
            child = _aggregate_objects(child_values)
            if child is not None:
                aggregated[key] = child
        return aggregated
    return None


def _build_raw_classifier_bgc_table(
    data_dir: str | Path,
    splits_path: str | Path,
    cv_fold: int,
    n_folds: int,
) -> pd.DataFrame:
    bgc_df = build_bgc_class_table(data_dir, splits_path=splits_path, cv_fold=cv_fold).copy()
    if "val" not in set(bgc_df["split"].astype(str).str.lower()) and "fold_id" in bgc_df.columns:
        val_fold = (int(cv_fold) % int(n_folds)) + 1
        bgc_df["fold_id"] = pd.to_numeric(bgc_df["fold_id"], errors="coerce")
        bgc_df["split"] = np.where(
            bgc_df["fold_id"] == int(cv_fold),
            "test",
            np.where(bgc_df["fold_id"] == int(val_fold), "val", "train"),
        )
    bgc_df["split"] = bgc_df["split"].astype(str).str.lower()
    if "bgc_class_list" not in bgc_df.columns:
        bgc_df["bgc_class_list"] = bgc_df["bgc_classes"].map(_parse_label_text)
    return bgc_df[bgc_df["bgc_class_list"].map(len) > 0].copy()


def _build_summary(fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "counts": _aggregate_objects([summary["counts"] for summary in fold_summaries]),
        "contrastive_metrics": _aggregate_objects([summary["contrastive_metrics"] for summary in fold_summaries]),
        "retrieval_test": _aggregate_objects([summary["retrieval_test"] for summary in fold_summaries]),
        "retrieval_baselines_test": _aggregate_objects(
            [summary["retrieval_baselines_test"] for summary in fold_summaries if "retrieval_baselines_test" in summary]
        ),
        "retrieval_class_test": _aggregate_objects([summary["retrieval_class_test"] for summary in fold_summaries]),
        "downstream": _aggregate_objects([summary["downstream"] for summary in fold_summaries]),
        "raw_bgc_classifier_baseline": _aggregate_objects(
            [
                summary["raw_bgc_classifier_baseline"]
                for summary in fold_summaries
                if "raw_bgc_classifier_baseline" in summary
            ]
        ),
    }


def _build_raw_classifier_baseline_summary(fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    fold_values = [
        {
            "fold_id": int(summary["fold_id"]),
            "output_dir": str(Path(summary["output_dir"]) / "raw_bgc_classifier_baseline"),
            "metrics": summary["raw_bgc_classifier_baseline"],
        }
        for summary in fold_summaries
        if summary.get("raw_bgc_classifier_baseline")
    ]
    return {
        "name": "raw_bgc_classifier_baseline",
        "input": "cached BGC encoder features",
        "model": "MLP multilabel classifier",
        "note": "This is OHE+MLP when the cache was built with the OHE BGC encoder.",
        "n_folds": int(len(fold_values)),
        "folds": fold_values,
        "aggregate": _aggregate_objects([fold["metrics"] for fold in fold_values]),
    }


def main() -> None:
    args = parse_args()
    logger = setup_logger("mibig_cv")

    if args.n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {args.n_folds}")

    cfg = apply_overrides(load_yaml(args.config), args.override)
    cfg["seed"] = int(args.seed)
    splits_path = _resolve_cv_splits_path(
        cfg,
        seed=int(args.seed),
        n_folds=int(args.n_folds),
        split_type=str(args.split_type),
        splits_path=args.splits_path,
    )
    split_df = _load_cv_assignments(splits_path, n_folds=int(args.n_folds))

    run_name = str(args.run_name) if args.run_name is not None else f"{args.split_type}_cv{args.n_folds}"
    root_outdir = Path("results") / run_name
    root_outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(1, int(args.n_folds) + 1):
        fold_outdir = root_outdir / f"fold_{fold_id}"
        fold_outdir.mkdir(parents=True, exist_ok=True)

        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["output"]["dir"] = str(fold_outdir)
        set_seed(int(args.seed))

        logger.info("Starting fold %d/%d", fold_id, args.n_folds)
        interactions = build_interactions(args.data_dir, splits_path=splits_path, cv_fold=fold_id)
        counts = _split_counts(interactions)
        logger.info("Fold %d counts: %s", fold_id, counts)

        model, contrastive_metrics, _ = train_contrastive(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            cfg=fold_cfg,
            device=device,
            splits_path=splits_path,
            cv_fold=fold_id,
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
        save_json(retrieval_test, fold_outdir / "retrieval_test.json")

        retrieval_class_test = evaluate_split_bgc_class_retrieval(
            model=model,
            interactions=interactions,
            split="test",
            bgc_cache_path=Path(args.cache_dir) / "bgc_features.pt",
            compound_cache_path=Path(args.cache_dir) / "compound_features.pt",
            device=device,
        )
        retrieval_class_test["plots"] = save_bgc_class_retrieval_plots(
            retrieval_class_test,
            fold_outdir,
            prefix="test",
        ) if bool(args.save_cm_png) else []
        save_json(retrieval_class_test, fold_outdir / "retrieval_class_test.json")

        retrieval_baselines_test: dict[str, Any] = {}
        if bool(args.retrieval_baselines):
            retrieval_baselines_test = run_retrieval_baseline_suite(
                interactions=interactions,
                split="test",
                cache_dir=args.cache_dir,
                cfg=fold_cfg,
                device=device,
                outdir=fold_outdir / "retrieval_baselines",
                seed=int(args.seed) + int(fold_id),
                random_trials=int(args.baseline_random_trials),
                k_values=[int(k) for k in args.baseline_k_values],
            )

        downstream_metrics = train_downstream(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            contrastive_ckpt=fold_outdir / "contrastive_model_best.pt",
            cfg=fold_cfg,
            device=device,
            splits_path=splits_path,
            cv_fold=fold_id,
            save_cm_png=bool(args.save_cm_png),
        )

        raw_bgc_classifier_baseline: dict[str, Any] = {}
        if bool(args.raw_bgc_classifier_baseline):
            raw_bgc_df = _build_raw_classifier_bgc_table(
                data_dir=args.data_dir,
                splits_path=splits_path,
                cv_fold=int(fold_id),
                n_folds=int(args.n_folds),
            )
            bgc_cache = torch.load(Path(args.cache_dir) / "bgc_features.pt", map_location="cpu")
            _, raw_bgc_classifier_baseline = _train_raw_bgc_baseline_member(
                bgc_df=raw_bgc_df,
                bgc_cache=bgc_cache,
                cfg=fold_cfg,
                device=device,
                output_dir=fold_outdir / "raw_bgc_classifier_baseline",
            )

        fold_summary = {
            "fold_id": int(fold_id),
            "output_dir": str(fold_outdir),
            "counts": counts,
            "contrastive_metrics": contrastive_metrics,
            "retrieval_test": retrieval_test,
            "retrieval_baselines_test": retrieval_baselines_test,
            "retrieval_class_test": retrieval_class_test,
            "downstream": downstream_metrics,
            "raw_bgc_classifier_baseline": raw_bgc_classifier_baseline,
        }
        save_json(fold_summary, fold_outdir / "fold_summary.json")
        fold_summaries.append(fold_summary)

    summary = {
        "seed": int(args.seed),
        "n_folds": int(args.n_folds),
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "splits_path": str(splits_path),
        "cv_assignments": {
            "n_bgcs": int(split_df["bgc_id"].nunique()),
            "n_pairs": int(len(split_df)),
            "n_compounds": int(split_df["compound_id"].nunique()) if "compound_id" in split_df.columns else None,
            "fold_sizes": {
                str(fold_id): int((split_df["fold_id"] == fold_id).sum())
                for fold_id in range(1, int(args.n_folds) + 1)
            },
        },
        "folds": fold_summaries,
        "aggregate": _build_summary(fold_summaries),
    }
    summary_path = root_outdir / "summary.json"
    save_json(summary, summary_path)
    logger.info("Saved CV summary to %s", summary_path)

    if bool(args.raw_bgc_classifier_baseline):
        raw_baseline_summary = _build_raw_classifier_baseline_summary(fold_summaries)
        raw_baseline_summary_path = root_outdir / "raw_bgc_classifier_baseline_summary.json"
        save_json(raw_baseline_summary, raw_baseline_summary_path)
        logger.info("Saved raw BGC classifier baseline summary to %s", raw_baseline_summary_path)

    try:
        baseline_artifacts = save_all_baseline_artifacts(root_outdir)
        summary["baseline_artifacts"] = baseline_artifacts
        save_json(summary, summary_path)
        logger.info("Saved visible baseline artifacts to %s", root_outdir / "baselines")
    except Exception as exc:
        logger.warning("Could not create visible baseline artifacts: %s", exc)

    try:
        from projects.mibig_bgc_np.scripts.plot_retrieval_summary import (
            build_retrieval_long,
            plot_class_retrieval,
            plot_mrr,
            plot_topk_recall,
            summarize_retrieval,
        )

        retrieval_plot_dir = root_outdir / "retrieval_plots"
        retrieval_plot_dir.mkdir(parents=True, exist_ok=True)
        top_k_values = [1, 5, 10, 20, 50, 100, 200, 500]
        retrieval_long = build_retrieval_long(summary, top_k_values)
        retrieval_long.to_csv(retrieval_plot_dir / "retrieval_long.csv", index=False)
        retrieval_summary = summarize_retrieval(retrieval_long)
        retrieval_summary.to_csv(retrieval_plot_dir / "retrieval_summary.csv", index=False)
        retrieval_plots = {
            "topk_recall": plot_topk_recall(retrieval_summary, retrieval_plot_dir, "retrieval", top_k_values),
            "mrr": plot_mrr(retrieval_summary, retrieval_plot_dir, "retrieval"),
            "class_retrieval": plot_class_retrieval(summary, retrieval_plot_dir, "retrieval"),
        }
        summary["retrieval_plot_artifacts"] = retrieval_plots
        save_json(summary, summary_path)
        logger.info("Saved retrieval plots to %s", retrieval_plot_dir)
    except Exception as exc:
        logger.warning("Could not create retrieval plots: %s", exc)

    if bool(args.save_cm_png):
        try:
            from scripts.plot_cv_summary_confusion_matrices import plot_summary

            outputs = plot_summary(summary_path, root_outdir / "summary_confusion_matrices")
            logger.info("Saved %d aggregate CV summary plots to %s", len(outputs), root_outdir / "summary_confusion_matrices")
        except Exception as exc:
            logger.warning("Could not create aggregate CV summary plots: %s", exc)


if __name__ == "__main__":
    main()
