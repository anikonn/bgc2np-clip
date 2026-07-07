from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from scripts._bootstrap import ensure_src_path
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from scripts._bootstrap import ensure_src_path

ensure_src_path()


def _load_class_names(path: str | None) -> list[str] | None:
    if path is None:
        return None
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def _top_confused_pairs(report: dict, class_names: list[str], k: int = 3) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    raw = report["confusion_matrix"]["raw"]
    for true_name in class_names:
        for pred_name in class_names:
            if true_name == pred_name:
                continue
            count = int(raw[true_name][pred_name])
            if count > 0:
                pairs.append((true_name, pred_name, count))
    return sorted(pairs, key=lambda item: item[2], reverse=True)[:k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train downstream MIBiG tasks on frozen CLIP embeddings.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument("--cv_fold", type=int, default=None)
    parser.add_argument("--config", type=str, default="projects/mibig_bgc_np/configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--save_cm_png", action="store_true")
    parser.add_argument("--class_names_path", type=str, default=None)
    parser.add_argument(
        "--task",
        action="append",
        choices=["bgc_class", "compound_mw", "origin_type"],
        default=None,
        help="Repeat to run a subset of downstream tasks. By default all downstream tasks are run.",
    )
    parser.add_argument("--npatlas_path", type=str, default="data/NPAtlas_download_2024_09.tsv")
    parser.add_argument("--mibig_pairs_path", type=str, default="data/MIBIG/processed/mibig_pairs.tsv")
    parser.add_argument("--mw_bins", type=int, default=50)
    parser.add_argument("--force_rebuild_match", action="store_true")
    return parser.parse_args()


def _log_classification_summary(logger, task_name: str, metrics: dict) -> None:
    class_names = metrics.get("class_names", [])
    for split in ("val", "test"):
        if split not in metrics:
            continue
        overall = metrics[split]["overall"]
        logger.info(
            "%s %s: loss=%.6f accuracy=%.6f macro_f1=%.6f micro_f1=%.6f",
            task_name,
            split,
            overall["loss"],
            overall["accuracy"],
            overall["macro_f1"],
            overall.get("micro_f1", 0.0),
        )
        if "positive_class" in metrics[split]:
            pos = metrics[split]["positive_class"]
            logger.info(
                "%s %s positive=%s precision=%.6f recall=%.6f f1=%.6f roc_auc=%.6f",
                task_name,
                split,
                pos["label"],
                pos["precision"],
                pos["recall"],
                pos["f1"],
                float(metrics[split].get("roc_auc", 0.0)),
            )
        if class_names:
            confused = _top_confused_pairs(metrics[split], class_names)
            logger.info("%s %s top confused pairs: %s", task_name, split, confused if confused else "none")
    baselines = metrics["test"].get("random_baselines", {})
    if baselines:
        logger.info("%s test baselines: %s", task_name, baselines)


def _log_regression_summary(logger, task_name: str, metrics: dict) -> None:
    for split in ("val", "test"):
        if split not in metrics:
            continue
        report = metrics[split]
        logger.info(
            "%s %s: loss=%.6f mse=%.6f rmse=%.6f r2=%.6f spearman=%.6f",
            task_name,
            split,
            report["loss"],
            report["mse"],
            report["rmse"],
            report["r2"],
            report["spearman"],
        )
    logger.info("%s test baselines: %s", task_name, metrics["test"].get("random_baselines", {}))


def main() -> None:
    args = parse_args()
    from clip_core.config import apply_overrides, load_yaml
    from clip_core.logging import setup_logger
    from projects.mibig_bgc_np.utils.seed import set_seed
    logger = setup_logger()
    from projects.mibig_bgc_np.training.downstream_trainer import train_downstream

    cfg = apply_overrides(load_yaml(args.config), args.override)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg["output"]["dir"] = str(Path(cfg["output"]["dir"]))
    splits_path = args.splits_path if args.splits_path is not None else cfg.get("data", {}).get("splits_path")
    metrics = train_downstream(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        contrastive_ckpt=args.checkpoint,
        cfg=cfg,
        device=device,
        splits_path=splits_path,
        cv_fold=args.cv_fold,
        baseline_trials=int(args.trials),
        class_names=_load_class_names(args.class_names_path),
        save_cm_png=bool(args.save_cm_png),
        tasks=args.task,
        npatlas_path=args.npatlas_path,
        mibig_pairs_path=args.mibig_pairs_path,
        mw_bins=int(args.mw_bins),
        force_rebuild_match=bool(args.force_rebuild_match),
    )
    if "compound_matching" in metrics:
        logger.info("Compound matching summary: %s", metrics["compound_matching"])
        logger.info("Matched compounds saved to %s", metrics["matched_compounds_path"])
    for task_name in metrics["tasks"]:
        task_metrics = metrics[task_name]
        if task_name in {"bgc_class", "origin_type"}:
            _log_classification_summary(logger, task_name, task_metrics)
        elif task_name == "compound_mw":
            _log_regression_summary(logger, task_name, task_metrics)


if __name__ == "__main__":
    main()
