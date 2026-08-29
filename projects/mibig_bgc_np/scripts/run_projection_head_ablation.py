from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.training.contrastive_trainer import evaluate_split_retrieval, train_contrastive
from projects.mibig_bgc_np.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen projection-head variants on cached BGC/NP features.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/MIBIG/processed"))
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/antismash_esm2_t30_domain_sequence_molformer"))
    parser.add_argument("--splits_path", type=Path, default=Path("data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10.tsv"))
    parser.add_argument("--config", type=Path, default=Path("projects/mibig_bgc_np/configs/domain_sequence_molformer.yaml"))
    parser.add_argument("--output_root", type=Path, default=Path("results/intermediate/projection_head_ablation"))
    parser.add_argument("--fold_ids", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--head", required=True, choices=("linear", "mlp_relu", "mlp_gelu", "layernorm_mlp_gelu"))
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def _validation_fold(fold_id: int, n_folds: int = 10) -> int:
    return (int(fold_id) % int(n_folds)) + 1


def _score(metrics: dict[str, dict[str, float]]) -> float:
    return 0.5 * (
        float(metrics["bgc_to_compound"]["mrr"]) + float(metrics["compound_to_bgc"]["mrr"])
    )


def _mean_std(values: list[float]) -> dict[str, float]:
    series = pd.Series(values, dtype="float64")
    return {"mean": float(series.mean()), "std": float(series.std(ddof=0))}


def _summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_folds": int(len(folds))}
    out["mean_bidirectional_mrr"] = _mean_std([float(f["mean_bidirectional_mrr"]) for f in folds])
    for direction, prefix in (("bgc_to_compound", "bgc_to_np"), ("compound_to_bgc", "np_to_bgc")):
        for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10"):
            out[f"{prefix}_{metric}"] = _mean_std(
                [float(f["retrieval_test"][direction][metric]) for f in folds]
            )
    return out


def main() -> None:
    args = parse_args()
    logger = setup_logger("projection_head_ablation")
    cfg = apply_overrides(load_yaml(args.config), args.override)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model", {})["projection_head"] = str(args.head)
    cfg.setdefault("train", {})["lr"] = float(args.lr)
    cfg.setdefault("eval", {})["selection_split"] = str(cfg.get("eval", {}).get("selection_split", "val"))
    run_name = args.run_name or f"{args.head}_lr{args.lr:g}".replace(".", "p")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    fold_results: list[dict[str, Any]] = []
    for fold_id in args.fold_ids:
        val_fold = _validation_fold(int(fold_id))
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["output"]["dir"] = str(run_dir / f"fold_{fold_id}")
        set_seed(int(cfg.get("seed", 42)) + int(fold_id))
        model, contrastive_metrics, _ = train_contrastive(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            cfg=fold_cfg,
            device=device,
            splits_path=args.splits_path,
            cv_fold=int(fold_id),
            val_fold=val_fold,
        )
        interactions = build_interactions(
            args.data_dir,
            splits_path=args.splits_path,
            cv_fold=int(fold_id),
            val_fold=val_fold,
        )
        retrieval_test = evaluate_split_retrieval(
            model=model,
            interactions=interactions,
            split="test",
            bgc_cache_path=args.cache_dir / "bgc_features.pt",
            compound_cache_path=args.cache_dir / "compound_features.pt",
            device=device,
            sim_batch_size=int(fold_cfg["eval"]["sim_batch_size"]),
        )
        fold_result = {
            "fold_id": int(fold_id),
            "val_fold": int(val_fold),
            "head": str(args.head),
            "lr": float(args.lr),
            "output_dir": fold_cfg["output"]["dir"],
            "contrastive_metrics": contrastive_metrics,
            "retrieval_test": retrieval_test,
            "mean_bidirectional_mrr": float(_score(retrieval_test)),
        }
        save_json(fold_result, run_dir / f"fold_{fold_id}" / "projection_head_result.json")
        fold_results.append(fold_result)
        logger.info(
            "fold=%d head=%s lr=%g test_mean_mrr=%.4f",
            int(fold_id),
            args.head,
            float(args.lr),
            float(fold_result["mean_bidirectional_mrr"]),
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "run_name": run_name,
        "head": str(args.head),
        "lr": float(args.lr),
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "splits_path": str(args.splits_path),
        "folds": fold_results,
        "summary": _summarize_folds(fold_results),
    }
    save_json(summary, run_dir / "summary.json")
    logger.info("Saved %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
