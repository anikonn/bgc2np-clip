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
from projects.mibig_bgc_np.training.contrastive_trainer import train_contrastive
from projects.mibig_bgc_np.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one pilot HPO trial for the frozen BGC-NP CLIP model.")
    parser.add_argument("--trial_id", required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/MIBIG/processed"))
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/antismash_esm2_t30_domain_sequence_molformer"))
    parser.add_argument("--splits_path", type=Path, default=Path("data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10.tsv"))
    parser.add_argument("--config", type=Path, default=Path("projects/mibig_bgc_np/configs/domain_sequence_molformer.yaml"))
    parser.add_argument("--output_root", type=Path, default=Path("results/intermediate/hparam_search_stage1"))
    parser.add_argument("--fold_ids", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--emb_dim", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--scheduler", choices=("none", "cosine_warmup", "linear_warmup_decay"), required=True)
    parser.add_argument("--warmup_fraction", type=float, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def _validation_fold(fold_id: int, n_folds: int = 10) -> int:
    return (int(fold_id) % int(n_folds)) + 1


def _metric(metrics: dict[str, Any], split: str, direction: str, name: str) -> float:
    return float(metrics[f"retrieval_{split}"][direction][name])


def _mean(values: list[float]) -> float:
    return float(pd.Series(values, dtype="float64").mean())


def _std(values: list[float]) -> float:
    return float(pd.Series(values, dtype="float64").std(ddof=0))


def _summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [fold["metrics"] for fold in folds]
    out: dict[str, Any] = {"n_folds": len(rows)}
    for split in ("val", "test"):
        for direction, prefix in (("bgc_to_compound", "bgc_to_np"), ("compound_to_bgc", "np_to_bgc")):
            for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10"):
                values = [_metric(row, split, direction, metric) for row in rows]
                out[f"{split}_{prefix}_{metric}_mean"] = _mean(values)
                out[f"{split}_{prefix}_{metric}_std"] = _std(values)
        val_bi_mrr = [
            0.5 * (_metric(row, split, "bgc_to_compound", "mrr") + _metric(row, split, "compound_to_bgc", "mrr"))
            for row in rows
        ]
        val_bi_r10 = [
            0.5
            * (
                _metric(row, split, "bgc_to_compound", "recall_at_10")
                + _metric(row, split, "compound_to_bgc", "recall_at_10")
            )
            for row in rows
        ]
        out[f"{split}_bidirectional_mrr_mean"] = _mean(val_bi_mrr)
        out[f"{split}_bidirectional_mrr_std"] = _std(val_bi_mrr)
        out[f"{split}_bidirectional_recall_at_10_mean"] = _mean(val_bi_r10)
        out[f"{split}_bidirectional_recall_at_10_std"] = _std(val_bi_r10)
    out["selection_score"] = out["val_bgc_to_np_recall_at_10_mean"]
    return out


def main() -> None:
    args = parse_args()
    logger = setup_logger("hparam_search")
    cfg = apply_overrides(load_yaml(args.config), args.override)
    cfg = copy.deepcopy(cfg)
    cfg["model"]["projection_head"] = "mlp_gelu"
    cfg["model"]["hidden_dim"] = int(args.hidden_dim)
    cfg["model"]["emb_dim"] = int(args.emb_dim)
    cfg["model"]["dropout"] = float(args.dropout)
    cfg["model"]["init_temperature"] = float(args.temperature)
    cfg["train"]["lr"] = float(args.lr)
    cfg["train"]["weight_decay"] = float(args.weight_decay)
    cfg["train"]["batch_size"] = int(args.batch_size)
    cfg["train"]["scheduler"] = str(args.scheduler)
    cfg["train"]["warmup_fraction"] = float(args.warmup_fraction)
    cfg["eval"]["selection_split"] = "val"
    cfg["eval"]["selection_metric"] = "bgc_to_np_recall_at_10"

    trial_dir = args.output_root / str(args.trial_id)
    trial_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "trial_id": str(args.trial_id),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "hidden_dim": int(args.hidden_dim),
            "emb_dim": int(args.emb_dim),
            "temperature": float(args.temperature),
            "scheduler": str(args.scheduler),
            "warmup_fraction": float(args.warmup_fraction),
            "selection_metric": "val_bgc_to_np_recall_at_10",
        },
        trial_dir / "trial_config.json",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("trial=%s device=%s", args.trial_id, device)
    folds: list[dict[str, Any]] = []
    for fold_id in args.fold_ids:
        val_fold = _validation_fold(int(fold_id))
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["output"]["dir"] = str(trial_dir / f"fold_{fold_id}")
        set_seed(int(cfg.get("seed", 42)) + int(fold_id))
        _, metrics, _ = train_contrastive(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            cfg=fold_cfg,
            device=device,
            splits_path=args.splits_path,
            cv_fold=int(fold_id),
            val_fold=val_fold,
        )
        fold_result = {
            "fold_id": int(fold_id),
            "val_fold": int(val_fold),
            "metrics": metrics,
        }
        save_json(fold_result, trial_dir / f"fold_{fold_id}" / "hparam_trial_result.json")
        folds.append(fold_result)
        logger.info(
            "trial=%s fold=%d val_bgc_to_np_r10=%.4f test_bgc_to_np_r10=%.4f",
            args.trial_id,
            int(fold_id),
            _metric(metrics, "val", "bgc_to_compound", "recall_at_10"),
            _metric(metrics, "test", "bgc_to_compound", "recall_at_10"),
        )

    summary = {
        "trial_id": str(args.trial_id),
        "fold_ids": [int(x) for x in args.fold_ids],
        "config": cfg,
        "folds": folds,
        "summary": _summarize(folds),
    }
    save_json(summary, trial_dir / "summary.json")
    logger.info("trial=%s selection_score=%.4f", args.trial_id, float(summary["summary"]["selection_score"]))


if __name__ == "__main__":
    main()
