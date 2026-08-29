from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.eval.retrieval_class_metrics import save_bgc_class_retrieval_plots
from projects.mibig_bgc_np.models.clip_dual import DualEncoderCLIP
from projects.mibig_bgc_np.scripts.run_bgcmac_ensemble import _build_bgcmac_interactions, _load_bgcmac_fold_table
from projects.mibig_bgc_np.training.contrastive_trainer import _get_cached_paths, evaluate_split_bgc_class_retrieval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill BGC-class retrieval ROC/AUC diagnostics into result folders.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--kind", choices=("cv", "bgcmac"), default="cv")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument("--bgcmac_splits_path", type=str, default=None)
    parser.add_argument("--test_fold", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _aggregate_objects(values: list[Any]) -> Any:
    if not values:
        return None
    if all(_is_number(value) for value in values):
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


def _load_model(checkpoint_path: Path, device: torch.device) -> DualEncoderCLIP:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = DualEncoderCLIP(
        bgc_input_dim=int(ckpt["bgc_input_dim"]),
        compound_input_dim=int(ckpt["compound_input_dim"]),
        emb_dim=int(cfg["model"]["emb_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        init_temperature=float(cfg["model"]["init_temperature"]),
        max_logit_scale=float(cfg["model"]["max_logit_scale"]),
        projection_head=str(cfg["model"].get("projection_head", "mlp_gelu")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _resolve(summary: dict[str, Any], args: argparse.Namespace, key: str, default: str | None = None) -> str:
    value = getattr(args, key)
    if value is not None:
        return str(value)
    if key in summary and summary[key] is not None:
        return str(summary[key])
    if default is not None:
        return default
    raise ValueError(f"Could not resolve {key}; pass --{key}.")


def _backfill_cv(summary_path: Path, summary: dict[str, Any], args: argparse.Namespace, device: torch.device) -> None:
    data_dir = _resolve(summary, args, "data_dir")
    cache_dir = _resolve(summary, args, "cache_dir")
    splits_path = _resolve(summary, args, "splits_path")
    bgc_cache_path, compound_cache_path = _get_cached_paths(cache_dir)

    for fold_summary in summary.get("folds", []):
        fold_id = int(fold_summary["fold_id"])
        outdir = Path(fold_summary["output_dir"])
        model = _load_model(outdir / "contrastive_model_best.pt", device)
        interactions = build_interactions(data_dir, splits_path=splits_path, cv_fold=fold_id)
        report = evaluate_split_bgc_class_retrieval(
            model=model,
            interactions=interactions,
            split="test",
            bgc_cache_path=bgc_cache_path,
            compound_cache_path=compound_cache_path,
            device=device,
        )
        report["plots"] = [] if bool(args.no_plots) else save_bgc_class_retrieval_plots(report, outdir, prefix="test")
        save_json(report, outdir / "retrieval_class_test.json")
        fold_summary["retrieval_class_test"] = report

    summary.setdefault("aggregate", {})["retrieval_class_test"] = _aggregate_objects(
        [fold["retrieval_class_test"] for fold in summary.get("folds", []) if "retrieval_class_test" in fold]
    )
    save_json(summary, summary_path)


def _backfill_bgcmac(summary_path: Path, summary: dict[str, Any], args: argparse.Namespace, device: torch.device) -> None:
    from projects.mibig_bgc_np.scripts.run_bgcmac_ensemble import _evaluate_ensemble

    data_dir = _resolve(summary, args, "data_dir")
    cache_dir = _resolve(summary, args, "cache_dir")
    splits_path = str(args.bgcmac_splits_path or summary.get("bgcmac_splits_path"))
    test_fold = int(args.test_fold if args.test_fold is not None else summary.get("test_fold", 10))
    outdir = summary_path.parent
    fold_table = _load_bgcmac_fold_table(splits_path, test_fold=test_fold)
    val_folds = [int(fold) for fold in summary.get("val_folds", [])]
    models = [
        _load_model(outdir / f"val_fold_{val_fold}" / "contrastive_model_best.pt", device)
        for val_fold in val_folds
    ]
    interactions = _build_bgcmac_interactions(data_dir, fold_table, val_fold=val_folds[0])
    ensemble = _evaluate_ensemble(models, interactions, cache_dir=cache_dir, device=device)
    ensemble["bgc_class_retrieval"]["plots"] = [] if bool(args.no_plots) else save_bgc_class_retrieval_plots(
        ensemble["bgc_class_retrieval"],
        outdir,
        prefix="ensemble_test",
    )
    save_json(ensemble, outdir / "ensemble_test_retrieval.json")
    summary["ensemble_test"] = ensemble
    save_json(summary, summary_path)


def main() -> None:
    args = _parse_args()
    logger = setup_logger("retrieval_class_backfill")
    summary = json.loads(args.summary.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.kind == "cv":
        _backfill_cv(args.summary, summary, args, device)
    else:
        _backfill_bgcmac(args.summary, summary, args, device)
    logger.info("Backfilled BGC-class retrieval diagnostics into %s", args.summary)


if __name__ == "__main__":
    main()
