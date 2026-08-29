from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from scripts._bootstrap import ensure_src_path

ensure_src_path()

NPCLASSIFIER_URL = "https://npclassifier.gnps2.org/classify"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNPS NPClassifier on unique MIBiG compound structures.")
    parser.add_argument("--pairs_path", type=Path, default=Path("data/MIBIG/processed/mibig_pairs.tsv"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/MIBIG/processed"))
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/npclassifier"))
    parser.add_argument("--report_path", type=Path, default=Path("results/EDA/npclassifier_report.json"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between uncached API calls.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit on unique structures.")
    parser.add_argument("--force", action="store_true", help="Ignore cached API responses and request again.")
    parser.add_argument(
        "--keep_error_cache",
        action="store_true",
        help="Reuse cached error responses instead of retrying them. By default cached errors are retried.",
    )
    parser.add_argument("--progress_every", type=int, default=25, help="Print progress every N structures.")
    return parser.parse_args()


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required to canonicalize MIBiG SMILES before NPClassifier calls."
        ) from exc
    return Chem


def _canonical_smiles(smiles: str, chem_module: Any) -> str | None:
    mol = chem_module.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return str(chem_module.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _cache_path(cache_dir: Path, canonical_smiles: str) -> Path:
    digest = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _fetch_npclassifier(smiles: str, *, timeout: float, retries: int) -> dict[str, Any]:
    query = urllib.parse.urlencode({"smiles": smiles})
    url = f"{NPCLASSIFIER_URL}?{query}"
    last_error: str | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                payload = response.read().decode("utf-8")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError(f"Unexpected NPClassifier response type: {type(parsed).__name__}")
            parsed["_request_url"] = url
            return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            if attempt < int(retries):
                time.sleep(min(10.0, 0.75 * attempt))
    return {"error": last_error or "unknown error", "_request_url": url}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    return [str(value)]


def _record_from_response(canonical_smiles: str, response: dict[str, Any]) -> dict[str, Any]:
    class_results = _as_list(response.get("class_results"))
    superclass_results = _as_list(response.get("superclass_results"))
    pathway_results = _as_list(response.get("pathway_results"))
    isglycoside = response.get("isglycoside")
    error = response.get("error")
    return {
        "canonical_smiles": canonical_smiles,
        "npclassifier_class": ";".join(class_results),
        "npclassifier_superclass": ";".join(superclass_results),
        "npclassifier_pathway": ";".join(pathway_results),
        "npclassifier_isglycoside": bool(isglycoside) if isglycoside is not None else None,
        "npclassifier_error": str(error) if error else "",
        "n_class_results": int(len(class_results)),
        "n_superclass_results": int(len(superclass_results)),
        "n_pathway_results": int(len(pathway_results)),
    }


def build_unique_mibig_structures(pairs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(pairs_path, sep="\t")
    if "smiles" not in pairs.columns:
        raise ValueError(f"{pairs_path} must contain a smiles column.")
    chem = _require_rdkit()
    pairs = pairs.copy()
    pairs["canonical_smiles"] = [_canonical_smiles(smiles, chem) for smiles in pairs["smiles"].tolist()]
    valid_pairs = pairs.dropna(subset=["canonical_smiles"]).copy()
    unique = (
        valid_pairs[["canonical_smiles", "smiles", "compound_name"] if "compound_name" in valid_pairs.columns else ["canonical_smiles", "smiles"]]
        .drop_duplicates(subset=["canonical_smiles"])
        .sort_values("canonical_smiles")
        .reset_index(drop=True)
    )
    return valid_pairs, unique


def classify_structures(
    unique_structures: pd.DataFrame,
    cache_dir: Path,
    *,
    timeout: float,
    sleep: float,
    retries: int,
    force: bool,
    keep_error_cache: bool,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    stats = {"n_total": int(len(unique_structures)), "n_cached": 0, "n_requested": 0, "n_errors": 0}
    for idx, row in enumerate(unique_structures.itertuples(index=False), start=1):
        canonical = str(row.canonical_smiles)
        path = _cache_path(cache_dir, canonical)
        if path.exists() and not force:
            response = json.loads(path.read_text(encoding="utf-8"))
            if response.get("error") and not keep_error_cache:
                response = _fetch_npclassifier(canonical, timeout=timeout, retries=retries)
                path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                stats["n_requested"] += 1
                if sleep > 0:
                    time.sleep(float(sleep))
            else:
                stats["n_cached"] += 1
        else:
            response = _fetch_npclassifier(canonical, timeout=timeout, retries=retries)
            path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stats["n_requested"] += 1
            if sleep > 0:
                time.sleep(float(sleep))
        if response.get("error"):
            stats["n_errors"] += 1
        record = _record_from_response(canonical, response)
        if hasattr(row, "smiles"):
            record["example_smiles"] = str(row.smiles)
        if hasattr(row, "compound_name"):
            record["example_compound_name"] = str(row.compound_name)
        records.append(record)
        if progress_every > 0 and (idx % int(progress_every) == 0 or idx == len(unique_structures)):
            print(
                f"NPClassifier progress {idx}/{len(unique_structures)} "
                f"(cached={stats['n_cached']} requested={stats['n_requested']} errors={stats['n_errors']})",
                flush=True,
            )
    return pd.DataFrame(records), stats


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    valid_pairs, unique = build_unique_mibig_structures(args.pairs_path)
    if args.limit is not None:
        unique = unique.head(int(args.limit)).copy()

    labels, stats = classify_structures(
        unique,
        args.cache_dir,
        timeout=float(args.timeout),
        sleep=float(args.sleep),
        retries=int(args.retries),
        force=bool(args.force),
        keep_error_cache=bool(args.keep_error_cache),
        progress_every=int(args.progress_every),
    )
    labels_path = args.out_dir / "mibig_npclassifier_labels.tsv"
    labels.to_csv(labels_path, sep="\t", index=False)

    pair_labels = valid_pairs.merge(labels, on="canonical_smiles", how="inner")
    pair_labels_path = args.out_dir / "mibig_pairs_npclassifier_labels.tsv"
    pair_labels.to_csv(pair_labels_path, sep="\t", index=False)

    report = {
        "api": NPCLASSIFIER_URL,
        "pairs_path": str(args.pairs_path),
        "labels_path": str(labels_path),
        "pair_labels_path": str(pair_labels_path),
        "cache_dir": str(args.cache_dir),
        "n_pair_rows": int(len(valid_pairs)),
        "n_unique_structures_in_pairs": int(valid_pairs["canonical_smiles"].nunique()),
        "n_unique_structures_classified_this_run": int(len(unique)),
        **stats,
    }
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
