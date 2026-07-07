from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip_core.logging import save_json, setup_logger


LOGGER = setup_logger("make_bccoe_processed")
BGC_ACCESSION_RE = re.compile(r"^(BGC\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a BCCoE-style MIBiG processed table by keeping paired BGCs "
            "with Pfam-domain annotations and canonical RDKit SMILES."
        )
    )
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=Path("data/MIBIG/processed/mibig_pairs.tsv"),
        help="Source processed MIBiG pair table.",
    )
    parser.add_argument(
        "--proteins_path",
        type=Path,
        default=Path("data/MIBIG/processed/bgc_proteins.jsonl"),
        help="Source processed BGC protein JSONL.",
    )
    parser.add_argument(
        "--pfam_path",
        type=Path,
        default=Path("data/MIBIG/bccoe/cand_BGCs_pfams.csv"),
        help="BCCoE/Pfam table with bgc_id and bgc_pfams columns.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("cache/bccoe"),
        help="Output directory for filtered mibig_pairs.tsv, bgc_proteins.jsonl, and summary.",
    )
    parser.add_argument("--target_bgcs", type=int, default=2625)
    parser.add_argument("--target_compounds", type=int, default=3000)
    parser.add_argument("--target_pairs", type=int, default=3422)
    return parser.parse_args()


def _base_bgc_id(value: Any) -> str | None:
    if value is None:
        return None
    match = BGC_ACCESSION_RE.match(str(value).strip())
    if match is None:
        return None
    return match.group(1)


def _canonicalize_smiles(smiles: Any) -> str | None:
    text = "" if smiles is None else str(smiles).strip()
    if not text:
        return None
    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RDKit is required to canonicalize BCCoE SMILES.") from exc
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _load_pfam_bgcs(path: Path) -> pd.DataFrame:
    pfam_df = pd.read_csv(path)
    required = {"bgc_id", "bgc_pfams"}
    missing = required.difference(pfam_df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    pfam_df["bgc_id"] = pfam_df["bgc_id"].astype(str)
    pfam_df["bgc_base_id"] = pfam_df["bgc_id"].map(_base_bgc_id)
    pfam_df["bgc_pfams"] = pfam_df["bgc_pfams"].fillna("").astype(str).str.strip()
    pfam_df = pfam_df.dropna(subset=["bgc_base_id"]).copy()
    pfam_df = pfam_df[pfam_df["bgc_pfams"] != ""].copy()
    return pfam_df.drop_duplicates(subset=["bgc_base_id"]).reset_index(drop=True)


def _write_filtered_proteins(source_path: Path, out_path: Path, kept_bgcs: set[str]) -> int:
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as source, out_path.open("w", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            bgc_id = str(record.get("bgc_id", ""))
            if bgc_id not in kept_bgcs:
                continue
            json.dump(record, out)
            out.write("\n")
            count += 1
    return count


def main() -> None:
    args = _parse_args()
    if not args.pairs_path.exists():
        raise FileNotFoundError(args.pairs_path)
    if not args.proteins_path.exists():
        raise FileNotFoundError(args.proteins_path)
    if not args.pfam_path.exists():
        raise FileNotFoundError(args.pfam_path)

    pair_df = pd.read_csv(args.pairs_path, sep="\t")
    pfam_df = _load_pfam_bgcs(args.pfam_path)
    pfam_bgcs = set(pfam_df["bgc_base_id"].astype(str).tolist())

    pair_df["bgc_id"] = pair_df["bgc_id"].astype(str)
    pair_df["bgc_base_id"] = pair_df["bgc_id"].map(_base_bgc_id)
    pair_df["canonical_smiles"] = pair_df["smiles"].map(_canonicalize_smiles)

    invalid_smiles = pair_df[pair_df["canonical_smiles"].isna()].copy()
    filtered = pair_df[
        pair_df["bgc_base_id"].isin(pfam_bgcs) & pair_df["canonical_smiles"].notna()
    ].copy()
    filtered["smiles"] = filtered["canonical_smiles"].astype(str)
    filtered["compound_id"] = filtered["canonical_smiles"].astype(str)
    filtered = (
        filtered.sort_values(["bgc_id", "compound_id", "compound_idx", "compound_name"])
        .drop_duplicates(subset=["bgc_id", "compound_id"])
        .reset_index(drop=True)
    )

    output_columns = [
        "bgc_id",
        "bgc_version",
        "compound_idx",
        "compound_name",
        "smiles",
        "compound_id",
        "n_genes",
        "protein_ids",
        "protein_seqs",
        "bgc_class",
        "bgc_classes",
        "n_bgc_classes",
        "taxon_name",
    ]
    filtered = filtered[[column for column in output_columns if column in filtered.columns]].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs_out = args.out_dir / "mibig_pairs.tsv"
    proteins_out = args.out_dir / "bgc_proteins.jsonl"
    invalid_out = args.out_dir / "invalid_smiles.tsv"
    summary_out = args.out_dir / "bccoe_processed_summary.json"

    filtered.to_csv(pairs_out, sep="\t", index=False)
    if invalid_smiles.empty:
        pd.DataFrame(columns=list(pair_df.columns) + ["reason"]).to_csv(invalid_out, sep="\t", index=False)
    else:
        invalid_smiles.assign(reason="rdkit_parse_failed").to_csv(invalid_out, sep="\t", index=False)

    kept_bgcs = set(filtered["bgc_id"].astype(str).tolist())
    proteins_written = _write_filtered_proteins(args.proteins_path, proteins_out, kept_bgcs)
    source_paired_bgcs = set(pair_df["bgc_id"].astype(str).tolist())
    source_paired_bgcs_absent_from_pfam = sorted(source_paired_bgcs.difference(pfam_bgcs))

    summary = {
        "source_pairs_path": str(args.pairs_path),
        "source_proteins_path": str(args.proteins_path),
        "pfam_path": str(args.pfam_path),
        "out_dir": str(args.out_dir),
        "filter": "paired BGCs with non-empty Pfam annotations; canonical RDKit SMILES; unique bgc_id/canonical_smiles pairs",
        "source": {
            "n_pairs": int(len(pair_df)),
            "n_bgcs": int(pair_df["bgc_id"].nunique()),
            "n_compounds": int(pair_df["canonical_smiles"].nunique(dropna=True)),
            "n_invalid_smiles": int(len(invalid_smiles)),
        },
        "pfam": {
            "n_pfam_rows": int(len(pfam_df)),
            "n_pfam_bgcs": int(pfam_df["bgc_base_id"].nunique()),
            "n_source_paired_bgcs_absent_from_pfam": int(len(source_paired_bgcs_absent_from_pfam)),
            "source_paired_bgcs_absent_from_pfam_examples": source_paired_bgcs_absent_from_pfam[:20],
        },
        "output": {
            "n_pairs": int(len(filtered)),
            "n_bgcs": int(filtered["bgc_id"].nunique()),
            "n_compounds": int(filtered["compound_id"].nunique()),
            "n_protein_records": int(proteins_written),
        },
        "paper_reference_counts": {
            "n_bgcs": int(args.target_bgcs),
            "n_compounds": int(args.target_compounds),
            "n_pairs": int(args.target_pairs),
        },
        "difference_output_minus_paper": {
            "n_bgcs": int(filtered["bgc_id"].nunique()) - int(args.target_bgcs),
            "n_compounds": int(filtered["compound_id"].nunique()) - int(args.target_compounds),
            "n_pairs": int(len(filtered)) - int(args.target_pairs),
        },
        "note": (
            "The BGC count remains below the paper count because the local pair extraction has only "
            "2114 BGCs with usable paired SMILES, and this PFAM table covers 2105 of them."
        ),
    }
    save_json(summary, summary_out)
    LOGGER.info("Wrote BCCoE-style pairs to %s", pairs_out)
    LOGGER.info("Wrote BCCoE-style proteins to %s", proteins_out)
    LOGGER.info("Wrote summary to %s", summary_out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
