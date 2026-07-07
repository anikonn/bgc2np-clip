from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


FASTA_HEADER_RE = re.compile(r"^(BGC\d+)\.(\d+)\|")
BGC_RE = re.compile(r"^(BGC\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BCCoE-style MIBiG BGC/product count losses.")
    parser.add_argument("--json_dir", type=Path, default=Path("data/MIBIG/mibig_json_4.0"))
    parser.add_argument("--fasta_path", type=Path, default=Path("data/MIBIG/mibig_prot_seqs_4.0.fasta"))
    parser.add_argument("--pfam_path", type=Path, default=Path("data/MIBIG/bccoe/cand_BGCs_pfams.csv"))
    parser.add_argument("--pairs_path", type=Path, default=Path("cache/bccoe/mibig_pairs.tsv"))
    parser.add_argument("--out_dir", type=Path, default=Path("cache/bccoe/diagnostics"))
    return parser.parse_args()


def _base_bgc_id(value: Any) -> str | None:
    match = BGC_RE.match("" if value is None else str(value).strip())
    return match.group(1) if match else None


def _canonicalize(smiles: Any) -> str | None:
    text = "" if smiles is None else str(smiles).strip()
    if not text:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _load_fasta_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    current: tuple[str, int] | None = None
    seq_parts: list[str] = []

    def flush() -> None:
        nonlocal current, seq_parts
        if current is None:
            return
        bgc_id, version = current
        entry = records.setdefault(bgc_id, {"bgc_id": bgc_id, "version": version, "n_proteins": 0})
        if int(entry["version"]) != int(version):
            raise ValueError(f"Conflicting FASTA versions for {bgc_id}: {entry['version']} and {version}")
        if seq_parts:
            entry["n_proteins"] += 1
        current = None
        seq_parts = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                match = FASTA_HEADER_RE.match(line[1:])
                if match is None:
                    raise ValueError(f"Unrecognized FASTA header: {line[:120]}")
                current = (match.group(1), int(match.group(2)))
                continue
            seq_parts.append(line)
    flush()
    return records


def _load_json_index(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for json_path in sorted(path.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        bgc_id = payload.get("accession")
        if not isinstance(bgc_id, str):
            continue
        compounds = payload.get("compounds", [])
        if not isinstance(compounds, list):
            compounds = []
        n_compounds = 0
        n_nonempty_structures = 0
        n_valid_structures = 0
        canonical_smiles: set[str] = set()
        for compound in compounds:
            if not isinstance(compound, dict):
                continue
            n_compounds += 1
            structure = compound.get("structure")
            if not isinstance(structure, str) or not structure.strip():
                continue
            n_nonempty_structures += 1
            canonical = _canonicalize(structure)
            if canonical is None:
                continue
            n_valid_structures += 1
            canonical_smiles.add(canonical)
        records[bgc_id] = {
            "bgc_id": bgc_id,
            "version": payload.get("version"),
            "status": payload.get("status"),
            "quality": payload.get("quality"),
            "n_compounds": n_compounds,
            "n_nonempty_structures": n_nonempty_structures,
            "n_valid_structures": n_valid_structures,
            "n_unique_valid_structures": len(canonical_smiles),
        }
    return records


def _load_pfam_ids(path: Path) -> set[str]:
    df = pd.read_csv(path)
    if "bgc_id" not in df.columns or "bgc_pfams" not in df.columns:
        raise ValueError(f"{path} must have bgc_id and bgc_pfams columns")
    df["base_id"] = df["bgc_id"].map(_base_bgc_id)
    df["bgc_pfams"] = df["bgc_pfams"].fillna("").astype(str).str.strip()
    return set(df.loc[df["base_id"].notna() & (df["bgc_pfams"] != ""), "base_id"].astype(str))


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fasta = _load_fasta_index(args.fasta_path)
    json_records = _load_json_index(args.json_dir)
    pfam_ids = _load_pfam_ids(args.pfam_path)

    rows: list[dict[str, Any]] = []
    all_ids = sorted(set(json_records) | set(fasta) | pfam_ids)
    for bgc_id in all_ids:
        json_entry = json_records.get(bgc_id, {})
        fasta_entry = fasta.get(bgc_id, {})
        status = json_entry.get("status")
        json_version = json_entry.get("version")
        fasta_version = fasta_entry.get("version")
        has_json = bgc_id in json_records
        has_fasta = bgc_id in fasta
        has_pfam = bgc_id in pfam_ids
        active = status == "active"
        version_match = has_json and has_fasta and int(json_version) == int(fasta_version)
        valid_smiles = int(json_entry.get("n_valid_structures", 0) or 0) > 0
        too_many_proteins = int(fasta_entry.get("n_proteins", 0) or 0) > 3000

        reason = "kept_candidate"
        if not has_pfam:
            reason = "not_in_pfam_table"
        elif not has_json:
            reason = "missing_json"
        elif not has_fasta:
            reason = "missing_fasta"
        elif not active:
            reason = "not_active"
        elif not version_match:
            reason = "version_mismatch"
        elif too_many_proteins:
            reason = "too_many_proteins"
        elif not valid_smiles:
            reason = "no_valid_smiles_structure"

        rows.append(
            {
                "bgc_id": bgc_id,
                "reason": reason,
                "has_json": has_json,
                "has_fasta": has_fasta,
                "has_pfam": has_pfam,
                "status": status,
                "quality": json_entry.get("quality"),
                "json_version": json_version,
                "fasta_version": fasta_version,
                "version_match": version_match,
                "n_proteins": fasta_entry.get("n_proteins"),
                "n_compounds": json_entry.get("n_compounds", 0),
                "n_nonempty_structures": json_entry.get("n_nonempty_structures", 0),
                "n_valid_structures": json_entry.get("n_valid_structures", 0),
                "n_unique_valid_structures": json_entry.get("n_unique_valid_structures", 0),
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_path = args.out_dir / "bgc_count_audit.tsv"
    audit_df.to_csv(audit_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    pairs = pd.read_csv(args.pairs_path, sep="\t") if args.pairs_path.exists() else pd.DataFrame()
    output_bgcs = set(pairs["bgc_id"].astype(str)) if "bgc_id" in pairs.columns else set()
    output_compounds = set(pairs["compound_id"].astype(str)) if "compound_id" in pairs.columns else set()

    def count(mask: pd.Series) -> int:
        return int(mask.sum())

    has_json = audit_df["has_json"] == True
    has_fasta = audit_df["has_fasta"] == True
    has_pfam = audit_df["has_pfam"] == True
    active = audit_df["status"] == "active"
    version_match = audit_df["version_match"] == True
    has_valid_smiles = audit_df["n_valid_structures"].fillna(0).astype(int) > 0
    protein_ok = audit_df["n_proteins"].fillna(0).astype(int) <= 3000

    summary = {
        "inputs": {
            "json_dir": str(args.json_dir),
            "fasta_path": str(args.fasta_path),
            "pfam_path": str(args.pfam_path),
            "pairs_path": str(args.pairs_path),
        },
        "raw_counts": {
            "json_bgcs": int(len(json_records)),
            "json_active_bgcs": int(sum(1 for item in json_records.values() if item.get("status") == "active")),
            "fasta_bgcs": int(len(fasta)),
            "pfam_bgcs": int(len(pfam_ids)),
        },
        "stage_counts": {
            "json_and_fasta": count(has_json & has_fasta),
            "json_and_fasta_and_pfam": count(has_json & has_fasta & has_pfam),
            "active_json_and_fasta_and_pfam": count(has_json & has_fasta & has_pfam & active),
            "active_version_matched_json_fasta_pfam": count(has_json & has_fasta & has_pfam & active & version_match),
            "active_version_matched_json_fasta_pfam_protein_ok": count(
                has_json & has_fasta & has_pfam & active & version_match & protein_ok
            ),
            "active_version_matched_json_fasta_pfam_protein_ok_with_valid_smiles": count(
                has_json & has_fasta & has_pfam & active & version_match & protein_ok & has_valid_smiles
            ),
        },
        "reason_counts_among_pfam_bgcs": dict(
            Counter(audit_df.loc[has_pfam, "reason"].astype(str).tolist()).most_common()
        ),
        "output_pairs_table": {
            "n_pairs": int(len(pairs)),
            "n_bgcs": int(len(output_bgcs)),
            "n_compounds": int(len(output_compounds)),
        },
        "audit_path": str(audit_path),
    }

    summary_path = args.out_dir / "bgc_count_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
