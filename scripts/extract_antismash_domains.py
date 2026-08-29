from __future__ import annotations

import argparse
import json
import re
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

try:
    from scripts._bootstrap import ensure_src_path
except ModuleNotFoundError:  # Support direct execution as python scripts/extract_antismash_domains.py.
    from _bootstrap import ensure_src_path

ensure_src_path()

from clip_core.logging import save_json, setup_logger


LOGGER = setup_logger("extract_antismash_domains")
FEATURE_RE = re.compile(r"^     (\S+)\s+(.+)$")
QUALIFIER_RE = re.compile(r"^                     /(\w+)(?:=(.*))?$")
COORDINATE_RE = re.compile(r"\d+")
VALID_AA_RE = re.compile(r"^[A-Z*]+$")


@dataclass
class GenBankFeature:
    feature_type: str
    location: str
    qualifiers: dict[str, str] = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int]:
        coordinates = [int(value) for value in COORDINATE_RE.findall(self.location)]
        if not coordinates:
            raise ValueError(f"Feature has no coordinates: {self.feature_type} {self.location}")
        return min(coordinates), max(coordinates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract antiSMASH aSDomain translations, retaining a complete CDS translation "
            "as one unsplit item when the CDS contains no aSDomain feature."
        )
    )
    parser.add_argument("--gbk_dir", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--required_bgc_ids_tsv", type=Path, default=None)
    parser.add_argument("--fallback_proteins_jsonl", type=Path, default=None)
    return parser.parse_args()


def _clean_quoted_value(value: str) -> str:
    return value.strip().strip('"').replace(" ", "")


def parse_genbank_features(path: Path) -> list[GenBankFeature]:
    """Parse the feature table fields needed here without requiring Biopython."""
    features: list[GenBankFeature] = []
    current: GenBankFeature | None = None
    current_qualifier: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("ORIGIN"):
                break
            feature_match = FEATURE_RE.match(line)
            if feature_match is not None:
                current = GenBankFeature(
                    feature_type=feature_match.group(1),
                    location=feature_match.group(2).strip(),
                )
                features.append(current)
                current_qualifier = None
                continue
            if current is None:
                continue
            qualifier_match = QUALIFIER_RE.match(line)
            if qualifier_match is not None:
                current_qualifier = qualifier_match.group(1)
                current.qualifiers[current_qualifier] = (qualifier_match.group(2) or "").strip()
                continue
            if current_qualifier is not None and line.startswith("                     "):
                current.qualifiers[current_qualifier] += line.strip()
    return features


def extract_domain_record(path: Path) -> dict[str, object]:
    features = parse_genbank_features(path)
    cds_indices = [index for index, feature in enumerate(features) if feature.feature_type == "CDS"]
    cds_features = [features[index] for index in cds_indices]
    domain_features = [feature for feature in features if feature.feature_type == "aSDomain"]

    item_ids: list[str] = []
    item_names: list[str] = []
    item_seqs: list[str] = []
    item_sources: list[str] = []
    parent_cds_indices: list[int] = []
    item_genomic_locations: list[str] = []
    item_protein_starts: list[int] = []
    item_protein_ends: list[int] = []
    parent_cds_locations: list[str] = []
    split_cds_count = 0
    unsplit_cds_count = 0

    for cds_index, (feature_index, cds) in enumerate(zip(cds_indices, cds_features, strict=True)):
        next_cds_index = (
            cds_indices[cds_index + 1]
            if cds_index + 1 < len(cds_indices)
            else len(features)
        )
        # antiSMASH emits child aSDomain features after their parent CDS and before
        # the next CDS. This avoids ambiguous coordinate containment for overlapping
        # joined CDS features, especially in fungal records.
        contained_domains = [
            domain
            for domain in features[feature_index + 1 : next_cds_index]
            if domain.feature_type == "aSDomain"
        ]
        if contained_domains:
            split_cds_count += 1
            emitted_features = contained_domains
            source = "antismash_domain"
        else:
            unsplit_cds_count += 1
            emitted_features = [cds]
            source = "unsplit_cds"

        for item_index, feature in enumerate(emitted_features):
            sequence = _clean_quoted_value(feature.qualifiers.get("translation", ""))
            if not sequence:
                raise ValueError(
                    f"{path}: {feature.feature_type} {feature.location} has no translation qualifier"
                )
            if VALID_AA_RE.fullmatch(sequence) is None:
                raise ValueError(
                    f"{path}: invalid characters in translation for {feature.feature_type} {feature.location}"
                )
            fallback_id = f"{path.stem}:cds{cds_index + 1}:item{item_index + 1}"
            item_ids.append(
                _clean_quoted_value(
                    feature.qualifiers.get(
                        "domain_id",
                        cds.qualifiers.get("protein_id", fallback_id),
                    )
                )
            )
            item_names.append(
                _clean_quoted_value(
                    feature.qualifiers.get(
                        "aSDomain",
                        cds.qualifiers.get("protein_id", "unsplit_CDS"),
                    )
                )
            )
            item_seqs.append(sequence.rstrip("*"))
            item_sources.append(source)
            parent_cds_indices.append(cds_index)
            item_genomic_locations.append(feature.location)
            if feature.feature_type == "aSDomain":
                item_protein_starts.append(int(_clean_quoted_value(feature.qualifiers["protein_start"])))
                item_protein_ends.append(int(_clean_quoted_value(feature.qualifiers["protein_end"])))
            else:
                item_protein_starts.append(0)
                item_protein_ends.append(len(sequence.rstrip("*")))
            parent_cds_locations.append(cds.location)

    return {
        "bgc_id": path.stem,
        "protein_ids": item_ids,
        "protein_seqs": item_seqs,
        "domain_names": item_names,
        "sequence_sources": item_sources,
        "parent_cds_indices": parent_cds_indices,
        "item_genomic_locations": item_genomic_locations,
        "item_protein_starts": item_protein_starts,
        "item_protein_ends": item_protein_ends,
        "parent_cds_locations": parent_cds_locations,
        "n_cds": len(cds_features),
        "n_split_cds": split_cds_count,
        "n_unsplit_cds": unsplit_cds_count,
        "n_antismash_domains": len(domain_features),
        "n_emitted_sequences": len(item_seqs),
    }


def main() -> None:
    args = parse_args()
    # Base BGC files are canonical here. Region files overlap them and would duplicate features.
    gbk_paths = sorted(
        path
        for path in args.gbk_dir.glob("BGC*.gbk")
        if ".region" not in path.name
    )
    if not gbk_paths:
        raise FileNotFoundError(f"No base BGC*.gbk files found under {args.gbk_dir}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {
        "n_bgcs": 0,
        "n_cds": 0,
        "n_split_cds": 0,
        "n_unsplit_cds": 0,
        "n_antismash_domains": 0,
        "n_emitted_sequences": 0,
    }
    records: list[dict[str, object]] = []
    with args.out_path.open("w", encoding="utf-8") as handle:
        for path in gbk_paths:
            record = extract_domain_record(path)
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            totals["n_bgcs"] += 1
            for key in totals:
                if key != "n_bgcs":
                    totals[key] += int(record[key])

        if args.required_bgc_ids_tsv is not None:
            if args.fallback_proteins_jsonl is None:
                raise ValueError("--fallback_proteins_jsonl is required with --required_bgc_ids_tsv")
            required = set(pd.read_csv(args.required_bgc_ids_tsv, sep="\t")["bgc_id"].astype(str))
            present = {str(record["bgc_id"]) for record in records}
            missing = required - present
            fallback_index = {
                str(record["bgc_id"]): record
                for line in args.fallback_proteins_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip() and (record := json.loads(line))
            }
            unresolved = missing - set(fallback_index)
            if unresolved:
                raise KeyError(f"No antiSMASH or fallback protein records for: {sorted(unresolved)[:5]}")
            for bgc_id in sorted(missing):
                source = fallback_index[bgc_id]
                sequences = [str(sequence).rstrip("*") for sequence in source["protein_seqs"]]
                ids = [str(value) for value in source.get("protein_ids", [])]
                if len(ids) != len(sequences):
                    ids = [f"{bgc_id}:protein{index + 1}" for index in range(len(sequences))]
                record = {
                    "bgc_id": bgc_id,
                    "protein_ids": ids,
                    "protein_seqs": sequences,
                    "domain_names": ["unsplit_protein_fallback"] * len(sequences),
                    "sequence_sources": ["unsplit_protein_fallback"] * len(sequences),
                    "parent_cds_indices": list(range(len(sequences))),
                    "item_genomic_locations": ["unknown"] * len(sequences),
                    "item_protein_starts": [0] * len(sequences),
                    "item_protein_ends": [len(sequence) for sequence in sequences],
                    "parent_cds_locations": ["unknown"] * len(sequences),
                    "n_cds": len(sequences),
                    "n_split_cds": 0,
                    "n_unsplit_cds": len(sequences),
                    "n_antismash_domains": 0,
                    "n_emitted_sequences": len(sequences),
                }
                handle.write(json.dumps(record) + "\n")
                totals["n_bgcs"] += 1
                totals["n_cds"] += len(sequences)
                totals["n_unsplit_cds"] += len(sequences)
                totals["n_emitted_sequences"] += len(sequences)

    summary_path = args.summary_path or args.out_path.with_suffix(".summary.json")
    summary = {
        "gbk_dir": str(args.gbk_dir),
        "out_path": str(args.out_path),
        "fallback_rule": "BGCs lacking a base antiSMASH GBK retain each complete protein as one unsplit item.",
        "rule": (
            "For each CDS, emit every contained aSDomain translation; "
            "if none exists, emit the complete CDS translation once."
        ),
        **totals,
    }
    save_json(summary, summary_path)
    LOGGER.info("Extracted %d sequences from %d BGCs", totals["n_emitted_sequences"], totals["n_bgcs"])
    LOGGER.info("Wrote domain records to %s", args.out_path)


if __name__ == "__main__":
    main()
