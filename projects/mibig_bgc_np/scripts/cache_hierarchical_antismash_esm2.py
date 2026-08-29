from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.logging import save_json, setup_logger
from projects.mibig_bgc_np.featurization.esm2 import ESM2Config, ESM2MeanPoolEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode antiSMASH domain sequences with ESM2 token-mean pooling, then "
            "mean-pool domains per CDS and CDS/proteins per BGC."
        )
    )
    parser.add_argument("--domains_path", type=Path, required=True)
    parser.add_argument("--compound_cache_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bgc_ids_tsv", type=Path, default=None)
    parser.add_argument("--bgc_id_column", default="bgc_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("cache_hierarchical_antismash_esm2")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to encode antiSMASH domains with ESM2-t30.")
    device = torch.device("cuda")
    records = [json.loads(line) for line in args.domains_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.bgc_ids_tsv is not None:
        id_table = pd.read_csv(args.bgc_ids_tsv, sep="\t")
        if args.bgc_id_column not in id_table.columns:
            raise ValueError(f"{args.bgc_ids_tsv} lacks column {args.bgc_id_column!r}")
        requested_ids = set(id_table[args.bgc_id_column].dropna().astype(str))
        records = [record for record in records if str(record["bgc_id"]) in requested_ids]
        missing = requested_ids.difference(str(record["bgc_id"]) for record in records)
        if missing:
            raise KeyError(f"Missing antiSMASH records for {len(missing)} requested BGCs; examples: {sorted(missing)[:5]}")

    sequences: list[str] = []
    owners: list[tuple[str, int]] = []
    n_truncated = 0
    for record in records:
        bgc_id = str(record["bgc_id"])
        item_sequences = [str(sequence) for sequence in record["protein_seqs"]]
        parent_indices = record.get("parent_cds_indices")
        if parent_indices is None or len(parent_indices) != len(item_sequences):
            raise ValueError(
                f"{bgc_id} lacks valid parent_cds_indices; regenerate the domain JSONL "
                "with scripts/extract_antismash_domains.py"
            )
        for sequence, parent_index in zip(item_sequences, parent_indices, strict=True):
            sequences.append(sequence)
            owners.append((bgc_id, int(parent_index)))
            n_truncated += int(len(sequence) + 2 > int(args.max_length))

    embedder = ESM2MeanPoolEmbedder(
        ESM2Config(
            model_name=str(args.model_name),
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
        ),
        device=device,
    )
    protein_sums: dict[tuple[str, int], torch.Tensor] = {}
    protein_counts: dict[tuple[str, int], int] = defaultdict(int)
    domain_embeddings_by_bgc: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in tqdm(range(0, len(sequences), int(args.batch_size)), desc="ESM2 domain batches"):
        batch_sequences = sequences[start : start + int(args.batch_size)]
        embeddings = embedder.encode(batch_sequences)
        for owner, embedding in zip(owners[start : start + len(batch_sequences)], embeddings, strict=True):
            domain_embeddings_by_bgc[owner[0]].append(embedding.float())
            if owner not in protein_sums:
                protein_sums[owner] = torch.zeros_like(embedding, dtype=torch.float32)
            protein_sums[owner] += embedding.float()
            protein_counts[owner] += 1

    proteins_by_bgc: dict[str, list[torch.Tensor]] = defaultdict(list)
    domains_per_protein: list[int] = []
    for owner, embedding_sum in protein_sums.items():
        count = protein_counts[owner]
        proteins_by_bgc[owner[0]].append(embedding_sum / float(count))
        domains_per_protein.append(count)

    bgc_cache = {
        bgc_id: torch.stack(protein_embeddings).mean(dim=0)
        for bgc_id, protein_embeddings in proteins_by_bgc.items()
    }
    compound_cache = torch.load(args.compound_cache_path, map_location="cpu", weights_only=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    domain_path = args.out_dir / "domain_features.pt"
    protein_path = args.out_dir / "protein_features.pt"
    bgc_path = args.out_dir / "bgc_features.pt"
    compound_path = args.out_dir / "compound_features.pt"
    domain_cache = {
        bgc_id: torch.stack(embeddings)
        for bgc_id, embeddings in domain_embeddings_by_bgc.items()
    }
    protein_cache = {
        bgc_id: torch.stack(protein_embeddings)
        for bgc_id, protein_embeddings in proteins_by_bgc.items()
    }
    torch.save(domain_cache, domain_path)
    torch.save(protein_cache, protein_path)
    torch.save(bgc_cache, bgc_path)
    torch.save(compound_cache, compound_path)

    protein_counts_by_bgc = [len(proteins) for proteins in proteins_by_bgc.values()]
    save_json(
        {
            "domain_cache": str(domain_path),
            "protein_cache": str(protein_path),
            "bgc_cache": str(bgc_path),
            "compound_cache": str(compound_path),
            "source_domains_path": str(args.domains_path),
            "bgc_ids_tsv": str(args.bgc_ids_tsv) if args.bgc_ids_tsv is not None else None,
            "model_name": str(args.model_name),
            "embedding_dim": int(next(iter(bgc_cache.values())).numel()),
            "max_length_including_special_tokens": int(args.max_length),
            "sequence_pooling": "mean across non-special amino-acid token embeddings",
            "protein_pooling": "mean across domain embeddings belonging to the same CDS; unsplit CDS has one item",
            "bgc_pooling": "mean across CDS/protein embeddings",
            "n_bgcs": len(bgc_cache),
            "n_domain_or_unsplit_sequences": len(sequences),
            "n_proteins": len(protein_sums),
            "n_sequences_truncated": n_truncated,
            "domains_per_protein_min": min(domains_per_protein),
            "domains_per_protein_mean": sum(domains_per_protein) / len(domains_per_protein),
            "domains_per_protein_max": max(domains_per_protein),
            "proteins_per_bgc_min": min(protein_counts_by_bgc),
            "proteins_per_bgc_mean": sum(protein_counts_by_bgc) / len(protein_counts_by_bgc),
            "proteins_per_bgc_max": max(protein_counts_by_bgc),
            "compound_dim": int(next(iter(compound_cache.values())).numel()),
            "n_compounds": len(compound_cache),
        },
        args.out_dir / "cache_index.json",
    )
    logger.info("Saved hierarchical ESM2 features for %d BGCs to %s", len(bgc_cache), args.out_dir)


if __name__ == "__main__":
    main()
