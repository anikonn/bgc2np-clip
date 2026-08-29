from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from clip_core.logging import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ordered variable-length domain tensors for BGC aggregation.")
    parser.add_argument("--domain_features", type=Path, required=True)
    parser.add_argument("--domain_metadata", type=Path, required=True)
    parser.add_argument("--compound_features", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_cache = torch.load(args.domain_features, map_location="cpu", weights_only=True)
    records = [json.loads(line) for line in args.domain_metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    protein_positions: dict[str, torch.Tensor] = {}
    domain_positions: dict[str, torch.Tensor] = {}
    for record in records:
        bgc_id = str(record["bgc_id"])
        if bgc_id not in domain_cache:
            continue
        parents = [int(value) for value in record["parent_cds_indices"]]
        if len(parents) != int(domain_cache[bgc_id].shape[0]):
            raise ValueError(f"Domain metadata/features disagree for {bgc_id}")
        within_protein: defaultdict[int, int] = defaultdict(int)
        domain_indices: list[int] = []
        for parent in parents:
            within_protein[parent] += 1
            domain_indices.append(within_protein[parent])
        # Zero is reserved for padding in both positional embedding tables.
        protein_positions[bgc_id] = torch.tensor([value + 1 for value in parents], dtype=torch.long)
        domain_positions[bgc_id] = torch.tensor(domain_indices, dtype=torch.long)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(domain_cache, args.out_dir / "bgc_features.pt")
    torch.save(torch.load(args.compound_features, map_location="cpu", weights_only=True), args.out_dir / "compound_features.pt")
    torch.save(protein_positions, args.out_dir / "protein_positions.pt")
    torch.save(domain_positions, args.out_dir / "domain_positions.pt")
    example = next(iter(domain_cache.values()))
    save_json(
        {
            "representation": "ordered frozen ESM2 domain embeddings",
            "bgc_cache": str(args.out_dir / "bgc_features.pt"),
            "compound_cache": str(args.out_dir / "compound_features.pt"),
            "protein_positions": str(args.out_dir / "protein_positions.pt"),
            "domain_positions": str(args.out_dir / "domain_positions.pt"),
            "n_bgcs": len(domain_cache),
            "domain_dim": int(example.shape[-1]),
            "max_domains": max(int(value.shape[0]) for value in domain_cache.values()),
        },
        args.out_dir / "cache_index.json",
    )


if __name__ == "__main__":
    main()
