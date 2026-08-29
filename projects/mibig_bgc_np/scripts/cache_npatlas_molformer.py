from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from projects.mibig_bgc_np.featurization.molecule_encoder import (
    MolFormerCompoundConfig,
    MolFormerCompoundEncoder,
)
from projects.mibig_bgc_np.scripts.eval_retrieval import _prepare_npatlas_candidates, _require_rdkit


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen MolFormer features for every valid NPAtlas molecule")
    parser.add_argument("--npatlas", type=Path, default=Path("data/NPAtlas_download_2024_09.tsv"))
    parser.add_argument("--outdir", type=Path, default=Path("cache/npatlas_molformer"))
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NPAtlas MolFormer caching requires a CUDA GPU")
    args.outdir.mkdir(parents=True, exist_ok=True)
    candidates = _prepare_npatlas_candidates(args.npatlas, _require_rdkit())
    unique_smiles = candidates["canonical_smiles"].drop_duplicates().astype(str).tolist()
    encoder = MolFormerCompoundEncoder(MolFormerCompoundConfig(), torch.device("cuda"))
    features: dict[str, torch.Tensor] = {}
    for start in tqdm(range(0, len(unique_smiles), args.batch_size), desc="NPAtlas MolFormer"):
        smiles = unique_smiles[start : start + args.batch_size]
        encoded = encoder.encode(smiles)
        features.update({key: value for key, value in zip(smiles, encoded, strict=True)})
    torch.save(features, args.outdir / "compound_features.pt")
    candidates.to_csv(args.outdir / "candidates.tsv", sep="\t", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps({
        "source": str(args.npatlas), "encoder": encoder.cfg.model_name,
        "pooling": "pooler_output (masked token mean)", "n_rows": int(len(candidates)),
        "n_unique_canonical_smiles": int(len(features)), "feature_dim": 768,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
