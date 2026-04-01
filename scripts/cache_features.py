from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.cache import FeatureCache
from clip_core.config import apply_overrides, load_yaml
from clip_core.logging import save_json, setup_logger
from kiba_clip.data.datasets import build_interactions
from kiba_clip.featurization.esm2 import ESM2Config, ESM2MeanPoolEmbedder
from kiba_clip.featurization.morgan import MorganConfig, MorganFingerprintFeaturizer
from kiba_clip.featurization.one_hot import ProteinOneHotConfig, ProteinOneHotEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache protein features (ESM2 or one-hot) and ligand Morgan fingerprints."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interactions = build_interactions(args.data_dir)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    protein_encoder_name = cfg["featurization"].get("protein_encoder", "esm2").lower()
    protein_batch_size = cfg["featurization"]["protein_batch_size"]
    if protein_encoder_name == "esm2":
        protein_encoder_cfg = ESM2Config(
            model_name=cfg["featurization"]["esm2_model_name"],
            max_length=cfg["featurization"]["protein_max_length"],
            batch_size=protein_batch_size,
        )
        protein_embedder = ESM2MeanPoolEmbedder(protein_encoder_cfg, device=device)
    elif protein_encoder_name == "one_hot":
        protein_encoder_cfg = ProteinOneHotConfig(
            max_length=cfg["featurization"]["protein_max_length"],
            alphabet=cfg["featurization"].get("one_hot_alphabet", "ACDEFGHIKLMNPQRSTVWYX"),
        )
        protein_embedder = ProteinOneHotEncoder(protein_encoder_cfg)
    else:
        raise ValueError(f"Unsupported protein encoder: {protein_encoder_name}")

    ligand_featurizer = MorganFingerprintFeaturizer(
        MorganConfig(radius=2, n_bits=2048)
    )

    prot_df = interactions[["Target_ID", "Target"]].drop_duplicates("Target_ID").reset_index(drop=True)
    lig_df = interactions[["Drug_ID", "Drug"]].drop_duplicates("Drug_ID").reset_index(drop=True)

    prot_cache = FeatureCache(outdir / "protein_embeddings.pt")
    for i in tqdm(range(0, len(prot_df), protein_batch_size), desc="Protein features"):
        chunk = prot_df.iloc[i : i + protein_batch_size]
        embs = protein_embedder.encode(chunk["Target"].tolist())
        for target_id, emb in zip(chunk["Target_ID"].tolist(), embs, strict=True):
            prot_cache.add(target_id, emb)
    prot_cache.save()

    lig_cache = FeatureCache(outdir / "ligand_fingerprints.pt")
    for row in tqdm(lig_df.itertuples(index=False), total=len(lig_df), desc="Ligand fingerprints"):
        try:
            fp = ligand_featurizer.encode(row.Drug)
            lig_cache.add(row.Drug_ID, fp)
        except ValueError:
            logger.warning("Skipping invalid SMILES for Drug_ID=%s", row.Drug_ID)
    lig_cache.save()

    index = {
        "protein_cache": str(outdir / "protein_embeddings.pt"),
        "ligand_cache": str(outdir / "ligand_fingerprints.pt"),
        "protein_featurizer": protein_encoder_name,
        "n_proteins": len(prot_cache.data),
        "n_ligands": len(lig_cache.data),
        "protein_dim": int(next(iter(prot_cache.data.values())).numel()),
        "ligand_dim": int(next(iter(lig_cache.data.values())).numel()),
    }
    save_json(index, outdir / "cache_index.json")

    logger.info("Saved caches to %s", outdir)


if __name__ == "__main__":
    main()
