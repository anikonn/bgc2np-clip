from __future__ import annotations

from pathlib import Path

import torch

from kiba_clip.data.datasets import CachedInteractionDataset, build_interactions
from kiba_clip.models.clip_dual import DualEncoderCLIP


def _write_tsv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_data_join_and_forward(tmp_path: Path) -> None:
    data_dir = tmp_path / "seed"
    data_dir.mkdir(parents=True)

    _write_tsv(
        data_dir / "interS.tsv",
        "Drug_ID\tTarget_ID\tY\tsplit",
        [
            "D1\tP1\t6.0\ttrain",
            "D2\tP2\t7.0\tval",
        ],
    )
    _write_tsv(
        data_dir / "lig.tsv",
        "Drug_ID\tDrug",
        ["D1\tCCO", "D2\tCCN"],
    )
    _write_tsv(
        data_dir / "prot.tsv",
        "Target_ID\tTarget",
        ["P1\tMKT", "P2\tMSS"],
    )

    interactions = build_interactions(data_dir)
    assert len(interactions) == 2

    p_cache = {"P1": torch.randn(320), "P2": torch.randn(320)}
    l_cache = {"D1": torch.randn(2048), "D2": torch.randn(2048)}
    torch.save(p_cache, data_dir / "protein_embeddings.pt")
    torch.save(l_cache, data_dir / "ligand_fingerprints.pt")

    ds = CachedInteractionDataset(
        interactions=interactions,
        protein_cache_path=data_dir / "protein_embeddings.pt",
        ligand_cache_path=data_dir / "ligand_fingerprints.pt",
        split="train",
    )
    sample = ds[0]

    model = DualEncoderCLIP(
        protein_input_dim=320,
        ligand_input_dim=2048,
        emb_dim=64,
        hidden_dim=128,
        dropout=0.1,
    )
    loss, logits = model(
        sample["protein_feature"].unsqueeze(0),
        sample["ligand_feature"].unsqueeze(0),
    )

    assert loss.item() >= 0.0
    assert logits.shape == (1, 1)
