from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from projects.mibig_bgc_np.featurization.esm2 import ESM2CLSProteinEmbedder, ESM2Config, ESM2MeanPoolEmbedder
from projects.mibig_bgc_np.featurization.one_hot import ProteinOneHotConfig, ProteinOneHotEncoder

@dataclass
class BGCOneHotConfig:
    max_length: int = 1024
    alphabet: str = "ACDEFGHIKLMNPQRSTVWYX"


class BGCOneHotEncoder:
    """Encode each protein with positional one-hot features, then mean-pool within each BGC."""

    def __init__(self, cfg: BGCOneHotConfig) -> None:
        self.cfg = cfg
        self.encoder = ProteinOneHotEncoder(
            ProteinOneHotConfig(max_length=cfg.max_length, alphabet=cfg.alphabet)
        )

    def encode_proteins(self, protein_sequences: Sequence[str]) -> torch.Tensor:
        if not protein_sequences:
            return torch.empty((0, self.cfg.max_length * len(self.cfg.alphabet)), dtype=torch.float32)
        return self.encoder.encode(list(protein_sequences))

    def encode_bgcs(self, bgc_records: Sequence[Sequence[str]]) -> torch.Tensor:
        if not bgc_records:
            return torch.empty((0, self.cfg.max_length * len(self.cfg.alphabet)), dtype=torch.float32)

        bgc_features: list[torch.Tensor] = []
        feature_dim = self.cfg.max_length * len(self.cfg.alphabet)
        for protein_sequences in bgc_records:
            protein_embs = self.encode_proteins(protein_sequences)
            if protein_embs.numel() == 0:
                bgc_features.append(torch.zeros(feature_dim, dtype=torch.float32))
            else:
                bgc_features.append(protein_embs.mean(dim=0))
        return torch.stack(bgc_features, dim=0)


@dataclass
class ESM2BGCConfig:
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    max_length: int = 1024
    batch_size: int = 8
    pooling: str = "mean"


class ESM2BGCEncoder:
    """Encode proteins with ESM2 and mean-pool the per-protein embeddings within each BGC."""

    def __init__(self, cfg: ESM2BGCConfig, device: torch.device) -> None:
        self.cfg = cfg
        embedder_cfg = ESM2Config(model_name=cfg.model_name, max_length=cfg.max_length, batch_size=cfg.batch_size)
        if cfg.pooling == "mean":
            self.embedder = ESM2MeanPoolEmbedder(embedder_cfg, device=device)
        elif cfg.pooling == "cls":
            self.embedder = ESM2CLSProteinEmbedder(embedder_cfg, device=device)
        else:
            raise ValueError(f"Unsupported ESM2 pooling mode: {cfg.pooling}")

    def encode_proteins(self, protein_sequences: Sequence[str]) -> torch.Tensor:
        if not protein_sequences:
            return torch.empty((0, 0), dtype=torch.float32)
        return self.embedder.encode(list(protein_sequences))

    def encode_bgcs(self, bgc_records: Sequence[Sequence[str]]) -> torch.Tensor:
        if not bgc_records:
            return torch.empty((0, 0), dtype=torch.float32)

        owner_indices: list[int] = []
        flat_sequences: list[str] = []
        for bgc_idx, protein_sequences in enumerate(bgc_records):
            for sequence in protein_sequences:
                owner_indices.append(bgc_idx)
                flat_sequences.append(sequence)

        if not flat_sequences:
            raise ValueError("Cannot encode BGC records because no protein sequences were provided.")

        sums: torch.Tensor | None = None
        counts = torch.zeros(len(bgc_records), dtype=torch.float32)
        for start in range(0, len(flat_sequences), self.cfg.batch_size):
            chunk_sequences = flat_sequences[start : start + self.cfg.batch_size]
            chunk_owner_indices = owner_indices[start : start + self.cfg.batch_size]
            protein_embs = self.embedder.encode(chunk_sequences)
            if sums is None:
                emb_dim = int(protein_embs.size(1))
                sums = torch.zeros((len(bgc_records), emb_dim), dtype=torch.float32)
            for owner_idx, emb in zip(chunk_owner_indices, protein_embs, strict=True):
                sums[owner_idx] += emb
                counts[owner_idx] += 1.0

        assert sums is not None
        counts = counts.clamp(min=1.0).unsqueeze(1)
        return sums / counts


def build_bgc_encoder(
    cfg: dict[str, object], device: torch.device
) -> BGCOneHotEncoder | ESM2BGCEncoder:
    encoder_name = str(cfg.get("bgc_encoder", "ohe")).lower()
    if encoder_name in {"ohe", "one_hot"}:
        return BGCOneHotEncoder(
            BGCOneHotConfig(
                max_length=int(cfg.get("protein_max_length", 1024)),
                alphabet=str(cfg.get("one_hot_alphabet", "ACDEFGHIKLMNPQRSTVWYX")),
            )
        )
    if encoder_name in {"esm2_mean", "esm2"}:
        return ESM2BGCEncoder(
            ESM2BGCConfig(
                model_name=str(cfg.get("esm2_model_name", "facebook/esm2_t6_8M_UR50D")),
                max_length=int(cfg.get("protein_max_length", 1024)),
                batch_size=int(cfg.get("protein_batch_size", 8)),
                pooling="mean",
            ),
            device=device,
        )
    if encoder_name == "esm2_cls":
        return ESM2BGCEncoder(
            ESM2BGCConfig(
                model_name=str(cfg.get("esm2_model_name", "facebook/esm2_t6_8M_UR50D")),
                max_length=int(cfg.get("protein_max_length", 1024)),
                batch_size=int(cfg.get("protein_batch_size", 8)),
                pooling="cls",
            ),
            device=device,
        )
    raise ValueError(f"Unsupported BGC encoder: {encoder_name}")
