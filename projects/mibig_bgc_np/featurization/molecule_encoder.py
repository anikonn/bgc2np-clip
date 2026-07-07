from __future__ import annotations

from dataclasses import dataclass

import torch

from projects.mibig_bgc_np.featurization.morgan import MorganConfig, MorganFingerprintFeaturizer


@dataclass
class MorganCompoundConfig:
    radius: int = 2
    n_bits: int = 2048


class MorganCompoundEncoder:
    """Encode compounds with Morgan fingerprints."""

    def __init__(self, cfg: MorganCompoundConfig) -> None:
        self.cfg = cfg
        self.featurizer = MorganFingerprintFeaturizer(MorganConfig(radius=cfg.radius, n_bits=cfg.n_bits))

    def encode(self, molecules: list[str]) -> torch.Tensor:
        if not molecules:
            return torch.empty((0, self.cfg.n_bits), dtype=torch.float32)
        return torch.stack([self.featurizer.encode(smiles) for smiles in molecules], dim=0)


def build_molecule_encoder(cfg: dict[str, object]) -> MorganCompoundEncoder:
    encoder_name = str(cfg.get("molecule_encoder", "morgan")).lower()
    if encoder_name != "morgan":
        raise ValueError(f"Unsupported molecule encoder: {encoder_name}")
    return MorganCompoundEncoder(
        MorganCompoundConfig(
            radius=int(cfg.get("morgan_radius", 2)),
            n_bits=int(cfg.get("morgan_bits", 2048)),
        )
    )
