from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray


@dataclass
class MorganConfig:
    radius: int = 2
    n_bits: int = 2048


class MorganFingerprintFeaturizer:
    """Compute Morgan fingerprints for SMILES."""

    def __init__(self, cfg: MorganConfig) -> None:
        self.cfg = cfg
        self._fpgen = AllChem.GetMorganGenerator(
            radius=self.cfg.radius,
            fpSize=self.cfg.n_bits,
        )

    def encode(self, smiles: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp = self._fpgen.GetFingerprint(mol)
        arr = np.zeros((self.cfg.n_bits,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return torch.from_numpy(arr)
