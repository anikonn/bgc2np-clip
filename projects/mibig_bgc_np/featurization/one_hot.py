from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProteinOneHotConfig:
    max_length: int = 1024
    alphabet: str = "ACDEFGHIKLMNPQRSTVWYX"


class ProteinOneHotEncoder:
    """Encode protein sequences as flattened fixed-length one-hot vectors."""

    def __init__(self, cfg: ProteinOneHotConfig) -> None:
        self.cfg = cfg
        self.alphabet = cfg.alphabet
        self.unknown_index = self.alphabet.index("X")
        self.token_to_index = {token: idx for idx, token in enumerate(self.alphabet)}

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        seq = (sequence or "").upper()
        output = torch.zeros((self.cfg.max_length, len(self.alphabet)), dtype=torch.float32)
        for pos, residue in enumerate(seq[: self.cfg.max_length]):
            token_idx = self.token_to_index.get(residue, self.unknown_index)
            output[pos, token_idx] = 1.0
        return output.flatten()

    def encode(self, sequences: list[str]) -> torch.Tensor:
        if not sequences:
            return torch.empty((0, self.cfg.max_length * len(self.alphabet)), dtype=torch.float32)
        return torch.stack([self._encode_sequence(sequence) for sequence in sequences], dim=0)
