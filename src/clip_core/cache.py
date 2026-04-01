from __future__ import annotations

from pathlib import Path

import torch


class FeatureCache:
    """Simple ID->tensor cache persisted with torch.save."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, torch.Tensor] = {}

    def add(self, item_id: str, tensor: torch.Tensor) -> None:
        self.data[item_id] = tensor.detach().cpu()

    def save(self) -> None:
        torch.save(self.data, self.path)

    @staticmethod
    def load(path: str | Path) -> dict[str, torch.Tensor]:
        return torch.load(path, map_location="cpu")
