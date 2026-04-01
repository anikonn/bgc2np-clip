from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head."""

    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
