from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """MLP regressor over interaction features built from paired embeddings."""

    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        in_dim = emb_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_p: torch.Tensor, z_l: torch.Tensor | None = None) -> torch.Tensor:
        if z_l is None:
            x = z_p
        else:
            x = torch.cat([z_p, z_l, z_p * z_l, torch.abs(z_p - z_l)], dim=-1)
        return self.net(x).squeeze(-1)
