from __future__ import annotations

import torch
from torch import nn


class ProjectionHead(nn.Module):
    """Configurable projection head for contrastive BGC-NP embeddings."""

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        head_type: str = "mlp_gelu",
    ) -> None:
        super().__init__()
        self.head_type = str(head_type)
        if self.head_type == "linear":
            self.net = nn.Sequential(nn.Linear(input_dim, emb_dim))
        elif self.head_type == "mlp_relu":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim),
            )
        elif self.head_type == "mlp_gelu":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim),
            )
        elif self.head_type == "layernorm_mlp_gelu":
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim),
            )
        else:
            raise ValueError(
                "Unknown projection head type "
                f"{self.head_type!r}; expected linear, mlp_relu, mlp_gelu, or layernorm_mlp_gelu"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
