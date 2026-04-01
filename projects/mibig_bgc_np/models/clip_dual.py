from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from clip_core.losses import symmetric_infonce_loss
from kiba_clip.models.projection import ProjectionHead


class DualEncoderCLIP(nn.Module):
    """Dual projection-head model over cached BGC and compound features."""

    def __init__(
        self,
        bgc_input_dim: int,
        compound_input_dim: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        init_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.bgc_proj = ProjectionHead(bgc_input_dim, emb_dim, hidden_dim, dropout)
        self.compound_proj = ProjectionHead(compound_input_dim, emb_dim, hidden_dim, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32))
        self.max_logit_scale = max_logit_scale

    def encode_bgc(self, bgc_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.bgc_proj(bgc_features), dim=-1)

    def encode_compound(self, compound_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.compound_proj(compound_features), dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_logit_scale)

    def forward(self, bgc_features: torch.Tensor, compound_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_bgc = self.encode_bgc(bgc_features)
        z_cmp = self.encode_compound(compound_features)
        logits = self.get_logit_scale() * (z_bgc @ z_cmp.t())
        return symmetric_infonce_loss(logits), logits
