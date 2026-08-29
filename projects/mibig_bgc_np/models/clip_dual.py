from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from clip_core.losses import multi_positive_infonce_loss, symmetric_infonce_loss
from projects.mibig_bgc_np.models.projection import ProjectionHead
from projects.mibig_bgc_np.models.bgc_aggregation import build_bgc_aggregator


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
        bgc_aggregation: str = "prepooled",
        bgc_aggregation_config: dict[str, object] | None = None,
        projection_head: str = "mlp_gelu",
    ) -> None:
        super().__init__()
        self.bgc_aggregation_mode = bgc_aggregation
        self.bgc_aggregator = build_bgc_aggregator(
            bgc_aggregation, bgc_input_dim, bgc_aggregation_config or {}
        )
        aggregated_dim = bgc_input_dim if self.bgc_aggregator is None else int(self.bgc_aggregator.output_dim)
        self.bgc_proj = ProjectionHead(aggregated_dim, emb_dim, hidden_dim, dropout, projection_head)
        self.compound_proj = ProjectionHead(compound_input_dim, emb_dim, hidden_dim, dropout, projection_head)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32))
        self.max_logit_scale = max_logit_scale

    def aggregate_bgc(
        self,
        bgc_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        protein_positions: torch.Tensor | None = None,
        domain_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.bgc_aggregator is None:
            if bgc_features.ndim != 2:
                raise ValueError(f"Pre-pooled BGC features must be 2D, got {tuple(bgc_features.shape)}")
            return bgc_features, None
        return self.bgc_aggregator(
            bgc_features,
            padding_mask=padding_mask,
            protein_positions=protein_positions,
            domain_positions=domain_positions,
        )

    def encode_bgc(self, bgc_features: torch.Tensor, padding_mask: torch.Tensor | None = None, **positions) -> torch.Tensor:
        pooled, _ = self.aggregate_bgc(bgc_features, padding_mask=padding_mask, **positions)
        return F.normalize(self.bgc_proj(pooled), dim=-1)

    def encode_bgc_with_weights(self, bgc_features: torch.Tensor, padding_mask: torch.Tensor | None = None, **positions):
        pooled, weights = self.aggregate_bgc(bgc_features, padding_mask=padding_mask, **positions)
        return F.normalize(self.bgc_proj(pooled), dim=-1), weights

    def encode_compound(self, compound_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.compound_proj(compound_features), dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_logit_scale)

    def forward(
        self,
        bgc_features: torch.Tensor,
        compound_features: torch.Tensor,
        positive_mask: torch.Tensor | None = None,
        bgc_padding_mask: torch.Tensor | None = None,
        protein_positions: torch.Tensor | None = None,
        domain_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_bgc = self.encode_bgc(
            bgc_features,
            padding_mask=bgc_padding_mask,
            protein_positions=protein_positions,
            domain_positions=domain_positions,
        )
        z_cmp = self.encode_compound(compound_features)
        logits = self.get_logit_scale() * (z_bgc @ z_cmp.t())
        if positive_mask is None:
            loss = symmetric_infonce_loss(logits)
        else:
            loss = multi_positive_infonce_loss(logits, positive_mask)
        return loss, logits
