from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from clip_core.losses import symmetric_infonce_loss
from kiba_clip.models.projection import ProjectionHead


class DualEncoderCLIP(nn.Module):
    """Dual projection-head model over cached protein/ligand features."""

    def __init__(
        self,
        protein_input_dim: int,
        ligand_input_dim: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        init_temperature: float = 0.07,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.prot_proj = ProjectionHead(protein_input_dim, emb_dim, hidden_dim, dropout)
        self.lig_proj = ProjectionHead(ligand_input_dim, emb_dim, hidden_dim, dropout)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature), dtype=torch.float32))
        self.max_logit_scale = max_logit_scale

    def encode_protein(self, protein_features: torch.Tensor) -> torch.Tensor:
        z = self.prot_proj(protein_features)
        return F.normalize(z, dim=-1)

    def encode_ligand(self, ligand_features: torch.Tensor) -> torch.Tensor:
        z = self.lig_proj(ligand_features)
        return F.normalize(z, dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_logit_scale)

    def forward(self, protein_features: torch.Tensor, ligand_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_p = self.encode_protein(protein_features)
        z_l = self.encode_ligand(ligand_features)
        scale = self.get_logit_scale()
        logits = scale * (z_p @ z_l.T)
        loss = symmetric_infonce_loss(logits)
        return loss, logits
