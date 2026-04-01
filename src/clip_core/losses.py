from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_infonce_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE over pairwise logits."""
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_left = F.cross_entropy(logits, targets)
    loss_right = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_left + loss_right)
