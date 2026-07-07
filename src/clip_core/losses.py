from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_infonce_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE over pairwise logits."""
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_left = F.cross_entropy(logits, targets)
    loss_right = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_left + loss_right)


def multi_positive_infonce_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE where each row/column may have multiple positives."""
    if logits.shape != positive_mask.shape:
        raise ValueError(
            f"logits and positive_mask must have the same shape, got {logits.shape} and {positive_mask.shape}"
        )
    if positive_mask.dtype != torch.bool:
        positive_mask = positive_mask.bool()
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every left item must have at least one positive in the batch.")
    if not bool(positive_mask.any(dim=0).all()):
        raise ValueError("Every right item must have at least one positive in the batch.")

    neg_inf = torch.finfo(logits.dtype).min
    log_probs_left = F.log_softmax(logits, dim=1)
    loss_left = -torch.logsumexp(log_probs_left.masked_fill(~positive_mask, neg_inf), dim=1)

    positive_mask_t = positive_mask.t()
    log_probs_right = F.log_softmax(logits.t(), dim=1)
    loss_right = -torch.logsumexp(log_probs_right.masked_fill(~positive_mask_t, neg_inf), dim=1)
    return 0.5 * (loss_left.mean() + loss_right.mean())
