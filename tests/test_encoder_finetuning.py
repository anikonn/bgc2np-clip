from __future__ import annotations

import torch
from torch import nn

from projects.mibig_bgc_np.models.online_finetune_clip import configure_encoder_trainability
from projects.mibig_bgc_np.scripts.run_encoder_finetuning import _queued_symmetric_loss


class DummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(3, 3)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(3, 3) for _ in range(6)])
        self.layer_norm = nn.LayerNorm(3)


def test_frozen_encoder_has_no_trainable_parameters() -> None:
    model = DummyEncoder()
    status = configure_encoder_trainability(model, 0)
    assert status["n_layers"] == 6
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_only_last_n_blocks_are_unfrozen() -> None:
    model = DummyEncoder()
    configure_encoder_trainability(model, 2)
    assert not any(parameter.requires_grad for parameter in model.encoder.layer[3].parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.layer[4].parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.layer[5].parameters())
    assert not any(parameter.requires_grad for parameter in model.layer_norm.parameters())
    assert not any(parameter.requires_grad for parameter in model.embedding.parameters())


def test_full_encoder_is_unfrozen() -> None:
    model = DummyEncoder()
    status = configure_encoder_trainability(model, "full")
    assert status["mode"] == "full"
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_cross_batch_queue_loss_is_finite_and_differentiable() -> None:
    class Scale(nn.Module):
        def scale(self):
            return torch.tensor(2.0)

    zb = nn.functional.normalize(torch.randn(2, 4, requires_grad=True), dim=-1)
    zc = nn.functional.normalize(torch.randn(2, 4, requires_grad=True), dim=-1)
    qb = [nn.functional.normalize(torch.randn(2, 4), dim=-1)]
    qc = [nn.functional.normalize(torch.randn(2, 4), dim=-1)]
    loss = _queued_symmetric_loss(
        Scale(), zb, zc, ["b1", "b2"], ["c1", "c2"], qb, qc,
        ["b3", "b4"], ["c3", "c4"],
        {("b1", "c1"), ("b2", "c2"), ("b3", "c3"), ("b4", "c4")},
    )
    assert torch.isfinite(loss)
    loss.backward()
