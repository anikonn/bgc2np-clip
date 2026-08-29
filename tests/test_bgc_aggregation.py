from __future__ import annotations

import pytest
import torch

from projects.mibig_bgc_np.models.bgc_aggregation import (
    AttentionDomainAggregator,
    HierarchicalDomainAggregator,
    MeanDomainAggregator,
    TransformerDomainAggregator,
)


def test_masked_mean_ignores_padding() -> None:
    features = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    pooled, weights = MeanDomainAggregator(2)(features, torch.tensor([[False, False, True]]))
    assert torch.allclose(pooled, torch.tensor([[2.0, 4.0]]))
    assert torch.allclose(weights, torch.tensor([[0.5, 0.5, 0.0]]))


def test_attention_weights_are_masked_and_normalized() -> None:
    aggregator = AttentionDomainAggregator(4, hidden_dim=3)
    _, weights = aggregator(torch.randn(2, 5, 4), torch.tensor([[False, False, True, True, True], [False] * 5]))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.equal(weights[0, 2:], torch.zeros(3))


def test_hierarchical_domain_attention_then_protein_mean() -> None:
    aggregator = HierarchicalDomainAggregator(2, hidden_dim=3, protein_pooling="mean")
    features = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [0.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[False, False, False, True]])
    proteins = torch.tensor([[1, 1, 2, 0]])
    pooled, weights = aggregator(features, mask, protein_positions=proteins)
    assert pooled.shape == (1, 2)
    assert weights.shape == (1, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert torch.allclose(weights[0, :2].sum(), torch.tensor(0.5))
    assert torch.allclose(weights[0, 2], torch.tensor(0.5))
    assert weights[0, 3] == 0


def test_hierarchical_domain_and_protein_attention_backpropagates() -> None:
    aggregator = HierarchicalDomainAggregator(4, hidden_dim=3, protein_pooling="attention")
    features = torch.randn(2, 5, 4, requires_grad=True)
    mask = torch.tensor([[False, False, False, True, True], [False] * 5])
    proteins = torch.tensor([[1, 1, 2, 0, 0], [1, 2, 2, 3, 3]])
    pooled, weights = aggregator(features, mask, protein_positions=proteins)
    pooled.sum().backward()
    assert pooled.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert features.grad is not None


def test_hierarchical_aggregation_requires_protein_positions() -> None:
    aggregator = HierarchicalDomainAggregator(3)
    with pytest.raises(ValueError, match="protein_positions"):
        aggregator(torch.randn(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))


def test_transformer_shapes_and_optional_positions() -> None:
    aggregator = TransformerDomainAggregator(
        input_dim=6,
        d_model=8,
        n_layers=2,
        n_heads=4,
        max_domains=10,
        use_protein_positions=True,
        use_domain_positions=True,
    ).eval()
    features = torch.randn(2, 4, 6)
    mask = torch.tensor([[False, False, True, True], [False, False, False, False]])
    proteins = torch.tensor([[1, 1, 0, 0], [1, 2, 2, 3]])
    domains = torch.tensor([[1, 2, 0, 0], [1, 1, 2, 1]])
    pooled, weights = aggregator(features, mask, proteins, domains)
    assert pooled.shape == (2, 8)
    assert weights is None


def test_rejects_all_padding() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MeanDomainAggregator(3)(torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool))
