from __future__ import annotations

import torch
from torch import nn


def validate_domain_batch(features: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"Domain features must have shape [batch, domains, dim], got {tuple(features.shape)}")
    if padding_mask is None:
        padding_mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=features.device)
    if padding_mask.shape != features.shape[:2]:
        raise ValueError(
            f"Padding mask must have shape {tuple(features.shape[:2])}, got {tuple(padding_mask.shape)}"
        )
    padding_mask = padding_mask.bool()
    if bool(padding_mask.all(dim=1).any()):
        raise ValueError("Every BGC must contain at least one non-padding domain")
    return padding_mask


class MeanDomainAggregator(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output_dim = input_dim

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None, **_: torch.Tensor):
        mask = validate_domain_batch(features, padding_mask)
        valid = (~mask).unsqueeze(-1).to(features.dtype)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        weights = (~mask).to(features.dtype) / (~mask).sum(dim=1, keepdim=True).clamp_min(1)
        return pooled, weights


class AttentionDomainAggregator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or input_dim
        self.output_dim = input_dim
        self.score = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None, **_: torch.Tensor):
        mask = validate_domain_batch(features, padding_mask)
        logits = self.score(features).squeeze(-1).masked_fill(mask, torch.finfo(features.dtype).min)
        weights = torch.softmax(logits, dim=1).masked_fill(mask, 0.0)
        return (features * weights.unsqueeze(-1)).sum(dim=1), weights


class HierarchicalDomainAggregator(nn.Module):
    """Aggregate domains within proteins, then proteins within each BGC.

    ``protein_positions`` contains positive, one-based protein/CDS identifiers
    for valid items. Zero is reserved for padding. Domain attention is always
    learned; the second stage is either an equal protein mean or learned protein
    attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        protein_pooling: str = "mean",
    ) -> None:
        super().__init__()
        if protein_pooling not in {"mean", "attention"}:
            raise ValueError(f"protein_pooling must be 'mean' or 'attention', got {protein_pooling!r}")
        hidden = hidden_dim or input_dim
        self.output_dim = input_dim
        self.protein_pooling = protein_pooling
        self.domain_score = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.protein_score = (
            nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            if protein_pooling == "attention"
            else None
        )

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        protein_positions: torch.Tensor | None = None,
        **_: torch.Tensor,
    ):
        mask = validate_domain_batch(features, padding_mask)
        if protein_positions is None or protein_positions.shape != mask.shape:
            raise ValueError(
                f"protein_positions must have shape {tuple(mask.shape)} for hierarchical aggregation"
            )
        protein_positions = protein_positions.long()
        if bool((protein_positions[~mask] <= 0).any()):
            raise ValueError("Valid domains must have positive one-based protein_positions")

        pooled_bgcs: list[torch.Tensor] = []
        effective_domain_weights = torch.zeros(mask.shape, dtype=features.dtype, device=features.device)
        for batch_index in range(features.shape[0]):
            valid = ~mask[batch_index]
            item_features = features[batch_index, valid]
            item_proteins = protein_positions[batch_index, valid]
            protein_ids = torch.unique(item_proteins, sorted=True)
            protein_embeddings: list[torch.Tensor] = []
            within_protein_weights: list[tuple[torch.Tensor, torch.Tensor]] = []
            for protein_id in protein_ids:
                belongs = item_proteins == protein_id
                domain_features = item_features[belongs]
                domain_weights = torch.softmax(self.domain_score(domain_features).squeeze(-1), dim=0)
                protein_embeddings.append((domain_features * domain_weights.unsqueeze(-1)).sum(dim=0))
                within_protein_weights.append((belongs, domain_weights))
            proteins = torch.stack(protein_embeddings)
            if self.protein_score is None:
                protein_weights = torch.full(
                    (proteins.shape[0],), 1.0 / proteins.shape[0], dtype=proteins.dtype, device=proteins.device
                )
            else:
                protein_weights = torch.softmax(self.protein_score(proteins).squeeze(-1), dim=0)
            pooled_bgcs.append((proteins * protein_weights.unsqueeze(-1)).sum(dim=0))

            # Return each domain's effective contribution after both stages.
            valid_weights = torch.zeros(item_features.shape[0], dtype=features.dtype, device=features.device)
            for protein_index, (belongs, domain_weights) in enumerate(within_protein_weights):
                valid_weights[belongs] = domain_weights * protein_weights[protein_index]
            effective_domain_weights[batch_index, valid] = valid_weights

        return torch.stack(pooled_bgcs), effective_domain_weights


class TransformerDomainAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_domains: int = 256,
        use_protein_positions: bool = False,
        use_domain_positions: bool = False,
        max_proteins: int = 256,
        max_domains_per_protein: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.output_dim = d_model
        self.max_domains = max_domains
        self.use_protein_positions = use_protein_positions
        self.use_domain_positions = use_domain_positions
        self.input_projection = nn.Linear(input_dim, d_model)
        self.bgc_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Embedding(max_domains + 1, d_model)
        self.protein_position_embedding = nn.Embedding(max_proteins + 1, d_model) if use_protein_positions else None
        self.domain_position_embedding = (
            nn.Embedding(max_domains_per_protein + 1, d_model) if use_domain_positions else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.bgc_token, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        protein_positions: torch.Tensor | None = None,
        domain_positions: torch.Tensor | None = None,
    ):
        mask = validate_domain_batch(features, padding_mask)
        batch_size, n_domains, _ = features.shape
        if n_domains > self.max_domains:
            raise ValueError(f"Received {n_domains} domains, but max_domains={self.max_domains}")
        x = self.input_projection(features)
        absolute = torch.arange(1, n_domains + 1, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(absolute)
        for enabled, values, embedding, name in (
            (self.use_protein_positions, protein_positions, self.protein_position_embedding, "protein_positions"),
            (self.use_domain_positions, domain_positions, self.domain_position_embedding, "domain_positions"),
        ):
            if enabled:
                if values is None or values.shape != mask.shape:
                    raise ValueError(f"{name} must have shape {tuple(mask.shape)} when enabled")
                assert embedding is not None
                if int(values.max()) >= embedding.num_embeddings:
                    raise ValueError(f"{name} contains an index outside the configured embedding range")
                x = x + embedding(values.clamp_min(0))
        token = self.bgc_token.expand(batch_size, -1, -1)
        x = torch.cat([token, x], dim=1)
        token_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=mask.device)
        encoded = self.encoder(x, src_key_padding_mask=torch.cat([token_mask, mask], dim=1))
        return self.norm(encoded[:, 0]), None


def build_bgc_aggregator(mode: str, input_dim: int, cfg: dict[str, object]) -> nn.Module | None:
    mode = mode.lower()
    if mode in {"prepooled", "none"}:
        return None
    if mode == "mean":
        return MeanDomainAggregator(input_dim)
    if mode == "attention":
        return AttentionDomainAggregator(input_dim, int(cfg.get("attention_hidden_dim", input_dim)))
    if mode in {"hierarchical_attention_mean", "domain_attention_protein_mean"}:
        return HierarchicalDomainAggregator(
            input_dim,
            hidden_dim=int(cfg.get("attention_hidden_dim", input_dim)),
            protein_pooling="mean",
        )
    if mode in {"hierarchical_attention_attention", "domain_attention_protein_attention"}:
        return HierarchicalDomainAggregator(
            input_dim,
            hidden_dim=int(cfg.get("attention_hidden_dim", input_dim)),
            protein_pooling="attention",
        )
    if mode == "transformer":
        return TransformerDomainAggregator(
            input_dim=input_dim,
            d_model=int(cfg.get("d_model", 256)),
            n_layers=int(cfg.get("n_layers", 2)),
            n_heads=int(cfg.get("n_heads", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
            max_domains=int(cfg.get("max_domains", 256)),
            use_protein_positions=bool(cfg.get("use_protein_positions", False)),
            use_domain_positions=bool(cfg.get("use_domain_positions", False)),
            max_proteins=int(cfg.get("max_proteins", 256)),
            max_domains_per_protein=int(cfg.get("max_domains_per_protein", 64)),
        )
    raise ValueError(f"Unsupported BGC aggregation mode: {mode}")
