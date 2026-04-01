from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class ESM2Config:
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    max_length: int = 1024
    batch_size: int = 8


class ESM2MeanPoolEmbedder:
    """Compute mean-pooled ESM2 embeddings excluding special tokens."""

    def __init__(self, cfg: ESM2Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sequences: list[str]) -> torch.Tensor:
        """Return [B, D] mean pooled sequence embeddings.
        
        Excludes special tokens (BOS at position 0, EOS at last real position)
        and padding tokens. Compatible with variable sequence lengths via attention_mask.
        """
        # Tokenize with padding and truncation
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Forward pass in eval mode without gradient computation
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, T, D]
        
        B, T = attention_mask.shape
        
        # Build mask: True for real residue tokens (exclude BOS, EOS, and padding)
        valid_mask = attention_mask.bool()  # [B, T]
        valid_mask[:, 0] = False  # Exclude BOS (always at position 0)
        
        # Exclude EOS (always at last real position for each sequence)
        # For sequences with only BOS, last_real_pos=0, so we skip EOS exclusion
        last_real_pos = attention_mask.sum(dim=1) - 1  # [B]
        valid_positions = last_real_pos > 0
        batch_idx = torch.arange(B, device=self.device)[valid_positions]
        valid_mask[batch_idx, last_real_pos[valid_positions]] = False
        
        # Compute masked mean pooling across sequence dimension
        valid_mask_f = valid_mask.unsqueeze(-1).float()  # [B, T, 1]
        summed = (hidden * valid_mask_f).sum(dim=1)  # [B, D]
        denom = valid_mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = summed / denom  # [B, D]
        
        return pooled.cpu()
        
        # Batching invariance check (unit test pseudocode):
        # from torch.nn.functional import cosine_similarity
        # emb_single = embedder.encode(["MKLAVS"])[0]
        # emb_batch = embedder.encode(["MKLAVS", "MSTNNE"])[0]
        # assert cosine_similarity(emb_single.unsqueeze(0), emb_batch.unsqueeze(0)) > 0.9999
        # This ensures mean pooling is independent of batch padding.
