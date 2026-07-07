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
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        batch_size, _ = attention_mask.shape
        valid_mask = attention_mask.bool()
        valid_mask[:, 0] = False

        last_real_pos = attention_mask.sum(dim=1) - 1
        valid_positions = last_real_pos > 0
        batch_idx = torch.arange(batch_size, device=self.device)[valid_positions]
        valid_mask[batch_idx, last_real_pos[valid_positions]] = False

        valid_mask_f = valid_mask.unsqueeze(-1).float()
        summed = (hidden * valid_mask_f).sum(dim=1)
        denom = valid_mask_f.sum(dim=1).clamp(min=1.0)
        return (summed / denom).cpu()


class ESM2CLSProteinEmbedder:
    """Use the BOS/CLS token embedding at position 0 as the sequence representation."""

    def __init__(self, cfg: ESM2Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sequences: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return hidden[:, 0, :].cpu()
