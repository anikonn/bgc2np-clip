from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from projects.mibig_bgc_np.models.projection import ProjectionHead


def _encoder_layers(model: nn.Module) -> list[nn.Module]:
    """Find the ordered Transformer block list in supported HF encoders."""
    candidates = [
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("model", "encoder", "layer"),
        ("model", "encoder", "layers"),
        ("bert", "encoder", "layer"),
    ]
    for path in candidates:
        node: Any = model
        for name in path:
            node = getattr(node, name, None)
            if node is None:
                break
        if isinstance(node, (nn.ModuleList, list, tuple)) and len(node):
            return list(node)
    module_lists = [module for module in model.modules() if isinstance(module, nn.ModuleList) and len(module)]
    if module_lists:
        return list(max(module_lists, key=len))
    raise ValueError(f"Could not locate Transformer layers in {type(model).__name__}")


def configure_encoder_trainability(model: nn.Module, unfreeze_last_n: int | str) -> dict[str, int | str]:
    """Freeze an encoder, unfreeze its last N blocks, or fully fine-tune it."""
    mode = str(unfreeze_last_n).lower()
    layers = _encoder_layers(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        n = int(mode)
        if n < 0 or n > len(layers):
            raise ValueError(f"unfreeze_last_n must be between 0 and {len(layers)}, or 'full'; got {n}")
        for layer in layers[len(layers) - n :] if n else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        if n:
            # MolFormer's pooled representation has a separate output pooler.
            # Do not broadly unfreeze every LayerNorm: those also occur inside
            # frozen blocks and would invalidate the meaning of "last N".
            for name, parameter in model.named_parameters():
                if "pooler" in name.lower():
                    parameter.requires_grad = True
    return {
        "mode": mode,
        "n_layers": len(layers),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_total_parameters": sum(p.numel() for p in model.parameters()),
    }


class OnlineFineTuneCLIP(nn.Module):
    """Differentiable ESM2-domain/MolFormer CLIP with hierarchical BGC means."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        ft = cfg["finetune"]
        model_cfg = cfg["model"]
        self.esm_tokenizer = AutoTokenizer.from_pretrained(ft["esm_model_name"])
        self.esm = AutoModel.from_pretrained(ft["esm_model_name"])
        self.mol_tokenizer = AutoTokenizer.from_pretrained(ft["molformer_model_name"], trust_remote_code=True)
        self.molformer = AutoModel.from_pretrained(
            ft["molformer_model_name"], deterministic_eval=True, trust_remote_code=True
        )
        self.esm_status = configure_encoder_trainability(self.esm, ft["esm_unfreeze"])
        self.molformer_status = configure_encoder_trainability(self.molformer, ft["molformer_unfreeze"])
        if any(parameter.requires_grad for parameter in self.esm.parameters()):
            self.esm.gradient_checkpointing_enable()
        if any(parameter.requires_grad for parameter in self.molformer.parameters()):
            checkpointing = getattr(self.molformer, "gradient_checkpointing_enable", None)
            if checkpointing is not None:
                checkpointing()
        self.esm_max_length = int(ft.get("esm_max_length", 1024))
        self.mol_max_length = int(ft.get("molformer_max_length", 202))
        self.domain_micro_batch = int(ft.get("domain_micro_batch", 8))
        esm_dim = int(self.esm.config.hidden_size)
        mol_dim = int(self.molformer.config.hidden_size)
        self.bgc_proj = ProjectionHead(esm_dim, int(model_cfg["emb_dim"]), int(model_cfg["hidden_dim"]), float(model_cfg["dropout"]))
        self.compound_proj = ProjectionHead(mol_dim, int(model_cfg["emb_dim"]), int(model_cfg["hidden_dim"]), float(model_cfg["dropout"]))
        self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.0 / float(model_cfg["init_temperature"]))))))
        self.max_logit_scale = float(model_cfg["max_logit_scale"])
        self.esm_frozen = not any(p.requires_grad for p in self.esm.parameters())
        self.molformer_frozen = not any(p.requires_grad for p in self.molformer.parameters())
        # Frozen encoder outputs are invariant. Keep them on CPU so repeated
        # epochs train only the projection heads instead of rerunning ESM/MolFormer.
        self._bgc_raw_cache: dict[str, torch.Tensor] = {}
        self._compound_raw_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _masked_token_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask.bool().clone()
        valid[:, 0] = False
        last = attention_mask.sum(dim=1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        valid[rows, last.clamp_min(0)] = False
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)

    def encode_domains(self, sequences: list[str], device: torch.device) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(sequences), self.domain_micro_batch):
            tokens = self.esm_tokenizer(
                sequences[start : start + self.domain_micro_batch], padding=True, truncation=True,
                max_length=self.esm_max_length, return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = self.esm(**tokens).last_hidden_state
            chunks.append(self._masked_token_mean(hidden, tokens["attention_mask"]))
        return torch.cat(chunks)

    def _encode_bgcs_raw(self, records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
        sequences = [sequence for record in records for sequence in record["protein_seqs"]]
        domain_embeddings = self.encode_domains(sequences, device)
        bgcs: list[torch.Tensor] = []
        offset = 0
        for record in records:
            count = len(record["protein_seqs"])
            domains = domain_embeddings[offset : offset + count]
            offset += count
            parents = [int(value) for value in record["parent_cds_indices"]]
            grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
            for parent, embedding in zip(parents, domains, strict=True):
                grouped[parent].append(embedding)
            proteins = torch.stack([torch.stack(grouped[index]).mean(0) for index in sorted(grouped)])
            bgcs.append(proteins.mean(0))
        return torch.stack(bgcs).float()

    def encode_bgcs(self, records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
        if not self.esm_frozen:
            raw = self._encode_bgcs_raw(records, device)
        else:
            missing = [r for r in records if str(r["bgc_id"]) not in self._bgc_raw_cache]
            if missing:
                with torch.no_grad():
                    values = self._encode_bgcs_raw(missing, device).cpu()
                for record, value in zip(missing, values, strict=True):
                    self._bgc_raw_cache[str(record["bgc_id"])] = value
            raw = torch.stack([self._bgc_raw_cache[str(r["bgc_id"])] for r in records]).to(device)
        return F.normalize(self.bgc_proj(raw), dim=-1)

    def _encode_compounds_raw(self, smiles: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.mol_tokenizer(
            smiles, padding=True, truncation=True, max_length=self.mol_max_length, return_tensors="pt"
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.molformer(**tokens)
        pooled = outputs.pooler_output
        if pooled is None:
            raise RuntimeError("MolFormer did not return pooler_output")
        return pooled.float()

    def encode_compounds(self, smiles: list[str], device: torch.device) -> torch.Tensor:
        if not self.molformer_frozen:
            raw = self._encode_compounds_raw(smiles, device)
        else:
            missing = list(dict.fromkeys(s for s in smiles if s not in self._compound_raw_cache))
            if missing:
                with torch.no_grad():
                    values = self._encode_compounds_raw(missing, device).cpu()
                self._compound_raw_cache.update(zip(missing, values, strict=True))
            raw = torch.stack([self._compound_raw_cache[s] for s in smiles]).to(device)
        return F.normalize(self.compound_proj(raw), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.max_logit_scale)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in trainable}
