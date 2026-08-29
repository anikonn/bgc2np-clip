from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer

from projects.mibig_bgc_np.featurization.morgan import MorganConfig, MorganFingerprintFeaturizer


@dataclass
class MorganCompoundConfig:
    radius: int = 2
    n_bits: int = 2048


class MorganCompoundEncoder:
    """Encode compounds with Morgan fingerprints."""

    def __init__(self, cfg: MorganCompoundConfig) -> None:
        self.cfg = cfg
        self.featurizer = MorganFingerprintFeaturizer(MorganConfig(radius=cfg.radius, n_bits=cfg.n_bits))

    def encode(self, molecules: list[str]) -> torch.Tensor:
        if not molecules:
            return torch.empty((0, self.cfg.n_bits), dtype=torch.float32)
        return torch.stack([self.featurizer.encode(smiles) for smiles in molecules], dim=0)


@dataclass
class MolFormerCompoundConfig:
    model_name: str = "ibm-research/MoLFormer-XL-both-10pct"
    max_length: int = 202


class MolFormerCompoundEncoder:
    """Encode canonical SMILES with the pretrained MoLFormer pooled representation."""

    def __init__(self, cfg: MolFormerCompoundConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            cfg.model_name,
            deterministic_eval=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, molecules: list[str]) -> torch.Tensor:
        if not molecules:
            hidden_size = int(self.model.config.hidden_size)
            return torch.empty((0, hidden_size), dtype=torch.float32)
        batch = self.tokenizer(
            molecules,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        outputs = self.model(**batch)
        if outputs.pooler_output is None:
            raise RuntimeError("MoLFormer did not return pooler_output")
        return outputs.pooler_output.float().cpu()


def build_molecule_encoder(
    cfg: dict[str, object],
    device: torch.device | None = None,
) -> MorganCompoundEncoder | MolFormerCompoundEncoder:
    encoder_name = str(cfg.get("molecule_encoder", "morgan")).lower()
    if encoder_name == "morgan":
        return MorganCompoundEncoder(
            MorganCompoundConfig(
                radius=int(cfg.get("morgan_radius", 2)),
                n_bits=int(cfg.get("morgan_bits", 2048)),
            )
        )
    if encoder_name in {"molformer", "molformer_xl"}:
        return MolFormerCompoundEncoder(
            MolFormerCompoundConfig(
                model_name=str(
                    cfg.get("molformer_model_name", "ibm-research/MoLFormer-XL-both-10pct")
                ),
                max_length=int(cfg.get("molformer_max_length", 202)),
            ),
            device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    raise ValueError(f"Unsupported molecule encoder: {encoder_name}")
