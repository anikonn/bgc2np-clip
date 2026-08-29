from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean, median, stdev

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from scripts._bootstrap import ensure_src_path

ensure_src_path()

from clip_core.logging import save_json
from projects.mibig_bgc_np.featurization.one_hot import ProteinOneHotConfig, ProteinOneHotEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OHE versus ESM2 on identical antiSMASH domains.")
    parser.add_argument("--domains_path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/paper_plots"))
    parser.add_argument("--model_name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--subset_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(function, repeats: int, device: torch.device) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        function()
        _synchronize(device)
        timings.append(time.perf_counter() - start)
    return timings


def _summarize(name: str, timings: list[float], n_sequences: int, n_residues: int) -> dict[str, object]:
    return {
        "encoder": name,
        "n_repeats": len(timings),
        "seconds_mean": mean(timings),
        "seconds_median": median(timings),
        "seconds_std": stdev(timings) if len(timings) > 1 else 0.0,
        "sequences_per_second": n_sequences / median(timings),
        "residues_per_second": n_residues / median(timings),
        "seconds_per_sequence": median(timings) / n_sequences,
        "raw_seconds": timings,
    }


def main() -> None:
    args = parse_args()
    sequences: list[str] = []
    with args.domains_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sequences.extend(
                str(sequence)
                for sequence, source in zip(
                    record["protein_seqs"],
                    record["sequence_sources"],
                    strict=True,
                )
                if source == "antismash_domain"
            )
    if args.subset_size > len(sequences):
        raise ValueError(f"Requested {args.subset_size} sequences but only {len(sequences)} are available")
    subset = random.Random(args.seed).sample(sequences, args.subset_size)
    n_residues = sum(min(len(sequence), args.max_length) for sequence in subset)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Run this benchmark on a GPU server.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    ohe = ProteinOneHotEncoder(ProteinOneHotConfig(max_length=args.max_length))
    ohe.encode(subset[: min(args.batch_size, len(subset))])
    ohe_timings = _measure(lambda: ohe.encode(subset), args.repeats, device)

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    model_load_seconds = time.perf_counter() - load_start

    @torch.inference_mode()
    def encode_esm2() -> None:
        for start in range(0, len(subset), args.batch_size):
            batch = tokenizer(
                subset[start : start + args.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            attention = batch["attention_mask"].bool()
            attention[:, 0] = False
            last_real = batch["attention_mask"].sum(dim=1) - 1
            attention[torch.arange(attention.size(0), device=device), last_real] = False
            weights = attention.unsqueeze(-1).to(hidden.dtype)
            _ = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    encode_esm2()  # Warm-up is excluded from timed repeats.
    esm_timings = _measure(encode_esm2, args.repeats, device)
    rows = [
        _summarize("OHE", ohe_timings, len(subset), n_residues),
        _summarize("ESM2-650M", esm_timings, len(subset), n_residues),
    ]
    speed_ratio = float(rows[1]["seconds_median"]) / float(rows[0]["seconds_median"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "raw_seconds"} for row in rows]
    ).to_csv(args.outdir / "antismash_ohe_vs_esm2_timing.csv", index=False)
    save_json(
        {
            "device": str(device),
            "model_name": args.model_name,
            "model_load_seconds": model_load_seconds,
            "subset_size": len(subset),
            "subset_seed": args.seed,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "total_residues_after_clipping": n_residues,
            "length_min": min(map(len, subset)),
            "length_mean": mean(map(len, subset)),
            "length_median": median(map(len, subset)),
            "length_max": max(map(len, subset)),
            "warmup_excluded": True,
            "esm_to_ohe_median_time_ratio": speed_ratio,
            "measurements": rows,
        },
        args.outdir / "antismash_ohe_vs_esm2_timing.json",
    )
    pd.DataFrame(
        {
            "sequence_index": range(len(subset)),
            "length_aa": [len(sequence) for sequence in subset],
        }
    ).to_csv(args.outdir / "antismash_timing_subset.csv", index=False)
    print(json.dumps({"esm_to_ohe_median_time_ratio": speed_ratio, "measurements": rows}, indent=2))


if __name__ == "__main__":
    main()
