from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot lengths of antiSMASH-derived model input sequences.")
    parser.add_argument("--domains_path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/EDA"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    with args.domains_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for sequence_id, sequence, source in zip(
                record["protein_ids"],
                record["protein_seqs"],
                record["sequence_sources"],
                strict=True,
            ):
                rows.append(
                    {
                        "bgc_id": str(record["bgc_id"]),
                        "sequence_id": str(sequence_id),
                        "source": str(source),
                        "length_aa": len(str(sequence)),
                        "clipped_at_1024": len(str(sequence)) > 1024,
                    }
                )

    lengths = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    lengths.to_csv(args.outdir / "antismash_domain_lengths.csv", index=False)

    summary = (
        lengths.groupby("source")["length_aa"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    total = pd.DataFrame(
        [
            {
                "source": "all_model_inputs",
                "count": len(lengths),
                "mean": lengths["length_aa"].mean(),
                "std": lengths["length_aa"].std(),
                "min": lengths["length_aa"].min(),
                "median": lengths["length_aa"].median(),
                "max": lengths["length_aa"].max(),
            }
        ]
    )
    summary = pd.concat([summary, total], ignore_index=True)
    clipped = (
        lengths.groupby("source")["clipped_at_1024"]
        .agg(n_clipped="sum", fraction_clipped="mean")
        .reset_index()
    )
    clipped = pd.concat(
        [
            clipped,
            pd.DataFrame(
                [
                    {
                        "source": "all_model_inputs",
                        "n_clipped": int(lengths["clipped_at_1024"].sum()),
                        "fraction_clipped": float(lengths["clipped_at_1024"].mean()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    summary = summary.merge(clipped, on="source", how="left")
    summary.to_csv(args.outdir / "antismash_domain_length_summary.csv", index=False)

    colors = {
        "antismash_domain": "#2B6CB0",
        "unsplit_cds": "#DD6B20",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bins = range(0, 2001, 25)
    for source in ("antismash_domain", "unsplit_cds"):
        values = lengths.loc[lengths["source"] == source, "length_aa"]
        axes[0].hist(
            values.clip(upper=2000),
            bins=bins,
            alpha=0.65,
            label=source.replace("_", " "),
            color=colors[source],
        )
    axes[0].axvline(1024, color="black", linestyle="--", linewidth=1.2, label="OHE clipping limit")
    axes[0].set_xlabel("Sequence length (amino acids; values >2,000 shown at 2,000)")
    axes[0].set_ylabel("Number of sequences")
    axes[0].set_title("Lengths of antiSMASH-derived OHE inputs")
    axes[0].legend(frameon=False)

    domain_values = lengths.loc[lengths["source"] == "antismash_domain", "length_aa"]
    axes[1].hist(domain_values, bins=60, color=colors["antismash_domain"], alpha=0.85)
    axes[1].axvline(
        domain_values.median(),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"median = {domain_values.median():.0f} aa",
    )
    axes[1].set_xlabel("Domain length (amino acids)")
    axes[1].set_ylabel("Number of domains")
    axes[1].set_title("Annotated antiSMASH domains only")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.outdir / "antismash_domain_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
