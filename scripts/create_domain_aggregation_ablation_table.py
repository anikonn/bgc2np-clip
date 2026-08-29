from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUNS = {
    "mean": Path("results/domain_mean_esm2_t30_molformer_strict_cv10/summary.json"),
    "attention": Path("results/domain_attention_esm2_t30_molformer_strict_cv10/summary.json"),
    "transformer": Path("results/domain_transformer_esm2_t30_molformer_strict_cv10/summary.json"),
    "domain_attention_protein_mean": Path(
        "results/domain_hierarchical_attention_mean_esm2_t30_molformer_strict_cv10/summary.json"
    ),
    "domain_attention_protein_attention": Path(
        "results/domain_hierarchical_attention_attention_esm2_t30_molformer_strict_cv10/summary.json"
    ),
}
OUTDIR = Path("results/paper_plots/domain_aggregation_ablation")


def main() -> None:
    rows: list[dict[str, object]] = []
    for mode, path in RUNS.items():
        summary = json.loads(path.read_text(encoding="utf-8"))
        retrieval = summary["aggregate"]["retrieval_test"]
        row: dict[str, object] = {"aggregation": mode, "n_folds": 10}
        for direction, prefix in (("bgc_to_compound", "bgc_to_np"), ("compound_to_bgc", "np_to_bgc")):
            for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10"):
                node = retrieval[direction][metric]
                row[f"{prefix}_{metric}_mean"] = float(node["mean"])
                row[f"{prefix}_{metric}_std"] = float(node["std"])
        rows.append(row)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUTDIR / "strict_cv10_domain_aggregation_retrieval.csv", index=False)
    (OUTDIR / "README.md").write_text(
        "# Domain aggregation ablation\n\n"
        "Strict CV10 using identical frozen ESM2-t30 domain and MolFormer embeddings, splits, loss, and metrics.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
