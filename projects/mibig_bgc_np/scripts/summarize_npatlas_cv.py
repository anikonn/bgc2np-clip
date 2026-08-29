from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p=argparse.ArgumentParser(description="Create a flat paper table from NPAtlas CV reports")
    p.add_argument("--indir",type=Path,default=Path("results/paper_plots/npatlas_retrieval"))
    args=p.parse_args(); rows=[]
    metrics=("mrr","median_rank","hit_at_1","hit_at_5","hit_at_10","hit_at_100",
             "recall_at_1","recall_at_5","recall_at_10","recall_at_100",
             "precision_at_1","max_maccs_tanimoto_at_100")
    for path in sorted(args.indir.glob("*_npatlas_retrieval.json")):
        report=json.loads(path.read_text())
        for mode,node in report["aggregate"].items():
            row={"split":report["split"],"candidate_mode":mode,"candidate_library_size":
                 report["npatlas_rows"] if mode=="all" else report["sample_size"],
                 "n_queries":node.get("n_queries",0),"fingerprint":"MACCS keys"}
            row.update({metric:node.get(metric) for metric in metrics}); rows.append(row)
    frame=pd.DataFrame(rows)
    args.indir.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.indir/"npatlas_retrieval_all_splits.csv",index=False)
    (args.indir/"README.txt").write_text(
        "NPAtlas BGC-to-NP retrieval for the baseline ESM2-domain + MolFormer model.\n"
        "sampled_10000 includes every true answer and samples remaining negatives with seed 42.\n"
        "all ranks against all valid NPAtlas rows. max_maccs_tanimoto_at_100 compares top-100 hits with true products.\n"
    )


if __name__=="__main__":main()
