from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT=Path("results/intermediate/encoder_finetuning")
RUNS=["b0_frozen_online","e2_esm_last2","e5_esm_last5","ef_esm_full","m2_molformer_last2","m4_molformer_last4","mf_molformer_full","em_esm5_molformer4","ff_both_full"]

def main() -> None:
    rows=[]
    for run in RUNS:
        data=json.loads((ROOT/run/"summary.json").read_text())
        row={"run":run,"n_folds":len(data["folds"])}
        for direction,prefix in [("bgc_to_compound","bgc_to_np"),("compound_to_bgc","np_to_bgc")]:
            for metric in ["mrr","recall_at_1","recall_at_5","recall_at_10"]:
                values=[float(f["retrieval_test"][direction][metric]) for f in data["folds"]]
                series=pd.Series(values)
                row[f"{prefix}_{metric}_mean"]=series.mean(); row[f"{prefix}_{metric}_std"]=series.std(ddof=0)
        row["mean_bidirectional_mrr"]=(row["bgc_to_np_mrr_mean"]+row["np_to_bgc_mrr_mean"])/2
        row["elapsed_hours_mean"]=pd.Series([f["elapsed_seconds"]/3600 for f in data["folds"]]).mean()
        rows.append(row)
    table=pd.DataFrame(rows).sort_values("mean_bidirectional_mrr",ascending=False)
    ROOT.mkdir(parents=True,exist_ok=True); table.to_csv(ROOT/"screen_strict_folds_1_2_3.csv",index=False)

if __name__=="__main__": main()
