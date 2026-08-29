from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys

from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.eval.retrieval_baselines import (
    LinearDualEncoderCLIP, _fixed_random_projection, split_entities_and_pairs,
)
from projects.mibig_bgc_np.scripts.eval_retrieval import _load_model

RDLogger.DisableLog("rdApp.*")


def _model_scores(model, bgc_ids, compound_ids, bgc_cache, compound_cache, batch=1024):
    model.eval(); zb=[]; zc=[]
    with torch.no_grad():
        for start in range(0,len(bgc_ids),batch):
            zb.append(model.encode_bgc(torch.stack([bgc_cache[x].float() for x in bgc_ids[start:start+batch]])))
        for start in range(0,len(compound_ids),batch):
            zc.append(model.encode_compound(torch.stack([compound_cache[x].float() for x in compound_ids[start:start+batch]])))
    return torch.cat(zb)@torch.cat(zc).t()


def _frozen_scores(bgc_ids,compound_ids,bgc_cache,compound_cache,seed,output_dim):
    b=torch.stack([bgc_cache[x].float() for x in bgc_ids]); c=torch.stack([compound_cache[x].float() for x in compound_ids])
    if b.shape[1]==c.shape[1]: bp,cp=b,c
    else:
        dim=min(output_dim,b.shape[1],c.shape[1])
        bp=_fixed_random_projection(b,dim,seed); cp=_fixed_random_projection(c,dim,seed+100_003)
    return F.normalize(bp,dim=-1)@F.normalize(cp,dim=-1).t()


def _knn_scores(interactions, bgc_ids, compound_ids, bgc_cache):
    train=interactions[interactions.split.astype(str).str.lower()=="train"]
    train_bgcs=sorted(train.bgc_id.astype(str).unique()); ti={x:i for i,x in enumerate(train_bgcs)}
    test=F.normalize(torch.stack([bgc_cache[x].float() for x in bgc_ids]),dim=-1)
    ref=F.normalize(torch.stack([bgc_cache[x].float() for x in train_bgcs]),dim=-1)
    sim=test@ref.t(); routes={}
    for r in train[["bgc_id","compound_id"]].drop_duplicates().itertuples(index=False):
        routes.setdefault(str(r.compound_id),[]).append(ti[str(r.bgc_id)])
    scores=torch.zeros((len(bgc_ids),len(compound_ids)))
    for j,cid in enumerate(compound_ids):
        idx=routes.get(cid,[])
        if idx:scores[:,j]=sim[:,idx].max(1).values
    return scores


def _linear_model(path: Path):
    ck=torch.load(path,map_location="cpu"); cfg=ck["config"]
    model=LinearDualEncoderCLIP(ck["bgc_input_dim"],ck["compound_input_dim"],int(cfg["model"]["emb_dim"]),
        float(cfg["model"].get("init_temperature",.07)),float(cfg["model"].get("max_logit_scale",100.)))
    model.load_state_dict(ck["model_state_dict"]); model.eval(); return model


def _rankwise_records(model_name, scores, fold, bgc_ids, compound_ids, true_by_bgc, fp_by_compound, trial=None):
    top=torch.topk(scores,k=min(10,len(compound_ids)),dim=1).indices
    rows=[]
    for i,bgc in enumerate(bgc_ids):
        true_fps=[fp_by_compound[x] for x in true_by_bgc.get(bgc,[]) if x in fp_by_compound]
        if not true_fps:continue
        for rank,j in enumerate(top[i].tolist(),1):
            cid=compound_ids[j]; candidate=fp_by_compound.get(cid)
            if candidate is None:continue
            value=max(DataStructs.TanimotoSimilarity(candidate,t) for t in true_fps)
            rows.append({"model":model_name,"fold":fold,"trial":trial,"bgc_id":bgc,"rank":rank,"max_maccs_tanimoto":value})
    return rows


def main() -> None:
    p=argparse.ArgumentParser(description="Plot mean top-10 positional MACCS similarity to true NPs")
    p.add_argument("--run_root",type=Path,default=Path("results/best_esm_domains_molformer_strict_cv10"))
    p.add_argument("--outdir",type=Path,default=Path("results/paper_plots")); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--random_trials",type=int,default=10); args=p.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.run_root/"summary.json").read_text()); cache=Path(summary["cache_dir"])
    bgc_cache=torch.load(cache/"bgc_features.pt",map_location="cpu"); compound_cache=torch.load(cache/"compound_features.pt",map_location="cpu")
    all_rows=[]; fp_global={}
    for fold in range(1,11):
        folder=args.run_root/f"fold_{fold}"; fs=json.loads((folder/"fold_summary.json").read_text())
        interactions=build_interactions(summary["data_dir"],splits_path=summary["splits_path"],cv_fold=fold,val_fold=fs.get("val_fold"))
        bgc_ids,compound_ids,_pairs,test=split_entities_and_pairs(interactions,"test")
        smiles_by_id=test[["compound_id","smiles"]].drop_duplicates("compound_id").set_index("compound_id").smiles.astype(str).to_dict()
        for cid in compound_ids:
            if cid not in fp_global:
                mol=Chem.MolFromSmiles(smiles_by_id.get(cid,cid)); fp_global[cid]=MACCSkeys.GenMACCSKeys(mol) if mol else None
        truth=test.groupby("bgc_id").compound_id.apply(lambda x:sorted(set(map(str,x)))).to_dict()
        fp={x:fp_global[x] for x in compound_ids if fp_global[x] is not None}
        model,cfg=_load_model(folder/"contrastive_model_best.pt",torch.device("cpu"))
        scores={
            "BGC2NP-CLIP":_model_scores(model,bgc_ids,compound_ids,bgc_cache,compound_cache),
            "Frozen encoders":_frozen_scores(bgc_ids,compound_ids,bgc_cache,compound_cache,args.seed+fold,int(cfg["model"]["emb_dim"])),
            "kNN transfer":_knn_scores(interactions,bgc_ids,compound_ids,bgc_cache),
        }
        linear=_linear_model(folder/"retrieval_baselines/linear_projection/linear_projection_baseline_best.pt")
        scores["Linear projection"]=_model_scores(linear,bgc_ids,compound_ids,bgc_cache,compound_cache)
        for name,matrix in scores.items():all_rows.extend(_rankwise_records(name,matrix,fold,bgc_ids,compound_ids,truth,fp))
        for trial in range(args.random_trials):
            rng=np.random.default_rng(args.seed+fold+trial)
            random=torch.tensor(rng.random((len(bgc_ids),len(compound_ids))),dtype=torch.float32)
            all_rows.extend(_rankwise_records("Random",random,fold,bgc_ids,compound_ids,truth,fp,trial))
    raw=pd.DataFrame(all_rows); raw.to_csv(args.outdir/"strict_rankwise_top10_maccs_per_query.csv",index=False)
    summary_rows=[]
    # Random trials contribute equally after first averaging within each query/fold/rank.
    reduced=raw.groupby(["model","fold","bgc_id","rank"],as_index=False).max_maccs_tanimoto.mean()
    for (name,rank),group in reduced.groupby(["model","rank"],sort=False):
        fold_means=group.groupby("fold").max_maccs_tanimoto.mean().to_numpy()
        summary_rows.append({"model":name,"rank":rank,"mean":group.max_maccs_tanimoto.mean(),
            "std_across_folds":fold_means.std(ddof=0),"n_queries":group.bgc_id.nunique(),"n_folds":len(fold_means)})
    curve=pd.DataFrame(summary_rows); curve.to_csv(args.outdir/"strict_rankwise_top10_maccs_summary.csv",index=False)
    colors={"BGC2NP-CLIP":"#d62728","Linear projection":"#1f77b4","Frozen encoders":"#2ca02c","kNN transfer":"#9467bd","Random":"#7f7f7f"}
    fig,ax=plt.subplots(figsize=(7.4,4.6))
    for name,g in curve.groupby("model",sort=False):
        g=g.sort_values("rank"); ax.plot(g["rank"],g["mean"],marker="o",linewidth=2 if name=="BGC2NP-CLIP" else 1.4,
            label=name,color=colors.get(name)); ax.fill_between(g["rank"],g["mean"]-g["std_across_folds"],g["mean"]+g["std_across_folds"],alpha=.09,color=colors.get(name))
    ax.set(xlabel="Retrieved candidate rank",ylabel="Mean max MACCS Tanimoto to true NP",xticks=range(1,11),ylim=(0,1.02))
    ax.grid(axis="y",linestyle=":",alpha=.5); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,ncol=2); fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(args.outdir/f"strict_rankwise_top10_maccs.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":main()
