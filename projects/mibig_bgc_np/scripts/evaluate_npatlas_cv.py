from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from clip_core.logging import save_json
from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.eval.retrieval_class_metrics import _parse_label_text
from projects.mibig_bgc_np.scripts.eval_retrieval import (
    _load_model, _match_true_products, _prepare_test_query_table, _require_rdkit,
)
from projects.mibig_bgc_np.training.contrastive_trainer import _pad_bgc_features


def _project_candidates(model, candidates: pd.DataFrame, raw: dict[str, torch.Tensor], device, batch_size=2048):
    parts=[]
    with torch.no_grad():
        for start in range(0,len(candidates),batch_size):
            smiles=candidates.canonical_smiles.iloc[start:start+batch_size].astype(str).tolist()
            features=torch.stack([raw[s] for s in smiles]).to(device)
            parts.append(model.encode_compound(features).cpu())
    return torch.cat(parts)


def _bgc_classes(interactions: pd.DataFrame, split: str) -> dict[str,list[str]]:
    df=interactions[interactions.split.astype(str).str.lower()==split]
    col="bgc_classes" if "bgc_classes" in df else "bgc_class"
    if col not in df:return {}
    return {str(r.bgc_id):_parse_label_text(getattr(r,col)) for r in df[["bgc_id",col]].drop_duplicates("bgc_id").itertuples()}


def _summarize(records: list[dict[str,Any]]) -> dict[str,Any]:
    if not records:return {"n_queries":0}
    out={"n_queries":len(records)}
    for key in ("reciprocal_rank","best_rank","hit_at_1","hit_at_5","hit_at_10","hit_at_100",
                "recall_at_1","recall_at_5","recall_at_10","recall_at_100",
                "precision_at_1","precision_at_5","precision_at_10","precision_at_100",
                "max_maccs_tanimoto_at_100"):
        vals=np.asarray([r[key] for r in records],dtype=float)
        name={"reciprocal_rank":"mrr","best_rank":"median_rank"}.get(key,key)
        out[name]=float(np.median(vals) if key=="best_rank" else vals.mean())
    out["per_class"]={}
    classes=sorted({c for r in records for c in r["bgc_classes"]})
    for cls in classes:
        subset=[r for r in records if cls in r["bgc_classes"]]
        out["per_class"][cls]={k:v for k,v in _summarize([{**r,"bgc_classes":[]} for r in subset]).items() if k!="per_class"}
    return out


def _evaluate_mode(
    *, mode: str, fold: int, seed: int, sample_size: int, candidate_embs: torch.Tensor,
    candidates: pd.DataFrame, model, bgc_cache, bgc_ids: list[str], truths: dict[str,list[int]],
    classes: dict[str,list[str]], device: torch.device,
) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    rng=np.random.default_rng(seed+fold*1009)
    all_ids=np.arange(len(candidates),dtype=np.int64)
    fp_cache: dict[int,Any]={}
    def fp(idx:int):
        if idx not in fp_cache:
            fp_cache[idx]=MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(str(candidates.iloc[idx].canonical_smiles)))
        return fp_cache[idx]
    records=[]
    for bgc_id in bgc_ids:
        true_ids=truths.get(bgc_id,[])
        if not true_ids or bgc_id not in bgc_cache:continue
        true_set=set(map(int,true_ids))
        if mode=="all": selected=all_ids
        else:
            background=all_ids[~np.isin(all_ids,list(true_set))]
            n_bg=sample_size-len(true_set)
            if n_bg<0:continue
            selected=np.concatenate([np.asarray(sorted(true_set)),rng.choice(background,n_bg,replace=False)])
        with torch.no_grad():
            bgc_features, bgc_padding_mask = _pad_bgc_features([bgc_cache[bgc_id].float()], device)
            q=model.encode_bgc(bgc_features, padding_mask=bgc_padding_mask).cpu()[0]
        scores=candidate_embs[selected]@q
        order=torch.argsort(scores,descending=True).numpy()
        ranked=selected[order]
        rank_lookup={int(cid):rank+1 for rank,cid in enumerate(ranked)}
        ranks=sorted(rank_lookup[x] for x in true_set)
        ranked_true=np.asarray([int(x in true_set) for x in ranked],dtype=int)
        top100=ranked[:min(100,len(ranked))]
        true_fps=[fp(x) for x in true_set]
        max_tan=max(DataStructs.TanimotoSimilarity(fp(int(x)),t) for x in top100 for t in true_fps)
        rec={"fold":fold,"bgc_id":bgc_id,"bgc_classes":classes.get(bgc_id,[]),"mode":mode,
             "candidate_count":int(len(selected)),"n_true":len(true_set),"best_rank":int(ranks[0]),
             "reciprocal_rank":1.0/ranks[0],"max_maccs_tanimoto_at_100":float(max_tan)}
        for k in (1,5,10,100):
            hits=int(ranked_true[:k].sum()); rec[f"hit_at_{k}"]=float(hits>0); rec[f"recall_at_{k}"]=hits/len(true_set)
            rec[f"precision_at_{k}"]=hits/float(min(k,len(ranked)))
        records.append(rec)
    return _summarize(records),records


def main() -> None:
    p=argparse.ArgumentParser(description="CV NPAtlas retrieval: full library and truth-preserving 10k samples")
    p.add_argument("--run_root",type=Path,required=True); p.add_argument("--split_name",required=True)
    p.add_argument("--cache_dir",type=Path,default=Path("cache/antismash_domain_esm2_molformer_full"))
    p.add_argument("--npatlas_cache",type=Path,default=Path("cache/npatlas_molformer"))
    p.add_argument("--outdir",type=Path,required=True); p.add_argument("--sample_size",type=int,default=10000)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--folds",type=int,nargs="+",default=list(range(1,11)))
    args=p.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.run_root/"summary.json").read_text())
    candidates=pd.read_csv(args.npatlas_cache/"candidates.tsv",sep="\t")
    raw=torch.load(args.npatlas_cache/"compound_features.pt",map_location="cpu")
    bgc_cache=torch.load(args.cache_dir/"bgc_features.pt",map_location="cpu")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_reports=[]; all_records=[]
    for fold in args.folds:
        fold_dir=args.run_root/f"fold_{fold}"; fs=json.loads((fold_dir/"fold_summary.json").read_text())
        interactions=build_interactions(summary["data_dir"],splits_path=summary["splits_path"],cv_fold=fold,val_fold=fs.get("val_fold"))
        query_df=_prepare_test_query_table(summary["data_dir"],interactions,"test",_require_rdkit())
        truths,match_counts=_match_true_products(query_df,candidates)
        bgc_ids=sorted(interactions[interactions.split=="test"].bgc_id.astype(str).unique())
        model,_=_load_model(fold_dir/"contrastive_model_best.pt",device)
        candidate_embs=_project_candidates(model,candidates,raw,device)
        classes=_bgc_classes(interactions,"test"); modes={}
        for mode in ("sampled_10000","all"):
            metrics,records=_evaluate_mode(mode=mode,fold=fold,seed=args.seed,sample_size=args.sample_size,
                candidate_embs=candidate_embs,candidates=candidates,model=model,bgc_cache=bgc_cache,bgc_ids=bgc_ids,
                truths=truths,classes=classes,device=device)
            modes[mode]=metrics; all_records.extend(records)
        fold_reports.append({"fold":fold,"n_test_bgcs":len(bgc_ids),"n_matched_bgcs":len(truths),"matches":match_counts,"modes":modes})
        del model,candidate_embs; torch.cuda.empty_cache()
    aggregate={}
    for mode in ("sampled_10000","all"):
        aggregate[mode]=_summarize([r for r in all_records if r["mode"]==mode])
    report={"split":args.split_name,"run_root":str(args.run_root),"npatlas_rows":len(candidates),
            "sample_size":args.sample_size,"seed":args.seed,"fingerprint":"MACCS keys","folds":fold_reports,"aggregate":aggregate}
    save_json(report,args.outdir/f"{args.split_name}_npatlas_retrieval.json")
    pd.DataFrame([{**r,"bgc_classes":";".join(r["bgc_classes"])} for r in all_records]).to_csv(
        args.outdir/f"{args.split_name}_npatlas_query_metrics.csv",index=False)


if __name__=="__main__":main()
