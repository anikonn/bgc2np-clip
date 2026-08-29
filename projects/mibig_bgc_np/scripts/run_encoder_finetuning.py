from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from clip_core.config import load_yaml
from clip_core.logging import save_json, setup_logger
from mibig_clip.eval.retrieval import evaluate_global_retrieval_multi
from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.models.online_finetune_clip import OnlineFineTuneCLIP


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
    def __len__(self) -> int:
        return len(self.frame)
    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.frame.iloc[index]
        return {"bgc_id": str(row.bgc_id), "compound_id": str(row.compound_id), "smiles": str(row.smiles)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective online fine-tuning of ESM2 and MolFormer for BGC-NP CLIP")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/MIBIG/processed"))
    parser.add_argument("--domains", type=Path, default=Path("cache/antismash_hierarchical_esm2_t30_molformer/antismash_domains_with_parents.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("data/MIBIG/splits/strict_bigscape_butina_cv_seed42_n10.tsv"))
    parser.add_argument("--fold_ids", type=int, nargs="+", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--esm_unfreeze", required=True, help="0, N, or full")
    parser.add_argument("--molformer_unfreeze", required=True, help="0, N, or full")
    return parser.parse_args()


def _collate(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {key: [row[key] for row in rows] for key in rows[0]}


def _unique(values: list[str]) -> tuple[list[str], torch.Tensor]:
    unique: list[str] = []
    index: dict[str, int] = {}
    inverse: list[int] = []
    for value in values:
        if value not in index:
            index[value] = len(unique); unique.append(value)
        inverse.append(index[value])
    return unique, torch.tensor(inverse, dtype=torch.long)


def _embed_entities(model, frame, split, domains, device, batch_size):
    selected = frame[frame.split == split]
    bgc_ids = sorted(selected.bgc_id.unique())
    compound_rows = selected[["compound_id", "smiles"]].drop_duplicates("compound_id").sort_values("compound_id")
    compound_ids = compound_rows.compound_id.astype(str).tolist()
    bgc_parts=[]; cmp_parts=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(bgc_ids),batch_size):
            ids=bgc_ids[start:start+batch_size]; bgc_parts.append(model.encode_bgcs([domains[x] for x in ids],device).cpu())
        smiles=compound_rows.smiles.astype(str).tolist()
        for start in range(0,len(smiles),batch_size): cmp_parts.append(model.encode_compounds(smiles[start:start+batch_size],device).cpu())
    bi={x:i for i,x in enumerate(bgc_ids)}; ci={x:i for i,x in enumerate(compound_ids)}
    pairs=[(bi[str(r.bgc_id)],ci[str(r.compound_id)]) for r in selected.itertuples()]
    return torch.cat(bgc_parts),torch.cat(cmp_parts),pairs


def _evaluate(model, frame, split, domains, device, batch_size, sim_batch_size):
    b,c,p=_embed_entities(model,frame,split,domains,device,batch_size)
    return evaluate_global_retrieval_multi(bgc_embs=b,compound_embs=c,interaction_pairs=p,sim_batch_size=sim_batch_size)


def _score(metrics):
    return .5*(metrics["bgc_to_compound"]["mrr"]+metrics["compound_to_bgc"]["mrr"])


def _directional_multi_positive_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if bool((~positive_mask.any(dim=1)).any()):
        raise ValueError("Every contrastive query must have at least one positive candidate")
    numerator=torch.logsumexp(logits.masked_fill(~positive_mask,float("-inf")),dim=1)
    denominator=torch.logsumexp(logits,dim=1)
    return -(numerator-denominator).mean()


def _queued_symmetric_loss(model,zb,zc,bgc_ids,compound_ids,queue_bgc,queue_cmp,queue_bgc_ids,queue_cmp_ids,positive_pairs):
    if not torch.isfinite(zb).all() or not torch.isfinite(zc).all():
        raise FloatingPointError("Non-finite encoder/projection embedding detected")
    candidate_cmp=torch.cat([zc]+queue_cmp,dim=0) if queue_cmp else zc
    candidate_bgc=torch.cat([zb]+queue_bgc,dim=0) if queue_bgc else zb
    candidate_cmp_ids=list(compound_ids)+queue_cmp_ids
    candidate_bgc_ids=list(bgc_ids)+queue_bgc_ids
    b2c_pos=torch.tensor([[(b,candidate_cmp_ids[j]) in positive_pairs for j in range(len(candidate_cmp_ids))] for b in bgc_ids],dtype=torch.bool,device=zb.device)
    c2b_pos=torch.tensor([[(candidate_bgc_ids[j],c) in positive_pairs for j in range(len(candidate_bgc_ids))] for c in compound_ids],dtype=torch.bool,device=zb.device)
    scale=model.scale()
    loss=.5*(_directional_multi_positive_loss(scale*(zb.float()@candidate_cmp.float().T),b2c_pos)+_directional_multi_positive_loss(scale*(zc.float()@candidate_bgc.float().T),c2b_pos))
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite contrastive loss detected (scale={float(scale.detach()):.4g})")
    return loss


def main() -> None:
    args=parse_args(); logger=setup_logger("encoder_finetuning"); cfg=load_yaml(args.config)
    torch.manual_seed(int(cfg.get("seed",42)))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(cfg.get("seed",42)))
    cfg["finetune"]["esm_unfreeze"]=str(args.esm_unfreeze); cfg["finetune"]["molformer_unfreeze"]=str(args.molformer_unfreeze)
    domains={r["bgc_id"]:r for r in (json.loads(line) for line in args.domains.read_text().splitlines() if line.strip())}
    if not torch.cuda.is_available(): raise RuntimeError("Encoder fine-tuning requires a CUDA GPU")
    device=torch.device("cuda"); all_results=[]
    for fold in args.fold_ids:
        val_fold=fold%10+1
        frame=build_interactions(args.data_dir,splits_path=args.splits,cv_fold=fold,val_fold=val_fold)
        if "smiles" not in frame: raise ValueError("Interaction table must contain smiles for online MolFormer")
        out=Path("results/intermediate/encoder_finetuning")/args.run_name/f"fold_{fold}"; out.mkdir(parents=True,exist_ok=True)
        model=OnlineFineTuneCLIP(cfg).to(device)
        groups=[]
        projection=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(("esm.","molformer."))]
        esm=[p for p in model.esm.parameters() if p.requires_grad]; mol=[p for p in model.molformer.parameters() if p.requires_grad]
        if projection: groups.append({"params":projection,"lr":float(cfg["train"]["projection_lr"])})
        if esm: groups.append({"params":esm,"lr":float(cfg["train"]["encoder_lr_full"] if str(args.esm_unfreeze)=="full" else cfg["train"]["encoder_lr_partial"])})
        if mol: groups.append({"params":mol,"lr":float(cfg["train"]["encoder_lr_full"] if str(args.molformer_unfreeze)=="full" else cfg["train"]["encoder_lr_partial"])})
        optimizer=AdamW(groups,weight_decay=float(cfg["train"]["weight_decay"]))
        train=frame[frame.split=="train"]
        loader=DataLoader(PairDataset(train),batch_size=int(cfg["train"]["micro_batch_size"]),shuffle=True,collate_fn=_collate)
        positives={(str(r.bgc_id),str(r.compound_id)) for r in train.itertuples()}; accumulation=int(cfg["train"]["gradient_accumulation"])
        updates_per_epoch=(len(loader)+accumulation-1)//accumulation
        total_updates=updates_per_epoch*int(cfg["train"]["epochs"])
        scheduler=get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1,int(total_updates*float(cfg["train"].get("warmup_fraction",0.1)))),
            num_training_steps=total_updates,
        )
        # MolFormer linear attention is unstable under FP16 on V100. Encoder
        # fine-tuning therefore runs in FP32; frozen outputs are cached instead.
        scaler=torch.amp.GradScaler("cuda",enabled=False)
        best=-1.; best_epoch=0; best_state={}; history=[]; started=time.time()
        queue_bgc=[]; queue_cmp=[]; queue_bgc_ids=[]; queue_cmp_ids=[]; queue_size=int(cfg["train"].get("negative_queue_size",256))
        for epoch in range(1,int(cfg["train"]["epochs"])+1):
            model.train()
            if not any(p.requires_grad for p in model.esm.parameters()): model.esm.eval()
            if not any(p.requires_grad for p in model.molformer.parameters()): model.molformer.eval()
            optimizer.zero_grad(set_to_none=True); total=0.; count=0
            for step,batch in enumerate(loader,1):
                ub,ib=_unique(batch["bgc_id"]); uc,ic=_unique(batch["compound_id"])
                smiles_by_id={cid:smi for cid,smi in zip(batch["compound_id"],batch["smiles"])}
                zb=model.encode_bgcs([domains[x] for x in ub],device)[ib.to(device)]
                zc=model.encode_compounds([smiles_by_id[x] for x in uc],device)[ic.to(device)]
                loss=_queued_symmetric_loss(model,zb,zc,batch["bgc_id"],batch["compound_id"],queue_bgc,queue_cmp,queue_bgc_ids,queue_cmp_ids,positives)/accumulation
                scaler.scale(loss).backward(); total+=float(loss.item())*accumulation*len(batch["bgc_id"]); count+=len(batch["bgc_id"])
                retained_bgc=torch.cat(queue_bgc+[zb.detach()],dim=0)[-queue_size:]
                retained_cmp=torch.cat(queue_cmp+[zc.detach()],dim=0)[-queue_size:]
                queue_bgc_ids=(queue_bgc_ids+list(batch["bgc_id"]))[-queue_size:]
                queue_cmp_ids=(queue_cmp_ids+list(batch["compound_id"]))[-queue_size:]
                queue_bgc=[retained_bgc]; queue_cmp=[retained_cmp]
                if step%accumulation==0 or step==len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg["train"]["gradient_clip"]))
                    scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            val=_evaluate(model,frame,"val",domains,device,int(cfg["eval"]["entity_batch_size"]),int(cfg["eval"]["sim_batch_size"]))
            score=_score(val); history.append({"epoch":epoch,"loss":total/max(count,1),"val_mean_mrr":score})
            logger.info("fold=%d epoch=%d loss=%.4f val_mean_mrr=%.4f",fold,epoch,total/max(count,1),score)
            if score>best: best=score; best_epoch=epoch; best_state=model.trainable_state_dict()
        model.load_state_dict(best_state,strict=False)
        test=_evaluate(model,frame,"test",domains,device,int(cfg["eval"]["entity_batch_size"]),int(cfg["eval"]["sim_batch_size"]))
        torch.save(best_state,out/"trainable_delta_best.pt")
        result={"fold":fold,"val_fold":val_fold,"best_epoch":best_epoch,"best_val_mean_mrr":best,"retrieval_test":test,"history":history,"esm":model.esm_status,"molformer":model.molformer_status,"elapsed_seconds":time.time()-started,"config":cfg}
        save_json(result,out/"result.json"); all_results.append(result)
        del model; torch.cuda.empty_cache()
    save_json({"run_name":args.run_name,"folds":all_results},Path("results/intermediate/encoder_finetuning")/args.run_name/"summary.json")


if __name__=="__main__": main()
