from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
# The cluster's umap/numba installation cannot create a cache locator on NFS.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from projects.mibig_bgc_np.data.datasets import build_interactions
from projects.mibig_bgc_np.scripts.eval_retrieval import _load_model
from projects.mibig_bgc_np.training.contrastive_trainer import build_unique_embeddings

RDLogger.DisableLog("rdApp.*")


def _reducers(seed: int, n: int):
    perplexity=min(30.0,max(5.0,(n-1)/3.0))
    return {
        "pca": lambda x,metric:PCA(n_components=2,random_state=seed).fit_transform(x),
        "umap": lambda x,metric:umap.UMAP(n_components=2,n_neighbors=min(15,n-1),min_dist=.1,
            metric=metric,random_state=seed).fit_transform(x),
        "tsne": lambda x,metric:TSNE(n_components=2,metric=metric,perplexity=perplexity,
            init="random",learning_rate="auto",random_state=seed,max_iter=1500).fit_transform(x),
    }


def _draw(ax,xy,scores,true_idx,top10,title,norm):
    points=ax.scatter(xy[:,0],xy[:,1],c=scores,cmap="viridis",norm=norm,s=25,alpha=.78,linewidths=0)
    ax.scatter(xy[top10,0],xy[top10,1],facecolors="none",edgecolors="red",s=78,linewidths=1.25,
               label="CLIP top-10")
    ax.scatter(xy[true_idx,0],xy[true_idx,1],marker="*",c="red",edgecolors="black",s=230,
               linewidths=.7,zorder=5,label="True NP")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    return points


def main() -> None:
    p=argparse.ArgumentParser(description="Six candidate-space views for a failed strict-CV test query")
    p.add_argument("--run_root",type=Path,default=Path("results/best_esm_domains_molformer_strict_cv10"))
    p.add_argument("--outdir",type=Path,default=Path("results/paper_plots/query_manifolds"))
    p.add_argument("--seed",type=int,default=42); args=p.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.run_root/"summary.json").read_text()); cache=Path(summary["cache_dir"])
    bgc_cache=torch.load(cache/"bgc_features.pt",map_location="cpu"); cmp_cache=torch.load(cache/"compound_features.pt",map_location="cpu")
    eligible=[]; fold_payload={}
    for fold in range(1,11):
        folder=args.run_root/f"fold_{fold}"; fs=json.loads((folder/"fold_summary.json").read_text())
        interactions=build_interactions(summary["data_dir"],splits_path=summary["splits_path"],cv_fold=fold,val_fold=fs.get("val_fold"))
        model,_=_load_model(folder/"contrastive_model_best.pt",torch.device("cpu"))
        bi,ci,zb,zc,pairs=build_unique_embeddings(model,interactions,"test",bgc_cache,cmp_cache,torch.device("cpu"))
        bgcs=list(bi); compounds=list(ci); truth={b:[] for b in bgcs}
        for i,j in pairs:truth[bgcs[i]].append(compounds[j])
        score=model.get_logit_scale().detach().cpu()*(zb@zc.t())
        for i,bgc in enumerate(bgcs):
            unique=sorted(set(truth[bgc]))
            if len(unique)!=1:
                continue
            true_idx=ci[unique[0]]
            true_rank=int((torch.argsort(score[i],descending=True)==true_idx).nonzero()[0,0])+1
            if true_rank>10:
                eligible.append((fold,bgc,true_rank))
        fold_payload[fold]=(interactions,model,bi,ci,score)
    rng=np.random.default_rng(args.seed); fold,bgc,_selected_rank=eligible[int(rng.integers(len(eligible)))]
    interactions,model,bi,ci,score=fold_payload[fold]; bgcs=list(bi); compounds=list(ci); row=bi[bgc]
    test=interactions[interactions.split.astype(str).str.lower()=="test"]
    true_id=str(test[test.bgc_id.astype(str)==bgc].compound_id.astype(str).drop_duplicates().iloc[0]); true_idx=ci[true_id]
    scores=score[row].numpy(); top10=torch.topk(score[row],k=min(10,len(compounds))).indices.numpy()
    smiles=test[["compound_id","smiles"]].drop_duplicates("compound_id").set_index("compound_id").smiles.astype(str).to_dict()
    maccs=[]
    for cid in compounds:
        mol=Chem.MolFromSmiles(smiles.get(cid,cid));
        if mol is None:raise ValueError(f"Invalid candidate SMILES for {cid}")
        fp=MACCSkeys.GenMACCSKeys(mol); maccs.append(np.asarray(list(fp),dtype=np.uint8))
    maccs=np.stack(maccs); molformer=torch.stack([cmp_cache[x].float() for x in compounds]).numpy()
    inputs={"maccs":(maccs,"jaccard","MACCS fingerprints"),"molformer":(molformer,"cosine","MolFormer embeddings")}
    reducers=_reducers(args.seed,len(compounds)); embeddings={}
    for modality,(features,metric,label) in inputs.items():
        for method,reduce in reducers.items():embeddings[(method,modality)]=reduce(features,metric)
    norm=Normalize(vmin=float(scores.min()),vmax=float(scores.max()))
    for (method,modality),xy in embeddings.items():
        fig,ax=plt.subplots(figsize=(6.3,5.2)); pts=_draw(ax,xy,scores,true_idx,top10,
            f"{method.upper()} — {inputs[modality][2]}",norm)
        ax.legend(frameon=False,loc="best"); fig.colorbar(pts,ax=ax,label="CLIP score",fraction=.046,pad=.04); fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(args.outdir/f"strict_{bgc}_{method}_{modality}.{ext}",dpi=300,bbox_inches="tight")
        plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,15)); last=None
    for r,method in enumerate(("pca","umap","tsne")):
        for c,modality in enumerate(("maccs","molformer")):
            last=_draw(axes[r,c],embeddings[(method,modality)],scores,true_idx,top10,
                f"{method.upper()} — {inputs[modality][2]}",norm)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle(f"Strict CV test query {bgc} (fold {fold})",y=.995)
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.975),ncol=2,frameon=False)
    fig.subplots_adjust(top=.925,hspace=.22,wspace=.12)
    fig.colorbar(last,ax=axes.ravel().tolist(),label="CLIP score",fraction=.02,pad=.02)
    fig.savefig(args.outdir/f"strict_{bgc}_six_manifolds.png",dpi=300,bbox_inches="tight"); fig.savefig(args.outdir/f"strict_{bgc}_six_manifolds.pdf",bbox_inches="tight"); plt.close(fig)
    rank=int((torch.argsort(score[row],descending=True)==true_idx).nonzero()[0,0])+1
    metadata={"seed":args.seed,"fold":fold,"bgc_id":bgc,"n_eligible_queries":len(eligible),"n_candidates":len(compounds),
        "true_compound_id":true_id,"true_rank":rank,"top1_compound_id":compounds[int(score[row].argmax())],
        "top10_compound_ids":[compounds[int(i)] for i in top10],"color":"CLIP score","true_marker":"red star",
        "top10_marker":"red outline","maccs_metrics":{"umap":"jaccard","tsne":"jaccard"},
        "molformer_metrics":{"umap":"cosine","tsne":"cosine"},"pca_metric":"not applicable"}
    (args.outdir/f"strict_{bgc}_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":main()
