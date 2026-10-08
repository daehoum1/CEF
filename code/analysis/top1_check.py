"""Is the Top-1 regression on WikiArt an artifact of selecting hyperparameters
on validation *balanced* accuracy (as the CGPR protocol does), or a genuine
limitation?  Re-selects on val Top-1 and reports test Top-1."""
import itertools, os, pickle
import numpy as np, pandas as pd
from scipy.sparse import eye
from cgpr_lab import *
from concept_hypergraph import _row_normalize

SEEDS = [42, 43, 44, 45, 46]
CACHE = "/tmp/claude-0/-workspace-src/919fc61a-352b-4d23-8d3f-5b610f63b2bf/scratchpad/knncache"
ALPHAS, KS, BETAS = [0.1,0.3,0.5,0.7,0.9], [1,2], [0.5,1.0,2.0,4.0]
TAUS = [0.0025,0.005,0.01,0.02,0.05]; WS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]; DEB=[False,True]
CG = "kNN+Concept Hypergraph + Uncertainty"

def knn(ds,tag,seed,split,e,k):
    f=f"{CACHE}/{ds}_{tag}_{seed}_{split}_k{k}.pkl"
    if os.path.exists(f):
        with open(f,"rb") as fh: return pickle.load(fh)
    W=build_knn_graph(e,k=k,metric="cosine")
    with open(f,"wb") as fh: pickle.dump(W,fh)
    return W

def pu(W,Y,a,K,b): return uchsp_propagate(_row_normalize(W+a*eye(W.shape[0],format="csr")),Y,steps=K,beta=b)["scores"]

rows=[]
for dataset in ["wikiart","mp100k"]:
    for tag,label in BACKBONES:
        p=Pool(dataset,tag); k=p.sp_cfg["k_graph"]; C=len(p.class_names)
        cg=p.cgpr_cfg[CG]
        d=np.load(f"raw_concept_embs/{dataset}_{tag}.npz",allow_pickle=True)
        ce=d["embs"]; cs=np.array([p.class_names.index(s) for s in d["styles"]])
        cols=[np.where(cs==c)[0] for c in range(C)]
        def ev(e,tau,db,mu):
            S=e@ce.T
            if db: S=S-mu
            S=S/tau; S-=S.max(1,keepdims=True); A=np.exp(S); A/=A.sum(1,keepdims=True)
            Yc=np.zeros((A.shape[0],C))
            for c,cc in enumerate(cols):
                if len(cc): Yc[:,c]=A[:,cc].sum(1)
            return Yc/(Yc.sum(1,keepdims=True)+1e-12)
        acc={m:[] for m in ["cef_b","cef_t","cgpr","van","sp_t","ugsp_t"]}
        for seed in SEEDS:
            vi,ti=p.split(seed); P_={}
            for nm,idx in (("val",vi),("test",ti)):
                e,Y,l=p.embs[idx],p.Y[idx],p.labels[idx]; P_[nm]=(e,Y,l,knn(dataset,tag,seed,nm,e,k))
            mu=(P_["val"][0]@ce.T).mean(0,keepdims=True)
            MM=lambda s,l: metrics(s,l,p.class_names)
            for sel in ["bacc","top1"]:
                ug=max(itertools.product(ALPHAS,KS,BETAS),key=lambda c:MM(pu(P_["val"][3],P_["val"][1],*c),P_["val"][2])[sel])
                if sel=="top1":
                    # baselines selected on the SAME metric they are reported on
                    acc["ugsp_t"].append(MM(pu(P_["test"][3],P_["test"][1],*ug),P_["test"][2])["top1"])
                    sp=max(itertools.product(ALPHAS,KS),key=lambda c:MM(
                        chsp_propagate(_row_normalize(P_["val"][3]+c[0]*eye(P_["val"][3].shape[0],format="csr")),
                                       P_["val"][1],steps=c[1]),P_["val"][2])["top1"])
                    acc["sp_t"].append(MM(chsp_propagate(_row_normalize(
                        P_["test"][3]+sp[0]*eye(P_["test"][3].shape[0],format="csr")),
                        P_["test"][1],steps=sp[1]),P_["test"][2])["top1"])
                def sc(pt,c):
                    tau,w,db=c; e,Y,l,W=pt
                    return pu(W,(1-w)*Y+w*ev(e,tau,db,mu),*ug)
                best=max(itertools.product(TAUS,WS,DEB),key=lambda c:MM(sc(P_["val"],c),P_["val"][2])[sel])
                acc["cef_b" if sel=="bacc" else "cef_t"].append(MM(sc(P_["test"],best),P_["test"][2])["top1"])
            te=P_["test"]
            B=build_incidence_matrix(te[0],p.concept_embs,top_r=cg["top_r"],normalize=cg["normalize_incidence"])
            acc["cgpr"].append(MM(uchsp_propagate(build_combined_propagation_matrix(te[3],B,alpha=cg["alpha"]),
                                te[1],steps=cg["propagation_steps"],beta=cg["beta"])["scores"],te[2])["top1"])
            acc["van"].append(MM(te[1],te[2])["top1"])
        r=dict(dataset=dataset,backbone=label,**{m:float(np.mean(v)) for m,v in acc.items()})
        rows.append(r)
        print(f"{dataset:8s} {label:14s} Top-1: van={r['van']:.4f} CGPR={r['cgpr']:.4f} "
              f"CEF(bAcc-sel)={r['cef_b']:.4f} CEF(Top1-sel)={r['cef_t']:.4f}",flush=True)
pd.DataFrame(rows).to_csv("top1_check.csv",index=False)
