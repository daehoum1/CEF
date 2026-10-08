"""Does the discriminability filter still pay off once concepts are used as
evidence rather than as hard hyperedges?  Compares the filtered dictionary
(40 / 36 concepts) against the full LLM dictionary (150 / 200)."""
import itertools, os, pickle, sys
import numpy as np, pandas as pd
from scipy.sparse import eye
from cgpr_lab import *
from concept_hypergraph import _row_normalize

SEEDS = [42, 43, 44, 45, 46]
CACHE = "/tmp/claude-0/-workspace-src/919fc61a-352b-4d23-8d3f-5b610f63b2bf/scratchpad/knncache"
ALPHAS, KS, BETAS = [0.1, 0.3, 0.5, 0.7, 0.9], [1, 2], [0.5, 1.0, 2.0, 4.0]
TAUS = [0.0025, 0.005, 0.01, 0.02, 0.05]
WS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
AGGS = ["sum", "mean"]

def knn(dataset, tag, seed, split, embs, k):
    f = f"{CACHE}/{dataset}_{tag}_{seed}_{split}_k{k}.pkl"
    if os.path.exists(f):
        with open(f, "rb") as fh: return pickle.load(fh)
    W = build_knn_graph(embs, k=k, metric="cosine")
    with open(f, "wb") as fh: pickle.dump(W, fh)
    return W

def prop(W, Y, alpha, steps, beta):
    P = _row_normalize(W + alpha * eye(W.shape[0], format="csr"))
    return uchsp_propagate(P, Y, steps=steps, beta=beta)["scores"]

rows = []
for dataset in ["wikiart", "mp100k"]:
    for tag, label in BACKBONES:
        p = Pool(dataset, tag)
        k = p.sp_cfg["k_graph"]
        C = len(p.class_names)
        d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
        DICTS = {
            "filtered": (p.concept_embs, p.phrase_style_idx),
            "full": (d["embs"], np.array([p.class_names.index(s) for s in d["styles"]])),
        }
        for dict_name, (cembs, cstyle) in DICTS.items():
            cols_per_class = [np.where(cstyle == c)[0] for c in range(C)]

            def evidence(embs, tau, agg):
                S = embs @ cembs.T / tau
                S -= S.max(1, keepdims=True)
                A = np.exp(S); A /= A.sum(1, keepdims=True)
                Yc = np.zeros((A.shape[0], C))
                for c, cc in enumerate(cols_per_class):
                    if len(cc): Yc[:, c] = A[:, cc].sum(1) if agg == "sum" else A[:, cc].mean(1)
                return Yc / (Yc.sum(1, keepdims=True) + 1e-12)

            vals = []
            for seed in SEEDS:
                vi, ti = p.split(seed)
                P_ = {}
                for nm, idx in (("val", vi), ("test", ti)):
                    e, Y, l = p.embs[idx], p.Y[idx], p.labels[idx]
                    P_[nm] = (e, Y, l, knn(dataset, tag, seed, nm, e, k))
                bacc = lambda s, l: metrics(s, l, p.class_names)["bacc"]
                ug = max(itertools.product(ALPHAS, KS, BETAS),
                         key=lambda c: bacc(prop(P_["val"][3], P_["val"][1], *c), P_["val"][2]))
                def sc(part, c):
                    tau, w, agg = c
                    e, Y, l, W = part
                    return prop(W, (1 - w) * Y + w * evidence(e, tau, agg), *ug)
                best = max(itertools.product(TAUS, WS, AGGS),
                           key=lambda c: bacc(sc(P_["val"], c), P_["val"][2]))
                vals.append((bacc(sc(P_["test"], best), P_["test"][2]), best))
            baccs = [v[0] for v in vals]
            rows.append(dict(dataset=dataset, backbone=label, dict=dict_name,
                             n_concepts=cembs.shape[0], bacc=float(np.mean(baccs)),
                             std=float(np.std(baccs)),
                             modal=str(pd.Series([v[1] for v in vals]).mode()[0])))
            print(f"{dataset:8s} {label:14s} {dict_name:8s} E={cembs.shape[0]:3d} "
                  f"bAcc={np.mean(baccs):.4f}  {rows[-1]['modal']}", flush=True)
pd.DataFrame(rows).to_csv("dict_ablation.csv", index=False)
