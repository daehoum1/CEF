"""How much of the combined graph is actually concept-restricted?"""
import numpy as np
from cgpr_lab import *

METHOD = "kNN+Concept Hypergraph + Uncertainty"
rows = []
for dataset in ["wikiart", "mp100k"]:
    for tag, label in BACKBONES:
        p = Pool(dataset, tag)
        cfg = p.cgpr_cfg[METHOD]
        vi, ti = p.split(42)
        embs, Y, lab = p.embs[ti], p.Y[ti], p.labels[ti]
        W = build_knn_graph(embs, k=p.sp_cfg["k_graph"], metric="cosine")
        B = build_incidence_matrix(embs, p.concept_embs, top_r=cfg["top_r"],
                                   normalize=cfg["normalize_incidence"])
        # masked graph WITHOUT the starved fallback
        C = concept_comembership(B); C.data = np.ones_like(C.data)
        Wm = W.multiply(C).tocsr(); Wm.eliminate_zeros()
        deg_knn = np.asarray((W > 0).sum(1)).ravel()
        deg_m = np.asarray((Wm > 0).sum(1)).ravel()
        starved = deg_m == 0
        # of the surviving edges, how many were kept?
        kept_frac = Wm.nnz / W.nnz
        # among non-starved nodes, avg degree
        rows.append(dict(dataset=dataset, backbone=label, N=len(ti), top_r=cfg["top_r"],
                         n_concepts=len(p.phrases),
                         starved_pct=100*starved.mean(),
                         edges_kept_pct=100*kept_frac,
                         deg_knn=deg_knn.mean(),
                         deg_masked_nonstarved=deg_m[~starved].mean() if (~starved).any() else 0,
                         # fraction of *edges* in the final graph that come from fallback rows
                         fallback_edge_pct=100*deg_knn[starved].sum()/(deg_knn[starved].sum()+deg_m[~starved].sum())))
import pandas as pd
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
df.to_csv("diag_mask.csv", index=False)
