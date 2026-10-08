"""The crux experiment.

Adds the matched baseline the original CGPR comparison is missing — kNN
propagation WITH the same entropy-based uncertainty gating CGPR itself uses
(UGSP) — tuned per seed on the same val partitions, over the same protocol.
CGPR's published kNN baseline is the *non*-uncertainty variant, so the
published gain conflates "concepts help" with "uncertainty gating helps".

Also evaluates concept evidence fusion (ours), tuned the same way.
"""
import itertools, os, pickle
import numpy as np, pandas as pd
from scipy.sparse import eye
from cgpr_lab import *
from concept_hypergraph import _row_normalize

SEEDS = [42, 43, 44, 45, 46]
CACHE = "/tmp/claude-0/-workspace-src/919fc61a-352b-4d23-8d3f-5b610f63b2bf/scratchpad/knncache"
os.makedirs(CACHE, exist_ok=True)
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
KS = [1, 2]
BETAS = [0.5, 1.0, 2.0, 4.0]
TAUS = [0.005, 0.01, 0.02, 0.05]
WS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
AGGS = ["sum", "mean"]
METHOD = "kNN+Concept Hypergraph + Uncertainty"

def knn(dataset, tag, seed, split, embs, k):
    f = f"{CACHE}/{dataset}_{tag}_{seed}_{split}_k{k}.pkl"
    if os.path.exists(f):
        with open(f, "rb") as fh: return pickle.load(fh)
    W = build_knn_graph(embs, k=k, metric="cosine")
    with open(f, "wb") as fh: pickle.dump(W, fh)
    return W

def prop(W, Y, alpha, steps, beta=None):
    P = _row_normalize(W + alpha * eye(W.shape[0], format="csr"))
    if beta is None:
        return chsp_propagate(P, Y, steps=steps)
    return uchsp_propagate(P, Y, steps=steps, beta=beta)["scores"]

rows = []
for dataset in ["wikiart", "mp100k"]:
    for tag, label in BACKBONES:
        p = Pool(dataset, tag)
        cfg = p.cgpr_cfg[METHOD]
        C = len(p.class_names)
        cols_per_class = [np.where(p.phrase_style_idx == c)[0] for c in range(C)]
        k = p.sp_cfg["k_graph"]

        def profile(embs, tau):
            S = embs @ p.concept_embs.T / tau
            S -= S.max(1, keepdims=True)
            A = np.exp(S); A /= A.sum(1, keepdims=True)
            return A

        def evidence(A, agg):
            Yc = np.zeros((A.shape[0], C))
            for c, cc in enumerate(cols_per_class):
                if len(cc): Yc[:, c] = A[:, cc].sum(1) if agg == "sum" else A[:, cc].mean(1)
            return Yc / (Yc.sum(1, keepdims=True) + 1e-12)

        per_seed = []
        for seed in SEEDS:
            vi, ti = p.split(seed)
            P_ = {}
            for nm, idx in (("val", vi), ("test", ti)):
                e, Y, l = p.embs[idx], p.Y[idx], p.labels[idx]
                P_[nm] = (e, Y, l, knn(dataset, tag, seed, nm, e, k))
            bacc = lambda s, l: metrics(s, l, p.class_names)["bacc"]

            # --- baseline 1: plain SP (no uncertainty) — the paper's kNN baseline
            sp_cfg = max(itertools.product(ALPHAS, KS),
                         key=lambda c: bacc(prop(P_["val"][3], P_["val"][1], c[0], c[1]), P_["val"][2]))
            # --- baseline 2: UGSP (kNN + uncertainty) — the MATCHED baseline, missing from the paper
            ug_cfg = max(itertools.product(ALPHAS, KS, BETAS),
                         key=lambda c: bacc(prop(P_["val"][3], P_["val"][1], c[0], c[1], c[2]), P_["val"][2]))
            # --- ours: concept evidence fusion, then UGSP propagation
            def ours_score(part, c):
                tau, w, agg, al, K_, be = c
                e, Y, l, W = part
                Yf = (1 - w) * Y + w * evidence(profile(e, tau), agg)
                return prop(W, Yf, al, K_, be)
            our_grid = list(itertools.product(TAUS, WS, AGGS, [ug_cfg[0]], [ug_cfg[1]], [ug_cfg[2]]))
            our_cfg = max(our_grid, key=lambda c: bacc(ours_score(P_["val"], c), P_["val"][2]))

            e, Y, l, W = P_["test"]
            B = build_incidence_matrix(e, p.concept_embs, top_r=cfg["top_r"],
                                       normalize=cfg["normalize_incidence"])
            r = dict(seed=seed,
                     vanilla=bacc(Y, l),
                     sp_knn=bacc(prop(W, Y, *sp_cfg), l),
                     ugsp_knn=bacc(prop(W, Y, *ug_cfg), l),
                     cgpr=bacc(uchsp_propagate(build_combined_propagation_matrix(W, B, alpha=cfg["alpha"]),
                                               Y, steps=cfg["propagation_steps"], beta=cfg["beta"])["scores"], l),
                     ours=bacc(ours_score(P_["test"], our_cfg), l),
                     tau=our_cfg[0], w=our_cfg[1], agg=our_cfg[2])
            per_seed.append(r)
        d = pd.DataFrame(per_seed)
        rows.append(dict(dataset=dataset, backbone=label,
                         **{c: d[c].mean() for c in ["vanilla","sp_knn","ugsp_knn","cgpr","ours"]},
                         ours_std=d["ours"].std(), cgpr_std=d["cgpr"].std(),
                         tau=d["tau"].mode()[0], w=d["w"].mode()[0], agg=d["agg"].mode()[0]))
        print(f"{dataset:8s} {label:14s} van={rows[-1]['vanilla']:.4f} SP={rows[-1]['sp_knn']:.4f} "
              f"UGSP={rows[-1]['ugsp_knn']:.4f} CGPR={rows[-1]['cgpr']:.4f} OURS={rows[-1]['ours']:.4f}", flush=True)
pd.DataFrame(rows).to_csv("crux.csv", index=False)
print("\nsaved crux.csv")
