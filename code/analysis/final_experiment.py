"""Definitive experiment for the follow-up paper.

Method (CEF — Concept-Evidence Fusion):
  1. dense concept-affinity profile for EVERY image (no top-r membership,
     so no coverage starvation)
  2. per-concept debiasing to suppress hub concepts  [val-tuned on/off]
  3. aggregate into per-class concept evidence, fuse with the VLM's zero-shot Y
  4. uncertainty-gated kNN propagation of the fused scores

Baselines: Vanilla, SP-kNN, UGSP-kNN (the matched baseline missing from the
CGPR paper), CGPR (published, re-run under this protocol).
Ablations: filtered vs full dictionary; debias on/off; evidence w/o fusion;
fusion w/o propagation.

Everything (propagation alpha/K/beta, tau, w, debias, dictionary) is tuned on
each seed's val partition and evaluated on that seed's test partition.
"""
import itertools, os, pickle, json
import numpy as np, pandas as pd
from scipy.sparse import eye
from cgpr_lab import *
from concept_hypergraph import _row_normalize

SEEDS = [42, 43, 44, 45, 46]
CACHE = "/tmp/claude-0/-workspace-src/919fc61a-352b-4d23-8d3f-5b610f63b2bf/scratchpad/knncache"
ALPHAS, KS, BETAS = [0.1, 0.3, 0.5, 0.7, 0.9], [1, 2], [0.5, 1.0, 2.0, 4.0]
TAUS = [0.0025, 0.005, 0.01, 0.02, 0.05]
WS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEBIAS = [False, True]
CGPR_METHOD = "kNN+Concept Hypergraph + Uncertainty"

def knn(ds, tag, seed, split, embs, k):
    f = f"{CACHE}/{ds}_{tag}_{seed}_{split}_k{k}.pkl"
    if os.path.exists(f):
        with open(f, "rb") as fh: return pickle.load(fh)
    W = build_knn_graph(embs, k=k, metric="cosine")
    with open(f, "wb") as fh: pickle.dump(W, fh)
    return W

def prop_unc(W, Y, a, K, b):
    return uchsp_propagate(_row_normalize(W + a*eye(W.shape[0], format="csr")), Y, steps=K, beta=b)["scores"]

def prop_plain(W, Y, a, K):
    return chsp_propagate(_row_normalize(W + a*eye(W.shape[0], format="csr")), Y, steps=K)

per_seed_rows, summary_rows = [], []
for dataset in ["wikiart", "mp100k"]:
    for tag, label in BACKBONES:
        p = Pool(dataset, tag)
        k = p.sp_cfg["k_graph"]; C = len(p.class_names)
        cgpr_cfg = p.cgpr_cfg[CGPR_METHOD]
        draw = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
        DICT = {
            "full": (draw["embs"], np.array([p.class_names.index(s) for s in draw["styles"]])),
            "filtered": (p.concept_embs, p.phrase_style_idx),
        }
        COLS = {n: [np.where(cs == c)[0] for c in range(C)] for n, (_, cs) in DICT.items()}
        # per-concept mean similarity is estimated on the val partition only
        def evidence(embs, dict_name, tau, debias, mu):
            ce = DICT[dict_name][0]
            S = embs @ ce.T
            if debias: S = S - mu[dict_name]
            S = S / tau
            S -= S.max(1, keepdims=True)
            A = np.exp(S); A /= A.sum(1, keepdims=True)
            Yc = np.zeros((A.shape[0], C))
            for c, cc in enumerate(COLS[dict_name]):
                if len(cc): Yc[:, c] = A[:, cc].sum(1)
            return Yc / (Yc.sum(1, keepdims=True) + 1e-12)

        for seed in SEEDS:
            vi, ti = p.split(seed)
            part = {}
            for nm, idx in (("val", vi), ("test", ti)):
                e, Y, l = p.embs[idx], p.Y[idx], p.labels[idx]
                part[nm] = (e, Y, l, knn(dataset, tag, seed, nm, e, k))
            mu = {n: (part["val"][0] @ DICT[n][0].T).mean(0, keepdims=True) for n in DICT}
            M = lambda s, l: metrics(s, l, p.class_names)
            bacc = lambda s, l: M(s, l)["bacc"]

            sp_cfg = max(itertools.product(ALPHAS, KS),
                         key=lambda c: bacc(prop_plain(part["val"][3], part["val"][1], *c), part["val"][2]))
            ug_cfg = max(itertools.product(ALPHAS, KS, BETAS),
                         key=lambda c: bacc(prop_unc(part["val"][3], part["val"][1], *c), part["val"][2]))

            def fused(pt, dict_name, tau, w, debias):
                e, Y, l, W = pt
                return (1-w)*Y + w*evidence(e, dict_name, tau, debias, mu)
            def full_pipe(pt, cfg):
                dn, tau, w, db = cfg
                return prop_unc(pt[3], fused(pt, dn, tau, w, db), *ug_cfg)

            grid_full = [("full", t, w, db) for t, w, db in itertools.product(TAUS, WS, DEBIAS)]
            grid_filt = [("filtered", t, w, db) for t, w, db in itertools.product(TAUS, WS, DEBIAS)]
            best = max(grid_full, key=lambda c: bacc(full_pipe(part["val"], c), part["val"][2]))
            best_filt = max(grid_filt, key=lambda c: bacc(full_pipe(part["val"], c), part["val"][2]))
            best_nodb = max([c for c in grid_full if not c[3]],
                            key=lambda c: bacc(full_pipe(part["val"], c), part["val"][2]))
            best_nofuse = max(TAUS, key=lambda t: bacc(
                prop_unc(part["val"][3], evidence(part["val"][0], "full", t, best[3], mu), *ug_cfg), part["val"][2]))

            te = part["test"]; l = te[2]
            B = build_incidence_matrix(te[0], p.concept_embs, top_r=cgpr_cfg["top_r"],
                                       normalize=cgpr_cfg["normalize_incidence"])
            row = dict(dataset=dataset, backbone=label, seed=seed,
                       cfg=json.dumps(dict(dict=best[0], tau=best[1], w=best[2], debias=best[3],
                                           alpha=ug_cfg[0], K=ug_cfg[1], beta=ug_cfg[2])))
            for name, sc in [
                ("Vanilla", te[1]),
                ("SP-kNN", prop_plain(te[3], te[1], *sp_cfg)),
                ("UGSP-kNN", prop_unc(te[3], te[1], *ug_cfg)),
                ("CGPR", uchsp_propagate(build_combined_propagation_matrix(te[3], B, alpha=cgpr_cfg["alpha"]),
                                         te[1], steps=cgpr_cfg["propagation_steps"], beta=cgpr_cfg["beta"])["scores"]),
                ("CEF (ours)", full_pipe(te, best)),
                ("CEF w/ filtered dict", full_pipe(te, best_filt)),
                ("CEF w/o debias", full_pipe(te, best_nodb)),
                ("CEF w/o fusion (evidence only)", prop_unc(te[3], evidence(te[0], "full", best_nofuse, best[3], mu), *ug_cfg)),
                ("CEF w/o propagation", fused(te, *best[:3], best[3])),
            ]:
                m = M(sc, l)
                row[f"{name}|bacc"] = m["bacc"]; row[f"{name}|top1"] = m["top1"]
            per_seed_rows.append(row)
        d = pd.DataFrame([r for r in per_seed_rows if r["dataset"] == dataset and r["backbone"] == label])
        s = dict(dataset=dataset, backbone=label)
        for col in [c for c in d.columns if "|" in c]:
            s[col + "_mean"] = d[col].mean(); s[col + "_std"] = d[col].std()
        summary_rows.append(s)
        print(f"{dataset:8s} {label:14s} " + "  ".join(
            f"{n}={d[n+'|bacc'].mean():.4f}" for n in ["Vanilla","UGSP-kNN","CGPR","CEF (ours)"]), flush=True)

pd.DataFrame(per_seed_rows).to_csv("final_per_seed.csv", index=False)
pd.DataFrame(summary_rows).to_csv("final_summary.csv", index=False)
print("\nsaved final_per_seed.csv / final_summary.csv")
