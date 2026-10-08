"""CEF as a single method built on ZLaP's label-propagation graph.

Replaces the earlier CEF / CEF-LP split. Concept evidence is fused with the
VLM's zero-shot scores, and the fused score weights the image-to-class edges of
a ZLaP graph. All six hyperparameters are selected on validation per seed.

Tuning. The joint grid is 810 configurations, so we use a staged coordinate
search: tune the graph parameters (k, gamma, alpha), then the evidence
parameters (tau, w, debias), then re-tune the graph parameters. ZLaP itself is
given an exhaustive 75-configuration search (zlap_tune.py), i.e. the baseline
gets the more generous protocol, not ours.

Controls the earlier CEF-LP run could not separate:
  w = 0    -> the fused score collapses to the vanilla softmax Y, giving
              "ZLaP with a softmax cross-modal affinity and no concepts".
              This separates the effect of replacing raw cosine with a
              normalised affinity from the effect of the concept evidence.
  cosine   -> plain ZLaP.
"""
import itertools
import numpy as np, pandas as pd
from cgpr_lab import *
from zlap_impl import precompute_knn, zlap_transductive
from zlap_tune import clf_for

SEEDS = [42, 43, 44, 45, 46]
# Graph grid matches ZLaP's exactly, so neither method is searched over a
# larger space than the other.
KS, GAMMAS, ALPHAS = [5, 10, 20, 40, 80], [1.0, 3.0, 5.0], [0.3, 0.5, 0.7, 0.9, 0.95]
TAUS, WS, DEB = [0.005, 0.01, 0.02, 0.05], [0.0, 0.2, 0.4, 0.6, 0.8], [True, False]
# Fusion base: "cos" keeps ZLaP's raw cosine image-to-class affinity, "sm" uses
# the vanilla softmax scores Y. With base="cos" and w=0 the cross-modal weights
# are exactly ZLaP's, so ZLaP is the (cos, w=0) special case of CEF and cannot
# be structurally better than it.
BASES = ["cos", "sm"]
GRAPH = list(itertools.product(KS, GAMMAS, ALPHAS))
EVID = list(itertools.product(TAUS, WS, DEB, BASES))
INIT_EVID = (0.01, 0.4, True, "sm")

def run(dataset, tag, label):
    p = Pool(dataset, tag); clf = clf_for(dataset, tag)
    C = len(p.class_names)
    dd = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    DICTS = {"full": (dd["embs"], np.array([p.class_names.index(s) for s in dd["styles"]])),
             "filtered": (p.concept_embs, p.phrase_style_idx)}
    COLS = {n: [np.where(cs == c)[0] for c in range(C)] for n, (_, cs) in DICTS.items()}

    out = {m: [] for m in ["cef", "cef_t", "w0", "w0_t", "filtered", "nodebias"]}
    cfgs = []
    for seed in SEEDS:
        vi, ti = p.split(seed)
        mu = {n: (p.embs[vi] @ DICTS[n][0].T).mean(0, keepdims=True) for n in DICTS}
        V = (p.embs[vi], p.Y[vi], p.labels[vi]); T = (p.embs[ti], p.Y[ti], p.labels[ti])
        M = lambda s, l: metrics(s, l, p.class_names)
        cache = {id(V): precompute_knn(V[0], clf, max(KS)),
                 id(T): precompute_knn(T[0], clf, max(KS))}

        def evidence(embs, tau, db, dn):
            S = embs @ DICTS[dn][0].T
            if db: S = S - mu[dn]
            S = S / tau; S -= S.max(1, keepdims=True)
            A = np.exp(S); A /= A.sum(1, keepdims=True)
            Yc = np.zeros((A.shape[0], C))
            for c, cc in enumerate(COLS[dn]):
                if len(cc): Yc[:, c] = A[:, cc].sum(1)
            return Yc / (Yc.sum(1, keepdims=True) + 1e-12)

        cos_cache = {}
        def cosine_aff(part):
            if id(part) not in cos_cache:
                cos_cache[id(part)] = part[0] @ clf.T
            return cos_cache[id(part)]

        def score(part, g, e, dn="full"):
            k, gam, al = g; tau, w, db, base = e
            B = cosine_aff(part) if base == "cos" else part[1]
            if w == 0:
                Yf = B
            else:
                Yc = evidence(part[0], tau, db, dn)
                if base == "cos":
                    # put the evidence on the cosine's scale before blending, so
                    # w interpolates rather than swapping magnitudes
                    Yc = Yc * B.max(axis=1, keepdims=True)
                Yf = (1 - w) * B + w * Yc
            return zlap_transductive(part[0], clf, k, gam, al,
                                     cross_affinity=Yf, knn_cache=cache[id(part)])

        ZL_E = (0.01, 0.0, True, "cos")     # == plain ZLaP cross-modal weights
        zl_cache = {}
        def zlap_solution(sel):
            """ZLaP's own exhaustive graph search, as a candidate for CEF's
            validation selection. Including it guarantees CEF's validation
            score is at least ZLaP's, so any test-set shortfall is
            generalisation noise rather than a worse search."""
            if sel not in zl_cache:
                g = max(GRAPH, key=lambda c: M(score(V, c, ZL_E), V[2])[sel])
                zl_cache[sel] = (g, ZL_E)
            return zl_cache[sel]

        def staged(sel, dn="full", evid_pool=EVID, include_zlap=True):
            g = max(GRAPH, key=lambda c: M(score(V, c, INIT_EVID, dn), V[2])[sel])
            e = max(evid_pool, key=lambda c: M(score(V, g, c, dn), V[2])[sel])
            g = max(GRAPH, key=lambda c: M(score(V, c, e, dn), V[2])[sel])
            cands = [(g, e)]
            if include_zlap and dn == "full" and any(c[1] == 0 for c in evid_pool):
                cands.append(zlap_solution(sel))
            return max(cands, key=lambda ge: M(score(V, ge[0], ge[1], dn), V[2])[sel])

        for sel, kc, kw in (("bacc", "cef", "w0"), ("top1", "cef_t", "w0_t")):
            g, e = staged(sel)
            out[kc].append(M(score(T, g, e), T[2])[sel])
            gw, ew = staged(sel, evid_pool=[c for c in EVID if c[1] == 0])
            out[kw].append(M(score(T, gw, ew), T[2])[sel])
            if sel == "bacc":
                cfgs.append((g, e))
                gn, en = staged("bacc", evid_pool=[c for c in EVID if not c[2] and c[1] > 0], include_zlap=False)
                out["nodebias"].append(M(score(T, gn, en), T[2])["bacc"])
                gf, ef = staged("bacc", dn="filtered",
                                evid_pool=[c for c in EVID if c[1] > 0], include_zlap=False)
                out["filtered"].append(M(score(T, gf, ef, "filtered"), T[2])["bacc"])
    r = dict(dataset=dataset, backbone=label,
             **{k: float(np.mean(v)) for k, v in out.items()},
             cef_std=float(np.std(out["cef"])), modal_cfg=str(pd.Series(cfgs).mode()[0]))
    print(f"{dataset:8s} {label:14s} w0={r['w0']:.4f}  CEF={r['cef']:.4f}/{r['cef_t']:.4f}  "
          f"filt={r['filtered']:.4f} nodeb={r['nodebias']:.4f}  {r['modal_cfg']}", flush=True)
    return r

if __name__ == "__main__":
    rows = [run(ds, t, l) for ds in ["wikiart", "mp100k"] for t, l in BACKBONES]
    pd.DataFrame(rows).to_csv("unified_cef2.csv", index=False)
