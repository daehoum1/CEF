"""Supplementary experiment: CEF without label propagation.

CEF-FusionOnly runs full CEF up to the fused score Y_f, then predicts
argmax_c Y_f[i, c] with no graph and no propagation, under the paper's protocol
without modification: the same pooled val+test split, the same five seeds, the same
per-seed class-stratified 20:80 resplit, the same cached embeddings and the same
evidence grid, with hyperparameters selected on the validation partition only,
separately per metric. The graph grids and the staged-search constants below are
also imported by supp_artist.py and r3_run.py.

Output: results/supplementary/per_seed_results.csv (append-safe, resumable).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cgpr_lab import BACKBONES, HYPER_ROOT, Pool, metrics  # noqa: E402
from zlap_impl import precompute_knn, zlap_transductive  # noqa: E402
from zlap_tune import clf_for  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
KS, GAMMAS, ALPHAS = [5, 10, 20, 40, 80], [1.0, 3.0, 5.0], [0.3, 0.5, 0.7, 0.9, 0.95]
TAUS, WS, DEB = [0.005, 0.01, 0.02, 0.05], [0.0, 0.2, 0.4, 0.6, 0.8], [True, False]
BASES = ["cos", "sm"]
GRAPH = list(itertools.product(KS, GAMMAS, ALPHAS))
EVID = list(itertools.product(TAUS, WS, DEB, BASES))
INIT_EVID = (0.01, 0.4, True, "sm")
ZL_E = (0.01, 0.0, True, "cos")

OUTDIR = os.path.join(HYPER_ROOT, "results", "supplementary")
OUT = os.path.join(OUTDIR, "per_seed_results.csv")


def load_variants(dataset, tag, class_names):
    """(embeddings, owner-class indices) of the full dictionary."""
    full = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    owners = np.array([class_names.index(s) for s in full["styles"]])
    return {"full": (full["embs"], owners)}


def run_pair(args):
    dataset, tag, label = args
    p = Pool(dataset, tag)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    DICTS = load_variants(dataset, tag, p.class_names)
    COLS = {n: [np.where(o == c)[0] for c in range(C)] for n, (_, o) in DICTS.items()}
    for n, (_, o) in DICTS.items():
        assert min(len(x) for x in COLS[n]) > 0, f"{dataset}/{n}: a style lost all concepts"

    rows = []
    for seed in SEEDS:
        vi, ti = p.split(seed)
        mu = {n: (p.embs[vi] @ DICTS[n][0].T).mean(0, keepdims=True) for n in DICTS}
        V = (p.embs[vi], p.Y[vi], p.labels[vi])
        T = (p.embs[ti], p.Y[ti], p.labels[ti])
        M = lambda s, l: metrics(s, l, p.class_names)
        knn = {id(V): precompute_knn(V[0], clf, max(KS)),
               id(T): precompute_knn(T[0], clf, max(KS))}

        def evidence(embs, tau, db, dn):
            S = embs @ DICTS[dn][0].T
            if db:
                S = S - mu[dn]
            S = S / tau
            S -= S.max(1, keepdims=True)
            A = np.exp(S)
            A /= A.sum(1, keepdims=True)
            Yc = np.zeros((A.shape[0], C))
            for c, cc in enumerate(COLS[dn]):
                if len(cc):
                    Yc[:, c] = A[:, cc].sum(1)
            return Yc / (Yc.sum(1, keepdims=True) + 1e-12)

        cos_cache = {}

        def cosine_aff(part):
            if id(part) not in cos_cache:
                cos_cache[id(part)] = part[0] @ clf.T
            return cos_cache[id(part)]

        def fused(part, e, dn):
            """Y_f exactly as CEF builds it immediately before propagation."""
            tau, w, db, base = e
            B = cosine_aff(part) if base == "cos" else part[1]
            if w == 0:
                return B
            Yc = evidence(part[0], tau, db, dn)
            if base == "cos":
                Yc = Yc * B.max(axis=1, keepdims=True)
            return (1 - w) * B + w * Yc

        memo = {}

        def val_metrics(g, e, dn):
            """Validation metrics for one configuration, cached so the bAcc and
            Top-1 selections share work. Test data is never touched here."""
            key = (g, e, dn)
            if key not in memo:
                memo[key] = M(zlap_transductive(V[0], clf, *g, cross_affinity=fused(V, e, dn),
                                                knn_cache=knn[id(V)]), V[2])
            return memo[key]

        def test_metrics(g, e, dn):
            return M(zlap_transductive(T[0], clf, *g, cross_affinity=fused(T, e, dn),
                                       knn_cache=knn[id(T)]), T[2])

        zl = {}

        def zlap_solution(sel):
            if sel not in zl:
                zl[sel] = (max(GRAPH, key=lambda c: val_metrics(c, ZL_E, "full")[sel]), ZL_E)
            return zl[sel]

        def staged(sel, dn):
            g = max(GRAPH, key=lambda c: val_metrics(c, INIT_EVID, dn)[sel])
            e = max(EVID, key=lambda c: val_metrics(g, c, dn)[sel])
            g = max(GRAPH, key=lambda c: val_metrics(c, e, dn)[sel])
            cands = [(g, e), zlap_solution(sel)]
            return max(cands, key=lambda ge: val_metrics(ge[0], ge[1], dn)[sel])

        def emit(exp, sel, score, cfg):
            rows.append(dict(experiment=exp, dataset=dataset, backbone=label, seed=seed,
                             metric="bAcc" if sel == "bacc" else "Top-1",
                             score=score, selected_config=str(cfg)))

        # --- fusion only, no graph, no propagation --------------
        fo_val = {e: M(fused(V, e, "full"), V[2]) for e in EVID}
        for sel in ("bacc", "top1"):
            e = max(EVID, key=lambda c: fo_val[c][sel])
            emit("CEF-FusionOnly", sel, M(fused(T, e, "full"), T[2])[sel], e)

        print(f"  [{dataset}/{label}] seed {seed} done ({len(memo)} cached configs)",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.backbone))
    jobs = [(ds, tag, label) for ds in ["wikiart", "mp100k"] for tag, label in BACKBONES
            if (ds, label) not in done and (not a.only or a.only in (ds, tag))]
    if not jobs:
        print("nothing to do; all pairs present in", OUT)
        return
    print(f"[run] {len(jobs)} pairs on {a.workers} workers", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    out = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rows in ex.map(run_pair, jobs):
            out.extend(rows)
            df = pd.DataFrame(out)
            if os.path.exists(OUT):
                df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
            df.drop_duplicates(subset=["experiment", "dataset", "backbone", "seed", "metric"],
                               keep="last").to_csv(OUT, index=False)
            out = []
            print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
