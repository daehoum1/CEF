"""Per-seed reference arms for the supplementary tables.

`unified_cef2.csv` reports CEF and the concept-free `w = 0` control for both
metrics, but only as means over the five seeds, and the per-seed file written by
review_controls.py selects on balanced accuracy only. The supplementary tables
need both metrics per seed, selected per metric, so this recomputes exactly those
two arms through the same code path supp_run.py uses.

It does not touch unified_cef2.csv or any existing result file; the values it
produces are checked against unified_cef2.csv by supp_report.py.

Output: results/supplementary/per_seed_reference.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import supp_run as S
from cgpr_lab import BACKBONES, Pool, metrics
from zlap_impl import precompute_knn, zlap_transductive
from zlap_tune import clf_for

OUT = os.path.join(S.OUTDIR, "per_seed_reference.csv")


def run_pair(args):
    dataset, tag, label = args
    p = Pool(dataset, tag)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    cemb = d["embs"]
    owners = np.array([p.class_names.index(s) for s in d["styles"]])
    cols = [np.where(owners == c)[0] for c in range(C)]

    rows = []
    for seed in S.SEEDS:
        vi, ti = p.split(seed)
        mu = (p.embs[vi] @ cemb.T).mean(0, keepdims=True)
        V = (p.embs[vi], p.Y[vi], p.labels[vi])
        T = (p.embs[ti], p.Y[ti], p.labels[ti])
        M = lambda s, l: metrics(s, l, p.class_names)
        knn = {id(V): precompute_knn(V[0], clf, max(S.KS)),
               id(T): precompute_knn(T[0], clf, max(S.KS))}
        cos_cache = {}

        def fused(part, e):
            tau, w, db, base = e
            if id(part) not in cos_cache:
                cos_cache[id(part)] = part[0] @ clf.T
            B = cos_cache[id(part)] if base == "cos" else part[1]
            if w == 0:
                return B
            Sm = part[0] @ cemb.T
            if db:
                Sm = Sm - mu
            Sm = Sm / tau
            Sm -= Sm.max(1, keepdims=True)
            A = np.exp(Sm)
            A /= A.sum(1, keepdims=True)
            Yc = np.zeros((A.shape[0], C))
            for c, cc in enumerate(cols):
                Yc[:, c] = A[:, cc].sum(1)
            Yc /= Yc.sum(1, keepdims=True) + 1e-12
            if base == "cos":
                Yc = Yc * B.max(axis=1, keepdims=True)
            return (1 - w) * B + w * Yc

        memo = {}

        def val_metrics(g, e):
            if (g, e) not in memo:
                memo[(g, e)] = M(zlap_transductive(V[0], clf, *g, cross_affinity=fused(V, e),
                                                   knn_cache=knn[id(V)]), V[2])
            return memo[(g, e)]

        zl = {}

        def zlap_solution(sel):
            if sel not in zl:
                zl[sel] = (max(S.GRAPH, key=lambda c: val_metrics(c, S.ZL_E)[sel]), S.ZL_E)
            return zl[sel]

        def staged(sel, pool):
            g = max(S.GRAPH, key=lambda c: val_metrics(c, S.INIT_EVID)[sel])
            e = max(pool, key=lambda c: val_metrics(g, c)[sel])
            g = max(S.GRAPH, key=lambda c: val_metrics(c, e)[sel])
            cands = [(g, e)]
            if any(c[1] == 0 for c in pool):
                cands.append(zlap_solution(sel))
            return max(cands, key=lambda ge: val_metrics(ge[0], ge[1])[sel])

        # ZLaP under per-metric selection comes free: zlap_solution() already
        # sweeps the graph grid for each metric.
        for sel in ("bacc", "top1"):
            g, e = zlap_solution(sel)
            sc = M(zlap_transductive(T[0], clf, *g, cross_affinity=fused(T, e),
                                     knn_cache=knn[id(T)]), T[2])[sel]
            rows.append(dict(experiment="ZLaP", dataset=dataset, backbone=label, seed=seed,
                             metric="bAcc" if sel == "bacc" else "Top-1",
                             score=sc, selected_config=str((g, e))))

        for exp, pool in (("CEF", S.EVID), ("CEF w=0", [c for c in S.EVID if c[1] == 0])):
            for sel in ("bacc", "top1"):
                g, e = staged(sel, pool)
                sc = M(zlap_transductive(T[0], clf, *g, cross_affinity=fused(T, e),
                                         knn_cache=knn[id(T)]), T[2])[sel]
                rows.append(dict(experiment=exp, dataset=dataset, backbone=label, seed=seed,
                                 metric="bAcc" if sel == "bacc" else "Top-1",
                                 score=sc, selected_config=str((g, e))))
        print(f"  [{dataset}/{label}] seed {seed} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(S.OUTDIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.backbone))
    jobs = [(ds, tag, label) for ds in ["wikiart", "mp100k"] for tag, label in BACKBONES
            if (ds, label) not in done]
    if not jobs:
        print("nothing to do")
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
