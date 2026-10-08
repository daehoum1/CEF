"""Review round 3: independently regenerated dictionaries.

Everything runs on the saved ARTIST-DISJOINT splits with the grids, staged search,
ZLaP candidate and per-metric validation selection of supp_artist.py.

Arms
  Vanilla, ZLaP, CEF w=0, CEF [v3], CEF-FusionOnly   reproduction of the main table
  CEF [orig_g0..4]                                   five independent draws of the
                                                     dictionary with the original prompt

Output: results/supplementary/round3/per_seed_r3.csv
  experiment, dataset, backbone, seed, metric, score, selected_config,
  recall (per-class test recall, bAcc selection only), n_val, n_test
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import supp_run as S  # noqa: E402
from cgpr_lab import BACKBONES, HYPER_ROOT, Pool, metrics  # noqa: E402
from zlap_impl import precompute_knn, zlap_transductive  # noqa: E402
from zlap_tune import clf_for  # noqa: E402

ADIR = os.path.join(HYPER_ROOT, "results", "supplementary", "artist_disjoint")
R3 = os.path.join(HYPER_ROOT, "results", "supplementary", "round3")
OUT = os.path.join(R3, "per_seed_r3.csv")
GENS = range(5)


def per_class_recall(scores, labels, C):
    pred = scores.argmax(1)
    return [float((pred[labels == c] == c).mean()) if (labels == c).any() else np.nan
            for c in range(C)]


def load_dicts(dataset, tag, class_names):
    idx = lambda styles: np.array([class_names.index(s) for s in styles])
    D = {}
    d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    D["v3"] = (d["embs"], idx(d["styles"]))
    for g in GENS:
        e = np.load(f"r3_embs/{dataset}_orig_g{g}_{tag}.npz", allow_pickle=True)
        D[f"orig_g{g}"] = (e["embs"], idx(e["styles"]))
    return D


def run_pair(args):
    dataset, tag, label = args
    p = Pool(dataset, tag)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    D = load_dicts(dataset, tag, p.class_names)
    COLS = {n: [np.where(o == c)[0] for c in range(C)] for n, (_, o) in D.items()}
    for n in D:
        assert min(len(x) for x in COLS[n]) > 0, (dataset, n)

    rows = []
    for seed in S.SEEDS:
        sp = np.load(os.path.join(ADIR, "splits", f"{dataset}_seed{seed}.npz"), allow_pickle=True)
        vi, ti = sp["val_idx"], sp["test_idx"]
        V = (p.embs[vi], p.Y[vi], p.labels[vi])
        T = (p.embs[ti], p.Y[ti], p.labels[ti])
        M = lambda s, l: metrics(s, l, p.class_names)
        knn = {id(V): precompute_knn(V[0], clf, max(S.KS)),
               id(T): precompute_knn(T[0], clf, max(S.KS))}
        sims, mus = {}, {}

        def sim(part, key):
            """image x phrase cosine, cached per part."""
            k = (id(part), key)
            if k not in sims:
                sims[k] = part[0] @ D[key][0].T
            return sims[k]

        def mu(key):
            if key not in mus:
                mus[key] = sim(V, key).mean(0, keepdims=True)
            return mus[key]

        def evidence(part, tau, db, dn):
            Sm = sim(part, dn) - mu(dn) if db else sim(part, dn)
            Sm = Sm / tau
            Sm = Sm - Sm.max(1, keepdims=True)
            A = np.exp(Sm)
            A /= A.sum(1, keepdims=True)
            Yc = np.stack([A[:, cc].sum(1) for cc in COLS[dn]], 1)
            return Yc / (Yc.sum(1, keepdims=True) + 1e-12)

        cos_cache = {}

        def fused(part, e, dn):
            tau, w, db, base = e
            if id(part) not in cos_cache:
                cos_cache[id(part)] = part[0] @ clf.T
            B = cos_cache[id(part)] if base == "cos" else part[1]
            if w == 0:
                return B
            Yc = evidence(part, tau, db, dn)
            if base == "cos":
                Yc = Yc * B.max(axis=1, keepdims=True)
            return (1 - w) * B + w * Yc

        memo = {}

        def vm(g, e, dn):
            key = (g, e if e[1] > 0 else (0, 0, 0, e[3]), dn if e[1] > 0 else "")
            if key not in memo:
                memo[key] = M(zlap_transductive(V[0], clf, *g, cross_affinity=fused(V, e, dn),
                                                knn_cache=knn[id(V)]), V[2])
            return memo[key]

        zl = {}

        def zlap_solution(sel):
            if sel not in zl:
                zl[sel] = (max(S.GRAPH, key=lambda c: vm(c, S.ZL_E, "v3")[sel]), S.ZL_E)
            return zl[sel]

        def staged(sel, dn, pool_e):
            g = max(S.GRAPH, key=lambda c: vm(c, S.INIT_EVID, dn)[sel])
            e = max(pool_e, key=lambda c: vm(g, c, dn)[sel])
            g = max(S.GRAPH, key=lambda c: vm(c, e, dn)[sel])
            cands = [(g, e), zlap_solution(sel)]
            return max(cands, key=lambda ge: vm(ge[0], ge[1], dn)[sel])

        def emit(exp, sel, test_scores, cfg):
            rec = json.dumps(per_class_recall(test_scores, T[2], C)) if sel == "bacc" else ""
            rows.append(dict(experiment=exp, dataset=dataset, backbone=label, seed=seed,
                             metric="bAcc" if sel == "bacc" else "Top-1",
                             score=M(test_scores, T[2])[sel], selected_config=str(cfg),
                             recall=rec, n_val=len(vi), n_test=len(ti)))

        def lp(g, e, dn):
            return zlap_transductive(T[0], clf, *g, cross_affinity=fused(T, e, dn),
                                     knn_cache=knn[id(T)])

        w0 = [c for c in S.EVID if c[1] == 0]
        for sel in ("bacc", "top1"):
            emit("Vanilla", sel, T[1], "")
            g, e = zlap_solution(sel)
            emit("ZLaP", sel, lp(g, e, "v3"), (g, e))
            g, e = staged(sel, "v3", w0)
            emit("CEF w=0", sel, lp(g, e, "v3"), (g, e))
            for dn in ["v3"] + [f"orig_g{k}" for k in GENS]:
                g, e = staged(sel, dn, S.EVID)
                emit(f"CEF [{dn}]", sel, lp(g, e, dn), (g, e))
            e = max(S.EVID, key=lambda c: M(fused(V, c, "v3"), V[2])[sel])
            emit("CEF-FusionOnly", sel, fused(T, e, "v3"), e)
        print(f"  [{dataset}/{label}] seed {seed} done (val {len(vi)}, test {len(ti)}, "
              f"{len(memo)} val configs)", flush=True)
    return rows


def run_job(job):
    dataset, tag, label, seed = job
    S.SEEDS = [seed]
    return run_pair((dataset, tag, label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="dataset or backbone tag")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args()
    os.makedirs(R3, exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.backbone, prev.seed))
    seeds = a.seeds or S.SEEDS
    # one job per (dataset, backbone, seed); the largest pools first
    jobs = [(ds, t, l, s) for ds in ["mp100k", "wikiart"] for t, l in BACKBONES for s in seeds
            if (ds, l, s) not in done and (not a.only or a.only in (ds, t))]
    print(f"[run] {len(jobs)} (pair, seed) jobs on {a.workers} workers", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for fu in as_completed(futs):
            rows = fu.result()
            df = pd.DataFrame(rows)
            if os.path.exists(OUT):
                df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
            df.drop_duplicates(subset=["experiment", "dataset", "backbone", "seed", "metric"],
                               keep="last").to_csv(OUT, index=False)
            print(f"[saved] {rows[0]['dataset']}/{rows[0]['backbone']}/seed {rows[0]['seed']}", flush=True)


if __name__ == "__main__":
    main()
