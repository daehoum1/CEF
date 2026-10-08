"""Artist-disjoint rebase: every main-table arm not already produced by supp_artist.py.

`supp_artist.py` produced Vanilla, CGPR (re-selected per seed), ZLaP, CEF, CEF w=0
and CEF-FusionOnly on the artist-disjoint splits. This produces the remaining arms
the manuscript reports, on the same saved splits
(`results/supplementary/artist_disjoint/splits/`), seeds, cached embeddings and
grids, with selection on validation only. Each arm mirrors the script that
produced it under the image-level protocol; that script is named per arm.

  ZLaP (default cfg)         zlap_tune.py -- published defaults k=5, gamma=5, alpha=0.3
  CEF w/ filtered dictionary unified_cef2.py -- staged search, w>0, no ZLaP candidate
  CEF w/o debias             unified_cef2.py -- staged search, debias off, w>0, no ZLaP candidate
  CEF w/ CGPR propagation    final_experiment.py -- fused score fed to uncertainty-gated kNN
  DCLIP, CuPL (+ ZLaP)       review_controls.py
  CEF [rand_chars|rand_words|shuffle_owner]   review_controls.py
  CGPR mask on / mask off    diag_attrib.py -- at the CGPR configuration supp_artist.py
                             selected for that seed, toggling only the concept mask

Output: results/supplementary/artist_disjoint/per_seed_extra.csv
"""
from __future__ import annotations

import argparse
import ast
import itertools
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import eye  # noqa: E402

import supp_run as S  # noqa: E402  (grids and constants only)
from cgpr_lab import (BACKBONES, HYPER_ROOT, Pool, build_combined_propagation_matrix,  # noqa: E402
                      build_incidence_matrix, build_knn_graph, metrics, uchsp_propagate)
from concept_hypergraph import _row_normalize  # noqa: E402
from zlap_impl import precompute_knn, zlap_transductive  # noqa: E402
from zlap_tune import clf_for  # noqa: E402

ADIR = os.path.join(HYPER_ROOT, "results", "supplementary", "artist_disjoint")
OUT = os.path.join(ADIR, "per_seed_extra.csv")
# final_experiment.py's grids for the CGPR-propagation host
FE_ALPHAS, FE_KS, FE_BETAS = [0.1, 0.3, 0.5, 0.7, 0.9], [1, 2], [0.5, 1.0, 2.0, 4.0]
FE_TAUS = [0.0025, 0.005, 0.01, 0.02, 0.05]
FE_WS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def prop_unc(W, Y, a, K, b):
    return uchsp_propagate(_row_normalize(W + a * eye(W.shape[0], format="csr")), Y,
                           steps=K, beta=b)["scores"]


def run_pair(args):
    dataset, tag, label = args
    p = Pool(dataset, tag)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    k_graph = p.sp_cfg["k_graph"]
    idx = lambda styles: np.array([p.class_names.index(s) for s in styles])

    DICTS = {}
    d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    DICTS["full"] = (d["embs"], idx(d["styles"]))
    DICTS["filtered"] = (p.concept_embs, p.phrase_style_idx)
    for v in ("rand_chars", "rand_words", "dclip"):
        dv = np.load(f"control_dicts/{dataset}_{tag}_{v}.npz", allow_pickle=True)
        DICTS[v] = (dv["embs"], idx(dv["styles"]))
    embs_f, own_f = DICTS["full"]
    DICTS["shuffle_owner"] = (embs_f, own_f[np.random.default_rng(0).permutation(len(own_f))])
    COLS = {n: [np.where(o == c)[0] for c in range(C)] for n, (_, o) in DICTS.items()}

    prev = pd.read_csv(os.path.join(ADIR, "per_seed_results.csv"))
    cgpr_cfg = {int(r.seed): ast.literal_eval(r.selected_config)
                for r in prev[(prev.dataset == dataset) & (prev.backbone == label)
                              & (prev.experiment == "CGPR") & (prev.metric == "bAcc")].itertuples()}

    rows = []
    for seed in S.SEEDS:
        sp = np.load(os.path.join(ADIR, "splits", f"{dataset}_seed{seed}.npz"), allow_pickle=True)
        vi, ti = sp["val_idx"], sp["test_idx"]
        mu = {n: (p.embs[vi] @ DICTS[n][0].T).mean(0, keepdims=True) for n in DICTS}
        V = (p.embs[vi], p.Y[vi], p.labels[vi])
        T = (p.embs[ti], p.Y[ti], p.labels[ti])
        M = lambda s, l: metrics(s, l, p.class_names)
        knn = {id(V): precompute_knn(V[0], clf, max(S.KS)),
               id(T): precompute_knn(T[0], clf, max(S.KS))}

        def emit(exp, m, cfg=""):
            rows.append(dict(experiment=exp, dataset=dataset, backbone=label, seed=seed,
                             bacc=m["bacc"], top1=m["top1"], selected_config=str(cfg),
                             split="artist-disjoint"))

        def evidence(embs, tau, db, dn):
            Sm = embs @ DICTS[dn][0].T
            if db:
                Sm = Sm - mu[dn]
            Sm = Sm / tau
            Sm -= Sm.max(1, keepdims=True)
            A = np.exp(Sm)
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

        def score(part, g, e, dn="full"):
            k, gam, al = g
            tau, w, db, base = e
            B = cosine_aff(part) if base == "cos" else part[1]
            if w == 0:
                Yf = B
            else:
                Yc = evidence(part[0], tau, db, dn)
                if base == "cos":
                    Yc = Yc * B.max(axis=1, keepdims=True)
                Yf = (1 - w) * B + w * Yc
            return zlap_transductive(part[0], clf, k, gam, al, cross_affinity=Yf,
                                     knn_cache=knn[id(part)])

        memo = {}

        def vm(g, e, dn):
            key = (g, e, dn)
            if key not in memo:
                memo[key] = M(score(V, g, e, dn), V[2])
            return memo[key]

        zl = {}

        def zlap_solution():
            if "b" not in zl:
                zl["b"] = (max(S.GRAPH, key=lambda c: vm(c, S.ZL_E, "full")["bacc"]), S.ZL_E)
            return zl["b"]

        def staged(dn, pool, include_zlap):
            g = max(S.GRAPH, key=lambda c: vm(c, S.INIT_EVID, dn)["bacc"])
            e = max(pool, key=lambda c: vm(g, c, dn)["bacc"])
            g = max(S.GRAPH, key=lambda c: vm(c, e, dn)["bacc"])
            cands = [(g, e)]
            if include_zlap and any(c[1] == 0 for c in pool):
                cands.append(zlap_solution())
            return max(cands, key=lambda ge: vm(ge[0], ge[1], dn)["bacc"])

        # ZLaP at its published defaults
        emit("ZLaP (default cfg)", M(zlap_transductive(T[0], clf, 5, 5.0, 0.3,
                                                       knn_cache=knn[id(T)]), T[2]), (5, 5.0, 0.3))

        # ablations exactly as unified_cef2.py selects them
        g, e = staged("filtered", [c for c in S.EVID if c[1] > 0], include_zlap=False)
        emit("CEF w/ filtered dict", M(score(T, g, e, "filtered"), T[2]), (g, e))
        g, e = staged("full", [c for c in S.EVID if not c[2] and c[1] > 0], include_zlap=False)
        emit("CEF w/o debias", M(score(T, g, e, "full"), T[2]), (g, e))

        # dictionary controls exactly as review_controls.py selects them
        for dn in ("rand_chars", "rand_words", "shuffle_owner"):
            g, e = staged(dn, S.EVID, include_zlap=True)
            emit(f"CEF [{dn}]", M(score(T, g, e, dn), T[2]), (g, e))

        # classification by description, standalone and on the LP host
        demb, downer = DICTS["dclip"]
        dcols = [np.where(downer == c)[0] for c in range(C)]
        proto = np.stack([demb[downer == c].mean(0) / (np.linalg.norm(demb[downer == c].mean(0)) + 1e-12)
                          for c in range(C)])
        for name, sv, st in (
                ("DCLIP", np.stack([(V[0] @ demb.T)[:, cc].mean(1) for cc in dcols], 1),
                 np.stack([(T[0] @ demb.T)[:, cc].mean(1) for cc in dcols], 1)),
                ("CuPL", V[0] @ proto.T, T[0] @ proto.T)):
            emit(name, M(st, T[2]))
            av, at = np.clip(sv, 0, None), np.clip(st, 0, None)
            gd = max(S.GRAPH, key=lambda c: M(zlap_transductive(
                V[0], clf, *c, cross_affinity=av, knn_cache=knn[id(V)]), V[2])["bacc"])
            emit(f"{name} + ZLaP", M(zlap_transductive(T[0], clf, *gd, cross_affinity=at,
                                                       knn_cache=knn[id(T)]), T[2]), gd)

        # CEF evidence on CGPR's propagation host, as final_experiment.py builds it
        Wv = build_knn_graph(V[0], k=k_graph, metric="cosine")
        Wt = build_knn_graph(T[0], k=k_graph, metric="cosine")
        ug = max(itertools.product(FE_ALPHAS, FE_KS, FE_BETAS),
                 key=lambda c: M(prop_unc(Wv, V[1], *c), V[2])["bacc"])

        def fe_fused(part, tau, w, db):
            return (1 - w) * part[1] + w * evidence(part[0], tau, db, "full")

        grid = list(itertools.product(FE_TAUS, FE_WS, [False, True]))
        best = max(grid, key=lambda c: M(prop_unc(Wv, fe_fused(V, *c), *ug), V[2])["bacc"])
        emit("CEF w/ CGPR propagation", M(prop_unc(Wt, fe_fused(T, *best), *ug), T[2]),
             dict(ug=ug, evidence=best))

        # concept mask toggled at this seed's artist-selected CGPR configuration
        cfg = cgpr_cfg[seed]
        B = build_incidence_matrix(T[0], p.concept_embs, top_r=min(cfg["top_r"], len(ti)),
                                   normalize=cfg["normalize_incidence"])
        emit("CGPR (mask on)", M(uchsp_propagate(
            build_combined_propagation_matrix(Wt, B, alpha=cfg["alpha"]), T[1],
            steps=cfg["propagation_steps"], beta=cfg["beta"])["scores"], T[2]), cfg)
        emit("CGPR (mask off)", M(prop_unc(Wt, T[1], cfg["alpha"], cfg["propagation_steps"],
                                           cfg["beta"]), T[2]), cfg)

        print(f"  [{dataset}/{label}] seed {seed} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.backbone))
    jobs = [(ds, t, l) for ds in ["wikiart", "mp100k"] for t, l in BACKBONES if (ds, l) not in done]
    if not jobs:
        print("nothing to do")
        return
    print(f"[run] {len(jobs)} pairs on {a.workers} workers", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rows in ex.map(run_pair, jobs):
            df = pd.DataFrame(rows)
            if os.path.exists(OUT):
                df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
            df.drop_duplicates(subset=["experiment", "dataset", "backbone", "seed"],
                               keep="last").to_csv(OUT, index=False)
            print(f"[saved] {rows[0]['dataset']}/{rows[0]['backbone']}", flush=True)


if __name__ == "__main__":
    main()
