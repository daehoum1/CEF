"""Main comparison under ARTIST-DISJOINT validation/test splits.

The reported protocol (`Pool.split`) resplits the pooled images per seed at the
image level, which leaves 96-99% of test images by artists also present in
validation. This reruns the main comparison with `artist_split.artist_disjoint_split`
instead: the same pooled images, the same five seeds, the same 20% target per
style, the same cached embeddings and grids, and selection on validation only,
per metric -- the only change is that no artist appears on both sides.

Arms
  Vanilla   no hyperparameters
  CGPR      its reported 400-configuration grid for "kNN+Concept Hypergraph +
            Uncertainty" (top_r, normalize_incidence, combined alpha,
            propagation_steps, beta), searched on each seed's validation images
            with run_concept_hypergraph.concept_grid_search and scored once on test
  ZLaP      exhaustive 75-configuration graph grid
  CEF w=0   concept-free control, staged search, ZLaP solution among candidates
  CEF       staged search, ZLaP solution among candidates
  CEF-FusionOnly  argmax of the fused score, evidence grid only

Outputs (results/supplementary/artist_disjoint/):
  per_seed_results.csv   experiment, dataset, backbone, seed, metric, score, selected_config
  splits/{dataset}_seed{seed}.npz   val_idx, test_idx, val_artists, test_artists
  split_report.csv       sizes, per-style validation share, artist counts, shared artists
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import supp_run as S  # noqa: E402  (grids and constants only)
from artist_split import artist_disjoint_split, pool_artists, split_report  # noqa: E402
from cgpr_lab import (BACKBONES, HYPER_ROOT, Pool, build_combined_propagation_matrix,  # noqa: E402
                      build_incidence_matrix, build_knn_graph, metrics, uchsp_propagate)
from zlap_impl import precompute_knn, zlap_transductive  # noqa: E402
from zlap_tune import clf_for  # noqa: E402

OUTDIR = os.path.join(HYPER_ROOT, "results", "supplementary", "artist_disjoint")
OUT = os.path.join(OUTDIR, "per_seed_results.csv")
CGPR_METHOD = "kNN+Concept Hypergraph + Uncertainty"
CGPR_GRID = dict(top_r=[50, 100, 200, 300, 500], propagation_steps=[1, 2],
                 beta=[0.5, 1.0, 2.0, 4.0], normalize_incidence=[False, True])
CGPR_ALPHA = [0.1, 0.3, 0.5, 0.7, 0.9]


def splits_for(dataset, p, art):
    os.makedirs(os.path.join(OUTDIR, "splits"), exist_ok=True)
    out = {}
    for seed in S.SEEDS:
        f = os.path.join(OUTDIR, "splits", f"{dataset}_seed{seed}.npz")
        vi, ti = artist_disjoint_split(p.labels, art, seed)
        if not os.path.exists(f):
            np.savez(f, val_idx=vi, test_idx=ti,
                     val_artists=np.array(sorted(set(art[vi])), dtype=object),
                     test_artists=np.array(sorted(set(art[ti])), dtype=object))
        out[seed] = (vi, ti)
    return out


def run_pair(args):
    dataset, tag, label = args
    from run_concept_hypergraph import concept_grid_search  # same search CGPR used

    p = Pool(dataset, tag)
    art = pool_artists(dataset, p.class_names)
    assert len(art) == len(p.labels)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    cemb = d["embs"]
    owners = np.array([p.class_names.index(s) for s in d["styles"]])
    cols = [np.where(owners == c)[0] for c in range(C)]
    k_graph = p.sp_cfg["k_graph"]
    splits = splits_for(dataset, p, art)

    rows, reports = [], []
    for seed in S.SEEDS:
        vi, ti = splits[seed]
        rep = split_report(p.labels, art, vi, ti)
        assert rep["artists_shared"] == 0
        reports.append(dict(dataset=dataset, backbone=label, seed=seed, **rep))

        mu = (p.embs[vi] @ cemb.T).mean(0, keepdims=True)
        V = (p.embs[vi], p.Y[vi], p.labels[vi])
        T = (p.embs[ti], p.Y[ti], p.labels[ti])
        M = lambda s, l: metrics(s, l, p.class_names)
        knn = {id(V): precompute_knn(V[0], clf, max(S.KS)),
               id(T): precompute_knn(T[0], clf, max(S.KS))}
        cos_cache = {}

        def emit(exp, sel, score, cfg):
            rows.append(dict(experiment=exp, dataset=dataset, backbone=label, seed=seed,
                             metric="bAcc" if sel == "bacc" else "Top-1", score=score,
                             selected_config=str(cfg), split="artist-disjoint"))

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

        def test_metrics(g, e):
            return M(zlap_transductive(T[0], clf, *g, cross_affinity=fused(T, e),
                                       knn_cache=knn[id(T)]), T[2])

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

        # Vanilla
        mv = M(T[1], T[2])
        for sel in ("bacc", "top1"):
            emit("Vanilla", sel, mv[sel], "no hyperparameters")

        # ZLaP, CEF w=0, CEF
        for sel in ("bacc", "top1"):
            g, e = zlap_solution(sel)
            emit("ZLaP", sel, test_metrics(g, e)[sel], (g, e))
        for exp, pool in (("CEF w=0", [c for c in S.EVID if c[1] == 0]), ("CEF", S.EVID)):
            for sel in ("bacc", "top1"):
                g, e = staged(sel, pool)
                emit(exp, sel, test_metrics(g, e)[sel], (g, e))

        # Fusion only
        fo_val = {e: M(fused(V, e), V[2]) for e in S.EVID}
        for sel in ("bacc", "top1"):
            e = max(S.EVID, key=lambda c: fo_val[c][sel])
            emit("CEF-FusionOnly", sel, M(fused(T, e), T[2])[sel], e)

        # CGPR: its own grid, on this seed's validation images only
        W_val = build_knn_graph(V[0], k=k_graph, metric="cosine")
        with contextlib.redirect_stdout(io.StringIO()):
            grid = concept_grid_search(V[0], p.concept_embs, V[1], V[2], p.class_names,
                                       CGPR_GRID, W_knn=W_val, combined_alpha_grid=CGPR_ALPHA)
        grid = grid[grid["method"] == CGPR_METHOD]
        W_test = build_knn_graph(T[0], k=k_graph, metric="cosine")
        for sel in ("bacc", "top1"):
            r = grid.sort_values(sel, ascending=False).iloc[0]
            cfg = dict(top_r=int(r.top_r), normalize_incidence=bool(r.normalize_incidence),
                       alpha=float(r.alpha), propagation_steps=int(r.propagation_steps),
                       beta=float(r.beta))
            B = build_incidence_matrix(T[0], p.concept_embs,
                                       top_r=min(cfg["top_r"], T[0].shape[0]),
                                       normalize=cfg["normalize_incidence"])
            P = build_combined_propagation_matrix(W_test, B, alpha=cfg["alpha"])
            sc = uchsp_propagate(P, T[1], steps=cfg["propagation_steps"], beta=cfg["beta"])["scores"]
            emit("CGPR", sel, M(sc, T[2])[sel], cfg)

        print(f"  [{dataset}/{label}] seed {seed} done (val {rep['n_val']}, "
              f"val artists {rep['artists_val']}, shared {rep['artists_shared']})", flush=True)
    return rows, reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.backbone))
    jobs = [(ds, t, l) for ds in ["wikiart", "mp100k"] for t, l in BACKBONES if (ds, l) not in done]
    if not jobs:
        print("nothing to do")
        return
    # build and save the split files once in the parent so workers never race
    for ds in sorted({j[0] for j in jobs}):
        p = Pool(ds, BACKBONES[0][0])
        splits_for(ds, p, pool_artists(ds, p.class_names))
    print(f"[run] {len(jobs)} pairs on {a.workers} workers", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    rep_path = os.path.join(OUTDIR, "split_report.csv")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rows, reps in ex.map(run_pair, jobs):
            for path, new in ((OUT, rows), (rep_path, reps)):
                df = pd.DataFrame(new)
                if os.path.exists(path):
                    df = pd.concat([pd.read_csv(path), df], ignore_index=True)
                key = (["experiment", "dataset", "backbone", "seed", "metric"] if path == OUT
                       else ["dataset", "backbone", "seed"])
                df.drop_duplicates(subset=key, keep="last").to_csv(path, index=False)
            print(f"[saved] {rows[0]['dataset']}/{rows[0]['backbone']}", flush=True)


if __name__ == "__main__":
    main()
