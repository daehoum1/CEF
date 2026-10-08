"""Descriptor-literature controls and baselines, per seed.

Answers three reviewer-facing questions that the original submission did not:

 1. Is the gain attributable to the CONTENT of the concepts, or would any
    dictionary of the same shape do? Three controls replace the real phrases
    while holding the per-class count, the phrase-length distribution and the
    entire search protocol fixed:
      rand_chars     random character strings
      rand_words     random sequences from the dictionary's own lexicon
      shuffle_owner  the real phrases, with their owning style permuted
 2. How does CEF compare with classification-by-description, which scores an
    image against LLM descriptors directly? DCLIP (mean cosine over a class's
    templated descriptors) and CuPL (cosine against the mean descriptor
    embedding) are evaluated standalone, and DCLIP is also given the same
    label-propagation host that CEF uses, tuned over the same graph grid.
 3. Are the differences larger than seed noise? Every number here is written
    per seed so paired tests can be run on the same partitions.

Output: review_controls_per_seed.csv (one row per dataset/backbone/seed/method).
Resumable: a (dataset, backbone) pair already present in the CSV is skipped.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

from cgpr_lab import BACKBONES, Pool, metrics
from zlap_impl import precompute_knn, zlap_transductive
from zlap_tune import clf_for

SEEDS = [42, 43, 44, 45, 46]
KS, GAMMAS, ALPHAS = [5, 10, 20, 40, 80], [1.0, 3.0, 5.0], [0.3, 0.5, 0.7, 0.9, 0.95]
TAUS, WS, DEB = [0.005, 0.01, 0.02, 0.05], [0.0, 0.2, 0.4, 0.6, 0.8], [True, False]
BASES = ["cos", "sm"]
GRAPH = list(itertools.product(KS, GAMMAS, ALPHAS))
EVID = list(itertools.product(TAUS, WS, DEB, BASES))
INIT_EVID = (0.01, 0.4, True, "sm")
ZL_E = (0.01, 0.0, True, "cos")
CTRL_VARIANTS = ["rand_chars", "rand_words", "shuffle_owner"]
OUT = "review_controls_per_seed.csv"
DICT_DIR = "control_dicts"


def load_dicts(dataset, tag, class_names):
    """Every dictionary the run needs, as (embeddings, owner-class indices)."""
    idx = lambda styles: np.array([class_names.index(s) for s in styles])
    out = {}
    d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    out["full"] = (d["embs"], idx(d["styles"]))
    for v in ["rand_chars", "rand_words", "dclip"]:
        d = np.load(f"{DICT_DIR}/{dataset}_{tag}_{v}.npz", allow_pickle=True)
        out[v] = (d["embs"], idx(d["styles"]))
    # shuffle_owner reuses the real embeddings and permutes the owner vector,
    # so per-class counts are identical by construction.
    embs, owners = out["full"]
    perm = np.random.default_rng(0).permutation(len(owners))
    out["shuffle_owner"] = (embs, owners[perm])
    return out


def run_pair(dataset, tag, label):
    p = Pool(dataset, tag)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    DICTS = load_dicts(dataset, tag, p.class_names)
    COLS = {n: [np.where(cs == c)[0] for c in range(C)] for n, (_, cs) in DICTS.items()}

    rows = []
    for seed in SEEDS:
        vi, ti = p.split(seed)
        mu = {n: (p.embs[vi] @ DICTS[n][0].T).mean(0, keepdims=True) for n in DICTS}
        V = (p.embs[vi], p.Y[vi], p.labels[vi])
        T = (p.embs[ti], p.Y[ti], p.labels[ti])
        M = lambda s, l: metrics(s, l, p.class_names)
        cache = {id(V): precompute_knn(V[0], clf, max(KS)),
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
            return zlap_transductive(part[0], clf, k, gam, al,
                                     cross_affinity=Yf, knn_cache=cache[id(part)])

        zl_cache = {}

        def zlap_solution(sel):
            if sel not in zl_cache:
                g = max(GRAPH, key=lambda c: M(score(V, c, ZL_E), V[2])[sel])
                zl_cache[sel] = (g, ZL_E)
            return zl_cache[sel]

        def staged(sel, dn="full", evid_pool=EVID, include_zlap=True):
            g = max(GRAPH, key=lambda c: M(score(V, c, INIT_EVID, dn), V[2])[sel])
            e = max(evid_pool, key=lambda c: M(score(V, g, c, dn), V[2])[sel])
            g = max(GRAPH, key=lambda c: M(score(V, c, e, dn), V[2])[sel])
            cands = [(g, e)]
            if include_zlap and any(c[1] == 0 for c in evid_pool):
                cands.append(zlap_solution(sel))
            return max(cands, key=lambda ge: M(score(V, ge[0], ge[1], dn), V[2])[sel])

        def emit(method, m, extra=""):
            rows.append(dict(dataset=dataset, backbone=label, seed=seed,
                             method=method, bacc=m["bacc"], top1=m["top1"],
                             cfg=extra))

        # --- reference arms, recomputed here so every number is paired ---------
        emit("Vanilla", M(T[1], T[2]))
        g_z, _ = zlap_solution("bacc")
        emit("ZLaP", M(score(T, g_z, ZL_E), T[2]), str(g_z))
        g, e = staged("bacc")
        emit("CEF", M(score(T, g, e), T[2]), str((g, e)))
        gw, ew = staged("bacc", evid_pool=[c for c in EVID if c[1] == 0])
        emit("CEF w=0", M(score(T, gw, ew), T[2]), str((gw, ew)))

        # --- control dictionaries, identical protocol -------------------------
        for dn in CTRL_VARIANTS:
            gc, ec = staged("bacc", dn=dn)
            emit(f"CEF [{dn}]", M(score(T, gc, ec, dn), T[2]), str((gc, ec)))

        # --- classification-by-description baselines --------------------------
        demb, downer = DICTS["dclip"]
        for name, fn in (("DCLIP", "mean_cos"), ("CuPL", "mean_emb")):
            def desc_scores(part):
                S = part[0] @ demb.T
                out = np.zeros((S.shape[0], C))
                for c in range(C):
                    cc = np.where(downer == c)[0]
                    out[:, c] = S[:, cc].mean(1) if len(cc) else -1e9
                return out
            if fn == "mean_cos":
                sc_v, sc_t = desc_scores(V), desc_scores(T)
            else:
                proto = np.stack([
                    demb[downer == c].mean(0) / (np.linalg.norm(demb[downer == c].mean(0)) + 1e-12)
                    for c in range(C)])
                sc_v, sc_t = V[0] @ proto.T, T[0] @ proto.T
            emit(name, M(sc_t, T[2]))
            # the same descriptor classifier given CEF's propagation host,
            # tuned over the identical graph grid
            aff_v = np.clip(sc_v, 0, None)
            aff_t = np.clip(sc_t, 0, None)
            gd = max(GRAPH, key=lambda c: M(zlap_transductive(
                V[0], clf, *c, cross_affinity=aff_v, knn_cache=cache[id(V)]), V[2])["bacc"])
            emit(f"{name} + ZLaP", M(zlap_transductive(
                T[0], clf, *gd, cross_affinity=aff_t, knn_cache=cache[id(T)]), T[2]), str(gd))

        print(f"  seed {seed} done", flush=True)

    return rows


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.backbone))
    all_rows = []
    for dataset in ["wikiart", "mp100k"]:
        for tag, label in BACKBONES:
            if only and only not in (dataset, tag):
                continue
            if (dataset, label) in done:
                print(f"[skip] {dataset} {label}", flush=True)
                continue
            print(f"[run] {dataset} {label}", flush=True)
            rows = run_pair(dataset, tag, label)
            all_rows.extend(rows)
            df = pd.DataFrame(all_rows)
            if os.path.exists(OUT):
                df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
            df.drop_duplicates(subset=["dataset", "backbone", "seed", "method"],
                               keep="last").to_csv(OUT, index=False)
            all_rows = []
            print(f"[saved] {dataset} {label}", flush=True)


if __name__ == "__main__":
    main()
