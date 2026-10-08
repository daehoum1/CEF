"""Fixed-configuration transfer on the artist-disjoint splits.

Same test as review_transfer.py, rebased: the configurations come from CEF's
balanced-accuracy selections on the artist-disjoint splits
(`results/supplementary/artist_disjoint/per_seed_results.csv`) and are scored on
the same saved artist-disjoint test partitions. Neither protocol consults the
evaluated pair's validation labels; its validation images enter only through the
debiasing offset.

  loo     modal CEF selection over the OTHER seven (dataset, backbone) pairs
  global  modal CEF selection over all pairs

Output: results/supplementary/artist_disjoint/per_seed_transfer.csv
"""
from __future__ import annotations

import ast
import collections
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cgpr_lab import BACKBONES, HYPER_ROOT, Pool, metrics  # noqa: E402
from zlap_impl import precompute_knn, zlap_transductive  # noqa: E402
from zlap_tune import clf_for  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ADIR = os.path.join(HYPER_ROOT, "results", "supplementary", "artist_disjoint")
OUT = os.path.join(ADIR, "per_seed_transfer.csv")


def modal(cfgs):
    return collections.Counter(cfgs).most_common(1)[0][0]


def evaluate(args):
    dataset, tag, label, cfgs = args
    p = Pool(dataset, tag)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    ce = d["embs"]
    owner = np.array([p.class_names.index(s) for s in d["styles"]])
    cols = [np.where(owner == c)[0] for c in range(C)]
    rows = []
    for seed in SEEDS:
        sp = np.load(os.path.join(ADIR, "splits", f"{dataset}_seed{seed}.npz"), allow_pickle=True)
        vi, ti = sp["val_idx"], sp["test_idx"]
        mu = (p.embs[vi] @ ce.T).mean(0, keepdims=True)
        Xt, Yt, Lt = p.embs[ti], p.Y[ti], p.labels[ti]
        knn = precompute_knn(Xt, clf, 80)
        for protocol, ((k, gam, al), (tau, w, db, base)) in cfgs.items():
            B = Xt @ clf.T if base == "cos" else Yt   # unclipped, as the tuned arm computes it
            if w == 0:
                Yf = B
            else:
                Sm = Xt @ ce.T
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
                Yf = (1 - w) * B + w * Yc
            m = metrics(zlap_transductive(Xt, clf, k, gam, al, cross_affinity=Yf, knn_cache=knn),
                        Lt, p.class_names)
            rows.append(dict(dataset=dataset, backbone=label, seed=seed,
                             method=f"CEF [{protocol}]", bacc=m["bacc"], top1=m["top1"],
                             cfg=str(((k, gam, al), (tau, w, db, base))), split="artist-disjoint"))
    print(f"  {dataset}/{label} done", flush=True)
    return rows


def main():
    src = pd.read_csv(os.path.join(ADIR, "per_seed_results.csv"))
    cef = src[(src.experiment == "CEF") & (src.metric == "bAcc")]
    assert len(cef) == 40, len(cef)
    pairs = list(dict.fromkeys(zip(cef.dataset, cef.backbone)))
    by_pair = {pr: list(cef[(cef.dataset == pr[0]) & (cef.backbone == pr[1])].selected_config)
               for pr in pairs}
    g_cfg = ast.literal_eval(modal([c for v in by_pair.values() for c in v]))
    tag_of = {label: tag for tag, label in BACKBONES}
    jobs = []
    for ds, label in pairs:
        others = [c for pr, v in by_pair.items() if pr != (ds, label) for c in v]
        jobs.append((ds, tag_of[label], label,
                     {"loo": ast.literal_eval(modal(others)), "global": g_cfg}))
        print(f"[cfg] {ds}/{label} loo={jobs[-1][3]['loo']}", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = [r for rs in ex.map(evaluate, jobs) for r in rs]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[saved] {OUT} ({len(rows)} rows); global cfg {g_cfg}")


if __name__ == "__main__":
    main()
