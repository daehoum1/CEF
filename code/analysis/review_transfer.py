"""Fixed-configuration transfer: does CEF still win when it is never tuned on
the pair it is evaluated on?

The submission's own Limitations section observes that CEF searches a larger
space than the baseline and that "evaluation with a fixed configuration would
provide a stricter test of transfer". This runs that test.

Two protocols, neither of which looks at the evaluated pair's validation split:

  loo     leave-one-pair-out. The configuration is the modal CEF selection over
          the OTHER seven (dataset, backbone) pairs, applied unchanged.
  global  one single configuration -- the modal selection over all pairs --
          applied to every pair.

Both are compared against ZLaP tuned per seed on the evaluated pair, i.e. the
baseline keeps its full advantage while CEF gives up tuning entirely.

Reads the CEF selections recorded by review_controls.py.
Output: review_transfer_per_seed.csv
"""
from __future__ import annotations

import ast
import collections
import os

import numpy as np
import pandas as pd

from cgpr_lab import BACKBONES, Pool, metrics
from zlap_impl import precompute_knn, zlap_transductive
from zlap_tune import clf_for

SEEDS = [42, 43, 44, 45, 46]
KMAX = 80
SRC = "review_controls_per_seed.csv"
OUT = "review_transfer_per_seed.csv"


def modal_cfg(cfgs):
    return collections.Counter(cfgs).most_common(1)[0][0]


def evaluate(dataset, tag, label, cfg_by_protocol):
    p = Pool(dataset, tag)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
    embs_c = d["embs"]
    owner = np.array([p.class_names.index(s) for s in d["styles"]])
    cols = [np.where(owner == c)[0] for c in range(C)]

    rows = []
    for seed in SEEDS:
        vi, ti = p.split(seed)
        # the validation split is used ONLY for the debiasing statistic, which
        # Equation (5) defines as a validation-derived offset; no selection.
        mu = (p.embs[vi] @ embs_c.T).mean(0, keepdims=True)
        Xt, Yt, Lt = p.embs[ti], p.Y[ti], p.labels[ti]
        knn = precompute_knn(Xt, clf, KMAX)

        for protocol, ((k, gam, al), (tau, w, db, base)) in cfg_by_protocol.items():
            # identical to review_controls.py: the cosine base is passed
            # unclipped, exactly as the tuned arm computes it
            B = Xt @ clf.T if base == "cos" else Yt
            if w == 0:
                Yf = B
            else:
                S = Xt @ embs_c.T
                if db:
                    S = S - mu
                S = S / tau
                S -= S.max(1, keepdims=True)
                A = np.exp(S)
                A /= A.sum(1, keepdims=True)
                Yc = np.zeros((A.shape[0], C))
                for c, cc in enumerate(cols):
                    if len(cc):
                        Yc[:, c] = A[:, cc].sum(1)
                Yc /= Yc.sum(1, keepdims=True) + 1e-12
                if base == "cos":
                    Yc = Yc * B.max(axis=1, keepdims=True)
                Yf = (1 - w) * B + w * Yc
            m = metrics(zlap_transductive(Xt, clf, k, gam, al,
                                          cross_affinity=Yf, knn_cache=knn), Lt,
                        p.class_names)
            rows.append(dict(dataset=dataset, backbone=label, seed=seed,
                             method=f"CEF [{protocol}]", bacc=m["bacc"],
                             top1=m["top1"], cfg=str(((k, gam, al), (tau, w, db, base)))))
        print(f"  seed {seed} done", flush=True)
    return rows


def main():
    src = pd.read_csv(SRC)
    cef = src[src.method == "CEF"]
    assert len(cef) == 40, f"expected 40 CEF rows, found {len(cef)}"
    pairs = list(dict.fromkeys(zip(cef.dataset, cef.backbone)))
    by_pair = {pr: list(cef[(cef.dataset == pr[0]) & (cef.backbone == pr[1])].cfg)
               for pr in pairs}
    g_cfg = ast.literal_eval(modal_cfg([c for v in by_pair.values() for c in v]))

    tag_of = {label: tag for tag, label in BACKBONES}
    all_rows = []
    for ds, label in pairs:
        others = [c for pr, v in by_pair.items() if pr != (ds, label) for c in v]
        cfgs = {"loo": ast.literal_eval(modal_cfg(others)), "global": g_cfg}
        print(f"[run] {ds} {label}  loo={cfgs['loo']}", flush=True)
        all_rows.extend(evaluate(ds, tag_of[label], label, cfgs))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)
    print(f"[saved] {OUT}  {len(df)} rows")
    print(f"[global config] {g_cfg}")


if __name__ == "__main__":
    main()
