"""Paired significance tests on the per-seed results.

Five seeds per (dataset, backbone) pair give a paired test too little power on
their own, so each comparison is reported at two levels:

  per pair    paired t-test over the 5 shared seeds (descriptive; n = 5)
  across      the 8 per-pair mean differences, tested with a two-sided Wilcoxon
              signed-rank test and an exact sign test (n = 8)

The across-pair level is the one quoted in the paper: the pairs, not the seeds,
are the units a claim about "all eight (dataset, backbone) pairs" is about.

Output: review_stats.csv and a printed summary.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

SRC = ["review_controls_per_seed.csv", "review_transfer_per_seed.csv"]
OUT = "review_stats.csv"

COMPARISONS = [
    ("CEF", "ZLaP", "CEF vs the tuned label-propagation baseline"),
    ("CEF", "CEF w=0", "CEF vs its own concept-free control (the honest reference)"),
    ("CEF", "CEF [rand_chars]", "real concepts vs random character strings"),
    ("CEF", "CEF [rand_words]", "real concepts vs random words from the same lexicon"),
    ("CEF", "CEF [shuffle_owner]", "real concepts vs the same phrases with permuted owners"),
    ("CEF", "DCLIP + ZLaP", "CEF vs classification-by-description under the same host"),
    ("CEF", "CuPL + ZLaP", "CEF vs the descriptor-prototype classifier under the same host"),
    ("CEF [loo]", "ZLaP", "CEF with a config never tuned on this pair, vs tuned ZLaP"),
    ("CEF [global]", "ZLaP", "one global CEF config vs per-pair tuned ZLaP"),
    ("CEF [rand_chars]", "CEF w=0", "do random strings add anything over no concepts?"),
    ("CEF [shuffle_owner]", "CEF w=0", "do misattributed real phrases add anything?"),
    ("ZLaP", "CGPR_ref", "baseline sanity check (skipped if CGPR is absent)"),
]


def sign_test(diffs):
    """Exact two-sided sign test on nonzero differences."""
    nz = diffs[diffs != 0]
    n, k = len(nz), int((nz > 0).sum())
    if n == 0:
        return 1.0, 0, 0
    p = min(1.0, 2 * stats.binom.sf(max(k, n - k) - 1, n, 0.5))
    return p, k, n


def main():
    df = pd.concat([pd.read_csv(f) for f in SRC if os.path.exists(f)], ignore_index=True)
    have = set(df.method)
    rows = []
    for a, b, note in COMPARISONS:
        if a not in have or b not in have:
            print(f"[skip] {a} vs {b} (missing)")
            continue
        pa = df[df.method == a].set_index(["dataset", "backbone", "seed"]).bacc
        pb = df[df.method == b].set_index(["dataset", "backbone", "seed"]).bacc
        idx = pa.index.intersection(pb.index)
        d = (pa[idx] - pb[idx]).rename("diff").reset_index()
        per_pair = d.groupby(["dataset", "backbone"])["diff"].mean()

        for (ds, bk), grp in d.groupby(["dataset", "backbone"]):
            t, pt = stats.ttest_rel(
                pa.loc[(ds, bk)].sort_index().values,
                pb.loc[(ds, bk)].sort_index().values)
            rows.append(dict(comparison=f"{a} - {b}", level="pair", dataset=ds,
                             backbone=bk, n=len(grp), mean_diff=grp["diff"].mean(),
                             p=pt, note=""))

        v = per_pair.values
        try:
            w_p = stats.wilcoxon(v).pvalue
        except ValueError:
            w_p = np.nan
        s_p, k, n = sign_test(v)
        rows.append(dict(comparison=f"{a} - {b}", level="across", dataset="-",
                         backbone="-", n=len(v), mean_diff=v.mean(), p=w_p,
                         note=f"wilcoxon; sign {k}/{n} p={s_p:.4g}; "
                              f"range [{v.min():+.4f},{v.max():+.4f}]"))
        print(f"{a:22s} - {b:22s}  mean {v.mean()*100:+.2f} pts  "
              f"range [{v.min()*100:+.2f},{v.max()*100:+.2f}]  "
              f"wilcoxon p={w_p:.4g}  sign {k}/{n} p={s_p:.4g}   ({note})")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
