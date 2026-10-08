"""Artist-disjoint validation/test resplitting.

`Pool.split` (cgpr_lab.py) resplits the pooled val+test images per seed with a
class-stratified permutation at the IMAGE level. That is what the paper's
reported numbers use, and it does not keep artists apart: on every seed 96-99%
of test images are by an artist who also appears in validation.

This module provides the artist-grouped alternative used by the robustness
analysis. Every artist is assigned whole to validation or to test, so the two
partitions share no artist by construction, while each style's validation share
is kept as close as the grouping allows to the same 20% target. Nothing in the
existing pipeline calls it; `Pool.split` is unchanged.
"""
from __future__ import annotations

import numpy as np

from cgpr_lab import _pool_dataframe


def pool_artists(dataset: str, class_names: list[str]) -> np.ndarray:
    """Artist of every pooled image, in exactly Pool.embs / Pool.labels order."""
    df = _pool_dataframe(dataset)
    out = []
    for split in ("val", "test"):
        sdf = df[(df["new_subset"] == split) & (df["style"].isin(class_names))]
        out.append(sdf["artist"].astype(str).str.strip().str.lower().values)
    return np.concatenate(out)


def artist_disjoint_split(labels: np.ndarray, artists: np.ndarray, seed: int,
                          val_frac: float = 0.2):
    """Return (val_idx, test_idx) with no artist in both.

    Greedy stratified group assignment: artists are visited in a seed-dependent
    random order, and each is placed in validation only if that moves the
    per-style validation counts closer to the per-style targets (squared
    deviation relative to style size). A random order, rather than largest
    first, keeps validation a sample of many artists instead of a few prolific
    ones, and makes the five seeds genuinely different partitions. A final
    repair guarantees every style keeps at least one image on each side.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    C = int(classes.max()) + 1
    n_c = np.bincount(labels, minlength=C).astype(float)
    target = val_frac * n_c

    groups, inv = np.unique(artists, return_inverse=True)
    counts = np.zeros((len(groups), C))
    np.add.at(counts, (inv, labels), 1)

    order = rng.permutation(len(groups))

    def dev(v):
        return float((((v - target) / np.maximum(n_c, 1)) ** 2).sum())

    in_val = np.zeros(len(groups), bool)
    v = np.zeros(C)
    for g in order:
        if dev(v + counts[g]) < dev(v):
            in_val[g] = True
            v += counts[g]

    # repair: no style may end up empty on either side
    for c in classes:
        if v[c] == 0:
            cand = [g for g in np.argsort(counts[:, c]) if not in_val[g] and counts[g, c] > 0
                    and counts[~in_val & (np.arange(len(groups)) != g), c].sum() > 0]
            g = cand[0]
            in_val[g] = True
            v += counts[g]
        t_c = counts[~in_val, c].sum()
        if t_c == 0:
            cand = [g for g in np.argsort(counts[:, c]) if in_val[g] and counts[g, c] > 0
                    and counts[in_val & (np.arange(len(groups)) != g), c].sum() > 0]
            g = cand[0]
            in_val[g] = False
            v -= counts[g]

    val_mask = in_val[inv]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(~val_mask)[0]
    assert not (set(artists[val_idx]) & set(artists[test_idx])), "artist leak"
    for c in classes:
        assert (labels[val_idx] == c).any() and (labels[test_idx] == c).any(), f"style {c} empty"
    return val_idx, test_idx


def split_report(labels, artists, val_idx, test_idx):
    C = int(labels.max()) + 1
    n_c = np.bincount(labels, minlength=C)
    v_c = np.bincount(labels[val_idx], minlength=C)
    frac = v_c / np.maximum(n_c, 1)
    return dict(n_val=len(val_idx), n_test=len(test_idx),
                val_frac=len(val_idx) / len(labels),
                style_val_frac_min=float(frac.min()), style_val_frac_max=float(frac.max()),
                artists_val=len(set(artists[val_idx])), artists_test=len(set(artists[test_idx])),
                artists_shared=len(set(artists[val_idx]) & set(artists[test_idx])))
