"""Faithful scipy port of transductive ZLaP (Stojnic et al., CVPR 2024).

Ported from the authors' reference implementation
(github.com/vladan-stojnic/ZLaP, zlap.py::do_transductive_lp and
utils.py::knn2laplacian / normalize_connection_graph), replacing
faiss/cupy with scipy so it runs on the same cached embeddings as every
other method here. Node order matches the reference: the C class-text
nodes come first, then the M image nodes.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import cg


def _topk_ip(Q: np.ndarray, X: np.ndarray, k: int, chunk: int = 4096):
    """Top-k inner-product search of X for each row of Q (faiss.knn_gpu with
    METRIC_INNER_PRODUCT). Returns (idx [Q,k], sim [Q,k]) sorted descending.
    As in the reference, a query that is also in X keeps itself as a
    neighbour; the diagonal is removed later during normalisation."""
    n = Q.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = Q[s:e] @ X.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        vals = S[rows, part]
        order = np.argsort(-vals, axis=1)
        idx[s:e] = part[rows, order]
        sim[s:e] = vals[rows, order]
    return idx, sim


def zlap_transductive(features: np.ndarray, clf: np.ndarray,
                      k: int = 5, gamma: float = 5.0, alpha: float = 0.3,
                      cg_tol: float = 1e-6, cg_maxiter: int = 50,
                      cross_affinity: np.ndarray | None = None,
                      knn_cache: tuple | None = None) -> np.ndarray:
    """Transductive ZLaP. Returns [M, C] class scores for the image nodes.

    Args:
        features: L2-normalised image embeddings [M, D] (the unlabelled pool).
        clf:      L2-normalised class-text embeddings [C, D] (prompt ensemble).
        k, gamma, alpha: ZLaP hyperparameters (paper defaults 5, 5.0, 0.3).
        cross_affinity: optional [M, C] matrix replacing the cosine
            image-to-text similarity that weights the cross-modal edges.
            ZLaP itself leaves this None; passing a concept-evidence-fused
            score here is the CEF-LP variant.
    """
    C, M = clf.shape[0], features.shape[0]
    N = C + M

    if knn_cache is not None:
        ck_ii, cs_ii, ck_it, cs_it = knn_cache
        assert ck_ii.shape[1] >= min(k, M), "knn_cache built with too small k"
        knn_ii, sim_ii = ck_ii[:, :min(k, M)], cs_ii[:, :min(k, M)]
        knn_it, sim_it = ck_it[:, :min(k, C)], cs_it[:, :min(k, C)]
    else:
        knn_ii, sim_ii = _topk_ip(features, features, min(k, M))
        knn_it, sim_it = _topk_ip(features, clf, min(k, C))
    if cross_affinity is not None:
        k_it = min(k, C)
        part = np.argpartition(-cross_affinity, k_it - 1, axis=1)[:, :k_it]
        rws = np.arange(M)[:, None]
        vv = cross_affinity[rws, part]
        oo = np.argsort(-vv, axis=1)
        knn_it, sim_it = part[rws, oo], vv[rws, oo]
    knn = np.concatenate([knn_ii + C, knn_it], axis=1)
    sim = np.concatenate([sim_ii, sim_it], axis=1)

    # text rows are never queries: no outgoing edges (reference uses -1 / 0)
    knn = np.concatenate([-np.ones((C, knn.shape[1]), dtype=knn.dtype), knn], 0)
    sim = np.concatenate([np.zeros((C, sim.shape[1])), sim], 0)

    sim[sim < 0] = 0.0
    cross = knn < C                       # image->text edges (text ids are < C)
    sim[cross] = sim[cross] ** gamma      # h(v) = v^gamma, balances the modality gap

    rows = np.repeat(np.arange(N), knn.shape[1])
    cols, vals = knn.ravel(), sim.ravel()
    keep = cols != -1
    W = csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(N, N))
    W = W + W.T
    W = W - diags(W.diagonal(), 0)        # drop self-loops before normalising
    d = np.asarray(W.sum(axis=1)).ravel()
    d[d == 0] = 1.0
    Dmh = diags(1.0 / np.sqrt(d))
    L = (eye(N, format="csr") - alpha * (Dmh @ W @ Dmh)).tocsr()

    scores = np.zeros((M, C))
    for c in range(C):
        y = np.zeros(N); y[c] = 1.0
        try:
            out = cg(L, y, rtol=cg_tol, maxiter=cg_maxiter)[0]
        except TypeError:                 # scipy < 1.12 uses `tol`
            out = cg(L, y, tol=cg_tol, maxiter=cg_maxiter)[0]
        scores[:, c] = out[C:]
    return scores


def precompute_knn(features: np.ndarray, clf: np.ndarray, k_max: int):
    """One neighbour search reusable for every k <= k_max in a sweep.

    The neighbour lists depend only on k, and the k grid is nested, so a single
    descending-sorted search at k_max can be sliced for any smaller k. gamma
    and alpha act after the search and need no recomputation.
    """
    C, M = clf.shape[0], features.shape[0]
    knn_ii, sim_ii = _topk_ip(features, features, min(k_max, M))
    knn_it, sim_it = _topk_ip(features, clf, min(k_max, C))
    return knn_ii, sim_ii, knn_it, sim_it
