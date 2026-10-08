"""
kNN graph construction from CLIP embeddings.

All operations use sparse matrices (scipy) to keep memory manageable
for N in the range of a few thousand to ~50k nodes.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
    self_loops: bool = False,
) -> csr_matrix:
    """Build a sparse kNN weight matrix from embeddings.

    Edge weight = cosine similarity (so w_ij ∈ [0, 1] for unit vectors).
    Returns a (N, N) sparse matrix W where W[i, j] > 0 iff j is among i's k-NN.
    The matrix is NOT symmetric and NOT row-normalised here — call
    row_normalize_graph() separately.
    """
    N = len(embeddings)
    nn = NearestNeighbors(
        n_neighbors=k + 1,  # +1 because the query point itself is always returned
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    rows, cols, data = [], [], []
    for i in range(N):
        for rank in range(len(indices[i])):
            j = indices[i, rank]
            if j == i and not self_loops:
                continue
            d = distances[i, rank]
            # cosine distance ∈ [0, 2] → similarity = 1 - d (clamped ≥ 0)
            w = max(1.0 - d, 0.0) if metric == "cosine" else np.exp(-d)
            rows.append(i)
            cols.append(j)
            data.append(w)

    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    return W


def row_normalize_graph(W: csr_matrix) -> csr_matrix:
    """D^{-1} W — each row sums to 1 (or 0 for isolated nodes)."""
    row_sums = np.array(W.sum(axis=1)).flatten()
    d_inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    D_inv = diags(d_inv)
    return D_inv @ W


def symmetric_normalize_graph(W: csr_matrix) -> csr_matrix:
    """D^{-1/2} W D^{-1/2} — preserves spectral properties."""
    row_sums = np.array(W.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(row_sums > 0, 1.0 / np.sqrt(row_sums), 0.0)
    D_inv_sqrt = diags(d_inv_sqrt)
    return D_inv_sqrt @ W @ D_inv_sqrt
