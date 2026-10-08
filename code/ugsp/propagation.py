"""
Score Propagation (SP) and Uncertainty-Guided Score Propagation (UGSP).

Propagation formulation
───────────────────────
Given raw kNN graph W_knn (no self-loops, unnormalized):

  1. Add self-loops:  W_sl  = W_knn + alpha * I
  2. Row-normalise:   W     = D^{-1} W_sl
  3. Propagate K steps:
        F_0 = Y   (vanilla CLIP probs)
        F_k = W @ F_{k-1}

  alpha controls how much each node retains its own score vs. aggregates
  from neighbours.  alpha → 0 : pure neighbour averaging per step.
                    alpha → ∞ : no propagation (stays at Y).

UGSP — layer-wise uncertainty-guided propagation
────────────────────────────────────────────────
  u is computed ONCE from Y (fixed across all K steps):

    H_i  = Shannon entropy of Y[i]
    u_i  = clip( (H_i / log C)^beta , 0, 1 )

  Each propagation step blends the current iterate with its neighbour
  aggregate, weighted per-node by u:

    F^(0) = Y
    F^(k+1) = (1 - u) ⊙ F^(k)  +  u ⊙ (W F^(k))

  where ⊙ broadcasts u [N] over the class dimension [C].

  beta < 1 : uncertainty saturates fast → more nodes get propagated.
  beta > 1 : only very uncertain nodes (near-uniform) rely on neighbours.
  beta = 1 : linear (original).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, eye

from .graph import row_normalize_graph


# ──────────────────────────────────────────────────────────────────────────────
# Entropy utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Shannon entropy H[i] = -Σ_c p_ic · log(p_ic).  Shape: [N]."""
    return -np.sum(probs * np.log(probs + eps), axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Propagation matrix (self-loops + row-normalise)
# ──────────────────────────────────────────────────────────────────────────────

def _make_prop_matrix(W_knn: csr_matrix, alpha: float) -> csr_matrix:
    """W_knn (no self-loops) → add alpha*I → row-normalise."""
    N = W_knn.shape[0]
    W_sl = W_knn + alpha * eye(N, format="csr")
    return row_normalize_graph(W_sl)


# ──────────────────────────────────────────────────────────────────────────────
# Score Propagation
# ──────────────────────────────────────────────────────────────────────────────

def score_propagation(
    W_knn: csr_matrix,
    Y: np.ndarray,
    K: int = 1,
    alpha: float = 0.5,
) -> np.ndarray:
    """K-step score propagation with self-loop weight alpha.

    Args:
        W_knn:  Raw kNN weight matrix WITHOUT self-loops [N, N].
        Y:      Vanilla CLIP softmax probs [N, C].
        K:      Number of propagation steps.
        alpha:  Self-loop weight (added before row-normalising).

    Returns:
        F_K [N, C] – score matrix after K propagation steps.
    """
    W = _make_prop_matrix(W_knn, alpha)
    F = Y.copy()
    for _ in range(K):
        F = W @ F
    return F


# ──────────────────────────────────────────────────────────────────────────────
# UGSP
# ──────────────────────────────────────────────────────────────────────────────

def ugsp(
    W_knn: csr_matrix,
    Y: np.ndarray,
    K: int = 1,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> dict[str, np.ndarray]:
    """Uncertainty-Guided Score Propagation.

    Args:
        W_knn:  Raw kNN weight matrix WITHOUT self-loops [N, N].
        Y:      Vanilla CLIP softmax probs [N, C].
        K:      Number of propagation steps.
        alpha:  Self-loop weight (propagation matrix construction).
        beta:   Entropy exponent for uncertainty: u = (H/logC)^beta.

    Returns a dict with keys:
        'scores_sp'           – plain SP result [N, C]
        'scores_ugsp'         – uncertainty-blended result [N, C]
        'entropy'             – raw Shannon entropy [N]
        'uncertainty'         – u = (H/logC)^beta ∈ [0,1] per node [N]
        'uncertainty_linear'  – linear uncertainty (H/logC) before beta [N]
    """
    F_sp = score_propagation(W_knn, Y, K=K, alpha=alpha)

    u_linear = compute_entropy(Y) / np.log(Y.shape[1])   # [N]
    u = np.clip(u_linear ** beta, 0.0, 1.0)              # [N]

    u_col = u[:, np.newaxis]                              # [N, 1] — broadcast over C

    W = _make_prop_matrix(W_knn, alpha)
    F_ugsp = Y.copy()
    for _ in range(K):
        F_ugsp = (1.0 - u_col) * F_ugsp + u_col * (W @ F_ugsp)

    return {
        "scores_sp": F_sp,
        "scores_ugsp": F_ugsp,
        "entropy": compute_entropy(Y),
        "uncertainty": u,
        "uncertainty_linear": u_linear,
    }
