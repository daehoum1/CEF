"""
concept_hypergraph.py — Concept Hypergraph Score Propagation (CHSP)
=====================================================================

Training-free extension of UGSP's node-level kNN score propagation
(see src/ugsp/ugsp/propagation.py) to label-informed but ground-truth-free
hyperedges: each style's *visual concept phrases* (e.g. "visible brush
strokes" for Impressionism) are encoded with the VLM text encoder, and the
top-r images most similar to each concept form a hyperedge. No image labels
are used anywhere in hyperedge construction — only VLM text/image similarity.

Hypergraph propagation matrix
------------------------------
Given incidence matrix B [N, E]  (B[i, e] = similarity if image i is among
the top-r members of concept-hyperedge e, else 0) and hyperedge weights We
(diagonal, defaults to identity):

    Dv[i]  = Σ_e B[i, e] · We[e, e]      (node degree)
    De[e]  = Σ_i B[i, e]                 (hyperedge degree)
    P      = Dv^{-1} B We De^{-1} B^T    (image-to-image propagation matrix)

Score propagation (mirrors ugsp.propagation.score_propagation / ugsp)
-----------------------------------------------------------------------
  CHSP   (no uncertainty):   F_0 = Y ;  F_{k+1} = P F_k
  UCHSP  (uncertainty-guided): u computed once from Y's entropy,
                               F_0 = Y ;  F_{k+1} = (1-u) F_k + u (P F_k)

kNN + Concept Hypergraph (combined propagation matrix)
--------------------------------------------------------
Pure concept-hypergraph propagation pools images that are semantically
concept-relevant but possibly nowhere near each other visually (a "blunt,
global" pool). build_combined_propagation_matrix instead restricts a kNN
visual-similarity graph to only the edges (i, j) where i and j also
co-occur in at least one shared concept hyperedge — i.e. propagation only
happens between images that are BOTH visually close AND concept-relevant.
The result is fed to the same chsp_propagate / uchsp_propagate functions
above (they operate on any propagation matrix P).
"""

from __future__ import annotations

import json

import numpy as np
from scipy.sparse import csr_matrix, diags, eye

EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# Concept profile loading
# ──────────────────────────────────────────────────────────────────────────────

def load_style_concepts(json_path: str) -> dict[str, list[str]]:
    """Load {style_name: [concept phrase, ...]} from assets/style_concepts.json.
    Ignores any "_meta" key."""
    with open(json_path) as f:
        data = json.load(f)
    return {k: list(v) for k, v in data.items() if not k.startswith("_")}


def flatten_concepts(
    style_concepts: dict[str, list[str]],
    class_names: list[str],
) -> tuple[list[str], list[str]]:
    """Flatten to (phrases, owning_style), restricted to class_names, in class order.

    A style present in class_names but missing from style_concepts still gets
    at least one hyperedge, falling back to the style name itself as its sole
    concept phrase.
    """
    phrases: list[str] = []
    styles: list[str] = []
    missing = []
    for c in class_names:
        concepts = style_concepts.get(c)
        if not concepts:
            missing.append(c)
            concepts = [c]
        for phrase in concepts:
            phrases.append(phrase)
            styles.append(c)
    if missing:
        print(f"[concept_hypergraph] no concept profile for {missing}; "
              f"using the style name itself as a fallback concept")
    return phrases, styles


# ──────────────────────────────────────────────────────────────────────────────
# Hypergraph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_incidence_matrix(
    image_embs: np.ndarray,
    concept_text_embs: np.ndarray,
    top_r: int,
    normalize: bool = False,
) -> csr_matrix:
    """Build the [N, E] incidence matrix from image-concept cosine similarity.

    For each concept (hyperedge) e, the top_r images with highest cosine
    similarity to concept e's text embedding become its members, weighted by
    their (clamped non-negative) similarity score. Ground-truth labels are
    never used — only VLM image/text similarity.

    Args:
        image_embs:         L2-normalised image embeddings [N, D].
        concept_text_embs:  L2-normalised concept text embeddings [E, D].
        top_r:               Hyperedge size (clamped to N).
        normalize:            If True, weights within each hyperedge are
                              renormalised to sum to 1 (removes the effect of
                              absolute similarity magnitude, keeping only the
                              relative ranking within the hyperedge).

    Returns:
        B — sparse [N, E] incidence matrix.
    """
    N = image_embs.shape[0]
    E = concept_text_embs.shape[0]
    r = max(1, min(top_r, N))

    sim = image_embs @ concept_text_embs.T  # [N, E], cosine (unit-norm inputs)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for e in range(E):
        col = sim[:, e]
        top_idx = np.argpartition(-col, r - 1)[:r]
        w = np.clip(col[top_idx], 0.0, None)
        if normalize:
            denom = w.sum()
            w = w / denom if denom > EPS else w
        rows.extend(top_idx.tolist())
        cols.extend([e] * r)
        data.extend(w.tolist())

    return csr_matrix((data, (rows, cols)), shape=(N, E))


def build_propagation_matrix(
    B: csr_matrix,
    we: np.ndarray | None = None,
) -> csr_matrix:
    """P = Dv^{-1} B We De^{-1} B^T.

    Unlike a kNN graph (which always has a self-loop, so every node keeps a
    non-zero row), a node may belong to zero concept hyperedges — e.g. with a
    small top_r and many nodes. Such isolated nodes get an explicit identity
    row in P (P[i, i] = 1) so their score simply passes through unchanged
    instead of being propagated to an all-zero vector.

    Args:
        B:  Incidence matrix [N, E].
        we: Diagonal hyperedge weights [E]. Defaults to identity (all ones).
    """
    N, E = B.shape
    if we is None:
        we = np.ones(E)
    We = diags(we)

    BW = B @ We
    node_deg = np.array(BW.sum(axis=1)).flatten()          # Dv[i] = Σ_e B[i,e] we[e]
    isolated = node_deg <= EPS
    dv_inv = np.divide(1.0, node_deg, out=np.zeros_like(node_deg), where=~isolated)
    Dv_inv = diags(dv_inv)

    edge_deg = np.array(B.sum(axis=0)).flatten()            # De[e] = Σ_i B[i,e]
    de_inv = np.divide(1.0, edge_deg, out=np.zeros_like(edge_deg), where=edge_deg > EPS)
    De_inv = diags(de_inv)

    P = Dv_inv @ B @ We @ De_inv @ B.T
    if isolated.any():
        P = P + diags(isolated.astype(np.float64))         # identity row for isolated nodes
    return P.tocsr()


# ──────────────────────────────────────────────────────────────────────────────
# Entropy-based uncertainty (identical formulation to ugsp.propagation)
# ──────────────────────────────────────────────────────────────────────────────

def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Shannon entropy H[i] = -Σ_c p_ic · log(p_ic). Shape: [N].

    Clamped at 0: the eps inside the log makes log(p + eps) > 0 for a
    near-one-hot row, so the raw sum can come out at ~-1e-10 instead of 0.
    Entropy is non-negative by definition, and a negative value would become
    NaN in uchsp_propagate's u = (H / log C)**beta for fractional beta (e.g.
    beta=0.5, which is in the tuning grid), silently zeroing those nodes'
    scores. Vanilla VLM softmax scores are never peaked enough to trigger
    this, but sharply-peaked score vectors are."""
    return np.maximum(-np.sum(probs * np.log(probs + eps), axis=1), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Propagation
# ──────────────────────────────────────────────────────────────────────────────

def chsp_propagate(
    P: csr_matrix,
    Y: np.ndarray,
    steps: int = 1,
) -> np.ndarray:
    """Concept Hypergraph Score Propagation — no uncertainty gating.

    F_0 = Y ;  F_{k+1} = P F_k
    """
    F = Y.copy()
    for _ in range(steps):
        F = P @ F
    return F


def uchsp_propagate(
    P: csr_matrix,
    Y: np.ndarray,
    steps: int = 1,
    beta: float = 1.0,
) -> dict[str, np.ndarray]:
    """Uncertainty-Guided Concept Hypergraph Score Propagation.

    u is computed once from Y (fixed across all propagation steps):
        H_i = Shannon entropy of Y[i]
        u_i = clip((H_i / log C)^beta, 0, 1)

    F_0 = Y ;  F_{k+1} = (1 - u) ⊙ F_k + u ⊙ (P F_k)

    Returns a dict with keys 'scores', 'entropy', 'uncertainty'.
    """
    C = Y.shape[1]
    u_linear = compute_entropy(Y) / np.log(C)
    u = np.clip(u_linear ** beta, 0.0, 1.0)
    u_col = u[:, np.newaxis]

    F = Y.copy()
    for _ in range(steps):
        F = (1.0 - u_col) * F + u_col * (P @ F)

    return {"scores": F, "entropy": compute_entropy(Y), "uncertainty": u}


def concept_hypergraph_scores(
    image_embs: np.ndarray,
    concept_text_embs: np.ndarray,
    Y: np.ndarray,
    top_r: int,
    propagation_steps: int,
    use_uncertainty: bool,
    normalize_incidence: bool,
    beta: float = 1.0,
) -> tuple[np.ndarray, csr_matrix]:
    """Convenience one-shot pipeline: build B/P once, propagate once.

    Returns (scores [N, C], B) — B is returned for diagnostics/caching.
    """
    B = build_incidence_matrix(image_embs, concept_text_embs, top_r, normalize=normalize_incidence)
    P = build_propagation_matrix(B)
    if use_uncertainty:
        return uchsp_propagate(P, Y, steps=propagation_steps, beta=beta)["scores"], B
    return chsp_propagate(P, Y, steps=propagation_steps), B


# ──────────────────────────────────────────────────────────────────────────────
# kNN + Concept Hypergraph combination
# ──────────────────────────────────────────────────────────────────────────────

def _row_normalize(W: csr_matrix) -> csr_matrix:
    """D^{-1} W — each row sums to 1 (or 0 for isolated nodes). Local copy of
    ugsp.graph.row_normalize_graph so this module has no dependency on the
    ugsp package."""
    row_sums = np.array(W.sum(axis=1)).flatten()
    d_inv = np.where(row_sums > EPS, 1.0 / row_sums, 0.0)
    return diags(d_inv) @ W


def concept_comembership(B: csr_matrix, we: np.ndarray | None = None) -> csr_matrix:
    """C[i, j] = Σ_e B[i, e] · we[e] · B[j, e] — how strongly images i and j
    co-occur in shared concept hyperedges. Symmetric [N, N]."""
    E = B.shape[1]
    if we is None:
        we = np.ones(E)
    We = diags(we)
    return (B @ We @ B.T).tocsr()


def mask_knn_with_concepts(
    W_knn: csr_matrix,
    B: csr_matrix,
    we: np.ndarray | None = None,
) -> csr_matrix:
    """A pure concept hypergraph pools images that are semantically relevant
    to a concept but may be nowhere near each other visually — a blunt,
    global pool that showed a net-negative propagation signal even after
    discriminability-filtering the concepts. This instead keeps a kNN edge
    (i, j) only where i and j also co-occur in >=1 shared concept hyperedge:
    propagation only happens between images that are BOTH visually close
    (kNN) AND concept-relevant to each other (hypergraph co-membership).

    Masking a k=10 kNN graph this way typically collapses average node
    degree by 2-15x (denser candidate concepts survive more; sparse/rare
    ones can leave most nodes isolated) — so the self-loop weight that was
    tuned for the *unmasked* kNN graph is usually no longer the right choice
    for the result; retune alpha separately (see build_combined_propagation_matrix).

    Starved-node fallback: a node with zero concept-qualified kNN neighbors
    (common when concept-hyperedge coverage is thin relative to N, e.g.
    MultitaskPainting100k's 20 styles sharing only ~36 surviving concepts)
    would otherwise end up with an all-zero row here, which
    build_propagation_matrix / build_combined_propagation_matrix would then
    turn into a pure self-loop (i.e. no propagation at all for that node —
    strictly worse than plain kNN Score Propagation, which such a node would
    still fully benefit from). Instead, starved nodes fall back to their
    full, unmasked kNN row, so the combined method never does worse than
    plain kNN SP on the subset of the graph where concept coverage is too
    sparse to add anything.

    Returns the raw masked weight matrix, WITHOUT self-loop or normalisation
    (split out so callers can cheaply sweep alpha without rebuilding the mask).
    """
    comembership = concept_comembership(B, we=we)
    mask = comembership.copy()
    mask.data = np.ones_like(mask.data)              # binarise: "share >=1 hyperedge"
    W_combined = W_knn.multiply(mask).tocsr()
    W_combined.eliminate_zeros()

    degree = np.asarray(W_combined.sum(axis=1)).flatten()
    starved = degree <= EPS
    if starved.any():
        W_combined = W_combined + diags(starved.astype(np.float64)) @ W_knn
        W_combined = W_combined.tocsr()
    return W_combined


def build_combined_propagation_matrix(
    W_knn: csr_matrix,
    B: csr_matrix,
    we: np.ndarray | None = None,
    alpha: float = 0.5,
    W_combined: csr_matrix | None = None,
) -> csr_matrix:
    """Combine a local kNN visual-similarity graph with a concept hypergraph
    (see mask_knn_with_concepts), then add a self-loop (alpha·I) before
    row-normalising — exactly like ugsp.propagation.score_propagation — so
    nodes with no qualifying neighbor simply keep their own score instead of
    losing it.

    Args:
        W_knn:       Raw kNN weight matrix WITHOUT self-loops [N, N] (see
                     ugsp.graph.build_knn_graph).
        B:           Concept incidence matrix [N, E] (see build_incidence_matrix).
        we:          Diagonal hyperedge weights [E]. Defaults to identity.
        alpha:       Self-loop weight, added before row-normalising. Because
                     masking sharply increases sparsity, this is typically NOT
                     the same alpha that was optimal for the unmasked kNN graph.
        W_combined:  Precomputed mask_knn_with_concepts(W_knn, B, we) result,
                     to avoid recomputing the mask when sweeping alpha alone.
    """
    N = W_knn.shape[0]
    if W_combined is None:
        W_combined = mask_knn_with_concepts(W_knn, B, we=we)
    W_sl = W_combined + alpha * eye(N, format="csr")
    return _row_normalize(W_sl)
