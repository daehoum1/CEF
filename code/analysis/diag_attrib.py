"""Attribution: how much of CGPR's gain over kNN SP is the concept mask,
and how much is just re-tuning the self-loop weight alpha?"""
import numpy as np, pandas as pd
from cgpr_lab import *
from scipy.sparse import eye

METHOD = "kNN+Concept Hypergraph + Uncertainty"
SEEDS = [42, 43, 44, 45, 46]

def ugsp_on(W_raw, Y, alpha, steps, beta):
    """uncertainty-gated propagation on W_raw + alpha*I, row-normalised."""
    N = W_raw.shape[0]
    P = _row_normalize(W_raw + alpha * eye(N, format="csr"))
    return uchsp_propagate(P, Y, steps=steps, beta=beta)["scores"]

from concept_hypergraph import _row_normalize

rows = []
for dataset in ["wikiart", "mp100k"]:
    for tag, label in BACKBONES:
        p = Pool(dataset, tag)
        cfg = p.cgpr_cfg[METHOD]
        acc = {k: [] for k in ["vanilla", "knnSP_own", "knnSP_cgprAlpha", "CGPR", "CGPR_noFallback"]}
        for seed in SEEDS:
            _, ti = p.split(seed)
            embs, Y, lab = p.embs[ti], p.Y[ti], p.labels[ti]
            W = build_knn_graph(embs, k=p.sp_cfg["k_graph"], metric="cosine")
            B = build_incidence_matrix(embs, p.concept_embs, top_r=cfg["top_r"],
                                       normalize=cfg["normalize_incidence"])
            st, be = cfg["propagation_steps"], cfg["beta"]

            acc["vanilla"].append(metrics(Y, lab, p.class_names)["bacc"])
            # kNN SP with ITS OWN tuned alpha/steps/beta (the paper's baseline)
            acc["knnSP_own"].append(metrics(
                ugsp_on(W, Y, p.sp_cfg["alpha"], p.sp_cfg["K"], p.sp_cfg["beta"]),
                lab, p.class_names)["bacc"])
            # kNN SP with CGPR's tuned alpha/steps/beta but NO concept mask
            acc["knnSP_cgprAlpha"].append(metrics(
                ugsp_on(W, Y, cfg["alpha"], st, be), lab, p.class_names)["bacc"])
            # full CGPR
            acc["CGPR"].append(metrics(
                uchsp_propagate(build_combined_propagation_matrix(W, B, alpha=cfg["alpha"]),
                                Y, steps=st, beta=be)["scores"], lab, p.class_names)["bacc"])
            # CGPR without the starved-node fallback
            C = concept_comembership(B); C.data = np.ones_like(C.data)
            Wm = W.multiply(C).tocsr(); Wm.eliminate_zeros()
            acc["CGPR_noFallback"].append(metrics(
                ugsp_on(Wm, Y, cfg["alpha"], st, be), lab, p.class_names)["bacc"])
        r = {k: float(np.mean(v)) for k, v in acc.items()}
        r.update(dataset=dataset, backbone=label)
        r["gain_CGPR_over_vanilla"] = r["CGPR"] - r["vanilla"]
        r["gain_from_alpha_only"] = r["knnSP_cgprAlpha"] - r["knnSP_own"]
        r["gain_from_mask"] = r["CGPR"] - r["knnSP_cgprAlpha"]
        rows.append(r)

df = pd.DataFrame(rows)[["dataset","backbone","vanilla","knnSP_own","knnSP_cgprAlpha","CGPR",
                          "CGPR_noFallback","gain_CGPR_over_vanilla","gain_from_alpha_only","gain_from_mask"]]
pd.set_option("display.width", 250)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
df.to_csv("diag_attrib.csv", index=False)
