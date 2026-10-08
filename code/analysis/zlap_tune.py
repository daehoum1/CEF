"""Tune ZLaP per seed on the same validation partitions every other method
uses, so the baseline is given the same treatment as our method. Also reports
the paper's fixed defaults (k=5, gamma=5, alpha=0.3) for reference."""
import itertools, sys
import numpy as np, pandas as pd
from cgpr_lab import *
from zlap_impl import precompute_knn, zlap_transductive

import cef_paths

SEEDS = [42, 43, 44, 45, 46]
KS, GAMMAS, ALPHAS = [5, 10, 20, 40, 80], [1.0, 3.0, 5.0], [0.3, 0.5, 0.7, 0.9, 0.95]
CLF_ROOT = {"wikiart": cef_paths.require("CEF_WIKIART_CACHE", cef_paths.WIKIART_CACHE),
            "mp100k": cef_paths.require("CEF_MP100K_CACHE", cef_paths.MP100K_CACHE)}

def clf_for(dataset, tag):
    return np.load(f"{CLF_ROOT[dataset]}/{tag}/val_embeddings.npz")["text_embs"]

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    for dataset in ["wikiart", "mp100k"]:
        for tag, label in BACKBONES:
            if only and only not in (dataset, tag): continue
            p = Pool(dataset, tag); clf = clf_for(dataset, tag)
            acc = {m: [] for m in ["default_b", "default_t", "tuned_b", "tuned_t"]}
            cfgs = []
            for seed in SEEDS:
                vi, ti = p.split(seed)
                V = (p.embs[vi], p.labels[vi]); T = (p.embs[ti], p.labels[ti])
                M = lambda s, l: metrics(s, l, p.class_names)
                # one neighbour search per split serves the whole k grid
                kmax = max(KS)
                cV = precompute_knn(V[0], clf, kmax)
                cT = precompute_knn(T[0], clf, kmax)
                sd = zlap_transductive(T[0], clf, 5, 5.0, 0.3, knn_cache=cT)
                acc["default_b"].append(M(sd, T[1])["bacc"])
                acc["default_t"].append(M(sd, T[1])["top1"])
                # score every config on val ONCE, then select per metric
                grid = list(itertools.product(KS, GAMMAS, ALPHAS))
                val_m = [M(zlap_transductive(V[0], clf, *c, knn_cache=cV), V[1]) for c in grid]
                test_cache = {}
                for sel, key in (("bacc", "tuned_b"), ("top1", "tuned_t")):
                    best = grid[int(np.argmax([m[sel] for m in val_m]))]
                    if best not in test_cache:
                        test_cache[best] = M(zlap_transductive(T[0], clf, *best, knn_cache=cT), T[1])
                    acc[key].append(test_cache[best][sel])
                    if sel == "bacc": cfgs.append(best)
            r = dict(dataset=dataset, backbone=label,
                     **{k: float(np.mean(v)) for k, v in acc.items()},
                     tuned_b_std=float(np.std(acc["tuned_b"])),
                     modal_cfg=str(pd.Series(cfgs).mode()[0]))
            rows.append(r)
            print(f"{dataset:8s} {label:14s} default(bAcc/Top1)={r['default_b']:.4f}/{r['default_t']:.4f}  "
                  f"tuned={r['tuned_b']:.4f}/{r['tuned_t']:.4f}  cfg={r['modal_cfg']}", flush=True)
    pd.DataFrame(rows).to_csv("zlap_results.csv", index=False)
