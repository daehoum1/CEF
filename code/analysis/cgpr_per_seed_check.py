"""Check how Table 3's CGPR row was configured, and what the described protocol gives.

The manuscript says CGPR's hyperparameters are reselected for each backbone and
seed on that seed's validation partition. `final_experiment.py`, which produced
the CGPR row, instead applies the single configuration stored in
`results/<ds>_v3/<tag>/best_config.json` to all five seeds. That file records
`val_bacc_mean`, i.e. it was chosen by averaging validation accuracy over the
five seeds' validation partitions -- and because the partitions are resplit per
seed, one seed's validation images are another seed's test images.

The original CGPR run also saved `best_config_per_seed.json`, where each seed's
configuration was selected on that seed's validation partition only. This script
applies both to exactly the same test partitions and graph as final_experiment.py,
so the only difference between the two columns is which configuration is used.

Output: results/supplementary/cgpr_per_seed_check.csv
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cgpr_lab import (BACKBONES, DATASETS, HYPER_ROOT, Pool, build_combined_propagation_matrix,
                      build_incidence_matrix, build_knn_graph, metrics, uchsp_propagate)

METHOD = "kNN+Concept Hypergraph + Uncertainty"
SEEDS = [42, 43, 44, 45, 46]
OUT = os.path.join(HYPER_ROOT, "results", "supplementary", "cgpr_per_seed_check.csv")


def cgpr_scores(p, idx, cfg, k):
    e, Y = p.embs[idx], p.Y[idx]
    W = build_knn_graph(e, k=k, metric="cosine")
    B = build_incidence_matrix(e, p.concept_embs, top_r=int(cfg["top_r"]),
                               normalize=bool(cfg["normalize_incidence"]))
    P = build_combined_propagation_matrix(W, B, alpha=float(cfg["alpha"]))
    return uchsp_propagate(P, Y, steps=int(cfg["propagation_steps"]),
                           beta=float(cfg["beta"]))["scores"]


def run(args):
    dataset, tag, label = args
    p = Pool(dataset, tag)
    k = p.sp_cfg["k_graph"]
    rdir = DATASETS[dataset]["results_dir"]
    fixed = json.load(open(os.path.join(rdir, tag, "best_config.json")))[METHOD]
    per_seed = json.load(open(os.path.join(rdir, tag, "best_config_per_seed.json")))
    rows = []
    for seed in SEEDS:
        _, ti = p.split(seed)
        l = p.labels[ti]
        for name, cfg in (("fixed (best_config.json)", fixed),
                          ("per-seed (best_config_per_seed.json)", per_seed[str(seed)][METHOD])):
            m = metrics(cgpr_scores(p, ti, cfg, k), l, p.class_names)
            rows.append(dict(dataset=dataset, backbone=label, seed=seed, config=name,
                             bacc=m["bacc"], top1=m["top1"],
                             cfg=json.dumps({kk: cfg[kk] for kk in
                                             ("top_r", "normalize_incidence", "alpha",
                                              "propagation_steps", "beta")})))
    print(f"  {dataset}/{label} done", flush=True)
    return rows


if __name__ == "__main__":
    jobs = [(ds, t, l) for ds in ["wikiart", "mp100k"] for t, l in BACKBONES]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = [r for rs in ex.map(run, jobs) for r in rs]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    s = df.groupby(["dataset", "backbone", "config"])[["bacc", "top1"]].mean().unstack("config")
    print(s.round(4).to_string())
    fx = df[df.config.str.startswith("fixed")].groupby(["dataset", "backbone"]).bacc.mean()
    ps = df[df.config.str.startswith("per-seed")].groupby(["dataset", "backbone"]).bacc.mean()
    print("\nper-seed minus fixed (bAcc pts):", ((ps - fx) * 100).round(2).to_dict())
    print(f"[saved] {OUT}")
