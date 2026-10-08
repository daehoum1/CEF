"""Measured execution cost of each method under the paper's protocol.

For every (dataset, backbone) pair, on seed 42's image-level split, each method is
run in its own fresh process so that peak memory is attributable to it, with the
thread count pinned. Reported per method:

  val_evals        distinct validation configurations actually evaluated by the
                   bAcc selection (and, separately, by both metric selections)
  select_s         wall-clock time of hyperparameter selection on validation
  infer_s          wall-clock time to score the test partition once
  peak_rss_gb      peak resident memory of the process

The 6,000-configuration Cartesian product quoted in the manuscript is the size of
CEF's search space; `val_evals` is what the staged search actually evaluates.

One-time costs shared by every method, including the vanilla VLM, are measured
separately: image-embedding throughput on 1,024 images, and encoding the concept
dictionary with the text encoder.

Outputs: results/supplementary/cost/{cost_per_pair.csv, cost_summary.csv,
cost_one_time.csv, environment.json}
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import platform
import resource
import time

import cef_paths

THREADS = "4"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = THREADS

COST_ROOT = str(cef_paths.project_root() / "results" / "supplementary")
OUTDIR = COST_ROOT + "/cost"          # image-level; overridden by --split artist
SEED = 42
ARTIST_SPLITS = COST_ROOT + "/artist_disjoint/splits"


def _measure(dataset, tag, method, split="image"):
    """Runs in a spawned child: one method, one pair, one seed."""
    import numpy as np

    import supp_run as S
    from cgpr_lab import (Pool, build_combined_propagation_matrix, build_incidence_matrix,
                          build_knn_graph, metrics, uchsp_propagate)
    from zlap_impl import precompute_knn, zlap_transductive
    from zlap_tune import clf_for

    p = Pool(dataset, tag)
    clf = clf_for(dataset, tag)
    C = len(p.class_names)
    if split == "artist":
        sp = np.load(f"{ARTIST_SPLITS}/{dataset}_seed{SEED}.npz", allow_pickle=True)
        vi, ti = sp["val_idx"], sp["test_idx"]
    else:
        vi, ti = p.split(SEED)
    V = (p.embs[vi], p.Y[vi], p.labels[vi])
    T = (p.embs[ti], p.Y[ti], p.labels[ti])
    M = lambda s, l: metrics(s, l, p.class_names)
    out = dict(dataset=dataset, backbone=tag, method=method)

    if method == "Vanilla":
        t0 = time.perf_counter()
        M(T[1], T[2])
        out.update(val_evals=0, val_evals_both=0, select_s=0.0,
                   infer_s=time.perf_counter() - t0)

    elif method == "CGPR":
        k = p.sp_cfg["k_graph"]
        t0 = time.perf_counter()
        W = build_knn_graph(V[0], k=k, metric="cosine")
        n = 0
        best = (-1, None)
        for top_r, norm in itertools.product([50, 100, 200, 300, 500], [False, True]):
            B = build_incidence_matrix(V[0], p.concept_embs, top_r=min(top_r, len(vi)),
                                       normalize=norm)
            for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
                P = build_combined_propagation_matrix(W, B, alpha=a)
                for steps in [1, 2]:
                    for beta in [0.5, 1.0, 2.0, 4.0]:
                        s = uchsp_propagate(P, V[1], steps=steps, beta=beta)["scores"]
                        b = M(s, V[2])["bacc"]
                        n += 1
                        if b > best[0]:
                            best = (b, (top_r, norm, a, steps, beta))
        sel = time.perf_counter() - t0
        top_r, norm, a, steps, beta = best[1]
        t0 = time.perf_counter()
        Wt = build_knn_graph(T[0], k=k, metric="cosine")
        Bt = build_incidence_matrix(T[0], p.concept_embs, top_r=min(top_r, len(ti)), normalize=norm)
        M(uchsp_propagate(build_combined_propagation_matrix(Wt, Bt, alpha=a), T[1],
                          steps=steps, beta=beta)["scores"], T[2])
        out.update(val_evals=n, val_evals_both=n, select_s=sel,
                   infer_s=time.perf_counter() - t0)

    else:
        d = np.load(f"raw_concept_embs/{dataset}_{tag}.npz", allow_pickle=True)
        cemb = d["embs"]
        owners = np.array([p.class_names.index(s) for s in d["styles"]])
        cols = [np.where(owners == c)[0] for c in range(C)]
        t_setup = time.perf_counter()
        mu = (V[0] @ cemb.T).mean(0, keepdims=True)

        def fused(part, e):
            tau, w, db, base = e
            Bm = part[0] @ clf.T if base == "cos" else part[1]
            if w == 0:
                return Bm
            Sm = part[0] @ cemb.T
            if db:
                Sm = Sm - mu
            Sm = Sm / tau
            Sm -= Sm.max(1, keepdims=True)
            A = np.exp(Sm)
            A /= A.sum(1, keepdims=True)
            Yc = np.zeros((A.shape[0], C))
            for c, cc in enumerate(cols):
                Yc[:, c] = A[:, cc].sum(1)
            Yc /= Yc.sum(1, keepdims=True) + 1e-12
            if base == "cos":
                Yc = Yc * Bm.max(axis=1, keepdims=True)
            return (1 - w) * Bm + w * Yc

        if method == "CEF-FusionOnly":
            t0 = time.perf_counter()
            val = {e: M(fused(V, e), V[2]) for e in S.EVID}
            e = max(S.EVID, key=lambda c: val[c]["bacc"])
            sel = time.perf_counter() - t0
            t0 = time.perf_counter()
            M(fused(T, e), T[2])
            out.update(val_evals=len(S.EVID), val_evals_both=len(S.EVID), select_s=sel,
                       infer_s=time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            kv = precompute_knn(V[0], clf, max(S.KS))
            memo = {}

            def vm(g, e):
                if (g, e) not in memo:
                    memo[(g, e)] = M(zlap_transductive(V[0], clf, *g, cross_affinity=fused(V, e),
                                                       knn_cache=kv), V[2])
                return memo[(g, e)]

            if method == "ZLaP":
                for sel_m in ("bacc",):
                    g = max(S.GRAPH, key=lambda c: vm(c, S.ZL_E)[sel_m])
                e = S.ZL_E
                n_b = len(memo)
                sel = time.perf_counter() - t0
                n_both = len(S.GRAPH)
            else:  # CEF
                def staged(sm):
                    g = max(S.GRAPH, key=lambda c: vm(c, S.INIT_EVID)[sm])
                    e = max(S.EVID, key=lambda c: vm(g, c)[sm])
                    g = max(S.GRAPH, key=lambda c: vm(c, e)[sm])
                    gz = max(S.GRAPH, key=lambda c: vm(c, S.ZL_E)[sm])
                    return max([(g, e), (gz, S.ZL_E)], key=lambda ge: vm(*ge)[sm])
                g, e = staged("bacc")
                n_b = len(memo)
                sel = time.perf_counter() - t0
                staged("top1")
                n_both = len(memo)
            t0 = time.perf_counter()
            kt = precompute_knn(T[0], clf, max(S.KS))
            M(zlap_transductive(T[0], clf, *g, cross_affinity=fused(T, e), knn_cache=kt), T[2])
            out.update(val_evals=n_b, val_evals_both=n_both, select_s=sel,
                       infer_s=time.perf_counter() - t0)
        _ = t_setup

    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    out["n_val"], out["n_test"] = len(vi), len(ti)
    return out


def _one_time(tag_spec):
    import numpy as np
    import sys
    cef_paths.add_bundled_paths()
    import torch
    from cgpr_lab import get_class_names, pool_paths
    from vlm_comparison import VLMEncoder
    from concept_hypergraph import flatten_concepts, load_style_concepts
    from cgpr_lab import DATASETS

    tag, (mn, pt) = tag_spec
    enc = VLMEncoder(mn, pt, device=None)
    paths = pool_paths("wikiart", get_class_names("wikiart"))[:1024]
    enc.encode_images_from_paths(paths[:64], batch_size=64)  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    enc.encode_images_from_paths(paths, batch_size=256)
    torch.cuda.synchronize()
    ips = len(paths) / (time.perf_counter() - t0)
    rows = []
    for ds in ["wikiart", "mp100k"]:
        cls = get_class_names(ds)
        ph, _ = flatten_concepts(load_style_concepts(DATASETS[ds]["concepts_raw_json"]), cls)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        enc.encode_texts(ph)
        torch.cuda.synchronize()
        rows.append(dict(backbone=tag, dataset=ds, image_throughput_per_s=ips,
                         n_concepts=len(ph), concept_encoding_s=time.perf_counter() - t0))
    return rows


def main():
    import pandas as pd

    from cgpr_lab import BACKBONES

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-one-time", action="store_true")
    ap.add_argument("--split", choices=["image", "artist"], default="image")
    a = ap.parse_args()
    global OUTDIR
    if a.split == "artist":
        OUTDIR = COST_ROOT + "/cost_artist"
    os.makedirs(OUTDIR, exist_ok=True)
    ctx = mp.get_context("spawn")

    rows = []
    for ds in ["wikiart", "mp100k"]:
        for tag, label in BACKBONES:
            for method in ["Vanilla", "CGPR", "ZLaP", "CEF-FusionOnly", "CEF"]:
                with ctx.Pool(1) as pool:
                    r = pool.apply(_measure, (ds, tag, method, a.split))
                r["backbone"] = label
                rows.append(r)
                print(f"{ds:8s} {label:14s} {method:15s} evals={r['val_evals']:>5} "
                      f"select={r['select_s']:8.1f}s infer={r['infer_s']:6.2f}s "
                      f"rss={r['peak_rss_gb']:.2f}GB", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUTDIR, "cost_per_pair.csv"), index=False)
    summ = (per.groupby(["dataset", "method"])
               [["val_evals", "val_evals_both", "select_s", "infer_s", "peak_rss_gb"]]
               .agg(["mean", "min", "max"]))
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(os.path.join(OUTDIR, "cost_summary.csv"), index=False)

    if not a.skip_one_time:
        specs = [("ViT-B-32_openai", ("ViT-B-32", "openai")),
                 ("ViT-B-32_metaclip_fullcc", ("ViT-B-32", "metaclip_fullcc")),
                 ("EVA02-B-16_merged2b_s8b_b131k", ("EVA02-B-16", "merged2b_s8b_b131k")),
                 ("ViT-B-16-SigLIP_webli", ("ViT-B-16-SigLIP", "webli"))]
        ot = []
        for spec in specs:
            with ctx.Pool(1) as pool:
                ot.extend(pool.apply(_one_time, (spec,)))
        pd.DataFrame(ot).to_csv(os.path.join(OUTDIR, "cost_one_time.csv"), index=False)

    import numpy, scipy, torch
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                if l.startswith("model name")), platform.processor())
    env = dict(split="artist-disjoint" if a.split == "artist" else "image-level", cpu=cpu, logical_cpus=os.cpu_count(), threads_per_method=int(THREADS),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               python=platform.python_version(), numpy=numpy.__version__,
               scipy=scipy.__version__, torch=torch.__version__, seed=SEED,
               note="one method per fresh process; CPU post-processing only; "
                    "image/text encoding measured separately on GPU")
    json.dump(env, open(os.path.join(OUTDIR, "environment.json"), "w"), indent=1)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
