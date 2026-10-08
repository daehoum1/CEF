"""
run_concept_hypergraph.py — Concept Hypergraph Score Propagation (CHSP)
==========================================================================

Training-free zero-shot painting-style classification: extends UGSP's
node-level kNN score propagation with *concept hyperedges* built from VLM
text-encoded style descriptions (e.g. "visible brush strokes" for
Impressionism). No ground-truth labels are used to build hyperedges — only
VLM image/text similarity. Labels are only used for evaluation and, on the
val split, for hyperparameter selection.

Baselines compared (per VLM):
  1. Vanilla VLM                        — raw zero-shot softmax scores
  2. Score Propagation (kNN)            — ugsp.propagation.score_propagation,
                                           reusing the tuned config already
                                           cached by ugsp/vlm_comparison.py
                                           (skipped if not found)
  3. Concept Hypergraph (no uncertainty)     — hyper.concept_hypergraph.chsp_propagate
  4. Concept Hypergraph + Uncertainty        — hyper.concept_hypergraph.uchsp_propagate
  5. kNN+Concept Hypergraph (no uncertainty) — chsp_propagate on the combined
                                                kNN∩hyperedge-co-membership graph
                                                (see build_combined_propagation_matrix);
                                                skipped if no kNN SP config found
  6. kNN+Concept Hypergraph + Uncertainty    — uchsp_propagate on the same
                                                combined graph

Protocol (mirrors ugsp/vlm_comparison.py for embedding caching, but NOT for
splitting — see below):
  • For each VLM, image/text embeddings + Y are loaded from the existing
    UGSP cache if present (no re-encoding); otherwise encoded fresh via
    open_clip and cached under --save_dir.
  • Concept phrases are encoded once per VLM's text encoder and cached.
  • The dataset's own val+test images are pooled together (no fixed train
    split is used at all, consistent with a training-free method). For
    each of --seeds, the pool is freshly partitioned val:test = --eval_frac
    : (1 - --eval_frac) (stratified by class), hyperparameters are
    grid-searched on THAT seed's val partition, and evaluated on THAT
    seed's test partition. Tuning on a single fixed val set across all
    seeds let hyperparameter selection overfit as the grid grew (observed
    concretely: near-zero val delta, large test regression on some VLMs);
    re-partitioning per seed makes the reported mean ± std a genuine
    repeated-tune-and-evaluate estimate instead of one tune + 5 test resamples.

Usage:
  python scripts/run_concept_hypergraph.py \\
      --config configs/concept_hypergraph.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix

# ──────────────────────────────────────────────────────────────────────────────
# Wire up this project's own src/ and the sibling src/ugsp project for reuse
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPER_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_ROOT = os.path.dirname(HYPER_ROOT)
UGSP_ROOT = os.path.join(SRC_ROOT, "ugsp")

sys.path.insert(0, os.path.join(HYPER_ROOT, "src"))
sys.path.insert(0, UGSP_ROOT)

from concept_hypergraph import (  # noqa: E402
    build_combined_propagation_matrix,
    build_incidence_matrix,
    build_propagation_matrix,
    chsp_propagate,
    flatten_concepts,
    load_style_concepts,
    mask_knn_with_concepts,
    uchsp_propagate,
)
from ugsp.clip_encoder import STYLE_DISPLAY_NAMES  # noqa: E402
from ugsp.dataset import maybe_filter_styles, prepare_dataframe  # noqa: E402
from ugsp.evaluate import (  # noqa: E402
    accuracy,
    aggregate_multi_seed_results,
    balanced_accuracy,
    compare_methods,
    g_means,
    macro_f1,
    print_comparison_table,
    print_multi_seed_summary,
    save_multi_seed_results,
)
from ugsp.graph import build_knn_graph  # noqa: E402
from ugsp.propagation import score_propagation  # noqa: E402
from vlm_comparison import VLMEncoder, get_split  # noqa: E402

METHOD_NO_UNC = "Concept Hypergraph (no uncertainty)"
METHOD_UNC = "Concept Hypergraph + Uncertainty"
METHOD_VANILLA = "Vanilla VLM"
METHOD_KNN_SP = "Score Propagation (kNN)"
METHOD_COMBINED_NO_UNC = "kNN+Concept Hypergraph (no uncertainty)"
METHOD_COMBINED_UNC = "kNN+Concept Hypergraph + Uncertainty"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y")


def resolve_path(path: str | None) -> str | None:
    """Resolve a config/CLI path against this project's root rather than the
    current working directory, so the project stays runnable from anywhere and
    can be copied/moved without editing configs/concept_hypergraph.yaml.
    Absolute paths are returned unchanged."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(HYPER_ROOT, path)


def stratified_two_way_split(
    labels: np.ndarray, seed: int, val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition ALL indices into disjoint (val_idx, test_idx), stratified by
    class (>=1 sample per class on the val side). Unlike
    vlm_comparison.stratified_random_split (which draws a val_frac SUBSET,
    leaving the rest unused), this assigns every index to exactly one side —
    used to re-partition the full val+test pool independently per seed."""
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    test_idx: list[int] = []
    for c in np.unique(labels):
        c_idx = rng.permutation(np.where(labels == c)[0])
        n_val = max(1, int(round(len(c_idx) * val_frac)))
        val_idx.extend(c_idx[:n_val].tolist())
        test_idx.extend(c_idx[n_val:].tolist())
    return np.array(val_idx), np.array(test_idx)


def load_or_encode_embeddings(
    model_name: str,
    pretrained: str,
    paths: list[str],
    class_names: list[str],
    cache_path: str,
    batch_size: int,
    device: str | None,
    display_names: dict[str, str] | None,
) -> tuple[np.ndarray, np.ndarray, VLMEncoder | None]:
    """Load (image_embs, Y) from cache_path if present; else build a VLMEncoder,
    encode, and save cache. Returns the encoder too (None on cache hit) so
    callers can reuse it for concept-phrase encoding without a second model
    load."""
    if os.path.exists(cache_path):
        print(f"  [cache] Loading {cache_path}")
        d = np.load(cache_path)
        return d["image_embs"], d["Y"], None

    encoder = VLMEncoder(model_name, pretrained, device=device)
    print(f"  [embed] Encoding {len(paths)} images …")
    image_embs = encoder.encode_images_from_paths(paths, batch_size=batch_size)
    text_embs = encoder.get_text_embeddings(class_names, display_names=display_names)
    Y, _ = encoder.zero_shot_scores(image_embs, text_embs)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, image_embs=image_embs, text_embs=text_embs, Y=Y)
    print(f"  [cache] Saved {cache_path}")
    return image_embs, Y, encoder


def load_or_encode_concepts(
    encoder: VLMEncoder | None,
    model_name: str,
    pretrained: str,
    device: str | None,
    concept_phrases: list[str],
    cache_path: str,
) -> tuple[np.ndarray, VLMEncoder | None]:
    """Load cached concept text embeddings if the phrase list matches;
    otherwise (re-)encode, reusing `encoder` if one is already loaded."""
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        if list(d["phrases"]) == concept_phrases:
            print(f"  [cache] Loading {cache_path}")
            return d["embs"], encoder
        print("  [concepts] cached phrases differ from current profile — re-encoding")

    if encoder is None:
        encoder = VLMEncoder(model_name, pretrained, device=device)
    embs = encoder.encode_texts(concept_phrases)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, phrases=np.array(concept_phrases, dtype=object), embs=embs)
    print(f"  [cache] Saved {cache_path}")
    return embs, encoder


def load_knn_sp_config(sp_config_root: str, tag: str, default_k_graph: int) -> dict | None:
    """Load a previously tuned kNN Score-Propagation config for this model tag,
    if available (see ugsp/vlm_comparison.py). Returns None if not found."""
    path = os.path.join(sp_config_root, tag, "best_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("k_graph", default_k_graph)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Val-split grid search
# ──────────────────────────────────────────────────────────────────────────────

def concept_grid_search(
    image_embs: np.ndarray,
    concept_text_embs: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    grid: dict,
    W_knn: csr_matrix | None = None,
    combined_alpha_grid: list[float] | None = None,
) -> pd.DataFrame:
    """Exhaustively evaluate (top_r, normalize_incidence, propagation_steps)
    for the no-uncertainty and uncertainty-guided variants (the latter
    additionally sweeping beta), both for the pure concept hypergraph and
    — if W_knn is given — for the combined kNN∩hypergraph graph (which also
    sweeps combined_alpha_grid: masking the kNN graph collapses its density,
    so the self-loop weight tuned for the *unmasked* kNN graph is usually
    not optimal for the combined graph and needs its own search). B/masks
    are rebuilt only once per (top_r, normalize_incidence) pair.

    Returns a long-form DataFrame with one row per (method, config)."""
    rows = []
    combos = list(itertools.product(grid["top_r"], grid["normalize_incidence"]))
    print(f"  [chsp-grid] {len(combos)} (top_r, normalize_incidence) combos …")

    def _add_rows(method_no_unc, method_unc, P, alpha_val=np.nan):
        for steps in grid["propagation_steps"]:
            scores_plain = chsp_propagate(P, Y, steps=steps)
            preds = np.argmax(scores_plain, axis=1)
            rows.append({
                "method": method_no_unc,
                "top_r": top_r, "propagation_steps": steps, "beta": np.nan,
                "normalize_incidence": normalize, "alpha": alpha_val,
                "bacc": balanced_accuracy(preds, labels),
                "top1": accuracy(preds, labels),
                "macro_f1": macro_f1(preds, labels),
                "g_means": g_means(preds, labels, class_names),
            })
            for beta in grid["beta"]:
                out = uchsp_propagate(P, Y, steps=steps, beta=beta)
                preds_u = np.argmax(out["scores"], axis=1)
                rows.append({
                    "method": method_unc,
                    "top_r": top_r, "propagation_steps": steps, "beta": beta,
                    "normalize_incidence": normalize, "alpha": alpha_val,
                    "bacc": balanced_accuracy(preds_u, labels),
                    "top1": accuracy(preds_u, labels),
                    "macro_f1": macro_f1(preds_u, labels),
                    "g_means": g_means(preds_u, labels, class_names),
                })

    for combo_i, (top_r, normalize) in enumerate(combos, 1):
        r = min(top_r, image_embs.shape[0])
        B = build_incidence_matrix(image_embs, concept_text_embs, top_r=r, normalize=normalize)

        _add_rows(METHOD_NO_UNC, METHOD_UNC, build_propagation_matrix(B))
        if W_knn is not None:
            W_masked = mask_knn_with_concepts(W_knn, B)
            for alpha_val in combined_alpha_grid:
                P_combined = build_combined_propagation_matrix(W_knn, B, alpha=alpha_val, W_combined=W_masked)
                _add_rows(METHOD_COMBINED_NO_UNC, METHOD_COMBINED_UNC, P_combined, alpha_val=alpha_val)

        print(f"  [{combo_i:>3}/{len(combos)}] top_r={top_r} normalize={normalize}  done")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-model: tune on val + evaluate on test
# ──────────────────────────────────────────────────────────────────────────────

def run_one_model(
    model_name: str,
    pretrained: str,
    display_name: str,
    val_paths: list[str],
    val_labels: np.ndarray,
    test_paths: list[str],
    test_labels: np.ndarray,
    class_names: list[str],
    concept_phrases: list[str],
    style_concepts_json: str,
    grid: dict,
    args,
    display_names: dict[str, str] | None,
) -> tuple[dict, dict]:
    tag = f"{model_name}_{pretrained}".replace("/", "-")
    cache_dir = os.path.join(args.cache_root, tag)
    model_dir = os.path.join(args.save_dir, tag)
    os.makedirs(model_dir, exist_ok=True)

    # ── Embeddings for the dataset's original val+test images (reuse UGSP
    #    cache if present). This split is ONLY used to reuse two separate
    #    cache files — the two halves are pooled and freely re-partitioned
    #    per seed below, so this is not the tuning/eval split. ────────────────
    print("\n[phase] VAL-CACHE EMBEDDINGS")
    val_embs, val_Y, encoder = load_or_encode_embeddings(
        model_name, pretrained, val_paths, class_names,
        os.path.join(cache_dir, "val_embeddings.npz"),
        args.batch_size, args.device, display_names,
    )
    val_preds = np.argmax(val_Y, 1)
    print(f"  val-cache: {len(val_labels)} images  "
          f"top-1={accuracy(val_preds, val_labels):.4f}  "
          f"bAcc={balanced_accuracy(val_preds, val_labels):.4f}")

    # ── Concept text embeddings ───────────────────────────────────────────────
    print("\n[phase] CONCEPT EMBEDDINGS")
    concept_text_embs, encoder = load_or_encode_concepts(
        encoder, model_name, pretrained, args.device,
        concept_phrases, os.path.join(model_dir, "concept_text_embs.npz"),
    )
    print(f"  {len(concept_phrases)} concept phrases encoded")

    # ── kNN SP baseline config (reused from ugsp/vlm_comparison tuning) ──────
    knn_sp_cfg = load_knn_sp_config(args.sp_config_root, tag, args.sp_default_k_graph)
    if knn_sp_cfg is None:
        print(f"  [knn-sp] no cached config found under {args.sp_config_root}/{tag} — "
              f"kNN SP and kNN+Concept Hypergraph baselines skipped")

    # ── Test embeddings (loaded now so the pool below is complete) ───────────
    print("\n[phase] TEST EMBEDDINGS")
    test_embs, test_Y, encoder = load_or_encode_embeddings(
        model_name, pretrained, test_paths, class_names,
        os.path.join(cache_dir, "test_embeddings.npz"),
        args.batch_size, args.device, display_names,
    )
    test_preds = np.argmax(test_Y, 1)
    print(f"  test: {len(test_labels)} images  "
          f"top-1={accuracy(test_preds, test_labels):.4f}  "
          f"bAcc={balanced_accuracy(test_preds, test_labels):.4f}")

    # ── Pool val+test — no fixed train/val/test split. Every seed draws its
    #    own val:test partition from this pool (val_frac = args.eval_frac),
    #    so hyperparameter tuning is never repeated against a single fixed
    #    val set (which let selection overfit as the grid grew). ─────────────
    pool_embs = np.concatenate([val_embs, test_embs], axis=0)
    pool_Y = np.concatenate([val_Y, test_Y], axis=0)
    pool_labels = np.concatenate([val_labels, test_labels], axis=0)
    print(f"\n[phase] POOL (val+test, re-split per seed): {len(pool_labels)} images  "
          f"val_frac={args.eval_frac}")

    combined_methods = {METHOD_COMBINED_NO_UNC, METHOD_COMBINED_UNC}
    unc_methods = [METHOD_NO_UNC, METHOD_UNC]
    if knn_sp_cfg is not None:
        unc_methods += [METHOD_COMBINED_NO_UNC, METHOD_COMBINED_UNC]

    # ── Per-seed: split pool -> tune on this seed's val -> evaluate on this
    #    seed's test ────────────────────────────────────────────────────────
    print(f"\n[phase] PER-SEED TUNE + EVALUATE  ({len(args.seeds)} seeds)")
    all_seed_results: list[dict] = []
    best_cfg_per_seed: dict[int, dict] = {}
    grid_dfs: list[pd.DataFrame] = []

    for i, seed in enumerate(args.seeds, 1):
        val_idx, test_idx = stratified_two_way_split(pool_labels, seed, val_frac=args.eval_frac)
        val_embs_s, val_Y_s, val_labels_s = pool_embs[val_idx], pool_Y[val_idx], pool_labels[val_idx]
        embs_s, Y_s, labels_s = pool_embs[test_idx], pool_Y[test_idx], pool_labels[test_idx]
        print(f"\n  [seed {seed}  ({i}/{len(args.seeds)})]  val={len(val_idx)}  test={len(test_idx)}")

        val_W_knn_s = None
        if knn_sp_cfg is not None:
            val_W_knn_s = build_knn_graph(val_embs_s, k=knn_sp_cfg["k_graph"], metric="cosine")
        grid_df = concept_grid_search(
            val_embs_s, concept_text_embs, val_Y_s, val_labels_s, class_names, grid,
            W_knn=val_W_knn_s, combined_alpha_grid=args.combined_alpha,
        )
        grid_df["seed"] = seed
        grid_dfs.append(grid_df)

        best_cfg_seed: dict[str, dict] = {}
        for method in unc_methods:
            sub = grid_df[grid_df["method"] == method].sort_values("bacc", ascending=False)
            row = sub.iloc[0]
            cfg = {
                "top_r": int(row["top_r"]),
                "propagation_steps": int(row["propagation_steps"]),
                "beta": None if pd.isna(row["beta"]) else float(row["beta"]),
                "normalize_incidence": bool(row["normalize_incidence"]),
                "val_bacc": float(row["bacc"]),
            }
            if method in combined_methods:
                cfg["alpha"] = float(row["alpha"])
            best_cfg_seed[method] = cfg
        if knn_sp_cfg is not None:
            best_cfg_seed[METHOD_KNN_SP] = knn_sp_cfg
        best_cfg_per_seed[seed] = best_cfg_seed
        print(f"    Best {METHOD_UNC}: {best_cfg_seed[METHOD_UNC]}")
        if knn_sp_cfg is not None:
            print(f"    Best {METHOD_COMBINED_UNC}: {best_cfg_seed[METHOD_COMBINED_UNC]}")

        # ── Evaluate this seed's tuned configs on this seed's test split ────
        method_scores = {METHOD_VANILLA: Y_s}

        W_knn_s = None
        if knn_sp_cfg is not None:
            W_knn_s = build_knn_graph(embs_s, k=knn_sp_cfg["k_graph"], metric="cosine")
            method_scores[METHOD_KNN_SP] = score_propagation(
                W_knn_s, Y_s, K=knn_sp_cfg["K"], alpha=knn_sp_cfg["alpha"],
            )

        cfg_plain = best_cfg_seed[METHOD_NO_UNC]
        B_plain = build_incidence_matrix(
            embs_s, concept_text_embs, top_r=cfg_plain["top_r"],
            normalize=cfg_plain["normalize_incidence"],
        )
        method_scores[METHOD_NO_UNC] = chsp_propagate(
            build_propagation_matrix(B_plain), Y_s, steps=cfg_plain["propagation_steps"],
        )

        cfg_unc = best_cfg_seed[METHOD_UNC]
        B_unc = build_incidence_matrix(
            embs_s, concept_text_embs, top_r=cfg_unc["top_r"],
            normalize=cfg_unc["normalize_incidence"],
        )
        method_scores[METHOD_UNC] = uchsp_propagate(
            build_propagation_matrix(B_unc), Y_s, steps=cfg_unc["propagation_steps"], beta=cfg_unc["beta"],
        )["scores"]

        if W_knn_s is not None:
            cfg_comb_plain = best_cfg_seed[METHOD_COMBINED_NO_UNC]
            B_comb_plain = build_incidence_matrix(
                embs_s, concept_text_embs, top_r=cfg_comb_plain["top_r"],
                normalize=cfg_comb_plain["normalize_incidence"],
            )
            method_scores[METHOD_COMBINED_NO_UNC] = chsp_propagate(
                build_combined_propagation_matrix(W_knn_s, B_comb_plain, alpha=cfg_comb_plain["alpha"]),
                Y_s, steps=cfg_comb_plain["propagation_steps"],
            )

            cfg_comb_unc = best_cfg_seed[METHOD_COMBINED_UNC]
            B_comb_unc = build_incidence_matrix(
                embs_s, concept_text_embs, top_r=cfg_comb_unc["top_r"],
                normalize=cfg_comb_unc["normalize_incidence"],
            )
            method_scores[METHOD_COMBINED_UNC] = uchsp_propagate(
                build_combined_propagation_matrix(W_knn_s, B_comb_unc, alpha=cfg_comb_unc["alpha"]),
                Y_s, steps=cfg_comb_unc["propagation_steps"], beta=cfg_comb_unc["beta"],
            )["scores"]

        seed_results = compare_methods(method_scores, labels_s, class_names)
        print_comparison_table(seed_results)
        all_seed_results.append(seed_results)

    pd.concat(grid_dfs, ignore_index=True).to_csv(
        os.path.join(model_dir, "grid_search_val.csv"), index=False,
    )
    with open(os.path.join(model_dir, "best_config_per_seed.json"), "w") as f:
        json.dump(best_cfg_per_seed, f, indent=2)

    # Modal (most frequent) hyperparameters across seeds, for a single
    # representative summary row — the authoritative per-seed configs (what
    # was actually used to produce each seed's test metrics) are in
    # best_config_per_seed.json.
    def _modal_cfg(method: str) -> dict | None:
        seed_cfgs = [best_cfg_per_seed[s][method] for s in args.seeds if method in best_cfg_per_seed[s]]
        if not seed_cfgs:
            return None
        modal = {}
        for k in seed_cfgs[0]:
            if k.startswith("val_"):
                continue
            vals = [c[k] for c in seed_cfgs]
            modal[k] = max(set(vals), key=vals.count)
        val_baccs = [c["val_bacc"] for c in seed_cfgs if "val_bacc" in c]
        if val_baccs:
            modal["val_bacc_mean"] = float(np.mean(val_baccs))
        return modal

    all_methods = [METHOD_NO_UNC, METHOD_UNC]
    if knn_sp_cfg is not None:
        all_methods += [METHOD_COMBINED_NO_UNC, METHOD_COMBINED_UNC, METHOD_KNN_SP]
    best_cfg = {m: _modal_cfg(m) for m in all_methods if _modal_cfg(m) is not None}

    with open(os.path.join(model_dir, "best_config.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    agg = aggregate_multi_seed_results(all_seed_results, class_names)
    print_multi_seed_summary(agg, n_seeds=len(args.seeds))

    extra = {
        "vlm": display_name, "model_name": model_name, "pretrained": pretrained,
        "modal_config": best_cfg, "concepts_json": style_concepts_json,
        "seeds": args.seeds, "val_frac": args.eval_frac,
    }
    save_multi_seed_results(agg, model_dir, extra=extra)
    return agg, best_cfg


# ──────────────────────────────────────────────────────────────────────────────
# Cross-model result saving
# ──────────────────────────────────────────────────────────────────────────────

def save_final_results(
    model_results: list[tuple[str, dict, dict]],
    save_dir: str,
    metrics: tuple[str, ...] = ("top1", "bacc", "g_means", "macro_f1"),
) -> None:
    rows = []
    for display_name, agg, best_cfg in model_results:
        for method, m_dict in agg.items():
            row: dict = {"model": display_name, "method": method}
            cfg = best_cfg.get(method)
            if cfg:
                for k, v in cfg.items():
                    if k != "val_bacc":
                        row[f"cfg_{k}"] = v
            for metric in metrics:
                stats = m_dict.get(metric)
                if stats:
                    row[f"{metric}_mean"] = stats["mean"]
                    row[f"{metric}_std"] = stats["std"]
            rows.append(row)

    results_path = os.path.join(save_dir, "concept_hypergraph_results.csv")
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"[save] Results → {results_path}")

    best_config_path = os.path.join(save_dir, "concept_hypergraph_best_config.json")
    with open(best_config_path, "w") as f:
        json.dump({d: c for d, _, c in model_results}, f, indent=2)
    print(f"[save] Best configs → {best_config_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    t0 = time.time()

    style_concepts = load_style_concepts(args.concepts_json)

    # ── Data (mirrors ugsp/vlm_comparison.py exactly, so indices line up with
    #    its cached embeddings) ─────────────────────────────────────────────
    if args.raw_csv:
        df = pd.read_csv(args.csv_path)
        print(f"[dataset] Loaded raw CSV: {len(df)} rows, {df['style'].nunique()} styles")
        if args.min_images > 0:
            counts = df["style"].value_counts()
            df = df[df["style"].isin(counts[counts >= args.min_images].index)].reset_index(drop=True)
        if args.selected_styles:
            df = df[df["style"].isin(args.selected_styles)].reset_index(drop=True)
        display_names = None
    else:
        df = prepare_dataframe(args.csv_path)
        df = maybe_filter_styles(
            df, min_images=args.min_images,
            top_n=args.top_n_styles, selected=args.selected_styles,
        )
        display_names = STYLE_DISPLAY_NAMES

    all_styles = set(df["style"].unique())
    for split in ("train", "val", "test"):
        all_styles &= set(df[df["new_subset"] == split]["style"].unique())
    class_names = sorted(all_styles)
    print(f"\n[data] {len(class_names)} styles: {class_names}")

    val_paths, val_labels = get_split(df, "val", args.image_root, class_names)
    test_paths, test_labels = get_split(df, "test", args.image_root, class_names)
    print(f"[data] val={len(val_labels)}  test={len(test_labels)}")

    concept_phrases, concept_styles = flatten_concepts(style_concepts, class_names)
    print(f"[concepts] {len(concept_phrases)} concept phrases across {len(class_names)} styles")

    grid = {
        "top_r": args.top_r,
        "propagation_steps": args.propagation_steps,
        "beta": args.beta,
        "normalize_incidence": args.normalize_incidence,
    }

    model_results: list[tuple[str, dict, dict]] = []
    for model_name, pretrained, display_name in args.models:
        print(f"\n{'━'*70}")
        print(f"MODEL: {display_name}")
        print(f"{'━'*70}")

        agg, best_cfg = run_one_model(
            model_name, pretrained, display_name,
            val_paths, val_labels, test_paths, test_labels,
            class_names, concept_phrases, args.concepts_json,
            grid, args, display_names,
        )
        model_results.append((display_name, agg, best_cfg))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_final_results(model_results, args.save_dir)

    with open(os.path.join(args.save_dir, "run_summary.json"), "w") as f:
        json.dump({
            "models": [display_name for _, _, display_name in args.models],
            "concepts_json": args.concepts_json,
            "seeds": args.seeds,
            "eval_frac": args.eval_frac,
            "elapsed_sec": round(time.time() - t0, 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    print(f"\n[done] All outputs in {args.save_dir}  ({round(time.time()-t0,1)}s)")
    return model_results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def get_args():
    p = argparse.ArgumentParser(description="Concept Hypergraph Score Propagation for zero-shot painting style classification")
    p.add_argument("--config", type=str, default=os.path.join(HYPER_ROOT, "configs", "concept_hypergraph.yaml"))

    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--image_root", type=str, default=None)
    p.add_argument("--raw_csv", action="store_true", default=None)
    p.add_argument("--min_images", type=int, default=None)
    p.add_argument("--top_n_styles", type=int, default=None)
    p.add_argument("--selected_styles", type=str, nargs="+", default=None)

    p.add_argument("--concepts_json", type=str, default=None)
    p.add_argument("--cache_root", type=str, default=None)
    p.add_argument("--sp_config_root", type=str, default=None)
    p.add_argument("--sp_default_k_graph", type=int, default=None)
    p.add_argument("--save_dir", type=str, default=None)

    p.add_argument("--models", type=str, nargs="+", default=None,
                    help="Override model list. Each entry: 'model_name/pretrained_tag/display_name'.")

    p.add_argument("--top_r", type=int, nargs="+", default=None)
    p.add_argument("--propagation_steps", type=int, nargs="+", default=None)
    p.add_argument("--beta", type=float, nargs="+", default=None)
    p.add_argument("--normalize_incidence", type=str2bool, nargs="+", default=None)
    p.add_argument("--combined_alpha", type=float, nargs="+", default=None,
                    help="Self-loop weight grid for the kNN+Concept Hypergraph combined graph. "
                         "Retuned separately from the plain kNN SP alpha because masking the kNN "
                         "graph by concept co-membership sharply increases its sparsity.")

    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--eval_frac", type=float, default=None)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    ns = p.parse_args()
    cfg = load_config(ns.config)

    def pick(cli_val, *cfg_path, default=None):
        if cli_val is not None:
            return cli_val
        node = cfg
        for key in cfg_path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node if node is not None else default

    ns.csv_path = resolve_path(pick(ns.csv_path, "data", "csv_path"))
    ns.image_root = resolve_path(pick(ns.image_root, "data", "image_root"))
    ns.raw_csv = pick(ns.raw_csv, "data", "raw_csv", default=False)
    ns.min_images = pick(ns.min_images, "data", "min_images", default=0)
    ns.top_n_styles = pick(ns.top_n_styles, "data", "top_n_styles", default=0)
    ns.selected_styles = pick(ns.selected_styles, "data", "selected_styles")

    ns.concepts_json = resolve_path(pick(ns.concepts_json, "concepts_json"))
    ns.cache_root = resolve_path(pick(ns.cache_root, "cache_root"))
    ns.sp_config_root = resolve_path(pick(ns.sp_config_root, "sp_config_root"))
    ns.sp_default_k_graph = pick(ns.sp_default_k_graph, "sp_default_k_graph", default=10)
    ns.save_dir = resolve_path(pick(ns.save_dir, "save_dir", default="results"))

    if ns.models is not None:
        parsed = []
        for spec in ns.models:
            parts = spec.split("/")
            if len(parts) == 3:
                parsed.append(tuple(parts))
            elif len(parts) == 2:
                parsed.append((parts[0], parts[1], f"{parts[0]}/{parts[1]}"))
            else:
                raise ValueError(f"--models entry must be model/pretrained[/display]: {spec}")
        ns.models = parsed
    else:
        cfg_models = cfg.get("models")
        if not cfg_models:
            raise ValueError("No models specified: pass --models or set 'models' in the config file")
        ns.models = [tuple(m) for m in cfg_models]

    ns.top_r = pick(ns.top_r, "grid", "top_r")
    ns.propagation_steps = pick(ns.propagation_steps, "grid", "propagation_steps")
    ns.beta = pick(ns.beta, "grid", "beta")
    ns.normalize_incidence = pick(ns.normalize_incidence, "grid", "normalize_incidence")
    ns.combined_alpha = pick(ns.combined_alpha, "grid", "combined_alpha", default=[0.1, 0.3, 0.5, 0.7, 0.9])

    ns.seeds = pick(ns.seeds, "eval", "seeds")
    ns.eval_frac = pick(ns.eval_frac, "eval", "eval_frac")

    ns.batch_size = pick(ns.batch_size, "runtime", "batch_size", default=64)
    ns.device = pick(ns.device, "runtime", "device")
    ns.seed = pick(ns.seed, "runtime", "seed", default=42)

    required = ["csv_path", "image_root", "concepts_json", "cache_root", "top_r",
                "propagation_steps", "beta", "normalize_incidence", "seeds", "eval_frac"]
    missing = [k for k in required if getattr(ns, k) is None]
    if missing:
        raise ValueError(f"Missing required settings (via --config or CLI flags): {missing}")

    return ns


if __name__ == "__main__":
    args = get_args()
    run(args)
