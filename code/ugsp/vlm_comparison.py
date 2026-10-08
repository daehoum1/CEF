"""
vlm_comparison.py — Cross-VLM consistency check for UGSP
=========================================================

Tests whether UGSP consistently improves over the base VLM and Score Propagation
across multiple pretrained VLMs (all loaded via open_clip).

Default models tested (all ViT-B scale for a fair comparison):
  1. ViT-B-32        | openai              — original CLIP
  2. ViT-B-32        | metaclip_fullcc     — MetaCLIP (Meta, curated data)
  3. EVA02-B-16      | merged2b_s8b_b131k  — EVA02-CLIP (BAAI)
  4. ViT-B-16-SigLIP | webli               — SigLIP (Google, sigmoid loss)

Protocol:
  • Each VLM is tuned independently on the VAL split via exhaustive grid search:
      K     ∈ {1, 2}
      alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
      beta  ∈ {0.25, 0.5, 1.0, 2.0, 4.0}
  • Best config (by val ugsp_bacc) is then used to evaluate on TEST split
    with 5 stratified random seeds.
  • Prints a cross-model summary table showing Vanilla → SP → UGSP delta.

Usage:
  python vlm_comparison.py \\
      --csv_path   /path/to/wikiart/classes_artist_disjoint_fixed.csv \\
      --image_root /path/to/wikiart \\
      --save_dir   ./outputs/vlm_comparison_tuned \\
      --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import softmax

from ugsp.clip_encoder import PROMPT_TEMPLATES, STYLE_DISPLAY_NAMES
from ugsp.dataset import maybe_filter_styles, prepare_dataframe
from ugsp.evaluate import (
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
from ugsp.graph import build_knn_graph
from ugsp.propagation import ugsp

# ──────────────────────────────────────────────────────────────────────────────
# Model registry & hyperparameter grid
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS: list[tuple[str, str, str]] = [
    ("ViT-B-32",         "openai",               "CLIP ViT-B/32 (OpenAI)"),
    ("ViT-B-32",         "metaclip_fullcc",       "MetaCLIP ViT-B/32"),
    ("EVA02-B-16",       "merged2b_s8b_b131k",    "EVA02-CLIP ViT-B/16"),
    ("ViT-B-16-SigLIP",  "webli",                 "SigLIP ViT-B/16"),
]

K_GRAPH_VALUES = [3, 5, 10, 20]
K_VALUES       = [1, 2]
ALPHA_VALUES   = [0.1, 0.3, 0.5, 0.7, 0.9]
BETA_VALUES    = [0.25, 0.5, 1.0, 2.0, 4.0]


# ──────────────────────────────────────────────────────────────────────────────
# Generic open_clip encoder
# ──────────────────────────────────────────────────────────────────────────────

class VLMEncoder:
    """Thin open_clip wrapper with the same interface as CLIPEncoder."""

    def __init__(self, model_name: str, pretrained: str, device: str | None = None):
        import open_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"[vlm] {model_name}/{pretrained}  device={device}")

    @torch.no_grad()
    def encode_images_from_paths(
        self, paths: list[str], batch_size: int = 64
    ) -> np.ndarray:
        all_feats: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            imgs = [self.preprocess(Image.open(p).convert("RGB")) for p in batch]
            imgs_t = torch.stack(imgs).to(self.device)
            feats = self.model.encode_image(imgs_t).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
            if (start // batch_size) % 10 == 0:
                print(f"  encoded {start + len(batch)}/{len(paths)} images", end="\r")
        print()
        return np.concatenate(all_feats, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def get_text_embeddings(
        self,
        class_names: list[str],
        templates: list[str] = PROMPT_TEMPLATES,
        display_names: dict[str, str] | None = STYLE_DISPLAY_NAMES,
    ) -> np.ndarray:
        resolved = [display_names.get(c, c) if display_names else c for c in class_names]
        stacked = []
        for tmpl in templates:
            prompts = [tmpl.format(name) for name in resolved]
            stacked.append(self.encode_texts(prompts))
        avg = np.mean(np.stack(stacked, axis=0), axis=0)
        avg = avg / (np.linalg.norm(avg, axis=1, keepdims=True) + 1e-8)
        return avg

    def zero_shot_scores(
        self, image_embs: np.ndarray, text_embs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        logit_scale = 1.0
        if hasattr(self.model, "logit_scale"):
            logit_scale = self.model.logit_scale.exp().item()
        logits = (image_embs @ text_embs.T) * logit_scale
        # SigLIP logit_bias is a constant shift across all classes → cancels in softmax
        probs = softmax(logits, axis=1)
        return probs, logits


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_split(df, subset: str, image_root: str, class_names: list) -> tuple:
    sdf = df[df["new_subset"] == subset].copy().reset_index(drop=True)
    sdf = sdf[sdf["style"].isin(class_names)].reset_index(drop=True)
    c2i = {c: i for i, c in enumerate(class_names)}
    paths  = [os.path.join(image_root, r["filename"]) for _, r in sdf.iterrows()]
    labels = np.array([c2i[s] for s in sdf["style"]])
    return paths, labels


def extract_or_load(
    encoder: VLMEncoder,
    paths: list[str],
    labels: np.ndarray,
    class_names: list[str],
    cache_path: str,
    batch_size: int,
    display_names: dict[str, str] | None = STYLE_DISPLAY_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        print(f"  [cache] Loading {cache_path}")
        d = np.load(cache_path)
        return d["image_embs"], d["Y"], d["text_embs"]

    print(f"  [embed] Encoding {len(paths)} images …")
    image_embs = encoder.encode_images_from_paths(paths, batch_size=batch_size)
    text_embs  = encoder.get_text_embeddings(class_names, display_names=display_names)
    Y, _       = encoder.zero_shot_scores(image_embs, text_embs)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, image_embs=image_embs, text_embs=text_embs, Y=Y)
    print(f"  [cache] Saved {cache_path}")
    return image_embs, Y, text_embs


def stratified_random_split(
    labels: np.ndarray, seed: int, eval_frac: float = 0.2
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eval_idx: list[int] = []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        n = max(1, int(len(c_idx) * eval_frac))
        eval_idx.extend(rng.choice(c_idx, size=n, replace=False).tolist())
    return np.array(eval_idx)


# ──────────────────────────────────────────────────────────────────────────────
# Grid search (val split)
# ──────────────────────────────────────────────────────────────────────────────

def grid_search(
    graphs: dict,  # {k_graph: W_knn}
    Y: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Exhaustively evaluate all (k_graph, K, alpha, beta) on the val split.

    Returns a DataFrame sorted by ugsp_bacc descending.
    """
    rows = []
    combos = list(itertools.product(K_GRAPH_VALUES, K_VALUES, ALPHA_VALUES, BETA_VALUES))
    print(f"  [grid] {len(combos)} combinations …")

    for i, (k_graph, K, alpha, beta) in enumerate(combos, 1):
        W_knn = graphs[k_graph]
        res = ugsp(W_knn, Y, K=K, alpha=alpha, beta=beta)

        preds_sp   = np.argmax(res["scores_sp"],   axis=1)
        preds_ugsp = np.argmax(res["scores_ugsp"], axis=1)

        rows.append({
            "k_graph": k_graph, "K": K, "alpha": alpha, "beta": beta,
            "sp_bacc":     balanced_accuracy(preds_sp,   labels),
            "sp_top1":     accuracy(preds_sp,   labels),
            "sp_g_means":  g_means(preds_sp,   labels, class_names),
            "sp_macro_f1": macro_f1(preds_sp,   labels),
            "ugsp_bacc":     balanced_accuracy(preds_ugsp, labels),
            "ugsp_top1":     accuracy(preds_ugsp, labels),
            "ugsp_g_means":  g_means(preds_ugsp, labels, class_names),
            "ugsp_macro_f1": macro_f1(preds_ugsp, labels),
        })

        if i % 20 == 0 or i == len(combos):
            best = max(rows, key=lambda r: r["ugsp_bacc"])
            print(
                f"  [{i:>3}/{len(combos)}] k={k_graph} K={K} α={alpha} β={beta}  "
                f"ugsp_bacc={rows[-1]['ugsp_bacc']:.4f}  "
                f"(best: k={best['k_graph']} K={best['K']} α={best['alpha']} β={best['beta']} "
                f"→ {best['ugsp_bacc']:.4f})"
            )

    return pd.DataFrame(rows).sort_values("ugsp_bacc", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Per-model: tune on val + evaluate on test
# ──────────────────────────────────────────────────────────────────────────────

def run_one_model(
    encoder: VLMEncoder,
    display_name: str,
    val_paths: list[str],
    val_labels: np.ndarray,
    test_paths: list[str],
    test_labels: np.ndarray,
    class_names: list[str],
    args,
    display_names: dict[str, str] | None = STYLE_DISPLAY_NAMES,
) -> tuple[dict, dict]:
    """Tune on val, evaluate on test. Returns (agg_results, best_config)."""
    tag = f"{encoder.model_name}_{encoder.pretrained}".replace("/", "-")
    model_dir = os.path.join(args.save_dir, tag)
    os.makedirs(model_dir, exist_ok=True)

    # ── Val embeddings ────────────────────────────────────────────────────────
    print(f"\n[phase] VAL EMBEDDINGS")
    val_cache = os.path.join(model_dir, "val_embeddings.npz")
    val_embs, val_Y, _ = extract_or_load(
        encoder, val_paths, val_labels, class_names, val_cache, args.batch_size,
        display_names=display_names,
    )
    val_preds = np.argmax(val_Y, 1)
    print(
        f"  val: {len(val_labels)} images  "
        f"top-1={accuracy(val_preds, val_labels):.4f}  "
        f"bAcc={balanced_accuracy(val_preds, val_labels):.4f}"
    )

    # ── Val kNN graphs (one per k_graph candidate) ───────────────────────────
    print(f"\n[phase] VAL GRAPHS")
    N_val = len(val_embs)
    val_graphs: dict = {}
    for k_g in K_GRAPH_VALUES:
        W = build_knn_graph(val_embs, k=k_g, metric="cosine")
        val_graphs[k_g] = W
        print(f"  k={k_g:2d}, density={W.nnz/(N_val*N_val):.6f} ({W.nnz} edges)")

    # ── Grid search on val ────────────────────────────────────────────────────
    print(f"\n[phase] GRID SEARCH (val)")
    grid_df = grid_search(val_graphs, val_Y, val_labels, class_names)
    grid_df.to_csv(os.path.join(model_dir, "grid_search_val.csv"), index=False)

    best_row     = grid_df.iloc[0]
    best_k_graph = int(best_row["k_graph"])
    best_K       = int(best_row["K"])
    best_alpha   = float(best_row["alpha"])
    best_beta    = float(best_row["beta"])

    print(f"\n  Best config: k={best_k_graph}  K={best_K}  alpha={best_alpha}  beta={best_beta}")
    print(f"  val SP   bAcc={best_row['sp_bacc']:.4f}  top-1={best_row['sp_top1']:.4f}")
    print(f"  val UGSP bAcc={best_row['ugsp_bacc']:.4f}  top-1={best_row['ugsp_top1']:.4f}")
    print(f"\n  Top-5 configs:")
    print(grid_df.head(5)[["k_graph","K","alpha","beta","sp_bacc","ugsp_bacc"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    best_config = {
        "k_graph": best_k_graph, "K": best_K, "alpha": best_alpha, "beta": best_beta,
        "val_sp_bacc":   float(best_row["sp_bacc"]),
        "val_ugsp_bacc": float(best_row["ugsp_bacc"]),
    }
    with open(os.path.join(model_dir, "best_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)

    # ── Test embeddings ───────────────────────────────────────────────────────
    print(f"\n[phase] TEST EMBEDDINGS")
    test_cache = os.path.join(model_dir, "test_embeddings.npz")
    test_embs, test_Y, _ = extract_or_load(
        encoder, test_paths, test_labels, class_names, test_cache, args.batch_size,
        display_names=display_names,
    )
    test_preds = np.argmax(test_Y, 1)
    print(
        f"  test: {len(test_labels)} images  "
        f"top-1={accuracy(test_preds, test_labels):.4f}  "
        f"bAcc={balanced_accuracy(test_preds, test_labels):.4f}"
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n[phase] TEST EVALUATION  (k={best_k_graph} K={best_K} α={best_alpha} β={best_beta})")
    all_seed_results: list[dict] = []

    for i, seed in enumerate(args.seeds, 1):
        eval_idx = stratified_random_split(test_labels, seed, args.eval_frac)
        embs_s   = test_embs[eval_idx]
        labels_s = test_labels[eval_idx]
        Y_s      = test_Y[eval_idx]

        N = len(eval_idx)
        W_knn = build_knn_graph(embs_s, k=best_k_graph, metric="cosine")
        print(f"  [seed {seed}  ({i}/{len(args.seeds)})]  n={N}  density={W_knn.nnz/(N*N):.6f}")

        res = ugsp(W_knn, Y_s, K=best_K, alpha=best_alpha, beta=best_beta)
        seed_results = compare_methods(
            {
                "Vanilla VLM":       Y_s,
                "Score Propagation": res["scores_sp"],
                "UGSP (ours)":       res["scores_ugsp"],
            },
            labels_s, class_names,
        )
        print_comparison_table(seed_results)
        all_seed_results.append(seed_results)

    agg = aggregate_multi_seed_results(all_seed_results, class_names)
    print_multi_seed_summary(agg, n_seeds=len(args.seeds))

    extra = {
        "vlm": display_name,
        "model_name": encoder.model_name,
        "pretrained": encoder.pretrained,
        **best_config,
        "seeds": args.seeds,
        "eval_frac": args.eval_frac,
    }
    save_multi_seed_results(agg, model_dir, extra=extra)
    return agg, best_config


# ──────────────────────────────────────────────────────────────────────────────
# Cross-model summary tables
# ──────────────────────────────────────────────────────────────────────────────

def print_cross_model_table(
    model_results: list[tuple[str, dict, dict]],
    metric: str = "bacc",
) -> None:
    label_map = {"bacc": "bAcc", "top1": "Top-1", "g_means": "GMeans", "macro_f1": "MacroF1"}
    mname = label_map.get(metric, metric)
    W = 30
    print("\n" + "=" * (W + 80))
    print(f"CROSS-MODEL SUMMARY  [{mname}]  (mean ± std  across seeds)")
    print("=" * (W + 80))
    header = (
        f"{'Model':<{W}} {'Best K/α/β':>12} {'Vanilla VLM':>18}"
        f" {'Score Prop.':>18} {'UGSP (ours)':>18} {'Δ UGSP-Vanilla':>14}"
    )
    print(header)
    print("-" * (W + 80))

    for display_name, agg, best_cfg in model_results:
        def fmt(method: str) -> str:
            m = agg.get(method, {}).get(metric)
            if m is None:
                return "      N/A      "
            return f"{m['mean']:.4f} ± {m['std']:.4f}"

        v_mean  = agg.get("Vanilla VLM",  {}).get(metric, {}).get("mean", float("nan"))
        ug_mean = agg.get("UGSP (ours)",  {}).get(metric, {}).get("mean", float("nan"))
        delta   = ug_mean - v_mean
        cfg_str = f"k={best_cfg['k_graph']}/{best_cfg['K']}/{best_cfg['alpha']}/{best_cfg['beta']}"

        print(
            f"{display_name:<{W}} {cfg_str:>12} {fmt('Vanilla VLM'):>18}"
            f" {fmt('Score Propagation'):>18} {fmt('UGSP (ours)'):>18}"
            f" {delta:>+14.4f}"
        )
    print("=" * (W + 80))


def save_cross_model_csv(
    model_results: list[tuple[str, dict, dict]],
    save_dir: str,
    metrics: tuple[str, ...] = ("top1", "bacc", "g_means", "macro_f1"),
) -> None:
    rows = []
    for display_name, agg, best_cfg in model_results:
        for method, m_dict in agg.items():
            row: dict = {
                "model": display_name, "method": method,
                "best_k_graph": best_cfg["k_graph"],
                "best_K": best_cfg["K"],
                "best_alpha": best_cfg["alpha"],
                "best_beta": best_cfg["beta"],
            }
            for metric in metrics:
                stats = m_dict.get(metric)
                if stats:
                    row[f"{metric}_mean"] = stats["mean"]
                    row[f"{metric}_std"]  = stats["std"]
            rows.append(row)
    path = os.path.join(save_dir, "cross_model_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[save] Cross-model summary → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    t0 = time.time()

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.raw_csv:
        # MultitaskPainting100k (or any pre-processed CSV with new_subset column)
        # — skip WikiArt-specific remapping and use style names as-is
        df = pd.read_csv(args.csv_path)
        print(f"[dataset] Loaded raw CSV: {len(df)} rows, {df['style'].nunique()} styles")
        if args.min_images > 0:
            counts = df["style"].value_counts()
            df = df[df["style"].isin(counts[counts >= args.min_images].index)].reset_index(drop=True)
        if args.selected_styles:
            df = df[df["style"].isin(args.selected_styles)].reset_index(drop=True)
        display_names = None  # use style names directly in CLIP prompts
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

    val_paths,  val_labels  = get_split(df, "val",  args.image_root, class_names)
    test_paths, test_labels = get_split(df, "test", args.image_root, class_names)
    print(f"[data] val={len(val_labels)}  test={len(test_labels)}")

    # ── Model list ────────────────────────────────────────────────────────────
    models_to_run: list[tuple[str, str, str]] = []
    if args.models:
        for spec in args.models:
            parts = spec.split("/")
            if len(parts) == 3:
                models_to_run.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                models_to_run.append((parts[0], parts[1], f"{parts[0]}/{parts[1]}"))
            else:
                raise ValueError(f"--models entry must be model/pretrained[/display]: {spec}")
    else:
        models_to_run = DEFAULT_MODELS

    # ── Per-model: tune → evaluate ────────────────────────────────────────────
    model_results: list[tuple[str, dict, dict]] = []

    for model_name, pretrained, display_name in models_to_run:
        print(f"\n{'━'*70}")
        print(f"MODEL: {display_name}")
        print(f"{'━'*70}")

        encoder = VLMEncoder(model_name, pretrained, device=args.device)
        agg, best_cfg = run_one_model(
            encoder, display_name,
            val_paths, val_labels,
            test_paths, test_labels,
            class_names, args,
            display_names=display_names,
        )
        model_results.append((display_name, agg, best_cfg))
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Cross-model summary ───────────────────────────────────────────────────
    for metric in ("bacc", "top1", "g_means", "macro_f1"):
        print_cross_model_table(model_results, metric=metric)

    save_cross_model_csv(model_results, args.save_dir)

    with open(os.path.join(args.save_dir, "run_summary.json"), "w") as f:
        json.dump({
            "models": [d for _, _, d in models_to_run],
            "per_model_best": {
                d: {"K": c["K"], "alpha": c["alpha"], "beta": c["beta"]}
                for d, _, c in model_results
            },
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

def get_args():
    p = argparse.ArgumentParser(description="Cross-VLM consistency check for UGSP (per-model tuning)")

    p.add_argument("--csv_path",   type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--save_dir",   type=str, default="./outputs/vlm_comparison_tuned")

    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Override model list. Each entry: "
                        "'model_name/pretrained_tag/display_name'.")

    p.add_argument("--raw_csv", action="store_true",
                   help="Skip WikiArt-specific remapping — use CSV style names as-is. "
                        "Required for non-WikiArt datasets (e.g. MultitaskPainting100k). "
                        "CSV must already have a 'new_subset' column.")
    p.add_argument("--min_images",      type=int,  default=0)
    p.add_argument("--top_n_styles",    type=int,  default=0)
    p.add_argument("--selected_styles", type=str,  nargs="+", default=None)

    p.add_argument("--k",         type=int,   default=10)
    p.add_argument("--seeds",     type=int,   nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--eval_frac", type=float, default=0.2)
    p.add_argument("--batch_size",type=int,   default=64)
    p.add_argument("--device",    type=str,   default=None)
    p.add_argument("--seed",      type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args)
