"""
filter_concepts.py — Discriminability-based concept filtering
================================================================

Diagnoses and filters assets/style_concepts.json: for each concept phrase,
measures cosine similarity (via the VLM text encoder) between the concept's
text embedding and every style's zero-shot class-prompt embedding (the
"style anchor" — the same ensemble-template embedding used to build Y).
A concept only earns its keep if its own style's anchor is the style it is
*most* similar to among all classes; otherwise it is not actually
discriminative and is dropped (or replaced) by build_incidence_matrix's
top-r selection, since it would just pull in generically-similar images
from any style, not this one.

This tests the hypothesis that broad style paraphrases (e.g. "visible
brush strokes" for Impressionism) may sit too close to *other* styles'
anchors in text-embedding space to make useful, style-specific hyperedges.

Usage:
  python scripts/filter_concepts.py --config configs/concept_hypergraph.yaml
  # then point run_concept_hypergraph.py at the filtered file:
  python scripts/run_concept_hypergraph.py --concepts_json assets/style_concepts_filtered.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYPER_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_ROOT = os.path.dirname(HYPER_ROOT)
UGSP_ROOT = os.path.join(SRC_ROOT, "ugsp")

sys.path.insert(0, os.path.join(HYPER_ROOT, "src"))
sys.path.insert(0, UGSP_ROOT)

from concept_hypergraph import flatten_concepts, load_style_concepts  # noqa: E402
from ugsp.clip_encoder import STYLE_DISPLAY_NAMES  # noqa: E402
from vlm_comparison import VLMEncoder  # noqa: E402


def resolve_path(path: str | None) -> str | None:
    """Resolve a config/CLI path against this project's root rather than the
    current working directory (see run_concept_hypergraph.resolve_path).
    Absolute paths are returned unchanged."""
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(HYPER_ROOT, path)


def compute_similarity_report(
    model_name: str,
    pretrained: str,
    display_name: str,
    class_names: list[str],
    concept_phrases: list[str],
    concept_styles: list[str],
    device: str | None,
) -> pd.DataFrame:
    """One row per concept: its similarity to its own style's anchor vs. the
    single most-similar *other* style's anchor."""
    encoder = VLMEncoder(model_name, pretrained, device=device)
    style_anchor_embs = encoder.get_text_embeddings(class_names, display_names=STYLE_DISPLAY_NAMES)  # [C, D]
    concept_text_embs = encoder.encode_texts(concept_phrases)                                          # [E, D]
    sim = concept_text_embs @ style_anchor_embs.T                                                       # [E, C]

    rows = []
    for e, (phrase, own_style) in enumerate(zip(concept_phrases, concept_styles)):
        own_idx = class_names.index(own_style)
        own_sim = float(sim[e, own_idx])
        other_names = [c for c in class_names if c != own_style]
        others = np.delete(sim[e], own_idx)
        best_other_i = int(np.argmax(others))
        rank = int(1 + (others > own_sim).sum())  # 1 = own style is the single closest of all C styles
        rows.append({
            "model": display_name,
            "style": own_style,
            "concept": phrase,
            "own_sim": own_sim,
            "best_other_style": other_names[best_other_i],
            "best_other_sim": float(others[best_other_i]),
            "margin": own_sim - float(others[best_other_i]),
            "rank": rank,
        })
    return pd.DataFrame(rows)


def run(args):
    style_concepts = load_style_concepts(args.concepts_json)
    class_names = sorted(style_concepts.keys())
    concept_phrases, concept_styles = flatten_concepts(style_concepts, class_names)
    print(f"[concepts] {len(concept_phrases)} phrases across {len(class_names)} styles")

    per_model_dfs = []
    for model_name, pretrained, display_name in args.models:
        print(f"\n[model] {display_name}")
        df = compute_similarity_report(
            model_name, pretrained, display_name,
            class_names, concept_phrases, concept_styles, args.device,
        )
        per_model_dfs.append(df)

    full_df = pd.concat(per_model_dfs, ignore_index=True)

    # Aggregate across VLMs: a concept is only trustworthy if it discriminates
    # its style consistently, not just for one particular text encoder.
    agg = full_df.groupby(["style", "concept"], sort=False).agg(
        mean_margin=("margin", "mean"),
        min_margin=("margin", "min"),
        mean_rank=("rank", "mean"),
        worst_rank=("rank", "max"),
    ).reset_index()

    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    full_df.to_csv(args.out_report, index=False)
    agg_report_path = args.out_report.replace(".csv", "_aggregated.csv")
    agg.sort_values(["style", "mean_margin"], ascending=[True, False]).to_csv(agg_report_path, index=False)
    print(f"\n[save] Per-model report      → {args.out_report}")
    print(f"[save] Aggregated report     → {agg_report_path}")

    # ── Filter: keep concepts whose own style is (on average) the closest
    #    style anchor; always keep at least min_keep_per_style per style so
    #    every class still gets hyperedges. ──────────────────────────────
    filtered: dict[str, list[str]] = {}
    kept_total, dropped_total = 0, 0
    print(f"\n[filter] min_margin={args.min_margin}  min_keep_per_style={args.min_keep_per_style}")
    for style in class_names:
        style_rows = agg[agg["style"] == style].sort_values("mean_margin", ascending=False)
        passing = style_rows[style_rows["mean_margin"] >= args.min_margin]
        if len(passing) < args.min_keep_per_style:
            passing = style_rows.head(args.min_keep_per_style)
        kept = passing["concept"].tolist()
        dropped = [c for c in style_rows["concept"].tolist() if c not in kept]
        filtered[style] = kept
        kept_total += len(kept)
        dropped_total += len(dropped)
        print(f"  {style:<24} kept {len(kept)}/{len(style_rows)}"
              + (f"   dropped: {dropped}" if dropped else ""))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"\n[save] Filtered concepts ({kept_total} kept, {dropped_total} dropped) → {args.out_json}")

    return filtered


def get_args():
    p = argparse.ArgumentParser(description="Filter style concepts by cross-style discriminability")
    p.add_argument("--config", type=str, default=os.path.join(HYPER_ROOT, "configs", "concept_hypergraph.yaml"))
    p.add_argument("--concepts_json", type=str, default=None)
    p.add_argument("--models", type=str, nargs="+", default=None,
                    help="Override model list. Each entry: 'model_name/pretrained_tag/display_name'.")
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--out_report", type=str, default=None)
    p.add_argument("--min_margin", type=float, default=0.0,
                    help="Keep a concept if mean(own_sim - best_other_sim) across models >= this.")
    p.add_argument("--min_keep_per_style", type=int, default=1,
                    help="Always keep at least this many (highest-margin) concepts per style.")
    p.add_argument("--device", type=str, default=None)
    ns = p.parse_args()

    cfg = {}
    if ns.config and os.path.exists(ns.config):
        with open(ns.config) as f:
            cfg = yaml.safe_load(f) or {}

    ns.concepts_json = resolve_path(
        ns.concepts_json or cfg.get("concepts_json") or os.path.join("assets", "style_concepts.json"))
    ns.out_json = resolve_path(
        ns.out_json or os.path.join("assets", "style_concepts_filtered.json"))
    ns.out_report = resolve_path(
        ns.out_report or os.path.join("results", "concept_discriminability.csv"))

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

    ns.device = ns.device or (cfg.get("runtime") or {}).get("device")
    return ns


if __name__ == "__main__":
    args = get_args()
    run(args)
