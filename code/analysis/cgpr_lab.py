"""
cgpr_lab.py — shared loading + diagnostics harness for CGPR follow-up work.

Loads exactly what run_concept_hypergraph.py loads (same cached embeddings,
same concept dictionaries, same pool-and-resplit-per-seed protocol), but as a
library so improvement ideas can be prototyped and compared without re-running
the full grid search.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import cef_paths

HERE = os.path.dirname(os.path.abspath(__file__))
HYPER_ROOT = str(cef_paths.project_root())
cef_paths.add_bundled_paths()


def _dictionary(name: str) -> str:
    """Concept dictionaries live in assets/ in a reproduction tree, dictionaries/ here."""
    for sub in ("assets", "dictionaries"):
        p = os.path.join(HYPER_ROOT, sub, name)
        if os.path.exists(p):
            return p
    return os.path.join(HYPER_ROOT, "dictionaries", name)

from concept_hypergraph import (  # noqa: E402
    concept_comembership,
    build_combined_propagation_matrix,
    build_incidence_matrix,
    build_propagation_matrix,
    chsp_propagate,
    flatten_concepts,
    load_style_concepts,
    mask_knn_with_concepts,
    uchsp_propagate,
)
from ugsp.dataset import maybe_filter_styles, prepare_dataframe  # noqa: E402
from ugsp.evaluate import accuracy, balanced_accuracy, macro_f1  # noqa: E402
from ugsp.graph import build_knn_graph  # noqa: E402
from ugsp.propagation import score_propagation  # noqa: E402

BACKBONES = [
    ("ViT-B-32_openai", "CLIP (OpenAI)"),
    ("ViT-B-32_metaclip_fullcc", "MetaCLIP"),
    ("EVA02-B-16_merged2b_s8b_b131k", "EVA02-CLIP"),
    ("ViT-B-16-SigLIP_webli", "SigLIP"),
]

DATASETS = {
    "wikiart": dict(
        csv_path=cef_paths.WIKIART_CSV,
        image_root=cef_paths.WIKIART_ROOT or "",
        cache_root=cef_paths.WIKIART_CACHE or "",
        sp_config_root=cef_paths.WIKIART_CACHE or "",
        results_dir=os.path.join(HYPER_ROOT, "results", "wikiart_v3"),
        concepts_json=_dictionary("style_concepts_v3_filtered.json"),
        concepts_raw_json=_dictionary("style_concepts_v3.json"),
        raw_csv=False, min_images=0,
    ),
    "mp100k": dict(
        csv_path=cef_paths.MP100K_CSV,
        image_root=cef_paths.MP100K_IMAGES,
        cache_root=cef_paths.MP100K_CACHE or "",
        sp_config_root=cef_paths.MP100K_CACHE or "",
        results_dir=os.path.join(HYPER_ROOT, "results", "mp100k_v3"),
        concepts_json=_dictionary("style_concepts_mp100k_v3_filtered.json"),
        concepts_raw_json=_dictionary("style_concepts_mp100k_v3.json"),
        raw_csv=True, min_images=1000,
    ),
}


def get_class_names(dataset: str) -> list[str]:
    """Reproduce run_concept_hypergraph.run()'s class_names derivation exactly."""
    ds = DATASETS[dataset]
    if ds["raw_csv"]:
        df = pd.read_csv(cef_paths.require("CEF_WIKIART_ROOT / CEF_MP100K_ROOT", ds["csv_path"]))
        if ds["min_images"] > 0:
            counts = df["style"].value_counts()
            df = df[df["style"].isin(counts[counts >= ds["min_images"]].index)].reset_index(drop=True)
    else:
        df = prepare_dataframe(cef_paths.require("CEF_WIKIART_ROOT / CEF_MP100K_ROOT", ds["csv_path"]))
        df = maybe_filter_styles(df, min_images=ds["min_images"], top_n=0, selected=None)
    all_styles = set(df["style"].unique())
    for split in ("train", "val", "test"):
        all_styles &= set(df[df["new_subset"] == split]["style"].unique())
    return sorted(all_styles)


class Pool:
    """The val+test pool for one (dataset, backbone), plus its concept embeddings."""

    def __init__(self, dataset: str, backbone_tag: str):
        ds = DATASETS[dataset]
        self.dataset = dataset
        self.tag = backbone_tag
        self.class_names = get_class_names(dataset)

        embs, Y = [], []
        for split in ("val", "test"):
            d = np.load(os.path.join(cef_paths.require("CEF_WIKIART_CACHE / CEF_MP100K_CACHE", ds["cache_root"]), backbone_tag, f"{split}_embeddings.npz"))
            embs.append(d["image_embs"])
            Y.append(d["Y"])
        self.embs = np.concatenate(embs, 0)
        self.Y = np.concatenate(Y, 0)

        # Labels come from the same dataframe ordering vlm_comparison.get_split uses.
        self.labels = _pool_labels(dataset, self.class_names)
        assert len(self.labels) == len(self.embs), (len(self.labels), len(self.embs))

        self.style_concepts = load_style_concepts(ds["concepts_json"])
        self.phrases, self.phrase_styles = flatten_concepts(self.style_concepts, self.class_names)
        ce = np.load(os.path.join(ds["results_dir"], backbone_tag, "concept_text_embs.npz"),
                     allow_pickle=True)
        assert list(ce["phrases"]) == self.phrases, "cached concept embeddings are stale"
        self.concept_embs = ce["embs"]
        self.phrase_style_idx = np.array([self.class_names.index(s) for s in self.phrase_styles])

        with open(os.path.join(ds["sp_config_root"], backbone_tag, "best_config.json")) as f:
            self.sp_cfg = json.load(f)
        self.sp_cfg.setdefault("k_graph", 10)

        with open(os.path.join(ds["results_dir"], backbone_tag, "best_config.json")) as f:
            self.cgpr_cfg = json.load(f)

    def split(self, seed: int, val_frac: float = 0.2):
        """Same stratified two-way split run_concept_hypergraph.py uses per seed."""
        rng = np.random.default_rng(seed)
        val_idx, test_idx = [], []
        for c in np.unique(self.labels):
            c_idx = rng.permutation(np.where(self.labels == c)[0])
            n_val = max(1, int(round(len(c_idx) * val_frac)))
            val_idx.extend(c_idx[:n_val].tolist())
            test_idx.extend(c_idx[n_val:].tolist())
        return np.array(val_idx), np.array(test_idx)


def pool_paths(dataset: str, class_names: list[str]) -> list[str]:
    """Image paths in the same order as Pool.embs / Pool.labels (mirrors
    vlm_comparison.get_split, val split then test split)."""
    ds = DATASETS[dataset]
    df = _pool_dataframe(dataset)
    out = []
    for split in ("val", "test"):
        sdf = df[df["new_subset"] == split].copy().reset_index(drop=True)
        sdf = sdf[sdf["style"].isin(class_names)].reset_index(drop=True)
        out.extend(os.path.join(ds["image_root"], r["filename"]) for _, r in sdf.iterrows())
    return out


def _pool_dataframe(dataset: str):
    ds = DATASETS[dataset]
    if ds["raw_csv"]:
        df = pd.read_csv(cef_paths.require("CEF_WIKIART_ROOT / CEF_MP100K_ROOT", ds["csv_path"]))
        if ds["min_images"] > 0:
            counts = df["style"].value_counts()
            df = df[df["style"].isin(counts[counts >= ds["min_images"]].index)].reset_index(drop=True)
    else:
        df = prepare_dataframe(cef_paths.require("CEF_WIKIART_ROOT / CEF_MP100K_ROOT", ds["csv_path"]))
        df = maybe_filter_styles(df, min_images=ds["min_images"], top_n=0, selected=None)
    return df


def _pool_labels(dataset: str, class_names: list[str]) -> np.ndarray:
    df = _pool_dataframe(dataset)
    out = []
    for split in ("val", "test"):
        sdf = df[(df["new_subset"] == split) & (df["style"].isin(class_names))]
        out.append(np.array([class_names.index(s) for s in sdf["style"]]))
    return np.concatenate(out)


def metrics(scores: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    preds = np.argmax(scores, 1)
    return {
        "top1": accuracy(preds, labels),
        "bacc": balanced_accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels),
    }
