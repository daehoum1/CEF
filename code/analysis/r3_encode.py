"""Review round 3: encode every generated dictionary with each backbone's text encoder.

Reads assets/r3_dicts/{dataset}_{family}_g{k}.json, flattens it in class order with
the same flatten_concepts the v3 embeddings use, and writes
analysis/r3_embs/{dataset}_{family}_g{k}_{tag}.npz (phrases, styles, embs).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from cgpr_lab import HYPER_ROOT, get_class_names

import cef_paths
cef_paths.add_bundled_paths()
from concept_hypergraph import flatten_concepts, load_style_concepts  # noqa: E402


SPECS = {
    "ViT-B-32_openai": ("ViT-B-32", "openai"),
    "ViT-B-32_metaclip_fullcc": ("ViT-B-32", "metaclip_fullcc"),
    "EVA02-B-16_merged2b_s8b_b131k": ("EVA02-B-16", "merged2b_s8b_b131k"),
    "ViT-B-16-SigLIP_webli": ("ViT-B-16-SigLIP", "webli"),
}
SRC = os.path.join(HYPER_ROOT, "assets", "r3_dicts")
OUT = os.path.join(HYPER_ROOT, "analysis", "r3_embs")


def main():
    os.makedirs(OUT, exist_ok=True)
    from vlm_comparison import VLMEncoder

    files = sorted(glob.glob(os.path.join(SRC, "*_g[0-9].json")))
    for tag, (mn, pt) in SPECS.items():
        enc = None
        for f in files:
            stem = os.path.basename(f)[:-5]
            dataset = stem.split("_")[0]
            out = os.path.join(OUT, f"{stem}_{tag}.npz")
            if os.path.exists(out):
                continue
            cls = get_class_names(dataset)
            sc = load_style_concepts(f)
            assert all(sc.get(c) for c in cls), f"{stem}: a class has no phrases"
            phrases, styles = flatten_concepts(sc, cls)
            if enc is None:
                enc = VLMEncoder(mn, pt, device=None)
            embs = enc.encode_texts(phrases)
            np.savez(out, phrases=np.array(phrases, dtype=object),
                     styles=np.array(styles, dtype=object), embs=embs)
            print(f"[saved] {os.path.basename(out)} {len(phrases)}", flush=True)
        del enc


if __name__ == "__main__":
    main()
