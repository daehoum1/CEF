"""Encode the UNFILTERED LLM concept dictionaries with each backbone's text
encoder, so the evidence path can be evaluated with the full dictionary (the
discriminability filter was designed for hard top-r hyperedges, which the
evidence path does not use)."""
import os, sys
import numpy as np
from cgpr_lab import DATASETS, BACKBONES, HYPER_ROOT, get_class_names
import cef_paths
cef_paths.add_bundled_paths()
from concept_hypergraph import flatten_concepts, load_style_concepts
from vlm_comparison import VLMEncoder


OUT = os.path.join(HYPER_ROOT, "analysis", "raw_concept_embs")
os.makedirs(OUT, exist_ok=True)
SPECS = {"ViT-B-32_openai": ("ViT-B-32", "openai"),
         "ViT-B-32_metaclip_fullcc": ("ViT-B-32", "metaclip_fullcc"),
         "EVA02-B-16_merged2b_s8b_b131k": ("EVA02-B-16", "merged2b_s8b_b131k"),
         "ViT-B-16-SigLIP_webli": ("ViT-B-16-SigLIP", "webli")}

for tag, (mn, pt) in SPECS.items():
    enc = None
    for dataset in ["wikiart", "mp100k"]:
        f = os.path.join(OUT, f"{dataset}_{tag}.npz")
        if os.path.exists(f):
            print("skip", f); continue
        cls = get_class_names(dataset)
        sc = load_style_concepts(DATASETS[dataset]["concepts_raw_json"])
        phrases, styles = flatten_concepts(sc, cls)
        if enc is None:
            enc = VLMEncoder(mn, pt, device=None)
        embs = enc.encode_texts(phrases)
        np.savez(f, phrases=np.array(phrases, dtype=object),
                 styles=np.array(styles, dtype=object), embs=embs)
        print(f"[saved] {f}  {len(phrases)} phrases")
    del enc
