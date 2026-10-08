"""Build and encode the descriptor-control dictionaries requested by the
descriptor-based zero-shot literature.

Three families, all holding the per-class phrase COUNT of the real LLM
dictionary fixed so that only the content of the phrases changes:

  rand_chars  random character strings (WaffleCLIP's random-token control).
  rand_words  random word sequences drawn from the real dictionary's own
              lexicon, with the real phrase-length distribution. Same words,
              destroyed composition; no external word list is needed, so the
              control is reproducible from this repository alone.
  dclip       the real phrases wrapped in the descriptor template
              "{class}, which has {phrase}" used by classification-by-
              description, so a faithful DCLIP/CuPL classifier can be scored.

A fourth control, shuffle_owner (real phrases, permuted owning style), needs no
encoding and is built at evaluation time from the full dictionary.

Outputs analysis/control_dicts/{dataset}_{tag}_{variant}.npz.
"""
from __future__ import annotations

import os
import string
import sys

import numpy as np

from cgpr_lab import BACKBONES, DATASETS, HYPER_ROOT, get_class_names

import cef_paths
cef_paths.add_bundled_paths()
from concept_hypergraph import flatten_concepts, load_style_concepts  # noqa: E402


OUT = os.path.join(HYPER_ROOT, "analysis", "control_dicts")
SPECS = {
    "ViT-B-32_openai": ("ViT-B-32", "openai"),
    "ViT-B-32_metaclip_fullcc": ("ViT-B-32", "metaclip_fullcc"),
    "EVA02-B-16_merged2b_s8b_b131k": ("EVA02-B-16", "merged2b_s8b_b131k"),
    "ViT-B-16-SigLIP_webli": ("ViT-B-16-SigLIP", "webli"),
}
RNG_SEED = 0
VARIANTS = ["rand_chars", "rand_words", "dclip"]


def real_dictionary(dataset: str):
    """The full (unfiltered) LLM dictionary, in flatten_concepts order."""
    cls = get_class_names(dataset)
    sc = load_style_concepts(DATASETS[dataset]["concepts_raw_json"])
    phrases, styles = flatten_concepts(sc, cls)
    return cls, phrases, styles


def build_variant(variant: str, cls, phrases, styles):
    """Return (phrases, styles) for one control. Per-class counts are preserved
    because the style vector is never changed."""
    rng = np.random.default_rng(RNG_SEED)
    lengths = [len(p.split()) for p in phrases]

    if variant == "rand_chars":
        alphabet = string.ascii_lowercase + string.digits
        out = []
        for n_words in lengths:
            # one random token per word of the phrase it replaces
            toks = ["".join(rng.choice(list(alphabet), size=int(rng.integers(3, 9))))
                    for _ in range(n_words)]
            out.append(" ".join(toks))
        return out, list(styles)

    if variant == "rand_words":
        lexicon = sorted({w.strip(string.punctuation).lower()
                          for p in phrases for w in p.split()
                          if w.strip(string.punctuation)})
        out = []
        for n_words in lengths:
            idx = rng.choice(len(lexicon), size=n_words, replace=False)
            out.append(" ".join(lexicon[i] for i in idx))
        return out, list(styles)

    if variant == "dclip":
        return [f"{s}, which has {p}" for p, s in zip(phrases, styles)], list(styles)

    raise ValueError(variant)


def main():
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    from vlm_comparison import VLMEncoder

    for tag, (mn, pt) in SPECS.items():
        if only and only != tag:
            continue
        enc = None
        for dataset in ["wikiart", "mp100k"]:
            cls, phrases, styles = real_dictionary(dataset)
            for variant in VARIANTS:
                f = os.path.join(OUT, f"{dataset}_{tag}_{variant}.npz")
                if os.path.exists(f):
                    print("skip", os.path.basename(f), flush=True)
                    continue
                vp, vs = build_variant(variant, cls, phrases, styles)
                assert len(vp) == len(phrases)
                if enc is None:
                    enc = VLMEncoder(mn, pt, device=None)
                embs = enc.encode_texts(vp)
                np.savez(f, phrases=np.array(vp, dtype=object),
                         styles=np.array(vs, dtype=object), embs=embs)
                print(f"[saved] {os.path.basename(f)} {len(vp)} phrases "
                      f"e.g. {vp[0]!r}", flush=True)
        del enc


if __name__ == "__main__":
    main()
