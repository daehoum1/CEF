"""Domain-transfer probe on Oxford-IIIT Pet.

The submission evaluates two painting-style benchmarks. This arm asks whether
the mechanism -- dense concept evidence fused into the cross-modal weights of a
label-propagation graph -- is specific to style, by repeating the identical
protocol on a fine-grained object domain where the descriptor literature is
usually evaluated.

Everything is held to the style experiments' protocol: the same four backbones,
a five-template prompt ensemble, the pretrained logit scale, a 20:80
class-stratified val/test resplit of the whole pool repeated over five seeds,
per-seed validation selection, and the same search grids. The concept
dictionary is produced by the same procedure with the prompt's axes adapted to
the domain (assets/pets_concepts.json records the exact template used).

  --stage encode   cache image embeddings, class-text embeddings and Y
  --stage eval     run every method and write pets_per_seed.csv
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import string
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax

import cef_paths

from cgpr_lab import HYPER_ROOT, metrics
from zlap_impl import precompute_knn, zlap_transductive

cef_paths.add_bundled_paths()

PETS_ROOT = os.path.join(HYPER_ROOT, "data_pets")
CACHE = os.path.join(HYPER_ROOT, "analysis", "pets_cache")
CONCEPTS = os.path.join(HYPER_ROOT, "assets", "pets_concepts.json")
OUT = "pets_per_seed.csv"

SPECS = {
    "ViT-B-32_openai": ("ViT-B-32", "openai", "CLIP (OpenAI)"),
    "ViT-B-32_metaclip_fullcc": ("ViT-B-32", "metaclip_fullcc", "MetaCLIP"),
    "EVA02-B-16_merged2b_s8b_b131k": ("EVA02-B-16", "merged2b_s8b_b131k", "EVA02-CLIP"),
    "ViT-B-16-SigLIP_webli": ("ViT-B-16-SigLIP", "webli", "SigLIP"),
}
TEMPLATES = [
    "a photo of a {}, a type of pet",
    "a photo of a {}",
    "a close-up photo of a {}",
    "a photo of a {} pet",
    "a cropped photo of a {}",
]
SEEDS = [42, 43, 44, 45, 46]
KS, GAMMAS, ALPHAS = [5, 10, 20, 40, 80], [1.0, 3.0, 5.0], [0.3, 0.5, 0.7, 0.9, 0.95]
TAUS, WS, DEB = [0.005, 0.01, 0.02, 0.05], [0.0, 0.2, 0.4, 0.6, 0.8], [True, False]
BASES = ["cos", "sm"]
GRAPH = list(itertools.product(KS, GAMMAS, ALPHAS))
EVID = list(itertools.product(TAUS, WS, DEB, BASES))
INIT_EVID = (0.01, 0.4, True, "sm")
ZL_E = (0.01, 0.0, True, "cos")
CTRL = ["rand_chars", "rand_words", "shuffle_owner"]


def inventory():
    """Image paths and integer labels, class names sorted for determinism."""
    paths = sorted(glob.glob(os.path.join(PETS_ROOT, "images", "*.jpg")))
    names = ["_".join(os.path.basename(p).split("_")[:-1]) for p in paths]
    classes = sorted(set(names))
    labels = np.array([classes.index(n) for n in names])
    return paths, labels, classes


def display(c):
    return c.replace("_", " ")


def load_concepts(classes):
    d = json.load(open(CONCEPTS))
    phrases, owners = [], []
    for ci, c in enumerate(classes):
        assert c in d, f"no concepts for {c}"
        for p in d[c]:
            phrases.append(p)
            owners.append(ci)
    return phrases, np.array(owners)


def control_phrases(variant, phrases):
    rng = np.random.default_rng(0)
    lengths = [len(p.split()) for p in phrases]
    if variant == "rand_chars":
        alpha = string.ascii_lowercase + string.digits
        return [" ".join("".join(rng.choice(list(alpha), size=int(rng.integers(3, 9))))
                         for _ in range(n)) for n in lengths]
    if variant == "rand_words":
        lex = sorted({w.strip(string.punctuation).lower()
                      for p in phrases for w in p.split() if w.strip(string.punctuation)})
        return [" ".join(lex[i] for i in rng.choice(len(lex), size=n, replace=False))
                for n in lengths]
    raise ValueError(variant)


def encode():
    os.makedirs(CACHE, exist_ok=True)
    from vlm_comparison import VLMEncoder

    paths, labels, classes = inventory()
    phrases, owners = load_concepts(classes)
    print(f"[pets] {len(paths)} images, {len(classes)} classes, {len(phrases)} concepts",
          flush=True)

    for tag, (mn, pt, _) in SPECS.items():
        f = os.path.join(CACHE, f"{tag}.npz")
        if os.path.exists(f):
            print("skip", tag, flush=True)
            continue
        enc = VLMEncoder(mn, pt, device=None)
        img = enc.encode_images_from_paths(paths, batch_size=256)
        txt = enc.get_text_embeddings([display(c) for c in classes],
                                      templates=TEMPLATES, display_names=None)
        scale = 1.0
        if hasattr(enc.model, "logit_scale"):
            scale = enc.model.logit_scale.exp().item()
        Y = softmax((img @ txt.T) * scale, axis=1)
        cemb = {"full": enc.encode_texts(phrases),
                "dclip": enc.encode_texts(
                    [f"{display(classes[o])}, which has {p}" for p, o in zip(phrases, owners)])}
        for v in ["rand_chars", "rand_words"]:
            cemb[v] = enc.encode_texts(control_phrases(v, phrases))
        np.savez(f, image_embs=img, text_embs=txt, Y=Y, labels=labels,
                 classes=np.array(classes, dtype=object), owners=owners,
                 **{f"c_{k}": v for k, v in cemb.items()})
        print(f"[saved] {f}", flush=True)
        del enc


def split(labels, seed, val_frac=0.2):
    rng = np.random.default_rng(seed)
    vi, ti = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n = max(1, int(round(len(idx) * val_frac)))
        vi.extend(idx[:n].tolist())
        ti.extend(idx[n:].tolist())
    return np.array(vi), np.array(ti)


def evaluate():
    rows = []
    for tag, (_, _, label) in SPECS.items():
        d = np.load(os.path.join(CACHE, f"{tag}.npz"), allow_pickle=True)
        X, clf, Yv, labels = d["image_embs"], d["text_embs"], d["Y"], d["labels"]
        classes = list(d["classes"])
        owners = d["owners"]
        C = len(classes)
        DICTS = {k: (d[f"c_{k}"], owners) for k in ["full", "dclip", "rand_chars", "rand_words"]}
        DICTS["shuffle_owner"] = (d["c_full"],
                                  owners[np.random.default_rng(0).permutation(len(owners))])
        COLS = {n: [np.where(o == c)[0] for c in range(C)] for n, (_, o) in DICTS.items()}
        print(f"[run] pets {label}", flush=True)

        for seed in SEEDS:
            vi, ti = split(labels, seed)
            mu = {n: (X[vi] @ DICTS[n][0].T).mean(0, keepdims=True) for n in DICTS}
            V = (X[vi], Yv[vi], labels[vi])
            T = (X[ti], Yv[ti], labels[ti])
            M = lambda s, l: metrics(s, l, classes)
            cache = {id(V): precompute_knn(V[0], clf, max(KS)),
                     id(T): precompute_knn(T[0], clf, max(KS))}

            def evidence(embs, tau, db, dn):
                S = embs @ DICTS[dn][0].T
                if db:
                    S = S - mu[dn]
                S = S / tau
                S -= S.max(1, keepdims=True)
                A = np.exp(S)
                A /= A.sum(1, keepdims=True)
                Yc = np.zeros((A.shape[0], C))
                for c, cc in enumerate(COLS[dn]):
                    if len(cc):
                        Yc[:, c] = A[:, cc].sum(1)
                return Yc / (Yc.sum(1, keepdims=True) + 1e-12)

            cos_cache = {}

            def cosine_aff(part):
                if id(part) not in cos_cache:
                    cos_cache[id(part)] = part[0] @ clf.T
                return cos_cache[id(part)]

            def score(part, g, e, dn="full"):
                k, gam, al = g
                tau, w, db, base = e
                B = cosine_aff(part) if base == "cos" else part[1]
                if w == 0:
                    Yf = B
                else:
                    Yc = evidence(part[0], tau, db, dn)
                    if base == "cos":
                        Yc = Yc * B.max(axis=1, keepdims=True)
                    Yf = (1 - w) * B + w * Yc
                return zlap_transductive(part[0], clf, k, gam, al,
                                         cross_affinity=Yf, knn_cache=cache[id(part)])

            zl = {}

            def zlap_solution(sel):
                if sel not in zl:
                    zl[sel] = (max(GRAPH, key=lambda c: M(score(V, c, ZL_E), V[2])[sel]), ZL_E)
                return zl[sel]

            def staged(sel, dn="full", pool=EVID, include_zlap=True):
                g = max(GRAPH, key=lambda c: M(score(V, c, INIT_EVID, dn), V[2])[sel])
                e = max(pool, key=lambda c: M(score(V, g, c, dn), V[2])[sel])
                g = max(GRAPH, key=lambda c: M(score(V, c, e, dn), V[2])[sel])
                cands = [(g, e)]
                if include_zlap and any(c[1] == 0 for c in pool):
                    cands.append(zlap_solution(sel))
                return max(cands, key=lambda ge: M(score(V, ge[0], ge[1], dn), V[2])[sel])

            def emit(method, m, cfg=""):
                rows.append(dict(dataset="pets", backbone=label, seed=seed,
                                 method=method, bacc=m["bacc"], top1=m["top1"], cfg=cfg))

            emit("Vanilla", M(T[1], T[2]))
            gz, _ = zlap_solution("bacc")
            emit("ZLaP", M(score(T, gz, ZL_E), T[2]), str(gz))
            g, e = staged("bacc")
            emit("CEF", M(score(T, g, e), T[2]), str((g, e)))
            gw, ew = staged("bacc", pool=[c for c in EVID if c[1] == 0])
            emit("CEF w=0", M(score(T, gw, ew), T[2]), str((gw, ew)))
            for dn in CTRL:
                gc, ec = staged("bacc", dn=dn)
                emit(f"CEF [{dn}]", M(score(T, gc, ec, dn), T[2]), str((gc, ec)))

            demb, downer = DICTS["dclip"]
            S_v, S_t = V[0] @ demb.T, T[0] @ demb.T
            dcols = [np.where(downer == c)[0] for c in range(C)]
            mc = lambda S: np.stack([S[:, cc].mean(1) for cc in dcols], 1)
            proto = np.stack([demb[downer == c].mean(0) /
                              (np.linalg.norm(demb[downer == c].mean(0)) + 1e-12)
                              for c in range(C)])
            for name, sv, st in (("DCLIP", mc(S_v), mc(S_t)),
                                 ("CuPL", V[0] @ proto.T, T[0] @ proto.T)):
                emit(name, M(st, T[2]))
                av, at = np.clip(sv, 0, None), np.clip(st, 0, None)
                gd = max(GRAPH, key=lambda c: M(zlap_transductive(
                    V[0], clf, *c, cross_affinity=av, knn_cache=cache[id(V)]), V[2])["bacc"])
                emit(f"{name} + ZLaP", M(zlap_transductive(
                    T[0], clf, *gd, cross_affinity=at, knn_cache=cache[id(T)]), T[2]), str(gd))
            print(f"  seed {seed} done", flush=True)

        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["encode", "eval"], required=True)
    a = ap.parse_args()
    encode() if a.stage == "encode" else evaluate()
