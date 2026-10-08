"""Generate the manuscript's revision tables and prose macros from per-seed CSVs.

Every number the revision adds or changes is produced here, so it traces to a
run rather than to prose. Writes into paper3/tables and emits
paper3/review_numbers.tex with the macros the prose uses.

Protocol. The main tables are computed on the ARTIST-DISJOINT validation/test
splits (results/supplementary/artist_disjoint/). The image-level resplit used by
the CGPR analysis is kept for the diagnosis section and for the secondary
comparison table (tables/tab_main_image.tex, preserved verbatim), and its
numbers are exposed under the `Img` macro prefix.
"""
from __future__ import annotations

import ast
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
HYPER = os.environ.get("CEF_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
OUT = os.path.join(HYPER, "paper3", "tables")
MAC = os.path.join(HYPER, "paper3", "review_numbers.tex")
ADIR = os.path.join(HYPER, "results", "supplementary", "artist_disjoint")
BB = ["CLIP (OpenAI)", "MetaCLIP", "EVA02-CLIP", "SigLIP"]
DS = [("wikiart", "WikiArt"), ("mp100k", "MultitaskPainting100k")]
f4 = lambda v: f"{v:.4f}"
macros: dict[str, str] = {}


def w(name, s):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(s)
    print("wrote", name)


def M(name, val):
    macros[name] = val


def pts(x):
    v = x * 100
    if abs(v) < 0.05:   # never print a signed zero
        v = abs(v)
    return f"{v:.1f}"


def pts2(x):
    v = x * 100
    if abs(v) < 0.005:
        v = abs(v)
    return f"{v:+.2f}" if v > 0 else f"{v:.2f}"


# ── loading ────────────────────────────────────────────────────────────────────
def load_artist():
    """Per-seed rows (method, bacc, top1, cfg) for every arm on the artist splits."""
    r = pd.read_csv(os.path.join(ADIR, "per_seed_results.csv"))
    key = ["dataset", "backbone", "seed", "experiment"]
    piv = r.pivot_table(index=key, columns="metric", values="score").reset_index()
    cfg = r[r.metric == "bAcc"][key + ["selected_config"]]
    piv = piv.merge(cfg, on=key).rename(columns={"experiment": "method", "bAcc": "bacc",
                                                 "Top-1": "top1", "selected_config": "cfg"})
    piv["method"] = piv.method.replace({"CEF-FusionOnly": "FusionOnly"})
    cols = ["dataset", "backbone", "seed", "method", "bacc", "top1", "cfg"]
    x = pd.read_csv(os.path.join(ADIR, "per_seed_extra.csv")).rename(
        columns={"experiment": "method", "selected_config": "cfg"})
    t = pd.read_csv(os.path.join(ADIR, "per_seed_transfer.csv"))
    out = pd.concat([piv[cols], x[cols], t[cols]], ignore_index=True)
    assert len(set(zip(out.dataset, out.backbone))) == 8
    return out


def load_image():
    """Per-seed rows on the image-level resplit (CGPR's protocol)."""
    d = pd.concat([pd.read_csv(f) for f in
                   ["review_controls_per_seed.csv", "review_transfer_per_seed.csv"]], ignore_index=True)
    fe = pd.read_csv("final_per_seed.csv")
    cg = fe[["dataset", "backbone", "seed", "CGPR|bacc", "CGPR|top1"]].rename(
        columns={"CGPR|bacc": "bacc", "CGPR|top1": "top1"}).assign(method="CGPR")
    return pd.concat([d, cg], ignore_index=True)


def means(ps, col="bacc"):
    return ps.groupby(["dataset", "backbone", "method"])[col].mean().rename("bacc").reset_index()


def cell(mean, m, ds, bb, best=None):
    v = mean[(mean.dataset == ds) & (mean.backbone == bb) & (mean.method == m)]
    if not len(v):
        return "--"
    x = float(v.bacc.iloc[0])
    s = f4(x)
    return rf"\textbf{{{s}}}" if best is not None and abs(x - best) < 5e-5 else s


def table(mean, methods, fname, header, datasets=DS, bold_best=True):
    L = [r"\begin{tabular}{ll" + "r" * len(methods) + "}", r"\toprule",
         "Dataset & Backbone & " + " & ".join(header) + r" \\", r"\midrule"]
    for di, (ds, dsl) in enumerate(datasets):
        for i, bb in enumerate(BB):
            vals = [mean[(mean.dataset == ds) & (mean.backbone == bb) &
                         (mean.method == m)].bacc for m in methods]
            vals = [float(v.iloc[0]) if len(v) else np.nan for v in vals]
            best = np.nanmax(vals) if bold_best else None
            L.append(f"{dsl if i == 0 else ''} & {bb} & " +
                     " & ".join(cell(mean, m, ds, bb, best) for m in methods) + r" \\")
        if di < len(datasets) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    w(fname, "\n".join(L))


def pair_mean_diffs(per_seed, a, b, col="bacc"):
    """Per-pair mean paired difference over the shared seeds (Series indexed by dataset, backbone)."""
    pa = per_seed[per_seed.method == a].set_index(["dataset", "backbone", "seed"])[col]
    pb = per_seed[per_seed.method == b].set_index(["dataset", "backbone", "seed"])[col]
    idx = pa.index.intersection(pb.index)
    return (pa[idx] - pb[idx]).rename("d").reset_index().groupby(["dataset", "backbone"]).d.mean()


def seed_level(per_seed, a, b, col="bacc"):
    """(runs with a > b, runs) over every (dataset, backbone, seed)."""
    pa = per_seed[per_seed.method == a].set_index(["dataset", "backbone", "seed"])[col]
    pb = per_seed[per_seed.method == b].set_index(["dataset", "backbone", "seed"])[col]
    idx = pa.index.intersection(pb.index)
    d = pa[idx] - pb[idx]
    return int((d > 0).sum()), int(len(d))


def across(per_seed, a, b, col="bacc"):
    """Per-pair mean differences and the across-pair tests."""
    d = pair_mean_diffs(per_seed, a, b, col)
    v = d.values
    wil = (1.0 if np.all(v == 0) else stats.wilcoxon(v).pvalue) if len(v) > 1 else np.nan
    n, k = len(v), int((v > 0).sum())
    sgn = min(1.0, 2 * stats.binom.sf(max(k, n - k) - 1, n, 0.5))
    return v, wil, sgn, k, n


def emit_cmp(ps, a, b, key, col="bacc"):
    v, wil, sgn, k, n = across(ps, a, b, col)
    M(key + "Mean", pts(v.mean()))
    M(key + "Min", pts(v.min()))
    M(key + "Max", pts(v.max()))
    M(key + "Pos", str(k))
    M(key + "N", str(n))
    M(key + "Wil", f"{wil:.4f}")
    M(key + "Sign", f"{sgn:.4f}")
    sp, sn = seed_level(ps, a, b, col)
    M(key + "SeedPos", str(sp))
    M(key + "SeedN", str(sn))
    return v


def main_table(ps, fname):
    """Table 3 layout: Top-1 and bAcc +- std (over seeds) per backbone block."""
    arms = [("Vanilla VLM", "Vanilla"), ("CGPR", "CGPR"), ("ZLaP", "ZLaP"),
            (r"\textbf{CEF (ours)}", "CEF")]
    g = ps.groupby(["dataset", "backbone", "method"])
    mt, mb, sb = g.top1.mean(), g.bacc.mean(), g.bacc.std(ddof=1)
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Method & \multicolumn{2}{c}{WikiArt} & \multicolumn{2}{c}{MultitaskPainting100k} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & Top-1 & bAcc & Top-1 & bAcc \\"]
    for bb in BB:
        L += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\textit{{{bb}}}}} \\"]
        best = {(ds, c): max(s[(ds, bb, m)] for _, m in arms)
                for ds, _ in DS for c, s in (("t", mt), ("b", mb))}
        for lab, m in arms:
            cells = []
            for ds, _ in DS:
                t, b, sd = mt[(ds, bb, m)], mb[(ds, bb, m)], sb[(ds, bb, m)]
                ts = rf"\textbf{{{f4(t)}}}" if abs(t - best[(ds, 't')]) < 5e-5 else f4(t)
                bs = rf"\textbf{{{f4(b)}}}" if abs(b - best[(ds, 'b')]) < 5e-5 else f4(b)
                cells += [ts, rf"{bs}\,$\pm$\,{f4(sd)}"]
            L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w(fname, "\n".join(L))
    stds = [sb[(ds, bb, m)] for ds, _ in DS for bb in BB for _, m in arms]
    M("MainStdMin", f"{100 * min(stds):.2f}")
    M("MainStdMax", f"{100 * max(stds):.2f}")


def main():
    per_seed = load_artist()
    img = load_image()
    mean = means(per_seed)

    # ── Table 3 on artist-disjoint splits ──────────────────────────────────
    main_table(per_seed, "tab_main.tex")
    emit_cmp(per_seed, "ZLaP", "CGPR", "MainZlapCgpr")
    emit_cmp(per_seed, "CEF", "CGPR", "MainCefCgpr")
    emit_cmp(per_seed, "CEF", "ZLaP", "MainCefZlap")
    emit_cmp(per_seed, "CEF", "CGPR", "MainCefCgprTop", col="top1")
    emit_cmp(per_seed, "CEF", "ZLaP", "MainCefZlapTop", col="top1")
    emit_cmp(per_seed, "CEF", "Vanilla", "MainCefVan")
    emit_cmp(per_seed, "CEF w/ CGPR propagation", "CGPR", "MainHostCgpr")
    emit_cmp(per_seed, "CEF", "CEF w/ CGPR propagation", "MainCefHost")
    emit_cmp(per_seed, "ZLaP", "ZLaP (default cfg)", "MainZlapTune")
    for ds, key in (("wikiart", "Wiki"), ("mp100k", "Mp")):
        emit_cmp(per_seed[per_seed.dataset == ds], "CEF", "ZLaP", "MainCefZlap" + key)

    # ── descriptor-literature baselines ────────────────────────────────────
    meths = ["Vanilla", "DCLIP", "CuPL", "ZLaP", "DCLIP + ZLaP", "CuPL + ZLaP", "CEF"]
    head = ["Vanilla", "DCLIP", "CuPL", "ZLaP", r"DCLIP\,+\,LP", r"CuPL\,+\,LP",
            r"\textbf{CEF}"]
    table(mean, meths, "tab_descriptor.tex", head)
    for a, b, key in (("CEF", "DCLIP + ZLaP", "CefOverDclipLp"),
                      ("CEF", "CuPL + ZLaP", "CefOverCuplLp"),
                      ("DCLIP + ZLaP", "ZLaP", "DclipLpOverZlap"),
                      ("DCLIP", "Vanilla", "DclipOverVanilla")):
        emit_cmp(per_seed, a, b, key)

    # ── dictionary controls ────────────────────────────────────────────────
    meths = ["CEF", "CEF w=0", "CEF [rand_chars]", "CEF [rand_words]", "CEF [shuffle_owner]"]
    head = [r"\textbf{CEF}", r"$w{=}0$", "Random chars", "Random words", "Permuted owners"]
    table(mean, meths, "tab_controls.tex", head)
    for m, key in (("CEF [rand_chars]", "RandChars"), ("CEF [rand_words]", "RandWords"),
                   ("CEF [shuffle_owner]", "ShufOwner")):
        v, wil, sgn, k, n = across(per_seed, "CEF", m)
        M("CefOver" + key + "Mean", pts(v.mean()))
        M("CefOver" + key + "Min", pts(v.min()))
        M("CefOver" + key + "Max", pts(v.max()))
        M("CefOver" + key + "Pos", str(k))
        M("CefOver" + key + "Wil", f"{wil:.4f}")
        v2, wil2, sgn2, k2, n2 = across(per_seed, m, "CEF w=0")
        M(key + "OverZeroMean", pts(v2.mean()))
        M(key + "OverZeroMax", pts(v2.max()))
        M(key + "OverZeroPos", str(k2))
        M(key + "OverZeroWil", f"{wil2:.4f}")
    v, wil, sgn, k, n = across(per_seed, "CEF", "CEF w=0")
    sp, sn = seed_level(per_seed, "CEF", "CEF w=0")
    for nm, val in (("Mean", pts(v.mean())), ("Min", pts(v.min())), ("Max", pts(v.max())),
                    ("Pos", str(k)), ("Wil", f"{wil:.4f}"), ("Sign", f"{sgn:.4f}"),
                    ("SeedPos", str(sp)), ("SeedN", str(sn))):
        M("CefOverZero" + nm, val)

    # ── fixed-configuration transfer ───────────────────────────────────────
    meths = ["ZLaP", "CEF [global]", "CEF [loo]", "CEF"]
    head = ["ZLaP (tuned)", "CEF (one global cfg)", "CEF (leave-pair-out cfg)",
            r"\textbf{CEF (tuned)}"]
    table(mean, meths, "tab_transfer.tex", head)
    for a, key in (("CEF [loo]", "Loo"), ("CEF [global]", "Glob")):
        v, wil, sgn, k, n = across(per_seed, a, "ZLaP")
        M(key + "OverZlapMean", pts(v.mean()))
        M(key + "OverZlapMin", pts(v.min()))
        M(key + "OverZlapMax", pts(v.max()))
        M(key + "OverZlapPos", str(k))
        M(key + "OverZlapAbsMin", pts(abs(v.min())))
        M(key + "OverZlapWil", f"{wil:.4f}")
        v, *_ = across(per_seed, "CEF", a)
        M(key + "CostMean", pts(v.mean()))
        M(key + "CostMax", pts(v.max()))
    lg = mean[mean.method.isin(["CEF [loo]", "CEF [global]"])].pivot_table(
        index=["dataset", "backbone"], columns="method", values="bacc")
    M("TransferSameN", str(int(((lg["CEF [loo]"] - lg["CEF [global]"]).abs() < 1e-12).sum())))

    # ── ablations (artist-disjoint) ────────────────────────────────────────
    rows = [(r"\textbf{CEF (full)}", "CEF"), (r"\quad w/o concepts ($w{=}0$)", "CEF w=0"),
            (r"\quad w/ filtered dictionary", "CEF w/ filtered dict"),
            (r"\quad w/o concept debiasing", "CEF w/o debias"),
            (r"\quad w/ CGPR propagation", "CEF w/ CGPR propagation"),
            (r"\quad w/o label propagation (fusion only)", "FusionOnly")]
    mb = per_seed.groupby(["dataset", "backbone", "method"]).bacc.mean()
    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Variant & \multicolumn{4}{c}{WikiArt} & \multicolumn{4}{c}{MultitaskPainting100k} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r" & CLIP & MetaCLIP & EVA02-CLIP & SigLIP & CLIP & MetaCLIP & EVA02-CLIP & SigLIP \\",
         r"\midrule"]
    colbest = {(ds, bb): max(mb[(ds, bb, m)] for _, m in rows) for ds, _ in DS for bb in BB}
    for lab, m in rows:
        cells = []
        for ds, _ in DS:
            for bb in BB:
                x = mb[(ds, bb, m)]
                cells.append(rf"\textbf{{{f4(x)}}}" if abs(x - colbest[(ds, bb)]) < 5e-5 else f4(x))
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_ablation.tex", "\n".join(L))
    emit_cmp(per_seed, "CEF w=0", "ZLaP", "AblZeroZlap")
    v = emit_cmp(per_seed, "CEF", "CEF w/ filtered dict", "AblFilt")
    M("AblFiltBetter", str(int((v < 0).sum())))
    for ds, key in (("wikiart", "Wiki"), ("mp100k", "Mp")):
        vv, *_ = across(per_seed[per_seed.dataset == ds], "CEF", "CEF w/ filtered dict")
        M("AblFiltMean" + key, pts(vv.mean()))
    v, *_ = across(per_seed, "CEF", "CEF w/o debias")
    M("AblDebAbsMax", pts(np.abs(v).max()))
    M("AblDebBetter", str(int((v < 0).sum())))

    # ── fusion without label propagation ───────────────────────────────────
    for a, b, key in (("FusionOnly", "Vanilla", "FoVan"), ("FusionOnly", "ZLaP", "FoZlap"),
                      ("FusionOnly", "DCLIP", "FoDclip"), ("FusionOnly", "CuPL", "FoCupl"),
                      ("CEF", "FusionOnly", "CefFo")):
        emit_cmp(per_seed, a, b, key)
    fv, *_ = across(per_seed, "FusionOnly", "Vanilla")
    cv, *_ = across(per_seed, "CEF", "Vanilla")
    M("FoSharePct", f"{100 * fv.mean() / cv.mean():.0f}")

    # ── diagnosis: concept mask toggled on the artist-disjoint splits ──────
    v, *_ = across(per_seed, "CGPR (mask on)", "CGPR (mask off)")
    M("ArtMaskMin", pts2(v.min()))
    M("ArtMaskMax", pts2(v.max()))
    M("ArtMaskAbsMax", f"{100 * np.abs(v).max():.2f}")

    # ── domain-transfer probe (unchanged data; style side is now artist) ───
    pets = pd.read_csv("pets_per_seed.csv")
    pmean = pets.groupby(["dataset", "backbone", "method"]).bacc.mean().reset_index()
    meths = ["Vanilla", "DCLIP", "ZLaP", "DCLIP + ZLaP", "CEF w=0", "CEF [rand_words]", "CEF"]
    head = ["Vanilla", "DCLIP", "ZLaP", r"DCLIP\,+\,LP", r"$w{=}0$", "Random words", r"\textbf{CEF}"]
    table(pmean, meths, "tab_pets.tex", head, datasets=[("pets", "Oxford-IIIT Pet")])
    for a, b, key in (("CEF", "ZLaP", "PetsCefZlap"), ("CEF", "CEF w=0", "PetsCefZero"),
                      ("CEF", "CEF [rand_words]", "PetsCefRandW"),
                      ("CEF", "CEF [rand_chars]", "PetsCefRandC"),
                      ("CEF", "CEF [shuffle_owner]", "PetsCefShuf"),
                      ("CEF w=0", "ZLaP", "PetsZeroZlap"), ("CEF", "DCLIP + ZLaP", "PetsCefDclip")):
        emit_cmp(pets, a, b, key)
    v, *_ = across(per_seed, "CEF w=0", "ZLaP")
    M("StyleZeroZlapMean", pts(v.mean()))
    v, *_ = across(per_seed, "CEF", "CEF w=0")
    M("StyleCefZeroMean", pts(v.mean()))

    def wsel(df):
        d = df[(df.method == "CEF") & df.cfg.notna()]
        ws = [ast.literal_eval(r)[1][1] for r in d.cfg]
        return np.mean(ws), int(sum(x == 0 for x in ws)), len(ws)
    mw, nz, n = wsel(pets)
    M("PetsMeanW", f"{mw:.2f}")
    M("PetsZeroWN", str(nz))
    M("PetsSelN", str(n))
    mw, nz, n = wsel(per_seed)
    M("StyleMeanW", f"{mw:.2f}")
    M("StyleZeroWN", str(nz))
    M("StyleSelN", str(n))

    # ── secondary: CGPR's image-level protocol ─────────────────────────────
    for a, b, key in (("CEF", "ZLaP", "ImgCefZlap"), ("CEF", "CGPR", "ImgCefCgpr"),
                      ("ZLaP", "CGPR", "ImgZlapCgpr"), ("CEF", "CEF w=0", "ImgCefZero")):
        emit_cmp(img, a, b, key)
    art_m = per_seed.groupby(["dataset", "backbone", "method"]).bacc.mean()
    img_m = img.groupby(["dataset", "backbone", "method"]).bacc.mean()
    for exp, key in (("CEF", "Cef"), ("ZLaP", "Zlap"), ("CGPR", "Cgpr")):
        dlt = (art_m.xs(exp, level="method") - img_m.xs(exp, level="method")) * 100
        M("ArtShift" + key + "Mean", f"{dlt.mean():+.1f}")
        M("ArtShift" + key + "Min", f"{dlt.min():+.1f}")
        M("ArtShift" + key + "Max", f"{dlt.max():+.1f}")

    chk = os.path.join(HYPER, "results", "supplementary", "cgpr_per_seed_check.csv")
    cg = pd.read_csv(chk)
    fx = cg[cg.config.str.startswith("fixed")].groupby(["dataset", "backbone"]).bacc.mean()
    ps_ = cg[cg.config.str.startswith("per-seed")].groupby(["dataset", "backbone"]).bacc.mean()
    M("CgprPsMin", pts2((ps_ - fx).min()))
    M("CgprPsMax", pts2((ps_ - fx).max()))
    zl = pd.read_csv("zlap_results.csv").set_index(["dataset", "backbone"]).tuned_b
    ce = pd.read_csv("unified_cef2.csv").set_index(["dataset", "backbone"]).cef
    M("ZlapOverCgprPsMin", pts((zl - ps_).min()))
    M("CefOverCgprPsMin", pts((ce - ps_).min()))
    cx = pd.read_csv("crux.csv").set_index(["dataset", "backbone"]).ugsp_knn
    r_ps, r_fx = cx - ps_, cx - fx
    M("RetunedPsMin", pts2(r_ps.min()))
    M("RetunedPsMax", pts2(r_ps.max()))
    M("RetunedPsNeg", str(int((r_ps < 0).sum())))
    M("RetunedAbsMax", f"{100 * max(r_ps.abs().max(), r_fx.abs().max()):.2f}")

    rep = pd.read_csv(os.path.join(ADIR, "split_report.csv")).drop_duplicates(["dataset", "seed"])
    M("ArtValFracMin", f"{100 * rep.val_frac.min():.1f}")
    M("ArtValFracMax", f"{100 * rep.val_frac.max():.1f}")
    M("ArtStyleShareMin", f"{100 * rep.style_val_frac_min.min():.1f}")
    M("ArtStyleShareMax", f"{100 * rep.style_val_frac_max.max():.1f}")
    M("ArtSharedMax", str(int(rep.artists_shared.max())))
    for ds, key in (("wikiart", "Wiki"), ("mp100k", "Mp")):
        r = rep[rep.dataset == ds]
        M(f"ArtValArtists{key}Min", str(int(r.artists_val.min())))
        M(f"ArtValArtists{key}Max", str(int(r.artists_val.max())))
    # Compute split overlap from released filenames/artist assignments;
    # regenerating tables must not require image embeddings or raw datasets.
    leak = []
    splitdir = os.path.join(HYPER, "paper3", "supplementary_material", "splits")
    for ds in ("wikiart", "mp100k"):
        split = pd.read_csv(os.path.join(splitdir, f"{ds}_image_level.csv.gz"))
        for seed, group in split.groupby("seed"):
            va = group[group.partition == "val"].artist
            te = group[group.partition == "test"].artist
            leak.append(100 * te.isin(set(va)).mean())
    M("ImgLeakMin", f"{min(leak):.1f}")
    M("ImgLeakMax", f"{max(leak):.1f}")

    # Final CEF qualitative counts on the artist-disjoint seed-42 test pools.
    qdir = os.path.join(HYPER, 'results/supplementary/qualitative_final')
    for ds, key in [('wikiart', 'Wiki'), ('mp100k', 'Mp')]:
        q = json.load(open(os.path.join(qdir, f'{ds}_metadata.json')))
        for suffix, field in [('Images', 'test_images'), ('Corrected', 'corrected_pool'), ('Regressed', 'regressed_pool')]:
            M('FinalQual' + key + suffix, str(q[field]))

    # ── execution cost ─────────────────────────────────────────────────────
    # main tables are artist-disjoint, so cost is reported on that partition when measured
    cdir = os.path.join(HYPER, "results", "supplementary", "cost_artist")
    if not os.path.exists(os.path.join(cdir, "environment.json")):
        cdir = os.path.join(HYPER, "results", "supplementary", "cost")
    cp = pd.read_csv(os.path.join(cdir, "cost_per_pair.csv"))
    order = ["Vanilla", "CGPR", "ZLaP", "CEF-FusionOnly", "CEF"]
    label = {"Vanilla": "Vanilla VLM", "CGPR": "CGPR", "ZLaP": "ZLaP",
             "CEF-FusionOnly": "CEF, fusion only", "CEF": r"\textbf{CEF}"}
    g = cp.groupby(["dataset", "method"])[["val_evals", "val_evals_both", "select_s",
                                           "infer_s", "peak_rss_gb"]].mean()
    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{WikiArt} & \multicolumn{4}{c}{MultitaskPainting100k} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Method & Val.\ configs & Select (s) & Test (s) & Peak mem. (GB) & "
         r"Val.\ configs & Select (s) & Test (s) & Peak mem. (GB) \\", r"\midrule"]
    for m in order:
        cells = []
        for ds in ("wikiart", "mp100k"):
            r = g.loc[(ds, m)]
            cells += [f"{r.val_evals:.0f}", f"{r.select_s:.1f}", f"{r.infer_s:.2f}",
                      f"{r.peak_rss_gb:.2f}"]
        L.append(f"{label[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    w("tab_cost.tex", "\n".join(L))
    cef = cp[cp.method == "CEF"]
    rng_ = lambda a, b: str(a) if a == b else f"{a}--{b}"
    M("CefEvalsRange", rng_(int(cef.val_evals.min()), int(cef.val_evals.max())))
    M("CefEvalsBothRange", rng_(int(cef.val_evals_both.min()), int(cef.val_evals_both.max())))
    M("CgprEvals", str(int(cp[cp.method == "CGPR"].val_evals.iloc[0])))
    for ds, key in (("wikiart", "Wiki"), ("mp100k", "Mp")):
        M(f"CefSelect{key}", f"{g.loc[(ds, 'CEF')].select_s:.0f}")
        M(f"ZlapSelect{key}", f"{g.loc[(ds, 'ZLaP')].select_s:.0f}")
        M(f"CgprSelect{key}", f"{g.loc[(ds, 'CGPR')].select_s:.0f}")
        M(f"FoSelect{key}", f"{g.loc[(ds, 'CEF-FusionOnly')].select_s:.1f}")
        M(f"CefTest{key}", f"{g.loc[(ds, 'CEF')].infer_s:.1f}")
        M(f"CefMem{key}", f"{g.loc[(ds, 'CEF')].peak_rss_gb:.1f}")
    o = pd.read_csv(os.path.join(cdir, "cost_one_time.csv"))
    M("ImgThroughputMin", f"{o.image_throughput_per_s.min():.0f}")
    M("ImgThroughputMax", f"{o.image_throughput_per_s.max():.0f}")
    M("ConceptEncMaxS", f"{o.concept_encoding_s.max():.2f}")
    env = json.load(open(os.path.join(cdir, "environment.json")))
    M("CostCpu", env["cpu"].replace("12-Core Processor", "").strip())
    M("CostThreads", str(env["threads_per_method"]))
    M("CostGpu", env["gpu"] or "none")
    M("CostSplit", env.get("split", "image-level"))

    # ── review round 3 (anticipated review): r3_run.py, r3_probe.py ──
    r3dir = os.path.join(HYPER, "results", "supplementary", "round3")
    if os.path.exists(os.path.join(r3dir, "per_seed_r3.csv")):
        import sys
        import r3_tables
        r3_tables.emit(sys.modules[__name__], r3_tables.load_r3(),
                       pd.read_csv(os.path.join(r3dir, "probe_per_seed.csv")), pets, per_seed)

    import sys
    import legacy_review_tables
    legacy_review_tables.emit(sys.modules[__name__])
    if os.path.exists(os.path.join(HYPER, "results", "supplementary", "round4", "per_seed_r4.csv")):
        import r4_tables
        r4_tables.emit(sys.modules[__name__], per_seed, r3_tables.load_r3())

    # Raw result ID CEF denotes the phrase-only branch; display its full name.
    from pathlib import Path
    for table_file in Path(OUT).glob("*.tex"):
        if table_file.name.startswith("tab_hybrid"):
            continue
        table_text = re.sub(r"(?<!phrase-only )\bCEF\b", "phrase-only CEF", table_file.read_text())
        table_text = table_text.replace("phrase-only CEF (ours)", "phrase-only CEF")
        table_file.write_text(table_text)

    if os.path.exists(os.path.join(HYPER, "results", "supplementary", "round5", "per_seed.csv")):
        import r5_tables
        r5_tables.emit(sys.modules[__name__])
        import current_cef_tables
        current_cef_tables.emit(sys.modules[__name__])

    with open(MAC, "w") as fh:
        fh.write("% Generated by analysis/make_review_tables.py -- do not hand edit.\n")
        for k in sorted(macros):
            v = macros[k]
            # a leading hyphen in text mode typesets as a hyphen, not a minus sign
            if re.fullmatch(r"-\d+(\.\d+)?", v):
                v = r"\ensuremath{-}" + v[1:]
            fh.write(f"\\newcommand{{\\R{k}}}{{{v}}}\n")
    print(f"wrote {MAC} with {len(macros)} macros")


if __name__ == "__main__":
    main()
