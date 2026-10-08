"""Review round 3: tables and prose macros.

Called from make_review_tables.main() so that every macro lands in the single
paper3/review_numbers.tex. Inputs:
  results/supplementary/round3/per_seed_r3.csv     r3_run.py
  results/supplementary/round3/probe_per_seed.csv  r3_probe.py
Tables written: tab_r3_gens.tex, tab_r3_probe.tex, tab_effects.tex
"""
from __future__ import annotations

import ast
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
HYPER = os.environ.get("CEF_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
R3 = os.path.join(HYPER, "results", "supplementary", "round3")
KEY = ["dataset", "backbone", "seed"]
GEN = range(5)


def load_r3():
    r = pd.read_csv(os.path.join(R3, "per_seed_r3.csv"))
    piv = r.pivot_table(index=KEY + ["experiment"], columns="metric", values="score").reset_index()
    cfg = r[r.metric == "bAcc"][KEY + ["experiment", "selected_config"]]
    piv = piv.merge(cfg, on=KEY + ["experiment"], how="left")
    piv = piv.rename(columns={"experiment": "method", "bAcc": "bacc", "Top-1": "top1",
                              "selected_config": "cfg"})
    names = [f"CEF [orig_g{g}]" for g in GEN]
    fam = (piv[piv.method.isin(names)].groupby(KEY)[["bacc", "top1"]].mean().reset_index()
           .assign(method="CEF [orig]", cfg=""))
    return pd.concat([piv, fam], ignore_index=True)


def seed_pos(df, a, b, col="bacc"):
    pa = df[df.method == a].set_index(KEY)[col]
    pb = df[df.method == b].set_index(KEY)[col]
    idx = pa.index.intersection(pb.index)
    d = pa[idx] - pb[idx]
    return int((d > 0).sum()), int(len(d))


def emit(mrt, r3, probe, pets_opus, per_seed):
    M, pts, f4 = mrt.M, mrt.pts, mrt.f4

    def cmp(df, a, b, key, col="bacc"):
        mrt.emit_cmp(df, a, b, key, col=col)

    all_ = r3
    for ds, k in (("wikiart", "Wiki"), ("mp100k", "Mp")):
        d0 = json.load(open(os.path.join(HYPER, "assets", "r3_dicts", f"{ds}_orig_g0.json")))
        M(f"RtPhrases{k}", str(sum(len(v) for kk, v in d0.items() if not kk.startswith("_"))))

    # ── C5: independent generations ─────────────────────────────────────────
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Dataset & Backbone & ZLaP & CEF (reported dict.) & CEF, 5 draws & draw range \\",
         r"\midrule"]
    gen_sd, seed_sd, rows = [], [], []
    for di, (ds, dsl) in enumerate(mrt.DS):
        for i, bb in enumerate(mrt.BB):
            d = all_[(all_.dataset == ds) & (all_.backbone == bb)]
            gmean = d.groupby("method").bacc.mean()
            og = np.array([gmean[f"CEF [orig_g{g}]"] for g in GEN])
            gen_sd.append(og.std(ddof=1))
            seed_sd.append(d[d.method == "CEF [v3]"].bacc.std(ddof=1))
            rows.append(dict(ds=ds, bb=bb, og_sd=og.std(ddof=1),
                             v3_minus_ogmean=gmean["CEF [v3]"] - og.mean(),
                             v3_rank=int((og > gmean["CEF [v3]"]).sum())))
            L.append(f"{dsl if i == 0 else ''} & {bb} & {f4(gmean['ZLaP'])} & {f4(gmean['CEF [v3]'])} & "
                     rf"{f4(og.mean())}\,$\pm$\,{f4(og.std(ddof=1))} & {f4(og.min())}--{f4(og.max())} \\")
        if di == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    mrt.w("tab_r3_gens.tex", "\n".join(L))
    rows = pd.DataFrame(rows)
    M("RtGenSdMin", f"{100 * min(gen_sd):.2f}")
    M("RtGenSdMax", f"{100 * max(gen_sd):.2f}")
    M("RtGenSdMean", f"{100 * np.mean(gen_sd):.2f}")
    M("RtSeedSdMean", f"{100 * np.mean(seed_sd):.2f}")
    M("RtVthreeAboveAllDraws", str(int((rows.v3_rank == 0).sum())))
    M("RtVthreeMinusDrawsMean", mrt.pts2(rows.v3_minus_ogmean.mean()))
    M("RtVthreeMinusDrawsMin", mrt.pts2(rows.v3_minus_ogmean.min()))
    M("RtVthreeMinusDrawsMax", mrt.pts2(rows.v3_minus_ogmean.max()))
    cmp(all_, "CEF [orig]", "ZLaP", "RtOgZlap")
    gm = all_.groupby(["dataset", "backbone", "method"]).bacc.mean()
    M("RtAllOgGenOverZlap", str(int(sum(gm[(ds, bb, f"CEF [orig_g{gg}]")] > gm[(ds, bb, "ZLaP")]
                                         for ds in ("wikiart", "mp100k") for bb in mrt.BB for gg in GEN))))
    cmp(all_, "CEF [orig]", "CEF w=0", "RtOgZero")
    cmp(all_, "CEF [v3]", "CEF [orig]", "RtVthreeOg")

    # ── C4: domain probes ───────────────────────────────────────────────────
    style = per_seed[per_seed.method.isin(["Vanilla", "ZLaP", "CEF w=0", "CEF", "CEF [rand_words]"])].copy()
    style["domain"] = style.dataset
    po = pets_opus.rename(columns={"dataset": "domain"})
    pr = probe.copy()
    order = [("wikiart", "WikiArt"), ("mp100k", "MultitaskPainting100k"), ("aircraft", "FGVC-Aircraft"),
             ("dtd", "DTD"), ("eurosat", "EuroSAT"), ("pets_s5", "Oxford-IIIT Pet"),
             ("pets", r"Oxford-IIIT Pet$^\dagger$")]
    allp = pd.concat([style[["domain", "backbone", "seed", "method", "bacc", "cfg"]],
                      po[["domain", "backbone", "seed", "method", "bacc", "cfg"]],
                      pr[["domain", "backbone", "seed", "method", "bacc", "cfg"]]], ignore_index=True)

    def wsel(d):
        ws = [ast.literal_eval(c)[1][1] for c in d[(d.method == "CEF") & d.cfg.notna() & (d.cfg != "")].cfg]
        return np.mean(ws), 100 * np.mean([x == 0 for x in ws])

    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Domain & Vanilla & ZLaP & $w{=}0$ & CEF & Concepts & Real$-$random & mean $w$ \\",
         r"\midrule"]
    pts_rows = []
    for dm, lab in order:
        d = allp[allp.domain == dm]
        if not len(d):
            continue
        g = d.groupby("method").bacc.mean()
        conc = g["CEF"] - g["CEF w=0"]
        rw = g["CEF"] - g["CEF [rand_words]"]
        mw, zpct = wsel(d)
        L.append(f"{lab} & {pts(g['Vanilla'])} & {pts(g['ZLaP'])} & {pts(g['CEF w=0'])} & {pts(g['CEF'])} & "
                 f"{mrt.pts2(conc)} & {mrt.pts2(rw)} & {mw:.2f} \\\\")
        k = {"wikiart": "Wiki", "mp100k": "Mp", "aircraft": "Air", "dtd": "Dtd", "eurosat": "Euro",
             "pets_s5": "PetS", "pets": "PetO"}[dm]
        M(f"RtPr{k}Van", pts(g["Vanilla"]))
        M(f"RtPr{k}Conc", mrt.pts2(conc))
        M(f"RtPr{k}CefZlap", mrt.pts2(g["CEF"] - g["ZLaP"]))
        M(f"RtPr{k}ZeroZlap", mrt.pts2(g["CEF w=0"] - g["ZLaP"]))
        M(f"RtPr{k}RandW", mrt.pts2(rw))
        M(f"RtPr{k}MeanW", f"{mw:.2f}")
        M(f"RtPr{k}ZeroWPct", f"{zpct:.0f}")
        if dm not in ("wikiart", "mp100k"):
            df = d.assign(dataset=dm)
            v, wil, sgn, kpos, n = mrt.across(df, "CEF", "CEF w=0")
            M(f"RtPr{k}ConcPos", str(kpos))
            M(f"RtPr{k}ConcN", str(n))
            v, _, _, kpos, n = mrt.across(df, "CEF", "CEF [rand_words]")
            M(f"RtPr{k}RandWPos", str(kpos))
        if dm != "pets":
            for bb, gb in d.groupby("backbone"):
                gg = gb.groupby("method").bacc.mean()
                pts_rows.append(dict(domain=dm, backbone=bb, van=gg["Vanilla"], conc=gg["CEF"] - gg["CEF w=0"],
                                     w=wsel(gb)[0]))
        if dm == "mp100k":
            L.append(r"\midrule")
        if dm not in ("wikiart", "mp100k", "pets"):
            pb = d.groupby(["backbone", "method"]).bacc.mean().unstack()
            cb = pb["CEF"] - pb["CEF w=0"]
            M(f"RtPr{k}ConcMax", mrt.pts2(cb.max()))
            M(f"RtPr{k}ConcMin", mrt.pts2(cb.min()))
            M(f"RtPr{k}ConcMaxVan", pts(pb["Vanilla"][cb.idxmax()]))
            M(f"RtPr{k}ConcMinVan", pts(pb["Vanilla"][cb.idxmin()]))
            M(f"RtPr{k}VanMin", pts(pb["Vanilla"].min()))
            M(f"RtPr{k}VanMax", pts(pb["Vanilla"].max()))
    L += [r"\bottomrule", r"\end{tabular}"]
    mrt.w("tab_r3_probe.tex", "\n".join(L))
    pr_df = pd.DataFrame(pts_rows)
    pr_df.to_csv(os.path.join(R3, "probe_grounding_points.csv"), index=False)
    rho, p = stats.spearmanr(pr_df.van, pr_df.conc)
    M("RtGroundRho", f"{rho:.2f}")
    M("RtGroundN", str(len(pr_df)))
    rho_w, _ = stats.spearmanr(pr_df.van, pr_df.w)
    M("RtGroundRhoW", f"{rho_w:.2f}")
    nonart = pr_df[~pr_df.domain.isin(["wikiart", "mp100k"])]
    rho_na, _ = stats.spearmanr(nonart.van, nonart.conc)
    M("RtGroundRhoNonArt", f"{rho_na:.2f}")
    M("RtGroundNNonArt", str(len(nonart)))
    # within a domain: centre each domain, then rank-correlate across backbones
    cen = pr_df.assign(van_c=pr_df.van - pr_df.groupby("domain").van.transform("mean"),
                       conc_c=pr_df.conc - pr_df.groupby("domain").conc.transform("mean"))
    M("RtGroundRhoWithin", f"{stats.spearmanr(cen.van_c, cen.conc_c)[0]:.2f}")
    helps = cen[cen.domain != "aircraft"]
    M("RtGroundRhoWithinHelps", f"{stats.spearmanr(helps.van_c, helps.conc_c)[0]:.2f}")
    M("RtGroundNWithinHelps", str(len(helps)))
    for dm, k in (("dtd", "Dtd"), ("eurosat", "Euro"), ("pets_s5", "PetS"), ("aircraft", "Air"),
                  ("wikiart", "Wiki"), ("mp100k", "Mp")):
        g = pr_df[pr_df.domain == dm]
        M(f"RtGroundRho{k}", f"{stats.spearmanr(g.van, g.conc)[0]:.2f}")

    # ── C3: effect-size summary ─────────────────────────────────────────────
    comps = [
        (per_seed, "CEF", "ZLaP", r"CEF $-$ ZLaP"),
        (per_seed, "CEF", "CGPR", r"CEF $-$ CGPR"),
        (per_seed, "CEF", "CEF w=0", r"CEF $-$ ($w{=}0$)"),
        (all_, "CEF [orig]", "ZLaP", r"CEF (5 new draws) $-$ ZLaP"),
    ]
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Difference (bAcc points) & Mean & Range over pairs & WikiArt & MP100k & Pairs $>0$ & Runs $>0$ \\",
         r"\midrule"]
    for df, a, b, lab in comps:
        d = mrt.pair_mean_diffs(df, a, b)
        sp, sn = seed_pos(df, a, b)
        L.append(f"{lab} & {mrt.pts2(d.mean())} & {mrt.pts2(d.min())} to {mrt.pts2(d.max())} & "
                 f"{mrt.pts2(d.xs('wikiart').mean())} & {mrt.pts2(d.xs('mp100k').mean())} & "
                 f"{int((d > 0).sum())}/{len(d)} & {sp}/{sn} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    mrt.w("tab_effects.tex", "\n".join(L))
