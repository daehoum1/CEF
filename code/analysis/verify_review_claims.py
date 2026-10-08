"""Assert that every directional claim the manuscript makes is the direction the
data actually shows.

Macros give the manuscript the right numbers automatically, but they do not
protect the sentences around them: a rerun that flipped a sign would leave the
prose asserting the opposite of the table. Each check names the claim it guards.

Protocol. Main-table claims are checked on the ARTIST-DISJOINT per-seed results;
claims about CGPR's image-level protocol (secondary table, CGPR configuration,
reproduction of stored scores) are checked on the image-level results. Run from
analysis/ after make_review_tables.py; non-zero exit means a claim in the
manuscript is no longer supported.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

from make_review_tables import load_artist, load_image

from pathlib import Path
HYPER = os.environ.get("CEF_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))
PAPER = os.path.join(HYPER, "paper3")
SUPP = os.path.join(HYPER, "results", "supplementary")
failures: list[str] = []
checks = 0


def pair_diffs(df, a, b, col="bacc"):
    pa = df[df.method == a].set_index(["dataset", "backbone", "seed"])[col]
    pb = df[df.method == b].set_index(["dataset", "backbone", "seed"])[col]
    idx = pa.index.intersection(pb.index)
    assert len(idx), f"no shared rows for {a} vs {b}"
    return (pa[idx] - pb[idx]).rename("d").reset_index().groupby(["dataset", "backbone"]).d.mean()


def check(name, ok, detail=""):
    global checks
    checks += 1
    print(f"[{'OK' if ok else 'FAIL'}]   {name}  {detail}")
    if not ok:
        failures.append(name)


def every(df, a, b, label, col="bacc"):
    d = pair_diffs(df, a, b, col)
    check(f"{label}: {a} beats {b} on every pair ({col})", bool((d > 0).all()),
          f"positive {int((d > 0).sum())}/{len(d)}, min {d.min()*100:+.2f} pts")
    return d


def main():
    art = load_artist()
    img = load_image()

    # ── Table 3 and main comparison (artist-disjoint) ──────────────────────
    arms = ["Vanilla", "CGPR", "ZLaP", "CEF"]
    for col in ("bacc", "top1"):
        m = art[art.method.isin(arms)].groupby(["dataset", "backbone", "method"])[col].mean().unstack()
        check(f"Base-reference table: CEF is best on every pair ({col})", bool((m.idxmax(axis=1) == "CEF").all()),
              f"CEF best on {int((m.idxmax(axis=1) == 'CEF').sum())}/8")
    every(art, "ZLaP", "CGPR", "main")
    every(art, "CEF", "ZLaP", "main")
    every(art, "CEF", "CGPR", "main")
    every(art, "CEF", "ZLaP", "main", col="top1")
    every(art, "CEF", "CGPR", "main", col="top1")
    every(art, "CEF", "Vanilla", "main")
    d = pair_diffs(art, "ZLaP", "ZLaP (default cfg)")
    check("tuning ZLaP improves on its published defaults on average", bool(d.mean() > 0),
          f"mean {d.mean()*100:+.2f} pts, positive {int((d > 0).sum())}/8")
    every(art, "CEF w/ CGPR propagation", "CGPR", "same evidence on CGPR's host")
    every(art, "CEF", "CEF w/ CGPR propagation", "label propagation is the better host")
    mp = pair_diffs(art[art.dataset == "mp100k"], "CEF", "ZLaP").mean()
    wk = pair_diffs(art[art.dataset == "wikiart"], "CEF", "ZLaP").mean()
    check("CEF's gain over ZLaP is larger on MP100k than on WikiArt", bool(mp > wk),
          f"MP100k {mp*100:.2f} vs WikiArt {wk*100:.2f} pts")
    d_art = pair_diffs(art, "CEF", "ZLaP")
    check("smallest CEF-ZLaP margin is WikiArt/SigLIP (artist-disjoint)",
          d_art.idxmin() == ("wikiart", "SigLIP"), f"{d_art.idxmin()} {d_art.min()*100:+.2f}")

    # ── classification by description ──────────────────────────────────────
    every(art, "CEF", "DCLIP + ZLaP", "descriptor")
    every(art, "CEF", "CuPL + ZLaP", "descriptor")
    d = pair_diffs(art, "DCLIP", "Vanilla")
    check("DCLIP standalone is not uniformly above vanilla", bool(not (d > 0).all()),
          f"positive {int((d > 0).sum())}/8")
    d = pair_diffs(art, "DCLIP + ZLaP", "ZLaP")
    check("DCLIP+LP improves on ZLaP on average", bool(d.mean() > 0),
          f"mean {d.mean()*100:+.2f} pts, positive {int((d > 0).sum())}/8")

    # ── dictionary controls ────────────────────────────────────────────────
    real = pair_diffs(art, "CEF", "CEF w=0")
    check("CEF beats its concept-free control on every pair", bool((real > 0).all()),
          f"min {real.min()*100:+.2f} pts")
    for ctrl in ["CEF [rand_chars]", "CEF [rand_words]", "CEF [shuffle_owner]"]:
        every(art, "CEF", ctrl, "controls")
        z = pair_diffs(art, ctrl, "CEF w=0")
        check(f"{ctrl} adds ~nothing over w=0 (mean < 0.1 pt and below the real dictionary)",
              bool(z.mean() < 0.001 and z.mean() < real.mean()),
              f"control {z.mean()*100:+.3f} pts vs real {real.mean()*100:+.2f} pts")

    # ── fixed-configuration transfer ───────────────────────────────────────
    for a in ["CEF [loo]", "CEF [global]"]:
        d = pair_diffs(art, a, "ZLaP")
        check(f"{a} beats tuned ZLaP on 7 of 8 pairs", bool(int((d > 0).sum()) == 7 and d.mean() > 0),
              f"positive {int((d > 0).sum())}/8, mean {d.mean()*100:+.2f} pts")
        check(f"{a}'s only loss is WikiArt/SigLIP", d.idxmin() == ("wikiart", "SigLIP"),
              f"worst {d.idxmin()}")
    d = pair_diffs(art, "CEF", "CEF [loo]")
    check("per-pair tuning is worth something on average", bool(d.mean() >= 0),
          f"mean {d.mean()*100:+.2f} pts")

    # ── ablations ─────────────────────────────────────────────────────────
    d = pair_diffs(art, "CEF w=0", "ZLaP")
    check("the w=0 parameterization alone contributes little (|mean| < 0.5 pt)",
          bool(abs(d.mean()) < 0.005), f"mean {d.mean()*100:+.2f} pts")
    f = every(art, "CEF", "CEF w/ filtered dict", "filtered dictionary is never better")
    fm = pair_diffs(art[art.dataset == "mp100k"], "CEF", "CEF w/ filtered dict").mean()
    fw = pair_diffs(art[art.dataset == "wikiart"], "CEF", "CEF w/ filtered dict").mean()
    check("filtering costs more on MP100k than on WikiArt", bool(fm > fw),
          f"MP100k {fm*100:.2f} vs WikiArt {fw*100:.2f} pts")
    d = pair_diffs(art, "CEF w/o debias", "CEF")
    check("removing debiasing helps only marginally where it helps (<= 0.5 pt)",
          bool(d.max() <= 0.005 and (d > 0).any()),
          f"best gain from removing it {d.max()*100:+.2f} pts on {int((d > 0).sum())} pairs")

    # ── fusion without propagation ─────────────────────────────────────────
    every(art, "FusionOnly", "Vanilla", "fusion only")
    every(art, "FusionOnly", "DCLIP", "fusion only")
    every(art, "FusionOnly", "CuPL", "fusion only")
    every(art, "CEF", "FusionOnly", "propagation adds to fusion")
    d = pair_diffs(art, "FusionOnly", "ZLaP")
    check("fusion only is not reliably different from ZLaP (sign changes across pairs, |mean| < 1.5 pt)",
          bool(0 < int((d > 0).sum()) < 8 and abs(d.mean()) < 0.015),
          f"positive {int((d > 0).sum())}/8, mean {d.mean()*100:+.2f} pts")

    # ── diagnosis: mask toggled on the artist-disjoint splits ─────────────
    d = pair_diffs(art, "CGPR (mask on)", "CGPR (mask off)")
    check("artist-disjoint mask effect is positive on every pair", bool((d > 0).all()),
          f"range {d.min()*100:+.3f}..{d.max()*100:+.3f} pts")

    # ── Oxford-IIIT Pet probe: the style side of each comparison ──────────
    pets = pd.read_csv("pets_per_seed.csv")
    s_zero_zlap = pair_diffs(art, "CEF w=0", "ZLaP").mean()
    s_cef_zero = pair_diffs(art, "CEF", "CEF w=0").mean()
    p_zero_zlap = pair_diffs(pets, "CEF w=0", "ZLaP").mean()
    p_cef_zero = pair_diffs(pets, "CEF", "CEF w=0").mean()
    check("gain decomposition runs opposite ways on style and on Pet",
          bool(s_cef_zero > s_zero_zlap and p_zero_zlap > p_cef_zero),
          f"style w0 {s_zero_zlap*100:.2f}/concepts {s_cef_zero*100:.2f}; "
          f"pet w0 {p_zero_zlap*100:.2f}/concepts {p_cef_zero*100:.2f}")
    import ast

    def wsel(df):
        c = df[(df.method == "CEF") & df.cfg.notna()]
        ws = [ast.literal_eval(r)[1][1] for r in c.cfg]
        return np.mean(ws), np.mean([x == 0 for x in ws])
    sw, sz = wsel(art)
    pw, pz = wsel(pets)
    check("selected fusion weight is higher on style, and zero less often",
          bool(sw > pw and sz < pz), f"style mean w {sw:.2f} (zero {sz:.1%}), pet {pw:.2f} (zero {pz:.1%})")
    rs = pair_diffs(art, "CEF", "CEF [rand_words]").mean()
    rp = pair_diffs(pets, "CEF", "CEF [rand_words]").mean()
    check("random-word separation on Pet is several times smaller than on style",
          bool(rs >= 3 * rp), f"style {rs*100:.2f} vs pet {rp*100:.2f} pts")

    # ── secondary: CGPR's image-level protocol ────────────────────────────
    am = art[art.method.isin(arms)].groupby(["dataset", "backbone", "method"]).bacc.mean().unstack()
    im = img[img.method.isin(arms)].groupby(["dataset", "backbone", "method"]).bacc.mean().unstack()
    same = all(list(am.loc[k].sort_values().index) == list(im.loc[k].sort_values().index) for k in am.index)
    check("method ordering is identical under both protocols (bAcc)", same)
    every(img, "CEF", "ZLaP", "image-level")
    every(img, "ZLaP", "CGPR", "image-level")
    for exp in ("CEF", "ZLaP", "CGPR"):
        a_ = art[art.method == exp].groupby(["dataset", "backbone"]).bacc.mean()
        i_ = img[img.method == exp].groupby(["dataset", "backbone"]).bacc.mean()
        sh = a_ - i_
        check(f"protocol shift of {exp} does not share a sign",
              bool((sh > 0).any() and (sh < 0).any()),
              f"positive {int((sh > 0).sum())}/8, range {sh.min()*100:+.2f}..{sh.max()*100:+.2f}")

    chk = os.path.join(SUPP, "cgpr_per_seed_check.csv")
    cg = pd.read_csv(chk)
    fx = cg[cg.config.str.startswith("fixed")].groupby(["dataset", "backbone"]).bacc.mean()
    ps = cg[cg.config.str.startswith("per-seed")].groupby(["dataset", "backbone"]).bacc.mean()
    t3 = pd.read_csv("final_summary.csv").set_index(["dataset", "backbone"])["CGPR|bacc_mean"]
    check("fixed-config CGPR reproduces the image-level CGPR row",
          bool((fx - t3[fx.index]).abs().max() < 1e-9), f"max |delta| = {(fx - t3[fx.index]).abs().max():.1e}")
    zl = pd.read_csv("zlap_results.csv").set_index(["dataset", "backbone"]).tuned_b
    ce = pd.read_csv("unified_cef2.csv").set_index(["dataset", "backbone"]).cef
    for name, ref in (("ZLaP", zl), ("CEF", ce)):
        dd = ref[ps.index] - ps
        check(f"image-level: {name} beats per-seed-selected CGPR on every pair",
              bool((dd > 0).all()), f"min {dd.min()*100:+.2f} pts")
    cx = pd.read_csv("crux.csv").set_index(["dataset", "backbone"]).ugsp_knn
    worst = max((cx - ps).abs().max(), (cx - fx).abs().max())
    check("concept-free CGPR within 0.31 pts of CGPR under either selection",
          bool(worst <= 0.0031), f"max |delta| = {worst*100:.2f} pts")

    ic = pd.read_csv("review_controls_per_seed.csv")
    for name, src, col in (("CEF", "unified_cef2.csv", "cef"), ("ZLaP", "zlap_results.csv", "tuned_b")):
        old = pd.read_csv(src).set_index(["dataset", "backbone"])[col]
        new = ic[ic.method == name].groupby(["dataset", "backbone"]).bacc.mean()
        delta = (old[new.index] - new).abs().max()
        check(f"image-level reruns reproduce stored {name} scores", bool(delta < 1e-9),
              f"max |delta| = {delta:.2e}")

    rep = pd.read_csv(os.path.join(SUPP, "artist_disjoint", "split_report.csv"))
    check("artist-disjoint splits share no artist on any seed", bool(rep.artists_shared.max() == 0),
          f"max shared = {int(rep.artists_shared.max())}")
    mac = open(os.path.join(PAPER, "review_numbers.tex")).read()
    leak = float(re.search(r"\\newcommand\{\\RImgLeakMin\}\{([\d.]+)\}", mac).group(1))
    check("image-level resplit leaks artists for nearly all test images", leak > 90, f"min {leak}%")

    # ── computational cost ─────────────────────────────────────────────────
    cdir = os.path.join(SUPP, "cost_artist")
    if not os.path.exists(os.path.join(cdir, "environment.json")):
        cdir = os.path.join(SUPP, "cost")
    cp = pd.read_csv(os.path.join(cdir, "cost_per_pair.csv"))
    g = cp.groupby(["dataset", "method"])[["val_evals", "select_s", "infer_s", "peak_rss_gb"]].mean()
    ev = cp.groupby("method").val_evals
    check(f"grid sizes: ZLaP 75, fusion-only 80, CGPR 400 ({os.path.basename(cdir)})",
          bool(ev.min()["ZLaP"] == ev.max()["ZLaP"] == 75 and ev.min()["CEF-FusionOnly"] == 80
               and ev.min()["CGPR"] == 400))
    check("CEF's staged search evaluates far fewer than 6,000 configurations",
          bool(ev.max()["CEF"] < 6000), f"max {ev.max()['CEF']}")
    for ds in ("wikiart", "mp100k"):
        c, z, cg_, fo = (g.loc[(ds, m)] for m in ("CEF", "ZLaP", "CGPR", "CEF-FusionOnly"))
        check(f"{ds}: CEF has the most expensive selection, still seconds",
              bool(c.select_s > z.select_s and c.select_s > cg_.select_s and c.select_s < 60),
              f"CEF {c.select_s:.1f}s, ZLaP {z.select_s:.1f}s, CGPR {cg_.select_s:.1f}s")
        check(f"{ds}: CEF peak memory close to ZLaP (within 15%)",
              bool(abs(c.peak_rss_gb - z.peak_rss_gb) <= 0.15 * z.peak_rss_gb),
              f"CEF {c.peak_rss_gb:.2f}GB, ZLaP {z.peak_rss_gb:.2f}GB")
        check(f"{ds}: fusion only an order of magnitude cheaper than CEF",
              bool(fo.select_s * 10 <= c.select_s and fo.infer_s * 10 <= c.infer_s),
              f"select {fo.select_s:.2f}s vs {c.select_s:.1f}s, test {fo.infer_s:.2f}s vs {c.infer_s:.2f}s")

    # Final-method cases are checked by verify_final_qualitative.py against R5.

    # ── review round 3: regenerated dictionaries ───────────────────────────
    r3path = os.path.join(SUPP, "round3", "per_seed_r3.csv")
    if os.path.exists(r3path):
        import json as _json
        import r3_tables
        r3 = pd.read_csv(r3path)
        ar = pd.read_csv(os.path.join(SUPP, "artist_disjoint", "per_seed_results.csv"))
        kk = ["dataset", "backbone", "seed", "metric"]
        for new, old in (("Vanilla", "Vanilla"), ("ZLaP", "ZLaP"), ("CEF w=0", "CEF w=0"),
                         ("CEF [v3]", "CEF"), ("CEF-FusionOnly", "CEF-FusionOnly")):
            a_ = ar[ar.experiment == old].set_index(kk).score
            b_ = r3[r3.experiment == new].set_index(kk).score
            idx = a_.index.intersection(b_.index)
            check(f"round 3 run reproduces the stored artist-disjoint {old} arm",
                  bool(len(idx) == 80 and (a_[idx] - b_[idx]).abs().max() < 1e-12),
                  f"{len(idx)} rows, max |d| {(a_[idx] - b_[idx]).abs().max():.1e}")
        A = os.path.join(HYPER, "assets", "r3_dicts")
        per_style = []
        for ds_ in ("wikiart", "mp100k"):
            for g_ in range(5):
                d_ = _json.load(open(os.path.join(A, f"{ds_}_orig_g{g_}.json")))
                per_style += [len(v_) for k_, v_ in d_.items() if not k_.startswith("_")]
        check("round 3: every draw holds twelve phrases per style", set(per_style) == {12}, f"{set(per_style)}")
        pr = pd.read_csv(os.path.join(SUPP, "round3", "probe_per_seed.csv"))
        cov = pr.groupby(["domain", "method"]).size().groupby("domain").min()
        check("round 3: probe results cover 4 domains x 4 backbones x 5 seeds for every arm",
              bool(set(cov.index) == {"dtd", "eurosat", "aircraft", "pets_s5"} and (cov == 20).all()),
              f"{cov.to_dict()}")
        R = r3_tables.load_r3()
        every(R, "CEF [orig]", "ZLaP", "round 3 draws (mean over five)")
        gm = R.groupby(["dataset", "backbone", "method"]).bacc.mean()
        og_mean = sum(gm.xs(f"CEF [orig_g{g_}]", level="method") for g_ in range(5)) / 5
        above = gm.xs("CEF [v3]", level="method") - og_mean
        check("round 3: the reported dictionary is above the draw mean on every pair", bool((above > 0).all()),
              f"min {above.min()*100:+.2f} pts")
        og_sd = pd.concat([gm.xs(f"CEF [orig_g{g_}]", level="method") for g_ in range(5)], axis=1).std(axis=1, ddof=1)
        seed_sd = R[R.method == "CEF [v3]"].groupby(["dataset", "backbone"]).bacc.std(ddof=1)
        check("round 3: across-draw SD is comparable to across-seed SD (within a factor of two)",
              bool(0.5 < og_sd.mean() / seed_sd.mean() < 2), f"{og_sd.mean()*100:.2f} vs {seed_sd.mean()*100:.2f}")

    # ── review round 3: domain probes ─────────────────────────────────────
    if os.path.exists(r3path):
        pr = pd.read_csv(os.path.join(SUPP, "round3", "probe_per_seed.csv")).rename(columns={"domain": "dataset"})
        def conc_by_backbone(dm):
            g = pr[pr.dataset == dm].groupby(["backbone", "method"]).bacc.mean().unstack()
            return g["CEF"] - g["CEF w=0"], g
        for dm in ("dtd", "eurosat"):
            c, g = conc_by_backbone(dm)
            check(f"round 3 probe {dm}: concept component positive for every backbone", bool((c > 0).all()),
                  f"min {c.min()*100:+.2f} pts")
            rw = (g["CEF"] - g["CEF [rand_words]"]).mean()
            check(f"round 3 probe {dm}: random words stay well below the real dictionary (> 1 pt)",
                  bool(rw > 0.01), f"{rw*100:+.2f} pts")
        c, g = conc_by_backbone("aircraft")
        check("round 3 probe aircraft: concept component is about zero (|mean| < 0.2 pt)",
              bool(abs(c.mean()) < 0.002), f"{c.mean()*100:+.3f} pts")
        ws = [ast.literal_eval(x)[1][1] for x in pr[(pr.dataset == "aircraft") & (pr.method == "CEF")].cfg]
        check("round 3 probe aircraft: fusion weight often set to zero", sum(w_ == 0 for w_ in ws) >= 4,
              f"{sum(w_ == 0 for w_ in ws)}/{len(ws)}")
        c, g = conc_by_backbone("pets_s5")
        check("round 3 probe pet: concept component is small (< 1 pt) and below DTD and EuroSAT",
              bool(c.mean() < 0.01 and c.mean() < conc_by_backbone("dtd")[0].mean()
                   and c.mean() < conc_by_backbone("eurosat")[0].mean()), f"{c.mean()*100:+.2f} pts")
        van = {dm: pr[(pr.dataset == dm) & (pr.method == "Vanilla")].bacc.mean() for dm in pr.dataset.unique()}
        van["wikiart"] = art[(art.dataset == "wikiart") & (art.method == "Vanilla")].bacc.mean()
        van["mp100k"] = art[(art.dataset == "mp100k") & (art.method == "Vanilla")].bacc.mean()
        check("round 3 probe: FGVC-Aircraft has the lowest vanilla bAcc of the six domains",
              min(van, key=van.get) == "aircraft", f"{ {k_: round(v_*100, 1) for k_, v_ in van.items()} }")
        gp = pd.read_csv(os.path.join(SUPP, "round3", "probe_grounding_points.csv"))
        rho = stats.spearmanr(gp.van, gp.conc)[0]
        check("round 3 probe: across (domain, backbone) the grounding correlation is weak (|rho| < 0.3)",
              bool(abs(rho) < 0.3), f"rho {rho:+.2f}")
        cen_v = gp.van - gp.groupby("domain").van.transform("mean")
        cen_c = gp.conc - gp.groupby("domain").conc.transform("mean")
        rw_ = stats.spearmanr(cen_v, cen_c)[0]
        check("round 3 probe: within domains, lower vanilla accuracy goes with a larger component (rho < -0.3)",
              bool(rw_ < -0.3), f"rho {rw_:+.2f}")
        for dm in ("dtd", "pets_s5"):
            g_ = gp[gp.dataset == dm] if "dataset" in gp else gp[gp.domain == dm]
            check(f"round 3 probe {dm}: the weakest backbone gains most, the strongest least",
                  bool(g_.loc[g_.van.idxmin()].conc == g_.conc.max() and g_.loc[g_.van.idxmax()].conc == g_.conc.min()))

    # ── every macro referenced in the manuscript must exist ───────────────
    defined = set(re.findall(r"\\newcommand\{\\(R[A-Za-z]+)\}", mac))
    used = set()
    for root, _, files in os.walk(PAPER):
        if "supplementary_material" in root:
            continue
        for fname in files:
            if fname.endswith(".tex") and fname != "review_numbers.tex":
                used |= set(re.findall(r"\\(R[A-Za-z]+)\b", open(os.path.join(root, fname)).read()))
    # Standard algpseudocode commands are not generated numerical macros.
    used -= {"Require", "Return", "Repeat"}
    missing = sorted(used - defined)
    check("every macro used in the text is defined", not missing,
          f"missing: {missing}" if missing else f"{len(used)} used, {len(defined)} defined")

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
