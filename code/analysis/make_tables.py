"""Generate the follow-up paper's LaTeX tables directly from the result CSVs,
so every number in the paper traces to a run."""
import numpy as np, pandas as pd
from scipy import stats
import os

import cef_paths

OUT = str(cef_paths.project_root() / "manuscript" / "tables")
BB = ["CLIP (OpenAI)", "MetaCLIP", "EVA02-CLIP", "SigLIP"]
DS = [("wikiart", "WikiArt"), ("mp100k", "MultitaskPainting100k")]
f4 = lambda v: f"{v:.4f}"

def w(name, s):
    with open(os.path.join(OUT, name), "w") as fh: fh.write(s)
    print("wrote", name)

# ── Table 1: diagnosis — coverage starvation ────────────────────────────────
m = pd.read_csv("diag_mask.csv")
L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
     r"Dataset & Backbone & $r^\star$ & Max cov. & Starved & Edges kept & Fallback mass \\",
     r"\midrule"]
for ds, dsl in DS:
    for i, bb in enumerate(BB):
        r = m[(m.dataset == ds) & (m.backbone == bb)].iloc[0]
        cov = min(1.0, r.n_concepts * r.top_r / r.N) * 100
        L.append(f"{dsl if i==0 else ''} & {bb} & {int(r.top_r)} & {cov:.1f}\\% & "
                 f"{r.starved_pct:.1f}\\% & {r.edges_kept_pct:.1f}\\% & {r.fallback_edge_pct:.1f}\\% \\\\")
    if ds == "wikiart": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_starvation.tex", "\n".join(L))

# ── Table 2: diagnosis — gain attribution ───────────────────────────────────
a = pd.read_csv("diag_attrib.csv")
cx = pd.read_csv("crux.csv")
L = [r"\begin{tabular}{llrrrr}", r"\toprule",
     r" & & \multicolumn{2}{c}{Mask toggled, tuning fixed} & \multicolumn{2}{c}{Independently re-tuned} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
     r"Dataset & Backbone & w/o mask & CGPR & w/o mask & CGPR \\",
     r"\midrule"]
for ds, dsl in DS:
    for i, bb in enumerate(BB):
        r = a[(a.dataset == ds) & (a.backbone == bb)].iloc[0]
        c = cx[(cx.dataset == ds) & (cx.backbone == bb)].iloc[0]
        L.append(f"{dsl if i==0 else ''} & {bb} & {f4(r.knnSP_cgprAlpha)} & {f4(r.CGPR)} & "
                 f"{f4(c.ugsp_knn)} & {f4(c.cgpr)} \\\\")
    if ds == "wikiart": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_attribution.tex", "\n".join(L))

# ── Table 3: main results ───────────────────────────────────────────────────
d = pd.read_csv("final_per_seed.csv")
t1 = pd.read_csv("top1_check.csv")
zl = pd.read_csv("zlap_results.csv")
un = pd.read_csv("unified_cef2.csv")
METH = [("Vanilla VLM", "Vanilla"), ("CGPR", "CGPR"), ("ZLaP", "ZLaP"),
        (r"\textbf{CEF (ours)}", "CEF")]
L = [r"\begin{tabular}{lrrrr}", r"\toprule",
     r"Method & \multicolumn{2}{c}{WikiArt} & \multicolumn{2}{c}{MP100k} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r" & Top-1 & bAcc & Top-1 & bAcc \\", r"\midrule"]
# Top-1 is reported with hyperparameters selected on validation Top-1 for
# EVERY method, so baselines are matched to ours. Bold marks the best value
# in each column, whichever method achieves it.
T1KEY = {"CEF (ours)": "cef_t", "Vanilla": "van", "CGPR": "cgpr"}

def _vals(ds, bb, key):
    if key == "ZLaP":
        zr = zl[(zl.dataset == ds) & (zl.backbone == bb)].iloc[0]
        return zr["tuned_t"], zr["tuned_b"], zr["tuned_b_std"]
    if key == "CEF":
        ur = un[(un.dataset == ds) & (un.backbone == bb)].iloc[0]
        return ur["cef_t"], ur["cef"], ur["cef_std"]
    g = d[(d.dataset == ds) & (d.backbone == bb)]
    tv = t1[(t1.dataset == ds) & (t1.backbone == bb)].iloc[0][T1KEY[key]]
    return tv, g[f"{key}|bacc"].mean(), g[f"{key}|bacc"].std()

for bb in BB:
    L.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{bb}}}}} \\")
    vals = {key: {ds: _vals(ds, bb, key) for ds, _ in DS} for _, key in METH}
    best = {(ds, j): max(vals[key][ds][j] for _, key in METH)
            for ds, _ in DS for j in (0, 1)}
    for disp, key in METH:
        cells = []
        for ds, _ in DS:
            tv, b, sd = vals[key][ds]
            bt = rf"\textbf{{{f4(tv)}}}" if tv == best[(ds, 0)] else f4(tv)
            bb_ = f"{f4(b)}\\,$\\pm$\\,{sd:.4f}"
            if b == best[(ds, 1)]: bb_ = rf"\textbf{{{f4(b)}}}\,$\pm$\,{sd:.4f}"
            cells += [bt, bb_]
        L.append(f"{disp} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule" if bb != BB[-1] else "")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_main.tex", "\n".join(L))

# ── Table 4: ablations ──────────────────────────────────────────────────────
AB = [(r"\textbf{CEF (full)}", "cef"),
      (r"\quad w/o concepts ($w{=}0$)", "w0"),
      (r"\quad w/ filtered dictionary", "filtered"),
      (r"\quad w/o concept debiasing", "nodebias")]
L = [r"\begin{tabular}{l" + "r"*8 + "}", r"\toprule",
     r"Variant & \multicolumn{4}{c}{WikiArt} & \multicolumn{4}{c}{MultitaskPainting100k} \\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
     " & " + " & ".join([b.replace(" (OpenAI)", "") for b in BB] * 2) + r" \\", r"\midrule"]
for disp, key in AB:
    cells = [f4(un[(un.dataset == ds) & (un.backbone == bb)].iloc[0][key])
             for ds, _ in DS for bb in BB]
    L.append(f"{disp} & " + " & ".join(cells) + r" \\")
# the alternative host: the same fused score under CGPR's own propagation
cells = [f4(d[(d.dataset == ds) & (d.backbone == bb)]["CEF (ours)|bacc"].mean())
         for ds, _ in DS for bb in BB]
L.append(r"\quad w/ CGPR propagation & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_ablation.tex", "\n".join(L))

# ── numbers quoted in the prose ─────────────────────────────────────────────
stat = {}
gains_b = [d[(d.dataset==ds)&(d.backbone==bb)]["CEF (ours)|bacc"].mean()
           - d[(d.dataset==ds)&(d.backbone==bb)]["CGPR|bacc"].mean() for ds,_ in DS for bb in BB]
gains_t = [t1[(t1.dataset==ds)&(t1.backbone==bb)].iloc[0]["cef_t"]
           - t1[(t1.dataset==ds)&(t1.backbone==bb)].iloc[0]["cgpr"] for ds,_ in DS for bb in BB]
gains_t_ug = [t1[(t1.dataset==ds)&(t1.backbone==bb)].iloc[0]["cef_t"]
              - t1[(t1.dataset==ds)&(t1.backbone==bb)].iloc[0]["ugsp_t"] for ds,_ in DS for bb in BB]
ug = [d[(d.dataset==ds)&(d.backbone==bb)]["UGSP-kNN|bacc"].mean()
      - d[(d.dataset==ds)&(d.backbone==bb)]["CGPR|bacc"].mean() for ds,_ in DS for bb in BB]
pv = [stats.ttest_rel(d[(d.dataset==ds)&(d.backbone==bb)]["CEF (ours)|bacc"],
                      d[(d.dataset==ds)&(d.backbone==bb)]["CGPR|bacc"]).pvalue
      for ds,_ in DS for bb in BB]
stat.update(bacc_min=min(gains_b), bacc_max=max(gains_b), bacc_mean=np.mean(gains_b),
            top1_min=min(gains_t), top1_max=max(gains_t), top1_mean=np.mean(gains_t),
            top1_vs_ugsp_min=min(gains_t_ug), top1_vs_ugsp_max=max(gains_t_ug),
            ugsp_vs_cgpr_min=min(ug), ugsp_vs_cgpr_max=max(ug), max_p=max(pv),
            mask_min=pd.read_csv("diag_attrib.csv").gain_from_mask.min(),
            mask_max=pd.read_csv("diag_attrib.csv").gain_from_mask.max(),
            starved_min=m.starved_pct.min(), starved_max=m.starved_pct.max(),
            kept_min=m.edges_kept_pct.min(), kept_max=m.edges_kept_pct.max(),
            fb_min=m.fallback_edge_pct.min(), fb_max=m.fallback_edge_pct.max())
def _pairs(col_a, col_b, frame_a, frame_b):
    out = []
    for ds, _ in DS:
        for bb in BB:
            a = frame_a[(frame_a.dataset == ds) & (frame_a.backbone == bb)].iloc[0][col_a]
            b = frame_b[(frame_b.dataset == ds) & (frame_b.backbone == bb)].iloc[0][col_b]
            out.append(a - b)
    return np.array(out)

def _col(frame, key):
    return np.array([frame[(frame.dataset == ds) & (frame.backbone == bb)].iloc[0][key]
                     for ds, _ in DS for bb in BB])

def _dcol(key):
    return np.array([d[(d.dataset == ds) & (d.backbone == bb)][key].mean()
                     for ds, _ in DS for bb in BB])

cef_b, zl_b = _col(un, "cef"), _col(zl, "tuned_b")
cef_t_, zl_t_ = _col(un, "cef_t"), _col(zl, "tuned_t")
cgpr_bb, cgpr_tt = _dcol("CGPR|bacc"), _col(t1, "cgpr")
w0_b, filt_b, nodeb_b = _col(un, "w0"), _col(un, "filtered"), _col(un, "nodebias")
oldhost = _dcol("CEF (ours)|bacc")

def rng(x, pref):
    return {f"{pref}_min": x.min(), f"{pref}_max": x.max(), f"{pref}_mean": x.mean(),
            f"{pref}_wins": int((x > 0).sum())}

stat.update(**rng(zl_b - cgpr_bb, "zlap_vs_cgpr"))
stat.update(**rng(cef_b - zl_b, "cef_vs_zlap"))
stat.update(**rng(cef_b - cgpr_bb, "cef_vs_cgpr"))
stat.update(**rng(cef_b - w0_b, "cef_vs_w0"))
stat.update(**rng(w0_b - zl_b, "w0_vs_zlap"))
stat.update(**rng(cef_b - filt_b, "cef_vs_filtered"))
stat.update(**rng(cef_b - nodeb_b, "cef_vs_nodebias"))
stat.update(**rng(cef_b - oldhost, "cef_vs_cgprhost"))
stat.update(**rng(cef_t_ - zl_t_, "cef_t_vs_zlap"))
stat.update(**rng(cef_t_ - cgpr_tt, "cef_t_vs_cgpr"))
stat.update(zlap_tuning_gain=(zl_b - _col(zl, "default_b")).mean())
pd.Series(stat).to_csv(os.path.join(OUT, "prose_numbers.csv"))
print("\n".join(f"{k:22s} {v:.4f}" for k, v in stat.items()))
