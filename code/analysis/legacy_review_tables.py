"""Regenerate the original diagnosis and image-level comparison tables."""
import numpy as np
import pandas as pd

def emit(mrt):
    w, BB, DS, f4 = mrt.w, mrt.BB, mrt.DS, mrt.f4
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
         r"Method & \multicolumn{2}{c}{WikiArt} & \multicolumn{2}{c}{MultitaskPainting100k} \\",
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
    w("tab_main_image.tex", "\n".join(L))

