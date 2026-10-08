"""Tables for CEF, generated only from saved validation selections and scores."""
from pathlib import Path
import os,json
import pandas as pd
import numpy as np
ROOT=Path(os.environ.get('CEF_PROJECT_ROOT',Path(__file__).resolve().parent.parent))
def emit(mrt):
    r=ROOT/'results/supplementary/round5';d=pd.read_csv(r/'per_seed.csv')
    old=pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv')
    h=d[d.method=='validation selected hybrid'];assert len(h)==40
    g=d.groupby(['dataset','backbone','method'])
    means=g.bAcc.mean();sd=g.bAcc.std();top=g.top1.mean()
    mrt.M('HybridMean',f'{100*h.bAcc.mean():.3f}')
    for name,key in [('CEF','Cef'),('CuPL fusion','Cupl')]:
        q=h.merge(d[d.method==name],on=['dataset','backbone','seed'],suffixes=('_h','_b'))
        delta=q.assign(delta=q.bAcc_h-q.bAcc_b).groupby(['dataset','backbone']).delta.mean()*100
        for stat,value in [('Mean',delta.mean()),('Min',delta.min()),('Max',delta.max())]:mrt.M('Hybrid'+key+stat,f'{value:.3f}')
        mrt.M('Hybrid'+key+'Pairs',str(int((delta>0).sum())));mrt.M('Hybrid'+key+'Runs',str(int((q.bAcc_h>q.bAcc_b).sum())))
    z=old[(old.method=='ZLaP')&(old.metric=='bAcc')]
    q=h.merge(z,on=['dataset','backbone','seed']);mrt.M('HybridZlapMean',f'{100*(q.bAcc-q.score).mean():.3f}')
    # Report original performance under identical bAcc-selected branch configurations.
    lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Dataset & Backbone & phrase-only CEF & CuPL fusion & Output mix & Graph mix & \textbf{CEF} \\',r'\midrule']
    for ds in ['wikiart','mp100k']:
        for i,b in enumerate(['CLIP (OpenAI)','MetaCLIP','EVA02-CLIP','SigLIP']):
            vals=[100*means[ds,b,m] for m in ['CEF','CuPL fusion','output mixture','graph mixture','validation selected hybrid']]
            cells=[f'{x:.2f}' for x in vals[:-1]]+[f'{vals[-1]:.2f}'+r'\,$\pm$\,'+f'{100*sd[ds,b,"validation selected hybrid"]:.2f}']
            lines.append(('WikiArt' if ds=='wikiart' else 'MP100k')+' & '+b+' & '+' & '.join(cells)+r' \\')
        lines.append(r'\midrule' if ds=='wikiart' else r'\bottomrule')
    lines.append(r'\end{tabular}');mrt.w('tab_hybrid.tex','\n'.join(lines))
    # Same bAcc-selected configurations: do not mix independently Top-1-tuned baselines.
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Dataset & Backbone & phrase-only CEF & CuPL fusion & CEF \\',r'\midrule']
    for ds in ['wikiart','mp100k']:
        for b in ['CLIP (OpenAI)','MetaCLIP','EVA02-CLIP','SigLIP']:
            lines.append(('WikiArt' if ds=='wikiart' else 'MP100k')+' & '+b+' & '+' & '.join(f'{100*top[ds,b,m]:.2f}' for m in ['CEF','CuPL fusion','validation selected hybrid'])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}']);mrt.w('tab_hybrid_top1.tex','\n'.join(lines))
    q=h.merge(d[d.method=='CuPL fusion'],on=['dataset','backbone','seed'],suffixes=('_h','_b'))
    mrt.M('HybridTopCuplPairs',str(int((q.assign(delta=q.top1_h-q.top1_b).groupby(['dataset','backbone']).delta.mean()>0).sum())))
    mrt.M('HybridTopMean',f'{100*h.top1.mean():.3f}')

    # Main external comparison: bAcc-selected results only, on identical splits.
    base = mrt.load_artist()
    arms = ['Vanilla', 'CGPR', 'ZLaP', 'DCLIP', 'CuPL', 'DCLIP + ZLaP', 'CuPL + ZLaP']
    bm = base.groupby(['dataset', 'backbone', 'method']).bacc.mean()
    lines = [r'\begin{tabular}{llrrrrrrrr}', r'\toprule',
             r'Dataset & Backbone & Vanilla & CGPR & ZLaP & DCLIP & CuPL & DCLIP\,+\,LP & CuPL\,+\,LP & \textbf{CEF} \\', r'\midrule']
    for ds in ['wikiart', 'mp100k']:
        for b in ['CLIP (OpenAI)', 'MetaCLIP', 'EVA02-CLIP', 'SigLIP']:
            vals = [100*bm[ds, b, arm] for arm in arms]
            final = 100*means[ds, b, 'validation selected hybrid']
            assert final > max(vals), (ds, b, vals, final)
            cells = [f'{v:.2f}' for v in vals]
            cells.append(r'\textbf{'+f'{final:.2f}'+r'}\,$\pm$\,'+f'{100*sd[ds,b,"validation selected hybrid"]:.2f}')
            lines.append(('WikiArt' if ds=='wikiart' else 'MP100k')+' & '+b+' & '+' & '.join(cells)+r' \\')
        lines.append(r'\midrule' if ds=='wikiart' else r'\bottomrule')
    lines.append(r'\end{tabular}')
    mrt.w('tab_cef_main.tex', '\n'.join(lines))
