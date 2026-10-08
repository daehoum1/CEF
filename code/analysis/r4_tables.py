"""Review-round-four tables and effect sizes, generated from saved results."""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('CEF_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
R4 = ROOT/'results/supplementary/round4'
KEY = ['dataset', 'backbone', 'seed']


def load():
    raw = pd.read_csv(R4/'per_seed_r4.csv')
    values = raw.pivot(index=KEY+['method'],columns='metric',values='score').reset_index()
    values = values.rename(columns={'bAcc':'bacc','Top-1':'top1'})
    ext = values[values.method.str.startswith('CEF external orig_g')]
    average = ext.groupby(KEY)[['bacc','top1']].mean().reset_index()
    average['method'] = 'CEF external draws'
    return pd.concat([values,average],ignore_index=True)


def emit(mrt, per_seed, r3):
    r4 = load()
    # Reference-dictionary and independently generated-dictionary results are
    # both visible in the main table. The latter averages accuracies, not scores.
    g=per_seed.groupby(['dataset','backbone','method'])
    mt,mb,sb=g.top1.mean(),g.bacc.mean(),g.bacc.std(ddof=1)
    draws=r3[r3.method.str.match(r'CEF \[orig_g[0-4]\]')]
    dg=draws.groupby(['dataset','backbone','method'])[['bacc','top1']].mean()
    dm=dg.groupby(['dataset','backbone']).mean()
    ds=dg.groupby(['dataset','backbone']).std(ddof=1)
    arms=[('Vanilla VLM','Vanilla'),('CGPR','CGPR'),('ZLaP','ZLaP'),
          ('CEF (reference dict.)','CEF'),(r'CEF (5 new dicts.)$^\dagger$','draws')]
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Method & \multicolumn{2}{c}{WikiArt} & \multicolumn{2}{c}{MultitaskPainting100k} \\',
           r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',r' & Top-1 & bAcc & Top-1 & bAcc \\']
    for bb in mrt.BB:
        lines += [r'\midrule',rf'\multicolumn{{5}}{{l}}{{\textit{{{bb}}}}} \\']
        for label,method in arms:
            cells=[]
            for dataset,_ in mrt.DS:
                if method=='draws':
                    t,b,sd=dm.loc[(dataset,bb),'top1'],dm.loc[(dataset,bb),'bacc'],ds.loc[(dataset,bb),'bacc']
                else:
                    k=dataset,bb,method
                    t,b,sd=mt[k],mb[k],sb[k]
                cells += [mrt.f4(t),rf'{mrt.f4(b)}\,$\pm$\,{mrt.f4(sd)}']
            lines.append(label+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    mrt.w('tab_main.tex','\n'.join(lines))
    mrt.emit_cmp(r3,'CEF [orig]','ZLaP','GenTopZlap',col='top1')
    methods=['ZLaP','DCLIP fusion','CuPL fusion','MeanPhrase fusion','CEF']
    headers=['ZLaP',r'DCLIP\,+\,fusion',r'CuPL\,+\,fusion',r'Mean phrase\,+\,fusion','CEF']
    mrt.table(mrt.means(r4),methods,'tab_matched_fusion.tex',headers)
    for method,key in [('DCLIP','Dclip'),('CuPL','Cupl'),('MeanPhrase','MeanPhrase')]:
        mrt.emit_cmp(r4,'CEF',method+' fusion','Matched'+key)
        mrt.emit_cmp(r4,method+' fusion','ZLaP','Matched'+key+'Zlap')
    methods=['ZLaP','CEF','CEF external full','CEF external draws']
    headers=['ZLaP (tuned)','CEF (tuned)',r'External cfg, ref. dict.',r'External cfg, 5 dicts.']
    mrt.table(mrt.means(r4),methods,'tab_external_transfer.tex',headers,bold_best=False)
    mrt.emit_cmp(r4,'CEF external full','ZLaP','ExternalRef')
    mrt.emit_cmp(r4,'CEF external draws','ZLaP','ExternalDraw')
    mrt.emit_cmp(r4,'CEF','CEF external full','ExternalCost')
    frame=pd.read_csv(R4/'per_seed_r4.csv')
    nf=frame[frame.method.isin([x+' fusion' for x in ['DCLIP','CuPL','MeanPhrase']])]
    mrt.M('MatchedEvalsMin',str(int(nf.val_evals.min())))
    mrt.M('MatchedEvalsMax',str(int(nf.val_evals.max())))
    protocol=json.loads((R4/'protocol.json').read_text())
    graph,evidence=protocol['fixed_config']
    for key,value in zip(['ExternalK','ExternalGamma','ExternalAlpha'],graph):
        mrt.M(key,f'{value:g}')
    for key,value in zip(['ExternalTau','ExternalW'],evidence[:2]):
        mrt.M(key,f'{value:g}')
    mrt.M('ExternalDebias','on' if evidence[2] else 'off')
    mrt.M('ExternalBase','softmax' if evidence[3]=='sm' else 'cosine')
    summary=[]
    for method in ['DCLIP fusion','CuPL fusion','MeanPhrase fusion','CEF external full','CEF external draws']:
        a,b=('CEF',method) if method.endswith('fusion') else (method,'ZLaP')
        diff=mrt.pair_mean_diffs(r4,a,b)
        wins,total=mrt.seed_level(r4,a,b)
        summary.append(dict(a=a,b=b,mean_pts=100*diff.mean(),min_pts=100*diff.min(),
                            max_pts=100*diff.max(),positive_pairs=int((diff>0).sum()),
                            positive_runs=wins,runs=total))
    pd.DataFrame(summary).to_csv(R4/'effect_sizes.csv',index=False)

    effect_path=Path(mrt.OUT)/'tab_effects.tex'
    text=effect_path.read_text()
    extra=[]
    labels=[(r'CEF $-$ DCLIP fusion','CEF','DCLIP fusion'),
            (r'CEF $-$ CuPL fusion','CEF','CuPL fusion'),
            (r'CEF $-$ MeanPhrase fusion','CEF','MeanPhrase fusion')]
    for label,a,b in labels:
        diff=mrt.pair_mean_diffs(r4,a,b)
        win,total=mrt.seed_level(r4,a,b)
        extra.append(f'{label} & {mrt.pts2(diff.mean())} & {mrt.pts2(diff.min())} to {mrt.pts2(diff.max())} & '
                     f'{mrt.pts2(diff.xs("wikiart").mean())} & {mrt.pts2(diff.xs("mp100k").mean())} & '
                     f'{int((diff>0).sum())}/{len(diff)} & {win}/{total} '+r' \\')
    text=text.replace(r'\bottomrule',r'\midrule'+'\n'+'\n'.join(extra)+'\n'+r'\bottomrule')
    mrt.w('tab_effects.tex',text)
