"""Current CEF comparisons, joined to retained controls by dataset/backbone/seed.

The historical raw ID 'CEF' denotes a constituent branch. The proposed method
is exclusively the frozen round-five ID 'validation selected hybrid'.
"""
from pathlib import Path
import os
import pandas as pd

ROOT = Path(os.environ.get('CEF_PROJECT_ROOT', Path(__file__).resolve().parent.parent))
KEY = ['dataset', 'backbone', 'seed']


def emit(mrt):
    current = pd.read_csv(ROOT/'results/supplementary/round5/per_seed.csv')
    current = current[current.method == 'validation selected hybrid'].copy()
    assert len(current) == 40 and not current.duplicated(KEY).any()
    current = current.rename(columns={'bAcc': 'bacc'})
    controls = pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv')
    controls = controls[controls.metric == 'bAcc'].rename(columns={'score': 'bacc'})
    methods = ['ZLaP', 'DCLIP fusion', 'CuPL fusion', 'MeanPhrase fusion']
    pieces = [controls[controls.method.isin(methods)][KEY+['method', 'bacc']],
              current[KEY+['method', 'bacc']].assign(method='CEF')]
    frame = pd.concat(pieces, ignore_index=True)
    assert len(frame) == 200 and not frame.duplicated(KEY+['method']).any()
    from current_cef_extended_tables import tab,cell
    means=frame.groupby(['dataset','backbone','method']).bacc.mean()
    rows=[]
    for ds in ['wikiart','mp100k']:
        for bb in ['CLIP (OpenAI)','MetaCLIP','EVA02-CLIP','SigLIP']:
            values=[means[ds,bb,m] for m in methods+['CEF']]
            rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),bb]+[cell(x,x==max(values)) for x in values])
    tab(mrt,'tab_matched_fusion.tex',['Dataset','Backbone','ZLaP',r'DCLIP\,+\,fusion',r'CuPL\,+\,fusion',r'Mean phrase\,+\,fusion','CEF'],rows,'llrrrrr')
    for name, prefix in [('DCLIP fusion', 'CurrentDclip'),
                         ('CuPL fusion', 'CurrentCupl'),
                         ('MeanPhrase fusion', 'CurrentMeanPhrase')]:
        mrt.emit_cmp(frame, 'CEF', name, prefix)

    # Constituent results occur only as explicitly named removal ablations.
    for name in ['tab_hybrid.tex', 'tab_hybrid_top1.tex']:
        f = Path(mrt.OUT)/name
        s = f.read_text().replace('phrase-only CEF', 'w/o prototype branch')
        s = s.replace('CuPL fusion', 'w/o phrase branch')
        f.write_text(s)


    # Secondary metric: compare the proposed CEF with fixed vanilla inference,
    # avoiding repeated presentation of historical constituent results.
    vanilla=mrt.load_artist();vanilla=vanilla[vanilla.method=='Vanilla']
    topframe=pd.concat([current.assign(method='CEF'),vanilla.assign(method='Vanilla')])
    mrt.emit_cmp(topframe,'CEF','Vanilla','CurrentTopVanilla',col='top1')
    stats=current.groupby(['dataset','backbone']).top1.agg(['mean','std'])
    vmeans=vanilla.groupby(['dataset','backbone']).top1.mean()
    lines=[r'\begin{tabular}{llrr}',r'\toprule',r'Dataset & Backbone & Vanilla Top-1 & CEF Top-1 \\',r'\midrule']
    for ds in ['wikiart','mp100k']:
        for bb in ['CLIP (OpenAI)','MetaCLIP','EVA02-CLIP','SigLIP']:
            q=stats.loc[ds,bb]
            lines.append(('WikiArt' if ds=='wikiart' else 'MP100k')+' & '+bb+' & '+f'{100*vmeans[ds,bb]:.2f}'+' & '+f'{100*q["mean"]:.2f}'+r'\,$\pm$\,'+f'{100*q["std"]:.2f}'+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    mrt.w('tab_hybrid_top1.tex','\n'.join(lines))

    import current_cef_extended_tables
    current_cef_extended_tables.emit(mrt)

    import table_uncertainty
    table_uncertainty.emit(mrt)

    import presubmission_tables
    presubmission_tables.emit(mrt)
