"""Tables for complete-method interventions, never historical branch substitutions."""
from current_cef_tables import ROOT,KEY
from pathlib import Path
import json,re
import numpy as np
import pandas as pd

DS=['wikiart','mp100k']; BB=['CLIP (OpenAI)','MetaCLIP','EVA02-CLIP','SigLIP']
def cell(x,bold=False):
    s=f'{100*x:.2f}'
    return r'\textbf{'+s+'}' if bold else s
def tab(mrt,name,headers,rows,align=None):
    text=[r'\begin{tabular}{'+(align or 'l'+'r'*(len(headers)-1))+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    text += [' & '.join(row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}'];mrt.w(name,'\n'.join(text))
def keyed(d):return d.set_index(KEY).bacc
def compare(mrt,a,b,prefix):
    q=pd.concat([a[KEY+['bacc']].assign(method='a'),b[KEY+['bacc']].assign(method='b')])
    return mrt.emit_cmp(q,'a','b',prefix)

def emit(mrt):
    path=ROOT/'results/supplementary/current_cef/per_seed.csv'
    if not path.exists():return
    new=pd.read_csv(path).rename(columns={'bAcc':'bacc'})
    art=new[new.dataset.isin(DS)]
    required=['rand_chars','rand_words','shuffle_owner','filtered','no_debias','no_graph','image_level']+[f'orig_g{i}' for i in range(5)]
    if any(len(art[art.variant==v])!=40 for v in required):return
    ref=pd.read_csv(ROOT/'results/supplementary/round5/per_seed.csv').rename(columns={'bAcc':'bacc'})
    cef=ref[ref.method=='validation selected hybrid'];cm=keyed(cef).groupby(['dataset','backbone']).mean()
    base=mrt.load_artist();z=base[base.method=='ZLaP'];zm=keyed(z).groupby(['dataset','backbone']).mean()
    means=art.groupby(['dataset','backbone','variant']).bacc.mean()
    rows=[]
    for ds in DS:
        for bb in BB:
            vals=[zm[ds,bb],cm[ds,bb]]+[means[ds,bb,v] for v in ['rand_chars','rand_words','shuffle_owner']]
            rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),bb]+[cell(v,v==max(vals)) for v in vals])
    tab(mrt,'tab_controls.tex',['Dataset','Backbone','ZLaP','CEF','Random chars','Random words','Permuted owners'],rows,'llrrrrr')
    for v,k in [('rand_chars','Chars'),('rand_words','Words'),('shuffle_owner','Owners'),('filtered','Filter'),('no_debias','Debias'),('no_graph','NoGraph')]:
        compare(mrt,cef,art[art.variant==v],'Current'+k)
    variants=[('CEF',cef)]
    variants += [(label,art[art.variant==v]) for label,v in [('w/o label propagation','no_graph'),('w/o centering','no_debias'),('w/ filtered dictionary','filtered')]]
    matrix=np.array([[keyed(d).groupby(['dataset','backbone']).mean()[ds,bb] for ds in DS for bb in BB] for _,d in variants])
    rows=[[label]+[cell(v,v==matrix[:,i].max()) for i,v in enumerate(matrix[r])] for r,(label,_) in enumerate(variants)]
    headers=['Variant']+[f'{ds} / {b}' for ds in ['WA','MP'] for b in ['CLIP','Meta','EVA','Sig']]
    tab(mrt,'tab_ablation.tex',headers,rows)
    draws=art[art.variant.str.startswith('orig_g')]
    gm=draws.groupby(['dataset','backbone','variant']).bacc.mean()
    gs=gm.groupby(['dataset','backbone']).agg(['mean','std','min','max'])
    rows=[]
    for ds in DS:
        for bb in BB:
            q=gs.loc[ds,bb]
            rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),bb,cell(zm[ds,bb]),cell(cm[ds,bb]),cell(q['mean'])+r'\,$\pm$\,'+cell(q['std']),cell(q['min'])+'--'+cell(q['max'])])
    tab(mrt,'tab_r3_gens.tex',['Dataset','Backbone','ZLaP','CEF (reference)','CEF (5 draws)','Draw range'],rows,'llrrrr')
    average=draws.groupby(KEY).bacc.mean().reset_index()
    compare(mrt,average,z,'CurrentGenZlap');compare(mrt,cef,average,'CurrentRefDraw')
    for k,v in [('SdMin',gs['std'].min()),('SdMax',gs['std'].max()),('SdMean',gs['std'].mean())]:mrt.M('CurrentGen'+k,f'{100*v:.2f}')
    delta=gm-gm.index.droplevel('variant').map(zm).to_numpy()
    mrt.M('CurrentGenPairWins',str(int((delta>0).sum())))
    # Image-level protocol: retain the comparator's stated per-metric selections.
    img=mrt.load_image();ni=art[art.variant=='image_level'].assign(method='CEF')
    parts=[img[img.method.isin(['Vanilla','CGPR','ZLaP'])],ni]
    im=pd.concat(parts).groupby(['dataset','backbone','method'])[['top1','bacc']].mean()
    rows=[]
    for ds in DS:
        for bb in BB:
            rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),bb]+[cell(im.loc[(ds,bb,m),metric]) for m in ['Vanilla','CGPR','ZLaP','CEF'] for metric in ['top1','bacc']])
    tab(mrt,'tab_main_image.tex',['Dataset','Backbone']+[m+' '+metric for m in ['Vanilla','CGPR','ZLaP','CEF'] for metric in ['Top-1','bAcc']],rows,'llrrrrrrrr')
    image_path=Path(mrt.OUT)/'tab_main_image.tex'
    image_lines=image_path.read_text().splitlines()
    image_lines[2:3]=[
        r'Dataset & Backbone & \multicolumn{2}{c}{Vanilla} & \multicolumn{2}{c}{CGPR} & \multicolumn{2}{c}{ZLaP} & \multicolumn{2}{c}{CEF} \\',
        r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
        r' & & Top-1 & bAcc & Top-1 & bAcc & Top-1 & bAcc & Top-1 & bAcc \\']
    image_path.write_text('\n'.join(image_lines)+'\n')
    for m,k in [('ZLaP','Zlap'),('CGPR','Cgpr')]:compare(mrt,ni,img[img.method==m].groupby(KEY).bacc.mean().reset_index(),'CurrentImage'+k)
    # Detailed effect-size summary now compares CEF itself.
    r4=pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv');r4=r4[r4.metric=='bAcc'].rename(columns={'score':'bacc'})
    comps=[('ZLaP',z),('CGPR',base[base.method=='CGPR'])]
    comps += [(m,r4[r4.method==m]) for m in ['DCLIP fusion','CuPL fusion','MeanPhrase fusion']]
    rows=[]
    for name,b in comps:
        d=keyed(cef)-keyed(b);pairs=d.groupby(['dataset','backbone']).mean()*100
        rows.append(['CEF $-$ '+name,f'{pairs.mean():+.2f}',f'{pairs.min():+.2f} to {pairs.max():+.2f}',f'{pairs.xs("wikiart").mean():+.2f}',f'{pairs.xs("mp100k").mean():+.2f}',f'{(pairs>0).sum()}/8',f'{(d>0).sum()}/40'])
    tab(mrt,'tab_effects.tex',['Comparison','Mean','Pair range','WikiArt','MP100k','Pairs','Runs'],rows,'lrrrrrr')
    probes=new[new.dataset.isin(['dtd','eurosat','aircraft','pets_s5'])]
    if len(probes)==160:
        old=pd.read_csv(ROOT/'results/supplementary/round3/probe_per_seed.csv').rename(columns={'domain':'dataset'})
        rows=[]
        for ds,label in [('wikiart','WikiArt'),('mp100k','MP100k'),('dtd','DTD'),('eurosat','EuroSAT'),('aircraft','Aircraft'),('pets_s5','Oxford-IIIT Pet')]:
            b=base[base.dataset==ds] if ds in DS else old[old.dataset==ds]
            r=cef[cef.dataset==ds] if ds in DS else probes[(probes.dataset==ds)&(probes.variant=='reference')]
            random=art[(art.dataset==ds)&(art.variant=='rand_words')] if ds in DS else probes[(probes.dataset==ds)&(probes.variant=='rand_words')]
            van=b[b.method=='Vanilla'].bacc.mean();zl=b[b.method=='ZLaP'].bacc.mean();c=r.bacc.mean();rw=random.bacc.mean()
            rows.append([label,cell(van),cell(zl),cell(c),cell(rw),f'{100*(c-zl):+.2f}',f'{100*(c-rw):+.2f}'])
            if ds not in DS:
                key={'dtd':'Dtd','eurosat':'Euro','aircraft':'Air','pets_s5':'Pet'}[ds]
                compare(mrt,r,b[b.method=='ZLaP'],'CurrentProbe'+key+'Zlap');compare(mrt,r,random,'CurrentProbe'+key+'Random')
        tab(mrt,'tab_r3_probe.tex',['Dataset','Vanilla','ZLaP','CEF','Random words','CEF $-$ ZLaP','Real $-$ random'],rows)
    transfer=ROOT/'results/supplementary/current_cef/transfer_per_seed.csv'
    if transfer.exists():
        tr=pd.read_csv(transfer).rename(columns={'bAcc':'bacc'})
        if all(len(tr[tr.setting==v])==40 for v in ['global_mode','leave_pair_out']):
            tm=tr.groupby(['dataset','backbone','setting']).bacc.mean();rows=[]
            for ds in DS:
                for bb in BB:rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),bb,cell(zm[ds,bb]),cell(tm[ds,bb,'global_mode']),cell(tm[ds,bb,'leave_pair_out']),cell(cm[ds,bb])])
            tab(mrt,'tab_transfer.tex',['Dataset','Backbone','ZLaP','CEF (global)','CEF (leave-pair-out)','CEF (tuned)'],rows,'llrrrr')
        ext=tr[tr.setting=='external']
        if len(ext)==240:
            er=ext[ext.variant=='reference'];ed=ext[ext.variant.str.startswith('orig_g')].groupby(KEY).bacc.mean().reset_index()
            erm=keyed(er).groupby(['dataset','backbone']).mean();edm=keyed(ed).groupby(['dataset','backbone']).mean();rows=[]
            for ds in DS:
                for bb in BB:rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),bb,cell(zm[ds,bb]),cell(cm[ds,bb]),cell(erm[ds,bb]),cell(edm[ds,bb])])
            tab(mrt,'tab_external_transfer.tex',['Dataset','Backbone','ZLaP (tuned)','CEF (tuned)','CEF (external, ref.)','CEF (external, 5 draws)'],rows,'llrrrr')
            compare(mrt,er,z,'CurrentExternalRef');compare(mrt,ed,z,'CurrentExternalDraw');compare(mrt,cef,er,'CurrentExternalCost')
    cost=ROOT/'results/supplementary/current_cef/cost.csv'
    if cost.exists():
        cp=pd.read_csv(cost);rows=[]
        for ds in DS:
            for m in ['ZLaP','CEF']:
                q=cp[(cp.dataset==ds)&(cp.method==m)];assert len(q)==4
                rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),m,f'{q.select_s.mean():.1f}',f'{q.test_s.mean():.2f}',f'{q.peak_gb.mean():.2f}'])
                if m=='CEF':
                    for key,col in [('Select','select_s'),('Test','test_s'),('Mem','peak_gb')]:mrt.M('CurrentCost'+('Wiki' if ds=='wikiart' else 'Mp')+key,f'{q[col].mean():.2f}')
        tab(mrt,'tab_cost.tex',['Dataset','Method','Selection (s)','Test (s)','Peak memory (GB)'],rows,'llrrr')

    if cost.exists():
        # Which tables the manuscript includes: from the released listing when the
        # LaTeX sources are not distributed, otherwise from the sources themselves.
        listing=ROOT/'paper3/referenced_tables.txt'
        active=set()
        if listing.exists():
            active={l.strip()[:-4] if l.strip().endswith('.tex') else l.strip()
                    for l in listing.read_text().splitlines() if l.strip() and not l.startswith('#')}
        else:
            for source in (ROOT/'paper3/tex').glob('*.tex'):
                active.update(re.findall(r'\\CEFTable\{tables/([^}]+)\}',source.read_text()))
        assert len(active)==15
        for source in Path(mrt.OUT).glob('*.tex'):
            if source.stem not in active:source.unlink()
