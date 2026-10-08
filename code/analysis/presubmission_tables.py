"""Additional presubmission evidence. Source inputs are immutable saved scores."""
from pathlib import Path
import json
import pandas as pd
from current_cef_tables import ROOT
from current_cef_extended_tables import tab,DS,BB

def emit(mrt):
 out=ROOT/'results/supplementary/reviewer_revision'
 d=pd.read_csv(out/'controls_per_seed.csv');labels=pd.read_csv(out/'labels_per_seed.csv');u=pd.read_csv(out/'paired_uncertainty.csv')
 assert len(d)==440 and len(labels)==320 and len(u)==10
 def fmt(a,stack=False):
  a=100*a;mean=f'{a.mean():.2f}';sd=f'{a.std(ddof=1):.2f}'
  return (r'\shortstack[r]{'+mean+r'\\$\pm$\,'+sd+'}') if stack else mean+r'\,$\pm$\,'+sd
 def arr(ds,bb,m,metric='bAcc'):return d[(d.dataset==ds)&(d.backbone==bb)&(d.method==m)][metric]
 def perpair(methods,metric='bAcc',stack=False):
  return [[('WikiArt' if ds=='wikiart' else 'MP100k'),bb]+[fmt(arr(ds,bb,m,metric),stack) for m in methods] for ds in DS for bb in BB]
 tab(mrt,'tab_hybrid.tex',['Dataset','Backbone','w/o prototype','w/o phrase','Branch selection','DCLIP+CuPL','CEF'],perpair(['Phrase','Prototype','Branch selection','DCLIP+CuPL selector','CEF'],stack=False),'llrrrrr')
 for m,key in [('Branch selection','BranchSelect'),('DCLIP+CuPL selector','Alternative')]:
  p=d[d.method.isin(['CEF',m])].pivot(index=['dataset','backbone','seed'],columns='method',values='bAcc');delta=p.CEF-p[m]
  mrt.M('Pre'+key+'Mean',f'{100*delta.mean():.2f}');mrt.M('Pre'+key+'Pairs',str(int((delta.groupby(['dataset','backbone']).mean()>0).sum())))
 # Top-1 under bAcc selection for all graph-based methods, plus vanilla.
 base=mrt.load_artist();rows=[]
 for ds in DS:
  for bb in BB:
   v=base[(base.dataset==ds)&(base.backbone==bb)&(base.method=='Vanilla')].top1
   rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),bb,fmt(v)]+[fmt(arr(ds,bb,m,'top1')) for m in ['ZLaP','DCLIP fusion','CuPL fusion','CEF']])
 tab(mrt,'tab_hybrid_top1.tex',['Dataset','Backbone','Vanilla','ZLaP','DCLIP fusion','CuPL fusion','CEF'],rows,'llrrrrr')
 p=d.pivot(index=['dataset','backbone','seed'],columns='method',values='top1').groupby(['dataset','backbone']).mean()
 for m,key in [('ZLaP','Zlap'),('DCLIP fusion','Dclip'),('CuPL fusion','Cupl')]:mrt.M('PreTop'+key+'Pairs',str(int((p.CEF>p[m]).sum())))
 # Replace redundant configuration reuse with controlled validation-label budgets.
 rows=[]
 for ds in DS:
  for fraction in [.1,.25,.5,1.]:
   q=labels[(labels.dataset==ds)&(labels.fraction==fraction)]
   rows.append([('WikiArt' if ds=='wikiart' else 'MP100k'),str(int(100*fraction))+r'\%',f'{q.n_labeled.min()}--{q.n_labeled.max()}']+[fmt(q[q.method==m].groupby('seed').bAcc.mean()) for m in ['ZLaP','CEF']])
 tab(mrt,'tab_transfer.tex',['Dataset','Labels used','Labeled images','ZLaP','CEF'],rows,'llrrr')
 for ds,k in [('wikiart','Wiki'),('mp100k','Mp')]:
  q=labels[(labels.dataset==ds)&(labels.fraction==.1)].groupby('method').bAcc.mean()
  mrt.M('PreBudget'+k+'Gain',f'{100*(q.CEF-q.ZLaP):.2f}')
 tab(mrt,'tab_external_transfer.tex',['Dataset','Backbone','ZLaP (source)','CEF (source)','ZLaP (target)','CEF (target)'],perpair(['Source ZLaP','Source CEF','ZLaP','CEF']),'llrrrr')
 p=d.pivot(index=['dataset','backbone','seed'],columns='method',values='bAcc');mrt.M('PreSourceGain',f'{100*(p["Source CEF"]-p["Source ZLaP"]).mean():.2f}')
 # Paired uncertainty; seed SD after backbone averaging; shared-artist bootstrap.
 rows=[]
 for m in ['ZLaP','DCLIP fusion','CuPL fusion','Branch selection','DCLIP+CuPL selector']:
  row=['CEF $-$ '+m.replace(' selector','')]
  for ds in DS:
   r=u[(u.dataset==ds)&(u.comparator==m)].iloc[0]
   row += [f'{100*r.delta:.2f}'+r'\,$\pm$\,'+f'{100*r.seed_sd:.2f}',f'[{100*r.lower:.2f}, {100*r.upper:.2f}]']
  rows.append(row)
 tab(mrt,'tab_effects.tex',['Comparison','WikiArt difference','95\% interval','MP100k difference','95\% interval'],rows,'lrrrr')
 # Convert diagnostic accuracy to percent, preserving four-decimal input precision.
 p=Path(mrt.OUT)/'tab_attribution.tex';s=p.read_text();import re
 s=re.sub(r'(0\.\d{4})\\,\$\\pm\$\\,(0\.\d{4})',lambda m:f'{100*float(m[1]):.2f}'+r'\,$\pm$\,'+f'{100*float(m[2]):.2f}',s);p.write_text(s)

 # Keep wide one-line score tables at the standard table font size.
 for name in ['tab_cef_main.tex','tab_ablation.tex','tab_main_image.tex']:
  f=Path(mrt.OUT)/name
  f.write_text(r'\begingroup'+ '\n'+r'\setlength{\tabcolsep}{2pt}'+ '\n'+f.read_text()+r'\endgroup'+'\n')
