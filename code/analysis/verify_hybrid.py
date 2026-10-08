"""Check CEF-H claims against the retained evaluation and frozen selections."""
from pathlib import Path
import os,json,re,hashlib
import numpy as np
import pandas as pd
root=Path(os.environ.get('CEF_PROJECT_ROOT',Path(__file__).resolve().parent.parent))
r=root/'results/supplementary/round5'
d=pd.read_csv(r/'per_seed.csv');assert len(d)==200
lock=json.loads((r/'frozen_selections.json').read_text());assert len(lock)==40
counts={}
for name,hashvalue in lock.items():
    f=r/'select'/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==hashvalue
    q=json.loads(f.read_text());chosen=q['selected']['validation selected hybrid']
    assert chosen['bAcc']==max(c['bAcc'] for c in q['validation_candidates'])
    assert not q['test_labels_used']
    counts[chosen['kind']]=counts.get(chosen['kind'],0)+1
assert counts=={'graph mixture':18,'output mixture':21,'CuPL fusion':1},counts
for row in d.itertuples():assert abs(np.mean(json.loads(row.recall))-row.bAcc)<1e-12
h=d[d.method=='validation selected hybrid'];old=pd.read_csv(root/'results/supplementary/round4/per_seed_r4.csv')
expected={'HybridMean':100*h.bAcc.mean(),'HybridTopMean':100*h.top1.mean()}
for name,key in [('CEF','Cef'),('CuPL fusion','Cupl')]:
    base=d[d.method==name]
    check=base.merge(old[(old.method==name)&(old.metric=='bAcc')],on=['dataset','backbone','seed']);assert len(check)==40
    assert np.max(np.abs(check.bAcc-check.score))<1e-12
    q=h.merge(base,on=['dataset','backbone','seed'],suffixes=('_h','_b'))
    delta=q.assign(delta=q.bAcc_h-q.bAcc_b).groupby(['dataset','backbone']).delta.mean()*100
    for stat,value in [('Mean',delta.mean()),('Min',delta.min()),('Max',delta.max())]:expected['Hybrid'+key+stat]=value
    expected['Hybrid'+key+'Pairs']=int((delta>0).sum());expected['Hybrid'+key+'Runs']=int((q.bAcc_h>q.bAcc_b).sum())
    if key=='Cupl':expected['HybridTopCuplPairs']=int((q.assign(delta=q.top1_h-q.top1_b).groupby(['dataset','backbone']).delta.mean()>0).sum())
q=h.merge(old[(old.method=='ZLaP')&(old.metric=='bAcc')],on=['dataset','backbone','seed']);expected['HybridZlapMean']=100*(q.bAcc-q.score).mean()
macros=dict(re.findall(r'\\newcommand\{\\R(Hybrid\w+)\}\{([^}]+)\}',(root/'paper3/review_numbers.tex').read_text()))
for k,v in expected.items():assert abs(float(macros[k])-v)<=.00050001,(k,v,macros[k])
print(f'Hybrid verification passed: 40 frozen choices, 200 recall means, 80 reproduced reference scores, {len(expected)} numeric macros and path-selection counts.')
