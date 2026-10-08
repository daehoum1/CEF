"""Verify final CEF figure cases against complete saved scores and frozen R5 runs."""
from pathlib import Path
import os,json,hashlib,re
import numpy as np
import pandas as pd
ROOT=Path(os.environ.get('CEF_PROJECT_ROOT',Path(__file__).resolve().parent.parent))
qdir=ROOT/'results/supplementary/qualitative_final'
r5=ROOT/'results/supplementary/round5'
saved=pd.read_csv(r5/'per_seed.csv')
macrotext=(ROOT/'paper3/review_numbers.tex').read_text()
checks=0
for ds,key in [('wikiart','Wiki'),('mp100k','Mp')]:
    d=np.load(qdir/f'{ds}_scores.npz');m=json.loads((qdir/f'{ds}_metadata.json').read_text())
    y=d['labels'];classes=d['class_names'].tolist();C=len(classes);rows=pd.read_csv(qdir/f'{ds}_examples.csv')
    sf=r5/'select'/f'{ds}_ViT-B-32_openai_42.json';frozen=json.loads(sf.read_text())
    h=hashlib.sha256(sf.read_bytes()).hexdigest()
    assert h==m['selection_sha256']==json.loads((r5/'frozen_selections.json').read_text())[sf.name]
    assert m['selection']==frozen['selected']['validation selected hybrid']
    assert m['protocol']=='artist-disjoint' and m['seed']==42 and m['test_images']==len(y)
    for arm,array in [('CEF','phrase_raw'),('CuPL fusion','prototype_raw'),('validation selected hybrid','cef_raw')]:
        a=d[array];pred=a.argmax(1);rec=[(pred[y==i]==i).mean() for i in range(C)]
        orig=saved[(saved.dataset==ds)&(saved.backbone=='CLIP (OpenAI)')&(saved.seed==42)&(saved.method==arm)].iloc[0]
        assert np.allclose(rec,json.loads(orig.recall),rtol=0,atol=1e-12)
        assert abs(np.mean(rec)-orig.bAcc)<1e-12 and abs((pred==y).mean()-orig.top1)<1e-12
        checks+=1
    # Display normalization preserves the final predicted class.
    raw=np.maximum(d['cef_raw'],0);expected=raw/np.maximum(raw.sum(1,keepdims=True),1e-30)
    assert np.allclose(expected,d['cef'],rtol=0,atol=1e-12)
    assert np.array_equal(d['cef'].argmax(1),d['cef_raw'].argmax(1))
    van,final=d['vanilla'],d['cef'];vp,fp=van.argmax(1),final.argmax(1)
    delta=[]
    for i,gt in enumerate(y):
        rivals=np.arange(C)!=gt
        delta.append((final[i,gt]-final[i,rivals].max())-(van[i,gt]-van[i,rivals].max()))
    delta=np.asarray(delta);good=np.flatnonzero((vp!=y)&(fp==y));bad=np.flatnonzero((vp==y)&(fp!=y))
    assert (len(good),len(bad))==(m['corrected_pool'],m['regressed_pool'])
    ordered=sorted(good,key=lambda i:(-delta[i],int(i)));expected_indices=[];seen=set()
    for i in ordered:
        if y[i] not in seen:expected_indices.append(int(i));seen.add(y[i])
        if len(expected_indices)==3:break
    expected_indices.append(int(min(bad,key=lambda i:(delta[i],int(i)))))
    assert rows.test_index.tolist()==expected_indices
    assert rows.kind.tolist()==['Correction']*3+['Regression']
    for row in rows.itertuples():
        i=int(row.test_index)
        assert (row.gt,row.vanilla,row.cef)==(classes[y[i]],classes[vp[i]],classes[fp[i]])
        assert row.pool_index==d['test_indices'][i]
        assert abs(row.margin_change-delta[i])<1e-12
        assert hashlib.sha256((qdir/row.image).read_bytes()).hexdigest()==row.image_sha256
    for suffix,value in [('Images',len(y)),('Corrected',len(good)),('Regressed',len(bad))]:
        match=re.search(r'\\newcommand\{\\RFinalQual'+key+suffix+r'\}\{([^}]+)\}',macrotext)
        assert match and int(match[1])==value
    checks+=6
    print(f'{ds}: {len(y)} final predictions; {len(good)} corrections, {len(bad)} regressions; four deterministic cases verified.')
print(f'Final CEF qualitative verification passed ({checks} grouped checks).')
