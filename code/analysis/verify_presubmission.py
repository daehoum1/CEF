"""Check new scores, frozen selections and paired conditional intervals from archived inputs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,re
ROOT=Path(os.environ.get('CEF_PROJECT_ROOT',Path(__file__).resolve().parent.parent));out=ROOT/'results/supplementary/reviewer_revision'
d=pd.read_csv(out/'controls_per_seed.csv');l=pd.read_csv(out/'labels_per_seed.csv');u=pd.read_csv(out/'paired_uncertainty.csv');r5=pd.read_csv(ROOT/'results/supplementary/round5/per_seed.csv')
key=['dataset','backbone','seed']
assert len(d)==440 and len(l)==320 and len(u)==10
assert not d.duplicated(key+['method']).any() and not l.duplicated(key+['method','fraction']).any()
for frame in [d,l]:
 for _,r in frame.iterrows():assert abs(r.bAcc-np.mean(json.loads(r.recall)))<1e-12
for a in [d[d.method=='CEF'],l[(l.method=='CEF')&(l.fraction==1.)]]:
 q=a.merge(r5[r5.method=='validation selected hybrid'],on=key);assert len(q)==40
 assert np.max(abs(q.bAcc_x-q.bAcc_y))<1e-12 and np.max(abs(q.top1_x-q.top1_y))<1e-12
r4=pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv');r4=r4[r4.metric=='bAcc']
for m in ['ZLaP','DCLIP fusion','CuPL fusion']:
 q=d[d.method==m].merge(r4[r4.method==m],on=key);assert len(q)==40;assert np.max(abs(q.bAcc-q.score))<1e-12
for part in ['controls','labels']:
 for f in (out/part).glob('*.json'):
  value=json.loads(f.read_text());suffix='_controls' if part=='controls' else '_labels_'+str(value['rows'][0]['fraction'])
  stem=f.stem if part=='controls' else f.stem.rsplit('_',1)[0]
  sf=out/'selections'/(stem+suffix+'.json');assert hashlib.sha256(sf.read_bytes()).hexdigest()==value['selection_sha256']
  selection=json.loads(sf.read_text());assert selection.get('test_labels_used',False)==False
# Nested label sets are disjoint from the fixed test pool.
for pf in (out/'predictions').glob('*.npz'):
 pred=np.load(pf);previous=set()
 for frac in [.1,.25,.5,1.]:
  s=json.loads((out/'selections'/(pf.stem+'_labels_'+str(frac)+'.json')).read_text());ii=set(s['label_indices'])
  assert previous<=ii and not ii.intersection(pred['indices']);previous=ii
# Recompute all paired intervals from artist sufficient statistics, no image files needed.
for ds in ['wikiart','mp100k']:
 arrays=np.load(out/(ds+'_bootstrap_counts.npz'));counts=arrays['counts'];G=counts.shape[0];rng=np.random.default_rng(20260916)
 comps=list(u[u.dataset==ds].comparator);boot={m:[] for m in comps};kept=0
 while kept<2000:
  w=rng.multinomial(G,np.full(G,1/G),size=100).astype(float);n=(w@counts.reshape(G,-1)).reshape(-1,*counts.shape[1:]);ok=(n>0).all(axis=(1,2));n=n[ok];w=w[ok];use=min(len(n),2000-kept);n=n[:use];w=w[:use]
  for m in comps:
   a=(w@arrays[m].reshape(G,-1)).reshape(-1,*counts.shape[1:]);boot[m].extend((a/n).mean(axis=(1,2)))
  kept+=use
 for m in comps:
  q=u[(u.dataset==ds)&(u.comparator==m)].iloc[0];lo,hi=np.quantile(boot[m],[.025,.975]);assert abs(lo-q.lower)<1e-12 and abs(hi-q.upper)<1e-12
  x=d[(d.dataset==ds)&(d.method.isin(['CEF',m]))].pivot(index=['backbone','seed'],columns='method',values='bAcc');delta=x.CEF-x[m]
  assert abs(delta.mean()-q.delta)<1e-12 and abs(delta.groupby('seed').mean().std()-q.seed_sd)<1e-12
# All newly reported macros trace back to exact scores.
text=(ROOT/'paper3/review_numbers.tex').read_text();mac=dict(re.findall(r'\\newcommand\{\\R([^}]+)\}\{([^}]+)\}',text));expected={}
for m,k in [('Branch selection','BranchSelect'),('DCLIP+CuPL selector','Alternative')]:
 q=d[d.method.isin(['CEF',m])].pivot(index=key,columns='method',values='bAcc');de=q.CEF-q[m];expected['Pre'+k+'Mean']=f'{100*de.mean():.2f}';expected['Pre'+k+'Pairs']=str(int((de.groupby(['dataset','backbone']).mean()>0).sum()))
q=d.pivot(index=key,columns='method',values='top1').groupby(['dataset','backbone']).mean()
for m,k in [('ZLaP','Zlap'),('DCLIP fusion','Dclip'),('CuPL fusion','Cupl')]:expected['PreTop'+k+'Pairs']=str(int((q.CEF>q[m]).sum()))
for ds,k in [('wikiart','Wiki'),('mp100k','Mp')]:
 q=l[(l.dataset==ds)&(l.fraction==.1)].groupby('method').bAcc.mean();expected['PreBudget'+k+'Gain']=f'{100*(q.CEF-q.ZLaP):.2f}'
q=d.pivot(index=key,columns='method',values='bAcc');expected['PreSourceGain']=f'{100*(q["Source CEF"]-q["Source ZLaP"]).mean():.2f}'
for k,v in expected.items():assert mac[k]==v,(k,v,mac[k])
print('Presubmission verification passed: 40 controls, 160 label-budget settings, 10 reconstructed artist-bootstrap intervals, 10 macros; original CEF and external baselines reproduced.')
