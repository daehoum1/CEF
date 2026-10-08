"""Paired artist-cluster bootstrap, sharing artist multiplicities across all splits/backbones.

Predictions and transductive graphs stay fixed. Percentile intervals describe
conditional evaluation-set uncertainty, not full retraining or unseen-dataset generalization.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np,pandas as pd,json
from reviewer_experiments import OUT,refs,identity
from artist_split import pool_artists
COMPS=['ZLaP','DCLIP fusion','CuPL fusion','Branch selection','DCLIP+CuPL selector']
rows=[]
for ds in ['wikiart','mp100k']:
 jobs=[j for j in refs() if j['dataset']==ds]
 root=Path(__file__).resolve().parent.parent
 cache=np.load(root/jobs[0]['image_cache']);names=cache['class_names'].tolist();artists=pool_artists(ds,names)
 groups,inv=np.unique(artists,return_inverse=True);G=len(groups);C=len(names)
 counts=np.zeros((G,20,C));diff={m:np.zeros_like(counts) for m in COMPS};k=0
 for j in jobs:
  for seed in [42,43,44,45,46]:
   d=np.load(OUT/'predictions'/(identity(j,seed)+'.npz'));idx=d['indices'];y=d['labels'];g=inv[idx]
   np.add.at(counts,(g,k,y),1)
   for m in COMPS:np.add.at(diff[m],(g,k,y),(d['CEF']==y).astype(float)-(d[m]==y).astype(float))
   k+=1
 np.savez_compressed(OUT/(ds+'_bootstrap_counts.npz'),counts=counts,**diff)
 rng=np.random.default_rng(20260916);boot={m:[] for m in COMPS};kept=0;attempted=0
 while kept<2000:
  w=rng.multinomial(G,np.full(G,1/G),size=100).astype(float);n=(w@counts.reshape(G,-1)).reshape(-1,20,C);ok=(n>0).all(axis=(1,2));w=w[ok];n=n[ok];attempted+=100
  if not len(n):continue
  use=min(len(n),2000-kept);n=n[:use];w=w[:use]
  for m in COMPS:
   a=(w@diff[m].reshape(G,-1)).reshape(-1,20,C)
   boot[m].extend((a/n).mean(axis=(1,2)).tolist())
  kept+=use
 point={m:(diff[m].sum(0)/counts.sum(0)).mean() for m in COMPS}
 controls=pd.read_csv(OUT/'controls_per_seed.csv');controls=controls[controls.dataset==ds]
 for m in COMPS:
  q=controls[controls.method.isin(['CEF',m])].pivot(index=['backbone','seed'],columns='method',values='bAcc');delta=q.CEF-q[m];seedmeans=delta.groupby('seed').mean()
  assert abs(point[m]-delta.mean())<1e-12
  lo,hi=np.quantile(boot[m],[.025,.975])
  rows.append(dict(dataset=ds,comparator=m,delta=point[m],seed_sd=seedmeans.std(),lower=lo,upper=hi,bootstrap_samples=2000,attempted=attempted,artists=G))
 print(ds,rows[-5:],flush=True)
pd.DataFrame(rows).to_csv(OUT/'paired_uncertainty.csv',index=False)
(OUT/'bootstrap_protocol.json').write_text(json.dumps(dict(seed=20260916,replicates=2000,unit='artist',shared_across='All four backbones and all five overlapping test splits within dataset',conditioning='Saved predictions and graph, rejecting resamples omitting a class in any split',interval='Percentile 2.5 and 97.5; no method reselection or graph reconstruction',comparators=COMPS),indent=2)+'\n')
