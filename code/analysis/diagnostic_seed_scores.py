"""Recover Table 2 per-seed scores with the original fixed CGPR configuration."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from cgpr_lab import *
from concept_hypergraph import _row_normalize
from scipy.sparse import eye
import pandas as pd
from pathlib import Path
rows=[]
for ds in ['wikiart','mp100k']:
 for tag,bb in BACKBONES:
  p=Pool(ds,tag);cfg=p.cgpr_cfg['kNN+Concept Hypergraph + Uncertainty']
  for seed in [42,43,44,45,46]:
   _,ti=p.split(seed);e,Y,l=p.embs[ti],p.Y[ti],p.labels[ti]
   W=build_knn_graph(e,k=p.sp_cfg['k_graph'],metric='cosine')
   P=_row_normalize(W+cfg['alpha']*eye(len(ti),format='csr'))
   F=uchsp_propagate(P,Y,steps=cfg['propagation_steps'],beta=cfg['beta'])['scores']
   rows.append(dict(dataset=ds,backbone=bb,seed=seed,mask_off_fixed=metrics(F,l,p.class_names)['bacc']))
  print(ds,bb,flush=True)
d=pd.DataFrame(rows);old=pd.read_csv('diag_attrib.csv').set_index(['dataset','backbone'])
means=d.groupby(['dataset','backbone']).mask_off_fixed.mean()
assert np.allclose(means,old.loc[means.index].knnSP_cgprAlpha,atol=1e-12,rtol=0)
d.to_csv('diagnostic_seed_scores.csv',index=False)
print('All eight means match original Table 2 exactly.',flush=True)
