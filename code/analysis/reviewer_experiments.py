"""Frozen reviewer-requested controls; no test labels enter selection.

Validation-size runs keep the unlabeled validation pool and centering fixed;
only the number of labeled validation images changes. Original test sets fixed.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,ast,collections,argparse,itertools,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
import current_cef_experiments as core
from current_cef_experiments import ROOT,CACHE,Experiment,GRAPH,ZERO,norm,metrics,sha
from current_cef_transfer import original
OUT=ROOT/'results/supplementary/reviewer_revision'

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def refs():return [j for j in json.loads((CACHE/'manifest.json').read_text()) if j['dataset'] in ['wikiart','mp100k'] and j['variant']=='reference']
def identity(j,s):return f"{j['dataset']}_{j['tag']}_{s}"
def frozen(j,s):return original(json.loads((ROOT/'results/supplementary/round5/select'/(identity(j,s)+'.json')).read_text()))
def chosen_score(e,sel,part):
 pc,pu,sc,su=e.prepare_mix(part,sel['branches']);r=sel['selected'];cf=r['config']
 if r['kind']=='phrase':return pc
 if r['kind']=='prototype':return pu
 if r['kind']=='output mixture':return (1-cf['weight'])*norm(pc)+cf['weight']*norm(pu)
 return e.lp(part,cf['graph'],(1-cf['weight'])*sc+cf['weight']*su)
class Alternative(Experiment):
 def sim(self,part,branch):
  if branch!='phrase':return super().sim(part,branch)
  key=part,branch
  if key not in self.sims:self.sims[key]=self.X[self.idx(part)]@self.mean_descriptor.T
  return self.sims[key]
 def select_branch(self,branch):return self.fixed[branch]

def controls(args):
 j,s=args;ident=identity(j,s);dest=OUT/'controls'/f'{ident}.json'
 if dest.exists():return
 e=Experiment(j,s);sel=frozen(j,s)
 old=pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv',usecols=['dataset','backbone','seed','method','metric','selected_config'])
 old=old[(old.dataset==j['dataset'])&(old.backbone==j['backbone'])&(old.seed==s)&(old.metric=='bAcc')]
 cfg=lambda m:json.loads(old[old.method==m].iloc[0].selected_config)
 alt=Alternative(j,s);d=np.load(ROOT/j['image_cache']);dm=d['dclip'];own=d['owner_dclip']
 alt.mean_descriptor=np.stack([dm[own==i].mean(0) for i in range(e.c)]) # mean cosine, no normalization
 alt.owner=np.arange(e.c);alt.fixed={'phrase':cfg('DCLIP fusion'),'prototype':cfg('CuPL fusion')}
 alt.knn=e.knn
 alt_sel=alt.select()
 source=json.loads((OUT/'protocol.json').read_text())
 pc,pu,_,_=e.prepare_mix('val',sel['branches'])
 branch='phrase' if metrics(pc,e.y[e.vi],e.c)['bAcc']>=metrics(pu,e.y[e.vi],e.c)['bAcc'] else 'prototype'
 decisions={'branch_select':branch,'alternative':alt_sel,'original':sel,'source_zlap':source['source_zlap_graph'],'test_labels_used':False}
 sf=OUT/'selections'/f'{ident}_controls.json';dump(sf,decisions)
 pc,pu,_,_=e.prepare_mix('test',sel['branches'])
 scores={'CEF':chosen_score(e,sel,'test'),'Phrase':pc,'Prototype':pu,'Branch selection':pc if branch=='phrase' else pu,'Fixed average':.5*(norm(pc)+norm(pu)),'DCLIP+CuPL selector':chosen_score(alt,alt_sel,'test'),'DCLIP fusion':alt.score('test','phrase',alt.fixed['phrase']),'CuPL fusion':pu}
 zg=cfg('ZLaP');zg=zg[0] if isinstance(zg,list) else zg['graph']
 scores['ZLaP']=e.lp('test',zg,e.cosine('test'))
 scores['Source ZLaP']=e.lp('test',source['source_zlap_graph'],e.cosine('test'))
 external=json.loads((ROOT/'results/supplementary/current_cef/external_protocol.json').read_text())['config']
 scores['Source CEF']=chosen_score(e,external,'test')
 rows=[dict(dataset=j['dataset'],backbone=j['backbone'],seed=s,method=m,**metrics(a,e.y[e.ti],e.c)) for m,a in scores.items()]
 ref=pd.read_csv(ROOT/'results/supplementary/round5/per_seed.csv');ref=ref[(ref.dataset==j['dataset'])&(ref.backbone==j['backbone'])&(ref.seed==s)&(ref.method=='validation selected hybrid')]
 assert abs(rows[0]['bAcc']-ref.iloc[0].bAcc)<1e-12
 pred=OUT/'predictions'/f'{ident}.npz';pred.parent.mkdir(exist_ok=True)
 np.savez_compressed(pred,indices=e.ti,labels=e.y[e.ti],**{m:a.argmax(1) for m,a in scores.items()})
 dump(dest,dict(rows=rows,selection_sha256=sha(sf),alternative_evaluations=len(alt_sel['candidates'])));print('controls',ident,flush=True)

def labels(args):
 j,s=args;ident=identity(j,s);e=Experiment(j,s);original_metric=core.metrics
 for frac in [.1,.25,.5,1.]:
  dest=OUT/'labels'/f'{ident}_{frac}.json'
  if dest.exists():continue
  rng=np.random.default_rng(71000+s);yy=e.y[e.vi];subset=np.concatenate([rng.permutation(np.flatnonzero(yy==c))[:max(1,int(np.ceil((yy==c).sum()*frac)))] for c in range(e.c)])
  e.memo={};e.ev={}
  def masked(a,y,c):return original_metric(a[subset],y[subset],c) if len(y)==len(e.vi) else original_metric(a,y,c)
  core.metrics=masked
  sel=e.select();gz=max(GRAPH,key=lambda g:e.vm('phrase',e.cfg(g,ZERO)))
  sf=OUT/'selections'/f'{ident}_labels_{frac}.json';dump(sf,dict(selection=sel,zlap_graph=gz,label_indices=e.vi[subset].tolist(),fraction=frac,test_labels_used=False))
  core.metrics=original_metric
  a=chosen_score(e,sel,'test');z=e.lp('test',gz,e.cosine('test'))
  rows=[dict(dataset=j['dataset'],backbone=j['backbone'],seed=s,fraction=frac,n_labeled=len(subset),method=m,**metrics(v,e.y[e.ti],e.c)) for m,v in [('CEF',a),('ZLaP',z)]]
  if frac==1.:
   ref=pd.read_csv(ROOT/'results/supplementary/round5/per_seed.csv');r=ref[(ref.dataset==j['dataset'])&(ref.backbone==j['backbone'])&(ref.seed==s)&(ref.method=='validation selected hybrid')].iloc[0]
   assert abs(rows[0]['bAcc']-r.bAcc)<1e-12,(ident,rows[0]['bAcc'],r.bAcc)
  dump(dest,dict(rows=rows,selection_sha256=sha(sf)));print('labels',ident,frac,flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','controls','labels'],required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();OUT.mkdir(exist_ok=True)
 if a.phase=='prepare':
  d=pd.read_csv(ROOT/'results/supplementary/round3/probe_per_seed.csv',usecols=['domain','backbone','seed','method','cfg']);z=d[d.method=='ZLaP'];assert len(z)==80
  configs=[ast.literal_eval(x)[0] for x in z.cfg];cnt=collections.Counter(json.dumps(g) for g in configs);g=json.loads(min(cnt,key=lambda g:(-cnt[g],g)))
  p=dict(source_zlap_graph=g,source_zlap_votes=dict(cnt),source_file_sha256=sha(ROOT/'results/supplementary/round3/probe_per_seed.csv'),seeds=[42,43,44,45,46],validation_fractions=[.1,.25,.5,1.],validation_rule='Nested class-stratified labeled subsets, fixed full validation pool for graph/centering, same test pool. All stages reselected using subset bAcc.',alternative='DCLIP fusion + CuPL fusion; same combination candidate enumeration as CEF; existing independently selected branch settings.',independent_confirmation=False,named_reference_interventions=False,source_sha256=sha(__file__))
  pf=OUT/'protocol.json'
  if pf.exists():assert json.loads(pf.read_text())==p
  else:dump(pf,p)
  print('frozen',p);return
 with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(controls if a.phase=='controls' else labels,[(j,s) for j in refs() for s in [42,43,44,45,46]]))
 rows=[]
 for f in sorted((OUT/a.phase).glob('*.json')):rows.extend(json.loads(f.read_text())['rows'])
 pd.DataFrame(rows).to_csv(OUT/(a.phase+'_per_seed.csv'),index=False)
if __name__=='__main__':main()
