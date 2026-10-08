"""Evaluate the complete CEF method for dictionary and component interventions.

Validation-only selection is saved before test inference. All variants use the
published branch searches and combination selector; dictionary interventions
affect both branches. Historical result IDs are never relabeled as CEF.
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
from pathlib import Path
import argparse, json, itertools, time, hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from zlap_impl import precompute_knn, zlap_transductive

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'results/supplementary/current_cef'
CACHE = ROOT/'analysis/current_cef_cache'
GRAPH = list(itertools.product([5,10,20,40,80], [1.,3.,5.], [.3,.5,.7,.9,.95]))
EVID = list(itertools.product([.005,.01,.02,.05], [0.,.2,.4,.6,.8], [True,False], ['cos','sm']))
INIT = (.01,.4,True,'sm')
ZERO = (.01,0.,True,'cos')
WEIGHTS = [i/10 for i in range(11)]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(a):
    a=np.maximum(a,0);return a/np.maximum(a.sum(1,keepdims=True),1e-30)
def metrics(a,y,c):
    pred=a.argmax(1);rec=[float((pred[y==i]==i).mean()) for i in range(c)]
    return dict(bAcc=float(np.mean(rec)),top1=float((pred==y).mean()),recall=rec)
def split(y,seed):
    rng=np.random.default_rng(seed);vi=[];ti=[]
    for c in np.unique(y):
        ii=rng.permutation(np.flatnonzero(y==c));n=max(1,int(round(len(ii)*.2)))
        vi.extend(ii[:n]);ti.extend(ii[n:])
    return np.array(vi),np.array(ti)

class Experiment:
    def __init__(self,job,seed):
        self.job=job;self.seed=seed
        d=np.load(ROOT/job['image_cache'],allow_pickle=True)
        if 'embs' in d:
            self.X=d['embs'];self.Y=d['Y'];self.clf=d['clf'];self.y=d['labels']
            self.names=d['class_names'].tolist()
            if job.get('image_level'):self.vi,self.ti=split(self.y,seed)
            else:self.vi,self.ti=d[f'val_{seed}'],d[f'test_{seed}']
        else:
            self.X=d['image_embs'];self.Y=d['Y'];self.clf=d['text_embs'];self.y=d['labels']
            self.names=d['classes'].tolist();self.vi,self.ti=split(self.y,seed)
        assert not np.intersect1d(self.vi,self.ti).size
        self.c=len(self.names);self.knn={};self.cos={};self.sims={};self.ev={};self.memo={}
        e=np.load(CACHE/job['evidence_cache'])
        self.full=e['full'];self.owner=e['owner'];self.proto=e['proto']
        self.no_graph=job['variant']=='no_graph';self.no_debias=job['variant']=='no_debias'
    def idx(self,part):return self.vi if part=='val' else self.ti
    def cosine(self,part):
        if part not in self.cos:self.cos[part]=self.X[self.idx(part)]@self.clf.T
        return self.cos[part]
    def sim(self,part,branch):
        key=part,branch
        if key not in self.sims:self.sims[key]=self.X[self.idx(part)]@(self.full if branch=='phrase' else self.proto).T
        return self.sims[key]
    def aff(self,part,branch,cfg):
        if cfg['path']=='raw':
            w=cfg['weight'];return (1-w)*self.cosine(part)+w*self.sim(part,branch)
        tau,w,db,base=cfg['evidence'];b=self.cosine(part) if base=='cos' else self.Y[self.idx(part)]
        if not w:return b
        key=part,branch,tau,db
        if key not in self.ev:
            a=self.sim(part,branch)
            if db:a=a-self.sim('val',branch).mean(0,keepdims=True)
            a=a/tau;a=np.exp(a-a.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
            if branch=='phrase':
                a=np.stack([a[:,self.owner==c].sum(1) for c in range(self.c)],1)
                a/=a.sum(1,keepdims=True)+1e-12
            self.ev[key]=a
        a=self.ev[key]
        if base=='cos':a=a*b.max(1,keepdims=True)
        return (1-w)*b+w*a
    def lp(self,part,g,a):
        if self.no_graph:return a
        idx=self.idx(part)
        if part not in self.knn:self.knn[part]=precompute_knn(self.X[idx],self.clf,80)
        return zlap_transductive(self.X[idx],self.clf,*g,cross_affinity=a,knn_cache=self.knn[part])
    def score(self,part,branch,cfg):return self.lp(part,cfg['graph'],self.aff(part,branch,cfg))
    def vm(self,branch,cfg):
        key=branch,json.dumps(cfg,sort_keys=True)
        if key not in self.memo:self.memo[key]=metrics(self.score('val',branch,cfg),self.y[self.vi],self.c)['bAcc']
        return self.memo[key]
    def cfg(self,g,e):return dict(path='calibrated',graph=g,evidence=e)
    def select_branch(self,branch):
        graph=[GRAPH[0]] if self.no_graph else GRAPH
        evidence=[e for e in EVID if not self.no_debias or not e[2]]
        initial=(.01,.4,False,'sm') if self.no_debias else INIT
        g=max(graph,key=lambda g:self.vm(branch,self.cfg(g,initial)))
        e=max(evidence,key=lambda e:self.vm(branch,self.cfg(g,e)))
        g=max(graph,key=lambda g:self.vm(branch,self.cfg(g,e)))
        best=self.cfg(g,e)
        gz=max(graph,key=lambda g:self.vm(branch,self.cfg(g,ZERO)))
        zero=self.cfg(gz,ZERO)
        if self.vm(branch,zero)>self.vm(branch,best):best=zero
        if branch=='prototype':
            raw=[dict(path='raw',graph=g,weight=w) for g,w in itertools.product(graph,[0.,.2,.4,.6,.8,1.])]
            r=max(raw,key=lambda cf:self.vm(branch,cf))
            if self.vm(branch,r)>self.vm(branch,best):best=r
        return best
    def prepare_mix(self,part,configs):
        ac=self.aff(part,'phrase',configs['phrase']);au=self.aff(part,'prototype',configs['prototype'])
        pc=self.lp(part,configs['phrase']['graph'],ac);pu=self.lp(part,configs['prototype']['graph'],au)
        cosine=self.cosine(part)
        return pc,pu,norm(ac)*cosine.max(1,keepdims=True),norm(au)*cosine.max(1,keepdims=True)
    def select(self):
        cfg={b:self.select_branch(b) for b in ['phrase','prototype']}
        pc,pu,sc,su=self.prepare_mix('val',cfg);nc,nu=norm(pc),norm(pu);records=[]
        def add(kind,a,config):
            v=metrics(a,self.y[self.vi],self.c)['bAcc']
            records.append(dict(kind=kind,config=config,bAcc=v));return v
        add('phrase',pc,{});add('prototype',pu,{})
        for w in WEIGHTS:add('output mixture',(1-w)*nc+w*nu,dict(weight=w))
        if not self.no_graph:
            memo={}
            def test(g,w):
                key=tuple(g),w
                if key not in memo:memo[key]=add('graph mixture',self.lp('val',g,(1-w)*sc+w*su),dict(graph=g,weight=w))
                return memo[key]
            anchors=list(dict.fromkeys(tuple(cfg[b]['graph']) for b in ['phrase','prototype']))
            g,w=max(itertools.product(anchors,WEIGHTS),key=lambda gw:test(*gw))
            g=max(GRAPH,key=lambda gg:test(gg,w));w=max(WEIGHTS,key=lambda ww:test(g,ww))
        return dict(branches=cfg,selected=max(records,key=lambda r:r['bAcc']),candidates=records,
                    branch_evaluations=len(self.memo),test_labels_used=False)
    def evaluate(self,selection):
        pc,pu,sc,su=self.prepare_mix('test',selection['branches']);r=selection['selected'];cfg=r['config']
        if r['kind']=='phrase':a=pc
        elif r['kind']=='prototype':a=pu
        elif r['kind']=='output mixture':a=(1-cfg['weight'])*norm(pc)+cfg['weight']*norm(pu)
        else:a=self.lp('test',cfg['graph'],(1-cfg['weight'])*sc+cfg['weight']*su)
        return metrics(a,self.y[self.ti],self.c)

def worker(args):
    job,seed,phase=args;ident=f"{job['dataset']}_{job['tag']}_{job['variant']}_{seed}"
    dest=OUT/phase/(ident+'.json')
    if dest.exists():return str(dest)
    start=time.monotonic();ex=Experiment(job,seed);loaded=time.monotonic()
    if phase=='select':
        value=ex.select();value.update(job=job,seed=seed,n_val=len(ex.vi),n_test=len(ex.ti),
            evidence_sha256=sha(CACHE/job['evidence_cache']),selection_seconds=time.monotonic()-loaded)
    else:
        frozen=OUT/'select'/(ident+'.json');sel=json.loads(frozen.read_text())
        assert sel['evidence_sha256']==sha(CACHE/job['evidence_cache'])
        value=dict(dataset=job['dataset'],backbone=job['backbone'],variant=job['variant'],seed=seed,
                   **ex.evaluate(sel),selection_sha256=sha(frozen),test_seconds=time.monotonic()-loaded)
    dest.write_text(json.dumps(value,indent=2)+'\n')
    print(f'{phase} {ident}: {time.monotonic()-start:.1f}s',flush=True)
    return str(dest)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['select','evaluate'],required=True)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--variants',nargs='+')
    ap.add_argument('--datasets',nargs='+');ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    ap.add_argument('--tags',nargs='+');a=ap.parse_args()
    (OUT/a.phase).mkdir(parents=True,exist_ok=True)
    jobs=json.loads((CACHE/'manifest.json').read_text())
    jobs=[j for j in jobs if (not a.variants or j['variant'] in a.variants) and (not a.datasets or j['dataset'] in a.datasets) and (not a.tags or j['tag'] in a.tags)]
    tasks=[(j,s,a.phase) for j in jobs for s in a.seeds]
    if a.phase=='evaluate':
        assert all((OUT/'select'/f"{j['dataset']}_{j['tag']}_{j['variant']}_{s}.json").exists() for j,s,_ in tasks)
    protocol=dict(graph=GRAPH,evidence=EVID,weights=WEIGHTS,selection_metric='bAcc',
                  intervention='Dictionary changes affect both phrase and prototype branches.',
                  source_sha256=sha(__file__),status='Follow-up experiments on previously inspected benchmarks.')
    pf=OUT/'protocol.json'
    if pf.exists():assert json.loads(pf.read_text())==json.loads(json.dumps(protocol))
    else:pf.write_text(json.dumps(protocol,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(worker,tasks))
    if a.phase=='evaluate':
        rows=[json.loads(f.read_text()) for f in sorted((OUT/'evaluate').glob('*.json'))]
        pd.DataFrame(rows).to_csv(OUT/'per_seed.csv',index=False)

if __name__=='__main__':main()
