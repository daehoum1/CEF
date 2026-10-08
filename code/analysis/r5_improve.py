"""Exploratory CEF/CuPL improvement: freeze all validation choices before testing."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
from pathlib import Path
import argparse,json,hashlib,itertools,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from zlap_impl import precompute_knn,zlap_transductive

ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/'analysis/r4_cache'
OUT=ROOT/'results/supplementary/round5'
GRAPH=list(itertools.product([5,10,20,40,80],[1.,3.,5.],[.3,.5,.7,.9,.95]))
WEIGHTS=[i/10 for i in range(11)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(a):
    a=np.maximum(a,0);return a/np.maximum(a.sum(1,keepdims=True),1e-30)
def metric(a,y,c):
    pred=a.argmax(1);rec=[float((pred[y==i]==i).mean()) for i in range(c)]
    return dict(bAcc=float(np.mean(rec)),top1=float((pred==y).mean()),recall=rec)
def worker(args):
    job,seed,phase=args;start=time.monotonic()
    ident=f"{job['dataset']}_{job['tag']}_{seed}"
    dest=OUT/phase/(ident+'.json')
    if dest.exists():return str(dest)
    d=np.load(CACHE/job['file']); X=d['embs'];clf=d['clf'];c=len(d['class_names'])
    vi=d[f'val_{seed}'];idx=vi if phase=='select' else d[f'test_{seed}']
    assert not np.intersect1d(vi,d[f'test_{seed}']).size
    # Load configurations only: no previously observed test performance enters selection.
    old=pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv',usecols=['dataset','backbone','seed','method','metric','selected_config'])
    old=old[(old.dataset==job['dataset'])&(old.backbone==job['backbone'])&(old.seed==seed)&(old.metric=='bAcc')]
    cfg={m:json.loads(old[old.method==m].iloc[0].selected_config) for m in ['CEF','CuPL fusion']}
    cosine=X[idx]@clf.T
    dm=d['dclip'];owner=d['owner_dclip']
    proto=np.stack([dm[owner==i].mean(0) for i in range(c)]);proto/=np.linalg.norm(proto,axis=1,keepdims=True)+1e-12
    def affinity(name,config):
        if name=='CEF':g,e=config
        else:
            g=config['graph']
            if config['path']=='raw':
                w=config['weight'];return tuple(g),(1-w)*cosine+w*(X[idx]@proto.T)
            e=config['evidence']
        tau,w,db,base=e;b=cosine if base=='cos' else d['Y'][idx]
        if not w:return tuple(g),b
        tex=d['full'] if name=='CEF' else proto
        s=X[idx]@tex.T
        if db:s=s-(X[vi]@tex.T).mean(0,keepdims=True)
        a=s/tau;a=np.exp(a-a.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
        if name=='CEF':
            a=np.stack([a[:,d['owner_full']==i].sum(1) for i in range(c)],1)
            a/=a.sum(1,keepdims=True)+1e-12
        if base=='cos':a=a*b.max(1,keepdims=True)
        return tuple(g),(1-w)*b+w*a
    gc,ac=affinity('CEF',cfg['CEF']);gu,au=affinity('CuPL fusion',cfg['CuPL fusion'])
    knn=precompute_knn(X[idx],clf,80)
    def lp(g,a):return zlap_transductive(X[idx],clf,*g,cross_affinity=a,knn_cache=knn)
    pc,pu=lp(gc,ac),lp(gu,au); nc,nu=norm(pc),norm(pu)
    # Match cross-modal scale to the original class cosine for both branches.
    sc=norm(ac)*cosine.max(1,keepdims=True);su=norm(au)*cosine.max(1,keepdims=True)
    if phase=='select':
        y=d['labels'][vi];records=[];predictions={}
        def add(key,a,config):
            score=metric(a,y,c)['bAcc'];records.append(dict(kind=key,config=config,bAcc=score));return score
        add('CEF',pc,{});add('CuPL fusion',pu,{})
        for w in WEIGHTS:add('output mixture',(1-w)*nc+w*nu,dict(weight=w))
        # Two original graph settings, then a full graph sweep at the selected weight.
        memo={}
        def test(g,w):
            key=(tuple(g),w)
            if key not in memo:memo[key]=add('graph mixture',lp(g,(1-w)*sc+w*su),dict(graph=g,weight=w))
            return memo[key]
        anchors=list(dict.fromkeys([gc,gu]))
        g,w=max(itertools.product(anchors,WEIGHTS),key=lambda gw:test(*gw))
        g=max(GRAPH,key=lambda gg:test(gg,w))
        # A final weight pass allows the new graph to adjust the evidence balance.
        w=max(WEIGHTS,key=lambda ww:test(g,ww))
        def best(kinds):return max([r for r in records if r['kind'] in kinds],key=lambda r:r['bAcc'])
        selected={'output mixture':best(['CEF','CuPL fusion','output mixture']),
                  'graph mixture':best(['CEF','CuPL fusion','graph mixture']),
                  'validation selected hybrid':max(records,key=lambda r:r['bAcc'])}
        result=dict(dataset=job['dataset'],backbone=job['backbone'],seed=seed,selected=selected,
                    baseline_configs=cfg,validation_candidates=records,n_val=len(vi),
                    selection_metric='bAcc',input_sha256=job['sha256'],test_labels_used=False)
    else:
        frozen=json.loads((OUT/'select'/(ident+'.json')).read_text())
        assert cfg==frozen['baseline_configs']
        y=d['labels'][idx];rows=[]
        def emit(name,a,selection):rows.append(dict(dataset=job['dataset'],backbone=job['backbone'],seed=seed,method=name,**metric(a,y,c),selection=selection))
        emit('CEF',pc,{});emit('CuPL fusion',pu,{})
        for name,r in frozen['selected'].items():
            kind=r['kind'];cf=r['config']
            if kind=='CEF':pred=pc
            elif kind=='CuPL fusion':pred=pu
            elif kind=='output mixture':pred=(1-cf['weight'])*nc+cf['weight']*nu
            else:pred=lp(tuple(cf['graph']),(1-cf['weight'])*sc+cf['weight']*su)
            emit(name,pred,r)
        result=rows
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{phase} {ident}: {time.monotonic()-start:.1f}s',flush=True)
    return str(dest)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['select','evaluate'],required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/a.phase).mkdir(exist_ok=True)
    jobs=json.loads((CACHE/'manifest.json').read_text());tasks=[(j,s,a.phase) for j in jobs for s in [42,43,44,45,46]]
    protocol=dict(method='CEF/CuPL hybrid',weights=WEIGHTS,graph=GRAPH,metric='bAcc',seeds=[42,43,44,45,46],
                  selection='Original bAcc-selected branches. Output mixture; graph mixture with two-anchor weight pass, full graph pass, final weight pass. Endpoints included. Ties favor original CEF, then CuPL, then first candidate.',
                  status='Exploratory development on previously inspected benchmarks; not fresh confirmatory evidence.',
                  source_sha256=sha(OUT / "selection_source.py") if (OUT / "selection_source.py").exists() else sha(__file__),r4_results_sha256=sha(ROOT/'results/supplementary/round4/per_seed_r4.csv'))
    pf=OUT/'protocol.json'
    if pf.exists():assert json.loads(pf.read_text())==json.loads(json.dumps(protocol))
    else:pf.write_text(json.dumps(protocol,indent=2)+'\n')
    if a.phase=='evaluate':
        paths=[OUT/'select'/f"{j['dataset']}_{j['tag']}_{s}.json" for j,s,_ in tasks];assert all(x.exists() for x in paths)
        lock={x.name:sha(x) for x in paths};fp=OUT/'frozen_selections.json'
        if fp.exists():assert json.loads(fp.read_text())==lock
        else:fp.write_text(json.dumps(lock,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=a.workers) as ex: list(ex.map(worker,tasks))
    if a.phase=='evaluate':
        rows=[]
        for f in sorted((OUT/'evaluate').glob('*.json')):rows.extend(json.loads(f.read_text()))
        df=pd.DataFrame(rows);df.to_csv(OUT/'per_seed.csv',index=False)
        old=pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv');old=old[old.metric=='bAcc']
        for m in ['CEF','CuPL fusion']:
            q=df[df.method==m].merge(old[old.method==m],on=['dataset','backbone','seed']);assert len(q)==40;assert np.max(np.abs(q.bAcc-q.score))<1e-12
        means=df.groupby(['dataset','backbone','method']).bAcc.mean().unstack();means.to_csv(OUT/'pair_means.csv')
        summary=[]
        for m in ['output mixture','graph mixture','validation selected hybrid']:
            delta=100*(means[m]-means['CuPL fusion']);summary.append(dict(method=m,mean_pp=delta.mean(),min_pp=delta.min(),max_pp=delta.max(),positive_pairs=int((delta>0).sum())))
        pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
