"""Reuse complete CEF configurations; freeze source-only choices before target runs."""
from current_cef_experiments import ROOT,CACHE,OUT,Experiment,sha
import json,collections,itertools,argparse,hashlib
import current_cef_experiments as experiment
_original_knn=experiment.precompute_knn
_neighbor_cache={}
def cached_knn(features,clf,k):
    key=(hashlib.sha256(features.tobytes()).digest(),hashlib.sha256(clf.tobytes()).digest(),k)
    if key not in _neighbor_cache:_neighbor_cache[key]=_original_knn(features,clf,k)
    return _neighbor_cache[key]
experiment.precompute_knn=cached_knn
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def clean(s):
    return dict(branches=s['branches'],selected={k:s['selected'][k] for k in ['kind','config']})
def mode(items):
    votes=collections.Counter(json.dumps(clean(x),sort_keys=True) for x in items)
    return json.loads(min(votes,key=lambda x:(-votes[x],x)))
def original(s):
    cf=s['baseline_configs'];g,e=cf['CEF'];chosen=dict(s['selected']['validation selected hybrid'])
    chosen['kind']={'CEF':'phrase','CuPL fusion':'prototype'}.get(chosen['kind'],chosen['kind'])
    return dict(branches=dict(phrase=dict(path='calibrated',graph=g,evidence=e),prototype=cf['CuPL fusion']),selected=chosen)
def worker(args):
    job,seed,setting,cfg=args
    dest=OUT/'transfer'/f"{job['dataset']}_{job['tag']}_{job['variant']}_{seed}_{setting}.json"
    if dest.exists():return
    e=Experiment(job,seed);m=e.evaluate(cfg)
    dest.write_text(json.dumps(dict(dataset=job['dataset'],backbone=job['backbone'],variant=job['variant'],seed=seed,setting=setting,config=cfg,**m),indent=2)+'\n')
    print('Transfer',dest.stem,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--external',action='store_true');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    (OUT/'transfer').mkdir(exist_ok=True)
    jobs=[j for j in json.loads((CACHE/'manifest.json').read_text()) if j['dataset'] in ['wikiart','mp100k']]
    tasks=[]
    if a.external:
        sources=[];hashes={}
        for f in sorted((OUT/'select').glob('*.json')):
            s=json.loads(f.read_text())
            if s['job']['dataset'] in ['dtd','eurosat','aircraft','pets_s5'] and s['job']['variant']=='reference':sources.append(s);hashes[f.name]=sha(f)
        assert len(sources)==80,len(sources)
        cfg=mode(sources)
        protocol=dict(source_files=hashes,config=cfg,rule='Joint mode of complete CEF branch and combination configurations; lexical tie breaking; no target labels.',test_labels_used=False)
        pf=OUT/'external_protocol.json'
        if pf.exists():assert json.loads(pf.read_text())==protocol
        else:pf.write_text(json.dumps(protocol,indent=2)+'\n')
        for job in jobs:
            if job['variant']=='reference' or job['variant'].startswith('orig_g'):
                tasks.extend((job,s,'external',cfg) for s in [42,43,44,45,46])
    else:
        sources=[json.loads(f.read_text()) for f in sorted((ROOT/'results/supplementary/round5/select').glob('*.json'))]
        assert len(sources)==40
        global_cfg=mode([original(s) for s in sources]);settings={}
        for job in jobs:
            if job['variant']!='reference':continue
            leave=mode([original(s) for s in sources if (s['dataset'],s['backbone'])!=(job['dataset'],job['backbone'])])
            settings[job['dataset']+'_'+job['tag']]=dict(global_mode=global_cfg,leave_pair_out=leave)
            for seed in [42,43,44,45,46]:
                tasks.extend([(job,seed,'global_mode',global_cfg),(job,seed,'leave_pair_out',leave)])
        pf=OUT/'reuse_protocol.json'
        protocol=dict(settings=settings,source_sha256=sha(ROOT/'results/supplementary/round5/frozen_selections.json'),independent_transfer=False)
        if pf.exists():assert json.loads(pf.read_text())==protocol
        else:pf.write_text(json.dumps(protocol,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(worker,tasks,chunksize=30 if a.external else 10))
    rows=[json.loads(f.read_text()) for f in sorted((OUT/'transfer').glob('*.json'))]
    pd.DataFrame(rows).to_csv(OUT/'transfer_per_seed.csv',index=False)
if __name__=='__main__':main()
