"""Serial CPU timing of complete CEF and ZLaP, seed 42, cached embeddings.

Run after parallel experiments finish. Each measurement uses a fresh process.
CEF selection includes both branch searches and the combination search. Test
inference executes only the selected path (one graph for a graph mixture).
"""
from current_cef_experiments import ROOT,OUT,CACHE,Experiment,GRAPH,ZERO,metrics,norm
import json,time,resource,subprocess,sys,argparse,platform
import pandas as pd

def measure(job,method):
    ex=Experiment(job,42);start=time.monotonic()
    if method=='CEF':cfg=ex.select()
    else:
        g=max(GRAPH,key=lambda g:ex.vm('phrase',ex.cfg(g,ZERO)))
        cfg=ex.cfg(g,ZERO)
    select_s=time.monotonic()-start
    # Fresh inference state prevents validation-search cache reuse from hiding work.
    del ex
    ex=Experiment(job,42);start=time.monotonic()
    if method=='ZLaP':scores=ex.score('test','phrase',cfg)
    else:
        r=cfg['selected'];b=cfg['branches'];cf=r['config']
        if r['kind'] in ['phrase','prototype']:scores=ex.score('test',r['kind'],b[r['kind']])
        elif r['kind']=='output mixture':
            pc=ex.score('test','phrase',b['phrase']);pu=ex.score('test','prototype',b['prototype'])
            scores=(1-cf['weight'])*norm(pc)+cf['weight']*norm(pu)
        else:
            ac=ex.aff('test','phrase',b['phrase']);au=ex.aff('test','prototype',b['prototype'])
            aff=((1-cf['weight'])*norm(ac)+cf['weight']*norm(au))*ex.cosine('test').max(1,keepdims=True)
            scores=ex.lp('test',cf['graph'],aff)
    test_s=time.monotonic()-start
    m=metrics(scores,ex.y[ex.ti],ex.c)
    return dict(dataset=job['dataset'],backbone=job['backbone'],method=method,seed=42,
                select_s=select_s,test_s=test_s,peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,**m)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=int);ap.add_argument('--method');a=ap.parse_args()
    jobs=[j for j in json.loads((CACHE/'manifest.json').read_text()) if j['dataset'] in ['wikiart','mp100k'] and j['variant']=='reference']
    dest=OUT/'cost';dest.mkdir(exist_ok=True)
    if a.job is not None:
        value=measure(jobs[a.job],a.method)
        (dest/f'{a.job}_{a.method}.json').write_text(json.dumps(value,indent=2)+'\n')
        return
    for i in range(len(jobs)):
        for m in ['ZLaP','CEF']:
            if not (dest/f'{i}_{m}.json').exists():subprocess.run([sys.executable,'-B',__file__,'--job',str(i),'--method',m],check=True)
            print('Measured',jobs[i]['dataset'],jobs[i]['backbone'],m,flush=True)
    rows=[json.loads(f.read_text()) for f in sorted(dest.glob('*.json'))]
    pd.DataFrame(rows).to_csv(OUT/'cost.csv',index=False)
    (OUT/'cost_environment.json').write_text(json.dumps(dict(platform=platform.platform(),threads_per_process=2,serial=True,cpu=subprocess.check_output(['lscpu'],text=True),scope='Cached-embedding CPU post-processing. Excludes model/image/text encoding, LLM generation, and disk loading. Peak process RSS includes inputs and validation state.'),indent=2)+'\n')
if __name__=='__main__':main()
