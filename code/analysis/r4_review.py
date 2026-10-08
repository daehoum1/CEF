"""Matched fusion controls and transfer from non-art development domains.

Preparation uses existing caches; evaluation accepts a portable --cache directory.
The frozen protocol is saved before target evaluation. No outcome-based choice
of development domains, configurations, baselines or seeds is made.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'

import numpy as np
import pandas as pd
from zlap_impl import precompute_knn, zlap_transductive

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = [42, 43, 44, 45, 46]
KS, GAMMAS, ALPHAS = [5, 10, 20, 40, 80], [1., 3., 5.], [.3, .5, .7, .9, .95]
GRAPH = list(itertools.product(KS, GAMMAS, ALPHAS))
EVID = list(itertools.product([.005, .01, .02, .05], [0., .2, .4, .6, .8],
                              [True, False], ['cos', 'sm']))
INIT = (.01, .4, True, 'sm')
ZERO = (.01, 0., True, 'cos')
RAW_WS = [0., .2, .4, .6, .8, 1.]
BASELINES = ['DCLIP', 'CuPL', 'MeanPhrase']
DOMAINS = ['aircraft', 'dtd', 'eurosat', 'pets_s5']
KEY = ['dataset', 'backbone', 'seed', 'method', 'metric']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(cache, out):
    """Freeze external selections, then export the existing target input caches."""
    from cgpr_lab import BACKBONES, Pool
    from zlap_tune import clf_for
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'results/supplementary/round3/probe_per_seed.csv'
    frame = pd.read_csv(source)
    cef = frame[(frame.method == 'CEF') & frame.domain.isin(DOMAINS)].copy()
    assert len(cef) == 80 and set(cef.domain) == set(DOMAINS)
    cef = cef.sort_values(['domain', 'backbone', 'seed'])
    counts = collections.Counter(str(ast.literal_eval(x)) for x in cef.cfg)
    fixed = min(counts, key=lambda x: (-counts[x], x))
    protocol = dict(
        version=1, source_domains=DOMAINS, source_rows=len(cef),
        source_sha256=digest(source), fixed_config=ast.literal_eval(fixed),
        selection='Mode over 80 non-art CEF bAcc selections; lexical tie break.',
        source_selections=cef[['domain', 'backbone', 'seed', 'cfg']].to_dict('records'),
        targets=['wikiart', 'mp100k'], seeds=SEEDS,
        baselines=BASELINES, evidence_grid=EVID, graph_grid=GRAPH, raw_weights=RAW_WS,
        baseline_selection='Same graph/evidence/graph stages as CEF plus ZLaP; '
                           'also exhaustive 75 x 6 raw-cosine mixtures; select '
                           'between both paths on validation separately per metric.',
        target_labels_for_external_config=False,
        external_dictionaries=['full'] + [f'orig_g{g}' for g in range(5)],
        target_debias='Validation-image similarities only; no target labels.',
    )
    frozen = out / 'protocol.json'
    encoded = json.dumps(protocol, indent=2, sort_keys=True) + '\n'
    if frozen.exists():
        assert frozen.read_text() == encoded, 'Protocol changed after freezing'
    else:
        frozen.write_text(encoded)
    print(f'[frozen] non-art configuration: {fixed}; votes {counts[fixed]}/80', flush=True)
    jobs = []
    for ds in protocol['targets']:
        for tag, label in BACKBONES:
            fname = f'{ds}_{tag}.npz'
            dest = cache / fname
            if not dest.exists():
                p = Pool(ds, tag)
                data = dict(embs=p.embs, Y=p.Y, labels=p.labels, clf=clf_for(ds, tag),
                            class_names=np.array(p.class_names))
                dict_paths = [('full', HERE/'raw_concept_embs'/fname),
                              ('dclip', HERE/'control_dicts'/f'{ds}_{tag}_dclip.npz')]
                dict_paths += [(f'orig_g{g}', HERE/'r3_embs'/f'{ds}_orig_g{g}_{tag}.npz')
                               for g in range(5)]
                for name, path in dict_paths:
                    d = np.load(path, allow_pickle=True)
                    data[name] = d['embs']
                    data['owner_'+name] = np.array([p.class_names.index(x) for x in d['styles']])
                for seed in SEEDS:
                    split = np.load(ROOT/f'results/supplementary/artist_disjoint/splits/{ds}_seed{seed}.npz')
                    data[f'val_{seed}'], data[f'test_{seed}'] = split['val_idx'], split['test_idx']
                np.savez_compressed(dest, **data)
                print(f'[cache] {fname}', flush=True)
            jobs.append(dict(dataset=ds, tag=tag, backbone=label, file=fname,
                             sha256=digest(dest)))
    (cache/'manifest.json').write_text(json.dumps(jobs, indent=2)+'\n')
    print('[ready] cache manifest exported', flush=True)


def metrics(scores, labels, classes):
    pred = scores.argmax(1)
    recall = np.array([(pred[labels == c] == c).mean() for c in range(classes)])
    return {'bAcc': float(recall.mean()), 'Top-1': float((pred == labels).mean())}


def run_job(args):
    job, seed, cache_str, fixed, out_str = args
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    d = np.load(Path(cache_str)/job['file'])
    X, Y, labels, clf = d['embs'], d['Y'], d['labels'], d['clf']
    C = len(d['class_names'])
    vi, ti = d[f'val_{seed}'], d[f'test_{seed}']
    assert not np.intersect1d(vi, ti).size
    ix = {'val': vi, 'test': ti}
    knn = {p: precompute_knn(X[ii], clf, max(KS)) for p, ii in ix.items()}
    cos = {p: X[ii] @ clf.T for p, ii in ix.items()}
    dm = d['dclip']; owner = d['owner_dclip']
    proto = np.stack([dm[owner == c].mean(0) for c in range(C)])
    proto /= np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12
    sims = {}
    for name in ['full']+[f'orig_g{g}' for g in range(5)]:
        for p, ii in ix.items():
            sims[p, name] = X[ii] @ d[name].T
    desc = {}
    for p, ii in ix.items():
        temp = X[ii] @ dm.T
        desc[p, 'DCLIP'] = np.stack([temp[:, owner == c].mean(1) for c in range(C)], 1)
        desc[p, 'CuPL'] = X[ii] @ proto.T
        desc[p, 'MeanPhrase'] = np.stack([
            sims[p, 'full'][:, d['owner_full'] == c].mean(1) for c in range(C)], 1)
    ev_cache = {}

    def evidence(part, name, tau, debias):
        key = part, name, tau, debias
        if key not in ev_cache:
            is_descriptor = name in BASELINES
            source = desc if is_descriptor else sims
            s = source[part, name]
            if debias:
                s = s - source['val', name].mean(0, keepdims=True)
            a = s / tau
            a = np.exp(a - a.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)
            if is_descriptor:
                e = a
            else:
                e = np.stack([a[:, d['owner_'+name] == c].sum(1) for c in range(C)], 1)
                e /= e.sum(1, keepdims=True) + 1e-12
            ev_cache[key] = e
        return ev_cache[key]

    def fused(part, name, e):
        tau, w, db, base = e
        b = cos[part] if base == 'cos' else Y[ix[part]]
        if w == 0:
            return b
        ec = evidence(part, name, tau, db)
        if base == 'cos':
            ec = ec * b.max(1, keepdims=True)
        return (1-w)*b + w*ec

    def lp(part, g, affinity):
        return zlap_transductive(X[ix[part]], clf, *g, cross_affinity=affinity,
                                 knn_cache=knn[part])

    memo = {}
    touched = set()

    def vm(name, g, e):
        canon = e if e[1] else (0, 0, False, e[3])
        key = (name if e[1] else '', g, canon)
        touched.add(key)
        if key not in memo:
            memo[key] = metrics(lp('val', g, fused('val', name, e)), labels[vi], C)
        return memo[key]

    zsol = {}
    def zlap_solution(metric):
        if metric not in zsol:
            zsol[metric] = max(GRAPH, key=lambda g: vm('full',g,ZERO)[metric]), ZERO
        return zsol[metric]

    def staged(name, metric, pool=EVID):
        g = max(GRAPH, key=lambda g: vm(name,g,INIT)[metric])
        e = max(pool, key=lambda e: vm(name,g,e)[metric])
        g = max(GRAPH, key=lambda g: vm(name,g,e)[metric])
        return max([(g,e),zlap_solution(metric)],key=lambda ge:vm(name,*ge)[metric])

    rows = []
    def emit(method, metric, scores, config, val_score=None, evaluations=0):
        pred = scores.argmax(1)
        rec = [(pred[labels[ti] == c] == c).mean() for c in range(C)]
        rows.append(dict(dataset=job['dataset'], backbone=job['backbone'],seed=seed,
                         method=method,metric=metric,score=metrics(scores,labels[ti],C)[metric],
                         selected_config=json.dumps(config), val_score=val_score,
                         val_evals=evaluations,n_val=len(vi),n_test=len(ti),
                         recall=json.dumps(rec),input_sha256=job['sha256']))

    start = time.monotonic()
    for metric in ['bAcc','Top-1']:
        emit('Vanilla',metric,Y[ti],{})
        for method, select in [('ZLaP',lambda:zlap_solution(metric)),
                               ('CEF',lambda:staged('full',metric)),
                               ('CEF w=0',lambda:staged('full',metric,[e for e in EVID if e[1]==0]))]:
            touched.clear(); g,e=select()
            emit(method,metric,lp('test',g,fused('test','full',e)),[g,e],
                 vm('full',g,e)[metric],len(touched))
        for name in BASELINES:
            touched.clear()
            g,e=staged(name,metric)
            match_score=vm(name,g,e)[metric]
            calibrated_count=len(touched)
            # Additional exhaustive raw-cosine fusion gives the control access
            # to w=1 and the original descriptor+LP pipeline as well as ZLaP.
            def rawvm(graph, weight):
                key=('raw',name,graph,weight)
                if key not in memo:
                    a=(1-weight)*cos['val']+weight*desc['val',name]
                    memo[key]=metrics(lp('val',graph,a),labels[vi],C)
                return memo[key][metric]
            rg,rw=max(itertools.product(GRAPH,RAW_WS),key=lambda gw:rawvm(*gw))
            raw_score=rawvm(rg,rw)
            emit(name+' calibrated fusion',metric,lp('test',g,fused('test',name,e)),
                 dict(path='calibrated',graph=g,evidence=e),match_score,calibrated_count)
            if raw_score>match_score:
                scores=lp('test',rg,(1-rw)*cos['test']+rw*desc['test',name])
                cfg=dict(path='raw',graph=rg,weight=rw)
            else:
                scores=lp('test',g,fused('test',name,e))
                cfg=dict(path='calibrated',graph=g,evidence=e)
            emit(name+' fusion',metric,scores,cfg,max(raw_score,match_score),calibrated_count+450)

    # External config is fixed before reading target performance and used on
    # every target, backbone, split, dictionary and metric without reselection.
    g,e=tuple(fixed[0]),tuple(fixed[1])
    for name in ['full']+[f'orig_g{g}' for g in range(5)]:
        scores=lp('test',g,fused('test',name,e))
        for metric in ['bAcc','Top-1']:
            emit('CEF external '+name,metric,scores,[g,e],None,0)
    path=Path(out_str)/'jobs'/f"{job['dataset']}_{job['tag']}_{seed}.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
    print(f"[done] {job['dataset']} / {job['backbone']} / {seed}: "
          f"{len(rows)} rows in {time.monotonic()-start:.1f}s",flush=True)
    return str(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--cache',type=Path,default=HERE/'r4_cache')
    ap.add_argument('--out',type=Path,default=ROOT/'results/supplementary/round4')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--only',default='')
    ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    a=ap.parse_args()
    if a.prepare:
        prepare(a.cache,a.out)
        return
    protocol=json.loads((a.out/'protocol.json').read_text())
    manifests=json.loads((a.cache/'manifest.json').read_text())
    jobs=[]
    for item in manifests:
        if a.only and a.only not in (item['dataset'],item['tag']):
            continue
        assert digest(a.cache/item['file'])==item['sha256']
        for seed in a.seeds:
            dest=a.out/'jobs'/f"{item['dataset']}_{item['tag']}_{seed}.csv"
            if not dest.exists():
                jobs.append((item,seed,str(a.cache),protocol['fixed_config'],str(a.out)))
    from concurrent.futures import ProcessPoolExecutor,as_completed
    print(f'[run] {len(jobs)} jobs on {a.workers} workers',flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(run_job,job) for job in jobs]
        for f in as_completed(futures):
            f.result()
            frames=[pd.read_csv(p) for p in sorted((a.out/'jobs').glob('*.csv'))]
            merged=pd.concat(frames,ignore_index=True).sort_values(KEY)
            assert not merged.duplicated(KEY).any()
            temp=a.out/'per_seed_r4.csv.tmp'
            merged.to_csv(temp,index=False)
            temp.replace(a.out/'per_seed_r4.csv')
    print('[saved]',a.out/'per_seed_r4.csv',flush=True)


if __name__=='__main__':
    main()
