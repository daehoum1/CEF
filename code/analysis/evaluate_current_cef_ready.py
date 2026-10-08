"""Evaluate completed, immutable validation selections as experiment jobs finish."""
import current_cef_experiments as experiment
from current_cef_experiments import OUT,worker
import hashlib
# Exact reusable neighbors depend only on the image/class embeddings and k.
# This cache changes no affinities, solver settings or validation choices.
_original_knn=experiment.precompute_knn
_neighbor_cache={}
def cached_knn(features,clf,k):
    key=(hashlib.sha256(features.tobytes()).digest(),hashlib.sha256(clf.tobytes()).digest(),k)
    if key not in _neighbor_cache:_neighbor_cache[key]=_original_knn(features,clf,k)
    return _neighbor_cache[key]
experiment.precompute_knn=cached_knn
from concurrent.futures import ProcessPoolExecutor
import json,time
import pandas as pd
(OUT/'evaluate').mkdir(exist_ok=True)
with ProcessPoolExecutor(max_workers=2) as pool:
    while True:
        todo=[]
        for f in sorted((OUT/'select').glob('*.json')):
            if (OUT/'evaluate'/f.name).exists():continue
            s=json.loads(f.read_text());todo.append((s['job'],s['seed'],'evaluate'))
        if todo:list(pool.map(worker,todo))
        files=sorted((OUT/'evaluate').glob('*.json'))
        if files:
            pd.DataFrame([json.loads(f.read_text()) for f in files]).to_csv(OUT/'per_seed.csv',index=False)
        print('Evaluation progress:',len(files),'/ 641',flush=True)
        if len(files)>=641:break
        time.sleep(10)
