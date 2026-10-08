"""Prepare both evidence branches for current-CEF dictionary interventions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,argparse
import numpy as np
from current_cef_experiments import CACHE,ROOT

import cef_paths

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference-only',action='store_true');a=ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    base=json.loads((ROOT/'analysis/r4_cache/manifest.json').read_text());jobs=[]
    cef_paths.add_bundled_paths()
    from control_dicts import SPECS
    from cgpr_lab import Pool
    from vlm_comparison import VLMEncoder
    for tag,(model,pretrained) in SPECS.items():
        enc=None
        for item in [j for j in base if j['tag']==tag]:
            ds=item['dataset'];d=np.load(ROOT/'analysis/r4_cache'/item['file'])
            names=d['class_names'].tolist();c=len(names)
            raw=np.load(ROOT/'analysis/raw_concept_embs'/item['file'],allow_pickle=True)
            defs={'reference':(d['full'],d['owner_full'],raw['phrases'].tolist(),d['dclip'])}
            if not a.reference_only:
                for v in ['rand_chars','rand_words']:
                    e=np.load(ROOT/'analysis/control_dicts'/f'{ds}_{tag}_{v}.npz',allow_pickle=True)
                    defs[v]=(e['embs'],np.array([names.index(x) for x in e['styles']]),e['phrases'].tolist(),None)
                owner=d['owner_full'][np.random.default_rng(0).permutation(len(d['owner_full']))]
                defs['shuffle_owner']=(d['full'],owner,raw['phrases'].tolist(),None)
                pool=Pool(ds,tag)
                defs['filtered']=(pool.concept_embs,pool.phrase_style_idx,pool.phrases,None)
                for g in range(5):
                    e=np.load(ROOT/'analysis/r3_embs'/f'{ds}_orig_g{g}_{tag}.npz',allow_pickle=True)
                    defs[f'orig_g{g}']=(e['embs'],np.array([names.index(x) for x in e['styles']]),e['phrases'].tolist(),None)
            for variant,(full,owner,phrases,templated) in defs.items():
                dest=CACHE/f'{ds}_{tag}_{variant}.npz'
                if not dest.exists():
                    if templated is None:
                        if enc is None:enc=VLMEncoder(model,pretrained)
                        templated=enc.encode_texts([f'{names[o]}, which has {phrase}' for o,phrase in zip(owner,phrases)])
                    proto=np.stack([templated[owner==i].mean(0) if (owner==i).any() else np.zeros(templated.shape[1],dtype=templated.dtype) for i in range(c)])
                    proto/=np.linalg.norm(proto,axis=1,keepdims=True)+1e-12
                    np.savez(dest,full=full,owner=owner,proto=proto)
                    (dest.with_suffix('.json')).write_text(json.dumps(dict(phrases=phrases,owners=owner.tolist(),class_names=names,variant=variant),indent=2)+'\n')
                jobs.append(dict(dataset=ds,tag=tag,backbone=item['backbone'],variant=variant,
                                 image_cache='analysis/r4_cache/'+item['file'],evidence_cache=dest.name))
            for variant in ['no_debias','no_graph','image_level']:
                if a.reference_only:continue
                j=dict(jobs[-len(defs)],variant=variant,evidence_cache=f'{ds}_{tag}_reference.npz')
                if variant=='image_level':j['image_level']=True
                jobs.append(j)
            print('Prepared',ds,tag,flush=True)
        if enc is not None:
            del enc
            import torch
            torch.cuda.empty_cache()
    (CACHE/'manifest.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print('Prepared jobs:',len(jobs))
if __name__=='__main__':main()
