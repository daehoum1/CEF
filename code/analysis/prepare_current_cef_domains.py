"""Extend current CEF inputs to four non-art domains without new image encoding."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json
import numpy as np
from current_cef_experiments import ROOT,CACHE
import cef_paths
cef_paths.add_bundled_paths()
from vlm_comparison import VLMEncoder
from pets_arm import SPECS,control_phrases
from r3_probe import EUROSAT_CLIP


def main():
    jobs=json.loads((CACHE/'manifest.json').read_text());jobs=[j for j in jobs if j['dataset'] in ['wikiart','mp100k']]
    for tag,(model,pretrained,label) in SPECS.items():
        enc=None
        for domain in ['dtd','eurosat','aircraft','pets_s5']:
            file=f'{domain}_{tag}.npz';d=np.load(ROOT/'analysis/r3_probe_cache'/file,allow_pickle=True)
            names=d['classes'].tolist();owner=d['owners'];dictionary=json.loads((ROOT/'assets/r3_dicts'/f'probe_{domain}.json').read_text())
            phrases=[s for c in names for s in dictionary[c]]
            assert len(phrases)==len(owner)
            for variant in ['reference','rand_words']:
                dest=CACHE/f'{domain}_{tag}_{variant}.npz'
                if not dest.exists():
                    vp=phrases if variant=='reference' else control_phrases('rand_words',phrases)
                    if variant=='reference':templated=d['c_dclip']
                    else:
                        if enc is None:enc=VLMEncoder(model,pretrained)
                        display=[EUROSAT_CLIP.get(c,c) if domain=='eurosat' else c.replace('_',' ') for c in names]
                        templated=enc.encode_texts([f'{display[o]}, which has {phrase}' for o,phrase in zip(owner,vp)])
                    proto=np.stack([templated[owner==i].mean(0) for i in range(len(names))]);proto/=np.linalg.norm(proto,axis=1,keepdims=True)+1e-12
                    np.savez(dest,full=d['c_full' if variant=='reference' else 'c_rand_words'],owner=owner,proto=proto)
                    dest.with_suffix('.json').write_text(json.dumps(dict(phrases=vp,owners=owner.tolist(),class_names=names,variant=variant),indent=2)+'\n')
                jobs.append(dict(dataset=domain,tag=tag,backbone=label,variant=variant,image_cache='analysis/r3_probe_cache/'+file,evidence_cache=dest.name))
            print('Prepared domain',domain,tag,flush=True)
        if enc is not None:
            del enc
            import torch
            torch.cuda.empty_cache()
    (CACHE/'manifest.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print('Total input jobs:',len(jobs))
if __name__=='__main__':main()
