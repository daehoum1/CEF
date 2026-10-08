"""Validate coverage, reproduction, external provenance and reported R4 effects."""
import ast
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('CEF_PROJECT_ROOT',Path(__file__).resolve().parent.parent))
SUPP=ROOT/'results/supplementary'
checks=0


def check(name,condition):
    global checks
    assert bool(condition),name
    checks+=1
    print('[OK]',name)


def main():
    raw=pd.read_csv(SUPP/'round4/per_seed_r4.csv')
    keys=['dataset','backbone','seed','method','metric']
    check('40 target jobs, 16 methods, two separately reported metrics',
          len(raw)==1280 and not raw.duplicated(keys).any() and
          (raw.groupby(['method','metric']).size()==40).all())
    check('all reported accuracies are finite probabilities',
          raw.score.notna().all() and raw.score.between(0,1).all())
    reference=pd.read_csv(SUPP/'artist_disjoint/per_seed_results.csv').rename(columns={'experiment':'method'})
    for method in ['Vanilla','ZLaP','CEF','CEF w=0']:
        matched=raw[raw.method==method].merge(reference,on=keys,suffixes=('_new','_old'),validate='one_to_one')
        check('fresh runner reproduces '+method,
              len(matched)==80 and (matched.score_new-matched.score_old).abs().max()<1e-12)
    ba=raw[raw.metric=='bAcc']
    recalls=ba.recall.apply(lambda s:np.mean(json.loads(s)))
    check('bAcc equals saved per-class mean recall',np.max(np.abs(recalls-ba.score))<1e-12)
    source_path=SUPP/'round3/probe_per_seed.csv'
    protocol=json.loads((SUPP/'round4/protocol.json').read_text())
    check('external source file matches frozen checksum',
          hashlib.sha256(source_path.read_bytes()).hexdigest()==protocol['source_sha256'])
    source=pd.read_csv(source_path)
    selected=source[(source.method=='CEF') & source.domain.isin(protocol['source_domains'])]
    check('external selection uses only four non-art datasets and 80 runs',
          len(selected)==80 and set(selected.domain)=={'aircraft','dtd','eurosat','pets_s5'})
    votes=collections.Counter(str(ast.literal_eval(x)) for x in selected.cfg)
    mode=min(votes,key=lambda x:(-votes[x],x))
    check('frozen external configuration equals source-only deterministic mode',
          json.loads(json.dumps(ast.literal_eval(mode)))==protocol['fixed_config'])
    external=raw[raw.method.str.startswith('CEF external ')]
    check('all external targets use the same configuration without target tuning',
          len(external)==480 and (external.val_evals==0).all() and external.val_score.isna().all()
          and all(json.loads(c)==protocol['fixed_config'] for c in external.selected_config))
    for name in ['DCLIP','CuPL','MeanPhrase']:
        a=raw[raw.method==name+' fusion'].set_index(['dataset','backbone','seed','metric'])
        b=raw[raw.method==name+' calibrated fusion'].set_index(['dataset','backbone','seed','metric'])
        check(name+' raw-path candidate never lowers selected validation score',
              (a.val_score-b.val_score>=-1e-12).all() and (a.val_evals-b.val_evals==450).all())
        z=raw[raw.method=='ZLaP'].set_index(['dataset','backbone','seed','metric'])
        check(name+' includes independently tuned ZLaP on validation',
              (a.val_score-z.val_score>=-1e-12).all())
    # Each printed mean, range and win count is recomputed from the raw CSV;
    # a negative result is valid and is never treated as a test failure.
    import r4_tables
    frame=r4_tables.load()
    macros=(ROOT/'paper3/review_numbers.tex').read_text()
    def value(name):
        match=re.search(r'\\newcommand\{\\R'+name+r'\}\{((?:[^{}]|\{[^{}]*\})*)\}',macros)
        assert match,name
        return match.group(1).replace(r'\ensuremath{-}','-')
    comparisons=[('CEF','DCLIP fusion','MatchedDclip'),('CEF','CuPL fusion','MatchedCupl'),
                 ('CEF','MeanPhrase fusion','MatchedMeanPhrase'),
                 ('CEF external full','ZLaP','ExternalRef'),
                 ('CEF external draws','ZLaP','ExternalDraw')]
    for a,b,prefix in comparisons:
        aa=frame[frame.method==a].set_index(['dataset','backbone','seed']).bacc
        bb=frame[frame.method==b].set_index(['dataset','backbone','seed']).bacc
        diff=aa-bb; pairs=diff.groupby(['dataset','backbone']).mean()
        for suffix,number in [('Mean',pairs.mean()),('Min',pairs.min()),('Max',pairs.max())]:
            check(prefix+suffix+' matches the raw paired differences',abs(float(value(prefix+suffix))-100*number)<=.051)
        check(prefix+' pair/run counts match the raw paired differences',
              int(value(prefix+'Pos'))==int((pairs>0).sum()) and
              int(value(prefix+'SeedPos'))==int((diff>0).sum()))
    print(f'{checks}/{checks} R4 checks passed')


if __name__=='__main__':
    main()
