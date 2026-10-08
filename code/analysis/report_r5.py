"""Verify frozen selections and summarize every R5 outcome without winner filtering."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent.parent/'results/supplementary/round5'
lock=json.loads((R/'frozen_selections.json').read_text());assert len(lock)==40
counts={};validated=0
for name,h in lock.items():
    f=R/'select'/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==h
    q=json.loads(f.read_text());c=q['validation_candidates']
    best=max(x['bAcc'] for x in c)
    assert q['selected']['validation selected hybrid']['bAcc']==best
    counts[q['selected']['validation selected hybrid']['kind']]=counts.get(q['selected']['validation selected hybrid']['kind'],0)+1
    assert not q['test_labels_used'];validated+=1
d=pd.read_csv(R/'per_seed.csv');assert len(d)==200
for rec,score in zip(d.recall,d.bAcc):assert abs(np.mean(json.loads(rec))-score)<1e-12
m=d.groupby(['dataset','backbone','method']).bAcc.mean().unstack()*100
methods=['CEF','CuPL fusion','output mixture','graph mixture','validation selected hybrid']
lines=['# CEF/CuPL improvement experiment','',
'This is exploratory development on previously inspected benchmarks, not an independent confirmatory evaluation. All 40 validation selections were frozen before this run evaluated test performance. Balanced accuracy selected every configuration; Top-1 is secondary under those same selections. Original CEF and CuPL configurations were reproduced from the previous experiment.','',
'All candidates and results are retained. No paper result has been replaced.','',
'| Dataset | Backbone | '+' | '.join(methods)+' |',
'|---|---|'+'---:|'*len(methods)]
for (ds,b),row in m.iterrows():lines.append('| '+ds+' | '+b+' | '+' | '.join(f'{row[k]:.3f}' for k in methods)+' |')
lines+=['','| Method | Mean bAcc (%) | vs CEF (pp) | vs CuPL fusion (pp) | Pairs above CuPL |','|---|---:|---:|---:|---:|']
for name in methods:
    delta=m[name]-m['CuPL fusion'];lines.append(f'| {name} | {m[name].mean():.3f} | {(m[name]-m.CEF).mean():+.3f} | {delta.mean():+.3f} | {int((delta>0).sum())}/8 |')
lines+=['','Validation-selected hybrid choices: '+json.dumps(counts)+'.',
f'Checks: {validated} frozen selection hashes; validation maxima; all 200 class-recall means; existing CEF/CuPL bAcc scores reproduced to 1e-12 by the runner.',
'','The post-propagation mixture uses two propagation outputs. The graph mixture combines normalized cross-modal affinities and requires its additional validation graph search. These are extensions using CuPL, not evidence that the unchanged CEF beats CuPL on all pairs.',
'','Implementation note: an initial evaluation launch stopped before evaluation because JSON converts graph tuples to lists. Equality was normalized through JSON serialization; the original selection source and protocol hashes are preserved. No search or scoring formula changed.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
shutil.copy2(Path(__file__).parent/'r5_improve.py',R/'evaluation_source.py')
print('\n'.join(lines))
