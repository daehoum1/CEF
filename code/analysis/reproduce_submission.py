"""Regenerate submission tables and verify claims from the released bundle.

This repository holds the code and the splits. The per-seed outputs live in the
reproducibility archive; extract it and point --bundle at it:
    python reproduce_submission.py --bundle /path/to/supplementary_material \\
                                   --output /tmp/cef-reproduced

No original image dataset, embeddings, GPU, API keys, network or author-specific paths needed.
The eight displayed images are included for integrity checks.
Run: python code/analysis/reproduce_submission.py --output /tmp/cef-reproduced
"""
import argparse
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    bundle=args.bundle.resolve(); output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Use an empty output directory; existing files are not overwritten.')
    output.mkdir(parents=True,exist_ok=True)
    # Table snapshots are called expected_tables/ in the code repository and
    # manuscript/ in the reproducibility archive.
    tables_dir='expected_tables' if (bundle/'expected_tables').is_dir() else 'manuscript'
    for src,dst in [('results','results/supplementary'),('dictionaries/round3','assets/r3_dicts'),
                    ('splits','paper3/supplementary_material/splits'),(tables_dir,'paper3')]:
        shutil.copytree(bundle/src,output/dst,dirs_exist_ok=True)
    analysis=output/'analysis'
    analysis.mkdir()
    for p in (bundle/'results').glob('*.csv'):
        shutil.copy2(p,analysis/p.name)
    scripts=bundle/'code/analysis'
    env=dict(os.environ,CEF_PROJECT_ROOT=str(output),PYTHONDONTWRITEBYTECODE='1')
    # Scripts whose inputs are large binary dumps kept out of the code repository.
    needs={'verify_final_qualitative.py':['results/supplementary/qualitative_final/wikiart_scores.npz']}
    skipped=[]
    for script in ['make_review_tables.py','verify_review_claims.py','verify_r4.py','verify_hybrid.py','verify_final_qualitative.py','verify_current_cef.py','verify_presubmission.py']:
        missing=[r for r in needs.get(script,[]) if not (output/r).exists()]
        if missing:
            skipped.append((script,missing[0]))
            print(f'[skip] {script}: {missing[0]} is not in this distribution; '
                  'it is in the full reproducibility archive.')
            continue
        subprocess.run([sys.executable,'-B',str(scripts/script)],cwd=analysis,env=env,check=True)
    # Compare all active tabular inputs after whitespace normalization. The table
    # files the manuscript includes are listed in expected_tables/referenced_tables.txt.
    listing=bundle/tables_dir/'referenced_tables.txt'
    if listing.exists():
        inputs={l.strip() for l in listing.read_text().splitlines() if l.strip() and not l.startswith('#')}
    else:
        inputs=set()
        for p in (bundle/tables_dir).rglob('*.tex'):
            if p.parent.name!='tables':
                inputs.update(re.findall(r'\\(?:input|CEFTable)\{tables/([^}]+)\}',p.read_text()))
    for name in sorted(inputs):
        filename=name if name.endswith('.tex') else name+'.tex'
        expected=(bundle/tables_dir/'tables'/filename).read_text()
        actual=(output/'paper3/tables'/filename).read_text()
        normalize=lambda t:re.sub(r'\s+','',t)
        if normalize(expected)!=normalize(actual):
            raise AssertionError(f'Table differs: {filename}')
    print(f'Reproduced and matched {len(inputs)} referenced tables. Output: {output}')
    if skipped:
        print('Skipped (inputs not redistributed here): ' + ', '.join(s for s,_ in skipped))


if __name__=='__main__':
    main()
