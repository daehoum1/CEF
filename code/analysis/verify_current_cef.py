"""Verify complete-CEF reruns, frozen selections, intervention counts and claims."""
from pathlib import Path
import os,json,re,hashlib,collections
import numpy as np
import pandas as pd
ROOT=Path(os.environ.get('CEF_PROJECT_ROOT',Path(__file__).resolve().parent.parent))
OUT=ROOT/'results/supplementary/current_cef'
KEY=['dataset','backbone','seed']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    d=pd.read_csv(OUT/'per_seed.csv');assert len(d)==641 and not d.duplicated(KEY+['variant']).any()
    art=d[d.dataset.isin(['wikiart','mp100k'])]
    variants=['rand_chars','rand_words','shuffle_owner','filtered','no_debias','no_graph','image_level']+[f'orig_g{i}' for i in range(5)]
    for v in variants:assert len(art[art.variant==v])==40,v
    for ds in ['dtd','eurosat','aircraft','pets_s5']:
        for v in ['reference','rand_words']:assert len(d[(d.dataset==ds)&(d.variant==v)])==20,(ds,v)
    hashes=json.loads((OUT/'input_hashes.json').read_text())
    for f in sorted((OUT/'evaluate').glob('*.json')):
        r=json.loads(f.read_text());sfile=OUT/'select'/f.name;s=json.loads(sfile.read_text())
        assert r['selection_sha256']==sha(sfile),f.name
        assert not s['test_labels_used'] and s['selected']['bAcc']==max(c['bAcc'] for c in s['candidates'])
        assert s['evidence_sha256']==hashes['analysis/current_cef_cache/'+s['job']['evidence_cache']]
        assert abs(np.mean(r['recall'])-r['bAcc'])<1e-12
        assert 0<=r['top1']<=1 and 0<=r['bAcc']<=1
        if r['variant']=='no_graph':assert all(c['kind']!='graph mixture' for c in s['candidates'])
        if r['variant']=='no_debias':
            for c in s['branches'].values():
                if c['path']=='calibrated' and c['evidence'][1]:assert not c['evidence'][2]
    # All saved costs reproduce reference CEF/ZLaP outcomes; timing values must be measured.
    cost=pd.read_csv(OUT/'cost.csv');assert len(cost)==16
    r5=pd.read_csv(ROOT/'results/supplementary/round5/per_seed.csv')
    cef=r5[r5.method=='validation selected hybrid'].rename(columns={'bAcc':'bacc'})
    r4=pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv');r4=r4[r4.metric=='bAcc'].rename(columns={'score':'bacc'})
    for r in cost.itertuples():
        src=cef if r.method=='CEF' else r4[r4.method=='ZLaP']
        expected=src[(src.dataset==r.dataset)&(src.backbone==r.backbone)&(src.seed==42)].iloc[0].bacc
        assert abs(r.bAcc-expected)<1e-12,(r,expected)
        assert r.select_s>0 and r.test_s>0 and r.peak_gb>0
    env=json.loads((OUT/'cost_environment.json').read_text());assert env['serial'] and env['threads_per_process']==2
    smoke=art[art.variant=='reference'].iloc[0]
    original=cef[(cef.dataset==smoke.dataset)&(cef.backbone==smoke.backbone)&(cef.seed==smoke.seed)].iloc[0]
    assert abs(smoke.bAcc-original.bacc)<1e-12 and abs(smoke.top1-original.top1)<1e-12
    # External setting is derived solely from all eighty non-art validation selections.
    ep=json.loads((OUT/'external_protocol.json').read_text());assert len(ep['source_files'])==80 and not ep['test_labels_used']
    votes=[]
    for name,h in ep['source_files'].items():
        f=OUT/'select'/name;assert sha(f)==h;s=json.loads(f.read_text())
        assert s['job']['dataset'] in ['dtd','eurosat','aircraft','pets_s5'] and s['job']['variant']=='reference'
        clean=dict(branches=s['branches'],selected={k:s['selected'][k] for k in ['kind','config']})
        votes.append(json.dumps(clean,sort_keys=True))
    counts=collections.Counter(votes);mode=json.loads(min(counts,key=lambda x:(-counts[x],x)))
    assert ep['config']==mode
    tr=pd.read_csv(OUT/'transfer_per_seed.csv').rename(columns={'bAcc':'bacc'})
    assert len(tr)==320 and not tr.duplicated(KEY+['variant','setting']).any()
    for f in (OUT/'transfer').glob('*.json'):
        r=json.loads(f.read_text());assert abs(np.mean(r['recall'])-r['bAcc'])<1e-12
        if r['setting']=='external':assert r['config']==ep['config']
    # Independently recompute every reported CEF comparison macro.
    import make_review_tables as mt
    base=mt.load_artist();new=d.rename(columns={'bAcc':'bacc'});na=new[new.dataset.isin(['wikiart','mp100k'])]
    z=base[base.method=='ZLaP'];cases=[]
    for name,key in [('DCLIP fusion','Dclip'),('CuPL fusion','Cupl'),('MeanPhrase fusion','MeanPhrase')]:cases.append((cef,r4[r4.method==name],'Current'+key,'bacc'))
    for v,key in [('rand_chars','Chars'),('rand_words','Words'),('shuffle_owner','Owners'),('filtered','Filter'),('no_debias','Debias'),('no_graph','NoGraph')]:cases.append((cef,na[na.variant==v],'Current'+key,'bacc'))
    draw=na[na.variant.str.startswith('orig_g')].groupby(KEY).bacc.mean().reset_index()
    cases += [(draw,z,'CurrentGenZlap','bacc'),(cef,draw,'CurrentRefDraw','bacc'),(cef,base[base.method=='Vanilla'],'CurrentTopVanilla','top1')]
    image=mt.load_image()
    for name,key in [('ZLaP','Zlap'),('CGPR','Cgpr')]:cases.append((na[na.variant=='image_level'],image[image.method==name].groupby(KEY).bacc.mean().reset_index(),'CurrentImage'+key,'bacc'))
    old=pd.read_csv(ROOT/'results/supplementary/round3/probe_per_seed.csv').rename(columns={'domain':'dataset'})
    for ds,k in [('dtd','Dtd'),('eurosat','Euro'),('aircraft','Air'),('pets_s5','Pet')]:
        a=new[(new.dataset==ds)&(new.variant=='reference')]
        cases += [(a,old[(old.dataset==ds)&(old.method=='ZLaP')],'CurrentProbe'+k+'Zlap','bacc'),(a,new[(new.dataset==ds)&(new.variant=='rand_words')],'CurrentProbe'+k+'Random','bacc')]
    er=tr[(tr.setting=='external')&(tr.variant=='reference')]
    ed=tr[(tr.setting=='external')&tr.variant.str.startswith('orig_g')].groupby(KEY).bacc.mean().reset_index()
    cases += [(er,z,'CurrentExternalRef','bacc'),(ed,z,'CurrentExternalDraw','bacc'),(cef,er,'CurrentExternalCost','bacc')]
    text=(ROOT/'paper3/review_numbers.tex').read_text();checked=0
    def macro(name):
        m=re.search(r'\\newcommand\{\\R'+name+r'\}\{((?:[^{}]|\{[^{}]*\})*)\}',text);assert m,name
        return float(m.group(1).replace(r'\ensuremath{-}','-'))
    for a,b,prefix,col in cases:
        aa=a.set_index(KEY)[col];bb=b.set_index(KEY)[col];assert len(aa)==len(bb) and aa.index.is_unique and bb.index.is_unique
        diff=aa-bb;assert not diff.isna().any();pairs=diff.groupby(['dataset','backbone']).mean()
        vals={'Mean':100*pairs.mean(),'Min':100*pairs.min(),'Max':100*pairs.max(),'Pos':int((pairs>0).sum()),'N':len(pairs),'SeedPos':int((diff>0).sum()),'SeedN':len(diff)}
        for suffix,v in vals.items():assert abs(macro(prefix+suffix)-v)<.051,(prefix,suffix,v,macro(prefix+suffix));checked+=1
    print(f'Current CEF verified: {len(d)} intervention/domain runs, 320 transfer runs, 16 timing measurements, {checked} comparison macros.')
if __name__=='__main__':main()
