"""Reproduce frozen final CEF predictions and render deterministic qualitative cases.

Uses CLIP/OpenAI, seed 42, artist-disjoint test pools, and the saved R5 selector.
No tuning or case-based configuration selection is performed. Corrections are
ranked by increase in the true-class margin, with one image per true class;
the regression has the largest margin decrease. Scores are clipped and row
normalized over ALL classes for display. These are not calibrated probabilities.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,hashlib,json,re,shutil,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from zlap_impl import precompute_knn,zlap_transductive
ROOT=Path(os.environ.get('CEF_PROJECT_ROOT',Path(__file__).resolve().parent.parent))
OUT=ROOT/'results/supplementary/qualitative_final'
FIG=ROOT/'paper3/figures'
SEED=42
TAG='ViT-B-32_openai'
BACKBONE='CLIP (OpenAI)'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(x):
    x=np.maximum(x,0);return x/np.maximum(x.sum(1,keepdims=True),1e-30)
def margin(a,y):
    rest=a.copy();rest[np.arange(len(y)),y]=-np.inf
    return a[np.arange(len(y)),y]-rest.max(1)
def choose(y,van,final):
    vp,fp=van.argmax(1),final.argmax(1)
    delta=margin(final,y)-margin(van,y)
    good=np.flatnonzero((vp!=y)&(fp==y));bad=np.flatnonzero((vp==y)&(fp!=y))
    picks=[];seen=set()
    for i in sorted(good,key=lambda i:(-delta[i],int(i))):
        if int(y[i]) in seen:continue
        picks.append((int(i),'Correction'));seen.add(int(y[i]))
        if len(picks)==3:break
    assert len(picks)==3 and len(bad)
    picks.append((int(min(bad,key=lambda i:(delta[i],int(i)))),'Regression'))
    return picks,delta,len(good),len(bad)
def predict(ds):
    from cgpr_lab import pool_paths,_pool_dataframe
    cache=ROOT/'analysis/r4_cache'/f'{ds}_{TAG}.npz'
    job=next(j for j in json.loads((cache.parent/'manifest.json').read_text()) if j['dataset']==ds and j['tag']==TAG)
    assert sha(cache)==job['sha256']
    selection=ROOT/'results/supplementary/round5/select'/f'{ds}_{TAG}_{SEED}.json'
    locks=json.loads((selection.parent.parent/'frozen_selections.json').read_text())
    assert sha(selection)==locks[selection.name]
    frozen=json.loads(selection.read_text());cfg=frozen['baseline_configs']
    assert frozen['input_sha256']==job['sha256'] and not frozen['test_labels_used']
    d=np.load(cache);X=d['embs'];clf=d['clf'];classes=d['class_names'].tolist();C=len(classes)
    vi,ti=d[f'val_{SEED}'],d[f'test_{SEED}'];assert not np.intersect1d(vi,ti).size
    cosine=X[ti]@clf.T;dm=d['dclip'];owner=d['owner_dclip']
    proto=np.stack([dm[owner==i].mean(0) for i in range(C)])
    proto/=np.linalg.norm(proto,axis=1,keepdims=True)+1e-12
    def affinity(name,config):
        if name=='CEF':g,e=config
        else:
            g=config['graph']
            if config['path']=='raw':
                w=config['weight'];return tuple(g),(1-w)*cosine+w*(X[ti]@proto.T)
            e=config['evidence']
        tau,w,db,base=e;b=cosine if base=='cos' else d['Y'][ti]
        if not w:return tuple(g),b
        tex=d['full'] if name=='CEF' else proto;s=X[ti]@tex.T
        if db:s=s-(X[vi]@tex.T).mean(0,keepdims=True)
        a=s/tau;a=np.exp(a-a.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
        if name=='CEF':
            a=np.stack([a[:,d['owner_full']==i].sum(1) for i in range(C)],1)
            a/=a.sum(1,keepdims=True)+1e-12
        if base=='cos':a=a*b.max(1,keepdims=True)
        return tuple(g),(1-w)*b+w*a
    gc,ac=affinity('CEF',cfg['CEF']);gu,au=affinity('CuPL fusion',cfg['CuPL fusion'])
    print(ds,'reconstructing graph neighbours',len(ti),flush=True)
    knn=precompute_knn(X[ti],clf,80)
    def lp(g,a):return zlap_transductive(X[ti],clf,*g,cross_affinity=a,knn_cache=knn)
    pc,pu=lp(gc,ac),lp(gu,au);selected=frozen['selected']['validation selected hybrid']
    kind=selected['kind'];cf=selected['config']
    if kind=='CEF':final=pc
    elif kind=='CuPL fusion':final=pu
    elif kind=='output mixture':final=(1-cf['weight'])*norm(pc)+cf['weight']*norm(pu)
    else:
        sc=norm(ac)*cosine.max(1,keepdims=True);su=norm(au)*cosine.max(1,keepdims=True)
        final=lp(tuple(cf['graph']),(1-cf['weight'])*sc+cf['weight']*su)
    y=d['labels'][ti]
    old=pd.read_csv(ROOT/'results/supplementary/round5/per_seed.csv')
    metrics={}
    for name,a in [('CEF',pc),('CuPL fusion',pu),('validation selected hybrid',final)]:
        pred=a.argmax(1);rec=np.array([(pred[y==i]==i).mean() for i in range(C)])
        row=old[(old.dataset==ds)&(old.backbone==BACKBONE)&(old.seed==SEED)&(old.method==name)].iloc[0]
        assert abs(rec.mean()-row.bAcc)<1e-12 and abs((pred==y).mean()-row.top1)<1e-12
        assert np.max(np.abs(rec-np.array(json.loads(row.recall))))<1e-12
        metrics[name]={'bAcc':float(rec.mean()),'top1':float((pred==y).mean())}
    df=_pool_dataframe(ds)
    meta=pd.concat([df[(df.new_subset==part)&df['style'].isin(classes)] for part in ['val','test']],ignore_index=True)
    assert np.array_equal(np.array([classes.index(s) for s in meta['style']]),d['labels'])
    assert not set(meta.iloc[vi].artist.astype(str).str.strip())&set(meta.iloc[ti].artist.astype(str).str.strip())
    paths=pool_paths(ds,classes);assert len(paths)==len(meta)
    van=norm(d['Y'][ti]);fn=norm(final)
    picks,delta,ng,nb=choose(y,van,fn)
    rows=[]
    for r,(i,kindcase) in enumerate(picks):
        src=Path(paths[ti[i]]);assert src.is_file(),src
        dest=OUT/'images'/f'{ds}_{r+1}{src.suffix.lower()}'
        shutil.copy2(src,dest)
        rows.append(dict(row=r+1,test_index=i,pool_index=int(ti[i]),kind=kindcase,
                         filename=str(meta.iloc[ti[i]].filename),artist=str(meta.iloc[ti[i]].artist),
                         image='images/'+dest.name,image_sha256=sha(dest),
                         gt=classes[int(y[i])],vanilla=classes[int(van[i].argmax())],cef=classes[int(fn[i].argmax())],
                         vanilla_margin=float(margin(van,y)[i]),cef_margin=float(margin(fn,y)[i]),margin_change=float(delta[i])))
    pd.DataFrame(rows).to_csv(OUT/f'{ds}_examples.csv',index=False)
    np.savez_compressed(OUT/f'{ds}_scores.npz',labels=y,test_indices=ti,class_names=d['class_names'],
                        vanilla=van,cef=fn,cef_raw=final,phrase_raw=pc,prototype_raw=pu)
    report=dict(dataset=ds,backbone=BACKBONE,seed=SEED,protocol='artist-disjoint',test_images=len(y),
                selection=selected,selection_sha256=sha(selection),cache_sha256=sha(cache),
                metrics=metrics,corrected_pool=ng,regressed_pool=nb,
                rule='Three corrections by largest increase in true-class margin, one per true class; one regression by largest margin decrease. Ties: test index.',
                score_normalization='Clamp negatives to zero; divide by row sum over all classes; not calibrated probabilities.')
    (OUT/f'{ds}_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(ds,json.dumps(report),flush=True)
    print(pd.DataFrame(rows)[['kind','artist','gt','vanilla','cef','margin_change']].to_string(index=False),flush=True)
def render(ds):
    d=np.load(OUT/f'{ds}_scores.npz');rows=pd.read_csv(OUT/f'{ds}_examples.csv')
    report=json.loads((OUT/f'{ds}_metadata.json').read_text());classes=d['class_names'].tolist()
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,axes=plt.subplots(4,2,figsize=(10.8,9.3),gridspec_kw={'width_ratios':[1.0,2.5],'hspace':.65,'wspace':.5})
    for r,row in rows.iterrows():
        i=int(row.test_index);gt=int(d['labels'][i]);van=d['vanilla'][i];final=d['cef'][i]
        vp,fp=int(van.argmax()),int(final.argmax())
        color='#25754b' if row.kind=='Correction' else '#ad3f36'
        ax=axes[r,0];im=Image.open(OUT/row.image).convert('RGB');ax.imshow(im)
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_color(color);sp.set_linewidth(1.8)
        artist=str(row.artist).replace('_',' ').strip()
        ax.set_title(f'({chr(97+r)}) {row.kind}',loc='left',fontsize=11,color=color,fontweight='bold',pad=5)
        ax.set_xlabel('\n'.join(textwrap.wrap(artist,30)),fontsize=9,labelpad=4)
        top=list(dict.fromkeys([gt,vp,fp,*np.argsort(-final)]))[:4]
        ax=axes[r,1];ys=np.arange(len(top));h=.32
        ax.barh(ys-h/2,[van[c] for c in top],height=h,color='#abb3bc',label='Vanilla')
        ax.barh(ys+h/2,[final[c] for c in top],height=h,color='#246495',label='CEF')
        ax.set_yticks(ys);ax.set_yticklabels(['\n'.join(textwrap.wrap(classes[c],19))+(' *' if c==gt else '') for c in top],fontsize=10)
        ax.invert_yaxis();ax.set_xlim(0,min(1,max(van[top].max(),final[top].max())*1.15))
        ax.grid(axis='x',alpha=.25,linewidth=.5);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',labelsize=9)
        ax.set_xlabel('Normalized class score',fontsize=9,labelpad=2)
        ax.set_title('Vanilla: '+classes[vp]+'   |   CEF: '+classes[fp],fontsize=10,loc='left',pad=7)
    cf=report['selection']['config'];weight=cf.get('weight',0)
    fig.text(.5,.957,f'Artist-disjoint test split  |  Selected graph fusion: prototype weight {weight:.1f}  |  * ground truth',ha='center',fontsize=10)
    handles,labels=axes[0,1].get_legend_handles_labels();fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.99,.942),ncol=2,frameon=False,fontsize=10)
    fig.subplots_adjust(top=.90,bottom=.06,left=.045,right=.985)
    for ext in ['pdf','png']:fig.savefig(FIG/f'qualitative_final_cef_{ds}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--render-only',action='store_true');a=ap.parse_args()
    (OUT/'images').mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    for ds in ['wikiart','mp100k']:
        if not a.render_only:predict(ds)
        render(ds)
if __name__=='__main__':main()
