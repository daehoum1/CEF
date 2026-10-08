"""Attach sample SDs to performance tables using their exact repeated units.

Runs after mean-table generation. Every original printed mean is checked before
replacement. ddof=1; no pooling of seed and backbone variability.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from current_cef_tables import ROOT,KEY
DS=['wikiart','mp100k'];BB=['CLIP (OpenAI)','MetaCLIP','EVA02-CLIP','SigLIP']

def emit(mrt):
 base=mrt.load_artist();img=mrt.load_image()
 r5=pd.read_csv(ROOT/'results/supplementary/round5/per_seed.csv').rename(columns={'bAcc':'bacc'})
 cef=r5[r5.method=='validation selected hybrid']
 r4=pd.read_csv(ROOT/'results/supplementary/round4/per_seed_r4.csv');r4=r4[r4.metric=='bAcc'].rename(columns={'score':'bacc'})
 new=pd.read_csv(ROOT/'results/supplementary/current_cef/per_seed.csv').rename(columns={'bAcc':'bacc'})
 tr=pd.read_csv(ROOT/'results/supplementary/current_cef/transfer_per_seed.csv').rename(columns={'bAcc':'bacc'})
 z=base[base.method=='ZLaP'];van=base[base.method=='Vanilla']
 def vals(frame,ds,bb,metric='bacc'):
  a=frame[(frame.dataset==ds)&(frame.backbone==bb)].sort_values('seed')[metric].to_numpy()
  assert len(a)==5,(ds,bb,len(a));return a
 def method(frame,m):return frame[frame.method==m]
 def variant(v):return new[new.variant==v]
 def pairs(frames,metric='bacc'):
  return [[vals(f,ds,bb,metric) for f in frames] for ds in DS for bb in BB]
 def patch(name,arrays,start=2,scale=100,digits=2,stack=False):
  p=Path(mrt.OUT)/name;lines=p.read_text().splitlines();j=0
  for i,line in enumerate(lines):
   if not line.rstrip().endswith(r'\\') or ' & ' not in line:continue
   cells=line[:-2].rstrip().split(' & ')
   if not any(bb in cells for bb in BB) and not (start==1 and j<len(arrays) and re.match(r'^[+-]?\d',re.sub(r'\\textbf\{','',cells[start]))):continue
   if cells[1:2]==['Backbone']:continue
   row=arrays[j];assert len(cells)-start==len(row),(name,cells,len(row))
   for k,a in enumerate(row,start):
    if a is None:continue # range is a range, not an estimated mean
    a=np.asarray(a,dtype=float);assert len(a)>1 and np.isfinite(a).all()
    mean,sd=a.mean()*scale,a.std(ddof=1)*scale
    old=cells[k];number=re.search(r'[+-]?\d+(?:\.\d+)?',old);assert number,old
    assert abs(float(number[0])-mean)<=.501*10**(-digits),(name,j,k,old,mean)
    m=f'{mean:.{digits}f}';s=f'{sd:.{digits}f}'
    if r'\textbf' in old:m=r'\textbf{'+m+'}'
    cells[k]=(r'\shortstack[r]{'+m+r'\\$\pm$\,'+s+'}') if stack else m+r'\,$\pm$\,'+s
   lines[i]=' & '.join(cells)+r' \\';j+=1
  assert j==len(arrays),(name,j,len(arrays));p.write_text('\n'.join(lines)+'\n')
  print('SD verified:',name,j,'rows')
 # Diagnostic means are on a 0--1 scale and must match archived averages.
 fixed=pd.read_csv(ROOT/'analysis/diagnostic_seed_scores.csv')
 final=pd.read_csv(ROOT/'analysis/final_per_seed.csv')
 patch('tab_attribution.tex',[[vals(fixed,ds,bb,'mask_off_fixed'),vals(final,ds,bb,'CGPR|bacc'),vals(final,ds,bb,'UGSP-kNN|bacc'),vals(final,ds,bb,'CGPR|bacc')] for ds in DS for bb in BB],scale=1,digits=4)
 arms=['Vanilla','CGPR','ZLaP','DCLIP','CuPL','DCLIP + ZLaP','CuPL + ZLaP']
 patch('tab_cef_main.tex',pairs([method(base,m) for m in arms]+[cef]),stack=False)
 patch('tab_hybrid.tex',pairs([method(r5,m) for m in ['CEF','CuPL fusion','output mixture','graph mixture','validation selected hybrid']]),stack=False)
 patch('tab_matched_fusion.tex',pairs([method(r4,m) for m in ['ZLaP','DCLIP fusion','CuPL fusion','MeanPhrase fusion']]+[cef]),stack=False)
 patch('tab_controls.tex',pairs([z,cef]+[variant(v) for v in ['rand_chars','rand_words','shuffle_owner']]),stack=False)
 patch('tab_ablation.tex',[[vals(f,ds,bb) for ds in DS for bb in BB] for f in [cef]+[variant(v) for v in ['no_graph','no_debias','filtered']]],start=1,stack=False)
 patch('tab_hybrid_top1.tex',pairs([van,cef],'top1'))
 draws=new[new.variant.str.startswith('orig_g')]
 patch('tab_r3_gens.tex',[[vals(z,ds,bb),vals(cef,ds,bb),draws[(draws.dataset==ds)&(draws.backbone==bb)].groupby('variant').bacc.mean().to_numpy(),None] for ds in DS for bb in BB])
 patch('tab_transfer.tex',pairs([z,tr[tr.setting=='global_mode'],tr[tr.setting=='leave_pair_out'],cef]))
 ext=tr[tr.setting=='external'];er=ext[ext.variant=='reference'];ed=ext[ext.variant.str.startswith('orig_g')].groupby(KEY).bacc.mean().reset_index()
 patch('tab_external_transfer.tex',pairs([z,cef,er,ed]))
 patch('tab_main_image.tex',[[vals(f,ds,bb,metric) for f in [method(img,m) for m in ['Vanilla','CGPR','ZLaP']]+[variant('image_level')] for metric in ['top1','bacc']] for ds in DS for bb in BB],stack=False)
 old=pd.read_csv(ROOT/'results/supplementary/round3/probe_per_seed.csv').rename(columns={'domain':'dataset'})
 rows=[]
 for ds in DS+['dtd','eurosat','aircraft','pets_s5']:
  b=base[base.dataset==ds] if ds in DS else old[old.dataset==ds]
  c=cef[cef.dataset==ds] if ds in DS else new[(new.dataset==ds)&(new.variant=='reference')]
  rw=new[(new.dataset==ds)&(new.variant=='rand_words')]
  # Aggregate four backbones within each seed, then report seed variability.
  seq=[f.groupby('seed').bacc.mean().sort_index() for f in [method(b,'Vanilla'),method(b,'ZLaP'),c,rw]]
  rows.append([*seq,seq[2]-seq[1],seq[2]-seq[3]])
 patch('tab_r3_probe.tex',rows,start=1,stack=False)
 cp=pd.read_csv(ROOT/'results/supplementary/current_cef/cost.csv')
 # Cost has one serial timing per backbone, so SD describes backbone variation.
 p=Path(mrt.OUT)/'tab_cost.tex';s=p.read_text();lines=s.splitlines();irow=0
 for i,line in enumerate(lines):
  if not any(line.startswith(x+' & ') for x in ['WikiArt','MP100k']):continue
  ds=DS[irow//2];m=['ZLaP','CEF'][irow%2];q=cp[(cp.dataset==ds)&(cp.method==m)];cells=line[:-2].rstrip().split(' & ')
  for k,(col,dig) in enumerate([('select_s',1),('test_s',2),('peak_gb',2)],2):
   assert len(q)==4
   cells[k]=f'{q[col].mean():.{dig}f}'+r'\,$\pm$\,'+f'{q[col].std(ddof=1):.{dig}f}'
  lines[i]=' & '.join(cells)+r' \\';irow+=1
 assert irow==4;p.write_text('\n'.join(lines)+'\n')
