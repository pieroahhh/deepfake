"""Publish retained Table III measurements. No new training or GPU work.

Run with the existing .venv-gpu Python. Progress is atomic JSON plus a log.
"""

# Workspace relocation: portable paths, historical metadata stays immutable.
import sys as _fk_sys
from pathlib import Path as _FkBootstrapPath
_fk_workspace = next((p for p in _FkBootstrapPath(__file__).resolve().parents
                      if (p / 'workspace-layout.json').is_file()), None)
if _fk_workspace is None:
    _fk_hint = _FkBootstrapPath(__import__('os').environ.get('FK_WORKSPACE_ROOT', '/inputs'))
    if (_fk_hint / 'workspace-layout.json').is_file():
        _fk_workspace = _fk_hint
if _fk_workspace is None:
    raise RuntimeError('workspace-layout.json is missing')
if str(_fk_workspace / '06_Tools/workspace') not in _fk_sys.path:
    _fk_sys.path.insert(0, str(_fk_workspace / '06_Tools/workspace'))
import workspace_paths as _fk_paths

import collections,csv,html,json,os,shutil,statistics,sys,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
W=_fk_paths.workspace_root(__file__);R=_fk_paths.relocate((W/'Deepfake-Detection'), __file__);OUT=_fk_paths.relocate((R/'runs/table-iii-complete-001'), __file__);PUB=_fk_paths.relocate((OUT/'publication'), __file__)
os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((R/'.cache/matplotlib'), __file__))
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__)), __file__).parent))
from journal_visuals_common import Pack,Sources,setup_style,axes_grid,sha
import matplotlib.pyplot as plt
import numpy as np
START=time.time();TOTAL=8
def write(p,v):
 p=_fk_paths.relocate((Path(p)), __file__);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def progress(stage,done,status='RUNNING',**extra):
 state={'status':status,'stage':stage,'processed':done,'total':TOTAL,'percent':round(done/TOTAL*100,1),'elapsed_seconds':round(time.time()-START,1),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
 write(_fk_paths.relocate((OUT/'progress.json'), __file__),state);print(json.dumps(state,ensure_ascii=False),flush=True)
def median(v):return statistics.median(v)
def main():
 OUT.mkdir(parents=True,exist_ok=True);progress('validate-retained-measurements',0);setup_style();sources=Sources();pack=Pack(PUB,sources)
 stages={s:sources.json(_fk_paths.relocate((OUT/s/'training-cost.json'), __file__),'training') for s in ('fk','avfakenet','audio','visual','audiovisual','fusion')}
 detail=sources.json(_fk_paths.relocate((OUT/'detail-cost.json'), __file__),'detail');old=_fk_paths.relocate((R/'runs/table-iii-cost-001'), __file__)
 inference=sources.csv(_fk_paths.relocate((old/'table-iii-comparison.csv'), __file__),'inference');models=sources.json(_fk_paths.relocate((old/'table-iii-models.json'), __file__),'inference')['models']
 assert all(v['source_hashes_unchanged'] for v in stages.values()) and detail['source_hashes_unchanged']
 assert stages['fk']['matched_online_cached']['final_weight_max_absolute_error']<=5e-6
 assert all(w['prepared_parity'] and all(s.get('max_output_error',0)<=5e-6 for s in w['samples']) for w in detail['windows'])
 for stage,v in stages.items():
  assert all(e['split']=='train' for e in v['examples'])
  for path,digest in v['sources'].items():
   p=_fk_paths.relocate((Path(path)), __file__)
   if str(path).startswith('/workspace/'):p=_fk_paths.relocate((W/'avh-a/code'/str(path)[len('/workspace/'):]), __file__)
   assert sha(p)==digest,'Selected checkpoint changed: '+str(p)
   sources.register(p,'selected-checkpoints')
 progress('inference-figures-and-tables',1)
 def table(slug,title,headers,rows,caption,group='training',**kw):pack.table(slug,title,headers,rows,caption,[group],**kw)
 def figure(slug,title,caption,fig,data,group='training'):pack.figure(slug,title,caption,fig,data,[group])
 common=[r for r in inference if r['population']=='common_assessed'];names=['FK','AVFakeNet','AVTENet (staged)'];components=['decode_io','preprocessing','feature_extraction','fusion_evidence','aggregation','forensic_record']
 colors=['#bdbdbd','#8dd3c7','#80b1d3','#bebada','#fdb462','#fb8072']
 fig,ax=plt.subplots(figsize=(7.16,3.4));bottom=np.zeros(len(common));data=[]
 for key,color in zip(components,colors):
  values=np.array([float(r[key]) for r in common]);ax.bar(names,values,bottom=bottom,label=key.replace('_',' '),color=color,edgecolor='black',linewidth=.35);bottom+=values
  data.extend({'model':r['model'],'component':key,'median_seconds_per_clip':float(r[key])} for r in common)
 ax.set_ylabel('Seconds per clip');axes_grid(ax);ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.29),frameon=False)
 figure('inference-components','Inference cost on the common assessed clips','Five common assessed clips, one warmup and three measured passes. Component bars are medians of exclusive self times; their sum need not equal the median total. AVTENet is a sum of separately executed stages. Model loading excluded.',fig,data,'inference')
 table('inference-cost','Measured inference cost on common assessed clips',['Model','Clips','Total s','Preprocess s','Features s','Evidence s','Aggregation s','Record s'],[[r['model'],r['assessed'],*[f'{float(r[k]):.5f}' for k in ('seconds','preprocessing','feature_extraction','fusion_evidence','aggregation','forensic_record')]] for r in common],'Decode/I/O is included in total but omitted as a separate table column. Baseline localization and forensic records are unavailable; a zero measured record time does not establish support.','inference')
 fig,ax=plt.subplots(figsize=(7.16,3.2));rows=[]
 for i,(name,clips) in enumerate(models.items()):
  values=[x['seconds'] for x in clips.values()];ax.scatter(np.full(len(values),i),values,alpha=.7,s=20,label=name);rows.extend({'model':name,'video_id':vid,'seconds':v['seconds'],'assessed':v['assessed']} for vid,v in clips.items())
 ax.set_xticks(range(len(models)),names);ax.set_ylabel('Seconds per attempted clip');axes_grid(ax)
 figure('inference-dispersion','Per-clip inference cost and assessment','All 12 attempted clips. Unassessed clips can stop early, so this panel is workload dispersion and must be read with the assessment counts.',fig,rows,'inference')
 progress('training-figures-and-tables',2)
 training=[]
 def add(name,unit,benches):
  assert benches
  samples=[x for b in benches for x in b['samples']];training.append({'name':name,'unit':unit,'samples':samples,'medians':{k:median(s[k] for s in samples) for k in samples[0]},'parameters':benches[0]['parameters']})
 add('FK cached','complete video (5 or 8 windows)',[x['benchmark'] for x in stages['fk']['cached_training']])
 for branch,v in stages['avfakenet']['branches'].items():add('AVFakeNet '+branch,'batch of 8 prepared inputs',[v['benchmark']])
 for stage in ('audio','visual','audiovisual'):
  valid=[v['benchmark'] for v in stages[stage]['experts'] if 'benchmark' in v]
  if not valid:raise RuntimeError('Expert benchmark unavailable: '+stage)
  add('AVTENet '+stage,'one prepared 2-second window',valid)
 add('AVTENet fusion','batch of 2 cached windows',[stages['fusion']['cached_training']])
 labels=[r['name'].replace(' ','\n',1) for r in training];x=np.arange(len(training));fig,ax=plt.subplots(figsize=(7.16,3.5));bottom=np.zeros(len(x));flat=[]
 for key,color in zip(('forward_loss_seconds','backward_seconds','clip_optimizer_seconds'),('#80b1d3','#fb8072','#b3de69')):
  vals=np.array([r['medians'][key]*1000 for r in training]);ax.bar(x,vals,bottom=bottom,label=key.replace('_seconds','').replace('_',' '),color=color,edgecolor='black',linewidth=.3);bottom+=vals
  flat.extend({'model':r['name'],'unit':r['unit'],'phase':key,'median_ms':r['medians'][key]*1000} for r in training)
 ax.set_xticks(x,labels,fontsize=6.5);ax.set_ylabel('Milliseconds per update');ax.legend(frameon=False,ncol=3);axes_grid(ax)
 train_note='Our selected models with discarded optimizer updates on two original TRAIN examples per branch/expert. One warmup, then 3 expert, 5 FK/AVFakeNet, or 10 fusion repeats. Prepared input and feature caches exclude raw media ingestion. Batch units differ; this is not a model speed ranking. New Adam/AdamW optimizer state; historical training duration is not inferred.'
 figure('training-phases','Measured forward, backward and optimizer cost',train_note,fig,flat)
 table('training-cost','Training update cost measured on our training inputs',['Stage','Unit','Forward ms','Backward ms','Optimizer ms','Total ms'],[[r['name'],r['unit'],*[f'{r["medians"][k]*1000:.3f}' for k in ('forward_loss_seconds','backward_seconds','clip_optimizer_seconds','total_seconds')]] for r in training],train_note)
 table('parameters-memory','Parameters and peak allocated CUDA memory',['Stage','Parameters','Trainable','Peak allocated MiB','Unit'],[[r['name'],r['parameters']['total'],r['parameters']['trainable'],f'{r["medians"]["peak_allocated_bytes"]/2**20:.1f}',r['unit']] for r in training],'Allocated memory from torch; excludes the GPU driver and other processes. Each stage executes separately. Optimizer buffers after warmup are included; this is not simultaneous three-expert deployment memory.')
 for metric,title,ylabel,slug in [('peak_allocated_bytes','Training memory by stage','Peak allocated CUDA memory (MiB)','training-memory'),('ratio','Measured training-to-forward cost ratio','Update time / forward-and-loss time','training-forward-ratio')]:
  fig,ax=plt.subplots(figsize=(7.16,3.2));values=[r['medians']['total_seconds']/r['medians']['forward_loss_seconds'] if metric=='ratio' else r['medians'][metric]/2**20 for r in training];ax.bar(x,values,color='#80b1d3',edgecolor='black',linewidth=.35);ax.set_xticks(x,labels,fontsize=6.5);ax.set_ylabel(ylabel);axes_grid(ax)
  if metric=='ratio':ax.axhline(3,color='black',linestyle='--',label='heuristic factor 3');ax.legend(frameon=False)
  figure(slug,title,'Different update units as stated in the training table. Ratios include clipping and optimizer work; a factor of three is a heuristic, not an equality. '+train_note,fig,[{'model':r['name'],'value':v,'unit':r['unit']} for r,v in zip(training,values)])
 progress('cache-and-scaling-figures',3)
 matched=stages['fk']['matched_online_cached'];cached=matched['cached']['medians']['total_seconds'];online=matched['online']['medians']['total_seconds'];speedup=online/cached
 fig,ax=plt.subplots(figsize=(4.9,3.0));ax.bar(['Online encoders','Cached features'],[online*1000,cached*1000],color=['#fb8072','#80b1d3'],edgecolor='black',linewidth=.4);ax.set_ylabel('Milliseconds per complete-video update');axes_grid(ax)
 figure('fk-online-cached','FK online and cached feature training',matched['scope']+f'. Eight usable windows; identical initial weights and optimizer settings. Feature and final weight maximum errors are zero. Observed speedup {speedup:.2f}x applies to this example and scope.',fig,[{'mode':k,'median_seconds':matched[k]['medians']['total_seconds'],'usable_windows':matched['usable_windows']} for k in ('online','cached')])
 table('fk-cache-parity','Matched FK online/cache replay and parity',['Mode','Windows','Update ms','Peak MiB','Feature max error','Final weight max error'],[[k,matched['usable_windows'],f'{matched[k]["medians"]["total_seconds"]*1000:.3f}',f'{matched[k]["medians"]["peak_allocated_bytes"]/2**20:.1f}',max(matched['feature_max_absolute_errors'].values()),matched['final_weight_max_absolute_error']] for k in ('online','cached')],matched['scope']+'. Both modes retain the same frozen encoder model in GPU memory. This absolute memory includes resident encoders.')
 epochs=[1,2,5,10,20,50];build=max(0,matched['online']['medians']['forward_loss_seconds']-matched['cached']['medians']['forward_loss_seconds']);fig,ax=plt.subplots(figsize=(5.2,3.0));ax.plot(epochs,[online*1000]*len(epochs),'o-',label='Online measured update');ax.plot(epochs,[(cached+build/e)*1000 for e in epochs],'s-',label='Cached + estimated one-time features / epochs');ax.set_xscale('log');ax.set_xlabel('Assumed epochs');ax.set_ylabel('Amortized milliseconds per update');ax.legend(frameon=False,fontsize=6.5);axes_grid(ax)
 figure('cache-amortization','Estimated amortization of frozen feature generation','Projection from one measured eight-window TRAIN video, not measured epochs. Build cost is the difference of median forward times. Excludes preprocessing, cache writes and cache reads; no whole-training speedup claim.',fig,[{'epochs':e,'online_seconds':online,'estimated_cached_amortized_seconds':cached+build/e} for e in epochs])
 for axis,title in [('usable_windows','FK workload versus usable windows'),('bins','FK workload versus temporal bins')]:
  values=[v for v in stages['fk']['scaling'] if v['axis']==axis and 'median_seconds' in v];fig,ax=plt.subplots(figsize=(5.0,3.0));ax.plot([v['value'] for v in values],[v['median_seconds']*1000 for v in values],'o-');ax.set_xlabel(axis.replace('_',' ').capitalize());ax.set_ylabel('Milliseconds per head forward');axes_grid(ax)
  figure('fk-scaling-'+axis,title,'Actual FK forward with authentic cached TRAIN features. Window workloads repeat the same feature window; bin settings reuse trained weights. One warmup and five repeats; no accuracy interpretation or empirical proof of asymptotic complexity.',fig,values)
 table('scaling-workloads','FK shape workloads and measured forward costs',['Axis','Value','Median ms','Measured repeats'],[[v['axis'],v['value'],f'{v["median_seconds"]*1000:.4f}',len(v['seconds'])] for v in stages['fk']['scaling'] if 'median_seconds' in v],'Shape microbenchmarks only. Small B and synchronization/validation overhead can obscure quadratic attention growth.')
 progress('detail-and-forensic-figures',4)
 detail_rows=[]
 for w in detail['windows']:
  labels_set=set().union(*(s['timers'] for s in w['samples'][1:]))
  for label in sorted(labels_set):detail_rows.append({'video_id':w['video_id'],'component':label,'median_self_seconds':median(s['timers'].get(label,{}).get('self_seconds',0) for s in w['samples'][1:]),'median_calls':median(s['timers'].get(label,{}).get('calls',0) for s in w['samples'][1:])})
 selected=['pre.quality_calculation','evidence.attention_matrix_softmax_value','feature.gru','evidence.reliability','feature.bin_pool'];fig,ax=plt.subplots(figsize=(7.16,3.1));values=[median(r['median_self_seconds'] for r in detail_rows if r['component']==k)*1000 for k in selected];ax.barh([k.split('.',1)[1].replace('_',' ') for k in selected],values,color='#80b1d3',edgecolor='black',linewidth=.3);ax.set_xlabel('Milliseconds per prepared two-second window');ax.grid(axis='x')
 figure('inline-components','Additional quality and attention measurements','Two TRAIN windows, one warmup and three measured passes. Runtime copies add explicit inline timers; prepared arrays and frozen model outputs match the original. Quality calculation here is the explicitly instrumented preparation assignments; shared tracker/image calculations remain in their enclosing components. Timers synchronize and add overhead.',fig,detail_rows,'detail')
 table('detailed-components','Additional inline timing components',['Component','Median self ms','Median calls/window'],[[k,f'{median(r["median_self_seconds"] for r in detail_rows if r["component"]==k)*1000:.5f}',median(r['median_calls'] for r in detail_rows if r['component']==k)] for k in selected],'Exclusive synchronized self times, pooled across two TRAIN windows. These timings are separate from the retained 12-clip end-to-end study.','detail')
 cpu=detail['cpu_scaling'];hashes=[r for r in cpu if r['operation']=='sha256_in_memory_source_prefix'];records=[r for r in cpu if r['operation']=='canonical_actual_record'];fig,ax=plt.subplots(figsize=(5.2,3.0));ax.loglog([r['bytes'] for r in hashes],[r['median_seconds']*1000 for r in hashes],'o-');ax.set_xlabel('Source bytes hashed');ax.set_ylabel('SHA-256 milliseconds');ax.grid(True,which='both',alpha=.3)
 figure('hash-scaling','Hashing cost versus source payload size','Actual prefixes of our TRAIN media in memory, 20 repeats after two warmups. Excludes file I/O and OS cache effects. No reference-paper measurements.',fig,hashes,'detail')
 if records:
  fig,ax=plt.subplots(figsize=(5.2,3.0));ax.scatter([r['bytes']/1024 for r in records],[r['median_seconds']*1000 for r in records]);ax.set_xlabel('Actual canonical record size (KiB)');ax.set_ylabel('Serialization milliseconds');axes_grid(ax)
  figure('record-serialization','Canonical serialization of our retained records','Four actual FK forensic records, 20 measured CPU repeats after two warmups. Serialization only; persistence and verification are separately profiled in the inference study.',fig,records,'detail')
 table('hashing-serialization','Measured CPU hashing and canonical serialization',['Operation','Bytes','Median ms','Repeats'],[[r['operation'],r['bytes'],f'{r["median_seconds"]*1000:.6f}',len(r['seconds'])] for r in cpu],'Uses our source media and actual canonical records. Hashes are in-memory payload operations, not cold disk measurements.','detail')
 # Retained full-record runs expose actual hash counts, including derived artifacts.
 calls=[]
 for p in sorted((_fk_paths.relocate((old/'fk'), __file__)).glob('[0-9][0-9].json')):
  v=sources.json(p,'inference');samples=v['samples'][1:];calls.append({'video_id':v['row']['video_id'],'hash_calls':median(s['timers'].get('record.file_hash',{}).get('calls',0) for s in samples),'seconds':median(s['timers'].get('record.file_hash',{}).get('self_seconds',0) for s in samples)})
 table('forensic-hash-calls','Actual file-hash calls in full FK record execution',['Clip','File hash calls','Hash self ms'],[[str(i+1),r['hash_calls'],f'{r["seconds"]*1000:.3f}'] for i,r in enumerate(calls)],'Counts include source integrity, derived artifacts and verification. The implementation re-hashes the source during window preparation; the analytical two-source-hash expression is an idealized policy and is not the measured call count.','inference')
 progress('coverage-and-analytical-table',5)
 coverage=[['Preprocessing','Measured','12 inference clips + 2 detailed train windows','Decode, detector, tracking/crops, VAD; inline quality subset'],['Feature extraction','Measured','Own frozen checkpoints','EfficientNet, GRU, wav2vec2, DST, AST, ViViT, AV-HuBERT'],['Cross-modal / evidence','Measured','FK and AVTENet fusion','Explicit inline attention; projections/gating/heads'],['Localization / aggregation','Measured / capability limited','FK timelines and native aggregation','Baselines have no native localization output'],['Forensic record','Measured','FK persisted records','Hashes, canonical construction, persistence and verification; baseline N/A'],['Training','Microbenchmarked','Original TRAIN examples only','Forward, backward, clipping and optimizer; no new selected model'],['Online / cached','Direct matched FK replay','One complete 8-window TRAIN video','Feature and final-weight parity; prepared-input scope'],['Big-O trends','Analytical + shape probes','Actual implementation and bounded shapes','Experiments do not prove an asymptotic bound'],['Historical full-training duration','Not remeasured','Existing logs only','No reconstructed full-epoch or 3-model training-time claim']]
 table('coverage','Table III coverage and remaining limitations',['Category','Evidence','Scope','Limit'],coverage,'Measurements use only our retained experiment data. Reference papers supply visual conventions only. Baselines are our explicit local adaptations; paper-reported benchmark values are not imported.','inference',grid=True,widths=[1,1,1.8,2.5])
 analytical=[['AVFakeNet adaptation','Frame detection/crops + clip mel','n_f visual DST + audio DST','Hard OR O(1)','Majority vote O(n_f); no native localization','N/A','Separate branch forward + backward + optimizer'],['AVTENet adaptation','Frame detection/crops, mel and filterbanks','ViViT + AST + AV-HuBERT + MS-TCN per usable window','Linear fusion O(N_w D_sum)','Window mean O(N_w); no native localization','N/A','S1 expert updates; S2 frozen features + fusion update'],['Ours (FK)','n_f T_det + T_VAD + N T_quality + decode','N_u (T_EN + T_GRU + T_w2v + B T_proj)','N_u [O(B d^2 + B^2 d) + B(T_disc + T_R) + |M| T_head]','O(N_u B), plus implementation top-k','Actual source/derived/record hashes + O(N + N_u B)','Online: T_enc + T_update; cached amortized: T_enc / n_ep + T_update']]
 table('analytical-cost','Analytical computational cost of our implemented schemes',['Scheme','Preprocessing','Feature extraction','Fusion / evidence','Localization / aggregation','Forensic record','Training'],analytical,'Analytical expressions describe our implementations, including explicit adaptations. T_update = measured forward-and-loss + backward + clipping/optimizer, rather than an exact 3x forward identity. Disk/cache I/O and provenance checks must be stated separately. B is the number of bins, d is the projected feature dimension, N is attempted windows, N_u is usable windows, and N_w is expert windows. Dense masked attention still constructs a B by B matrix.','inference',grid=True,widths=[1.1,1.4,1.6,2.0,1.5,1.5,1.8])
 (_fk_paths.relocate((PUB/'TABLE_III_analytical.tex'), __file__)).write_text(r'''% Own implementation analysis. Requires booktabs, array, graphicx.
\begin{table*}[t]\centering\scriptsize
\caption{Analytical computational cost of our implemented audio-visual schemes.}
\label{tab:analytical-cost-implemented}
\resizebox{\textwidth}{!}{\begin{tabular}{lllllll}\toprule
Scheme & Preprocessing & Features & Fusion/evidence & Localization/aggregation & Forensic record & Training \\ \midrule
AVFakeNet adapt. & Face detection/crops + mel & $n_fT^V_{DST}+T^A_{DST}$ & $O(1)$ hard OR & $O(n_f)$ majority; localization N/A & N/A & Branch updates \\
AVTENet adapt. & Face/lip crops + mel/filterbanks & $N_w(T_{ViViT}+T_{AST}+T_{AVH}+T_{TCN})$ & $O(N_wD_{sum})$ & $O(N_w)$ mean; localization N/A & N/A & S1 experts; S2 frozen features + fusion \\
Ours (FK) & $n_fT_{det}+T_{VAD}+NT_q+T_{decode}$ & $N_u(T_{EN}+T_{GRU}+T_{w2v}+BT_{proj})$ & $N_u[O(Bd^2+B^2d)+B(T_{disc}+T_R)+|M|T_{head}]$ & $O(N_uB)$ plus top-$k$ & Actual hashes $+O(N+N_uB)$ & On: $T_{enc}+T_{update}$; Ca: $T_{enc}/n_{ep}+T_{update}$ \\
\bottomrule\end{tabular}}
\parbox{\textwidth}{\footnotesize Analytical costs, not measurements copied from reference papers. $T_{update}$ includes forward/loss, backward, clipping and optimizer. Cache I/O is separate. The current implementation repeats source hashing during window preparation; a fixed two-source-hash budget is not its measured execution.}
\end{table*}
''',encoding='utf-8')
 progress('publish-gallery-and-pdf',6)
 # Bind every exported item to its own numerical sources.
 for item in pack.items:
  item['source_files']=sorted(set().union(*(sources.groups.get(g,set()) for g in item['source_groups'])))
  assert item['source_files']
  write(_fk_paths.relocate((PUB/'metadata'/(item['stem']+'.json')), __file__),item)
 write(_fk_paths.relocate((PUB/'catalog.json'), __file__),{'items':pack.items});write(_fk_paths.relocate((PUB/'source-manifest.json'), __file__),{'files':sources.files,'groups':{k:sorted(v) for k,v in sources.groups.items()},'reference_papers_used_as_data':False})
 cards=[]
 for item in pack.items:cards.append(f'<article><h2>{html.escape(item["label"]+". "+item["title"])}</h2><a href="{item["png"]}"><img loading="lazy" src="{item["png"]}"></a><p>{html.escape(item["caption"])}</p><a href="{item["data"]}">CSV</a> | <a href="{item["svg"]}">SVG</a></article>')
 (_fk_paths.relocate((PUB/'index.html'), __file__)).write_text('<!doctype html><meta charset="utf-8"><title>Table III — our measurements</title><style>body{max-width:1050px;margin:30px auto;font:16px Georgia;color:#182733}article{border-top:1px solid #bbb;padding:20px 0}img{max-width:100%}p{line-height:1.5}h2{font-size:20px}</style><h1>Table III: our measurements and implementation analysis</h1><p>Own data only. Training updates are discarded microbenchmarks; selected models stay unchanged. Each visual links its numerical data.</p>'+''.join(cards),encoding='utf-8')
 sys.path.insert(0,str(_fk_paths.relocate((W/'.review-cache/archive-v0/1PiHw63lG2QUPDDZ'), __file__)));import pymupdf as fitz
 doc=fitz.open()
 for item in pack.items:
  pix=fitz.Pixmap(str(_fk_paths.relocate((PUB/item['png']), __file__)));height=519*pix.height/pix.width
  page_height=max(842.,height+310.);page=doc.new_page(width=595,height=page_height)
  page.insert_textbox(fitz.Rect(38,28,557,85),item['label']+'. '+item['title'],fontname='tibo',fontsize=13)
  page.insert_image(fitz.Rect(38,96,557,96+height),pixmap=pix)
  remain=page.insert_textbox(fitz.Rect(38,112+height,557,page_height-38),item['caption']+'\n\nData: '+item['data'],fontname='tiro',fontsize=9,lineheight=1.2)
  assert remain>=0,'Caption overflow: '+item['stem']
 doc.save(_fk_paths.relocate((PUB/'TABLE_III_FIGURES_AND_TABLES.pdf'), __file__),garbage=4,deflate=True);doc.close()
 qa=_fk_paths.relocate((OUT/'qa'), __file__);qa.mkdir(exist_ok=True);doc=fitz.open(_fk_paths.relocate((PUB/'TABLE_III_FIGURES_AND_TABLES.pdf'), __file__))
 for i,page in enumerate(doc):
  page.get_pixmap(matrix=fitz.Matrix(.8,.8)).save(_fk_paths.relocate((qa/f'page-{i+1:02d}.png'), __file__))
  for block in page.get_text('blocks'):
   assert block[0]>=0 and block[1]>=0 and block[2]<=page.rect.width+.5 and block[3]<=page.rect.height+.5
 doc.close();progress('verify-and-package',7)
 changed=[p for p,v in sources.files.items() if sha(_fk_paths.relocate((W/p), __file__))!=v['sha256']];assert not changed,changed
 counts=dict(collections.Counter(i['kind'] for i in pack.items));validation={'counts':counts,'all_own_source_hashes_unchanged':True,'training_examples_are_train_only':True,'matched_fk_feature_max_error':max(matched['feature_max_absolute_errors'].values()),'matched_fk_final_weight_max_error':matched['final_weight_max_absolute_error'],'fk_matched_speedup_in_scope':speedup,'pdf_pages_rendered':len(pack.items),'pdf_text_bounds_checked':True,'visual_review':'Rendered QA pages provided; manual final inspection still required.','reference_papers_used_as_data':False,'new_training_in_this_publisher':False}
 write(_fk_paths.relocate((PUB/'validation.json'), __file__),validation)
 report=f'''# Table III results\n\nCompleted measurements cover six training stages and the retained 12-clip inference study. The publication folder contains {counts.get('figure',0)} figures and {counts.get('table',0)} tables, with CSV/SVG/LaTeX exports.\n\nThe matched eight-window FK example took {online*1000:.3f} ms online versus {cached*1000:.3f} ms cached ({speedup:.2f}x in the stated prepared-input scope). Frozen features and final weights matched exactly. This is a short training replay, not full training, and does not establish an end-to-end epoch speedup.\n\nRemaining limits: only two TRAIN examples per expert/branch; no full-epoch remeasurement; no proof of Big-O from small shape probes; no monolithic three-expert memory/timing claim. Baselines do not produce native localization or forensic records. The fixed two-source-hash formula understates actual repeated source hashing.\n\nOpen publication/index.html or publication/TABLE_III_FIGURES_AND_TABLES.pdf. Rendered pages for manual review are under qa/.\n'''
 (_fk_paths.relocate((OUT/'REPORT.md'), __file__)).write_text(report,encoding='utf-8')
 (_fk_paths.relocate((OUT/'REPORT_TH.md'), __file__)).write_text(f'ผลวัด Table III: เพิ่ม {counts.get("figure",0)} กราฟ และ {counts.get("table",0)} ตารางจากข้อมูลของเราเอง\n\nFK ทดลองวิดีโอ train 8 หน้าต่าง: online {online*1000:.3f} ms / cached {cached*1000:.3f} ms เร็วขึ้น {speedup:.2f} เท่า เฉพาะขอบเขตที่วัด โดยฟีเจอร์และน้ำหนักหลังอัปเดตตรงกันทุกค่า\n\nการเทรนเป็น microbenchmark สั้น ๆ ไม่ได้เทรนใหม่ทั้งชุด และไม่ได้ใช้ข้อมูลทดสอบสำหรับอัปเดตโมเดล ค่า Big-O เป็นการวิเคราะห์โค้ด ค่าคาดการณ์การใช้แคชหลาย epoch แสดงแยกจากค่าที่วัดจริง\n\nยังต้องตรวจหน้า PDF ด้วยสายตาจากโฟลเดอร์ qa ก่อนนำไปใช้ในบทความ\n',encoding='utf-8')
 reproduce=_fk_paths.relocate((PUB/'reproduce'), __file__);reproduce.mkdir(exist_ok=True)
 for name in ('finish_table_iii_complete.py','journal_visuals_common.py','training_cost_probe.py','detail_cost_probe.py'):shutil.copy2(_fk_paths.relocate((Path(__file__)), __file__).with_name(name),_fk_paths.relocate((reproduce/name), __file__))
 write(_fk_paths.relocate((PUB/'complete.json'), __file__),{'files':{p.relative_to(PUB).as_posix():sha(p) for p in PUB.rglob('*') if p.is_file() and p.name!='complete.json'},'counts':counts})
 archive=_fk_paths.relocate((OUT/'TABLE_III_COMPLETE.zip'), __file__)
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in PUB.rglob('*'):
   if p.is_file():z.write(p,'publication/'+p.relative_to(PUB).as_posix())
  for name in ('REPORT.md','REPORT_TH.md'):z.write(_fk_paths.relocate((OUT/name), __file__),name)
 progress('complete',TOTAL,'FINISHED',counts=counts,publication=str(PUB),archive=str(archive))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  progress('failed',0,'FAILED',error=str(e));traceback.print_exc();raise
