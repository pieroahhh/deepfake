"""Isolated training/shape microbenchmarks on original TRAIN media only.

Selected checkpoint files are read only. Optimizer updates are discarded.
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

import argparse,copy,gc,hashlib,json,os,platform,sqlite3,statistics,sys,time
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--workspace',required=True);P.add_argument('--stage',required=True);P.add_argument('--code-root');P.add_argument('--output');P.add_argument('--assets');A=P.parse_args()
W=_fk_paths.relocate((Path(A.workspace)), __file__);R=_fk_paths.relocate((W/'Deepfake-Detection'), __file__);C=_fk_paths.relocate((Path(A.code_root)), __file__) if A.code_root else R
OUT=_fk_paths.relocate((Path(A.output)), __file__) if A.output else _fk_paths.relocate((R/'runs/table-iii-complete-001'), __file__)
OUT.mkdir(parents=True,exist_ok=True)
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['HF_HOME']=str(_fk_paths.relocate((R/'.cache/huggingface'), __file__));os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((R/'.cache/matplotlib'), __file__))
os.environ['PATH']=str(_fk_paths.relocate((R/'.cache/ffmpeg-tools/static_ffmpeg/bin/win32'), __file__))+os.pathsep+os.environ.get('PATH','')
sys.path[:0]=[str(C),str(_fk_paths.relocate((C/'src'), __file__)),str(_fk_paths.relocate((C/'scripts'), __file__))]
import numpy as np
import torch
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
torch.manual_seed(7)
def read(p):return json.loads(_fk_paths.relocate((Path(p)), __file__).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with _fk_paths.relocate((Path(p)), __file__).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,v):
 p=_fk_paths.relocate((OUT/name), __file__);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sync():torch.cuda.synchronize()
def measure(fn,n=5,warm=1):
 vals=[];result=None
 for i in range(n+warm):
  sync();start=time.perf_counter();result=fn();sync();elapsed=time.perf_counter()-start
  if i>=warm:vals.append(elapsed)
 return {'seconds':vals,'median_seconds':statistics.median(vals)},result
def params(m):return {'total':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad)}
def bench(m,lossfn,optimizer,mode=None,repeats=5):
 (mode or (lambda:m.train()))();rows=[]
 for i in range(repeats+1):
  optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();sync();t=time.perf_counter()
  loss=lossfn();sync();t1=time.perf_counter();loss.backward();sync();t2=time.perf_counter()
  assert torch.isfinite(loss)
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step();sync();t3=time.perf_counter()
  if i:rows.append({'forward_loss_seconds':t1-t,'backward_seconds':t2-t1,'clip_optimizer_seconds':t3-t2,'total_seconds':t3-t,'loss':float(loss.detach()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 return {'samples':rows,'medians':{k:statistics.median(x[k] for x in rows) for k in rows[0]},'parameters':params(m),'warmups':1,'repeats':repeats,'optimizer_state':'new optimizer; warmup initializes state; consecutive discarded updates'}
POP=_fk_paths.relocate((R/'runs/two-day-001/population/manifest.json'), __file__);population=read(POP)
assert all(r['split'] in ('train','validation') for r in population)
TRAIN=[r for r in population if r['split']=='train'];BYID={r['video_id']:r for r in TRAIN}
base={'stage':A.stage,'torch':str(torch.__version__),'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'threads':2,'population_sha256':sha(POP),'seed':7,'scope':'training-only microbenchmark; no model selection, accuracy measurement, or checkpoint writes','sources':{},'examples':[]}
def lock(p):base['sources'][str(p)]=sha(p);return p
def verify():
 for p,h in base['sources'].items():assert sha(p)==h,p
def fk():
 from deepfake_detection.training import load_fk_checkpoint,forward_video,TENSORS
 from deepfake_detection.model import multilevel_loss
 from deepfake_detection.feature_cache import read_cache
 from deepfake_detection.core import Decision
 ck=lock(_fk_paths.relocate((R/'runs/two-day-001/experiment/fk/training/attempt-001/selected.pt'), __file__))
 model,meta=load_fk_checkpoint(ck,device='cuda');model.requires_grad_(True)
 manifest=lock(_fk_paths.relocate((R/'runs/two-day-001/experiment/fk/cache/training-manifest.json'), __file__));cache=read(manifest)
 rows=[]
 for label in (0,1):
  row=next(x for x in cache['train'] if x['label']==label and x['window_intervals'])
  values=read_cache(manifest.parent/row['cache'],row['cache_provenance']);rows.append({**row,**values})
  base['examples'].append({k:row[k] for k in ('video_id','source_sha256','split','window_intervals')})
 config=meta.get('config',{});base['training_config']=config
 def loss(row):
  out,score,labels=forward_video(model,row)
  return multilevel_loss(out,score,row['label'],labels,config.get('window_loss_weight',1.),config.get('bin_loss_weight',1.))
 results=[]
 for row in rows:
  opt=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',.0001),weight_decay=config.get('weight_decay',.01))
  results.append({'video_id':row['video_id'],'usable_windows':len(row['window_intervals']),'benchmark':bench(model,lambda:loss(row),opt)})
 base['cached_training']=results
 row=rows[0];values={k:row[k].cuda() for k in TENSORS};shapes=[]
 with torch.no_grad():
  for count in (1,2,4,8,16):
   x={k:(v[:1].repeat((count,)+(1,)*(v.ndim-1)) if v.ndim>1 and k not in ('visual_times','audio_times') else v) for k,v in values.items()}
   for k in ('visual_times','audio_times'):
    if values[k].ndim==2:x[k]=values[k][:1].repeat(count,1)
   timing,_=measure(lambda:model(**x));shapes.append({'axis':'usable_windows','value':count,**timing})
  original=model.decision
  for bins in (2,5,10,20):
   model.decision=Decision(**{**original.to_dict(),'bins':bins,'top_k':min(original.top_k,bins)})
   try:timing,_=measure(lambda:model(**values));shapes.append({'axis':'bins','value':bins,**timing})
   except ValueError as e:shapes.append({'axis':'bins','value':bins,'status':'unsupported','reason':str(e)})
  model.decision=original
 base['scaling']=shapes
 del opt,model;gc.collect();torch.cuda.empty_cache()
 # Recreate genuine prepared windows and measure frozen feature generation.
 from run_remaining_supplement import FrozenFK
 from deepfake_detection.preparation import prepare_fk_window,fk_source_speech,encode_fk_window
 frozen=FrozenFK();enc=[]
 for row in rows:
  path=_fk_paths.relocate((Path('D:/AV-Deepfake1M')/row['path']), __file__);assert sha(path)==row['source_sha256']
  speech=fk_source_speech(path,row['duration'],frozen.vad,pcm16=True)
  for start,end in row['window_intervals'][:2]:
   prep=prepare_fk_window(path,start,row['duration'],frozen.detector,speech,fit_to_frame=frozen.fit,pcm16=True)
   assert prep['status']=='USABLE'
   with torch.no_grad():timing,features=measure(lambda:encode_fk_window(prep,frozen.encoders),n=3)
   enc.append({'video_id':row['video_id'],'start':start,'scope':'prepared two-second window; frozen EfficientNet and wav2vec2; excludes media preparation',**timing})
 base['frozen_feature_generation']=enc
 # Direct matched online/cache replay on one complete TRAIN video.
 row=rows[0];path=_fk_paths.relocate((Path('D:/AV-Deepfake1M')/row['path']), __file__);speech=fk_source_speech(path,row['duration'],frozen.vad,pcm16=True)
 prepared=[prepare_fk_window(path,a,row['duration'],frozen.detector,speech,fit_to_frame=frozen.fit,pcm16=True) for a,b in row['window_intervals']]
 assert all(x['status']=='USABLE' for x in prepared)
 def generate():
  with torch.no_grad():parts=[encode_fk_window(p,frozen.encoders) for p in prepared]
  values={k:torch.cat([p[k] for p in parts],dim=0) for k in ('visual_features','audio_features')}
  for k in ('visual_times','audio_times'):values[k]=parts[0][k]
  return {**row,**values}
 online=generate();parity={k:float((online[k].cpu()-row[k].cpu()).abs().max()) for k in ('visual_features','audio_features')}
 assert max(parity.values())<=5e-6,parity
 model=frozen.model;initial={k:v.detach().clone() for k,v in model.state_dict().items()};matched={}
 for name in ('cached','online'):
  model.load_state_dict(initial);model.requires_grad_(True);torch.manual_seed(7)
  opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
  def matched_loss():
   current=generate() if name=='online' else row
   out,score,labels=forward_video(model,current)
   return multilevel_loss(out,score,row['label'],labels,config['window_loss_weight'],config['bin_loss_weight'])
  matched[name]=bench(model,matched_loss,opt,repeats=3)
  matched[name]['final_parameters']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 final_error=max(float((matched['cached']['final_parameters'][k]-matched['online']['final_parameters'][k]).abs().max()) for k in initial)
 for v in matched.values():del v['final_parameters']
 matched['feature_max_absolute_errors']=parity;matched['final_weight_max_absolute_error']=final_error;matched['usable_windows']=len(prepared);matched['video_id']=row['video_id'];matched['scope']='complete video update; online starts from prepared CPU faces/audio and runs frozen encoders each step; cached starts from verified stored CPU feature tensors; both exclude media preprocessing and disk I/O'
 assert final_error<=5e-6,final_error
 base['matched_online_cached']=matched
 base['online_cost_note']='Online per-video estimate = N_u * measured frozen-feature time per window + measured cached head update. Decode/preparation and verified cache I/O are separate; this is not a directly timed online training epoch.'
def avfake():
 from baselines.common.training import load_checkpoint
 from run_avfakenet_one_epoch import make_model
 from baselines.avfakenet.model import classification_loss
 from baselines.avfakenet.normalized_recovery import set_training_mode,trainable_groups
 ident=read(_fk_paths.relocate((R/'runs/avfakenet-one-epoch-001/identity.json'), __file__));profile=ident['profile'];records=read(_fk_paths.relocate((R/'runs/avfakenet-one-epoch-001/prepared-index.json'), __file__))
 cache=_fk_paths.relocate((Path(ident['prepared_cache_provenance']['source'])), __file__);base['branches']={}
 for branch in ('visual','audio'):
  ck=lock(_fk_paths.relocate((R/('runs/avfakenet-one-epoch-001/'+branch+'/best.pt')), __file__));state=load_checkpoint(ck);m=make_model(profile,branch,state['model_state'],'cuda');del state
  arrays=[];labels=[]
  for label in (0,1):
   for record in records:
    if record['split']!='train' or branch in record['reasons']:continue
    candidates=[_fk_paths.relocate((R/'runs/avfakenet-machine-b-prepared-001'/record['key']), __file__),_fk_paths.relocate((cache/record['key']), __file__),_fk_paths.relocate((R/'runs/avfakenet-normalized-pilot-001/prepared'/record['key']), __file__)]
    p=next((p for p in candidates if (_fk_paths.relocate((p/(branch+'.npy')), __file__)).exists() and (branch=='audio' or (_fk_paths.relocate((p/'visual-labels.npy'), __file__)).exists())),None)
    if p is None:continue
    source=BYID[record['video_id']]
    if sha(_fk_paths.relocate((p/(branch+'.npy')), __file__))!=record['files'][branch+'.npy']:continue
    if branch=='audio':
     if record['audio_label']!=label:continue
     value=np.load(_fk_paths.relocate_value((_fk_paths.relocate((p/'audio.npy'), __file__)), __file__),allow_pickle=False);y=label
    else:
     ys=np.load(_fk_paths.relocate_value((_fk_paths.relocate((p/'visual-labels.npy'), __file__)), __file__),allow_pickle=False);idx=np.flatnonzero(ys==label)
     if not len(idx):continue
     value=np.array(np.load(_fk_paths.relocate_value((_fk_paths.relocate((p/'visual.npy'), __file__)), __file__),mmap_mode='r')[int(idx[0])],copy=True);y=label
    assert sha(_fk_paths.relocate((p/(branch+'.npy')), __file__))==record['files'][branch+'.npy']
    arrays.extend([value]*4);labels.extend([y]*4);base['examples'].append({'branch':branch,'video_id':source['video_id'],'source_sha256':source['source_sha256'],'split':'train','label':y});break
  x=torch.from_numpy(np.stack(arrays))
  if branch=='visual':x=x.permute(0,3,1,2).contiguous().float()/255
  x=x.cuda();y=torch.tensor(labels,device='cuda');assert len(y)==8
  groups=trainable_groups(m);cfg=profile['training'];opt=torch.optim.AdamW([{'params':[p for k,ps in groups.items() if k!='classifier' for p in ps],'lr':1e-8},{'params':groups['classifier'],'lr':1e-4}],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['epsilon'])
  base['branches'][branch]={'batch_size':8,'input_shape':list(x.shape),'benchmark':bench(m,lambda:classification_loss(m(x),y,cfg['label_smoothing']),opt,lambda:set_training_mode(m,'off'))}
  del m,opt,x,y;gc.collect();torch.cuda.empty_cache()
def avtenet():
 from baselines.avtenet.model import load_component_profile,FeatureFusion,native_binary_loss
 from baselines.common.training import load_checkpoint,move
 from baselines.common.population import target
 plan=read(_fk_paths.relocate((C/'runs/auto-next-001/test-plan.json'), __file__));profile=load_component_profile(_fk_paths.relocate((C/plan['config']), __file__));stage=A.stage
 ck=lock(_fk_paths.relocate((C/plan['checkpoints'][stage]), __file__));state=load_checkpoint(ck)
 if stage=='fusion':
  m=FeatureFusion(profile).cuda();m.load_state_dict(state['model_state']);del state
  db=_fk_paths.relocate((C/'runs/worker-c/avtenet/fusion-features/features.sqlite'), __file__);con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);selected=[]
  for vid,start,end,payload,source in con.execute('SELECT video_id,start,end,features,source_sha256 FROM features ORDER BY video_id,start'):
   if vid not in BYID:continue
   row=BYID[vid];assert row['source_sha256']==source;y=target(row,'audiovisual',start,end)
   if y is None or any(z[1]==y for z in selected):continue
   selected.append((np.frombuffer(payload,dtype='<f4').copy().reshape(3,768),y));base['examples'].append({'video_id':vid,'start':start,'end':end,'source_sha256':source,'split':'train'})
   if len(selected)==2:break
  assert len(selected)==2;con.close();arr=np.stack([x[0] for x in selected]);inputs={s:torch.from_numpy(arr[:,i,:]).cuda() for i,s in enumerate(('visual','audio','audiovisual'))};y=torch.tensor([x[1] for x in selected],device='cuda')
  opt=torch.optim.Adam(m.parameters(),lr=.0001);base['benchmark_optimizer']={'name':'Adam','learning_rate':.0001};base['cached_training']=bench(m,lambda:native_binary_loss(m(inputs),y),opt,repeats=10);return
 from baselines.avtenet.pipeline import Prepared,stage_model
 from baselines.avtenet.assets import registry
 from baselines.common.media import detector
 assets=registry(A.assets);m,backend,proof=stage_model(stage,profile,assets);m.load_state_dict(state['model_state']);del state;m.cuda().requires_grad_(True)
 detect,_=detector('cuda');prep=Prepared(None,stage,backend,assets,detect,reuse_video_decoder=True,cache_face_detections=True,fast_vivit_processor=stage=='visual');examples=[]
 # Use two distinct original TRAIN videos with known window targets.
 for row in TRAIN:
  y=target(row,stage,0.,2.)
  if y is None or any(z[1]==y for z in examples):continue
  path=(_fk_paths.relocate((Path('/data')), __file__) if os.name!='nt' else _fk_paths.relocate((Path('D:/AV-Deepfake1M')), __file__))/row['path']
  try:
   assert sha(path)==row['source_sha256'];inputs=move(prep.inputs(path,0.,row),torch.device('cuda'))
  except (ValueError,OSError) as e:print('SKIP',row['video_id'],str(e)[:100],flush=True);continue
  examples.append((inputs,y));base['examples'].append({'video_id':row['video_id'],'source_sha256':row['source_sha256'],'split':'train','start':0.,'end':2.})
  if len(examples)==2:break
 assert len(examples)==2
 base['experts']=[];base['benchmark_optimizer']={'name':'Adam','learning_rate':.00001};initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 for inputs,label in examples:
  m.load_state_dict(initial)
  y=torch.tensor([label],device='cuda');opt=torch.optim.Adam(m.parameters(),lr=.00001)
  try:result=bench(m,lambda:native_binary_loss(m(inputs),y),opt,repeats=3)
  except torch.cuda.OutOfMemoryError:
   # Report the real hardware limit; never silently benchmark a smaller model.
   base['experts'].append({'status':'OUT_OF_MEMORY','batch_size':1,'scope':'full expert gradient update in float32'});break
  base['experts'].append({'batch_size':1,'benchmark':result});del opt
  m.load_state_dict(initial);m.eval()
  with torch.no_grad():timing,_=measure(lambda:m.extract_features(inputs),n=3)
  base['experts'][-1]['frozen_feature_generation']=timing
if __name__=='__main__':
 print('START',A.stage,flush=True)
 if A.stage=='fk':fk()
 elif A.stage=='avfakenet':avfake()
 else:avtenet()
 verify();base['source_hashes_unchanged']=True;write(A.stage+'/training-cost.json',base);print('COMPLETE',A.stage,flush=True)
