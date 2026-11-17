"""Discarded training updates on equal original two-second TRAIN windows.

No selected checkpoint is written. Media preprocessing is excluded from timing.
Each presentation runs each method's declared native stage workload; this is
not an equal-optimizer-step/full-epoch comparison. All stage counts are recorded.
"""
import argparse,copy,gc,hashlib,json,math,os,platform,statistics,sys,time
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--workspace',required=True);P.add_argument('--stage',required=True);P.add_argument('--output');P.add_argument('--extract-only',action='store_true');A=P.parse_args()
W=Path(A.workspace);sys.path.insert(0,str(W/'06_Tools/workspace'))
from workspace_paths import workspace_path,relocate
R=workspace_path('main');C=workspace_path('avhubert');OUT=Path(A.output) if A.output else workspace_path('main_results')/'training-scaling-001'
OUT.mkdir(parents=True,exist_ok=True)
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['HF_HOME']=str(workspace_path('runtime')/'caches/main/huggingface');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PATH']=str(workspace_path('runtime')/'caches/main/ffmpeg-tools/static_ffmpeg/bin/win32')+os.pathsep+os.environ.get('PATH','')
root=C if A.stage.startswith('avt-') else R
sys.path[:0]=[str(root),str(root/'src'),str(root/'scripts')]
import numpy as np
import torch
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,p)
def sync():torch.cuda.synchronize()
def move(x):return {k:move(v) for k,v in x.items()} if isinstance(x,dict) else x.cuda()
def lock(p):
 p=Path(p);protected[str(p)]=sha(p);return p
protected={}
def progress(stage,**extra):print(json.dumps({'stage':stage,**extra}),flush=True)
def rawfile(i):return OUT/'prepared'/f'window-{i:02d}.npz'
def prepare():
 from run_remaining_supplement import FrozenFK
 from deepfake_detection.preparation import fk_source_speech,prepare_fk_window
 from baselines.common.media import detector,faces,audio
 from baselines.avfakenet.preprocessing import prepare_audio
 manifest=lock(workspace_path('main_results')/'two-day-001/population/manifest.json');population=read(manifest)
 row=next(r for r in population if r['video_id']=='train/id00012/Pdni5zyZJYA/00082/real.mp4')
 assert row['split']=='train' and row['label']==row['audio_label']==row['visual_label']==0
 source=Path('D:/AV-Deepfake1M')/row['path'];assert sha(source)==row['source_sha256']
 fk=FrozenFK();torch.backends.cudnn.allow_tf32=False
 detect,proof=detector('cuda');identity=read(workspace_path('main_results')/'avfakenet-one-epoch-001/identity.json');profile=identity['profile']
 speech=fk_source_speech(source,row['duration'],fk.vad,pcm16=True);intervals=[]
 for i,start in enumerate((0.,1.)):
  progress('prepare-window',index=i,start=start)
  ours=prepare_fk_window(source,start,row['duration'],fk.detector,speech,fit_to_frame=fk.fit,pcm16=True);assert ours['status']=='USABLE',ours['reasons']
  avf=faces(source,start,2.,detect,fps=25,pixels=128,prefetch_decode=False)
  visual=faces(source,start,2.,detect,fps=16,pixels=224,prefetch_decode=False)
  av=faces(source,start,2.,detect,fps=25,pixels=224,prefetch_decode=False)
  waveform=audio(source,start,2.)
  import cv2
  lips=np.stack([cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) for rgb in av['lips']])
  rawfile(i).parent.mkdir(exist_ok=True)
  np.savez(rawfile(i),ours_faces=ours['faces'],ours_audio=ours['audio'],ours_times=ours['visual_times'],ours_quality=fk.mapper(ours['quality_measurements']).numpy(),
           avf_faces=avf['faces'],avf_audio=prepare_audio(waveform,16000,profile).numpy(),visual_faces=visual['faces'],av_lips=lips,waveform=waveform)
  intervals.append({'index':i,'start':start,'end':start+2.,'visual_frames':len(avf['faces']),'source_sha256':row['source_sha256'],'prepared_sha256':sha(rawfile(i))})
 protocol={'schema':1,'source_manifest_sha256':sha(manifest),'row':row,'windows':intervals,'window_counts':[1,2,4,8],'repeats':3,'seed':7,
   'scope':'equal source window presentations, repeated cyclically from two genuine TRAIN windows; native stage workloads and update counts differ; prepared inputs only',
   'excluded':'media preparation, disk/cache I/O, model/optimizer initialization; selected-checkpoint writes, validation and model selection',
   'ours':'one head update per presentation, with either online frozen encoding or precomputed features',
   'avfakenet':'all 50 native 25-fps visual frames in batches of 8 plus one audio update per two-second window',
   'avtenet':'one update of each of AST, ViViT and AV-HuBERT/TCN plus one cached fusion-head update per window; expert-to-fusion cache creation excluded',
   'precision':'float32; CUDA matmul TF32 false; cuDNN TF32 false; two CPU threads; stages run sequentially',
   'source_files':protected}
 write(OUT/'protocol.json',protocol);progress('prepared',windows=len(intervals))
def load_windows():
 protocol=read(OUT/'protocol.json');windows=[]
 assert protocol['row']['split']=='train'
 for item in protocol['windows']:
  path=rawfile(item['index']);assert sha(path)==item['prepared_sha256'];windows.append(dict(np.load(path,allow_pickle=False)))
 return protocol,windows
def curves(model,opt,loss_step,mode,stage,updates):
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 def reset():
  model.load_state_dict(initial);mode();opt.zero_grad(set_to_none=True);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
  for state in opt.state.values():
   for key,value in state.items():
    if isinstance(value,torch.Tensor):value.zero_()
    elif key=='step':state[key]=0
 mode();loss_step(0);sync() # warm kernels and optimizer state, including partial visual minibatches
 results=[]
 for count in (1,2,4,8):
  samples=[]
  for repeat in range(3):
   reset();torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter()
   for i in range(count):loss_step(i%2)
   sync();elapsed=time.perf_counter()-start
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
   samples.append({'repeat':repeat,'seconds':elapsed,'optimizer_updates':count*updates,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
  results.append({'window_presentations':count,'source_seconds_presented':count*2.,'samples':samples,'median_seconds':statistics.median(s['seconds'] for s in samples),'updates_per_presentation':updates})
  progress(stage,count=count,median_seconds=results[-1]['median_seconds'])
 model.load_state_dict(initial);return results
def update(m,opt,loss):
 assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
def extraction_curve(forward,stage):
 results=[]
 with torch.no_grad():
  forward(0);forward(1);sync()
  for count in (1,2,4,8):
   samples=[]
   for repeat in range(3):
    sync();start=time.perf_counter()
    for i in range(count):forward(i%2)
    sync();samples.append({'repeat':repeat,'seconds':time.perf_counter()-start})
   results.append({'window_presentations':count,'samples':samples,'median_seconds':statistics.median(s['seconds'] for s in samples)})
   progress('extraction '+stage,count=count,median_seconds=results[-1]['median_seconds'])
 write(OUT/('extract-'+A.stage+'.json'),{'stage':A.stage,'curves':results,'scope':'native feature-return boundary on same prepared windows; output transfer included; classification head included for DST native branch API','source_files':protected,'checkpoint_unchanged':all(sha(p)==h for p,h in protected.items())})
def ours(protocol,windows):
 from run_remaining_supplement import FrozenFK
 from deepfake_detection.preparation import encode_fk_window
 from deepfake_detection.model import multilevel_loss
 from deepfake_detection.core import supervision
 ck=lock(workspace_path('main_results')/'two-day-001/experiment/fk/training/attempt-001/selected.pt')
 fk=FrozenFK();torch.backends.cudnn.allow_tf32=False;model=fk.model.requires_grad_(True);config=fk.metadata['config'];prepared=[];cached=[]
 for w in windows:
  p={'faces':w['ours_faces'],'audio':w['ours_audio'],'visual_times':w['ours_times']};prepared.append(p)
  with torch.no_grad():values=encode_fk_window(p,fk.encoders)
  cached.append({k:v.detach().clone() for k,v in values.items()})
 quality=[torch.from_numpy(w['ours_quality']).cuda() for w in windows];labels=supervision(0,True,[],[(0.,2.)],bins=model.decision.bins)
 if A.extract_only:
  model.eval();fk.encoders.eval()
  def forward(i):
   values=move(encode_fk_window(prepared[i],fk.encoders));states,_=model.gru(values['visual_features'])
   visual=model.visual(model.pool(states,values['visual_times']));audio=model.audio(model.pool(values['audio_features'],values['audio_times']))
   return visual.cpu(),audio.cpu()
  extraction_curve(forward,'Ours');return
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};answer={};last={}
 for mode in ('online','cached'):
  model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
  def step(i):
   if mode=='online':
    with torch.no_grad():values=encode_fk_window(prepared[i],fk.encoders)
   else:values=cached[i]
   outputs=model(**move(values),quality=quality[i]);score=model.aggregate_video(outputs['window_scores'])
   loss=multilevel_loss(outputs,score,0,labels,config['window_loss_weight'],config['bin_loss_weight']);update(model,opt,loss)
  answer[mode]=curves(model,opt,step,lambda:model.train(),'Ours '+mode,1)
  del opt
 write(OUT/'ours.json',{'stage':'ours','curves':answer,'source_files':protected,'checkpoint_unchanged':all(sha(p)==h for p,h in protected.items())})
def avf(protocol,windows,branch):
 from baselines.common.training import load_checkpoint
 from run_avfakenet_one_epoch import make_model
 from baselines.avfakenet.model import classification_loss
 from baselines.avfakenet.normalized_recovery import trainable_groups,set_training_mode
 identity=read(workspace_path('main_results')/'avfakenet-one-epoch-001/identity.json');profile=identity['profile'];cfg=profile['training']
 checkpoint=lock(workspace_path('main_results')/f'avfakenet-one-epoch-001/{branch}/best.pt');selected=load_checkpoint(checkpoint);m=make_model(profile,branch,selected['model_state'],'cuda');del selected
 groups=trainable_groups(m);opt=torch.optim.AdamW([{'params':[p for key,ps in groups.items() if key!='classifier' for p in ps],'lr':1e-8},{'params':groups['classifier'],'lr':1e-4}],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['epsilon'])
 batches=[]
 for w in windows:
  if branch=='visual':
   x=torch.from_numpy(w['avf_faces']).permute(0,3,1,2).contiguous().float()/255
   batches.append([x[i:i+8].cuda() for i in range(0,len(x),8)])
  else:batches.append([torch.from_numpy(w['avf_audio']).unsqueeze(0).cuda()])
 assert len(batches[0])==len(batches[1])
 if A.extract_only:
  m.eval()
  def forward(i):return [m(x).detach().cpu() for x in batches[i]]
  extraction_curve(forward,'AVFakeNet '+branch);return
 def step(i):
  for x in batches[i]:update(m,opt,classification_loss(m(x),torch.zeros(len(x),dtype=torch.long,device='cuda'),cfg['label_smoothing']))
 result=curves(m,opt,step,lambda:set_training_mode(m,'off'),'AVFakeNet '+branch,len(batches[0]))
 write(OUT/f'avf-{branch}.json',{'stage':'avf-'+branch,'curves':result,'source_files':protected,'checkpoint_unchanged':all(sha(p)==h for p,h in protected.items())})
def avt(protocol,windows,stage):
 from baselines.avtenet.model import load_component_profile,FeatureFusion,native_binary_loss
 from baselines.common.training import load_checkpoint
 plan=read(workspace_path('avhubert_results')/'auto-next-001/test-plan.json');profile=load_component_profile(C/plan['config'])
 checkpoint=lock(relocate(C/plan['checkpoints'][stage],__file__));selected=load_checkpoint(checkpoint)
 if stage=='fusion':
  m=FeatureFusion(profile).cuda();inputs=[]
  for i in range(2):
   values={}
   for name in ('visual','audio','audiovisual'):
    p=OUT/f'avt-{name}-features.npz';assert p.is_file();values[name]=torch.from_numpy(np.load(p,allow_pickle=False)[f'window_{i}'].copy()).cuda()
   inputs.append(values)
 else:
  from baselines.avtenet.pipeline import Prepared,stage_model
  from baselines.avtenet.assets import registry
  assets=registry(workspace_path('avhubert_results')/('worker-c/assets.local.json' if os.name=='nt' else 'worker-c/assets.docker.local.json'))
  m,backend,proof=stage_model(stage,profile,assets);m.cuda();prep=Prepared(None,stage,backend,assets,None,fast_vivit_processor=stage=='visual')
  inputs=[]
  for w in windows:
   raw={'faces':w['visual_faces']} if stage=='visual' else {'waveform':w['waveform']} if stage=='audio' else {'waveform':w['waveform'],'lips':w['av_lips']}
   inputs.append(move(prep.transform(raw)))
 m.load_state_dict(selected['model_state']);del selected;m.requires_grad_(True)
 if A.extract_only:
  assert stage!='fusion';m.eval()
  def forward(i):return m.extract_features(inputs[i]).detach().cpu()
  extraction_curve(forward,'AVTENet '+stage);return
 if stage!='fusion':
  m.eval();vectors={}
  with torch.no_grad():
   for i,x in enumerate(inputs):vectors[f'window_{i}']=m.extract_features(x).detach().cpu().numpy()
  np.savez(OUT/f'avt-{stage}-features.npz',**vectors)
 opt=torch.optim.Adam(m.parameters(),lr=.0001 if stage=='fusion' else .00001)
 def step(i):update(m,opt,native_binary_loss(m(inputs[i]),torch.zeros(1,dtype=torch.long,device='cuda')))
 result=curves(m,opt,step,lambda:m.train(),'AVTENet '+stage,1)
 write(OUT/f'avt-{stage}.json',{'stage':'avt-'+stage,'curves':result,'source_files':protected,'checkpoint_unchanged':all(sha(p)==h for p,h in protected.items())})
if __name__=='__main__':
 progress('START',stage_name=A.stage)
 if A.stage=='prepare':prepare()
 else:
  protocol,windows=load_windows()
  if A.stage=='ours':ours(protocol,windows)
  elif A.stage.startswith('avf-'):avf(protocol,windows,A.stage[4:])
  else:avt(protocol,windows,A.stage[4:])
  assert all(sha(p)==h for p,h in protected.items()),'Original checkpoint changed'
 progress('COMPLETE',stage_name=A.stage)
