"""Refine inline costs by instrumenting a runtime copy; source files stay unchanged."""

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

import ast,copy,hashlib,inspect,json,os,statistics,sys,textwrap,time
from pathlib import Path
W=_fk_paths.workspace_root(__file__);R=_fk_paths.relocate((W/'Deepfake-Detection'), __file__);OUT=_fk_paths.relocate((R/'runs/table-iii-complete-001'), __file__)
sys.path[:0]=[str(R),str(_fk_paths.relocate((R/'src'), __file__)),str(_fk_paths.relocate((R/'scripts'), __file__))]
os.environ['HF_HOME']=str(_fk_paths.relocate((R/'.cache/huggingface'), __file__));os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((R/'.cache/matplotlib'), __file__))
os.environ['PATH']=str(_fk_paths.relocate((R/'.cache/ffmpeg-tools/static_ffmpeg/bin/win32'), __file__))+os.pathsep+os.environ.get('PATH','')
import numpy as np
import torch
from table_iii_profiler import Timer,media_timers,fk_timers,read,write,digest
from run_remaining_supplement import FrozenFK
import deepfake_detection.preparation as prep
from deepfake_detection.model import EvidenceModel
torch.set_num_threads(2)
def instrument(fn,timer,kind):
 tree=ast.parse(textwrap.dedent(inspect.getsource(fn)));root=tree.body[0]
 def wrap(stmts,label,gpu=False):
  ctx=ast.Call(func=ast.Attribute(value=ast.Name(id='_detail_timer',ctx=ast.Load()),attr='phase',ctx=ast.Load()),args=[ast.Constant(label),ast.Constant(gpu)],keywords=[])
  return [ast.With(items=[ast.withitem(context_expr=ctx,optional_vars=None)],body=stmts)]
 class Visitor(ast.NodeTransformer):
  def visit_FunctionDef(self,node):
   self.generic_visit(node)
   if kind=='model' and node.name=='attend':node.body=wrap(node.body,'evidence.attention_matrix_softmax_value',True)
   return node
  def visit_Assign(self,node):
   self.generic_visit(node)
   if kind=='prep':
    target=' '.join(ast.unparse(t) for t in node.targets)
    if 'quality_measurements' in target or target in ('fraction','speech_fraction','signal','noise','speech_mask'):
     return wrap([node],'pre.quality_calculation')
   return node
 tree=Visitor().visit(tree);ast.fix_missing_locations(tree);env={**fn.__globals__,'_detail_timer':timer};exec(compile(tree,fn.__code__.co_filename,'exec'),env);return env[fn.__name__]
def fingerprint(value):
 h=hashlib.sha256()
 def walk(v):
  if isinstance(v,np.ndarray):h.update(str((v.shape,str(v.dtype))).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v):h.update(k.encode());walk(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v:walk(x)
  else:h.update(repr(v).encode())
 walk(value);return h.hexdigest()
def main():
 sources={str(p):digest(p) for p in [_fk_paths.relocate((R/'src/deepfake_detection/model.py'), __file__),_fk_paths.relocate((R/'src/deepfake_detection/preparation.py'), __file__),_fk_paths.relocate((R/'scripts/run_remaining_supplement.py'), __file__)]}
 fk=FrozenFK();timer=Timer(torch);population=read(_fk_paths.relocate((R/'runs/two-day-001/population/manifest.json'), __file__));chosen=[]
 for label in (0,1):chosen.append(next(r for r in population if r['split']=='train' and r['label']==label))
 original=prep.prepare_fk_window;prepared=[]
 for row in chosen:
  path=_fk_paths.relocate((Path('D:/AV-Deepfake1M')/row['path']), __file__);assert digest(path)==row['source_sha256'];speech=prep.fk_source_speech(path,row['duration'],fk.vad,pcm16=True)
  for start in (0.,1.,2.):
   value=original(path,start,row['duration'],fk.detector,speech,fit_to_frame=fk.fit,pcm16=True)
   if value['status']=='USABLE':
    with torch.no_grad():
     features=prep.encode_fk_window(value,fk.encoders)
     baseline=fk.model(**{k:v.cuda() for k,v in features.items()},quality=fk.mapper(value['quality_measurements']).cuda())
    prepared.append((row,path,speech,start,value,baseline));break
 assert len(prepared)==2
 prep.prepare_fk_window=instrument(original,timer,'prep');EvidenceModel.forward=instrument(EvidenceModel.forward,timer,'model')
 media_timers(timer);fk_timers(timer,fk);rows=[]
 for row,path,speech,start,expected,baseline in prepared:
  samples=[]
  for repeat in range(4):
   timer.reset();torch.cuda.synchronize();t=time.perf_counter()
   actual=prep.prepare_fk_window(path,start,row['duration'],fk.detector,speech,fit_to_frame=fk.fit,pcm16=True)
   assert fingerprint(actual)==fingerprint(expected)
   with torch.no_grad():
    features=prep.encode_fk_window(actual,fk.encoders);result=fk.model(**{k:v.cuda() for k,v in features.items()},quality=fk.mapper(actual['quality_measurements']).cuda())
   torch.cuda.synchronize();elapsed=time.perf_counter()-t
   error=max(float((v-baseline[k]).abs().max()) for k,v in result.items() if isinstance(v,torch.Tensor))
   assert error<=5e-6,error
   samples.append({'seconds':elapsed,'timers':copy.deepcopy(timer.values),'max_output_error':error})
  rows.append({'video_id':row['video_id'],'split':'train','start':start,'source_sha256':row['source_sha256'],'prepared_parity':True,'samples':samples})
 # CPU hashing and canonical serialization use bytes/records from our experiments.
 from deepfake_detection.forensics import canonical
 payload=_fk_paths.relocate((Path(prepared[0][1])), __file__).read_bytes();scaling=[]
 for size in sorted(set([n for n in (4096,16384,65536,262144,1048576,len(payload)) if n<=len(payload)])):
  data=payload[:size];samples=[]
  for i in range(22):
   t=time.perf_counter();h=hashlib.sha256(data).hexdigest();elapsed=time.perf_counter()-t
   if i>=2:samples.append(elapsed)
  assert h==hashlib.sha256(data).hexdigest();scaling.append({'operation':'sha256_in_memory_source_prefix','bytes':size,'seconds':samples,'median_seconds':statistics.median(samples)})
 for p in sorted((_fk_paths.relocate((R/'runs/table-iii-cost-001/fk'), __file__)).glob('artifacts-*-1/record.json'))[:4]:
  record=read(p);samples=[]
  for i in range(22):
   t=time.perf_counter();data=canonical(record);elapsed=time.perf_counter()-t
   if i>=2:samples.append(elapsed)
  scaling.append({'operation':'canonical_actual_record','path':str(p),'bytes':len(data),'seconds':samples,'median_seconds':statistics.median(samples)})
 assert all(digest(p)==v for p,v in sources.items())
 write(_fk_paths.relocate((OUT/'detail-cost.json'), __file__),{'scope':'two original TRAIN windows, one warmup and three measured passes; nested synchronized exclusive self times; no source edits','source_hashes':sources,'source_hashes_unchanged':True,'windows':rows,'cpu_scaling':scaling})
 print('COMPLETE detail',flush=True)
if __name__=='__main__':main()
