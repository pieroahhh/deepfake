"""Bounded CPU benchmark; does not run any GPU models or fit test data."""

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

import os,sys,time,json,tempfile
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(2)
from baselines.common.io import read,write
from baselines.avtenet.input_cache import InputCache
from baselines.avtenet.heldout_preparation import FastRawCache,FastPrepared
from baselines.avtenet.pipeline import Prepared
from baselines.common.population import Population
from transformers import VivitImageProcessor

report={}
processor=VivitImageProcessor.from_pretrained(_fk_paths.relocate_value((_fk_paths.relocate((ROOT/'checkpoints/vivit-joint-adaptation-002'), __file__)), __file__),local_files_only=True)
rng=np.random.RandomState(7);frames=rng.randint(0,256,size=(32,224,224,3),dtype=np.uint8)
started=time.perf_counter();reference=processor(list(frames),return_tensors='pt')['pixel_values'];ordinary=time.perf_counter()-started
started=time.perf_counter();fast=torch.from_numpy(np.ascontiguousarray(processor(list(frames),return_tensors='np')['pixel_values']));optimized=time.perf_counter()-started
assert torch.equal(reference,fast)
report['vivit_tensor_packing']={'reference_seconds':ordinary,'optimized_seconds':optimized,'bit_exact':True,'fixture':'32 synthetic RGB frames; local production image processor'}
with tempfile.TemporaryDirectory() as folder:
    timings={}
    for cls in (InputCache,FastRawCache):
        cache=cls(_fk_paths.relocate((Path(folder)/cls.__name__), __file__),{})
        t=time.perf_counter();value=cache.get({'source_sha256':'a'*64},0.,lambda:{'faces':frames});saved=time.perf_counter()-t
        t=time.perf_counter();loaded=cache.load({'source_sha256':'a'*64},0.);loaded_seconds=time.perf_counter()-t
        np.testing.assert_array_equal(loaded['faces'],frames)
        timings[cls.__name__]={'write_seconds':saved,'read_seconds':loaded_seconds,'bytes':next(cache.root.glob('*.npz')).stat().st_size}
    report['raw_cache']=timings
assets=read(_fk_paths.relocate((ROOT/'runs/worker-c/assets.local.json'), __file__))
pop=Population(_fk_paths.relocate((ROOT/'runs/auto-next-001/index'), __file__),_fk_paths.relocate((ROOT.parent/'dataset'), __file__),'avtenet')
selected=list(pop.rows('test'))[:3]
class Subset:
    root=pop.root
    def rows(self,split):return iter(selected)
    def path(self,row):return pop.path(row)
with tempfile.TemporaryDirectory() as folder:
    outputs=[];times=[]
    for cls in (Prepared,FastPrepared):
        p=cls(Subset(),'audio',None,assets,None,_fk_paths.relocate((Path(folder)/(cls.__name__+'.events.jsonl')), __file__),
            FastRawCache(_fk_paths.relocate((Path(folder)/cls.__name__), __file__),{}),prefetch_cached_inputs=4)
        before=time.perf_counter();rows=list(p.records('test'));times.append(time.perf_counter()-before);outputs.append(rows)
    assert len(outputs[0])==len(outputs[1])
    for a,b in zip(*outputs):
        assert a[:3]==b[:3] and a[4]==b[4] and torch.equal(a[3],b[3])
    report['real_ast_cpu_pipeline']={'clips':3,'windows':len(outputs[0]),'reference_seconds':times[0],'optimized_seconds':times[1],'bit_exact':True,
        'note':'CPU only; GPU overlap benefit is not measured here'}
write(_fk_paths.relocate((ROOT/'runs/auto-next-001/preprocessing-benchmark.json'), __file__),report,replace=True)
print(json.dumps(report,indent=2))
