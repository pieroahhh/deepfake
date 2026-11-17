"""Read-only CPU timing of existing AV-HuBERT caches; no model/GPU allocation."""

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

import sys,time,json
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
import torch
from baselines.common.io import read,fingerprint
from baselines.common.population import Population,windows
from baselines.avtenet.input_cache import InputCache
from deepfake_detection.avhubert_preparation import prepare_avhubert
from threadpoolctl import threadpool_info,threadpool_limits
run=_fk_paths.relocate((ROOT/'runs/worker-c/avtenet'), __file__)
identity=read(_fk_paths.relocate((run/'audiovisual-features/producer.json'), __file__))['identity']
identity={k:v for k,v in identity.items() if k not in ('schedule','max_steps')}
assert (_fk_paths.relocate((run/'prepared-inputs'/fingerprint(identity)/'identity.json'), __file__)).exists()
cache=InputCache(_fk_paths.relocate((run/'prepared-inputs'), __file__),identity)
population=Population(_fk_paths.relocate((ROOT/'runs/two-day-001/avtenet/shared-index'), __file__),'/dataset','avtenet')
settings=read(_fk_paths.relocate((ROOT/'runs/worker-c/assets.docker.local.json'), __file__))['experts']['audiovisual']['preparation']
examples=[]
for row in population.rows('train'):
    for start,end in windows(row['duration']):
        raw=cache.load(row,start)
        if raw is not None:examples.append((row,start,raw))
        if len(examples)==12:break
    if len(examples)==12:break
torch.set_num_threads(8)
print('POOLS',json.dumps(threadpool_info()),flush=True)
reference=[prepare_avhubert(raw['waveform'],raw['lips'],settings) for _,_,raw in examples]
for threads in (None,1,2,4):
    with threadpool_limits(limits=threads,user_api='blas'):
        times={'load':0.,'transform':0.}
        for number,(row,start,raw) in enumerate(examples):
            before=time.perf_counter();cache.load(row,start);times['load']+=time.perf_counter()-before
            before=time.perf_counter();actual=prepare_avhubert(raw['waveform'],raw['lips'],settings);times['transform']+=time.perf_counter()-before
            for key in actual:
                if not torch.equal(actual[key],reference[number][key]):raise ValueError('Input parity failed: '+key)
        print(json.dumps({'blas_threads':threads,'windows':len(examples),**times}),flush=True)
print('ALL INPUTS BIT-EXACT',flush=True)
