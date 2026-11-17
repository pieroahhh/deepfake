"""Verified FK cache load and actual majority/OR scaling; no fitting."""

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

import hashlib,json,os,statistics,sys,time
from pathlib import Path
W=_fk_paths.workspace_root(__file__);R=_fk_paths.relocate((W/'Deepfake-Detection'), __file__);sys.path[:0]=[str(R),str(_fk_paths.relocate((R/'src'), __file__))]
from deepfake_detection.feature_cache import read_cache
from deepfake_detection.baselines import avfakenet_decision
from deepfake_detection.forensics import sha256_file
P=_fk_paths.relocate((R/'runs/two-day-001/experiment/fk/cache/training-manifest.json'), __file__);d=json.loads(P.read_text());rows=[]
for label in (0,1):
 r=next(x for x in d['train'] if x['label']==label and x['window_intervals']);root=_fk_paths.relocate((P.parent/r['cache']), __file__);samples=[]
 for i in range(7):
  start=time.perf_counter();values=read_cache(root,r['cache_provenance']);elapsed=time.perf_counter()-start
  if i:samples.append(elapsed)
 rows.append({'video_id':r['video_id'],'split':'train','usable_windows':len(r['window_intervals']),'feature_bytes':(_fk_paths.relocate((root/'features.safetensors'), __file__)).stat().st_size,'payload_sha256':sha256_file(_fk_paths.relocate((root/'features.safetensors'), __file__)),'seconds':samples,'median_seconds':statistics.median(samples),'scope':'content hash, metadata/schema/finite validation, CPU safetensors load; uncontrolled warm OS cache'})
scale=[]
# Decisions are from our retained AVFakeNet visual predictions, never reference papers.
p=_fk_paths.relocate((R/'runs/avfakenet-one-epoch-001/validation-predictions.jsonl'), __file__);pred=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
decisions=[int(x['visual_prediction']) for x in pred if x.get('visual_prediction') is not None]
assert decisions
for n in (1,8,32,128,512,2048):
 x=(decisions*((n+len(decisions)-1)//len(decisions)))[:n];samples=[]
 for repeat in range(22):
  start=time.perf_counter()
  for i in range(100):result=avfakenet_decision(x,0,tie_break='fake')
  elapsed=(time.perf_counter()-start)/100
  if repeat>=2:samples.append(elapsed)
 assert result['score'] is None
 scale.append({'frames':n,'seconds':samples,'median_seconds':statistics.median(samples),'scope':'actual majority + OR function; retained validation decisions repeated to vary workload; no neural accuracy claim'})
out={'cache_loads':rows,'majority_or_scaling':scale,'decision_source':str(p),'decision_source_sha256':sha256_file(p),'training_manifest_sha256':sha256_file(P)}
target=_fk_paths.relocate((R/'runs/table-iii-complete-001/cache-cost.json'), __file__);target.write_text(json.dumps(out,indent=2),encoding='utf-8');print('COMPLETE cache and majority costs')
