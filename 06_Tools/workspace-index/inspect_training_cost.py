
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

import json
from pathlib import Path
W=_fk_paths.workspace_root(__file__)
R=_fk_paths.relocate((W/'Deepfake-Detection'), __file__)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
d=read(_fk_paths.relocate((R/'runs/two-day-001/experiment/fk/cache/training-manifest.json'), __file__))
print('FK_KEYS',list(d))
for k,v in d.items():
 if k!='assessments': print('FK',k,str(v)[:650])
print('CACHE_DIR', [str(p.relative_to(R)) for p in (_fk_paths.relocate((R/'runs/two-day-001/experiment/fk/cache/caches/000000'), __file__)).glob('*')])
print('PREP_INDEX',read(_fk_paths.relocate((R/'runs/avfakenet-one-epoch-001/prepared-index.json'), __file__))[0])
print('AVF_CACHE', read(_fk_paths.relocate((R/'runs/avfakenet-one-epoch-001/identity.json'), __file__))['prepared_cache_provenance'].get('source'))
row=d['train'][0];print('FK_ROW', {k:v for k,v in row.items() if k not in ('cache_provenance','preprocessing')})
base=_fk_paths.relocate((W/'avh-a/code/runs/worker-c/avtenet'), __file__)
for name in ['prepared-inputs','fusion-features','audio-features']:
 p=_fk_paths.relocate((base/name), __file__)
 print('AVT_DIR',name,[x.name for x in p.iterdir()][:12])
 for f in ['cache.json','identity.json','index.json']:
  if (_fk_paths.relocate((p/f), __file__)).exists():
   v=read(_fk_paths.relocate((p/f), __file__));print('AVT',name,f,str(v)[:1200])
