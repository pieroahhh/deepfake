"""Machine A queue: resume FK only; machines B/C own the comparator models."""

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

from pathlib import Path
import argparse
import subprocess
import sys

ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-root',default='D:/AV-Deepfake1M')
    args=p.parse_args()
    commands=[
        ['scripts/run_two_day.py','--methods','fk','--manifest','runs/two-day-001/population/manifest.json',
         '--dataset-root',args.dataset_root,'--assets','checkpoints/paper-alignment-001/assets.local.json',
         '--output','runs/two-day-001/experiment']]
    for command in commands:
        subprocess.run([sys.executable,'-u',*command],cwd=ROOT,check=True)
    print('FK complete; import machine B/C results on machine A for comparison.',flush=True)
