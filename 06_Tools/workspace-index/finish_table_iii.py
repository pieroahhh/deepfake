"""Wait for native profiling, then measure the isolated Linux expert and fusion."""

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

import importlib.util
import sys
import time
from pathlib import Path

ROOT = _fk_paths.workspace_root(__file__)
REPO = _fk_paths.relocate((ROOT / 'Deepfake-Detection'), __file__)
OUT = _fk_paths.relocate((REPO / 'runs/table-iii-cost-001'), __file__)
sys.path[:0] = [str(_fk_paths.relocate((REPO / 'src'), __file__))]
from deepfake_detection.quality_fitting import exclusive_run
spec = importlib.util.spec_from_file_location('table_iii_runner', _fk_paths.relocate((Path(__file__)), __file__).with_name('run_table_iii.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
deadline = runner.read(_fk_paths.relocate((OUT / 'protocol.json'), __file__))['deadline_unix']
while True:
    if time.time() > deadline - 180:
        runner.report()
        raise RuntimeError('Not enough time left for Linux profiling')
    state = runner.read(_fk_paths.relocate((OUT / 'progress.json'), __file__))
    if state.get('stage') == 'FAILED':
        raise RuntimeError('Native profiling requires attention: ' + str(state))
    complete = all((_fk_paths.relocate((OUT / stage / 'complete.json'), __file__)).exists() for stage in ('fk', 'avfakenet', 'audio', 'visual'))
    if complete:
        try:
            with exclusive_run(_fk_paths.relocate((OUT / 'worker.lock'), __file__)):
                pass
            break
        except ValueError:
            pass
    print('Waiting for native stages: ' + str(state.get('stage')), flush=True)
    time.sleep(20)
runner.run()
