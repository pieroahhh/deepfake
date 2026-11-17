"""Continue unchanged preparation, then dispatch the new one-epoch pilot."""

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

import os,sys,time,subprocess
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from avfakenet_progress_io import write_json


def main():
    import scale_avfakenet_prepare as scale
    run=_fk_paths.relocate((ROOT/'runs/avfakenet-normalized-pilot-001'), __file__)
    target=_fk_paths.relocate((ROOT/'runs/avfakenet-one-epoch-001'), __file__)
    write_json(_fk_paths.relocate((run/'next-run.json'), __file__),{'output':str(target),'epochs_per_branch':1,
        'reason':'User authorized lower epoch budget on same population; two-hour deadline cannot be guaranteed',
        'created_unix':time.time(),'automatic_heldout':False})
    scale.base.write=write_json
    original_run=subprocess.run
    def dispatch(command,*positional,**keywords):
        if len(command)>2 and command[2]==str(_fk_paths.relocate((ROOT/'scripts/run_avfakenet_normalized_pilot.py'), __file__)):
            command=list(command);command[2]=str(_fk_paths.relocate((ROOT/'scripts/run_avfakenet_one_epoch.py'), __file__))
            command[command.index('--output')+1]=str(target)
            command+=['--prepared-cache',str(_fk_paths.relocate((run/'prepared'), __file__))]
        return original_run(command,*positional,**keywords)
    scale.subprocess.run=dispatch
    sys.argv=[str(_fk_paths.relocate((Path(__file__)), __file__)),'--run',str(run),'--dataset-root',str(_fk_paths.relocate((ROOT.parent/'avh-a/dataset'), __file__)),
              '--evidence',str(_fk_paths.relocate((ROOT/'runs/avfakenet-worker-scaling-001'), __file__))]
    scale.main()


if __name__=='__main__':main()
