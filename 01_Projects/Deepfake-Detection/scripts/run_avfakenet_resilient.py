"""Versioned I/O-only wrapper; original hashed preparation/training code unchanged."""

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

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from avfakenet_progress_io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare','pilot'), required=True)
    args, remaining=parser.parse_known_args()
    option='--run' if args.mode=='prepare' else '--output'
    run=_fk_paths.relocate((Path(remaining[remaining.index(option)+1])), __file__).resolve()
    sources=[_fk_paths.relocate((Path(__file__)), __file__),_fk_paths.relocate((ROOT/'scripts/avfakenet_progress_io.py'), __file__)]
    provenance={'scope':'I/O-only runtime wrapper; no changes to data, models, sampler or optimizer',
                'mode':args.mode,'pid':os.getpid(),'timestamp_unix':time.time(),
                'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'progress_policy':'unique atomic temp; retry sharing failures for 2 seconds; skip only locked progress updates',
                'other_json_policy':'retry then raise; never suppress metadata failures'}
    write_json(_fk_paths.relocate((run/('io-wrapper-'+str(time.time_ns())+'.json')), __file__),provenance)
    sys.argv=[sys.argv[0],*remaining]
    if args.mode=='prepare':
        import scale_avfakenet_prepare as scale
        scale.base.write=write_json
        original_run=scale.subprocess.run
        def continue_with_wrapper(command, *positional, **keywords):
            expected=str(_fk_paths.relocate((ROOT/'scripts/run_avfakenet_normalized_pilot.py'), __file__))
            if len(command)>2 and command[2]==expected:
                command=[command[0],'-u',str(_fk_paths.relocate((Path(__file__)), __file__)), '--mode','pilot',*command[3:]]
            return original_run(command,*positional,**keywords)
        scale.subprocess.run=continue_with_wrapper
        scale.main()
    else:
        import run_avfakenet_normalized_pilot as pilot
        pilot.write=write_json
        pilot.main()


if __name__=='__main__':main()
