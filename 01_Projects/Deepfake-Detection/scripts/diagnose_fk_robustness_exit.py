"""Isolated reproduction of an unexpected native FK robustness worker exit."""

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

import sys,os,json,time,faulthandler,traceback
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
OUT=_fk_paths.relocate((ROOT/'runs/fk-robustness-exit-diagnosis-001'), __file__);OUT.mkdir(exist_ok=True)
native=(_fk_paths.relocate((OUT/'native-trace.log'), __file__)).open('a',encoding='utf-8',buffering=1)
faulthandler.enable(native,all_threads=True)
from run_remaining_supplement import FrozenFK
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json

def status(stage,**extra):
    write_json(_fk_paths.relocate((OUT/'progress.json'), __file__),{'stage':stage,'updated_unix':time.time(),**extra});print(stage,extra,flush=True)

def main():
    import torch
    torch.set_num_threads(2)
    rows=read(_fk_paths.relocate((ROOT/'runs/remaining-derivatives-001/manifest.json'), __file__));row=rows[113]
    path=_fk_paths.relocate((ROOT/'runs/remaining-derivatives-001'/row['path']), __file__)
    status('source_integrity',row=row)
    if digest(path)!=row['source_sha256']:raise ValueError('Source hash mismatch')
    status('loading_fk');model=FrozenFK()
    import deepfake_detection.preparation as prep
    original=prep.prepare_fk_window
    def tracked(*args,**kwargs):
        status('prepare_window',start=args[1]);value=original(*args,**kwargs);status('prepared_window',start=args[1],assessment=value['status']);return value
    prep.prepare_fk_window=tracked
    status('predict');result=model.predict(path,row)
    write_json(_fk_paths.relocate((OUT/'result.json'), __file__),result);status('complete')

if __name__=='__main__':
    try:main()
    except Exception as error:status('python_error',error=str(error));traceback.print_exc();raise
