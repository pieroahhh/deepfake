"""Preserve an unsaved first epoch, then start existing hidden experiment jobs."""

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
import os,time,subprocess,traceback
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
log=_fk_paths.relocate((ROOT/'runs/remaining-reboot-recovery-python.log'), __file__)
def report(message):
    with log.open('a',encoding='utf-8') as stream:stream.write(message+'\n')
try:
    report('Recovery started '+str(time.time()))
    target=_fk_paths.relocate((ROOT/'runs/remaining-ablations-001/no_av'), __file__)
    if target.exists() and not (_fk_paths.relocate((target/'completed.json'), __file__)).exists():
        if (_fk_paths.relocate((target/'attempt-001/last.pt'), __file__)).exists():raise RuntimeError('Saved checkpoint requires explicit resume')
        dest=target.with_name('no_av-interrupted-'+str(time.time_ns()))
        os.rename(target,dest);report('Preserved interrupted run '+str(dest))
    for name in ('FK-Remaining-Core-Ablations-001','FK-Remaining-Ablation-Evaluation-001','FK-Remaining-Derivatives-001','FK-Remaining-Supplement-001'):
        value=subprocess.run(['schtasks.exe','/Run','/TN',name],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW,timeout=30)
        report(name+': '+value.stdout+value.stderr)
        value.check_returncode()
except Exception:
    report(traceback.format_exc());raise
