"""Serialize this repository's explicit CUDA CLI jobs; no global machine settings."""

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

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time


@contextmanager
def gpu_slot(enabled=True):
    if not enabled or os.environ.get('FK_GPU_QUEUE_MANAGED')=='1':
        yield;return
    root=_fk_paths.relocate((Path(os.environ.get('FK_GPU_QUEUE_DIRECTORY',str(Path(__file__).resolve().parents[2]/'.cache')))), __file__)
    root.mkdir(parents=True,exist_ok=True)
    with (_fk_paths.relocate((root/'gpu-work.lock'), __file__)).open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        waiting=False
        while True:
            stream.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:
                if not waiting:print('GPU queue: waiting for another experiment stage to finish...',file=sys.stderr,flush=True)
                waiting=True;time.sleep(5)
        print('GPU queue: acquired',file=sys.stderr,flush=True)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def explicit_cuda(argv):
    return any(value=='--device' and index+1<len(argv) and argv[index+1].startswith('cuda')
               for index,value in enumerate(argv))
