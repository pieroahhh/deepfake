"""Atomic JSON with bounded Windows sharing retries; progress is noncritical."""

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
import os
import sys
import tempfile
import time
from pathlib import Path


def write_json(path, value, *, timeout=2.0):
    path = _fk_paths.relocate((Path(path)), __file__)
    payload = json.dumps(value, indent=2, allow_nan=False)
    # Unique temp names prevent retrying writers from sharing the same scratch file.
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = _fk_paths.relocate((Path(name)), __file__)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
        deadline = time.monotonic()+timeout
        while True:
            try:
                os.replace(temporary, path)
                return True
            except PermissionError:
                if time.monotonic() >= deadline:
                    if path.name != 'progress.json':
                        raise
                    # A stale status display must never cancel data preparation.
                    print('Progress update skipped: Windows still holds progress.json; work continues.', file=sys.stderr, flush=True)
                    return False
                time.sleep(.05)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass
