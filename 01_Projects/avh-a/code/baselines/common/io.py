"""Small portable IO helpers, including the isolated Python 3.8 worker."""

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

import hashlib
import json
import os
from pathlib import Path


def read(path):
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(_fk_paths.relocate((Path(path)), __file__).read_text(encoding='utf-8'), parse_constant=reject)


def digest(path):
    value = hashlib.sha256()
    with _fk_paths.relocate((Path(path)), __file__).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value, *, replace=False):
    path = _fk_paths.relocate((Path(path)), __file__)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if not replace:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(data)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(data, encoding='utf-8')
        os.replace(temporary, path)


def checked_path(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative or any(p in ('', '.', '..') for p in relative.split('/')):
        raise ValueError('expected safe relative POSIX path')
    path = (_fk_paths.relocate((Path(root).resolve() / relative), __file__)).resolve()
    path.relative_to(_fk_paths.relocate((Path(root)), __file__).resolve())
    return path
