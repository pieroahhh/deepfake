"""Alias pinned upstream sibling modules once, avoiding duplicate registration."""

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

import importlib,sys
from pathlib import Path


def bootstrap(repository):
    # Upstream chooses flat debug imports only when sys.argv contains no CLI
    # arguments. Production CLI invocations already use proper relative imports.
    if len(sys.argv)>1:
        return
    directory=_fk_paths.relocate((Path(repository)/'avhubert'), __file__)
    sys.path.insert(0,str(directory))
    names=('hubert_pretraining','hubert','hubert_asr','hubert_dataset','hubert_criterion')
    for name in names:
        importlib.import_module(name)
        for sibling in names:
            if sibling in sys.modules:
                sys.modules['avhubert.'+sibling]=sys.modules[sibling]
