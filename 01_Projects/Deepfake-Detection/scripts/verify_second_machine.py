"""Verify exported code/media without importing training dependencies."""

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
from pathlib import Path


def verify(root,role=None):
    root=_fk_paths.relocate((Path(root)), __file__).resolve();metadata=json.loads((_fk_paths.relocate((root/'bundle.json'), __file__)).read_text(encoding='utf-8'))
    if role and metadata.get('worker_role')!=role:
        raise ValueError('bundle role mismatch; use the separately exported '+role+' bundle')
    for number,(relative,expected) in enumerate(metadata['files'].items(),1):
        path=(_fk_paths.relocate((root/relative), __file__)).resolve();path.relative_to(root)
        value=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
        if value.hexdigest()!=expected:raise ValueError('bundle checksum mismatch: '+relative)
        if number%500==0:print('Verified',number,'files',flush=True)
    print('Bundle verified:',metadata['clips'],'clips')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True)
    p.add_argument('--role',choices=('B','C'))
    args=p.parse_args();verify(args.bundle,args.role)
