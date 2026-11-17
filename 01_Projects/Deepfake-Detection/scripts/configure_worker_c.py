"""Rebuild host/container registries for all three AVTENet experts after relocation."""

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

import copy
from pathlib import Path
import sys

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(_fk_paths.relocate((ROOT / 'src'), __file__))]
from baselines.common.io import read, write, digest, checked_path


def configure(root=ROOT):
    root = _fk_paths.relocate((Path(root)), __file__).resolve()
    value = read(_fk_paths.relocate((root / 'baselines/avtenet/configs/assets.example.json'), __file__))
    for stage, relative in (('visual', 'checkpoints/vivit-joint-adaptation-002'),
                            ('audio', 'checkpoints/comparators-ast-001/ast')):
        directory = _fk_paths.relocate((root / relative), __file__)
        metadata = read(_fk_paths.relocate((directory / 'artifact.json'), __file__))
        for name, expected in metadata['files'].items():
            if digest(checked_path(directory, name)) != expected:
                raise ValueError(stage + ' asset checksum mismatch: ' + name)
        value['experts'][stage] = {'directory': str(directory), 'manifest': str(_fk_paths.relocate((directory / 'artifact.json'), __file__))}
    checkpoint = _fk_paths.relocate((root / 'checkpoints/comparators-native-001/avhubert_native/base_lrs3_iter5.pt'), __file__)
    metadata = read(_fk_paths.relocate((checkpoint.parent / 'artifact.json'), __file__))
    actual = digest(checkpoint)
    if actual != metadata['files'][checkpoint.name]:
        raise ValueError('AV-HuBERT asset checksum mismatch')
    value['experts']['audiovisual'].update(repository='/opt/av_hubert',
        commit='258fb50e155134eec2c4b49c2ae8de267075fd18', checkpoint=str(checkpoint), checkpoint_sha256=actual)
    docker = copy.deepcopy(value)
    for entry in docker['experts'].values():
        for key in ('directory', 'manifest', 'checkpoint'):
            if entry.get(key):
                entry[key] = '/workspace/' + _fk_paths.relocate((Path(entry[key])), __file__).relative_to(root).as_posix()
    destination = _fk_paths.relocate((root / 'runs/worker-c'), __file__)
    write(_fk_paths.relocate((destination / 'assets.local.json'), __file__), value, replace=True)
    write(_fk_paths.relocate((destination / 'assets.docker.local.json'), __file__), docker, replace=True)
    print('Verified and configured all AVTENet expert assets:', destination)
    return value, docker


if __name__ == '__main__':
    configure()
