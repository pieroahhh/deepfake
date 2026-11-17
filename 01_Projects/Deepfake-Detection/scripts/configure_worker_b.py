"""Generate local and Docker asset paths after relocating the second-machine bundle."""

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
import sys
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest

if __name__=='__main__':
    value=read(_fk_paths.relocate((ROOT/'baselines/avtenet/configs/assets.example.json'), __file__))
    checkpoint=_fk_paths.relocate((ROOT/'checkpoints/comparators-native-001/avhubert_native/base_lrs3_iter5.pt'), __file__)
    value['experts']['audiovisual'].update(repository='/opt/av_hubert',commit='258fb50e155134eec2c4b49c2ae8de267075fd18',
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
    directory=_fk_paths.relocate((ROOT/'runs/worker-b'), __file__)
    write(_fk_paths.relocate((directory/'assets.local.json'), __file__),value,replace=True)
    value['experts']['audiovisual']['checkpoint']='/workspace/'+checkpoint.relative_to(ROOT).as_posix()
    write(_fk_paths.relocate((directory/'assets.docker.local.json'), __file__),value,replace=True)
