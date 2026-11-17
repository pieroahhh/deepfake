"""Record the Git revision of the source checkout that produced an artifact."""

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
import subprocess


def git_provenance():
    root = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[2]
    if not (_fk_paths.relocate((root / '.git'), __file__)).exists():
        return {'git_commit': None, 'git_source_dirty': None}
    git = ['git', '-c', 'safe.directory='+root.as_posix()]
    commit = subprocess.check_output([*git, 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    status = subprocess.check_output(
        [*git, 'status', '--porcelain', '--untracked-files=all', '--', 'src', 'baselines', 'pyproject.toml'],
        cwd=root, text=True,
    )
    return {'git_commit': commit, 'git_source_dirty': bool(status)}
