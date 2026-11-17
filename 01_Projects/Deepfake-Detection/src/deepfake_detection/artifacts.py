"""Local artifact manifests. Missing weights never trigger a random fallback."""

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

from .cli import load_json
from .forensics import sha256_file


def verified_files(directory, manifest, *, require_pretrained=True):
    """Verify files against independently supplied source/revision/checksum metadata."""
    root = _fk_paths.relocate((Path(directory)), __file__).resolve()
    metadata = load_json(manifest) if isinstance(manifest, (str, Path)) else manifest
    if metadata.get('schema') != 'model-artifact-v1':
        raise ValueError('unsupported model artifact manifest')
    for key in ('source', 'revision', 'architecture', 'weights_status'):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f'artifact manifest requires {key}')
    if require_pretrained and metadata['weights_status'] != 'pretrained':
        raise ValueError('pretrained weights required; synthetic/random artifact rejected')
    if not metadata.get('files'):
        raise ValueError('artifact manifest contains no files')
    for relative, expected in metadata['files'].items():
        path = (_fk_paths.relocate((root / relative), __file__)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'missing or escaping artifact path: {relative}')
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected)):
            raise ValueError('invalid artifact SHA-256')
        if sha256_file(path) != expected:
            raise ValueError(f'artifact checksum mismatch: {relative}')
    return metadata


def require_recorded(metadata, *files):
    if any(name not in metadata['files'] for name in files):
        raise ValueError('every consumed model/configuration file must be checksummed')
