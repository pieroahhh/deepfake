"""Content-verified frozen-feature cache with explicit window and encoder provenance."""

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

from safetensors.torch import load_file, save_file
import torch

from .cli import load_json, save
from .forensics import sha256_file
from .avtenet import fingerprint


def write_cache(directory, tensors, provenance):
    required = ('source_sha256', 'preprocessing_sha256', 'encoder_sha256',
                'feature_layers', 'window_intervals', 'precision')
    if any(name not in provenance for name in required):
        raise ValueError('cache provenance is incomplete')
    for name, tensor in tensors.items():
        if tensor.requires_grad or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'{name} must be finite detached encoder features')
    root = _fk_paths.relocate((Path(directory)), __file__)
    root.mkdir(parents=True, exist_ok=False)
    payload = _fk_paths.relocate((root / 'features.safetensors'), __file__)
    # Clone expanded/shared/inference-mode views into independent stored tensors.
    save_file({name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()}, str(payload))
    metadata = {'schema': 'frozen-feature-cache-v1', 'provenance': provenance,
                'provenance_sha256': fingerprint(provenance), 'content_sha256': sha256_file(payload),
                'tensors': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in tensors.items()}}
    save(_fk_paths.relocate((root / 'metadata.json'), __file__), metadata)
    return metadata


def read_cache(directory, expected_provenance):
    root = _fk_paths.relocate((Path(directory)), __file__)
    metadata = load_json(_fk_paths.relocate((root / 'metadata.json'), __file__))
    if (metadata['schema'] != 'frozen-feature-cache-v1'
            or metadata['provenance_sha256'] != fingerprint(expected_provenance)
            or fingerprint(metadata['provenance']) != fingerprint(expected_provenance)):
        raise ValueError('cache provenance mismatch')
    payload = _fk_paths.relocate((root / 'features.safetensors'), __file__)
    if sha256_file(payload) != metadata['content_sha256']:
        raise ValueError('feature cache content hash mismatch')
    tensors = load_file(_fk_paths.relocate_value((str(payload)), __file__))
    actual = {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in tensors.items()}
    if actual != metadata['tensors']:
        raise ValueError('cache tensor schema mismatch')
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError('cache contains nonfinite values')
    return tensors
