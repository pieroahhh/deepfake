"""Reproducible synthetic reconstruction check; never evaluates real media."""

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
from pathlib import Path

import torch

from .avfakenet import DSTNet, classification_loss, load_profile, profile_hash
from .cli import save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True, help='new output directory')
    args = parser.parse_args(argv)
    profile = load_profile(args.profile)
    output = _fk_paths.relocate((Path(args.output)), __file__)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    report = {
        'status': 'SYNTHETIC_SOFTWARE_CHECK_NOT_DETECTION_RESULTS',
        'profile_id': profile['profile_id'], 'profile_sha256': profile_hash(profile),
        'seed': 7, 'device': 'cpu', 'torch_version': str(torch.__version__), 'branches': {},
    }
    for modality in ('visual', 'audio'):
        model = DSTNet(profile, modality)
        pixels = profile['preprocessing'][f'{modality}_pixels']
        logits, shapes = model(torch.rand(1, 3, pixels, pixels), return_shapes=True)
        loss = classification_loss(logits, torch.tensor([1]), profile['training']['label_smoothing'])
        loss.backward()
        finite_gradients = all(
            p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()
        )
        if not finite_gradients:
            raise RuntimeError(f'{modality} branch has missing or nonfinite gradients')
        report['branches'][modality] = {
            'shapes': shapes, 'parameters': sum(p.numel() for p in model.parameters()),
            'finite_gradients': finite_gradients,
        }
    save(_fk_paths.relocate((output / 'profile.json'), __file__), profile)
    save(_fk_paths.relocate((output / 'smoke.json'), __file__), report)
    print(f'Synthetic forward/backward checks passed. Report: {_fk_paths.relocate((output / "smoke.json"), __file__)}')


if __name__ == '__main__':
    main()
