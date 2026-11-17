"""Run checksummed official AV-HuBERT inference inside its legacy environment.

This entry point is Python 3.8 compatible and consumes explicitly prepared paired
inputs. It is a frozen feature-extraction bridge, not a cross-process fine-tuner.
"""

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
import subprocess
import sys
from types import SimpleNamespace


def digest(path):
    result = hashlib.sha256()
    with _fk_paths.relocate((Path(path)), __file__).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--repository-commit', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--input', required=True, help='NPZ: audio[B,F,T], video[B,1,T,H,W]')
    parser.add_argument('--preparation-json', required=True, help='pinned crop/filterbank/alignment settings')
    parser.add_argument('--output', required=True, help='new directory')
    args = parser.parse_args()
    repo = _fk_paths.relocate((Path(args.repository)), __file__).resolve()
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if commit != args.repository_commit or digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('upstream commit or checkpoint digest mismatch')
    # Refuse tracked source modifications; hash the submodule revision separately.
    dirty = subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code'], capture_output=True)
    staged = subprocess.run(['git', '-C', str(repo), 'diff', '--cached', '--exit-code'], capture_output=True)
    if dirty.returncode or staged.returncode:
        raise ValueError('modified upstream source is not the requested commit')
    preparation = json.loads(_fk_paths.relocate((Path(args.preparation_json)), __file__).read_text())
    for key in ('image_mean', 'image_std', 'crop_size', 'stack_order_audio', 'normalize_audio', 'filterbank_settings', 'alignment_policy'):
        if key not in preparation:
            raise ValueError('explicit preparation metadata missing: ' + key)
    import numpy as np
    import torch
    from fairseq import checkpoint_utils, utils
    sys.path.insert(0, str(_fk_paths.relocate((Path(__file__).resolve().parents[1] / 'src'), __file__)))
    from deepfake_detection.backbones import AVHubertBackend
    utils.import_user_module(SimpleNamespace(user_dir=str(_fk_paths.relocate((repo / 'avhubert'), __file__))))
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.checkpoint])
    if len(models) != 1:
        raise ValueError('expected one core AV-HuBERT checkpoint')
    for key, saved_key in (('image_mean', 'image_mean'), ('image_std', 'image_std'),
                           ('crop_size', 'image_crop_size'), ('stack_order_audio', 'stack_order_audio'),
                           ('normalize_audio', 'normalize')):
        saved = getattr(cfg.task, saved_key, None)
        if saved is None or saved != preparation[key]:
            raise ValueError('preparation differs from checkpoint task setting: ' + key)
    adapter = AVHubertBackend(models[0]).eval()
    with np.load(_fk_paths.relocate_value((args.input), __file__), allow_pickle=False) as payload:
        source = {key: torch.from_numpy(payload[key]).float() for key in ('audio', 'video')}
    if (source['video'].shape[-2:] != (preparation['crop_size'], preparation['crop_size'])
            or source['audio'].shape[1] != preparation['filterbank_settings']['nfilt'] * preparation['stack_order_audio']):
        raise ValueError('prepared source dimensions differ from preparation metadata')
    with torch.no_grad():
        features = adapter(source).cpu().numpy()
    output = _fk_paths.relocate((Path(args.output)), __file__)
    output.mkdir(parents=True, exist_ok=False)
    with (_fk_paths.relocate((output / 'features.npz'), __file__)).open('xb') as stream:
        np.savez_compressed(stream, features=features)
    fairseq_commit = subprocess.run(['git', '-C', str(_fk_paths.relocate((repo / 'fairseq'), __file__)), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    metadata = {
        'status': 'EXTRACTED_OFFICIAL_AVHUBERT_FEATURES', 'upstream_commit': commit,
        'fairseq_commit': fairseq_commit, 'checkpoint_sha256': args.checkpoint_sha256,
        'input_sha256': digest(args.input), 'preparation_sha256': digest(args.preparation_json),
        'output_sha256': digest(_fk_paths.relocate((output / 'features.npz'), __file__)), 'shape': list(features.shape),
        'python': sys.version, 'torch_version': str(torch.__version__),
    }
    (_fk_paths.relocate((output / 'metadata.json'), __file__)).write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
