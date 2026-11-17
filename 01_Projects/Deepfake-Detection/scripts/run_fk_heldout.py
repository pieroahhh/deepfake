"""Check or run FK local held-out evaluation with frozen training settings."""

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
import os
import shutil
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(_fk_paths.relocate((ROOT / 'src'), __file__)), str(_fk_paths.relocate((ROOT / 'scripts'), __file__))]
from baselines.common.io import digest, read, write
from prepare_heldout import verify


def check(args):
    import torch
    from deepfake_detection.artifacts import verified_files
    from deepfake_detection.commands import fk_asset_hashes
    from deepfake_detection.infrastructure import fk_device
    from deepfake_detection.quality import profile_hash
    from deepfake_detection.fk_caching import WINDOW_LABEL_POLICY
    from deepfake_detection.preparation import IN_FRAME_CROP_POLICY

    report = verify(args.population, args.dataset_root)
    heldout = read(_fk_paths.relocate((Path(args.population) / 'manifest.json'), __file__))
    used = read(_fk_paths.relocate((Path(args.population) / 'used-manifest.json'), __file__))
    checkpoint = torch.load(_fk_paths.relocate_value((args.checkpoint), __file__), map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != 'fk-training-v1' or 'validation_selection' not in checkpoint:
        raise ValueError('a validation-selected FK checkpoint is required')
    training_run_path = _fk_paths.relocate((Path(args.checkpoint).parent / 'run.json'), __file__)
    training_run = read(training_run_path)
    if checkpoint.get('data_signatures') != training_run['data_signatures']:
        raise ValueError('checkpoint differs from its training run')
    origin = read(args.training_cache_identity)
    if origin['source_manifest_sha256'] != report['used_manifest_sha256']:
        raise ValueError('training cache was not built from the excluded used population')
    for key in ('preprocessing', 'asset_sha256', 'window_label_policy', 'source_overlap_policy'):
        if checkpoint.get(key) != origin.get(key) or checkpoint.get(key) != training_run.get(key):
            raise ValueError('training provenance mismatch: ' + key)
    if (origin['window_label_policy'] != WINDOW_LABEL_POLICY or
            origin['preprocessing']['face_crop_policy'] != IN_FRAME_CROP_POLICY):
        raise ValueError('current preprocessing constants differ from training')
    quality = read(args.quality)
    if quality != checkpoint['preprocessing']['quality_normalization']:
        raise ValueError('quality normalization differs from selected checkpoint')
    training_rows = [{'video_id': r['video_id'], 'source_sha256': r['source_sha256']}
                     for r in used if r['split'] == 'train']
    if profile_hash(training_rows) != quality['training_population_sha256']:
        raise ValueError('quality profile does not belong to excluded training population')
    device = fk_device(args.device)
    if (str(device) != checkpoint['preprocessing']['encoder_device'] or
            str(torch.__version__) != checkpoint['preprocessing']['torch_version']):
        raise ValueError('use the same encoder device and PyTorch version as the training cache')
    assets = read(args.assets)
    artifacts = {}
    for name, key in (('visual', 'efficientnet_b0'), ('audio', 'wav2vec2_base')):
        spec = assets['artifacts'][key]
        artifacts[name] = verified_files(spec['directory'], spec['manifest'])
    for key in ('detector', 'silero'):
        if digest(assets[key]['path']) != assets[key]['sha256']:
            raise ValueError('asset checksum mismatch: ' + key)
    if fk_asset_hashes(assets, SimpleNamespace(artifacts=artifacts)) != checkpoint['asset_sha256']:
        raise ValueError('encoder/detector/VAD assets differ from selected checkpoint')
    context = {
        'schema': 'fk-local-heldout-evaluation-v1', 'scope': report['scope'],
        'population_sha256': report['manifest_sha256'],
        'used_manifest_sha256': report['used_manifest_sha256'],
        'population_audit_sha256': digest(_fk_paths.relocate((Path(args.population) / 'audit.json'), __file__)),
        'training_cache_identity_sha256': digest(args.training_cache_identity),
        'training_run_sha256': digest(training_run_path),
        'training_data_signatures': checkpoint['data_signatures'],
        'checkpoint_sha256': digest(args.checkpoint),
        'expected_labels': {row['video_id']: row['label'] for row in heldout},
    }
    print(f'PRECHECK PASSED: {len(heldout)} held-out clips; checkpoint, quality and assets verified', flush=True)
    return used + heldout, context


def run(args, combined, context):
    from deepfake_detection.fk_caching import cache_fk_manifest
    from deepfake_detection.study import evaluate_checkpoint
    from deepfake_detection.quality_fitting import exclusive_run
    from deepfake_detection.gpu_queue import gpu_slot
    root = _fk_paths.relocate((Path(args.output)), __file__).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(_fk_paths.relocate((root / 'workflow.lock'), __file__)), gpu_slot(args.device == 'cuda'):
        def ensure_json(path, value):
            if path.exists():
                if read(path) != value:
                    raise ValueError('existing workflow inputs changed; choose a new output: ' + str(path))
            else:
                write(path, value)
        ensure_json(_fk_paths.relocate((root / 'evaluation-context.json'), __file__), context)
        combined_path = _fk_paths.relocate((root / 'combined-manifest.json'), __file__)
        ensure_json(combined_path, combined)
        cache = _fk_paths.relocate((root / 'cache'), __file__)
        try:
            if not (_fk_paths.relocate((cache / 'training-manifest.json'), __file__)).exists() and shutil.disk_usage(root).free < 10 * 1024**3:
                raise ValueError('need at least 10 GiB free on the cache output drive')
            write(_fk_paths.relocate((root / 'status.json'), __file__), {'stage': 'cache', 'status': 'RUNNING'}, replace=True)
            # Always validate completed clips on rerun; only missing/corrupt clips are encoded.
            cache_fk_manifest(combined_path, args.dataset_root, args.assets, cache,
                              quality_normalization=args.quality, device=args.device,
                              resume=cache.exists(), only_split='test', evaluation_context=context)
            write(_fk_paths.relocate((root / 'status.json'), __file__), {'stage': 'evaluate', 'status': 'RUNNING'}, replace=True)
            with TemporaryDirectory(prefix='.evaluation-', dir=root) as temporary:
                staged = _fk_paths.relocate((Path(temporary) / 'test.json'), __file__)
                report = evaluate_checkpoint(_fk_paths.relocate((cache / 'training-manifest.json'), __file__), args.checkpoint,
                                             staged, split='test', device=args.device)
                staged.replace(_fk_paths.relocate((root / 'test.json'), __file__))
            write(_fk_paths.relocate((root / 'summary.json'), __file__), {'scope': context['scope'],
                  'population_sha256': context['population_sha256'],
                  'video_metrics': report['video_metrics'], 'decision': report['decision']}, replace=True)
            write(_fk_paths.relocate((root / 'status.json'), __file__), {'stage': 'complete', 'status': 'FINISHED'}, replace=True)
            print('FINISHED:', _fk_paths.relocate((root / 'test.json'), __file__), flush=True)
        except BaseException as error:
            write(_fk_paths.relocate((root / 'status.json'), __file__), {'stage': 'failed', 'status': 'FAILED', 'reason': str(error)}, replace=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='verify only; no cache/inference')
    mode.add_argument('--run', action='store_true', help='cache held-out clips then evaluate; rerun to resume')
    experiment = _fk_paths.relocate((ROOT / 'runs/two-day-001/experiment/fk'), __file__)
    parser.add_argument('--population', default=str(_fk_paths.relocate((ROOT / 'runs/heldout-001/population'), __file__)))
    parser.add_argument('--dataset-root', default='D:/AV-Deepfake1M')
    parser.add_argument('--checkpoint', default=str(_fk_paths.relocate((experiment / 'training/attempt-001/selected.pt'), __file__)))
    parser.add_argument('--training-cache-identity', default=str(_fk_paths.relocate((experiment / 'cache/cache-run.json'), __file__)))
    parser.add_argument('--quality', default=str(_fk_paths.relocate((experiment / 'quality.json'), __file__)))
    parser.add_argument('--assets', default=str(_fk_paths.relocate((ROOT / 'checkpoints/paper-alignment-001/assets.local.json'), __file__)))
    parser.add_argument('--output', default=str(_fk_paths.relocate((ROOT / 'runs/heldout-001/fk'), __file__)))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    os.environ.setdefault('HF_HOME', str(_fk_paths.relocate((ROOT / '.cache/huggingface-two-day'), __file__)))
    import torch
    torch.set_num_threads(args.threads)
    try:
        combined, context = check(args)
        if args.run:
            run(args, combined, context)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, 'error: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
