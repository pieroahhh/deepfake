"""Train four controls sequentially, then score frozen checkpoints on shared held-out caches."""

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
from contextlib import redirect_stdout, redirect_stderr
import copy
import gc
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(_fk_paths.relocate((ROOT / 'src'), __file__))]
from baselines.common.io import read, write, digest

KINDS = ('visual_only', 'audio_only', 'concatenation', 'late_fusion')
POLICIES = ('preprocessing', 'asset_sha256', 'window_label_policy', 'source_overlap_policy')


def load_checkpoint(path):
    import torch
    return torch.load(_fk_paths.relocate_value((path), __file__), map_location='cpu', weights_only=True)


def ensure_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('workflow inputs changed: ' + str(path))
    else:
        write(path, value)


def check_inputs(args):
    from deepfake_detection.infrastructure import fk_device
    from deepfake_detection.data import audit
    from deepfake_detection.forensics import canonical
    import hashlib
    training, test = read(args.training_cache), read(args.test_cache)
    reference = load_checkpoint(args.reference_checkpoint)
    heldout_path = _fk_paths.relocate((Path(args.population) / 'manifest.json'), __file__)
    heldout, report = read(heldout_path), read(_fk_paths.relocate((Path(args.population) / 'audit.json'), __file__))
    if digest(heldout_path) != report['manifest_sha256']:
        raise ValueError('held-out manifest checksum mismatch')
    used_path = _fk_paths.relocate((Path(args.population) / 'used-manifest.json'), __file__)
    used = read(used_path)
    if digest(used_path) != report['used_manifest_sha256'] or training['source_manifest_sha256'] != report['used_manifest_sha256']:
        raise ValueError('training cache differs from held-out exclusion population')
    audit(used); audit(heldout)
    for key in ('video_id', 'source_family', 'source_sha256'):
        values = [r[key] for r in heldout]
        if len(set(values)) != len(values) or set(values) & {r[key] for r in used}:
            raise ValueError('held-out duplication or leakage: ' + key)
    if len(heldout) != 1000 or any(r['split'] != 'test' for r in heldout):
        raise ValueError('expected the frozen 1000-clip held-out population')
    context = test.get('evaluation_context', {})
    expected = {r['video_id']: r['label'] for r in heldout}
    if (context.get('population_sha256') != report['manifest_sha256'] or
            context.get('expected_labels') != expected or
            context.get('checkpoint_sha256') != digest(args.reference_checkpoint) or
            context.get('training_data_signatures') != reference['data_signatures']):
        raise ValueError('test cache provenance differs from FK reference/held-out population')
    if reference.get('schema') != 'fk-training-v1' or 'validation_selection' not in reference:
        raise ValueError('selected FK reference checkpoint required')
    for key in POLICIES:
        if training.get(key) != test.get(key) or training.get(key) != reference.get(key):
            raise ValueError('cache/reference policy mismatch: ' + key)
    for manifest, population, splits in ((training, used, ('train', 'validation')), (test, heldout, ('test',))):
        expected_rows = {r['video_id']: r for r in population}
        assessments = manifest['assessments']
        if len(assessments) != len(expected_rows) or {r['video_id'] for r in assessments} != set(expected_rows):
            raise ValueError('cache assessments do not cover the frozen population')
        for item in assessments:
            row = expected_rows[item['video_id']]
            if any(item[k] != row[k] for k in ('label', 'split', 'source_family')):
                raise ValueError('assessment population mismatch')
        for split in splits:
            for item in manifest[split]:
                row = expected_rows[item['video_id']]
                if any(item[k] != row[k] for k in ('label', 'split', 'source_family', 'source_sha256')):
                    raise ValueError('cached row population mismatch')
                provenance = item['cache_provenance']
                policy_sha = hashlib.sha256(canonical(manifest['preprocessing'])).hexdigest()
                if (provenance['preprocessing_sha256'] != policy_sha or
                        provenance['asset_sha256'] != manifest['asset_sha256'] or
                        provenance['window_label_policy'] != manifest['window_label_policy']):
                    raise ValueError('cached feature policy mismatch')
    config = read(args.training_config)
    if config != reference['config'] or args.epochs != reference['epoch']:
        raise ValueError('controls must use the FK reference training config and epoch budget')
    fk_device(args.device)
    print(f"PRECHECK PASSED: train {len(training['train'])} usable / 3000; "
          f"validation {len(training['validation'])} / 500; test {len(test['test'])} / 1000", flush=True)
    return training, test, reference


def validate_selected(checkpoint, kind, config, reference):
    if (checkpoint.get('schema') != 'fk-training-v1' or
            checkpoint.get('model_spec', {}).get('model_kind') != kind or
            checkpoint.get('config') != config or
            checkpoint.get('data_signatures') != reference['data_signatures']):
        raise ValueError('control checkpoint model/config/training population mismatch')
    for key in POLICIES:
        if checkpoint.get(key) != reference.get(key):
            raise ValueError('control checkpoint policy mismatch: ' + key)


def fit_control(kind, directory, args, reference):
    """Restart at the last saved epoch; recover calibration after a completed epoch budget."""
    import torch
    from deepfake_detection.infrastructure import train_fk_manifest, CachedVideoPopulation
    from deepfake_detection.training import load_fk_checkpoint, _save_checkpoint
    from deepfake_detection.calibration import select_decision
    config = read(args.training_config)
    profile = copy.deepcopy(read(args.model_config))
    profile.pop('ablation', None)
    profile['model_kind'] = kind
    ensure_json(_fk_paths.relocate((directory / 'model.json'), __file__), profile)
    attempts = sorted(directory.glob('attempt-*'))
    latest = attempts[-1] if attempts else None
    if latest and (_fk_paths.relocate((latest / 'selected.pt'), __file__)).exists():
        selected = load_checkpoint(_fk_paths.relocate((latest / 'selected.pt'), __file__))
        validate_selected(selected, kind, config, reference)
        if 'validation_selection' not in selected:
            raise ValueError('selected control has no validation calibration')
        print(kind + ': reusing selected checkpoint', flush=True)
        return _fk_paths.relocate((latest / 'selected.pt'), __file__)
    last = load_checkpoint(_fk_paths.relocate((latest / 'last.pt'), __file__)) if latest and (_fk_paths.relocate((latest / 'last.pt'), __file__)).exists() else None
    if last is not None:
        validate_selected(last, kind, config, reference)
    finished = last is not None and (last['epoch'] >= args.epochs or last['stale_epochs'] >= config['patience'])
    if not finished:
        destination = _fk_paths.relocate((directory / f'attempt-{len(attempts)+1:03d}'), __file__)
        print(f'{kind}: training {destination.name}', flush=True)
        train_fk_manifest(args.training_cache, _fk_paths.relocate((directory / 'model.json'), __file__), args.training_config,
                          destination, epochs=args.epochs, resume_from=_fk_paths.relocate((latest / 'last.pt'), __file__) if last else None,
                          calibrate=False, device=args.device)
        latest = destination
        last = load_checkpoint(_fk_paths.relocate((latest / 'last.pt'), __file__))
    # Publish the best inherited state even if a crash happened just after last.pt.
    validate_selected(last, kind, config, reference)
    best = dict(last)
    best['model_state'] = last['best_model_state']
    best['selected_epoch'] = min(last['history'], key=lambda e: e['validation']['loss'])['epoch']
    best.pop('optimizer_state', None)
    _save_checkpoint(_fk_paths.relocate((latest / 'best.pt'), __file__), best)
    model, checkpoint = load_fk_checkpoint(_fk_paths.relocate((latest / 'best.pt'), __file__), device=args.device)
    manifest = read(args.training_cache)
    validation = CachedVideoPopulation(manifest['validation'], _fk_paths.relocate((Path(args.training_cache)), __file__).resolve().parent)
    print(kind + ': selecting decision parameters on validation', flush=True)
    selection = select_decision(model, validation)
    selection['source_checkpoint_sha256'] = digest(_fk_paths.relocate((latest / 'best.pt'), __file__))
    checkpoint['decision'] = selection['decision']
    checkpoint['validation_selection'] = selection
    write(_fk_paths.relocate((latest / 'selection.json'), __file__), selection, replace=True)
    _save_checkpoint(_fk_paths.relocate((latest / 'selected.pt'), __file__), checkpoint)
    del model
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return _fk_paths.relocate((latest / 'selected.pt'), __file__)


def evaluation_view(test, source_path, checkpoint_path):
    """Rebind the verified shared features to a control trained on the same tensors."""
    checkpoint = load_checkpoint(checkpoint_path)
    context = test['evaluation_context']
    if checkpoint['data_signatures'] != context['training_data_signatures']:
        raise ValueError('cannot reuse held-out features for a different training population')
    for key in POLICIES:
        if checkpoint.get(key) != test.get(key):
            raise ValueError('control cache policy mismatch: ' + key)
    view = copy.deepcopy(test)
    for split in ('train', 'validation', 'test'):
        for row in view.get(split, []):
            path = (_fk_paths.relocate((Path(source_path).resolve().parent / row['cache']), __file__)).resolve()
            path.relative_to(_fk_paths.relocate((Path(source_path)), __file__).resolve().parent)
            row['cache'] = str(path)
    view['evaluation_context'].update(
        checkpoint_sha256=digest(checkpoint_path),
        training_run_sha256=digest(_fk_paths.relocate((Path(checkpoint_path).parent / 'run.json'), __file__)),
        shared_feature_manifest_sha256=digest(source_path),
        model_kind=checkpoint['model_spec']['model_kind'])
    return view


def evaluate_once(manifest, checkpoint, output, device):
    from deepfake_detection.study import evaluate_checkpoint
    if output.exists():
        report = read(output)
        if report['checkpoint_sha256'] != digest(checkpoint) or report['manifest_sha256'] != digest(manifest):
            raise ValueError('existing evaluation belongs to different inputs: ' + str(output))
        return report
    with TemporaryDirectory(prefix='.evaluation-', dir=output.parent) as temporary:
        staged = _fk_paths.relocate((Path(temporary) / 'report.json'), __file__)
        report = evaluate_checkpoint(manifest, checkpoint, staged, device=device,
                                     split='test' if output.name == 'test.json' else 'validation')
        staged.rename(output)
    return report


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile
    def write(self, value):
        self.console.write(value); self.logfile.write(value)
        self.flush()
        return len(value)
    def flush(self):
        self.console.flush(); self.logfile.flush()


def workflow(args, training, test, reference):
    from deepfake_detection.quality_fitting import exclusive_run
    from deepfake_detection.gpu_queue import gpu_slot
    root = _fk_paths.relocate((Path(args.output)), __file__).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from deepfake_detection.provenance import git_provenance
    # Hash source bytes as well as Git metadata so uncommitted edits cannot pass resume.
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted((_fk_paths.relocate((ROOT / 'src/deepfake_detection'), __file__)).glob('*.py'))}
    source_hashes['scripts/run_shared_controls.py'] = digest(__file__)
    identity = {'schema': 'shared-controls-workflow-v1', 'models': list(KINDS), 'epochs': args.epochs,
                'seed': reference['config']['seed'], 'device': args.device, 'threads': args.threads,
                'training_cache_sha256': digest(args.training_cache), 'test_cache_sha256': digest(args.test_cache),
                'reference_checkpoint_sha256': digest(args.reference_checkpoint),
                'model_config_sha256': digest(args.model_config), 'training_config_sha256': digest(args.training_config),
                'source_sha256': source_hashes, **git_provenance()}
    with exclusive_run(_fk_paths.relocate((root / 'workflow.lock'), __file__)), gpu_slot(args.device == 'cuda'):
        ensure_json(_fk_paths.relocate((root / 'workflow.json'), __file__), identity)
        with (_fk_paths.relocate((root / 'workflow.log'), __file__)).open('a', encoding='utf-8') as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                selected = {}
                for index, kind in enumerate(KINDS, 1):
                    print(f'[{index}/4] {kind}', flush=True)
                    directory = _fk_paths.relocate((root / kind), __file__)
                    directory.mkdir(exist_ok=True)
                    write(_fk_paths.relocate((root / 'status.json'), __file__), {'status': 'RUNNING', 'model': kind, 'stage': 'train/calibrate'}, replace=True)
                    selected[kind] = fit_control(kind, directory, args, reference)
                    evaluate_once(args.training_cache, selected[kind], _fk_paths.relocate((directory / 'validation.json'), __file__), args.device)
                # Freeze every selected checkpoint before opening test features for scoring.
                frozen = {kind: digest(path) for kind, path in selected.items()}
                ensure_json(_fk_paths.relocate((root / 'selected-checkpoints.json'), __file__), frozen)
                reports = {}
                for kind, checkpoint in selected.items():
                    if digest(checkpoint) != frozen[kind]:
                        raise ValueError('selected checkpoint changed before test')
                    directory = _fk_paths.relocate((root / kind), __file__)
                    write(_fk_paths.relocate((root / 'status.json'), __file__), {'status': 'RUNNING', 'model': kind, 'stage': 'test'}, replace=True)
                    print(kind + ': evaluating held-out cache', flush=True)
                    view = _fk_paths.relocate((directory / 'test-cache-view.json'), __file__)
                    ensure_json(view, evaluation_view(test, args.test_cache, checkpoint))
                    report = evaluate_once(view, checkpoint, _fk_paths.relocate((directory / 'test.json'), __file__), args.device)
                    reports[kind] = {'video_metrics': report['video_metrics'], 'decision': report['decision'],
                                     'checkpoint_sha256': frozen[kind]}
                write(_fk_paths.relocate((root / 'summary.json'), __file__), {'scope': test['evaluation_context']['scope'],
                      'population_sha256': test['evaluation_context']['population_sha256'],
                      'epochs': args.epochs, 'seed': reference['config']['seed'], 'models': reports}, replace=True)
                write(_fk_paths.relocate((root / 'status.json'), __file__), {'status': 'FINISHED', 'stage': 'complete'}, replace=True)
                print('FINISHED:', _fk_paths.relocate((root / 'summary.json'), __file__), flush=True)
            except BaseException as error:
                write(_fk_paths.relocate((root / 'status.json'), __file__), {'status': 'FAILED', 'reason': str(error)}, replace=True)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--run', action='store_true')
    fk = _fk_paths.relocate((ROOT / 'runs/two-day-001/experiment/fk'), __file__)
    parser.add_argument('--training-cache', default=str(_fk_paths.relocate((fk / 'cache/training-manifest.json'), __file__)))
    parser.add_argument('--test-cache', default=str(_fk_paths.relocate((ROOT / 'runs/heldout-001/fk/cache/training-manifest.json'), __file__)))
    parser.add_argument('--reference-checkpoint', default=str(_fk_paths.relocate((fk / 'training/attempt-001/selected.pt'), __file__)))
    parser.add_argument('--population', default=str(_fk_paths.relocate((ROOT / 'runs/heldout-001/population'), __file__)))
    parser.add_argument('--model-config', default=str(_fk_paths.relocate((ROOT / 'configs/ours.provisional.json'), __file__)))
    parser.add_argument('--training-config', default=str(_fk_paths.relocate((ROOT / 'configs/fk.training-v1.json'), __file__)))
    parser.add_argument('--output', default=str(_fk_paths.relocate((ROOT / 'runs/shared-controls-001'), __file__)))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.threads < 1 or args.epochs < 1:
        parser.error('threads and epochs must be positive')
    os.environ.setdefault('HF_HOME', str(_fk_paths.relocate((ROOT / '.cache/huggingface-two-day'), __file__)))
    os.environ['FK_PROGRESS_SECONDS'] = '10'
    import torch
    torch.set_num_threads(args.threads)
    try:
        inputs = check_inputs(args)
        if args.run:
            workflow(args, *inputs)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, 'error: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
