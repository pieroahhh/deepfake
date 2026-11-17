"""Training-only, CPU head diagnostics; never releases the full-pilot gate.

Compare unchanged DSTNet heads on raw versus training-standardized pooled
features. Each arm uses the same initialization, batches and 200-update budget.
Original models and inputs are read-only. --resume reuses verified completed
stages; an interrupted 200-update stage restarts from its fixed initialization.
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

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(_fk_paths.relocate((ROOT / 'src'), __file__))]
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from baselines.avfakenet.model import DSTNet, classification_loss
from baselines.common.training import load_checkpoint


def digest(path):
    h = hashlib.sha256()
    with _fk_paths.relocate((Path(path)), __file__).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(_fk_paths.relocate((Path(path)), __file__).read_text(encoding='utf-8'))


def write(path, value):
    path = _fk_paths.relocate((Path(path)), __file__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalization(features, fit_indices, standardized):
    """Fit on training rows only, including inside grouped folds."""
    if standardized:
        scaler = StandardScaler().fit(features[fit_indices])
        mean, scale = scaler.mean_, scaler.scale_
    else:
        mean, scale = np.zeros(features.shape[1]), np.ones(features.shape[1])
    transformed = ((features.astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError('Non-finite normalized features')
    return transformed, mean, scale


def balanced_batches(labels, indices, seed=7, steps=200, batch_size=8):
    rng = np.random.default_rng(seed)
    classes = [np.asarray(indices)[labels[indices] == label] for label in (0, 1)]
    half = batch_size // 2
    if batch_size % 2 or any(len(rows) < half for rows in classes):
        raise ValueError('Insufficient rows for balanced batches')
    queues = [[], []]
    for _ in range(steps):
        batch = []
        for label in (0, 1):
            if len(queues[label]) < half:
                queues[label].extend(rng.permutation(classes[label]).tolist())
            batch.extend(queues[label][:half])
            del queues[label][:half]
        yield np.asarray(batch)[rng.permutation(batch_size)]


def score(head, features, labels, smoothing):
    head.eval()
    relu_stats = {}
    hooks = []
    for name, module in head.named_modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(
                lambda module, args, out, key=name: relu_stats.__setitem__(
                    key, {'zero_fraction': float((out == 0).float().mean()),
                          'dead_units': int((out == 0).all(dim=0).sum())})))
    try:
        with torch.inference_mode():
            logits = head(features)
            probs = logits.softmax(-1)[:, 1].numpy()
            pred = logits.argmax(-1).numpy()
            loss = float(classification_loss(logits, labels, smoothing))
    finally:
        for hook in hooks:
            hook.remove()
    y = labels.numpy()
    return {'loss': loss, 'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'roc_auc': float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
            'confusion': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            'probability_range': [float(probs.min()), float(probs.max())],
            'relu': relu_stats}, probs


def fit_stage(template, features, labels, train_rows, test_rows, standardized, profile,
              stage, destination, progress, identity):
    destination.mkdir(parents=True, exist_ok=True)
    stamp = _fk_paths.relocate((destination / 'complete.json'), __file__)
    if stamp.exists():
        saved = read(stamp)
        if saved['identity'] != identity:
            raise ValueError('Completed stage identity changed: ' + stage)
        for name, expected in saved['files'].items():
            if digest(destination / name) != expected:
                raise ValueError('Completed stage modified: ' + stage)
        progress(stage, 200, 'REUSED')
        return read(_fk_paths.relocate((destination / 'metrics.json'), __file__))
    transformed, mean, scale = normalization(features, train_rows, standardized)
    x = torch.from_numpy(transformed)
    y = torch.tensor(labels, dtype=torch.long)
    head = copy.deepcopy(template)
    initial_state = {key: value.clone() for key, value in head.state_dict().items()}
    torch.manual_seed(7)  # Match dropout sequence between paired arms.
    config = profile['training']
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0001,
        weight_decay=config['weight_decay'], betas=tuple(config['betas']), eps=config['epsilon'])
    initial, _ = score(head, x[train_rows], y[train_rows], config['label_smoothing'])
    history = []
    for step, rows in enumerate(balanced_batches(labels, train_rows), 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = classification_loss(head(x[rows]), y[rows], config['label_smoothing'])
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss')
        loss.backward()
        gradient_norms = {name: float(param.grad.norm()) for name, param in head.named_parameters()}
        norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True))
        optimizer.step()
        if step % 20 == 0:
            scores, _ = score(head, x[train_rows], y[train_rows], config['label_smoothing'])
            history.append({'update': step, 'training': scores, 'gradient_norms_before_clip': gradient_norms,
                            'total_gradient_norm_before_clip': norm})
            write(_fk_paths.relocate((destination / 'history.json'), __file__), history)
            progress(stage, step, 'RUNNING')
    final, _ = score(head, x[train_rows], y[train_rows], config['label_smoothing'])
    evaluated, probabilities = score(head, x[test_rows], y[test_rows], config['label_smoothing'])
    reduction = (initial['loss'] - final['loss']) / initial['loss']
    result = {'stage': stage, 'updates': 200, 'standardized_features': standardized,
              'normalizer_fitted_rows': train_rows.tolist(), 'evaluation_rows': test_rows.tolist(),
              'initial_training': initial, 'final_training': final, 'evaluation': evaluated,
              'evaluation_probabilities': probabilities.tolist(), 'relative_loss_reduction': reduction,
              'changed_head_parameter_tensors': sum(not torch.equal(v, initial_state[k]) for k, v in head.state_dict().items()),
              'head_training_targets_passed': final['balanced_accuracy'] >= .8 and reduction >= .2,
              'full_pilot_released': False}
    torch.save({'head_state': head.state_dict(), 'mean': torch.tensor(mean), 'scale': torch.tensor(scale),
                'updates': 200, 'identity': identity, 'diagnostic_only': True}, _fk_paths.relocate((destination / 'head.pt'), __file__))
    write(_fk_paths.relocate((destination / 'metrics.json'), __file__), result)
    write(stamp, {'identity': identity, 'files': {name: digest(destination / name)
          for name in ('metrics.json', 'head.pt', 'history.json')}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    bundle, output = args.bundle.resolve(), args.output.resolve()
    if output == bundle or bundle in output.parents:
        raise ValueError('Output must be separate from original bundle')
    if output.exists() and not args.resume:
        raise ValueError('Use a new directory or --resume')
    sealed = {'SHA256SUMS.txt': digest(_fk_paths.relocate((bundle / 'SHA256SUMS.txt'), __file__))}
    for line in (_fk_paths.relocate((bundle / 'SHA256SUMS.txt'), __file__)).read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (bundle / name).resolve()
        if bundle not in path.parents or digest(path) != expected:
            raise ValueError('Bundle checksum/path mismatch: ' + name)
        sealed[name] = expected
    profile = read(_fk_paths.relocate((bundle / 'recovery-profile.json'), __file__))
    code = read(_fk_paths.relocate((bundle / 'tiny/attempt-003/code.json'), __file__))
    if digest(_fk_paths.relocate((ROOT / 'baselines/avfakenet/model.py'), __file__)) != code['files']['baselines/avfakenet/model.py']:
        raise ValueError('Model source differs from returned recovery code')
    manifest_path = _fk_paths.relocate((ROOT / 'runs/two-day-001/population/manifest.json'), __file__)
    manifest_hash = digest(manifest_path)
    manifest = {row['video_id']: row for row in read(manifest_path)}
    identity = {'protocol': 'avfakenet-head-scaling-diagnostic-v1', 'bundle': sealed,
        'manifest_sha256': manifest_hash, 'script_sha256': digest(_fk_paths.relocate((Path(__file__)), __file__)),
        'model_sha256': digest(_fk_paths.relocate((ROOT / 'baselines/avfakenet/model.py'), __file__)), 'seed': 7,
        'updates_per_stage': 200, 'learning_rate': .0001, 'batch_size': 8,
        'gradient_clip_norm': 1.0, 'torch': torch.__version__, 'full_pilot': 'ON HOLD'}
    output.mkdir(parents=True, exist_ok=True)
    if (_fk_paths.relocate((output / 'identity.json'), __file__)).exists() and read(_fk_paths.relocate((output / 'identity.json'), __file__)) != identity:
        raise ValueError('Resume inputs/code differ')
    write(_fk_paths.relocate((output / 'identity.json'), __file__), identity)
    torch.set_num_threads(2)
    started = time.monotonic()
    def progress(stage, update, status):
        event = {'status': status, 'stage': stage, 'update': update, 'updates_in_head_stage': 200,
                 'elapsed_seconds_this_session': round(time.monotonic() - started, 2), 'full_pilot': 'ON HOLD'}
        write(_fk_paths.relocate((output / 'progress.json'), __file__), event)
        print(json.dumps(event), flush=True)
    result = {'protocol': identity['protocol'], 'device': 'cpu', 'full_pilot': 'ON HOLD', 'branches': {}}
    try:
        for branch in ('visual', 'audio'):
            definitions = read(bundle / f'tiny/{branch}/samples.json')
            rows = definitions['samples']
            if definitions['population_sha256'] != manifest_hash:
                raise ValueError('Wrong population')
            for row in rows:
                source = manifest[row['video_id']]
                if source['split'] != 'train' or row['split'] != 'train' or source['source_sha256'] != row['source_sha256'] or source['source_family'] != row['source_family']:
                    raise ValueError('Wrong sample membership')
                expected = source['audio_label'] if branch == 'audio' else int(any(
                    a <= row['source_timestamp'] < b for a, b in source['visual_manipulated_intervals']))
                if row['annotation_label'] != expected:
                    raise ValueError('Wrong annotation label')
            checkpoint = load_checkpoint(bundle / f'tiny/attempt-003/{branch}/last.pt')
            run = read(bundle / f'tiny/attempt-003/{branch}/run.json')
            if checkpoint['identity'] != run['identity'] or run['identity']['profile'] != profile:
                raise ValueError('Wrong checkpoint identity')
            prepared = read(bundle / f'tiny/{branch}/prepared.json')
            if prepared != run['identity']['source']['prepared'] or any(
                prepared[name + '_sha256'] != digest(bundle / f'tiny/{branch}/{name}.{ext}')
                for name, ext in [('pixels', 'npy'), ('samples', 'json')]):
                raise ValueError('Prepared input binding differs')
            model = DSTNet(profile, branch).eval().requires_grad_(False)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            pixels = np.load(_fk_paths.relocate_value((bundle / f'tiny/{branch}/pixels.npy'), __file__), allow_pickle=False)
            inputs = torch.from_numpy(pixels.copy()).float()
            if branch == 'visual':
                inputs = inputs.permute(0, 3, 1, 2).contiguous() / 255
            chunks = []
            hook = model.output_block.register_forward_pre_hook(
                lambda module, args: chunks.append(args[0].detach().numpy().copy()))
            try:
                with torch.inference_mode():
                    for start in range(0, len(rows), 8):
                        model(inputs[start:start + 8])
                        progress(branch + '/extract', min(start + 8, len(rows)), 'EXTRACTING')
            finally:
                hook.remove()
            features = np.concatenate(chunks)
            labels = np.array([r['annotation_label'] for r in rows], dtype=np.int64)
            groups = np.array([r['source_family'] for r in rows])
            torch.manual_seed(7)
            template = copy.deepcopy(model.output_block).requires_grad_(True)
            for module in template.modules():
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
            source_unchanged = all(torch.equal(value, checkpoint['model_state'][name]) for name, value in model.state_dict().items())
            if not source_unchanged:
                raise ValueError('Source model was changed')
            del model, checkpoint, inputs
            np.savez(output / f'{branch}-features.npz', features=features, labels=labels, groups=groups)
            std = features.std(axis=0)
            branch_result = {'samples': len(rows), 'families': len(set(groups)),
                'feature_shape': list(features.shape), 'feature_std_min_median_max':
                [float(std.min()), float(np.median(std)), float(std.max())],
                'feature_abs_mean': float(np.abs(features).mean()), 'source_model_unchanged': source_unchanged,
                'arms': {}}
            folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=7).split(features, labels, groups))
            for standardized in (False, True):
                arm = 'standardized' if standardized else 'raw'
                all_rows = np.arange(len(rows))
                tiny = fit_stage(template, features, labels, all_rows, all_rows, standardized, profile,
                    f'{branch}/{arm}/tiny', _fk_paths.relocate((output / branch / arm / 'tiny'), __file__), progress, fingerprint(identity))
                oof = np.full(len(rows), np.nan)
                fold_results = []
                for fold, (train, test) in enumerate(folds):
                    if set(groups[train]) & set(groups[test]):
                        raise ValueError('Family leakage')
                    fitted = fit_stage(template, features, labels, train, test, standardized, profile,
                        f'{branch}/{arm}/fold-{fold}', output / branch / arm / f'fold-{fold}', progress, fingerprint(identity))
                    oof[test] = fitted['evaluation_probabilities']
                    fold_results.append({'fold': fold, 'train': train.tolist(), 'test': test.tolist(),
                                         'evaluation': fitted['evaluation']})
                if not np.isfinite(oof).all():
                    raise ValueError('Incomplete grouped predictions')
                branch_result['arms'][arm] = {'tiny': tiny, 'grouped_folds': fold_results,
                    'pooled_grouped_balanced_accuracy': float(balanced_accuracy_score(labels, oof > .5)),
                    'pooled_grouped_roc_auc': float(roc_auc_score(labels, oof)),
                    'out_of_fold_probabilities': oof.tolist()}
                result['branches'][branch] = branch_result
                write(_fk_paths.relocate((output / 'summary.json'), __file__), result)
        if not all(digest(bundle / name) == value for name, value in sealed.items()) or digest(manifest_path) != manifest_hash:
            raise ValueError('Original inputs changed')
        write(_fk_paths.relocate((output / 'source-preservation.json'), __file__), {'all_original_files_unchanged': True, 'bundle_files': sealed,
                                                   'manifest_sha256': manifest_hash})
        result['limitations'] = ['Tiny training diagnostics only; not held-out performance.',
            'Backbone saw all tiny examples during attempt 003; grouped head folds are not end-to-end unseen tests.',
            'Standardization is an experimental preparation change, not an established paper reproduction.',
            'Head-target success does not release the full-training sanity gate or full pilot.',
            'The audio experiment uses frozen whole-clip inputs; no annotation-guided test cropping.']
        write(_fk_paths.relocate((output / 'summary.json'), __file__), result)
        progress('complete', 200, 'FINISHED')
    except Exception as exc:
        write(_fk_paths.relocate((output / 'progress.json'), __file__), {'status': 'FAILED', 'error': str(exc), 'full_pilot': 'ON HOLD'})
        raise


if __name__ == '__main__':
    main()
