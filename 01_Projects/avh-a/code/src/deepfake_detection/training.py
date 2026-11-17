"""Whole-video FK training, validation-only selection, and resumable checkpoints.

Each optimizer step retains all usable windows from one video in a single graph.
Frozen feature extraction and audited cache loading occur upstream.
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


import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from .cli import save
from .core import Decision, supervision
from .evaluation import detection_metrics
from .model import EvidenceModel, multilevel_loss
from .provenance import git_provenance

TENSORS = ('visual_features', 'audio_features', 'visual_times', 'audio_times', 'quality')


def model_spec(model):
    if hasattr(model, 'spec'):
        return model.spec
    return {
        'visual_dim': model.gru.input_size, 'audio_dim': model.audio.in_features,
        'hidden': model.gru.hidden_size, 'dimension': model.visual.out_features,
        'quality_dim': model.quality_dim,
        'disabled_evidence': list(model.disabled_evidence), 'uniform_weights': model.uniform_weights,
    }


def seed_everything(seed, deterministic):
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def _rng_state(device):
    state = np.random.get_state()
    return {
        'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
        'torch': torch.get_rng_state(),
        'cuda': [torch.cuda.get_rng_state(device)] if device.type == 'cuda' else [],
    }


def _restore_rng(state, device):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'][0].cpu(), device)


def _population_signature(rows, split):
    if not rows:
        raise ValueError(f'{split} population is empty')
    ids, families, hashes = set(), set(), set()
    digest = hashlib.sha256()
    metadata = getattr(rows, 'metadata', rows)
    order = sorted(range(len(rows)), key=lambda index: metadata[index]['video_id'])
    for index in order:
        row = rows[index]
        if row['split'] != split:
            raise ValueError(f'{split} loader contains a different split; test inference is forbidden here')
        if not isinstance(row['video_id'], str) or not row['video_id'] or row['video_id'] in ids:
            raise ValueError('missing/duplicate video ID')
        if not isinstance(row['source_family'], str) or not row['source_family']:
            raise ValueError('known source family is required')
        source_hash = row['source_sha256']
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(c not in '0123456789abcdef' for c in source_hash)):
            raise ValueError('source SHA-256 is required')
        ids.add(row['video_id'])
        families.add(row['source_family'])
        hashes.add(source_hash)
        metadata = {key: row[key] for key in (
            'video_id', 'source_family', 'source_sha256', 'split', 'label',
            'temporal_complete', 'manipulated_intervals', 'window_intervals',
        )}
        metadata['feature_provenance'] = row.get('feature_provenance')
        digest.update(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
        for key in TENSORS:
            tensor = row[key]
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f'{key} must contain finite floating-point cached features')
            # Trainable extractor graphs must never masquerade as frozen caches.
            if tensor.requires_grad:
                raise ValueError('feature caches must be detached frozen encoder outputs')
            digest.update(key.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest(), ids, families, hashes


def forward_video(model, row):
    labels = supervision(
        row['label'], row['temporal_complete'], row['manipulated_intervals'],
        row['window_intervals'], bins=model.decision.bins,
    )
    if len(row['window_intervals']) != row['visual_features'].shape[0]:
        raise ValueError('usable-window intervals and feature batch differ')
    if not row['window_intervals']:
        return None, None, labels
    if any(abs(end - start - 2.0) > 1e-9 for start, end in row['window_intervals']):
        raise ValueError('FK feature windows must each span two seconds')
    device = next(model.parameters()).device
    values = {name: row[name].to(device) for name in TENSORS}
    outputs = model(**values)
    score = model.aggregate_video(outputs['window_scores'])
    return outputs, score, labels


def _epoch(model, rows, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    order = list(range(len(rows)))
    if training:
        random.shuffle(order)
    total, contributing = 0.0, 0
    predictions = []
    with torch.set_grad_enabled(training):
        for index in order:
            row = rows[index]
            outputs, score, labels = forward_video(model, row)
            prediction = None if score is None else int(score.detach().item() >= model.decision.video_threshold)
            predictions.append({
                'video_id': row['video_id'], 'label': row['label'],
                'status': 'UNASSESSED' if score is None else 'ASSESSED',
                'prediction': prediction, 'score': None if score is None else score.detach().item(),
            })
            known = labels['video_eligible'] or any(
                target['window'] is not None or any(value is not None for value in target['bins'])
                for target in labels['windows']
            )
            if outputs is None or not known:
                continue
            loss = multilevel_loss(
                outputs, score, row['label'], labels,
                config['window_loss_weight'], config['bin_loss_weight'],
            )
            if 'stream_bins' in outputs:
                # Independent branch losses: no fused-score gradient couples the branches.
                losses = []
                for bins in outputs['stream_bins']:
                    ws = model._hybrid(bins, model.decision.top_k, model.decision.peak_weight)
                    losses.append(multilevel_loss({'bin_scores': bins, 'window_scores': ws},
                        model.aggregate_video(ws), row['label'], labels,
                        config['window_loss_weight'], config['bin_loss_weight']))
                loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite whole-video loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
            total += loss.detach().item()
            contributing += 1
    if contributing == 0:
        raise ValueError('population contains no videos with usable supervised terms')
    return {'loss': total / contributing, 'supervised_videos': contributing,
            'detection': detection_metrics(predictions)}


def _save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_fk(model, train_rows, validation_rows, config, output, *, epochs=None, resume_from=None,
           manifest_policy=None):
    """Train grouped cached features; preserve Eq.52 sums within each video.

    Per-video losses are averaged only for epoch reporting/selection. Input rows
    must already describe usable windows. Test split records are rejected.
    """
    if (config['profile_id'] != 'fk-training-reconstruction-v1' or config['optimizer'] != 'AdamW'
            or config['batch_unit'] != 'one_complete_video_all_usable_windows'
            or config['selection_metric'] != 'mean_validation_video_loss'
            or config['checkpoint_ties'] != 'first_strict_improvement'):
        raise ValueError('unsupported training profile')
    for key in ('learning_rate', 'gradient_clip_norm'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('weight_decay', 'window_loss_weight', 'bin_loss_weight'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('maximum_epochs', 'patience'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    budget = config['maximum_epochs'] if epochs is None else epochs
    if type(budget) is not int or not 1 <= budget <= config['maximum_epochs']:
        raise ValueError('invalid epoch budget')
    training = _population_signature(train_rows, 'train')
    validation = _population_signature(validation_rows, 'validation')
    manifest_policy = manifest_policy or {}
    overlap_policy = manifest_policy.get('source_overlap_policy','error')
    if overlap_policy not in ('error','drop-train','keep-official'):
        raise ValueError('invalid source overlap policy')
    checked = (1,) if overlap_policy == 'keep-official' else (1,2,3)
    if any(training[i] & validation[i] for i in checked):
        raise ValueError('train/validation video, source-family or source-hash leakage')
    signatures = {'train': training[0], 'validation': validation[0]}
    manifest_policy = {**manifest_policy, 'train_validation_overlap':{
        'source_families':len(training[2] & validation[2]),
        'content_hashes':len(training[3] & validation[3])}}
    spec = model_spec(model)
    device = next(model.parameters()).device
    manifest_policy = manifest_policy or {}
    provenance = git_provenance()
    gpu_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    history, best_loss, best_state, stale, start = [], math.inf, None, 0, 1
    step, resume_device = 0, None
    seed_everything(config['seed'], config['deterministic'])
    if resume_from is not None:
        previous = torch.load(_fk_paths.relocate_value((resume_from), __file__), map_location='cpu', weights_only=True)
        if previous.get('schema') != 'fk-training-v1' or 'optimizer_state' not in previous:
            raise ValueError('resume requires last.pt with optimizer/RNG state, not an inference checkpoint')
        if (previous['config'] != config or previous['model_spec'] != spec
                or previous['data_signatures'] != signatures or previous['decision'] != model.decision.to_dict()
                or any(previous.get(key) != value for key, value in manifest_policy.items())):
            raise ValueError('resume configuration, feature population or architecture changed')
        model.load_state_dict(previous['model_state'])
        optimizer.load_state_dict(previous['optimizer_state'])
        history, best_loss = previous['history'], previous['best_loss']
        step = previous.get('step', sum(entry['training']['supervised_videos'] for entry in history))
        resume_device = previous.get('device')
        best_state, stale, start = previous['best_model_state'], previous['stale_epochs'], previous['epoch'] + 1
        _restore_rng(previous['rng'], device)
        if start > budget or stale >= config['patience']:
            raise ValueError('checkpoint has already completed the requested budget or early stopping')
    output = _fk_paths.relocate((Path(output)), __file__)
    output.mkdir(parents=True, exist_ok=False)
    save(_fk_paths.relocate((output / 'run.json'), __file__), {
        'config': config, 'model_spec': spec, 'decision': model.decision.to_dict(),
        'data_signatures': signatures, 'epoch_budget': budget,
        'resume_from': None if resume_from is None else str(resume_from),
        'torch_version': str(torch.__version__), 'device': str(device), 'gpu_name': gpu_name,
        'resume_device': resume_device, **manifest_policy, **provenance,
        'status': 'STARTED',
    })
    epoch_timing = []
    for epoch in range(start, budget + 1):
        started = perf_counter()
        train_metrics = _epoch(model, train_rows, config, optimizer)
        train_seconds = perf_counter() - started
        step += train_metrics['supervised_videos']
        validation_metrics = _epoch(model, validation_rows, config)
        epoch_timing.append({'epoch': epoch, 'training_seconds': train_seconds,
                             'validation_seconds': perf_counter() - started - train_seconds})
        history.append({'epoch': epoch, 'training': train_metrics, 'validation': validation_metrics})
        improved = validation_metrics['loss'] < best_loss
        if improved:
            best_loss, stale = validation_metrics['loss'], 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        checkpoint = {
            'schema': 'fk-training-v1', 'config': config, 'model_spec': spec,
            'decision': model.decision.to_dict(), 'data_signatures': signatures,
            'epoch': epoch, 'step': step, 'history': history, 'best_loss': best_loss,
            'best_model_state': best_state, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': None,
            'stale_epochs': stale, 'rng': _rng_state(device),
            'device': str(device), 'gpu_name': gpu_name, **manifest_policy, **provenance,
        }
        _save_checkpoint(_fk_paths.relocate((output / 'last.pt'), __file__), checkpoint)
        if improved:
            _save_checkpoint(_fk_paths.relocate((output / 'best.pt'), __file__), checkpoint)
        if stale >= config['patience']:
            break
    # Resume may not improve the inherited best; always publish its best state.
    checkpoint['model_state'] = best_state
    checkpoint['selected_epoch'] = min(history, key=lambda item: item['validation']['loss'])['epoch']
    checkpoint.pop('optimizer_state')  # Best is inference-only; resume uses last.pt.
    _save_checkpoint(_fk_paths.relocate((output / 'best.pt'), __file__), checkpoint)
    model.load_state_dict(best_state)
    model.eval()
    result = {'status': 'FINISHED', 'best_validation_loss': best_loss,
              'selected_epoch': checkpoint['selected_epoch'], 'epochs_completed': len(history), 'history': history}
    save(_fk_paths.relocate((output / 'metrics.json'), __file__), result)
    save(_fk_paths.relocate((output / 'timing.json'), __file__), {'device': str(device), 'epochs': epoch_timing})
    return result


def load_fk_checkpoint(path, *, device='cpu'):
    checkpoint = torch.load(_fk_paths.relocate_value((path), __file__), map_location='cpu', weights_only=True)
    if checkpoint['schema'] != 'fk-training-v1':
        raise ValueError('unsupported FK checkpoint')
    spec = dict(checkpoint['model_spec'])
    if 'model_kind' in spec:
        from .shared_baselines import SharedBaseline
        model = SharedBaseline(**spec, decision=Decision(**checkpoint['decision'])).to(device)
    else:
        model = EvidenceModel(**spec, decision=Decision(**checkpoint['decision'])).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    return model.eval(), checkpoint


def select_video_threshold(model, validation_rows):
    """Select balanced accuracy on validation only; never inspect test records."""
    _population_signature(validation_rows, 'validation')
    rows = []
    model.eval()
    with torch.inference_mode():
        for row in validation_rows:
            _, score, _ = forward_video(model, row)
            if score is not None:
                rows.append((score.item(), row['label']))
    if {label for _, label in rows} != {0, 1}:
        raise ValueError('threshold selection requires assessed validation examples of both classes')
    candidates = {0.0, .5, 1.0, *(score for score, _ in rows)}
    best = None
    for threshold in sorted(candidates):
        positives = sum(label == 1 for _, label in rows)
        negatives = len(rows) - positives
        sensitivity = sum(score >= threshold and label == 1 for score, label in rows) / positives
        specificity = sum(score < threshold and label == 0 for score, label in rows) / negatives
        candidate = ((sensitivity + specificity) / 2, -abs(threshold - .5), -threshold)
        if best is None or candidate > best[0]:
            best = (candidate, threshold)
    return {'video_threshold': best[1], 'balanced_accuracy': best[0][0],
            'split': 'validation', 'assessed_videos': len(rows)}
