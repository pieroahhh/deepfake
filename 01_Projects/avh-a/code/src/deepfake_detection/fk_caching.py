"""Build FK training caches and retain per-window assessment and annotation provenance."""

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


from collections import Counter
import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from time import perf_counter

import torch

from .cli import load_json, save
from .commands import fk_asset_hashes, load_fk_assets
from .core import overlap, supervision
from .data import audit
from .feature_cache import read_cache, write_cache
from .forensics import canonical, sha256_file
from .infrastructure import fk_device
from .provenance import git_provenance
from .preparation import (FK_QUALITY_PROFILE, IN_FRAME_CROP_POLICY, encode_fk_window,
                          fk_availability_quality, fk_source_speech, prepare_fk_window)


WINDOW_LABEL_POLICY = {
    'interval_convention': 'half-open source seconds',
    'fake_rule': 'positive-duration intersection with a published manipulation interval',
    'minimum_overlap_seconds_exclusive': 0,
    'complete_annotation_with_no_overlap': 0,
    'unknown_modality_with_combined_overlap': None,
    'unusable_windows_supply_training_targets': False,
}


def _modality_target(periods, window, combined_target):
    if periods is None:
        return 0 if combined_target == 0 else None
    return int(any(overlap(window, period) > 0 for period in periods))


def _read_completed_clip(directory, row, relative_cache, provenance):
    completed = load_json(_fk_paths.relocate((directory / 'completed.json'), __file__))
    assessment = completed['assessment']
    usable = [window['interval'] for window in assessment['windows'] if window['status'] == 'USABLE']
    if assessment['video_id'] != row['video_id'] or assessment['usable_windows'] != len(usable):
        raise ValueError('completed clip does not match the input row')
    entry = completed['cache_entry']
    if usable:
        expected = {**provenance, 'window_intervals': usable}
        if (entry['cache'] != relative_cache or entry['cache_provenance'] != expected
                or entry['window_intervals'] != usable or any(entry.get(key) != value for key, value in row.items())
                or sha256_file(_fk_paths.relocate((directory / 'metadata.json'), __file__)) != entry['cache_metadata_sha256']):
            raise ValueError('completed cache metadata does not match')
        read_cache(directory, expected)
    elif entry is not None or assessment['exclusion_reason'] != 'no_usable_windows':
        raise ValueError('completed exclusion does not match the assessment')
    return completed


def cache_fk_manifest(manifest_path, dataset_root, assets_path, output, *, device='auto', resume=False, quality_normalization=None):
    """Cache the named study population using its recorded labels and fixed crop policy."""
    rows = load_json(manifest_path)
    population_audit = audit(rows)
    root = _fk_paths.relocate((Path(dataset_root)), __file__).resolve(strict=True)
    assets = load_json(assets_path)
    device = fk_device(device)
    encoders, detector, vad = load_fk_assets(assets)
    encoders.to(device)
    policy = {
        'face_crop_policy': IN_FRAME_CROP_POLICY, 'quality_profile': FK_QUALITY_PROFILE,
        'window_seconds': 2, 'stride_seconds': 1, 'detector': detector.method,
        'detector_sha256': assets['detector']['sha256'], 'vad_sha256': assets['silero']['sha256'],
        'torch_version': str(torch.__version__), 'encoder_device': str(device),
    }
    policy_digest = hashlib.sha256(canonical(policy)).hexdigest()
    asset_hashes = fk_asset_hashes(assets, encoders)
    if quality_normalization is not None:
        from .quality import QUALITY_V2, quality_tensor, profile_hash
        normalization = load_json(quality_normalization)
        if normalization.get('asset_sha256') != asset_hashes:
            raise ValueError('quality fitting assets differ from caching assets')
        population = [{'video_id':r['video_id'], 'source_sha256':r['source_sha256']} for r in rows if r['split']=='train']
        if normalization.get('training_population_sha256') != profile_hash(population):
            raise ValueError('quality fitting training population differs from caching manifest')
        policy.update(quality_profile=QUALITY_V2, quality_normalization=normalization,
                      quality_normalization_sha256=profile_hash(normalization), audio_pcm_policy='pcm16-roundtrip-v1')
        policy_digest = hashlib.sha256(canonical(policy)).hexdigest()
    encoder_hashes = {name: asset_hashes[name]['model.safetensors'] for name in ('visual', 'audio')}
    run_identity = {
        'source_overlap_policy':population_audit['source_overlap_policy'],
        'population_audit':population_audit,
        'schema': 'fk-cache-manifest-v2' if quality_normalization else 'fk-cache-manifest-v1', 'source_manifest_sha256': sha256_file(manifest_path),
        'preprocessing': policy, 'window_label_policy': WINDOW_LABEL_POLICY,
        'asset_sha256': asset_hashes, **git_provenance(),
    }
    manifest = {
        **run_identity,
        'train': [], 'validation': [], 'test': [], 'assessments': [],
    }
    output = _fk_paths.relocate((Path(output)), __file__).resolve()
    if resume:
        if load_json(_fk_paths.relocate((output / 'cache-run.json'), __file__)) != run_identity:
            raise ValueError('resume requires the same source manifest, code, assets, and preprocessing')
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(_fk_paths.relocate((output / 'cache-run.json'), __file__), run_identity)
    cache_root = _fk_paths.relocate((output / 'caches'), __file__)
    cache_root.mkdir(exist_ok=True)
    cache_root.resolve().relative_to(output)
    reused = 0
    for index, row in enumerate(rows):
        started = perf_counter()
        source = (_fk_paths.relocate((root / row['path']), __file__)).resolve(strict=True)
        source.relative_to(root)
        if sha256_file(source) != row['source_sha256']:
            raise ValueError(f"source checksum differs from manifest: {row['video_id']}")
        relative_cache = f'caches/{index:06d}'
        directory = _fk_paths.relocate((output / relative_cache), __file__)
        provenance = {
            'source_sha256': row['source_sha256'], 'preprocessing_sha256': policy_digest,
            'face_crop_policy': IN_FRAME_CROP_POLICY, 'window_label_policy': WINDOW_LABEL_POLICY,
            'asset_sha256': asset_hashes, 'encoder_sha256': encoder_hashes,
            'feature_layers': ['visual_avgpool', 'audio_last_hidden'], 'precision': 'float32',
        }
        if resume and directory.exists():
            directory.resolve().relative_to(cache_root.resolve())
            try:
                completed = _read_completed_clip(directory, row, relative_cache, provenance)
            except (OSError, ValueError, KeyError, TypeError):
                quarantine = _fk_paths.relocate((Path(mkdtemp(prefix='.invalid-', dir=cache_root))), __file__).resolve()
                quarantine.relative_to(cache_root.resolve())
                directory.rename(_fk_paths.relocate((quarantine / 'clip'), __file__))
            else:
                if completed['cache_entry'] is not None:
                    manifest[row['split']].append(completed['cache_entry'])
                manifest['assessments'].append(completed['assessment'])
                reused += 1
                continue
        duration = row['duration']
        try:
            speech = fk_source_speech(source, duration, vad, pcm16=quality_normalization is not None)
            speech_error = None
        except ValueError as error:
            speech, speech_error = [], str(error)

        intervals = [[float(start), float(start + 2)] for start in range(math.ceil(duration))]
        targets = supervision(row['label'], row['temporal_complete'], row['manipulated_intervals'], intervals)
        assessments, usable, features = [], [], []
        for window, labels in zip(intervals, targets['windows']):
            prepared = prepare_fk_window(source, window[0], duration, detector, speech, fit_to_frame=True,
                                         pcm16=quality_normalization is not None)
            if prepared['source_sha256'] != row['source_sha256'] or 'source_changed_during_preparation' in prepared['reasons']:
                raise ValueError(f"source changed during caching: {row['video_id']}")
            assessment = {key: value for key, value in prepared.items() if key not in ('faces', 'lips', 'audio', 'visual_times')}
            assessment['labels'] = {
                'combined': labels['window'],
                'audio': _modality_target(row.get('audio_manipulated_intervals'), window, labels['window']),
                'visual': _modality_target(row.get('visual_manipulated_intervals'), window, labels['window']),
            }
            assessments.append(assessment)
            if prepared['status'] == 'USABLE':
                encoded = encode_fk_window(prepared, encoders)
                encoded['quality'] = (quality_tensor(prepared['quality_measurements'], normalization)
                                      if quality_normalization else fk_availability_quality(prepared['quality_measurements']))
                features.append(encoded)
                usable.append(window)

        assessment = {
            'video_id': row['video_id'], 'split': row['split'], 'source_family': row['source_family'],
            'duration': row['duration'], 'manipulated_intervals': row['manipulated_intervals'],
            'temporal_complete': row['temporal_complete'],
            'label': row['label'], 'audio_label': row['audio_label'], 'visual_label': row['visual_label'],
            'speech_intervals': speech, 'speech_detection_error': speech_error,
            'windows': assessments, 'usable_windows': len(usable),
            'exclusion_reason': None if features else 'no_usable_windows',
            'processing_seconds': perf_counter() - started,
        }
        entry = None
        with TemporaryDirectory(prefix='.staging-', dir=cache_root) as temporary:
            staged = _fk_paths.relocate((Path(temporary) / 'clip'), __file__)
            if features:
                tensors = {
                    key: torch.stack([item[key] for item in features]) if key.endswith('_times')
                    else torch.cat([item[key] for item in features])
                    for key in features[0]
                }
                provenance['window_intervals'] = usable
                write_cache(staged, tensors, provenance)
                entry = {
                    **row, 'window_intervals': usable, 'cache': relative_cache, 'cache_provenance': provenance,
                    'cache_metadata_sha256': sha256_file(_fk_paths.relocate((staged / 'metadata.json'), __file__)),
                }
            save(_fk_paths.relocate((staged / 'completed.json'), __file__), {'cache_entry': entry, 'assessment': assessment})
            staged.rename(directory)
        if entry is not None:
            manifest[row['split']].append(entry)
        manifest['assessments'].append(assessment)
    with TemporaryDirectory(prefix='.manifest-', dir=output) as temporary:
        staged = _fk_paths.relocate((Path(temporary) / 'training-manifest.json'), __file__)
        save(staged, manifest)
        staged.replace(_fk_paths.relocate((output / 'training-manifest.json'), __file__))
    return {
        'manifest': str(_fk_paths.relocate((output / 'training-manifest.json'), __file__)), 'device': str(device),
        'attempted_clips': len(rows), 'reused_clips': reused,
        'cached_clips': {split: len(manifest[split]) for split in ('train', 'validation', 'test')},
        'usable_windows': sum(item['usable_windows'] for item in manifest['assessments']),
        'unusable_window_reasons': dict(Counter(reason for item in manifest['assessments']
                                                for window in item['windows'] for reason in window['reasons'])),
    }
