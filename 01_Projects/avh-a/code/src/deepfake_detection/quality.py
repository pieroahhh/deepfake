"""Measured quality summaries; normalization is fitted on training clips only."""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .forensics import canonical, sha256_file

QUALITY_V2 = 'measured-quality-v2'


def profile_hash(profile):
    return hashlib.sha256(canonical(profile)).hexdigest()


def fit_profile(rows):
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('quality normalization requires training rows only')
    measures = [w['quality_measurements'] for row in rows for w in row['windows']
                if w['status'] == 'USABLE']
    if not measures:
        raise ValueError('no usable training windows for quality normalization')
    bounds = {}
    for key in ('mean_face_laplacian_variance', 'estimated_snr_db'):
        values = [m[key] for m in measures if m.get(key) is not None]
        if any(not math.isfinite(v) for v in values):
            raise ValueError('nonfinite quality measurement')
        bounds[key] = None if not values else np.percentile(values, [5, 95]).tolist()
    return {'schema': QUALITY_V2, 'fit_split': 'train', 'bounds': bounds,
            'training_population_sha256': profile_hash([
                {'video_id': r['video_id'], 'source_sha256': r['source_sha256']} for r in rows]),
            'usable_windows': len(measures), 'missing_or_constant_policy': 'neutral_0.5',
            'snr_method': 'speech_power_minus_nonspeech_power_v1'}


def quality_tensor(measurements, profile):
    if profile.get('schema') != QUALITY_V2 or profile.get('fit_split') != 'train':
        raise ValueError('unsupported quality normalization profile')

    def normalized(key):
        value, limits = measurements.get(key), profile['bounds'][key]
        if value is None or limits is None or limits[0] == limits[1]:
            return .5
        if not math.isfinite(value) or not all(math.isfinite(x) for x in limits) or limits[1] < limits[0]:
            raise ValueError('invalid quality bounds/measurement')
        return float(np.clip((value-limits[0])/(limits[1]-limits[0]), 0, 1))

    keys = ('face_coverage', 'mean_detection_confidence', 'mean_tracking_iou',
            'speech_coverage', 'audio_clipping_fraction')
    if any(not math.isfinite(measurements[k]) or not 0 <= measurements[k] <= 1 for k in keys):
        raise ValueError('quality fractions must be finite in [0,1]')
    visual = (measurements['face_coverage'] + measurements['mean_detection_confidence']
              + measurements['mean_tracking_iou'] + normalized('mean_face_laplacian_variance')) / 4
    audio = (measurements['speech_coverage'] + 1-measurements['audio_clipping_fraction']
             + normalized('estimated_snr_db')) / 3
    return torch.tensor([[visual, audio]], dtype=torch.float32)


def fit_quality_manifest(manifest_path, dataset_root, assets_path, output, *, resume=False, progress_seconds=10):
    """Fit exact training percentiles with durable per-window progress."""
    from .quality_fitting import fit_quality_manifest as run
    return run(manifest_path, dataset_root, assets_path, output,
               resume=resume, progress_seconds=progress_seconds)
