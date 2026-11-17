"""Pinned local Silero JIT adapter and declared source-time speech interval policy."""

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


import math

import numpy as np
import torch

from .core import union
from .forensics import sha256_file


def speech_intervals(probabilities, hop_seconds, duration, *, threshold=.5,
                     minimum_speech=.25, merge_gap=.3, padding=.1):
    values = np.asarray(probabilities, dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()
            or not math.isfinite(duration) or duration <= 0 or not math.isfinite(hop_seconds) or hop_seconds <= 0
            or not 0 <= threshold <= 1 or min(minimum_speech, merge_gap, padding) < 0):
        raise ValueError('invalid VAD probabilities, timing or interval policy')
    if len(values) != math.ceil(duration / hop_seconds):
        raise ValueError('VAD probability count must cover exactly the source duration')
    runs, start = [], None
    for index, probability in enumerate([*values, 0.]):
        voiced = probability >= threshold and index < len(values)
        if voiced and start is None:
            start = index * hop_seconds
        if not voiced and start is not None:
            end = min(index * hop_seconds, duration)
            if end - start + 1e-12 >= minimum_speech:
                runs.append((start, end))
            start = None
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return union((max(0., start-padding), min(duration, end+padding)) for start, end in merged)


class SileroVAD:
    """No hub download. Use a checksummed 16kHz/512-sample compatible JIT model."""

    def __init__(self, path, sha256):
        if sha256_file(path) != sha256:
            raise ValueError('Silero model checksum mismatch')
        self.model = torch.jit.load(_fk_paths.relocate_value((str(path)), __file__), map_location='cpu').eval()
        if not callable(getattr(self.model, 'reset_states', None)):
            raise ValueError('incompatible Silero JIT state interface')

    @torch.inference_mode()
    def __call__(self, waveform):
        samples = torch.as_tensor(waveform, dtype=torch.float32)
        if samples.ndim != 1 or samples.numel() == 0 or not torch.isfinite(samples).all():
            raise ValueError('VAD requires finite nonempty mono 16kHz PCM')
        if samples.abs().max() > 1:
            raise ValueError('VAD PCM must be normalized to [-1,1]')
        self.model.reset_states()
        probabilities = []
        for chunk in samples.split(512):
            if len(chunk) < 512:
                chunk = torch.nn.functional.pad(chunk, (0, 512-len(chunk)))
            probabilities.append(self.model(chunk, 16000).item())
        return speech_intervals(probabilities, 512/16000, len(samples)/16000)
