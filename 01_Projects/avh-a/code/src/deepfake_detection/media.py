"""Bounded-window decoding on a shared source timeline, with explicit coverage.

Subtract the container start once for both modalities; never shift each stream
independently to make it appear synchronized. No incomplete audio is silently
accepted as fully assessed.
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


from pathlib import Path
import json
import math
import shutil
import subprocess
import sys

import av
import numpy as np


def _origin(container):
    return 0.0 if container.start_time is None else container.start_time / av.time_base


def _window(start, duration):
    if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
        raise ValueError('window requires finite start >=0 and duration >0')


def media_tool(name):
    if name not in ('ffmpeg', 'ffprobe'):
        raise ValueError('unsupported media tool')
    executable = shutil.which(name)
    if executable is None:
        candidate = _fk_paths.relocate((Path(sys.executable).parent / (name+'.exe' if sys.platform=='win32' else name)), __file__)
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        raise RuntimeError(name+' is unavailable; activate the prepared environment')
    return _fk_paths.relocate((Path(executable)), __file__)


def probe(path):
    executable = media_tool('ffprobe')
    result = subprocess.run(
        [executable, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return json.loads(result.stdout)


def source_metadata(path):
    """Read actual stream metadata with PyAV; record the inspection backend."""
    from datetime import datetime, timezone
    with av.open(str(path)) as container:
        streams = [{'index': s.index, 'type': s.type, 'codec': s.codec_context.name,
                    'start_seconds': None if s.start_time is None else float(s.start_time*s.time_base),
                    'time_base': str(s.time_base),
                    'sample_rate': getattr(s.codec_context, 'sample_rate', None),
                    'average_rate': str(getattr(s, 'average_rate', None))} for s in container.streams]
        video = next((s for s in streams if s['type']=='video'), None)
        audio = next((s for s in streams if s['type']=='audio'), None)
        offset = (audio['start_seconds']-video['start_seconds'] if audio and video
                  and audio['start_seconds'] is not None and video['start_seconds'] is not None else None)
        if container.duration is None:
            raise ValueError('source duration unavailable')
        variable = None
        if container.streams.video:
            times = [float(f.pts*f.time_base) for f in container.decode(container.streams.video[0]) if f.pts is not None]
            if len(times)>2:
                deltas = np.diff(times)
                variable = bool(np.max(deltas)-np.min(deltas)>max(1e-6,float(container.streams.video[0].time_base)*1.1))
        return {'filename': _fk_paths.relocate((Path(path)), __file__).name, 'bytes': _fk_paths.relocate((Path(path)), __file__).stat().st_size,
                'duration': container.duration/av.time_base, 'streams': streams,
                'av_start_offset_seconds': offset, 'backend': 'PyAV/libavformat',
                'ingestion_time_utc': datetime.now(timezone.utc).isoformat(),
                'acquisition_time': None, 'source_context': None,
                'limitations': {'missing_audio': audio is None,
                               'multiple_audio_tracks': sum(s['type']=='audio' for s in streams)>1,
                               'variable_frame_rate': variable}}


def sample_video(path, start, duration, *, fps=25, maximum_time_error=.06):
    """Nearest source-PTS frames, rejecting large gaps; record duplicates explicitly.

    Decodes only up to the requested interval end, retaining nearby frames, but
    currently scans from the stream start for correctness across codecs/keyframes.
    """
    _window(start, duration)
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(maximum_time_error) or maximum_time_error < 0:
        raise ValueError('invalid sampling rate or timestamp tolerance')
    count = round(duration * fps)
    if count < 1 or abs(count - duration * fps) > 1e-6:
        raise ValueError('duration*fps must be a positive whole frame count')
    targets = start + np.arange(count, dtype=np.float64) / fps
    times, images = [], []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('missing video stream')
        stream = container.streams.video[0]
        stream_index = stream.index
        origin = _origin(container)
        previous = -math.inf
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('video frame lacks a source timestamp')
            timestamp = float(frame.pts * frame.time_base) - origin
            if timestamp <= previous:
                raise ValueError('video timestamps must increase strictly')
            previous = timestamp
            if timestamp >= start - maximum_time_error:
                times.append(timestamp)
                images.append(frame.to_ndarray(format='rgb24'))
            if timestamp > targets[-1] + maximum_time_error:
                break
    if not times:
        raise ValueError('no video frames in requested source interval')
    times = np.asarray(times)
    right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
    errors = np.abs(times[indices] - targets)
    if (errors > maximum_time_error + 1e-9).any():
        raise ValueError('source-frame gap exceeds declared sampling tolerance')
    return {
        'rgb': np.stack([images[i] for i in indices]),
        'target_times': targets, 'source_times': times[indices],
        'duplicate_samples': count - len(set(indices.tolist())),
        'maximum_time_error': float(errors.max()), 'stream_index': stream_index,
        'source_origin_seconds': origin,
    }


def _mono_float(frame):
    values = frame.to_ndarray()
    channels = len(frame.layout.channels)
    if not frame.format.is_planar:
        values = values.reshape(-1, channels).T
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        if info.min == 0:
            midpoint = (info.max + 1) / 2
            values = (values.astype(np.float32) - midpoint) / midpoint
        else:
            values = values.astype(np.float32) / max(abs(info.min), abs(info.max))
    return values.astype(np.float32).mean(axis=0, keepdims=True)


def aligned_audio(path, start, duration, *, sample_rate=16000, require_complete=True):
    """Arithmetic-mean mono, resample, retain timestamps and a per-sample mask."""
    _window(start, duration)
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('sample rate must be a positive integer')
    count = round(duration * sample_rate)
    if count < 1:
        raise ValueError('requested audio interval is shorter than one output sample')
    samples = np.zeros(count, dtype=np.float32)
    present = np.zeros(count, dtype=bool)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError('missing audio stream')
        stream = container.streams.audio[0]
        stream_index = stream.index
        origin = _origin(container)
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=sample_rate)

        def insert(output):
            if output.pts is None:
                raise ValueError('resampled audio lacks a source timestamp')
            timestamp = float(output.pts * output.time_base) - origin
            values = output.to_ndarray().reshape(-1)
            offset = round((timestamp - start) * sample_rate)
            begin, end = max(0, offset), min(count, offset + len(values))
            if end > begin:
                if present[begin:end].any():
                    raise ValueError('overlapping audio timestamps require an explicit repair policy')
                samples[begin:end] = values[begin - offset:end - offset]
                present[begin:end] = True

        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('audio frame lacks a source timestamp')
            timestamp = float(frame.pts * frame.time_base) - origin
            if timestamp > start + duration + .1:
                break
            mono = av.AudioFrame.from_ndarray(_mono_float(frame), format='fltp', layout='mono')
            mono.sample_rate, mono.pts, mono.time_base = frame.sample_rate, frame.pts, frame.time_base
            for output in resampler.resample(mono):
                insert(output)
        for output in resampler.resample(None):
            insert(output)
    if require_complete and not present.all():
        raise ValueError('audio does not fully cover the requested source-time interval')
    return {'samples': samples, 'present': present, 'sample_rate': sample_rate,
            'coverage': float(present.mean()), 'source_origin_seconds': origin, 'stream_index': stream_index}
