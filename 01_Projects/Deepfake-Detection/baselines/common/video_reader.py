"""Bounded sequential decoding with the reference sampler's frame selection.

One reader belongs to one preparation iterator. No seeking, resampling changes,
or cross-window face tracking are introduced. Backwards requests reopen the
stream; forward/overlapping requests reuse already decoded frames.
"""
import math
import numpy as np
import av

from deepfake_detection.media import _origin, _window


class VideoWindowReader:
    def __init__(self):
        self.close()

    def close(self):
        if getattr(self, 'container', None) is not None:
            self.container.close()
        self.container = None
        self.path = None
        self.times = []
        self.images = []
        self.previous = -math.inf
        self.last_start = -math.inf
        self.exhausted = False

    def sample(self, path, start, duration, *, fps=25, maximum_time_error=.06):
        _window(start, duration)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(maximum_time_error) or maximum_time_error < 0:
            raise ValueError('invalid sampling rate or timestamp tolerance')
        count = round(duration * fps)
        if count < 1 or abs(count - duration * fps) > 1e-6:
            raise ValueError('duration*fps must be a positive whole frame count')
        try:
            return self._sample(str(path), start, count, fps, maximum_time_error)
        except Exception:
            # A failed request must not hide a malformed earlier timestamp on
            # the next window. Reopen and validate from the beginning again.
            self.close()
            raise

    def _sample(self, path, start, count, fps, tolerance):
        lower = start - tolerance
        # Retained RGB frames may not cover a backwards request / wider tolerance.
        if (self.path != path or start < self.last_start or
                (self.times and lower < self.retained_lower)):
            self.close()
        if self.container is None:
            self.container = av.open(path)
            self.path = path
            if not self.container.streams.video:
                raise ValueError('missing video stream')
            stream = self.container.streams.video[0]
            self.stream_index = stream.index
            self.origin = _origin(self.container)
            self.decoder = iter(self.container.decode(stream))
        self.last_start = start
        targets = start + np.arange(count, dtype=np.float64) / fps
        upper = targets[-1] + tolerance
        # Drop old RGB arrays before continuing. Memory is bounded by the
        # requested interval, rather than the duration of the source clip.
        keep = next((i for i, t in enumerate(self.times) if t >= lower), len(self.times))
        self.times = self.times[keep:]
        self.images = self.images[keep:]
        self.retained_lower = lower
        while not self.exhausted and self.previous <= upper:
            frame = next(self.decoder, None)
            if frame is None:
                self.exhausted = True
                break
            if frame.pts is None:
                raise ValueError('video frame lacks a source timestamp')
            timestamp = float(frame.pts * frame.time_base) - self.origin
            if timestamp <= self.previous:
                raise ValueError('video timestamps must increase strictly')
            self.previous = timestamp
            if timestamp >= lower:
                self.times.append(timestamp)
                self.images.append(frame.to_ndarray(format='rgb24'))
        # Match the reference's inclusive first frame AFTER the interval end.
        end = next((i + 1 for i, t in enumerate(self.times) if t > upper), len(self.times))
        times = np.asarray(self.times[:end])
        if not len(times):
            raise ValueError('no video frames in requested source interval')
        right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
        errors = np.abs(times[indices] - targets)
        if (errors > tolerance + 1e-9).any():
            raise ValueError('source-frame gap exceeds declared sampling tolerance')
        return {'rgb': np.stack([self.images[i] for i in indices]),
                'target_times': targets, 'source_times': times[indices],
                'duplicate_samples': count - len(set(indices.tolist())),
                'maximum_time_error': float(errors.max()), 'stream_index': self.stream_index,
                'source_origin_seconds': self.origin}
