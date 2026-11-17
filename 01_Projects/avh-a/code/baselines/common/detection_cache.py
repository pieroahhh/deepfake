"""Bounded reuse of pure per-frame MTCNN results, never tracking state."""
from collections import OrderedDict
import copy
import hashlib
import numpy as np


class FrameDetectionCache:
    """For deterministic, stateless detectors only; reset between source clips.

    Keys bind exact RGB bytes/shape/dtype. Only small detection metadata is kept,
    not source images. Tracking/cropping still runs in original frame order.
    """
    def __init__(self, detector, maximum_frames=128):
        if not 1 <= maximum_frames <= 1024:
            raise ValueError('detection cache must hold between 1 and 1024 frames')
        self.detector = detector
        self.method = detector.method
        self.maximum_frames = maximum_frames
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def clear(self):
        self.entries.clear()

    def __call__(self, rgb):
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('detection cache requires uint8 HWC RGB frames')
        image = np.ascontiguousarray(image)
        key = (image.shape, image.dtype.str, hashlib.sha256(memoryview(image).cast('B')).digest())
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return copy.deepcopy(self.entries[key])
        # Exceptions are never cached, and callers cannot mutate a stored result.
        result = self.detector(rgb)
        self.misses += 1
        self.entries[key] = copy.deepcopy(result)
        if len(self.entries) > self.maximum_frames:
            self.entries.popitem(last=False)
        return result
