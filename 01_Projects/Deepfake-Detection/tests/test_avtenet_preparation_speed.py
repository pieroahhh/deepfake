"""Exact-equivalence checks for pure detection reuse and tensor packing."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from baselines.common.detection_cache import FrameDetectionCache
from baselines.avtenet.hf_vivit import HFVivitBackend
from deepfake_detection.faces import track_and_crop, MTCNNDetector


class Detector:
    method = 'fixture'
    def __init__(self):self.calls = 0
    def __call__(self, image):
        self.calls += 1
        marker = int(image[0, 0, 0])
        if marker == 0:return []
        if marker == 9:raise ValueError('decode fixture failure')
        detection = {'box':[16., 16., 48., 48.], 'score':.99,
                     'landmarks':[[25., 25.], [39., 25.], [32., 33.], [27., 40.], [37., 40.]]}
        if marker == 2:return [detection, copy.deepcopy(detection)]
        if marker == 3:
            detection['box'] = [0., 0., 8., 8.]
        return [detection]


class DetectionReuseTests(unittest.TestCase):
    def test_exact_bytes_shape_eviction_mutation_and_exceptions(self):
        source = Detector()
        cache = FrameDetectionCache(source, maximum_frames=2)
        a = np.full((64, 64, 3), 1, dtype=np.uint8)
        result = cache(a)
        result[0]['box'][0] = -999
        self.assertEqual(cache(a.copy())[0]['box'][0], 16.)
        self.assertEqual(source.calls, 1)
        # Shape and one changed byte must make a different key.
        cache(a.reshape(32, 128, 3))
        b = a.copy(); b[1, 1, 1] = 2
        cache(b)
        cache(a)
        self.assertEqual(source.calls, 4)
        self.assertEqual(len(cache.entries), 2)
        cache.clear(); cache(a)
        self.assertEqual(source.calls, 5)
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                cache(np.full_like(a, 9))
        self.assertEqual(source.calls, 7)
        with self.assertRaises(ValueError):cache(a.astype(np.float32))

    def test_crops_tracking_and_failure_outcomes_are_unchanged(self):
        for pattern in ((1, 1, 1, 1), (2, 1, 0, 1), (1, 3, 1, 0)):
            frames = [np.full((64, 64, 3), value, dtype=np.uint8) for value in pattern]
            reference = track_and_crop(frames, Detector(), fit_to_frame=True)
            detector = Detector()
            optimized = track_and_crop(frames, FrameDetectionCache(detector), fit_to_frame=True)
            for left, right in zip(reference, optimized):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    if isinstance(left[key], np.ndarray):
                        np.testing.assert_array_equal(left[key], right[key])
                    else:self.assertEqual(left[key], right[key])
            self.assertEqual(detector.calls, len(set(pattern)))

    def test_prepared_resets_cache_between_clips_and_rejects_other_detectors(self):
        from baselines.avtenet.pipeline import Prepared
        class Population:
            def rows(self, split):
                for i in range(2):yield {'video_id':str(i), 'duration':2., 'visual_label':0}
            def path(self, row):return row['video_id']
        class Backend:
            def prepare(self, frames):return frames
        class Native:
            def __init__(self):self.calls = 0
            def detect(self, rgb, landmarks):
                self.calls += 1
                return None, None, None
        class Fixture(Prepared):
            def raw_inputs(self, path, start):
                self.detect(np.zeros((4, 4, 3), dtype=np.uint8))
                self.detect(np.zeros((4, 4, 3), dtype=np.uint8))
                return {'faces':np.zeros((1, 4, 4, 3), dtype=np.uint8)}
        with self.assertRaisesRegex(ValueError, 'MTCNN'):
            Fixture(Population(), 'visual', Backend(), {}, Detector(), cache_face_detections=True)
        native = Native()
        prepared = Fixture(Population(), 'visual', Backend(), {}, MTCNNDetector(native), cache_face_detections=True)
        list(prepared.records('train', supervised=True))
        self.assertEqual(native.calls, 2)
        self.assertEqual(len(prepared.detection_cache.entries), 0)


class TensorPackingTests(unittest.TestCase):
    def backend(self, processor):
        class Model(torch.nn.Module):
            config = SimpleNamespace(hidden_size=768)
        return HFVivitBackend(Model(), processor, efficient=False)

    def test_default_processor_full_window_is_bit_exact(self):
        from transformers import VivitImageProcessor
        torch.set_num_threads(1)
        frames = np.random.default_rng(7).integers(0, 256, (32, 224, 224, 3), dtype=np.uint8)
        backend = self.backend(VivitImageProcessor())
        left = backend.prepare(frames)
        right = backend.prepare(frames, fast=True)
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(tuple(left.shape), (1, 32, 3, 224, 224))
        self.assertTrue(torch.equal(left, right))
        # No views onto mutable source images; packing owns its processed pixels.
        saved = right.clone()
        frames.fill(0)
        self.assertTrue(torch.equal(saved, right))

    def test_custom_processor_configs_and_noncontiguous_frames_are_exact(self):
        from transformers import VivitImageProcessor
        frames = np.random.default_rng(11).integers(0, 256, (4, 48, 64, 3), dtype=np.uint8)[:, :, ::-1]
        for configuration in (
            {'size':{'shortest_edge':32}, 'crop_size':{'height':24, 'width':24}},
            {'do_resize':False, 'do_center_crop':False, 'do_normalize':False, 'offset':False},
            {'size':{'shortest_edge':32}, 'crop_size':{'height':24, 'width':24},
             'image_mean':[.1,.2,.3], 'image_std':[.8,.7,.6]},
        ):
            backend = self.backend(VivitImageProcessor(**configuration))
            self.assertTrue(torch.equal(backend.prepare(frames), backend.prepare(frames, fast=True)))


if __name__ == '__main__':unittest.main()
