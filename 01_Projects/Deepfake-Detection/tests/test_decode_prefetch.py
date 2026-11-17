"""CPU-only parity, error ordering, bounded lookahead and thread ownership."""

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

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from baselines.common.decode_prefetch import decoded_chunks
from baselines.common.media import faces
from baselines.common.video_reader import VideoWindowReader
from test_avtenet_io_optimization import video


class DecodePrefetchTests(unittest.TestCase):
    def assert_clean(self):
        self.assertFalse(any(t.name.startswith('avtenet-decode') for t in threading.enumerate()))

    def test_overlap_is_real_but_lookahead_and_threads_are_bounded(self):
        second_started = threading.Event()
        calls = []
        caller = threading.get_ident()
        def sample(path, start, duration, *, fps):
            calls.append((start, threading.get_ident()))
            if start == 1:
                second_started.set()
            return start
        stream = decoded_chunks(sample, 'fixture', ((i, 1) for i in range(100)), fps=16, enabled=True)
        try:
            self.assertEqual(next(stream), 0)
            # While the consumer still owns chunk zero, chunk one can decode.
            self.assertTrue(second_started.wait(2))
            self.assertEqual([c[0] for c in calls], [0, 1])
            self.assertEqual(len({c[1] for c in calls}), 1)
            self.assertNotEqual(calls[0][1], caller)
        finally:
            stream.close()
        self.assert_clean()

    def test_default_has_no_worker_and_preserves_lazy_order(self):
        caller = threading.get_ident()
        calls = []
        def sample(path, start, duration, *, fps):
            self.assertEqual(threading.get_ident(), caller)
            calls.append(start)
            return start
        stream = decoded_chunks(sample, 'fixture', [(0, 1), (1, 1)], fps=25)
        self.assertEqual(next(stream), 0)
        self.assertEqual(calls, [0])
        stream.close()
        self.assert_clean()

    def test_decode_failure_is_delivered_in_order_without_extra_requests(self):
        calls = []
        def sample(path, start, duration, *, fps):
            calls.append(start)
            if start == 1:
                raise ValueError('source-frame gap')
            return start
        stream = decoded_chunks(sample, 'fixture', [(0, 1), (1, 1), (2, 1)], fps=16, enabled=True)
        self.assertEqual(next(stream), 0)
        with self.assertRaisesRegex(ValueError, 'source-frame gap'):
            next(stream)
        self.assertEqual(calls, [0, 1])
        self.assert_clean()

    def test_empty_input_does_not_create_worker(self):
        self.assertEqual(list(decoded_chunks(None, None, [], fps=16, enabled=True)), [])
        self.assert_clean()

    def test_real_decode_exact_for_cfr_vfr_overlap_tail_and_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            for variable, gap in ((False, False), (True, False), (False, True)):
                path = _fk_paths.relocate((Path(folder) / f'{variable}-{gap}.mkv'), __file__)
                video(path, variable=variable, gap=gap)
                sequential, parallel = VideoWindowReader(), VideoWindowReader()
                try:
                    for start in (0., 1., 1.75, 2.75, 0., 6., 20.):
                        requests = [(start, 1.), (start+1., 1.)]
                        outcomes = []
                        for reader, enabled in ((sequential, False), (parallel, True)):
                            values = []
                            try:
                                values.extend(decoded_chunks(reader.sample, path, requests, fps=16, enabled=enabled))
                                error = None
                            except ValueError as exc:
                                error = str(exc)
                            outcomes.append((values, error))
                        self.assertEqual(outcomes[0][1], outcomes[1][1])
                        self.assertEqual(len(outcomes[0][0]), len(outcomes[1][0]))
                        for expected, actual in zip(outcomes[0][0], outcomes[1][0]):
                            self.assertEqual(expected.keys(), actual.keys())
                            for key in expected:
                                if isinstance(expected[key], np.ndarray):
                                    np.testing.assert_array_equal(expected[key], actual[key])
                                else:
                                    self.assertEqual(expected[key], actual[key])
                finally:
                    sequential.close()
                    parallel.close()
        self.assert_clean()

    def test_faces_crop_parity_and_detector_remains_on_calling_thread(self):
        caller = threading.get_ident()
        class Detector:
            method = 'deterministic-cpu-fixture'
            def __call__(self, rgb):
                if threading.get_ident() != caller:
                    raise AssertionError('Detector was moved to a worker')
                return [{'box': np.array([16.,16.,48.,48.]), 'confidence': 1.,
                         'landmarks': np.array([[24.,26.],[40.,26.],[32.,34.],[26.,40.],[38.,40.]])}]
        with tempfile.TemporaryDirectory() as folder:
            path = _fk_paths.relocate((Path(folder) / 'crop.mkv'), __file__)
            video(path)
            for fps in (16, 25):
                results = []
                for enabled in (False, True):
                    reader = VideoWindowReader()
                    try:
                        results.append(faces(path, 0, 2, Detector(), fps=fps, pixels=224,
                                             sampler=reader.sample, prefetch_decode=enabled))
                    finally:
                        reader.close()
                for key in results[0]:
                    np.testing.assert_array_equal(results[0][key], results[1][key])
                self.assertEqual(len(results[1]['faces']), 2*fps)
        self.assert_clean()

    def test_detector_exception_closes_decoder_worker(self):
        def fail(rgb):
            raise RuntimeError('detector stopped')
        with tempfile.TemporaryDirectory() as folder:
            path = _fk_paths.relocate((Path(folder) / 'stop.mkv'), __file__)
            video(path)
            reader = VideoWindowReader()
            try:
                with self.assertRaisesRegex(RuntimeError, 'detector stopped'):
                    faces(path, 0, 2, fail, sampler=reader.sample, prefetch_decode=True)
                self.assert_clean()
                # A completed speculative decode must not corrupt a later seek.
                expected = VideoWindowReader()
                try:
                    np.testing.assert_array_equal(reader.sample(path, 0, 1)['rgb'],
                                                  expected.sample(path, 0, 1)['rgb'])
                finally:
                    expected.close()
            finally:
                reader.close()


if __name__ == '__main__':
    unittest.main()
