"""Input equivalence, bounded decoding and ordered CPU-only cache prefetch."""

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

from fractions import Fraction
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch

import av
import numpy as np
import torch

from baselines.common.video_reader import VideoWindowReader
from baselines.avtenet.input_cache import InputCache, prefetch_reads
from baselines.avtenet.pipeline import Prepared
from deepfake_detection.media import sample_video


def video(path, *, variable=False, gap=False):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=25)
        stream.width = stream.height = 64
        stream.pix_fmt = 'bgr0'
        for i in range(200):
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            rgb[:, :, 0] = i % 256
            rgb[:, :, 1] = np.arange(64, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts = i + (i // 7 if variable else 0) + (25 if gap and i >= 50 else 0)
            frame.time_base = Fraction(1, 25)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


class DecoderTests(unittest.TestCase):
    def compare(self, path, requests):
        reader = VideoWindowReader()
        try:
            for start, duration, fps in requests:
                reference = sample_video(path, start, duration, fps=fps)
                result = reader.sample(path, start, duration, fps=fps)
                self.assertEqual(result.keys(), reference.keys())
                for key in result:
                    if isinstance(result[key], np.ndarray):
                        np.testing.assert_array_equal(result[key], reference[key], err_msg=key)
                    else:self.assertEqual(result[key], reference[key], key)
        finally:reader.close()

    def test_exact_cfr_vfr_overlap_backwards_and_end_aligned(self):
        with tempfile.TemporaryDirectory() as folder:
            for variable in (False, True):
                path = _fk_paths.relocate((Path(folder) / ('vfr.mkv' if variable else 'cfr.mkv')), __file__)
                video(path, variable=variable)
                self.compare(path, [(0, 1, 16), (1, 1, 16), (1, 1, 16), (2, 1, 16),
                                    (1.75, 1, 16), (2.75, 1, 16), (0, 2, 25), (5, 1, 25)])

    def test_single_open_and_bounded_retained_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _fk_paths.relocate((Path(folder) / 'sample.mkv'), __file__)
            video(path)
            real_open = av.open
            reader = VideoWindowReader()
            with patch('baselines.common.video_reader.av.open', wraps=real_open) as opened:
                for start in (0, 1, 1, 2, 2, 3, 3, 4, 4, 5):
                    reader.sample(path, start, 1, fps=16)
                    self.assertLessEqual(len(reader.times), 30)
                self.assertEqual(opened.call_count, 1)
            reader.close()

    def test_gap_eof_and_invalid_inputs_preserve_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _fk_paths.relocate((Path(folder) / 'gap.mkv'), __file__)
            video(path, gap=True)
            reader = VideoWindowReader()
            for start in (0, 1, 2, 2.5, 3, 20):
                try:reference = sample_video(path, start, 1, fps=25)
                except ValueError as error:
                    with self.assertRaisesRegex(ValueError, str(error)):
                        reader.sample(path, start, 1, fps=25)
                else:
                    np.testing.assert_array_equal(reader.sample(path, start, 1, fps=25)['rgb'], reference['rgb'])
            for start, duration, fps in ((-1, 1, 25), (0, 0, 25), (0, 1, 0), (0, .11, 25)):
                with self.assertRaises(ValueError):reader.sample(path, start, duration, fps=fps)
            reader.close()

    def test_path_switch_and_wider_tolerance(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [_fk_paths.relocate((Path(folder) / f'{i}.mkv'), __file__) for i in range(2)]
            for path in paths:video(path)
            reader = VideoWindowReader()
            try:
                for path, tolerance in ((paths[0], .01), (paths[0], .08), (paths[1], .06)):
                    expected = sample_video(path, 1, 1, fps=25, maximum_time_error=tolerance)
                    result = reader.sample(path, 1, 1, fps=25, maximum_time_error=tolerance)
                    np.testing.assert_array_equal(result['rgb'], expected['rgb'])
                    np.testing.assert_array_equal(result['source_times'], expected['source_times'])
            finally:reader.close()


class PrefetchTests(unittest.TestCase):
    def test_order_and_exceptions_are_delivered_at_the_right_item(self):
        def load(item):
            if item == 2:raise ValueError('corrupt cache')
            return item * 2
        output = []
        for item, future in prefetch_reads(range(5), load, 2):
            try:output.append((item, future.result()))
            except ValueError:output.append((item, 'error'))
        self.assertEqual(output, [(0, 0), (1, 2), (2, 'error'), (3, 6), (4, 8)])
        self.assertFalse(any(t.name.startswith('avtenet-cache') for t in threading.enumerate()))

    def test_early_close_cleans_up_and_lookahead_is_bounded(self):
        visited = []
        def source():
            for i in range(100):
                visited.append(i)
                yield i
        iterator = prefetch_reads(source(), lambda x:x, 2)
        self.assertEqual(next(iterator)[1].result(), 0)
        iterator.close()
        self.assertEqual(visited, [0, 1])
        self.assertFalse(any(t.name.startswith('avtenet-cache') for t in threading.enumerate()))

    def test_cache_still_rejects_corruption_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as folder:
            row = {'source_sha256':'a' * 64}
            cache = InputCache(folder, {'stage':'visual'})
            expected = {'faces':np.arange(24, dtype=np.uint8).reshape(2, 4, 3)}
            cache.get(row, 0., lambda:expected)
            np.testing.assert_array_equal(cache.load(row, 0.)['faces'], expected['faces'])
            header = next(p for p in cache.root.glob('*.json') if p.name != 'identity.json')
            value = json.loads(header.read_text())
            value['source_sha256'] = 'b' * 64
            header.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'source mismatch'):cache.load(row, 0.)
            value['source_sha256'] = 'a' * 64
            header.write_text(json.dumps(value))
            next(cache.root.glob('*.npz')).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):cache.load(row, 0.)

    def test_prepared_inputs_targets_order_and_rng_match_on_hits_and_misses(self):
        main_thread = threading.get_ident()
        class Backend:
            def prepare(self, frames):
                if threading.get_ident() != main_thread:raise AssertionError('background model preparation')
                return torch.from_numpy(frames.copy()).float()
        rows = [dict(video_id=str(i), source_sha256=str(i)*64, duration=3.25,
                     visual_label=0 if i < 2 else None) for i in range(3)]
        class Population:
            def rows(self, split):return iter(rows)
            def path(self, row):return row['video_id']
        class Fixture(Prepared):
            def raw_inputs(self, path, start):
                if threading.get_ident() != main_thread:raise AssertionError('background GPU provider')
                return {'faces':np.full((2, 2, 3), int(path)*10 + int(start*4), dtype=np.uint8)}
        def run(root, depth):
            cache = InputCache(root, {'stage':'visual'})
            prepared = Fixture(Population(), 'visual', Backend(), {}, None, input_cache=cache,
                               prefetch_cached_inputs=depth)
            return [(r['video_id'], start, end, values.clone(), label)
                    for r, start, end, values, label in prepared.records('train', supervised=True)]
        with tempfile.TemporaryDirectory() as folder:
            torch.set_num_threads(1)
            before = (random.getstate(), torch.get_rng_state().clone(), np.random.get_state())
            reference = run(_fk_paths.relocate((Path(folder)/'baseline'), __file__), 0)
            for result in (run(_fk_paths.relocate((Path(folder)/'prefetch'), __file__), 2), run(_fk_paths.relocate((Path(folder)/'prefetch'), __file__), 2)):
                self.assertEqual(len(result), len(reference))
                for expected, actual in zip(reference, result):
                    self.assertEqual(expected[:3], actual[:3])
                    self.assertEqual(expected[4], actual[4])
                    torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)
            self.assertEqual(random.getstate(), before[0])
            self.assertTrue(torch.equal(torch.get_rng_state(), before[1]))
            after = np.random.get_state()
            self.assertEqual(after[0], before[2][0])
            np.testing.assert_array_equal(after[1], before[2][1])
            self.assertEqual(after[2:], before[2][2:])

    def test_prefetch_preserves_cpu_training_weights_and_history(self):
        from baselines.common.training import fit, seed_all, load_checkpoint
        from baselines.common.io import read
        from baselines.avtenet.model import native_binary_loss
        schedule = {'optimizer':'Adam', 'learning_rate':.01, 'weight_decay':0.,
                    'betas':[.9,.999], 'epsilon':1e-8, 'maximum_epochs':1,
                    'patience':5, 'selection':'loss'}
        def run(root, depth):
            seed_all(7)
            model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
            def loader(item):
                return torch.full((1, 4), item / 10.), torch.tensor([item % 2])
            def batches(split):
                if depth:
                    yield from (future.result() for _, future in prefetch_reads(range(5), loader, depth))
                else:
                    yield from (loader(item) for item in range(5))
            fit(model, batches, root, identity={'model':'CPU-prefetch-check'}, schedule=schedule,
                loss_fn=lambda logits, labels:native_binary_loss(logits.squeeze(-1), labels),
                fake_predictions=lambda logits:(logits.squeeze(-1) < 0).long(), epochs=1,
                execution={'prefetch_cached_inputs':depth})
            return load_checkpoint(_fk_paths.relocate((root/'last.pt'), __file__)), read(_fk_paths.relocate((root/'metrics.json'), __file__))
        with tempfile.TemporaryDirectory() as folder:
            torch.set_num_threads(1)
            reference, metrics = run(_fk_paths.relocate((Path(folder)/'baseline'), __file__), 0)
            optimized, optimized_metrics = run(_fk_paths.relocate((Path(folder)/'prefetch'), __file__), 2)
            self.assertEqual(reference['identity_sha256'], optimized['identity_sha256'])
            self.assertEqual(metrics, optimized_metrics)
            for key, value in reference['model_state'].items():
                torch.testing.assert_close(value, optimized['model_state'][key], rtol=0, atol=0)


if __name__ == '__main__':unittest.main()
