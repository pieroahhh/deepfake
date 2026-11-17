
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
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch

from deepfake_detection.cli import load_json, save
from deepfake_detection.commands import infer_fk_from_assets
from deepfake_detection.fk_caching import cache_fk_manifest
from deepfake_detection.forensics import sha256_file
from deepfake_detection.preparation import FK_QUALITY_PROFILE


class CacheResumeTests(unittest.TestCase):
    def test_resume_reuses_completed_clip_and_rebuilds_corrupt_cache(self):
        with TemporaryDirectory() as temporary:
            root = _fk_paths.relocate((Path(temporary)), __file__)
            rows = []
            for name in ('first', 'second'):
                source = _fk_paths.relocate((root / f'{name}.mp4'), __file__)
                source.write_bytes(name.encode())
                rows.append({'video_id': name, 'path': source.name, 'source_sha256': sha256_file(source),
                             'source_family': name, 'dataset': 'fixture', 'release': 'v1', 'split': 'train',
                             'duration': 2., 'label': 0, 'audio_label': 0, 'visual_label': 0,
                             'temporal_complete': True, 'manipulated_intervals': []})
            source_manifest, assets, output = _fk_paths.relocate((root / 'source.json'), __file__), _fk_paths.relocate((root / 'assets.json'), __file__), _fk_paths.relocate((root / 'output'), __file__)
            save(source_manifest, rows)
            save(assets, {'detector': {'sha256': '1' * 64}, 'silero': {'sha256': '2' * 64}})
            hashes = {name: {'model.safetensors': '3' * 64} for name in ('visual', 'audio')}
            processed = []
            interrupted = True

            def prepare(path, start, duration, detector, speech, **kwargs):
                if path.stem == 'second' and interrupted:
                    raise RuntimeError('simulated interruption')
                processed.append(path.stem)
                return {'status': 'USABLE' if start == 0 else 'UNASSESSED', 'interval': [start, start + 2.],
                        'source_sha256': sha256_file(path), 'reasons': [] if start == 0 else ['incomplete_window'],
                        'quality_measurements': {'face_coverage': 1., 'speech_coverage': 1.}}

            detector = Mock(method='fixture')
            with patch('deepfake_detection.fk_caching.load_fk_assets', return_value=(Mock(), detector, Mock())), \
                    patch('deepfake_detection.fk_caching.fk_asset_hashes', return_value=hashes), \
                    patch('deepfake_detection.fk_caching.fk_source_speech', return_value=[(0., 2.)]), \
                    patch('deepfake_detection.fk_caching.prepare_fk_window', side_effect=prepare), \
                    patch('deepfake_detection.fk_caching.encode_fk_window', return_value={'visual_features': torch.ones(1, 2, 4)}):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    cache_fk_manifest(source_manifest, root, assets, output, device='cpu')
                self.assertFalse((_fk_paths.relocate((output / 'training-manifest.json'), __file__)).exists())

                interrupted = False
                processed.clear()
                result = cache_fk_manifest(source_manifest, root, assets, output, device='cpu', resume=True)
                self.assertEqual(result['reused_clips'], 1)
                self.assertEqual(processed, ['second', 'second'])
                manifest = load_json(_fk_paths.relocate((output / 'training-manifest.json'), __file__))
                self.assertEqual([row['video_id'] for row in manifest['train']], ['first', 'second'])

                (_fk_paths.relocate((output / 'caches/000000/features.safetensors'), __file__)).write_bytes(b'corrupt')
                processed.clear()
                result = cache_fk_manifest(source_manifest, root, assets, output, device='cpu', resume=True)
                self.assertEqual(result['reused_clips'], 1)
                self.assertEqual(processed, ['first', 'first'])

    def test_inference_rejects_checkpoint_asset_mismatch(self):
        with patch('deepfake_detection.commands.load_json', return_value={}), \
                patch('deepfake_detection.commands.load_fk_assets', return_value=(Mock(), Mock(), Mock())), \
                patch('deepfake_detection.commands.fk_asset_hashes', return_value={'silero': 'actual'}), \
                patch('deepfake_detection.training.load_fk_checkpoint',
                      return_value=(Mock(), {'asset_sha256': {'silero': 'expected'}})), \
                patch('deepfake_detection.inference.run_fk_video') as run:
            with self.assertRaisesRegex(ValueError, 'inference assets differ'):
                infer_fk_from_assets('source', 2., 'checkpoint', 'assets', 'output', FK_QUALITY_PROFILE)
            run.assert_not_called()
