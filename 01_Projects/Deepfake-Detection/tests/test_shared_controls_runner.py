
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

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from test_paper_alignment import rows
from deepfake_detection.feature_cache import write_cache
from deepfake_detection.training import TENSORS, _population_signature
from deepfake_detection.infrastructure import CachedVideoPopulation
from deepfake_detection.calibration import select_decision
from deepfake_detection.core import supervision

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('controls_runner', _fk_paths.relocate((ROOT / 'scripts/run_shared_controls.py'), __file__))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SharedControlWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = _fk_paths.relocate((Path(self.temporary.name)), __file__)
        torch.set_num_threads(1)

    def cache(self, name, splits):
        directory = self.root / name
        manifest = {split: [] for split in ('train', 'validation', 'test')}
        manifest.update(assessments=[], preprocessing={}, asset_sha256={}, window_label_policy={}, source_overlap_policy='error')
        for split in splits:
            for row in rows(split):
                provenance = {'source_sha256': row['source_sha256'], 'preprocessing_sha256': 'a'*64,
                              'encoder_sha256': {}, 'feature_layers': ['fixture'], 'precision': 'float32',
                              'window_intervals': row['window_intervals']}
                path = directory / row['video_id']
                write_cache(path, {k: row[k] for k in TENSORS}, provenance)
                metadata = {k: v for k, v in row.items() if k not in TENSORS}
                metadata.update(cache=row['video_id'], cache_provenance=provenance,
                                cache_metadata_sha256=runner.digest(_fk_paths.relocate((path/'metadata.json'), __file__)))
                manifest[split].append(metadata)
                target = supervision(row['label'], True, row['manipulated_intervals'], row['window_intervals'])
                assessment = {k: row[k] for k in ('video_id','source_family','label','split','duration','temporal_complete','manipulated_intervals')}
                assessment['windows'] = [{'interval': iv, 'status': 'USABLE', 'labels': {'combined': label['window']}}
                                         for iv, label in zip(row['window_intervals'], target['windows'])]
                manifest['assessments'].append(assessment)
        path = _fk_paths.relocate((directory/'manifest.json'), __file__)
        runner.write(path, manifest)
        return path, manifest

    def test_all_four_train_calibrate_evaluate_and_rerun_without_training(self):
        training_path, training = self.cache('training', ('train','validation'))
        test_path, test = self.cache('test', ('test',))
        config = runner.read(_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__))
        runner.write(_fk_paths.relocate((self.root/'training.json'), __file__), config)
        profile = runner.read(_fk_paths.relocate((ROOT/'configs/ours.provisional.json'), __file__))
        profile['paper_fixed']['dimension'] = 8
        profile['reconstruction_choices'].update(visual_feature_dimension=4, audio_feature_dimension=6, gru_hidden=8)
        runner.write(_fk_paths.relocate((self.root/'model.json'), __file__), profile)
        signatures = {split: _population_signature(CachedVideoPopulation(training[split], training_path.parent), split)[0]
                      for split in ('train','validation')}
        reference = {'config': config, 'data_signatures': signatures,
                     **{k: training[k] for k in runner.POLICIES}}
        reference_path = _fk_paths.relocate((self.root/'reference.pt'), __file__)
        torch.save(reference, reference_path)
        test['evaluation_context'] = {'schema':'fk-local-heldout-evaluation-v1',
            'scope':'synthetic integration test', 'population_sha256':'fixture',
            'training_data_signatures':signatures, 'checkpoint_sha256':runner.digest(reference_path),
            'expected_labels':{r['video_id']:r['label'] for r in test['assessments']}}
        runner.write(test_path, test, replace=True)
        args = SimpleNamespace(training_cache=training_path, test_cache=test_path, reference_checkpoint=reference_path,
                               model_config=_fk_paths.relocate((self.root/'model.json'), __file__), training_config=_fk_paths.relocate((self.root/'training.json'), __file__),
                               epochs=1, threads=1, device='cpu', output=_fk_paths.relocate((self.root/'controls'), __file__))
        grid = {'top_k':[3], 'peak_weight':[.5], 'top_ratio':[.2],
                'video_peak_weight':[.5], 'reliability_temperature':[1.]}
        def calibrate(model, population):
            return select_decision(model, population, grid=grid)
        with patch('deepfake_detection.calibration.select_decision', side_effect=calibrate):
            runner.workflow(args, training, test, reference)
            summary = runner.read(_fk_paths.relocate((args.output/'summary.json'), __file__))
            self.assertEqual(set(summary['models']), set(runner.KINDS))
            for kind in runner.KINDS:
                report = runner.read(_fk_paths.relocate((args.output/kind/'test.json'), __file__))
                selected = _fk_paths.relocate((args.output/kind/'attempt-001/selected.pt'), __file__)
                checkpoint = runner.load_checkpoint(selected)
                self.assertEqual(report['checkpoint_sha256'], runner.digest(selected))
                self.assertEqual(report['decision'], checkpoint['validation_selection']['decision'])
                self.assertEqual(report['video_metrics']['attempted'], 2)
                self.assertEqual(checkpoint['data_signatures'], signatures)
            original_cache = test_path.read_bytes()
            with patch('deepfake_detection.infrastructure.train_fk_manifest', side_effect=AssertionError('unexpected retraining')):
                runner.workflow(args, training, test, reference)
                # Simulate a crash after the last epoch but before selected.pt.
                selected = _fk_paths.relocate((args.output/'audio_only/attempt-001/selected.pt'), __file__)
                selected.unlink()
                recovered = runner.fit_control('audio_only', _fk_paths.relocate((args.output/'audio_only'), __file__), args, reference)
                self.assertTrue(recovered.is_file())
            self.assertEqual(test_path.read_bytes(), original_cache)
            self.assertEqual(runner.read(_fk_paths.relocate((args.output/'status.json'), __file__))['status'], 'FINISHED')
            wrong = copy.deepcopy(test)
            wrong['evaluation_context']['training_data_signatures'] = {'train': 'wrong'}
            with self.assertRaisesRegex(ValueError, 'different training population'):
                runner.evaluation_view(wrong, test_path, recovered)


if __name__ == '__main__':
    unittest.main()
