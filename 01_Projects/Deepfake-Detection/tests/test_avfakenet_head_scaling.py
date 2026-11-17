
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

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(_fk_paths.relocate((Path(__file__).resolve().parents[1] / 'scripts'), __file__)))
import numpy as np
from diagnose_avfakenet_head_scaling import normalization, balanced_batches


class HeadScalingTests(unittest.TestCase):
    def test_scaler_does_not_fit_evaluation_rows_or_change_input(self):
        features = np.array([[1., 5.], [3., 5.], [1000., 5.]], dtype=np.float32)
        original = features.copy()
        transformed, mean, scale = normalization(features, np.array([0, 1]), True)
        np.testing.assert_array_equal(features, original)
        np.testing.assert_allclose(mean, [2, 5])
        np.testing.assert_allclose(scale, [1, 1])
        np.testing.assert_allclose(transformed, [[-1, 0], [1, 0], [998, 0]])

    def test_paired_batches_are_deterministic_balanced_and_stay_in_training_fold(self):
        labels = np.array([0] * 8 + [1] * 8)
        train = np.array([0, 1, 2, 3, 8, 9, 10, 11])
        first = list(balanced_batches(labels, train, steps=20))
        second = list(balanced_batches(labels, train, steps=20))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(int(labels[a].sum()), 4)
            self.assertTrue(set(a) <= set(train))


if __name__ == '__main__':
    unittest.main()
