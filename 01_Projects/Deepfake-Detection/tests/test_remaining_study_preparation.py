
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

import sys,unittest
from pathlib import Path
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'scripts'), __file__)))
from prepare_remaining_study import calibration_grid,select_balanced,localization_summary


class PreparationTests(unittest.TestCase):
    def test_aggregation_axes_stay_fixed_during_validation_search(self):
        self.assertEqual(calibration_grid('window_max')['peak_weight'],[1.])
        self.assertEqual(calibration_grid('window_mean')['top_k'],[10])
        self.assertEqual(calibration_grid('window_mean')['peak_weight'],[0.])
        self.assertEqual(calibration_grid('video_max')['video_peak_weight'],[1.])
        self.assertEqual(calibration_grid('video_mean')['top_ratio'],[1.])
        self.assertEqual(calibration_grid('video_mean')['video_peak_weight'],[0.])

    def test_metadata_selection_is_balanced_and_order_independent(self):
        rows=[{'video_id':f'{kind}-{i}','source_modify_type':kind} for kind in ('real','audio','visual','both') for i in range(8)]
        a=select_balanced(rows,2);b=select_balanced(list(reversed(rows)),2)
        self.assertEqual(a,b);self.assertEqual(len(a),8)

    def test_localization_keeps_unassessed_positive_time_in_full_denominator(self):
        result=localization_summary([{'duration':10,'seconds':{'tp':2.,'tn':3.,'fp':1.,'fn':1.,'unassessed_positive':2.,'unassessed_negative':1.}}])
        self.assertAlmostEqual(result['time_assessment_coverage'],.7)
        self.assertAlmostEqual(result['duration_weighted_assessed_region']['recall'],2/3)
        self.assertAlmostEqual(result['detected_fraction_of_all_manipulated_time'],.4)


if __name__=='__main__':unittest.main()
