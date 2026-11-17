
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
from review_avfakenet_audio_intervals import merged_intervals,column_geometry


class AudioReviewTests(unittest.TestCase):
    def test_annotations_are_unioned_without_mutating_source(self):
        row={'split':'train','audio_label':1,'duration':3.,'audio_manipulated_intervals':[[1.,2.],[1.5,2.5]]}
        self.assertEqual(merged_intervals(row),[[1.,2.5]])
        self.assertEqual(row['audio_manipulated_intervals'],[[1.,2.],[1.5,2.5]])
        with self.assertRaises(ValueError):merged_intervals({**row,'split':'test'})

    def test_short_edit_can_have_no_column_center_but_nonzero_stft_overlap(self):
        geometry=column_geometry(10.,[[1.31,1.32]],100,10,.1,.2)
        self.assertEqual(geometry['stft_center_inside_edit_columns'],[])
        self.assertTrue(geometry['stft_support_overlapping_edit_columns'])

    def test_resize_can_skip_a_short_edit_between_selected_stft_columns(self):
        geometry=column_geometry(10.,[[1.01,1.02]],100,10,.1,.2)
        self.assertEqual(geometry['stft_support_overlapping_edit_columns'],[])


if __name__=='__main__':unittest.main()
