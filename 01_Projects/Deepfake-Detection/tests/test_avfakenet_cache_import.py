
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
import numpy as np
from import_avfakenet_machine_b_cache import converted


class CacheImportTests(unittest.TestCase):
    def test_timestamp_labels_half_open_and_arrays_unchanged(self):
        row={'duration':.08,'visual_label':1,'visual_manipulated_intervals':[[.04,.08]]}
        face=np.zeros((2,128,128,3),dtype=np.uint8);mel=np.zeros((3,175,175),dtype=np.float32)
        values=converted(row,{'faces':face,'source_times':np.array([0.,.04]),'mel':mel},{})
        self.assertEqual(values['visual-labels.npy'].tolist(),[0,1])
        self.assertIs(values['visual.npy'],face);self.assertIs(values['audio.npy'],mel)
        with self.assertRaises(ValueError):converted(row,{'faces':face,'source_times':np.array([0.,.04]),'mel':mel},{'audio':'failure'})

    def test_rejected_modalities_remain_absent(self):
        self.assertEqual(converted({'duration':1.},{},{'audio':'bad','visual':'bad'}),{})


if __name__=='__main__':unittest.main()
