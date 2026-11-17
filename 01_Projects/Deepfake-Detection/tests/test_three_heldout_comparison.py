
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
from compare_three_heldout import compare


class ComparisonTests(unittest.TestCase):
    def test_coverage_and_common_clip_denominators(self):
        manifest={str(i):{'label':i%2} for i in range(4)}
        def rows(missing):return [{'video_id':str(i),'label':i%2,'prediction':None if i in missing else i%2,'score':None} for i in range(4)]
        result=compare({'A':rows({0}),'B':rows({1}),'C':rows(set())},manifest)
        self.assertEqual(result['common_assessed_ids'],['2','3'])
        self.assertEqual(result['own_assessed']['A']['coverage'],.75)
        for m in result['common_assessed'].values():
            self.assertEqual(m['assessed'],2);self.assertEqual(m['balanced_accuracy'],1.)

    def test_wrong_labels_and_duplicate_ids_are_rejected(self):
        manifest={'a':{'label':0},'b':{'label':1}}
        with self.assertRaises(ValueError):compare({'A':[{'video_id':'a','label':0,'prediction':0}]*2},manifest)
        with self.assertRaises(ValueError):compare({'A':[{'video_id':'a','label':1,'prediction':0},{'video_id':'b','label':1,'prediction':1}]},manifest)


if __name__=='__main__':unittest.main()
