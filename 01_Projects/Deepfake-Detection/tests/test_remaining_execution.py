
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

import unittest,sys,copy,tempfile
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(_fk_paths.relocate((ROOT/'scripts'), __file__)),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(ROOT)]
from prepare_remaining_derivatives import validate_timeline
from run_remaining_supplement import mutation_checks
from deepfake_detection.forensics import build_record,commitment,sha256_file
from deepfake_detection.core import Decision


class RemainingExecutionTests(unittest.TestCase):
    def test_timeline_rejects_shift_or_missing_frames(self):
        validate_timeline([0,.04,.08],[.001,.041,.081])
        for changed in ([.064,.104,.144],[0,.04]):
            with self.assertRaises(ValueError):validate_timeline([0,.04,.08],changed)

    def test_mutation_runner_preserves_source_and_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=_fk_paths.relocate((Path(temporary)), __file__);source=_fk_paths.relocate((root/'source'), __file__);source.write_bytes(b'fixture-only')
            window={'window_id':0,'interval':[0.,2.],'status':'USABLE','reasons':[],'quality':{},
                'bins':[{'interval':[i/5,(i+1)/5],'alpha':[.25]*4,'contributions':[.1]*4} for i in range(10)]}
            record=build_record(sha256_file(source),2.,[window],Decision(),model_version='synthetic-fixture',preprocessing={})
            original=copy.deepcopy(record);expected=source.read_bytes()
            result=mutation_checks(record,commitment(record),source,root,_fk_paths.relocate((root/'mutations'), __file__))
            self.assertTrue(result['original']['valid'])
            self.assertEqual(result['artifact_bytes']['status'],'NOT_APPLICABLE')
            self.assertTrue(all(result[k]['status']=='REJECTED_AS_EXPECTED' for k in ('contribution','timestamp','threshold','coverage','source_bytes')))
            self.assertEqual(record,original);self.assertEqual(source.read_bytes(),expected)


if __name__=='__main__':unittest.main()
