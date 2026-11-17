
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

import sys,unittest,copy,tempfile
from pathlib import Path
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'scripts'), __file__)))
from evaluate_completed_ablations import rebind_manifest


class AblationContextTests(unittest.TestCase):
    def test_rebinding_preserves_original_cache_and_ground_truth(self):
        source={'test':[{'cache':'caches/0001','label':1,'cache_metadata_sha256':'unchanged'}],
            'evaluation_context':{'expected_labels':{'v':1},'training_data_signatures':{'train':'same'},'checkpoint_sha256':'old'}}
        before=copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as directory:
            result=rebind_manifest(source,_fk_paths.relocate((Path(directory)), __file__),{'data_signatures':{'train':'same'}},'new-model','new-run',{'v':1})
            self.assertEqual(source,before)
            self.assertEqual(result['test'][0]['cache'],str((_fk_paths.relocate((Path(directory)/'caches/0001'), __file__)).resolve()))
            self.assertEqual(result['test'][0]['cache_metadata_sha256'],'unchanged')
            self.assertEqual(result['evaluation_context']['expected_labels'],{'v':1})
            self.assertEqual(result['evaluation_context']['checkpoint_sha256'],'new-model')

    def test_new_training_population_is_rejected(self):
        source={'evaluation_context':{'expected_labels':{'v':1},'training_data_signatures':{'train':'old'}}}
        with self.assertRaises(ValueError):rebind_manifest(source,_fk_paths.relocate((Path('.')), __file__),{'data_signatures':{'train':'other'}},'new','run',{'v':1})


if __name__=='__main__':unittest.main()
