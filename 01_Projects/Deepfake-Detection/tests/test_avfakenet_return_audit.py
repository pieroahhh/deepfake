
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
from audit_avfakenet_return import inspect_predictions,visual_duration_diagnostic


class ReturnAuditTests(unittest.TestCase):
    def rows(self):
        manifest=[{'video_id':'r','label':0,'visual_label':0,'audio_label':0},
                  {'video_id':'f','label':1,'visual_label':1,'audio_label':0},
                  {'video_id':'u','label':1,'visual_label':0,'audio_label':1}]
        rows=[{'video_id':r['video_id'],'label':r['label'],'split':'test','status':'ASSESSED',
               'visual_prediction':0,'audio_prediction':1,'prediction':1,'score':None} for r in manifest]
        rows[-1].update(status='UNASSESSED',prediction=None,reason='missing_face')
        return rows,manifest

    def test_constant_branches_explain_or_and_baseline_not_detection(self):
        rows,manifest=self.rows();result=inspect_predictions(rows,manifest)
        self.assertEqual(result['fusion_inconsistent_ids'],[])
        self.assertEqual(result['metrics']['balanced_accuracy'],.5)
        self.assertTrue(result['constant_fake_baseline_matches_return'])
        self.assertEqual(result['metrics']['coverage'],2/3)
        self.assertIsNone(result['metrics']['roc_auc'])
        self.assertEqual(result['branch_confusion_against_modality_labels']['audio'],{'fp':2})

    def test_bad_labels_and_duplicate_ids_fail_closed(self):
        rows,manifest=self.rows();rows[0]['label']=1
        with self.assertRaises(ValueError):inspect_predictions(rows,manifest)
        rows,manifest=self.rows();rows[-1]['video_id']='r'
        with self.assertRaises(ValueError):inspect_predictions(rows,manifest)

    def test_fusion_error_is_distinguished_from_branch_error(self):
        rows,manifest=self.rows();rows[0]['prediction']=0
        self.assertEqual(inspect_predictions(rows,manifest)['fusion_inconsistent_ids'],['r'])

    def test_short_edit_diagnostic_does_not_require_collapsed_frame_scores(self):
        manifest=[{'video_id':'short','visual_label':1,'duration':10.,'visual_manipulated_intervals':[[2.,2.4]]},
                  {'video_id':'long','visual_label':1,'duration':10.,'visual_manipulated_intervals':[[0.,8.]]}]
        result=visual_duration_diagnostic(manifest,{'short'})
        self.assertEqual(result['all_heldout']['below_half_manipulated_duration'],1)
        self.assertEqual(result['assessed_heldout']['known_visually_manipulated_clips'],1)
        self.assertAlmostEqual(result['assessed_heldout']['median_manipulated_duration_fraction'],.04)
        self.assertIn('not actual',visual_duration_diagnostic.__doc__)


if __name__=='__main__':unittest.main()
