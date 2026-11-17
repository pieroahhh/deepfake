"""Regression checks for paper alignment; fixtures are not benchmark evidence."""

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
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deepfake_detection.cli import load_json
from deepfake_detection.core import Decision
from deepfake_detection.model import EvidenceModel
from deepfake_detection.quality import fit_profile, quality_tensor, profile_hash
from deepfake_detection.shared_baselines import SharedBaseline
from deepfake_detection.training import fit_fk, load_fk_checkpoint

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]


def rows(split):
    generator = torch.Generator().manual_seed({'train':7, 'validation':8, 'test':9}[split])
    return [dict(video_id=f'{split}-{label}',source_family=f'{split}-{label}',
                 source_sha256=str(label+{'train':1,'validation':3,'test':5}[split])*64,
                 split=split,label=label,temporal_complete=True,duration=3.,
                 manipulated_intervals=[] if label==0 else [[.2,.4]],window_intervals=[[0.,2.],[1.,3.]],
                 visual_features=torch.randn(2,50,4,generator=generator),
                 audio_features=torch.randn(2,100,6,generator=generator),
                 visual_times=torch.arange(50)/25,audio_times=torch.arange(100)/50,
                 quality=torch.ones(2,2)) for label in (0,1)]


def measurement(blur=50., snr=10.):
    return dict(face_coverage=1.,mean_detection_confidence=.95,mean_tracking_iou=.9,
                mean_face_laplacian_variance=blur,speech_coverage=.75,
                audio_clipping_fraction=0.,estimated_snr_db=snr)


class AlignmentTests(unittest.TestCase):
    def test_official_overlap_training_is_explicit_and_checkpointed(self):
        config=load_json(_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__))
        train,val=rows('train'),rows('validation')
        val[0]['source_family']=train[0]['source_family']
        val[0]['source_sha256']=train[0]['source_sha256']
        model=SharedBaseline('audio_only',visual_dim=4,audio_dim=6,hidden=8,dimension=8)
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__)
            with self.assertRaisesRegex(ValueError,'leakage'):
                fit_fk(model,train,val,config,_fk_paths.relocate((root/'strict'), __file__),epochs=1)
            policy={'source_overlap_policy':'keep-official'}
            fit_fk(model,train,val,config,_fk_paths.relocate((root/'official'), __file__),epochs=1,manifest_policy=policy)
            checkpoint=torch.load(_fk_paths.relocate_value((_fk_paths.relocate((root/'official/last.pt'), __file__)), __file__),weights_only=True)
            self.assertEqual(checkpoint['source_overlap_policy'],'keep-official')
            self.assertEqual(checkpoint['train_validation_overlap'],{'source_families':1,'content_hashes':1})
            fit_fk(model,train,val,config,_fk_paths.relocate((root/'resume'), __file__),epochs=2,
                   resume_from=_fk_paths.relocate((root/'official/last.pt'), __file__),manifest_policy=policy)

    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_quality_train_only_degradation_and_missing_snr(self):
        training=[{'split':'train','video_id':'a','source_sha256':'a'*64,'windows':[
            {'status':'USABLE','quality_measurements':measurement(10,0)},
            {'status':'USABLE','quality_measurements':measurement(100,20)}]}]
        profile=fit_profile(training)
        good=quality_tensor(measurement(100,20),profile)
        bad=quality_tensor({**measurement(10,0),'audio_clipping_fraction':.8},profile)
        self.assertTrue((bad<good).all())
        self.assertTrue(torch.isfinite(quality_tensor(measurement(snr=None),profile)).all())
        changed=copy.deepcopy(training);changed[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'training rows only'):fit_profile(changed)
        self.assertNotEqual(profile_hash(profile),profile_hash({**profile,'usable_windows':3}))

    def test_pcm_matches_saved_audio(self):
        import soundfile as sf
        from deepfake_detection.preparation import pcm16_samples,save_prepared_window
        values=pcm16_samples(np.array([-1.1,-.000012,0.,.12345,1.1],dtype=np.float32))
        prepared={'status':'USABLE','faces':np.zeros((1,2,2,3),dtype=np.uint8),
                  'lips':np.zeros((1,2,2,3),dtype=np.uint8),'audio':values,'visual_times':np.zeros(1)}
        with tempfile.TemporaryDirectory() as tmp:
            save_prepared_window(prepared,_fk_paths.relocate((Path(tmp)/'derived'), __file__))
            actual,_=sf.read(_fk_paths.relocate((Path(tmp)/'derived/audio-16k-pcm16.wav'), __file__),dtype='float32')
            np.testing.assert_array_equal(actual,values)

    def test_calibration_recomputes_temperature_and_rejects_test(self):
        from deepfake_detection.calibration import select_decision
        torch.manual_seed(10)
        model=EvidenceModel(visual_dim=4,audio_dim=6,hidden=8,dimension=8)
        grid={'top_k':[1,3], 'peak_weight':[.5], 'top_ratio':[.2],
              'video_peak_weight':[.5], 'reliability_temperature':[.25,1.]}
        original=model.decision
        result=select_decision(model,rows('validation'),grid)
        self.assertEqual(len(result['candidates']),4)
        self.assertEqual(model.decision,original)
        self.assertEqual(set(result['decision']),set(original.to_dict()))
        from deepfake_detection.training import forward_video
        selected=Decision(**result['decision']);model.decision=selected
        direct=[]
        for row in rows('validation'):
            _,score,_=forward_video(model,row)
            direct.append(int(score>=selected.video_threshold)==row['label'])
        self.assertAlmostEqual(sum(direct)/2,result['balanced_accuracy'])
        with self.assertRaisesRegex(ValueError,'test inference'):
            select_decision(model,rows('test'),grid)

    def test_baseline_train_resume_reload_and_modality_isolation(self):
        config=load_json(_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__))
        train,val=rows('train'),rows('validation')
        for kind in ('visual_only','audio_only','concatenation','late_fusion'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=_fk_paths.relocate((Path(tmp)), __file__)
                torch.manual_seed(20)
                model=SharedBaseline(kind,visual_dim=4,audio_dim=6,hidden=8,dimension=8)
                initial=copy.deepcopy(model)
                fit_fk(model,train,val,config,_fk_paths.relocate((root/'first'), __file__),epochs=1)
                fit_fk(model,train,val,config,_fk_paths.relocate((root/'resume'), __file__),epochs=2,resume_from=_fk_paths.relocate((root/'first/last.pt'), __file__))
                fit_fk(initial,train,val,config,_fk_paths.relocate((root/'continuous'), __file__),epochs=2)
                resumed=torch.load(_fk_paths.relocate_value((_fk_paths.relocate((root/'resume/last.pt'), __file__)), __file__),weights_only=True)
                continuous=torch.load(_fk_paths.relocate_value((_fk_paths.relocate((root/'continuous/last.pt'), __file__)), __file__),weights_only=True)
                for key,value in resumed['model_state'].items():
                    torch.testing.assert_close(value,continuous['model_state'][key],rtol=0,atol=0)
                loaded,_=load_fk_checkpoint(_fk_paths.relocate((root/'resume/best.pt'), __file__))
                self.assertEqual(loaded.model_kind,kind)
                values={k:val[0][k] for k in ('visual_features','audio_features','visual_times','audio_times','quality')}
                expected=loaded(**values)['bin_scores']
                changed=dict(values)
                if kind=='visual_only':changed['audio_features']=values['audio_features']+10
                if kind=='audio_only':changed['visual_features']=values['visual_features']+10
                if kind in ('visual_only','audio_only'):
                    torch.testing.assert_close(expected,loaded(**changed)['bin_scores'],rtol=0,atol=0)

    def test_v2_artifact_tamper_and_escape(self):
        from deepfake_detection.forensics import build_record,commitment,verify,sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__);artifact=_fk_paths.relocate((root/'audio.wav'), __file__);artifact.write_bytes(b'evidence')
            record=build_record('a'*64,1.,[{'window_id':0,'interval':[0.,2.],
                'status':'UNASSESSED','quality':{},'reasons':['incomplete_window']}],Decision(),
                model_version='fixture',preprocessing={})
            record.update(schema_version='fk-record-v2',source_metadata={'duration':1.},derived_evidence=[
                {'path':'audio.wav','sha256':sha256_file(artifact),'source_sha256':'a'*64,'interval':[0.,1.]}])
            trusted=commitment(record)
            self.assertEqual(verify(record,trusted,artifact_root=root)['derived_artifacts_checked'],1)
            artifact.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'artifact hash'):verify(record,trusted,artifact_root=root)
            record['derived_evidence'][0]['path']='../outside.wav'
            with self.assertRaises(ValueError):verify(record,commitment(record),artifact_root=root)

    def test_evaluation_retains_unassessed_videos_and_overlap_windows(self):
        from deepfake_detection.feature_cache import write_cache
        from deepfake_detection.forensics import sha256_file
        from deepfake_detection.core import supervision
        from deepfake_detection.training import TENSORS
        from deepfake_detection.cli import save
        from deepfake_detection.study import evaluate_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__);manifest={'test':[],'assessments':[]}
            for row in rows('test'):
                provenance={'source_sha256':row['source_sha256'],'preprocessing_sha256':'a'*64,
                            'encoder_sha256':{},'feature_layers':['fixture'],'precision':'float32',
                            'window_intervals':row['window_intervals']}
                directory=_fk_paths.relocate((root/row['video_id']), __file__)
                write_cache(directory,{k:row[k] for k in TENSORS},provenance)
                entry={k:v for k,v in row.items() if k not in TENSORS}
                entry.update(cache=row['video_id'],cache_provenance=provenance,
                             cache_metadata_sha256=sha256_file(_fk_paths.relocate((directory/'metadata.json'), __file__)))
                manifest['test'].append(entry)
                labels=supervision(row['label'],True,row['manipulated_intervals'],row['window_intervals'])
                assessment={k:row[k] for k in ('video_id','source_family','label','split','duration','temporal_complete','manipulated_intervals')}
                assessment['windows']=[{'interval':iv,'status':'USABLE','labels':{'combined':label['window']}}
                                       for iv,label in zip(row['window_intervals'],labels['windows'])]
                manifest['assessments'].append(assessment)
            manifest['assessments'].append({'video_id':'excluded','source_family':'excluded','split':'test',
                'label':1,'duration':1.,'temporal_complete':True,'manipulated_intervals':[[.1,.3]],
                'windows':[{'interval':[0.,2.],'status':'UNASSESSED','labels':{'combined':1}}]})
            save(_fk_paths.relocate((root/'manifest.json'), __file__),manifest)
            model=SharedBaseline('visual_only',visual_dim=4,audio_dim=6,hidden=8,dimension=8)
            checkpoint={'schema':'fk-training-v1','model_spec':model.spec,'decision':model.decision.to_dict(),
                        'model_state':model.state_dict(),'validation_selection':{'split':'validation'}}
            torch.save(checkpoint,_fk_paths.relocate((root/'selected.pt'), __file__))
            report=evaluate_checkpoint(_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((root/'selected.pt'), __file__),_fk_paths.relocate((root/'report.json'), __file__))
            self.assertEqual(report['video_metrics']['attempted'],3)
            self.assertEqual(report['video_metrics']['assessed'],2)
            self.assertEqual(report['window_metrics']['attempted'],5)
            self.assertEqual(report['window_coverage_by_label']['1'],{'total':2,'assessed':1})
            self.assertAlmostEqual(report['timeline_metrics_supplement'][-1]['seconds']['unassessed_positive'],.2)
            context={'schema':'fk-local-heldout-evaluation-v1',
                     'checkpoint_sha256':sha256_file(_fk_paths.relocate((root/'selected.pt'), __file__)),
                     'training_data_signatures':None,
                     'expected_labels':{r['video_id']:r['label'] for r in manifest['assessments']}}
            manifest['evaluation_context']=context
            save(_fk_paths.relocate((root/'heldout.json'), __file__),manifest)
            heldout_report=evaluate_checkpoint(_fk_paths.relocate((root/'heldout.json'), __file__),_fk_paths.relocate((root/'selected.pt'), __file__),_fk_paths.relocate((root/'heldout-report.json'), __file__))
            self.assertEqual(heldout_report['evaluation_context'],context)
            context['expected_labels']['missing']=0
            save(_fk_paths.relocate((root/'heldout-missing.json'), __file__),manifest)
            with self.assertRaisesRegex(ValueError,'frozen population'):
                evaluate_checkpoint(_fk_paths.relocate((root/'heldout-missing.json'), __file__),_fk_paths.relocate((root/'selected.pt'), __file__),_fk_paths.relocate((root/'invalid-report.json'), __file__))
            context['checkpoint_sha256']='wrong'
            save(_fk_paths.relocate((root/'heldout-wrong.json'), __file__),manifest)
            with self.assertRaisesRegex(ValueError,'checkpoint provenance'):
                evaluate_checkpoint(_fk_paths.relocate((root/'heldout-wrong.json'), __file__),_fk_paths.relocate((root/'selected.pt'), __file__),_fk_paths.relocate((root/'invalid-report.json'), __file__))
            manifest.pop('evaluation_context')
            checkpoint.pop('validation_selection');torch.save(checkpoint,_fk_paths.relocate((root/'raw.pt'), __file__))
            with self.assertRaisesRegex(ValueError,'validation-selected'):
                evaluate_checkpoint(_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((root/'raw.pt'), __file__),_fk_paths.relocate((root/'bad.json'), __file__))
            manifest['train']=[{**manifest['test'][0],'video_id':'train-copy','split':'train'}]
            save(_fk_paths.relocate((root/'overlap.json'), __file__),manifest)
            with self.assertRaisesRegex(ValueError,'leakage'):
                evaluate_checkpoint(_fk_paths.relocate((root/'overlap.json'), __file__),_fk_paths.relocate((root/'selected.pt'), __file__),_fk_paths.relocate((root/'strict.json'), __file__))
            manifest['source_overlap_policy']='keep-official'
            save(_fk_paths.relocate((root/'official.json'), __file__),manifest)
            with self.assertRaisesRegex(ValueError,'overlap policy'):
                evaluate_checkpoint(_fk_paths.relocate((root/'official.json'), __file__),_fk_paths.relocate((root/'selected.pt'), __file__),_fk_paths.relocate((root/'mismatch.json'), __file__))
            checkpoint.update(source_overlap_policy='keep-official',validation_selection={'split':'validation'})
            torch.save(checkpoint,_fk_paths.relocate((root/'official.pt'), __file__))
            report=evaluate_checkpoint(_fk_paths.relocate((root/'official.json'), __file__),_fk_paths.relocate((root/'official.pt'), __file__),_fk_paths.relocate((root/'official-report.json'), __file__))
            self.assertEqual(report['source_overlap_policy'],'keep-official')
            self.assertEqual(report['video_metrics']['attempted'],3)

    def test_v2_shared_experiment_runner_smoke(self):
        from deepfake_detection.cli import save
        from deepfake_detection.feature_cache import write_cache
        from deepfake_detection.forensics import sha256_file
        from deepfake_detection.training import TENSORS
        from deepfake_detection.experiments import prepare_experiments
        from deepfake_detection.study import run_experiments
        from deepfake_detection.infrastructure import train_fk_manifest
        from deepfake_detection.quality import QUALITY_V2
        profile=fit_profile([{'split':'train','video_id':'fixture','source_sha256':'a'*64,
            'windows':[{'status':'USABLE','quality_measurements':measurement()}]}])
        policy={'quality_profile':QUALITY_V2,'quality_normalization':profile,
                'quality_normalization_sha256':profile_hash(profile),'face_crop_policy':'fixture'}
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__)
            manifest={'schema':'fk-cache-manifest-v2','preprocessing':policy,
                      'asset_sha256':{},'window_label_policy':{'rule':'any_overlap'}}
            for split in ('train','validation'):
                manifest[split]=[]
                for row in rows(split):
                    provenance={'source_sha256':row['source_sha256'],'preprocessing_sha256':profile_hash(policy),
                        'face_crop_policy':'fixture','asset_sha256':{},'window_label_policy':manifest['window_label_policy'],
                        'encoder_sha256':{},'feature_layers':['fixture'],'precision':'float32',
                        'window_intervals':row['window_intervals']}
                    directory=_fk_paths.relocate((root/row['video_id']), __file__)
                    write_cache(directory,{k:row[k] for k in TENSORS},provenance)
                    entry={k:v for k,v in row.items() if k not in TENSORS}
                    entry.update(cache=row['video_id'],cache_provenance=provenance,
                                 cache_metadata_sha256=sha256_file(_fk_paths.relocate((directory/'metadata.json'), __file__)))
                    manifest[split].append(entry)
            save(_fk_paths.relocate((root/'manifest.json'), __file__),manifest)
            config=load_json(_fk_paths.relocate((ROOT/'configs/ours.provisional.json'), __file__))
            config['paper_fixed']['dimension']=8
            config['reconstruction_choices'].update(visual_feature_dimension=4,audio_feature_dimension=6,gru_hidden=8)
            plan=prepare_experiments(config,_fk_paths.relocate((root/'plan'), __file__),seeds=(7,))
            plan['runs']=[r for r in plan['runs'] if r['method'] in ('visual_only','audio_only','concatenation','late_fusion') or (r['method']=='FK' and r['variant']=='full')]
            save(_fk_paths.relocate((root/'plan/smoke.json'), __file__),plan)
            report=run_experiments(_fk_paths.relocate((root/'plan/smoke.json'), __file__),_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__),_fk_paths.relocate((root/'runs'), __file__),epochs=1)
            self.assertEqual(len(report['runs']),5)
            for run in report['runs']:
                model,checkpoint=load_fk_checkpoint(run['selected_checkpoint'])
                self.assertEqual(checkpoint['preprocessing'],policy)
                self.assertEqual(checkpoint['validation_selection']['split'],'validation')
            manifest['preprocessing']['quality_normalization']['fit_split']='test'
            save(_fk_paths.relocate((root/'bad.json'), __file__),manifest)
            with self.assertRaisesRegex(ValueError,'quality normalization'):
                train_fk_manifest(_fk_paths.relocate((root/'bad.json'), __file__),_fk_paths.relocate((root/'plan/full.json'), __file__),_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__),_fk_paths.relocate((root/'bad-run'), __file__))


if __name__=='__main__':unittest.main()
