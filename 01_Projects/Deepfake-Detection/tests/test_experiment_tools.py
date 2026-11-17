"""Dataset-independent tests. Generated media and explicit synthetic providers only."""

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
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from deepfake_detection.cli import load_json
from deepfake_detection.media import media_tool

ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]


class ExperimentToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def inputs(self):
        return dict(visual_features=torch.randn(1,50,4),audio_features=torch.randn(1,100,6),
                    visual_times=torch.arange(50)/25,audio_times=torch.arange(100)/50,quality=torch.ones(1,2))

    def test_retrained_ablation_masks_renormalize_and_remove_branch_leakage(self):
        from deepfake_detection.model import EvidenceModel
        for disabled,channel in (('av',2),('temporal',3)):
            model=EvidenceModel(visual_dim=4,audio_dim=6,hidden=8,dimension=8,disabled_evidence=[disabled]).eval()
            inputs=self.inputs();before=model(**inputs)
            self.assertTrue(torch.equal(before['alpha'][...,channel],torch.zeros(1,10)))
            torch.testing.assert_close(before['alpha'].sum(-1),torch.ones(1,10))
            with torch.no_grad():
                for layer in ((model.av,model.vq,model.heads[2]) if disabled=='av' else (model.temporal,model.dv,model.heads[3])):
                    for parameter in layer.parameters():parameter.fill_(1000)
            after=model(**inputs)
            torch.testing.assert_close(before['bin_scores'],after['bin_scores'],rtol=0,atol=0)
            after['window_scores'].sum().backward()
            self.assertIsNotNone(model.visual.weight.grad)
        uniform=EvidenceModel(visual_dim=4,audio_dim=6,hidden=8,dimension=8,uniform_weights=True).eval()
        inputs=self.inputs();first=uniform(**inputs)
        inputs['quality'].zero_();second=uniform(**inputs)
        torch.testing.assert_close(first['alpha'],torch.full((1,10,4),.25))
        torch.testing.assert_close(first['bin_scores'],second['bin_scores'])

    def test_experiment_plan_has_retraining_and_separate_aggregation_variants(self):
        from deepfake_detection.experiments import prepare_experiments
        with tempfile.TemporaryDirectory() as tmp:
            output=_fk_paths.relocate((Path(tmp)/'plan'), __file__)
            plan=prepare_experiments(load_json(_fk_paths.relocate((ROOT/'configs/ours.provisional.json'), __file__)),output)
            self.assertEqual(len(plan['runs']),42)
            self.assertTrue({'visual_only','audio_only','concatenation','late_fusion'} <= {r['method'] for r in plan['runs']})
            self.assertTrue(all(row['status']=='PLANNED_NOT_RUN' and row['requires_retraining'] for row in plan['runs']))
            self.assertEqual(load_json(_fk_paths.relocate((output/'window_mean.json'), __file__))['decision']['top_k'],10)
            self.assertEqual(load_json(_fk_paths.relocate((output/'video_mean.json'), __file__))['decision']['top_ratio'],1.)
            self.assertEqual(load_json(_fk_paths.relocate((output/'no_av.json'), __file__))['ablation']['disabled_evidence'],['av'])

    def test_robustness_is_deterministic_nonmutating_and_snr_measured(self):
        from deepfake_detection.robustness import corrupt_frames,audio_noise
        original=np.random.default_rng(1).integers(0,256,(3,24,32,3),dtype=np.uint8)
        before=original.copy()
        for kind,strength in (('jpeg',60),('blur',1.),('downsample',.5),('gaussian_noise',5.)):
            a=corrupt_frames(original,kind,strength,seed=7);b=corrupt_frames(original,kind,strength,seed=7)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(a.shape,original.shape)
        np.testing.assert_array_equal(original,before)
        signal=(.1*np.sin(np.arange(16000)*.1)).astype(np.float32)
        noisy,metadata=audio_noise(signal,20,seed=7)
        snr=10*np.log10(np.mean(signal.astype(float)**2)/np.mean((noisy-signal).astype(float)**2))
        self.assertAlmostEqual(snr,20,places=4)
        np.testing.assert_array_equal(noisy,audio_noise(signal,20,seed=7)[0])
        self.assertEqual(metadata['alignment'],'unchanged')

    def test_localization_overlap_average_and_unassessed_manipulated_time(self):
        from deepfake_detection.reporting import timeline_metrics
        result=timeline_metrics(4.,[{'interval':[0.,2.],'score':.8},{'interval':[1.,3.],'score':.2}],
                                [(1.,2.),(3.,4.)],.5)
        self.assertAlmostEqual(result['assessed_region']['precision'],.5)
        self.assertAlmostEqual(result['assessed_region']['recall'],1.)
        self.assertAlmostEqual(result['manipulated_time_assessment_coverage'],.5)
        self.assertAlmostEqual(result['detected_fraction_of_all_manipulated_time'],.5)
        self.assertIsNone(result['segments'][-1]['score'])
        self.assertAlmostEqual(result['assessment_coverage'],.75)

    def rows(self):
        return [dict(video_id=str(index),source_family='family-'+str(index//2),label=label,
                     status='ASSESSED',prediction=label,score=.9 if label else .1)
                for index,label in enumerate((0,0,1,1))]

    def test_metrics_preserve_hard_baseline_and_pair_bootstrap_groups(self):
        from deepfake_detection.reporting import scored_metrics,paired_group_bootstrap,write_detection_report
        rows=self.rows();hard=[{**row,'score':None} for row in rows]
        self.assertEqual(scored_metrics(rows)['roc_auc'],1.)
        self.assertIsNone(scored_metrics(hard)['roc_auc'])
        methods={'continuous':rows,'hard':hard}
        result=paired_group_bootstrap(methods,reference='continuous',replicates=100)
        self.assertEqual(result['paired_differences']['hard']['balanced_accuracy']['ci95'],[0.,0.])
        self.assertGreater(result['paired_differences']['hard']['balanced_accuracy']['undefined_resamples'],0)
        with tempfile.TemporaryDirectory() as tmp:
            output=_fk_paths.relocate((Path(tmp)/'report'), __file__)
            report=write_detection_report(methods,output,provenance={'protocol_id':'synthetic-fixture','data_kind':'synthetic'},
                                          reference='continuous',replicates=20)
            self.assertTrue((_fk_paths.relocate((output/'metrics.csv'), __file__)).is_file())
            self.assertIsNone(report['per_method_assessed']['hard']['roc_auc'])

    def test_profile_reports_measured_boundary_and_units(self):
        from deepfake_detection.profiling import profile_stage,parameter_storage
        model=nn.Linear(4,2)
        count=[0]
        def callback():
            count[0]+=1;return model(torch.ones(1,4))
        result=profile_stage(callback,name='synthetic-linear-forward',unit='one_fixture',iterations=4,warmup=2)
        self.assertEqual(count[0],6)
        self.assertEqual(len(result['milliseconds']['samples']),4)
        self.assertGreater(result['milliseconds']['median'],0)
        self.assertEqual(parameter_storage(model)['total_parameters'],10)

    def test_production_inference_refuses_missing_pretrained_assets_without_output(self):
        from deepfake_detection.commands import infer_fk_from_assets
        with tempfile.TemporaryDirectory() as tmp:
            output=_fk_paths.relocate((Path(tmp)/'must-not-exist'), __file__)
            with self.assertRaisesRegex(ValueError,'actual pretrained artifact required'):
                infer_fk_from_assets('missing-video.mp4',3.,'missing-checkpoint.pt',
                                     _fk_paths.relocate((ROOT/'configs/assets.example.json'), __file__),output,'availability-speech-v1-provisional')
            self.assertFalse(output.exists())

    def test_parity_rejects_wrong_inputs_and_detects_permuted_intermediate_vectors(self):
        from deepfake_detection.validation_tools import compare_backend_exports
        reference={'spatial':np.arange(12).reshape(1,3,4),'temporal':np.ones((1,4))}
        provenance={'implementation_revision':'fixture-v1','checkpoint_sha256':'1'*64,'input_sha256':'2'*64}
        self.assertTrue(compare_backend_exports(reference,reference,reference_provenance=provenance,
                                               actual_provenance=provenance)['passed'])
        changed={'spatial':reference['spatial'][:,::-1],'temporal':reference['temporal']}
        self.assertFalse(compare_backend_exports(reference,changed,reference_provenance=provenance,
                                                actual_provenance=provenance)['passed'])
        with self.assertRaises(ValueError):
            compare_backend_exports(reference,reference,reference_provenance=provenance,
                                    actual_provenance={**provenance,'input_sha256':'3'*64})

    def test_explicit_flax_array_conversion_and_pretrained_promotion_gate(self):
        from deepfake_detection.backbones import FactorizedViViT
        from deepfake_detection.checkpoint_conversion import convert_factorized_arrays
        config=dict(image_size=16,frames=4,tubelet=[2,8,8],dimension=8,spatial_depth=1,
                    temporal_depth=1,heads=2,mlp_ratio=2,dropout=0.,layer_norm_epsilon=1e-6)
        model=FactorizedViViT(config).eval();arrays={};rules={}
        for name,tensor in model.state_dict().items():
            value=tensor.numpy()
            if name.endswith('in_proj_weight'):
                names=[name+'/'+part for part in ('query','key','value')]
                for key,part in zip(names,np.split(value,3)):arrays[key]=part.T.reshape(8,2,4)
                operation='qkv_weight'
            elif name.endswith('in_proj_bias'):
                names=[name+'/'+part for part in ('query','key','value')]
                for key,part in zip(names,np.split(value,3)):arrays[key]=part.reshape(2,4)
                operation='qkv_bias'
            else:
                names=[name]
                if name=='tubelets.weight':arrays[name]=value.transpose(2,3,4,1,0);operation='conv3d_flax_to_torch'
                elif name.endswith('out_proj.weight'):arrays[name]=value.T.reshape(2,4,8);operation='attention_output'
                elif value.ndim==2:arrays[name]=value.T;operation='dense_transpose'
                else:arrays[name]=value;operation='identity'
            rules[name]={'sources':names,'operation':operation}
        mapping={'schema':'explicit-flax-tensor-map-v1','tensors':rules,'ignored_sources':[]}
        with tempfile.TemporaryDirectory() as tmp:
            output=_fk_paths.relocate((Path(tmp)/'converted'), __file__)
            metadata=convert_factorized_arrays(arrays,mapping,config,output,source='synthetic-reference-arrays',
                                                revision='synthetic-v1',source_checkpoint_sha256='1'*64)
            self.assertEqual(metadata['weights_status'],'converted_unverified')
            with self.assertRaisesRegex(ValueError,'pretrained'):
                FactorizedViViT.from_local(output,_fk_paths.relocate((output/'artifact.json'), __file__))
            reloaded=FactorizedViViT.from_local(output,_fk_paths.relocate((output/'artifact.json'), __file__),require_pretrained=False).eval()
            pixels=torch.randn(1,4,3,16,16)
            torch.testing.assert_close(model(pixels),reloaded(pixels),rtol=0,atol=0)
            invalid=copy.deepcopy(mapping);invalid['tensors'].pop(next(iter(rules)))
            with self.assertRaisesRegex(ValueError,'cover every target'):
                convert_factorized_arrays(arrays,invalid,config,_fk_paths.relocate((Path(tmp)/'bad'), __file__),source='fixture',
                                           revision='fixture',source_checkpoint_sha256='1'*64)

    def media(self,root):
        path=_fk_paths.relocate((root/'generated.mkv'), __file__);ffmpeg=media_tool('ffmpeg')
        subprocess.run([str(ffmpeg),'-v','error','-f','lavfi','-i','testsrc=size=96x96:rate=25:duration=3',
                        '-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=3',
                        '-c:v','ffv1','-pix_fmt','bgr0','-c:a','pcm_s16le','-threads','1',str(path)],
                       check=True,capture_output=True,timeout=30)
        return path

    def test_media_derivatives_preserve_original_and_offset_label_semantics(self):
        from deepfake_detection.robustness import transcode_media
        from deepfake_detection.forensics import sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__);path=self.media(root);before=sha256_file(path)
            for index,(kind,strength) in enumerate((('video_compression',28),('video_blur',1.),
                                                    ('video_downsample',.5),('audio_compression',64),('av_offset',.16))):
                metadata=transcode_media(path,_fk_paths.relocate((root/f'derivative-{index}'), __file__),kind,strength)
                self.assertEqual(metadata['source_sha256'],before)
                self.assertTrue(_fk_paths.relocate((Path(metadata['derivative'])), __file__).is_file())
                if kind=='av_offset':self.assertIn('do_not_inherit',metadata['label_policy'])
            self.assertEqual(sha256_file(path),before)

    def test_generated_video_to_float32_model_to_verified_forensic_record(self):
        from deepfake_detection.inference import run_fk_video
        from deepfake_detection.model import EvidenceModel
        from deepfake_detection.forensics import verify,commitment
        class Detector:
            method='SYNTHETIC_FIXED_LANDMARKS_NOT_A_LEARNED_DETECTOR'
            def __call__(self,rgb):
                return [{'box':[10,10,86,86],'score':1.,'landmarks':[[30,30],[66,30],[48,48],[36,65],[60,65]]}]
        class Encoders(nn.Module):
            def __init__(self):
                super().__init__();self.visual=nn.Linear(3,4);self.audio=nn.Linear(1,6);self.calls=0
            def encode_window(self,faces,waveform,times):
                self.calls+=1
                return {'visual_features':self.visual(faces.mean((-2,-1))).unsqueeze(0),
                        'audio_features':self.audio(waveform.reshape(100,320).mean(-1,keepdim=True)).unsqueeze(0),
                        'visual_times':times,'audio_times':(torch.arange(100)+.5)/50}
        torch.manual_seed(7)
        model=EvidenceModel(visual_dim=4,audio_dim=6,hidden=8,dimension=8)
        encoders=Encoders()
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__);path=self.media(root)
            result=run_fk_video(path,3.,model,encoders,Detector(),lambda samples:[(0.,3.)],
                                lambda measurements:torch.tensor([[measurements['face_coverage'],measurements['speech_coverage']]]),
                                _fk_paths.relocate((root/'inference'), __file__),model_version='SYNTHETIC_UNTRAINED_FIXTURE',
                                quality_profile='SYNTHETIC_AVAILABILITY_SPEECH',provider_status='synthetic')
            record=load_json(result['record'])
            self.assertEqual(encoders.calls,2)
            self.assertEqual(record['preprocessing']['provider_status'],'synthetic')
            self.assertTrue(result['verification']['valid'])
            self.assertEqual(record['schema_version'],'fk-record-v2')
            self.assertEqual(result['verification']['derived_artifacts_checked'],8)
            self.assertEqual(record['source_metadata']['av_start_offset_seconds'],0.)
            self.assertTrue((_fk_paths.relocate((root/'inference'/record['windows'][0]['representative_frame']), __file__)).is_file())
            self.assertEqual(result['verification']['max_absolute_error'],0.)
            self.assertEqual(record['windows'][-1]['status'],'UNASSESSED')
            self.assertLessEqual(record['preprocessing']['numeric_export']['maximum_model_rounding_difference'],5e-6)
            trusted=commitment(record);record['windows'][0]['bins'][0]['contributions'][0]+=.1
            with self.assertRaisesRegex(ValueError,'commitment mismatch'):verify(record,trusted)
