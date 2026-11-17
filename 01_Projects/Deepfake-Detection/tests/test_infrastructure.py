"""Synthetic checks for backbone, cache, timing, tracking, and training contracts."""

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

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]


class InfrastructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_factorized_encoder_real_forward_backward_and_strict_reload(self):
        from safetensors.torch import save_file
        from deepfake_detection.backbones import FactorizedViViT
        from deepfake_detection.forensics import sha256_file
        config = dict(image_size=16, frames=4, tubelet=[2, 8, 8], dimension=16,
                      spatial_depth=2, temporal_depth=1, heads=2, mlp_ratio=2,
                      dropout=0., layer_norm_epsilon=1e-6)
        from deepfake_detection.backbones import prepare_vivit
        config.update(input_normalization='rgb_unit_interval_to_minus_one_plus_one',resize='bilinear_stretch_antialias_true')
        red=np.zeros((4,12,20,3),dtype=np.uint8);red[...,0]=255
        normalized=prepare_vivit(red,config)
        self.assertEqual(tuple(normalized.shape),(1,4,3,16,16))
        torch.testing.assert_close(normalized[:,:,0],torch.ones_like(normalized[:,:,0]))
        torch.testing.assert_close(normalized[:,:,1:],-torch.ones_like(normalized[:,:,1:]))
        model = FactorizedViViT(config)
        pixels = torch.randn(2, 4, 3, 16, 16, requires_grad=True)
        output = model(pixels)
        self.assertEqual(tuple(output.shape), (2, 16))
        output.square().mean().backward()
        self.assertGreater(pixels.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(torch.equal(model.spatial_encoder.layers[0].self_attn.in_proj_weight,
                                     model.spatial_encoder.layers[1].self_attn.in_proj_weight))
        with tempfile.TemporaryDirectory() as tmp:
            root = _fk_paths.relocate((Path(tmp)), __file__)
            (_fk_paths.relocate((root/'config.json'), __file__)).write_text(json.dumps(config))
            save_file(model.state_dict(), str(_fk_paths.relocate((root/'model.safetensors'), __file__)))
            manifest = {'schema':'model-artifact-v1', 'source':'synthetic-test', 'revision':'fixture-v1',
                        'architecture':'factorized-vivit-reconstruction-v1', 'weights_status':'synthetic',
                        'files':{name:sha256_file(_fk_paths.relocate((root/name), __file__)) for name in ('config.json','model.safetensors')}}
            with self.assertRaisesRegex(ValueError,'pretrained'):
                FactorizedViViT.from_local(root, manifest)
            reloaded = FactorizedViViT.from_local(root, manifest, require_pretrained=False)
            torch.testing.assert_close(model(pixels), reloaded(pixels))
            (_fk_paths.relocate((root/'config.json'), __file__)).write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                FactorizedViViT.from_local(root, manifest, require_pretrained=False)

    def test_ast_actual_transformer_tokens_and_local_checkpoint(self):
        from transformers import ASTConfig, ASTModel, ASTFeatureExtractor
        from deepfake_detection.backbones import ASTBackend
        from deepfake_detection.forensics import sha256_file
        config = ASTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=32, max_length=32, num_mel_bins=32,
                           patch_size=16, frequency_stride=10, time_stride=10)
        model = ASTModel(config).eval()
        inputs = torch.randn(2,32,32)
        sequence = model(inputs).last_hidden_state
        torch.testing.assert_close(ASTBackend(model,token='cls')(inputs),sequence[:,0])
        torch.testing.assert_close(ASTBackend(model,token='mean_cls_distillation')(inputs),sequence[:,:2].mean(1))
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__)
            model.save_pretrained(root, safe_serialization=True)
            ASTFeatureExtractor(max_length=32,num_mel_bins=32).save_pretrained(root)
            manifest={'schema':'model-artifact-v1','source':'synthetic-test','revision':'fixture-v1',
                      'architecture':'transformers-ast-base','weights_status':'synthetic',
                      'files':{p.name:sha256_file(p) for p in root.iterdir() if p.is_file()}}
            reloaded=ASTBackend.from_local(root,manifest,token='cls',require_pretrained=False).eval()
            torch.testing.assert_close(reloaded(inputs),sequence[:,0])

    def test_avhubert_adapter_keeps_gradients_and_rejects_misalignment(self):
        from deepfake_detection.backbones import AVHubertBackend
        class OfficialInterfaceFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.project=nn.Linear(4,768)
            def extract_finetune(self, source, padding_mask, mask):
                return self.project(source['audio'].transpose(1,2)), None
        adapter=AVHubertBackend(OfficialInterfaceFixture())
        audio=torch.randn(2,4,5,requires_grad=True)
        source={'audio':audio,'video':torch.randn(2,1,5,8,8)}
        adapter(source).square().mean().backward()
        self.assertGreater(audio.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):
            adapter({'audio':audio,'video':torch.randn(2,1,6,8,8)})

    def test_feature_centers_match_wav2vec_receptive_field(self):
        from deepfake_detection.encoders import convolution_centers
        centers=convolution_centers(32000,[10,3,3,3,3,2,2],[5,2,2,2,2,2,2],16000)
        self.assertEqual(len(centers),99)
        self.assertAlmostEqual(centers[0].item(),199.5/16000,places=7)
        torch.testing.assert_close(centers[1:]-centers[:-1],torch.full((98,),.02))
        with self.assertRaises(ValueError):
            convolution_centers(100,[10,3,3,3,3,2,2],[5,2,2,2,2,2,2],16000)

    def test_content_cache_rejects_stale_provenance_and_tampering(self):
        from deepfake_detection.feature_cache import write_cache,read_cache
        provenance={'source_sha256':'1'*64,'preprocessing_sha256':'2'*64,
                    'encoder_sha256':{'visual':'3'*64,'audio':'4'*64},
                    'feature_layers':['visual_avgpool','audio_last_hidden'],
                    'window_intervals':[[0.,2.]],'precision':'float32'}
        with tempfile.TemporaryDirectory() as tmp:
            directory=_fk_paths.relocate((Path(tmp)/'cache'), __file__)
            tensor=torch.randn(1,50,4)
            write_cache(directory,{'visual_features':tensor},provenance)
            torch.testing.assert_close(read_cache(directory,provenance)['visual_features'],tensor)
            changed=copy.deepcopy(provenance); changed['precision']='float16'
            with self.assertRaisesRegex(ValueError,'provenance'):
                read_cache(directory,changed)
            payload=_fk_paths.relocate((directory/'features.safetensors'), __file__)
            content=bytearray(payload.read_bytes()); content[-1]^=1;payload.write_bytes(content)
            with self.assertRaisesRegex(ValueError,'content hash'):
                read_cache(directory,provenance)

    def test_generated_media_preserves_audio_offset_and_sampling(self):
        from deepfake_detection.media import aligned_audio,sample_video,probe
        ffmpeg=media_tool('ffmpeg')
        self.assertTrue(ffmpeg.is_file(),'run scripts/setup_environment.sh to install FFmpeg')
        with tempfile.TemporaryDirectory() as tmp:
            path=_fk_paths.relocate((Path(tmp)/'offset.mkv'), __file__)
            subprocess.run([str(ffmpeg),'-v','error','-f','lavfi','-i','testsrc=size=96x96:rate=25:duration=2',
                            '-itsoffset','0.25','-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=1.75',
                            '-c:v','ffv1','-pix_fmt','bgr0','-c:a','pcm_s16le','-threads','1',str(path)],
                           check=True,capture_output=True,timeout=30)
            info=probe(path)
            self.assertEqual(len(info['streams']),2)
            frames=sample_video(path,0,1,fps=25)
            self.assertEqual(frames['rgb'].shape,(25,96,96,3))
            self.assertEqual(frames['duplicate_samples'],0)
            audio=aligned_audio(path,0,.5,require_complete=False)
            self.assertFalse(audio['present'][:3900].any())
            self.assertTrue(audio['present'][4100:].all())
            with self.assertRaisesRegex(ValueError,'fully cover'):
                aligned_audio(path,0,.5)
            complete=aligned_audio(path,.5,1)
            self.assertEqual(complete['coverage'],1.)
            self.assertGreater(float(np.abs(complete['samples']).max()),0)

    def test_face_track_gaps_and_geometric_crops(self):
        from deepfake_detection.faces import aligned_crops,track_and_crop
        image=np.full((120,120,3),127,dtype=np.uint8)
        detection={'box':[20,20,100,100],'score':1.,'landmarks':[[40,45],[80,45],[60,60],[45,75],[75,75]]}
        face,lips=aligned_crops(image,detection)
        self.assertEqual(face.shape,(224,224,3));self.assertEqual(lips.shape,(96,96,3))
        class FixtureDetector:
            method='SYNTHETIC_NOT_A_LEARNED_DETECTOR'
            def __init__(self):self.count=0
            def __call__(self,rgb):
                self.count+=1
                return [] if self.count==2 else [detection]
        records=track_and_crop([image]*3,FixtureDetector())
        self.assertEqual([r['status'] for r in records],['USABLE','UNASSESSED','USABLE'])
        self.assertNotIn('face',records[1])

    def test_boundary_face_crop_preserves_observed_pixels(self):
        from deepfake_detection.faces import aligned_crops

        image = np.full((100, 100, 3), 127, dtype=np.uint8)
        detection = {'box': [-10, 10, 90, 90],
                     'landmarks': [[20, 35], [55, 35], [40, 51], [30, 67], [50, 67]]}
        with self.assertRaisesRegex(ValueError, 'extends beyond observed source pixels'):
            aligned_crops(image, detection)
        face, lips = aligned_crops(image, detection, fit_to_frame=True)
        self.assertEqual(face.shape, (224, 224, 3))
        self.assertEqual(lips.shape, (96, 96, 3))
        self.assertTrue((face == 127).all())
        self.assertTrue((lips == 127).all())

        detection['landmarks'][2] = [-1, 51]
        with self.assertRaisesRegex(ValueError, 'cannot retain facial landmarks'):
            aligned_crops(image, detection, fit_to_frame=True)

    def test_speech_intervals_minimum_merge_padding_and_clipping(self):
        from deepfake_detection.vad import speech_intervals
        probabilities=[1.]*3+[0.]*2+[1.]*3+[0.]*2
        self.assertEqual(speech_intervals(probabilities,.1,1.),[(0.,.9)])
        self.assertEqual(speech_intervals([1.,0.,0.,0.],.1,.4),[])
        with self.assertRaises(ValueError):speech_intervals([1.],.1,1.)

    def test_avtenet_stage_training_preserves_unknown_targets(self):
        from deepfake_detection.avtenet import VectorExpert,load_component_profile
        from deepfake_detection.avtenet_training import fit_avtenet_stage
        profile=load_component_profile(_fk_paths.relocate((ROOT/'configs/avtenet.components-v1.json'), __file__))
        schedule=load_json(_fk_paths.relocate((ROOT/'configs/avtenet.training-v1.json'), __file__))
        def rows(split):
            return [dict(video_id=f'{split}-{index}',source_family=f'{split}-family-{index}',
                         source_sha256=str(index+(1 if split=='train' else 4))*64,
                         split=split,label=0 if index==0 else 1,visual_label=None if index==2 else index,
                         audio_label=0 if index==0 else 1,inputs=torch.randn(1,3)) for index in range(3)]
        model=VectorExpert(nn.Linear(3,768),768)
        with tempfile.TemporaryDirectory() as tmp:
            result=fit_avtenet_stage(model,rows('train'),rows('validation'),profile,schedule,'visual',_fk_paths.relocate((Path(tmp)/'stage'), __file__),epochs=1)
            self.assertEqual(result['history'][0]['training']['supervised_inputs'],2)
            self.assertEqual(result['history'][0]['validation']['unknown_targets'],1)

    def test_frozen_encoders_return_trainable_head_compatible_features(self):
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        from deepfake_detection.encoders import FrozenFKEncoders
        from deepfake_detection.model import EvidenceModel
        class VisualFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.features=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(3,4,1))
        config=Wav2Vec2Config(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                             intermediate_size=32,conv_dim=[8]*7,conv_kernel=[10,3,3,3,3,2,2],
                             conv_stride=[5,2,2,2,2,2,2],num_conv_pos_embeddings=8,num_conv_pos_embedding_groups=2)
        encoders=FrozenFKEncoders(VisualFixture(),Wav2Vec2Model(config),image_mean=[.5]*3,image_std=[.5]*3)
        encoders.train()
        self.assertFalse(encoders.training)
        features=encoders.encode_window(torch.rand(50,3,224,224),torch.randn(32000)*.1,torch.arange(50)/25)
        self.assertEqual(tuple(features['audio_features'].shape),(1,99,16))
        self.assertTrue(all(not tensor.is_inference() and not tensor.requires_grad for tensor in features.values()))
        model=EvidenceModel(visual_dim=4,audio_dim=16,hidden=8,dimension=8)
        output=model(**features,quality=torch.ones(1,2))
        output['window_scores'].sum().backward()
        self.assertTrue(all(p.grad is None for p in encoders.parameters()))
        self.assertIsNotNone(model.gru.weight_ih_l0.grad)

    def test_preparation_pipeline_generated_media_and_missing_speech(self):
        from deepfake_detection.preparation import prepare_fk_window,save_prepared_window
        class DetectorFixture:
            method='SYNTHETIC_DETECTOR_NOT_A_LEARNED_MODEL'
            def __call__(self,image):
                return [{'box':[10,10,86,86],'score':1.,
                         'landmarks':[[30,30],[66,30],[48,48],[36,65],[60,65]]}]
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__)
            path=_fk_paths.relocate((root/'aligned.mkv'), __file__)
            ffmpeg=media_tool('ffmpeg')
            subprocess.run([str(ffmpeg),'-v','error','-f','lavfi','-i','testsrc=size=96x96:rate=25:duration=2',
                            '-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=2',
                            '-c:v','ffv1','-pix_fmt','bgr0','-c:a','pcm_s16le','-threads','1',str(path)],
                           check=True,capture_output=True,timeout=30)
            prepared=prepare_fk_window(path,0,2.,DetectorFixture(),[(0.,2.)])
            self.assertEqual(prepared['status'],'USABLE',prepared['reasons'])
            self.assertEqual(prepared['faces'].shape,(50,224,224,3))
            self.assertEqual(prepared['audio'].shape,(32000,))
            metadata=save_prepared_window(prepared,_fk_paths.relocate((root/'prepared'), __file__))
            self.assertIn('prepared.npz',metadata['derived_sha256'])
            rejected=prepare_fk_window(path,0,2.,DetectorFixture(),[])
            self.assertEqual(rejected['status'],'UNASSESSED')
            self.assertIn('insufficient_speech',rejected['reasons'])
            short=prepare_fk_window(path,1.,2.,DetectorFixture(),[(0.,2.)])
            self.assertIn('incomplete_window',short['reasons'])


class FKTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads=torch.get_num_threads();torch.set_num_threads(2)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.previous_threads)

    def model(self):
        from deepfake_detection.model import EvidenceModel
        torch.manual_seed(13)
        return EvidenceModel(visual_dim=4,audio_dim=6,hidden=8,dimension=8)

    def rows(self,split):
        torch.manual_seed(5 if split=='train' else 6)
        return [dict(video_id=f'{split}-{label}',source_family=f'{split}-family-{label}',
                     source_sha256=(str(label+1) if split=='train' else str(label+3))*64,
                     split=split,label=label,temporal_complete=True,
                     manipulated_intervals=[] if label==0 else [[.2,.4]],window_intervals=[[0.,2.],[1.,3.]],
                     visual_features=torch.randn(2,50,4),audio_features=torch.randn(2,100,6),
                     visual_times=torch.arange(50)/25,audio_times=torch.arange(100)/50,
                     quality=torch.ones(2,2)) for label in (0,1)]

    def test_training_resume_matches_continuous_and_selects_validation_only(self):
        from deepfake_detection.training import fit_fk,load_fk_checkpoint,select_video_threshold
        config=load_json(_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__))
        train,val=self.rows('train'),self.rows('validation')
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__)
            continuous=self.model(); fit_fk(continuous,train,val,config,_fk_paths.relocate((root/'continuous'), __file__),epochs=2)
            first=self.model();fit_fk(first,train,val,config,_fk_paths.relocate((root/'first'), __file__),epochs=1)
            resumed=self.model();fit_fk(resumed,train,val,config,_fk_paths.relocate((root/'resumed'), __file__),epochs=2,resume_from=_fk_paths.relocate((root/'first/last.pt'), __file__))
            full=torch.load(_fk_paths.relocate_value((_fk_paths.relocate((root/'continuous/last.pt'), __file__)), __file__),weights_only=True)
            resumed_state=torch.load(_fk_paths.relocate_value((_fk_paths.relocate((root/'resumed/last.pt'), __file__)), __file__),weights_only=True)
            for key,value in full['model_state'].items():
                torch.testing.assert_close(value,resumed_state['model_state'][key],rtol=0,atol=0)
            loaded,metadata=load_fk_checkpoint(_fk_paths.relocate((root/'resumed/best.pt'), __file__))
            for key,value in loaded.state_dict().items():
                torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(select_video_threshold(loaded,val)['split'],'validation')
            bad=copy.deepcopy(val);bad[0]['split']='test'
            with self.assertRaisesRegex(ValueError,'test inference'):
                select_video_threshold(loaded,bad)
            changed=copy.deepcopy(val);changed[0]['quality'][0,0]=.1
            with self.assertRaisesRegex(ValueError,'changed'):
                fit_fk(self.model(),train,changed,config,_fk_paths.relocate((root/'invalid'), __file__),epochs=2,resume_from=_fk_paths.relocate((root/'first/last.pt'), __file__))

    def test_training_rejects_family_and_hash_leakage(self):
        from deepfake_detection.training import fit_fk
        config=load_json(_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__))
        train,val=self.rows('train'),self.rows('validation')
        for key in ('source_family','source_sha256'):
            bad=copy.deepcopy(val);bad[0][key]=train[0][key]
            with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'leakage'):
                fit_fk(self.model(),train,bad,config,_fk_paths.relocate((Path(tmp)/'bad'), __file__),epochs=1)

    def test_trusted_cache_manifest_training_and_calibration(self):
        from deepfake_detection.feature_cache import write_cache
        from deepfake_detection.forensics import sha256_file
        from deepfake_detection.infrastructure import train_fk_manifest
        from deepfake_detection.training import TENSORS,load_fk_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=_fk_paths.relocate((Path(tmp)), __file__);manifest={}
            for split in ('train','validation'):
                manifest[split]=[]
                for row in self.rows(split):
                    cache=_fk_paths.relocate((root/row['video_id']), __file__)
                    provenance={'source_sha256':row['source_sha256'],'preprocessing_sha256':'a'*64,
                                'encoder_sha256':{'visual':'b'*64,'audio':'c'*64},'feature_layers':['synthetic'],
                                'window_intervals':row['window_intervals'],'precision':'float32'}
                    write_cache(cache,{name:row[name] for name in TENSORS},provenance)
                    entry={key:value for key,value in row.items() if key not in TENSORS}
                    entry.update(cache=cache.name,cache_metadata_sha256=sha256_file(_fk_paths.relocate((cache/'metadata.json'), __file__)),cache_provenance=provenance)
                    manifest[split].append(entry)
            (_fk_paths.relocate((root/'manifest.json'), __file__)).write_text(json.dumps(manifest))
            model_config=load_json(_fk_paths.relocate((ROOT/'configs/ours.provisional.json'), __file__))
            model_config['paper_fixed']['dimension']=8
            model_config['reconstruction_choices'].update(visual_feature_dimension=4,audio_feature_dimension=6,gru_hidden=8)
            (_fk_paths.relocate((root/'model.json'), __file__)).write_text(json.dumps(model_config))
            result=train_fk_manifest(_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((root/'model.json'), __file__),_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__),_fk_paths.relocate((root/'run'), __file__),epochs=1,calibrate=True)
            self.assertEqual(result['validation_selection']['split'],'validation')
            model,_=load_fk_checkpoint(_fk_paths.relocate((root/'run/selected.pt'), __file__))
            self.assertEqual(model.decision.video_threshold,result['validation_selection']['video_threshold'])
            metadata=_fk_paths.relocate((root/manifest['train'][0]['cache']/'metadata.json'), __file__)
            metadata.write_text('{}')
            with self.assertRaisesRegex(ValueError,'trusted training manifest'):
                train_fk_manifest(_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((root/'model.json'), __file__),_fk_paths.relocate((ROOT/'configs/fk.training-v1.json'), __file__),_fk_paths.relocate((root/'tampered'), __file__),epochs=1)
