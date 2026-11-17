"""Comparator workflow contracts; synthetic weights are never pretrained evidence."""

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
from contextlib import closing
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util

import torch

from baselines.common.io import write,digest
from baselines.common.population import prepare,Population,target,windows
from baselines.common.training import fit,seed_all,load_checkpoint
from baselines.common.metrics import metrics,compare


def row(vid,split,**updates):
    result={'video_id':vid,'path':vid+'.mp4','dataset':'AV-Deepfake1M','release':'test',
            'source_family':'shared','source_sha256':'a'*64,'source_overlap_policy':'keep-official',
            'split':split,'duration':3.25,'label':1,'visual_label':1,'audio_label':0,
            'temporal_complete':True,'manipulated_intervals':[[1.,1.5]],
            'visual_manipulated_intervals':[[1.,1.5]],'audio_manipulated_intervals':[]}
    return {**result,**updates}


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_population_portable_and_keep_official(self):
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            write(_fk_paths.relocate((root/'manifest.json'), __file__),[row('a','train'),row('b','validation')])
            result=prepare(_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((root/'index'), __file__),'avfakenet')
            self.assertEqual(result['counts'],{'train':1,'validation':1})
            self.assertEqual(result['population_audit']['cross_split_source_families'],1)
            population=Population(_fk_paths.relocate((root/'index'), __file__),root,'avfakenet')
            self.assertEqual(next(population.rows('validation'))['video_id'],'b')
            with (_fk_paths.relocate((root/'index/index.sqlite'), __file__)).open('ab') as stream:stream.write(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):Population(_fk_paths.relocate((root/'index'), __file__),root,'avfakenet')

    def test_strict_policy_duplicate_id_and_unsafe_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            for name,records in (
                ('strict',[row('a','train',source_overlap_policy='error'),row('b','validation',source_overlap_policy='error')]),
                ('duplicate',[row('a','train'),row('a','validation')]),
                ('path',[row('a','train',path='../outside.mp4')])):
                write(_fk_paths.relocate((root/(name+'.json')), __file__),records)
                with self.assertRaises(ValueError):prepare(_fk_paths.relocate((root/(name+'.json')), __file__),_fk_paths.relocate((root/name), __file__),'avtenet')

    def test_partial_edit_targets_and_windows(self):
        value=row('a','train')
        self.assertEqual(windows(3.25),[(0.,2.),(1.,3.),(1.25,3.25)])
        self.assertEqual(windows(1.99),[])
        self.assertEqual(target(value,'visual',1.),1)
        self.assertEqual(target(value,'visual',1.5),0)
        self.assertEqual(target(value,'visual',0.,1.),0)
        self.assertEqual(target(value,'audio',0.,2.),0)
        value['visual_label']=None
        self.assertIsNone(target(value,'visual',1.1))
        value['temporal_complete']=False
        self.assertIsNone(target(value,'audiovisual',0.,2.))

    @unittest.skipUnless(importlib.util.find_spec('python_speech_features'),'optional AV-HuBERT preparation dependency')
    def test_real_logfbank_api_and_alignment(self):
        import numpy as np
        from baselines.common.io import read
        from deepfake_detection.avhubert_preparation import prepare_avhubert
        settings=read(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'baselines/avtenet/configs/assets.example.json'), __file__))['experts']['audiovisual']['preparation']
        waveform=(.1*np.sin(np.arange(32000)*2*np.pi*220/16000)).astype(np.float32)
        values=prepare_avhubert(waveform,np.full((50,96,96),128,dtype=np.uint8),settings)
        self.assertEqual(tuple(values['audio'].shape),(1,104,50))
        self.assertEqual(tuple(values['video'].shape),(1,1,50,88,88))
        self.assertTrue(torch.isfinite(values['audio']).all())
        with self.assertRaisesRegex(ValueError,'aligned'):
            prepare_avhubert(waveform[:-1000],np.zeros((50,96,96),dtype=np.uint8),settings)

    def test_coverage_and_hard_decision_metrics(self):
        predictions=[{'video_id':'a','prediction':1,'label':1,'score':None},
                     {'video_id':'b','prediction':0,'label':0,'score':None},
                     {'video_id':'c','prediction':None,'label':1,'score':None}]
        report=metrics(predictions)
        self.assertEqual(report['coverage'],2/3)
        self.assertEqual(report['balanced_accuracy'],1.)
        self.assertIsNone(report['roc_auc'])
        self.assertEqual(compare(predictions,predictions)['common_assessed'],2)

    def test_exact_cpu_resume_and_identity_rejection(self):
        from baselines.avtenet.model import native_binary_loss
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(2,1);self.dropout=torch.nn.Dropout(.2)
            def forward(self,x):return self.linear(self.dropout(x)).squeeze(-1)
        def batches(split):
            yield torch.tensor([[.2,.7],[.4,.8]]),torch.tensor([0,1])
        schedule={'optimizer':'Adam','learning_rate':.01,'weight_decay':0.,'betas':[.9,.999],
                  'epsilon':1e-8,'maximum_epochs':3,'patience':5,'selection':'loss'}
        options={'identity':{'model':'fixture','index':'fixed'},'schedule':schedule,'loss_fn':native_binary_loss,
                 'fake_predictions':lambda logits:(logits<0).long()}
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            seed_all(7);fit(Tiny(),batches,_fk_paths.relocate((root/'full'), __file__),epochs=2,**options)
            seed_all(7);fit(Tiny(),batches,_fk_paths.relocate((root/'one'), __file__),epochs=1,**options)
            fit(Tiny(),batches,_fk_paths.relocate((root/'resume'), __file__),epochs=2,resume=_fk_paths.relocate((root/'one/last.pt'), __file__),**options)
            full=load_checkpoint(_fk_paths.relocate((root/'full/last.pt'), __file__));resumed=load_checkpoint(_fk_paths.relocate((root/'resume/last.pt'), __file__))
            for key,value in full['model_state'].items():self.assertTrue(torch.equal(value,resumed['model_state'][key]))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                fit(Tiny(),batches,_fk_paths.relocate((root/'wrong'), __file__),epochs=2,resume=_fk_paths.relocate((root/'one/last.pt'), __file__),
                    **{**options,'identity':{'model':'different'}})

    def test_fusion_merge_alignment_and_freezing(self):
        from types import SimpleNamespace
        from baselines.avtenet.pipeline import merge_features,open_cache
        import sqlite3
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            for stage in ('visual','audio','audiovisual'):
                directory=_fk_paths.relocate((root/stage), __file__);directory.mkdir()
                with closing(sqlite3.connect(_fk_paths.relocate((directory/'features.sqlite'), __file__))) as conn, conn:
                    conn.execute('CREATE TABLE features (video_id TEXT,start REAL,end REAL,split TEXT,source_sha256 TEXT,features BLOB,PRIMARY KEY(video_id,start,end))')
                    conn.execute('INSERT INTO features VALUES (?,?,?,?,?,?)',('a',0.,2.,'train','a'*64,np.ones(768,dtype='<f4').tobytes()))
                write(_fk_paths.relocate((directory/'cache.json'), __file__),{'schema':'avtenet-expert-cache-v1','stage':stage,'index_sha256':'b'*64,
                      'profile_sha256':'c'*64,'preparation_sha256':'d'*64,'checkpoint_sha256':stage,
                      'integration_only':True,'payload_sha256':digest(_fk_paths.relocate((directory/'features.sqlite'), __file__))})
            args=SimpleNamespace(**{s:str(_fk_paths.relocate((root/s), __file__)) for s in ('visual','audio','audiovisual')},output=str(_fk_paths.relocate((root/'fusion'), __file__)))
            report=merge_features(args)
            self.assertEqual(report['windows'],1)
            self.assertTrue(report['integration_only'])
            meta,conn=open_cache(_fk_paths.relocate((root/'fusion'), __file__))
            self.assertEqual(len(conn.execute('SELECT features FROM features').fetchone()[0]),3*768*4)
            conn.close()

    def test_complete_fusion_workflow_keeps_unassessed(self):
        from types import SimpleNamespace
        import sqlite3
        import numpy as np
        from baselines.common.io import fingerprint
        from baselines.avtenet.pipeline import CONFIG,PREPARATION,merge_features,train,evaluate
        from baselines.avtenet.model import load_component_profile
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            records=[row('a','train'),row('b','validation'),row('c','validation')]
            write(_fk_paths.relocate((root/'manifest.json'), __file__),records)
            indexed=prepare(_fk_paths.relocate((root/'manifest.json'), __file__),_fk_paths.relocate((root/'index'), __file__),'avtenet')
            population=Population(_fk_paths.relocate((root/'index'), __file__),root,'avtenet')
            profile=load_component_profile(CONFIG)
            for stage in ('visual','audio','audiovisual'):
                directory=_fk_paths.relocate((root/stage), __file__);directory.mkdir()
                with closing(sqlite3.connect(_fk_paths.relocate((directory/'features.sqlite'), __file__))) as conn,conn:
                    conn.execute('CREATE TABLE features (video_id TEXT,start REAL,end REAL,split TEXT,source_sha256 TEXT,features BLOB,PRIMARY KEY(video_id,start,end))')
                    for record in records[:2]:
                        conn.execute('INSERT INTO features VALUES (?,?,?,?,?,?)',(record['video_id'],0.,2.,record['split'],'a'*64,np.ones(768,dtype='<f4').tobytes()))
                write(_fk_paths.relocate((directory/'cache.json'), __file__),{'schema':'avtenet-expert-cache-v1','stage':stage,'index_sha256':indexed['index_sha256'],
                      'profile_sha256':fingerprint(profile),'preparation_sha256':fingerprint(PREPARATION),
                      'checkpoint_sha256':stage,'integration_only':True,'payload_sha256':digest(_fk_paths.relocate((directory/'features.sqlite'), __file__))})
            merge_features(SimpleNamespace(**{s:str(_fk_paths.relocate((root/s), __file__)) for s in ('visual','audio','audiovisual')},output=str(_fk_paths.relocate((root/'features'), __file__))))
            args=SimpleNamespace(config=None,schedule=None,stage='fusion',seed=7,feature_cache=str(_fk_paths.relocate((root/'features'), __file__)),
                                 output=str(_fk_paths.relocate((root/'trained'), __file__)),device='cpu',epochs=1,resume=None,max_steps=1)
            train(args,population)
            args.output=str(_fk_paths.relocate((root/'evaluation'), __file__));args.checkpoint=str(_fk_paths.relocate((root/'trained/best.pt'), __file__));args.split='validation';args.compare_with=None
            result=evaluate(args,population)
            self.assertEqual(result['metrics']['coverage'],.5)
            self.assertTrue(result['integration_only'])


if __name__=='__main__':unittest.main()
