"""Partial feature resume preserves committed bytes and validates provenance."""

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

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.common.io import write,read,digest,fingerprint
from baselines.common.population import windows
from baselines.avtenet.feature_resume import validate_snapshot
from baselines.avtenet import pipeline


class Population:
    metadata={'index_sha256':'i'*64}
    def rows(self,split):
        for vid,kind in (('a','train'),('b','validation')):
            if kind==split:yield {'video_id':vid,'split':kind,'source_sha256':vid*64,'duration':3.25}
    def path(self,row):return _fk_paths.relocate((Path(row['video_id'])), __file__)


class ResumeTests(unittest.TestCase):
    def test_owned_snapshot_preserves_source_and_rejects_changed_producer(self):
        from baselines.avtenet.feature_resume import snapshot_owned
        from baselines.common import training
        with tempfile.TemporaryDirectory() as folder:
            root=_fk_paths.relocate((Path(folder)), __file__)
            source,checkpoint,identity,expected=self.fixture(root)
            identity={**identity,'profile':pipeline.load_component_profile(pipeline.CONFIG),
                      'preparation':pipeline.PREPARATION}
            producer={**expected,'identity':identity,'schema':'avtenet-owned-feature-producer-v1'}
            write(_fk_paths.relocate((source/'producer.json'), __file__),producer)
            before=digest(_fk_paths.relocate((source/'features.sqlite'), __file__))
            with patch.object(training,'load_checkpoint',return_value={'identity':identity}):
                snapshot_owned(source,_fk_paths.relocate((root/'copy'), __file__),Population(),checkpoint,'visual')
                _,keys,failures=validate_snapshot(_fk_paths.relocate((root/'copy'), __file__),{**expected,'identity':identity},Population())
                self.assertEqual(len(keys),1);self.assertEqual(len(failures),1)
                producer['checkpoint_sha256']='wrong';write(_fk_paths.relocate((source/'producer.json'), __file__),producer,replace=True)
                with self.assertRaisesRegex(ValueError,'producer differs'):
                    snapshot_owned(source,_fk_paths.relocate((root/'bad-copy'), __file__),Population(),checkpoint,'visual')
            self.assertEqual(digest(_fk_paths.relocate((source/'features.sqlite'), __file__)),before)

    def fixture(self,root):
        checkpoint=_fk_paths.relocate((root/'best.pt'), __file__);checkpoint.write_bytes(b'selected checkpoint')
        identity={'stage':'visual','seed':7,'max_steps':None}
        expected={'stage':'visual','index_sha256':'i'*64,
            'profile_sha256':fingerprint(pipeline.load_component_profile(pipeline.CONFIG)),
            'preparation_sha256':fingerprint(pipeline.PREPARATION),
            'checkpoint_sha256':digest(checkpoint),'identity':identity}
        source=_fk_paths.relocate((root/'snapshot'), __file__);source.mkdir()
        with closing(sqlite3.connect(_fk_paths.relocate((source/'features.sqlite'), __file__))) as conn,conn:
            conn.execute('CREATE TABLE features(video_id TEXT,start REAL,end REAL,split TEXT,source_sha256 TEXT,features BLOB,PRIMARY KEY(video_id,start,end))')
            conn.execute('INSERT INTO features VALUES(?,?,?,?,?,?)',('a',0.,2.,'train','a'*64,np.ones(768,dtype='<f4').tobytes()))
        (_fk_paths.relocate((source/'preparation-events.jsonl'), __file__)).write_text('{"video_id":"a","split":"train","start":1.0,"end":3.0,"reason":"preserved failure"}\n')
        metadata={**expected,'schema':'avtenet-attested-partial-features-v1','windows':1,
            'payload_sha256':digest(_fk_paths.relocate((source/'features.sqlite'), __file__)),'events_sha256':digest(_fk_paths.relocate((source/'preparation-events.jsonl'), __file__))}
        write(_fk_paths.relocate((source/'resume.json'), __file__),metadata)
        return source,checkpoint,identity,expected

    def test_identity_and_payload_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,_,_,expected=self.fixture(_fk_paths.relocate((Path(folder)), __file__))
            _,keys,failures=validate_snapshot(source,expected,Population())
            self.assertEqual(keys,{('a',0.,2.)});self.assertEqual(failures,{('a',1.,3.)})
            with self.assertRaisesRegex(ValueError,'mismatch'):
                validate_snapshot(source,{**expected,'checkpoint_sha256':'wrong'},Population())
            with (_fk_paths.relocate((source/'features.sqlite'), __file__)).open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):validate_snapshot(source,expected,Population())

    def test_rehashed_wrong_source_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,_,_,expected=self.fixture(_fk_paths.relocate((Path(folder)), __file__))
            with closing(sqlite3.connect(_fk_paths.relocate((source/'features.sqlite'), __file__))) as conn,conn:
                conn.execute('UPDATE features SET source_sha256=?',('other',))
            metadata=read(_fk_paths.relocate((source/'resume.json'), __file__));metadata['payload_sha256']=digest(_fk_paths.relocate((source/'features.sqlite'), __file__));write(_fk_paths.relocate((source/'resume.json'), __file__),metadata,replace=True)
            with self.assertRaisesRegex(ValueError,'frozen population'):validate_snapshot(source,expected,Population())

    def run_resume(self,root,*,parity_error=False):
        source,checkpoint,identity,_=self.fixture(root)
        before=digest(_fk_paths.relocate((source/'features.sqlite'), __file__))
        class Model(torch.nn.Module):
            def extract_features(self,values):return values.repeat(1,768)+(1 if parity_error else 0)
        class Prepared:
            def __init__(self,*args,**kwargs):pass
            def inputs(self,path,start,row):return torch.tensor([[start+1.]])
            def records(self,split,skip_windows):
                for row in Population().rows(split):
                    for start,end in windows(row['duration']):
                        if (row['video_id'],start,end) not in skip_windows:
                            yield row,start,end,self.inputs(None,start,row),0
        args=SimpleNamespace(config=None,assets='fixture',stage='visual',checkpoint=str(checkpoint),
            device='cpu',output=str(_fk_paths.relocate((root/'resumed'), __file__)),resume_feature_cache=str(source))
        with patch.object(pipeline,'registry',return_value={}),patch.object(pipeline,'stage_model',return_value=(Model(),None,{})),\
             patch.object(pipeline,'detector',return_value=(None,{})),patch.object(pipeline,'load_checkpoint',return_value={'identity':identity,'model_state':{}}),\
             patch.object(pipeline,'stage_identity',return_value={'stage':'visual','seed':7}),patch.object(pipeline,'Prepared',Prepared):
            if parity_error:
                with self.assertRaisesRegex(ValueError,'parity failed'):pipeline.cache_features(args,Population())
            else:
                report=pipeline.cache_features(args,Population())
                self.assertEqual(report['windows'],5)
                with closing(sqlite3.connect(_fk_paths.relocate((root/'resumed/features.sqlite'), __file__))) as conn:
                    blob=conn.execute('SELECT features FROM features WHERE video_id=? AND start=0',('a',)).fetchone()[0]
                    self.assertEqual(blob,np.ones(768,dtype='<f4').tobytes())
                    self.assertEqual(conn.execute('SELECT COUNT(*) FROM features WHERE video_id=? AND start=1',('a',)).fetchone()[0],0)
                self.assertTrue((_fk_paths.relocate((root/'resumed/cache.json'), __file__)).is_file())
            self.assertEqual(digest(_fk_paths.relocate((source/'features.sqlite'), __file__)),before)

    def test_complete_resume_preserves_old_rows_and_skips_recorded_failures(self):
        with tempfile.TemporaryDirectory() as folder:self.run_resume(_fk_paths.relocate((Path(folder)), __file__))

    def test_parity_failure_leaves_original_snapshot_untouched(self):
        with tempfile.TemporaryDirectory() as folder:self.run_resume(_fk_paths.relocate((Path(folder)), __file__),parity_error=True)


if __name__=='__main__':unittest.main()
