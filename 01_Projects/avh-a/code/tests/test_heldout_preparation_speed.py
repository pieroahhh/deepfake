
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

import sqlite3,tempfile,threading,unittest,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from baselines.avtenet.input_cache import InputCache
from baselines.avtenet.heldout_preparation import FastRawCache,FastPrepared,aligned_vectors
from baselines.avtenet.pipeline import Prepared


class SpeedTests(unittest.TestCase):
    def test_lossless_cache_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            row={'source_sha256':'a'*64};arrays={'waveform':np.arange(32000,dtype=np.float32)/32000,
                'lips':np.arange(50*96*96,dtype=np.uint8).reshape(50,96,96)}
            ordinary=InputCache(_fk_paths.relocate((Path(folder)/'compressed'), __file__),{});fast=FastRawCache(_fk_paths.relocate((Path(folder)/'raw'), __file__),{})
            for cache in (ordinary,fast):
                values=cache.get(row,0.,lambda:arrays)
                for key in arrays:np.testing.assert_array_equal(values[key],arrays[key])
                values=cache.get(row,0.,lambda:(_ for _ in ()).throw(AssertionError('provider repeated')))
                for key in arrays:np.testing.assert_array_equal(values[key],arrays[key])
            next(fast.root.glob('*.npz')).write_bytes(b'corruption')
            with self.assertRaisesRegex(ValueError,'checksum'):fast.load(row,0.)

    def test_ast_cold_prefetch_matches_sequential_real_processor(self):
        from transformers import ASTFeatureExtractor
        extractor=ASTFeatureExtractor()
        rows=[{'video_id':'a','path':'a.mp4','split':'test','duration':4.,'audio_label':0,'source_sha256':'a'*64}]
        class Population:
            root=_fk_paths.relocate((Path('.')), __file__)
            def rows(self,split):return iter(rows)
            def path(self,row):return self.root/row['path']
        main=threading.get_ident();threads=[]
        def provider(self,path,start):
            threads.append(threading.get_ident())
            if start==1:raise ValueError('fixture bad audio')
            return {'waveform':np.sin(np.arange(32000,dtype=np.float32)/100+start)*.1}
        assets={'experts':{'audio':{'directory':'fixture'}}}
        with tempfile.TemporaryDirectory() as folder,patch.object(ASTFeatureExtractor,'from_pretrained',return_value=extractor):
            def run(cls,name):
                p=cls(Population(),'audio',None,assets,None,input_cache=FastRawCache(_fk_paths.relocate((Path(folder)/name), __file__),{}),
                    failure_log=_fk_paths.relocate((Path(folder)/(name+'.jsonl')), __file__),prefetch_cached_inputs=4 if cls is FastPrepared else 0)
                with patch.object(cls,'raw_inputs',provider):return list(p.records('test'))
            torch.set_num_threads(2);rng=torch.get_rng_state().clone();before=np.random.get_state()
            reference=run(Prepared,'reference');self.assertTrue(all(t==main for t in threads));threads.clear()
            result=run(FastPrepared,'optimized');self.assertTrue(all(t!=main for t in threads))
            self.assertEqual(len(result),2)
            for a,b in zip(reference,result):
                self.assertEqual(a[:3],b[:3]);self.assertEqual(a[4],b[4]);self.assertTrue(torch.equal(a[3],b[3]))
            self.assertEqual((_fk_paths.relocate((Path(folder)/'reference.jsonl'), __file__)).read_text(),(_fk_paths.relocate((Path(folder)/'optimized.jsonl'), __file__)).read_text())
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));np.testing.assert_array_equal(before[1],np.random.get_state()[1])
            self.assertFalse(any(t.name.startswith('avtenet-cache') for t in threading.enumerate()))

    def test_clip_reads_preserve_window_alignment_and_reject_wrong_source(self):
        stores={}
        try:
            for stage in ('visual','audio','audiovisual'):
                conn=sqlite3.connect(':memory:');stores[stage]=conn
                conn.execute('CREATE TABLE features(video_id TEXT,start REAL,end REAL,source_sha256 TEXT,features BLOB)')
                for start in ([0.,2.] if stage=='audio' else [0.,1.,2.]):
                    conn.execute('INSERT INTO features VALUES(?,?,?,?,?)',('a',start,start+2,'hash',np.full(768,start,dtype='<f4').tobytes()))
            values=list(aligned_vectors(stores,{'video_id':'a','source_sha256':'hash'},[(2.,4.),(1.,3.),(0.,2.)]))
            self.assertEqual(len(values),2);self.assertEqual(values[0]['audio'][0,0],2.)
            self.assertEqual(values[1]['audio'][0,0],0.)
            stores['visual'].execute("UPDATE features SET source_sha256='wrong'")
            with self.assertRaisesRegex(ValueError,'source mismatch'):list(aligned_vectors(stores,{'video_id':'a','source_sha256':'hash'},[(0.,2.)]))
        finally:
            for conn in stores.values():conn.close()

if __name__=='__main__':unittest.main()
