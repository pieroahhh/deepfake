
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

import random,tempfile,threading,unittest
from pathlib import Path
import numpy as np
import torch
from baselines.avtenet.pipeline import Prepared
from baselines.avtenet.input_cache import InputCache

class Tests(unittest.TestCase):
    def test_inputs_order_rng_cache_misses_and_resume_progress(self):
        main_thread=threading.get_ident()
        settings={'image_mean':.421,'image_std':.165,'crop_size':88,'stack_order_audio':4,'normalize_audio':True,
            'filterbank_settings':{'winlen':.025,'winstep':.01,'nfilt':26,'nfft':512,'lowfreq':0,'highfreq':8000,'preemph':.97}}
        rows=[{'video_id':'a','duration':4.,'source_sha256':'a'*64,'label':0}]
        class Population:
            def rows(self,split):return iter(rows)
            def path(self,row):return 'fixture'
        class Fixture(Prepared):
            def raw_inputs(self,path,start):
                assert threading.get_ident()==main_thread,'provider moved off main thread'
                return {'waveform':np.sin(np.arange(32000,dtype=np.float32)/100)*.1,
                        'lips':np.full((50,96,96),int(start)*20+80,dtype=np.uint8)}
        assets={'experts':{'audiovisual':{'preparation':settings}}}
        with tempfile.TemporaryDirectory() as folder:
            torch.set_num_threads(2)
            def collect(depth,name,skip=None):
                p=Fixture(Population(),'audiovisual',None,assets,None,input_cache=InputCache(_fk_paths.relocate((Path(folder)/name), __file__),{}),prefetch_processed_inputs=depth)
                return list(p.records('train',skip_windows=skip)),p.progress('train')
            rng=torch.get_rng_state().clone();np_rng=np.random.get_state();py_rng=random.getstate()
            reference,_=collect(0,'reference')
            for name in ('cold','cold','reference'):
                actual,_=collect(4,name)
                self.assertEqual(len(actual),3)
                for a,b in zip(reference,actual):
                    self.assertEqual(a[:3],b[:3]);self.assertEqual(a[4],b[4])
                    for key in a[3]:self.assertTrue(torch.equal(a[3][key],b[3][key]))
            resumed,progress=collect(4,'reference',{('a',0.,2.),('a',1.,3.)})
            self.assertEqual(len(resumed),1);self.assertEqual(progress['windows_processed'],3)
            self.assertEqual(progress['initial_windows'],2)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(random.getstate(),py_rng)
            np.testing.assert_array_equal(np.random.get_state()[1],np_rng[1])

if __name__=='__main__':unittest.main()
