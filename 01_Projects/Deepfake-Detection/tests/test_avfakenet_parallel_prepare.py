
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

import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'scripts'), __file__)))
import numpy as np
import torch
import parallel_avfakenet_prepare as worker
from baselines.common.io import digest


class ParallelPreparationTests(unittest.TestCase):
    def test_identical_payloads_and_completed_cache_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base=_fk_paths.relocate((Path(directory)), __file__);dataset=_fk_paths.relocate((base/'dataset'), __file__);dataset.mkdir();(_fk_paths.relocate((dataset/'clip.mp4'), __file__)).write_bytes(b'source')
            row={'video_id':'clip.mp4','path':'clip.mp4','source_sha256':digest(_fk_paths.relocate((dataset/'clip.mp4'), __file__)),
                 'split':'train','duration':.08,'label':1,'visual_label':1,'audio_label':0,
                 'visual_manipulated_intervals':[[.02,.06]]}
            sample={'faces':np.zeros((2,128,128,3),dtype=np.uint8),'source_times':np.array([0.,.04])}
            worker.DETECT=object();worker.PROFILE={}
            with patch('baselines.common.media.faces',return_value=sample),patch('baselines.common.media.audio',return_value=np.zeros(1280,dtype=np.float32)),patch('baselines.avfakenet.preprocessing.prepare_audio',return_value=torch.zeros(3,175,175)):
                ordinary=worker.prepare_one(row,str(dataset),str(_fk_paths.relocate((base/'ordinary'), __file__)),False)
                optimized=worker.prepare_one(row,str(dataset),str(_fk_paths.relocate((base/'optimized'), __file__)),True)
            self.assertEqual(ordinary['record'],optimized['record'])
            header=_fk_paths.relocate((base/'optimized'/optimized['record']['key']/'record.json'), __file__);before=digest(header)
            with patch('baselines.common.media.faces',side_effect=AssertionError('Must not recompute')):
                reused=worker.prepare_one(row,str(dataset),str(_fk_paths.relocate((base/'optimized'), __file__)),True)
            self.assertTrue(reused['reused']);self.assertEqual(digest(header),before)
            (_fk_paths.relocate((dataset/'clip.mp4'), __file__)).write_bytes(b'changed')
            with self.assertRaises(ValueError):worker.prepare_one(row,str(dataset),str(_fk_paths.relocate((base/'optimized'), __file__)),True)

    def test_data_exclusion_reason_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            base=_fk_paths.relocate((Path(directory)), __file__);dataset=_fk_paths.relocate((base/'dataset'), __file__);dataset.mkdir();(_fk_paths.relocate((dataset/'clip.mp4'), __file__)).write_bytes(b'source')
            row={'video_id':'clip.mp4','path':'clip.mp4','source_sha256':digest(_fk_paths.relocate((dataset/'clip.mp4'), __file__)),
                 'split':'train','duration':1.,'label':0,'visual_label':0,'audio_label':0}
            worker.DETECT=object();worker.PROFILE={}
            with patch('baselines.common.media.faces',side_effect=ValueError('missing_face')),patch('baselines.common.media.audio',side_effect=ValueError('missing audio')):
                a=worker.prepare_one(row,str(dataset),str(_fk_paths.relocate((base/'a'), __file__)),False)
                b=worker.prepare_one(row,str(dataset),str(_fk_paths.relocate((base/'b'), __file__)),True)
            self.assertEqual(a['record'],b['record'])
            self.assertEqual(a['record']['files'],{})


if __name__=='__main__':unittest.main()
