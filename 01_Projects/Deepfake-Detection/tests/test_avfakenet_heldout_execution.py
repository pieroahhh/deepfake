
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

import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'scripts'), __file__)))
import numpy as np
import torch
import tune_avfakenet_heldout as tuned


class Constant(torch.nn.Module):
    def __init__(self,values):super().__init__();self.values=values;self.calls=[]
    def forward(self,x):
        self.calls.append(len(x));return torch.tensor(self.values).repeat(len(x),1)


class ExecutionTests(unittest.TestCase):
    def test_near_tie_recomputes_original_aligned_batches_and_keeps_native_or(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=_fk_paths.relocate((Path(directory)), __file__);(_fk_paths.relocate((cache/'clip'), __file__)).mkdir()
            np.save(_fk_paths.relocate((cache/'clip/visual.npy'), __file__),np.zeros((70,128,128,3),dtype=np.uint8))
            np.save(_fk_paths.relocate((cache/'clip/audio.npy'), __file__),np.zeros((3,175,175),dtype=np.float32))
            visual=Constant([0.,0.]);audio=Constant([0.,1.]);tuned.BATCH_SIZE=64
            row={'video_id':'test','label':1};record={'key':'clip','reasons':{}}
            with patch.object(torch.Tensor,'cuda',lambda self:self):
                result=tuned.optimized_prediction(row,record,cache,{'visual':visual,'audio':audio},'unassessed')
            self.assertEqual(visual.calls,[64,*([8]*8),6,6])
            self.assertEqual(result['prediction'],1);self.assertEqual(result['visual_prediction'],0)
            self.assertEqual(result['audio_prediction'],1);self.assertIsNone(result['score'])

    def test_preparation_rejection_uses_unchanged_original_path(self):
        tuned.original_predict=tuned.original.predict
        result=tuned.optimized_prediction({'video_id':'x','label':0},{'reasons':{'visual':'missing_face'}},_fk_paths.relocate((Path('.')), __file__),{},'unassessed')
        self.assertIsNone(result['prediction']);self.assertEqual(result['reasons'],{'visual':'missing_face'})


if __name__=='__main__':unittest.main()
