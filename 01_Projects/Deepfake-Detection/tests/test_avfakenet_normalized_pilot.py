
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
import numpy as np
import tempfile
import torch
from run_avfakenet_normalized_pilot import balanced_epoch,Inputs,references
from baselines.common.training import atomic_checkpoint,load_checkpoint,rng_state,restore_rng


class BalancedEpochTests(unittest.TestCase):
    def test_all_examples_retained_and_batches_balanced(self):
        labels=np.array([0]*11+[1]*3)
        order=balanced_epoch(labels,7)
        self.assertEqual(set(order),set(range(14)))
        self.assertEqual(len(order),22)
        self.assertEqual(int((labels[order]==0).sum()),11)
        self.assertEqual(int((labels[order]==1).sum()),11)
        for start in range(0,len(order),8):
            batch=labels[order[start:start+8]]
            self.assertEqual(int(batch.sum())*2,len(batch))
        np.testing.assert_array_equal(order,balanced_epoch(labels,7))

    def test_one_class_is_rejected(self):
        with self.assertRaises(ValueError):balanced_epoch(np.zeros(9,dtype=int),7)

    def test_prepared_rows_keep_temporal_labels_and_visual_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=_fk_paths.relocate((Path(directory)), __file__);(_fk_paths.relocate((root/'clip'), __file__)).mkdir()
            pixels=np.zeros((2,128,128,3),dtype=np.uint8);pixels[1]=255
            np.save(_fk_paths.relocate((root/'clip/visual.npy'), __file__),pixels)
            np.save(_fk_paths.relocate((root/'clip/visual-labels.npy'), __file__),np.array([0,1],dtype=np.int8))
            records=[{'key':'clip','split':'train','reasons':{}}]
            refs=references(records,'visual','train',root)
            loader=Inputs(root,records,'visual')
            try:
                x,y=loader.batch(refs,np.array([1,0]))
                self.assertEqual(tuple(x.shape),(2,3,128,128))
                self.assertEqual(y.tolist(),[1,0])
                self.assertEqual(float(x[0].min()),1.)
                self.assertEqual(float(x[1].max()),0.)
            finally:loader.close()

    def test_sampler_cursor_optimizer_and_rng_resume_reproduce_remaining_updates(self):
        torch.manual_seed(7)
        model=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.arange(28,dtype=torch.float32).reshape(14,2)/28
        y=torch.tensor([0]*11+[1]*3)
        order=balanced_epoch(y.numpy(),8)
        def update(net,opt,ids):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(net(x[ids]),y[ids]);loss.backward();opt.step()
        update(model,optimizer,order[:8])
        with tempfile.TemporaryDirectory() as directory:
            path=_fk_paths.relocate((Path(directory)/'last.pt'), __file__)
            atomic_checkpoint(path,{'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'rng':rng_state(),'cursor':8})
            for start in range(8,len(order),8):update(model,optimizer,order[start:start+8])
            resumed=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.2),torch.nn.Linear(4,2))
            resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=.001)
            saved=load_checkpoint(path);resumed.load_state_dict(saved['model_state']);resumed_optimizer.load_state_dict(saved['optimizer_state']);restore_rng(saved['rng'])
            for start in range(saved['cursor'],len(order),8):update(resumed,resumed_optimizer,order[start:start+8])
            for key,value in model.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
