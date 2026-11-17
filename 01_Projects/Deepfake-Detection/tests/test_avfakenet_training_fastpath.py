
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

import sys,unittest,tempfile,copy
from pathlib import Path
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'scripts'), __file__)))
import numpy as np
import torch
import run_avfakenet_one_epoch as pilot
from avfakenet_training_fastpath import enable,input_class,loss
from baselines.avfakenet.model import DSTNet,load_profile,classification_loss


class FastpathTests(unittest.TestCase):
    def test_logits_loss_and_all_gradients_match(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        profile=load_profile(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'baselines/avfakenet/configs/avfakenet.reconstruction-v1.json'), __file__))
        a=DSTNet(profile,'visual').eval();b=enable(copy.deepcopy(a))
        x=torch.rand(2,3,128,128);y=torch.tensor([0,1])
        out_a=a(x);out_b=b(x)
        torch.testing.assert_close(out_a,out_b,rtol=0,atol=0)
        left=classification_loss(out_a,y,.1);right=loss(out_b,y,.1)
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        left.backward();right.backward()
        for (_,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_prefetch_matches_reference_sampler_and_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            cache=_fk_paths.relocate((Path(directory)), __file__);records=[];refs=[]
            for clip in range(3):
                folder=_fk_paths.relocate((cache/str(clip)), __file__);folder.mkdir();rng=np.random.default_rng(clip)
                np.save(_fk_paths.relocate((folder/'visual.npy'), __file__),rng.integers(0,256,(10,128,128,3),dtype=np.uint8))
                records.append({'key':str(clip),'split':'train'})
                refs.extend((clip,i,int(i==0)) for i in range(10))
            refs=np.array(refs,dtype=np.int64);order=pilot.balanced_epoch(refs[:,2],8)
            base=pilot.Inputs(cache,records,'visual');fast=input_class(pilot.Inputs,pilot.balanced_epoch)(cache,records,'visual')
            try:
                for start in range(8,len(order),8):
                    ids=order[start:start+8];x,y=base.batch(refs,ids);u,v=fast.batch(refs,ids)
                    torch.testing.assert_close(x,u,rtol=0,atol=0);torch.testing.assert_close(y,v,rtol=0,atol=0)
                # Switching to validation rows drains pending work first.
                val=refs[:2].copy();records[0]['split']='validation'
                x,y=base.batch(val,np.arange(2));u,v=fast.batch(val,np.arange(2))
                torch.testing.assert_close(x,u,rtol=0,atol=0)
            finally:base.close();fast.close()


if __name__=='__main__':unittest.main()
