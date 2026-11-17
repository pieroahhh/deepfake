
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
from probe_avfakenet_checkpoints import activation_stats,probe


class ProbeTests(unittest.TestCase):
    def test_activation_statistics_distinguish_dead_and_input_sensitive_values(self):
        import torch
        dead=activation_stats(torch.zeros(4,8))
        self.assertEqual(dead['zero_fraction'],1)
        self.assertEqual(dead['spread_across_inputs'],0)
        active=activation_stats(torch.tensor([[1.,2.],[3.,4.]]))
        self.assertEqual(active['spread_across_inputs'],2)
        self.assertEqual(active['zero_fraction'],0)
        self.assertTrue(active['finite'])

    def test_hooks_are_removed_when_invalid_input_raises(self):
        import torch
        from baselines.avfakenet.model import DSTNet,load_profile
        root=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
        profile=load_profile(_fk_paths.relocate((root/'baselines/avfakenet/configs/avfakenet.reconstruction-v1.json'), __file__))
        model=DSTNet(profile,'visual').eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        with self.assertRaises(ValueError):probe(model,torch.zeros(1,3,1,1))
        self.assertFalse(any(m._forward_hooks for m in model.modules()))
        for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[name]))


if __name__=='__main__':unittest.main()
