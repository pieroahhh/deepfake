import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch


class AdaptationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_sdpa_matches_eager_outputs_and_gradients(self):
        from transformers import VivitConfig
        from transformers.models.vivit.modeling_vivit import VivitSelfAttention
        from baselines.avtenet.hf_vivit import SDPAAttention
        config=VivitConfig(hidden_size=32,num_attention_heads=4,attention_probs_dropout_prob=0.)
        eager=VivitSelfAttention(config);efficient=SDPAAttention(copy.deepcopy(eager))
        a=torch.randn(2,17,32,requires_grad=True);b=a.detach().clone().requires_grad_(True)
        left=eager(a)[0];right=efficient(b)[0]
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
        left.square().mean().backward();right.square().mean().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)
        for first,second in zip(eager.parameters(),efficient.parameters()):
            torch.testing.assert_close(first.grad,second.grad,rtol=1e-4,atol=1e-7)

    def test_checkpointing_backbone_has_gradients(self):
        from transformers import VivitConfig,VivitModel
        from baselines.avtenet.hf_vivit import HFVivitBackend
        config=VivitConfig(image_size=16,num_frames=4,tubelet_size=[2,8,8],hidden_size=32,
                           num_hidden_layers=1,num_attention_heads=4,intermediate_size=64)
        model=HFVivitBackend(VivitModel(config)).train()
        output=model(torch.randn(1,4,3,16,16))
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_lossless_input_cache(self):
        from baselines.avtenet.input_cache import InputCache
        with tempfile.TemporaryDirectory() as folder:
            cache=InputCache(folder,{'model':'fixture'});row={'video_id':'a','source_sha256':'a'*64}
            expected={'faces':np.full((2,4,4,3),25,dtype=np.uint8),'waveform':np.linspace(-1,1,32000,dtype=np.float32)}
            actual=cache.get(row,0.,lambda:expected)
            reused=cache.get(row,0.,lambda:self.fail('provider called for existing window'))
            for key in expected:np.testing.assert_array_equal(actual[key],reused[key])


if __name__=='__main__':unittest.main()
