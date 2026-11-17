import unittest
import torch
from baselines.avfakenet.normalized_recovery import FixedNormalizedHead, set_training_mode


class FixedNormalizationTests(unittest.TestCase):
    def test_gradient_reaches_extractor_and_statistics_are_fixed(self):
        head=torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():head.weight.copy_(torch.tensor([[2.,3.]]))
        model=FixedNormalizedHead(head,[1.,2.],[2.,4.])
        x=torch.tensor([[5.,10.]],requires_grad=True)
        model(x).sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([[1.,.75]]))
        self.assertNotIn('mean',dict(model.named_parameters()))
        self.assertNotIn('scale',dict(model.named_parameters()))
        torch.testing.assert_close(model.mean,torch.tensor([1.,2.]))
        clone=FixedNormalizedHead(torch.nn.Linear(2,1,bias=False),[0.,0.],[1.,1.])
        clone.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(model(x),clone(x))

    def test_rejects_invalid_scales(self):
        for scale in ([1.,0.],[1.,float('nan')],[1.,-1.]):
            with self.assertRaises(ValueError):FixedNormalizedHead(torch.nn.Identity(),[0.,0.],scale)

    def test_feature_eval_disables_dropout_but_keeps_gradients_and_head_training(self):
        model=torch.nn.Module()
        for name in ('input_block','patch_embedding','stages','merges','output_block'):
            setattr(model,name,torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.5)))
        set_training_mode(model,'off')
        self.assertTrue(model.output_block.training)
        for name in ('input_block','patch_embedding','stages','merges'):
            module=getattr(model,name)
            self.assertFalse(module.training)
            self.assertTrue(all(p.requires_grad for p in module.parameters()))
        x=torch.ones(1,2)
        model.output_block(model.input_block(x)).sum().backward()
        self.assertIsNotNone(model.input_block[0].weight.grad)


if __name__=='__main__':unittest.main()
