import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is unavailable; run these on the training environment")
class ModelTests(unittest.TestCase):
    def test_pool_matches_reference_values_and_gradients(self):
        import torch
        from deepfake_detection.model import EvidenceModel

        torch.manual_seed(11)
        model = EvidenceModel(visual_dim=4, audio_dim=6, hidden=8, dimension=8)
        sequence = torch.randn(2, 50, 4, dtype=torch.float64, requires_grad=True)
        times = (torch.arange(50, dtype=torch.float64) + .5) / 25
        # Reverse one window to check that pooling uses timestamps, not ordering.
        times = torch.stack((times, times.flip(0)))
        indices = torch.floor(times * 5).long()
        reference = torch.stack([
            (sequence * (indices == b).unsqueeze(-1)).sum(1)
            / (indices == b).sum(1, keepdim=True)
            for b in range(10)
        ], dim=1)
        actual = model.pool(sequence, times)
        torch.testing.assert_close(actual, reference)
        weights = torch.randn_like(actual)
        actual_grad = torch.autograd.grad((actual * weights).sum(), sequence)[0]
        reference_grad = torch.autograd.grad((reference * weights).sum(), sequence)[0]
        torch.testing.assert_close(actual_grad, reference_grad)

    def test_loss_masks_unknown_targets_and_preserves_gradients(self):
        import torch
        from torch.nn import functional as F
        from deepfake_detection.model import multilevel_loss

        windows = torch.tensor([.2, .8], dtype=torch.float64, requires_grad=True)
        bins = torch.tensor([[.1, .7], [.3, .9]], dtype=torch.float64, requires_grad=True)
        video = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
        labels = {"video_eligible": True, "windows": [
            {"window": 0, "bins": [0, None]},
            {"window": None, "bins": [None, 1]},
        ]}
        actual = multilevel_loss(
            {"window_scores": windows, "bin_scores": bins}, video, 1, labels,
            window_weight=2, bin_weight=3,
        )
        reference = (
            F.binary_cross_entropy(video, video.new_tensor(1.))
            + 2 * F.binary_cross_entropy(windows[0], windows.new_tensor(0.))
            + 3 * F.binary_cross_entropy(bins[0, 0], bins.new_tensor(0.))
            + 3 * F.binary_cross_entropy(bins[1, 1], bins.new_tensor(1.))
        )
        torch.testing.assert_close(actual, reference)
        actual_grads = torch.autograd.grad(actual, (windows, bins, video))
        reference_grads = torch.autograd.grad(reference, (windows, bins, video))
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad)

    def test_fully_unknown_loss_is_differentiable_zero(self):
        import torch
        from deepfake_detection.model import multilevel_loss

        windows = torch.tensor([.4], requires_grad=True)
        bins = torch.tensor([[.2, .8]], requires_grad=True)
        labels = {"video_eligible": False, "windows": [
            {"window": None, "bins": [None, None]},
        ]}
        loss = multilevel_loss(
            {"window_scores": windows, "bin_scores": bins}, windows.mean(), 1, labels,
        )
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(windows.grad, torch.zeros_like(windows)))
        self.assertTrue(torch.equal(bins.grad, torch.zeros_like(bins)))

    def test_shapes_reliability_and_video_gradients(self):
        import torch
        from deepfake_detection.model import EvidenceModel, multilevel_loss
        from deepfake_detection.core import supervision
        torch.manual_seed(7)
        model=EvidenceModel(visual_dim=4,audio_dim=6,hidden=8,dimension=8)
        visual=torch.randn(2,50,4)
        audio=torch.randn(2,100,6)
        out=model(visual,audio,torch.arange(50)/25,torch.arange(100)/50,torch.ones(2,2))
        self.assertEqual(tuple(out["bin_scores"].shape),(2,10))
        self.assertEqual(tuple(out["alpha"].shape),(2,10,4))
        self.assertTrue(torch.allclose(out["alpha"].sum(-1),torch.ones(2,10)))
        self.assertTrue(torch.equal(out["temporal_evidence"][:,0],torch.zeros(2,8)))
        p=model.aggregate_video(out["window_scores"])
        labels=supervision(1,True,[(.2,.4)],[(0.,2.),(1.,3.)])
        loss=multilevel_loss(out,p,1,labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_empty_feature_bins_fail(self):
        import torch
        from deepfake_detection.model import EvidenceModel
        model=EvidenceModel(visual_dim=4,audio_dim=6,hidden=8,dimension=8)
        with self.assertRaisesRegex(ValueError,"empty feature bin"):
            model.pool(torch.ones(1,2,4),torch.tensor([0.,.1]))

    def test_loss_preserves_sums(self):
        import torch
        from deepfake_detection.model import multilevel_loss
        score=torch.tensor([.5,.5],requires_grad=True)
        bins=torch.full((2,10),.5,requires_grad=True)
        labels={"video_eligible":True,"windows":[{"window":0,"bins":[0]*10} for _ in range(2)]}
        loss=multilevel_loss({"window_scores":score,"bin_scores":bins},score.mean(),0,labels)
        # 23 BCE terms: one video, two windows, and twenty bins.
        # Float32 accumulation needs a relative tolerance against the exact sum.
        torch.testing.assert_close(loss, loss.new_tensor(23*0.6931471805599453),
                                   rtol=1e-6, atol=0)
