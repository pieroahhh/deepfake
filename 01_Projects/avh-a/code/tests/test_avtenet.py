"""Component mechanics only: no pretrained-expert or detection-accuracy claims."""

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


import copy
import importlib.util
from pathlib import Path
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
PROFILE = _fk_paths.relocate((Path(__file__).resolve().parents[1] / 'configs/avtenet.components-v1.json'), __file__)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for AVTENet component tests')
class AVTENetTests(unittest.TestCase):
    def profile(self):
        from deepfake_detection.avtenet import load_component_profile
        return load_component_profile(PROFILE)

    def test_modality_target_mapping_and_unknowns(self):
        from deepfake_detection.avtenet import expert_target
        for visual, audio in ((0, 0), (0, 1), (1, 0), (1, 1)):
            labels = dict(visual_label=visual, audio_label=audio, video_label=int(bool(visual or audio)))
            self.assertEqual(expert_target('visual', **labels), 1 - visual)
            self.assertEqual(expert_target('audio', **labels), 1 - audio)
            self.assertEqual(expert_target('audiovisual', **labels), 1 - labels['video_label'])
        self.assertIsNone(expert_target('audio', visual_label=1, audio_label=None, video_label=1))
        with self.assertRaises(ValueError):
            expert_target('visual', visual_label=1, audio_label=0, video_label=0)

    def test_temporal_shape_and_all_gradients(self):
        import torch
        from deepfake_detection.avtenet import MultiScaleTemporalHead
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            head = MultiScaleTemporalHead(self.profile())
            inputs = torch.randn(2, 7, 768, requires_grad=True)
            outputs = head(inputs)
            self.assertEqual(tuple(outputs.shape), (2, 768))
            outputs.square().mean().backward()
            self.assertTrue(torch.isfinite(inputs.grad).all())
            for name, parameter in head.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        finally:
            torch.set_num_threads(previous)

    def test_fusion_is_single_linear_with_declared_parameter_count(self):
        import torch
        from deepfake_detection.avtenet import FeatureFusion
        fusion = FeatureFusion(self.profile())
        self.assertEqual(sum(p.numel() for p in fusion.parameters()), 2305)
        features = {name: torch.randn(2, 768) for name in ('visual', 'audio', 'audiovisual')}
        with torch.no_grad():
            fusion.classifier.weight.zero_()
            fusion.classifier.weight[0, 0] = 1
            fusion.classifier.weight[0, 768] = 2
            fusion.classifier.weight[0, 1536] = 3
            fusion.classifier.bias.zero_()
        actual = fusion(features)
        expected = features['visual'][:, 0] + 2 * features['audio'][:, 0] + 3 * features['audiovisual'][:, 0]
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            fusion({'visual': features['visual']})

    def test_fusion_training_does_not_update_experts_or_batchnorm(self):
        import torch
        from torch import nn
        from deepfake_detection.avtenet import FrozenExpertFusion, VectorExpert, native_binary_loss
        experts = {
            name: VectorExpert(nn.Sequential(nn.Linear(4, 768), nn.BatchNorm1d(768), nn.Dropout(.5)), 768)
            for name in ('visual', 'audio', 'audiovisual')
        }
        model = FrozenExpertFusion(experts, self.profile()).train()
        before = {name: copy.deepcopy(expert.state_dict()) for name, expert in experts.items()}
        optimizer = torch.optim.Adam(model.fusion.parameters(), lr=.002)
        inputs = {name: torch.randn(2, 4) for name in experts}
        logits = model(inputs)
        loss = native_binary_loss(logits, torch.tensor([0, 1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(all(p.grad is not None for p in model.fusion.parameters()))
        for name, expert in experts.items():
            self.assertFalse(expert.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in expert.parameters()))
            for key, tensor in expert.state_dict().items():
                torch.testing.assert_close(tensor, before[name][key], rtol=0, atol=0)

    def test_label_direction_and_stable_binary_loss(self):
        import torch
        from deepfake_detection.avtenet import fake_scores, native_binary_loss
        logits = torch.tensor([1000., -1000.], requires_grad=True)
        scores = fake_scores(logits)
        self.assertLess(scores[0].item(), .01)
        self.assertGreater(scores[1].item(), .99)
        loss = native_binary_loss(logits, torch.tensor([0, 1]))
        self.assertLess(loss.item(), .001)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_cache_invalidates_on_any_expert_change(self):
        from deepfake_detection.avtenet import cache_header, verify_cache_header
        hashes = {'visual': '1' * 64, 'audio': '2' * 64, 'audiovisual': '3' * 64}
        header = cache_header(self.profile(), hashes, preprocessing_sha256='4' * 64, split_sha256='5' * 64)
        verify_cache_header(header, copy.deepcopy(header))
        for name in hashes:
            changed = copy.deepcopy(header)
            changed['expert_checkpoint_sha256'][name] = '6' * 64
            with self.assertRaisesRegex(ValueError, 'stale'):
                verify_cache_header(changed, header)
        for name in ('profile_sha256', 'preprocessing_sha256', 'split_sha256'):
            changed = copy.deepcopy(header)
            changed[name] = '7' * 64
            with self.assertRaises(ValueError):
                verify_cache_header(changed, header)
