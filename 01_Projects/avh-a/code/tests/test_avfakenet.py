"""Synthetic mechanical checks, not evidence of baseline detection accuracy."""

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
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
PROFILE = _fk_paths.relocate((Path(__file__).resolve().parents[1] / "configs/avfakenet.reconstruction-v1.json"), __file__)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for baseline reconstruction tests")
class AVFakeNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    def profile(self):
        from deepfake_detection.avfakenet import load_profile
        return load_profile(PROFILE)

    def uniform_attention(self, shift):
        import torch
        from deepfake_detection.avfakenet import WindowAttention
        attention = WindowAttention(2, 1, 4, shift, 0, False, False)
        with torch.no_grad():
            attention.qkv.weight.zero_()
            attention.qkv.weight[4:].copy_(torch.eye(2))
            attention.projection.weight.copy_(torch.eye(2))
            attention.projection.bias.zero_()
        return attention

    def test_attention_padding_cannot_dilute_valid_values(self):
        import torch
        for shift in (0, 2):
            attention = self.uniform_attention(shift)
            # Partial windows and one-window axes exercise both padding and shifts.
            for height, width in ((5, 6), (2, 6), (2, 2)):
                inputs = torch.ones(2, height, width, 2, requires_grad=True)
                output = attention(inputs)
                torch.testing.assert_close(output, inputs)
                output.sum().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_shift_connects_neighbors_without_cyclic_wraparound(self):
        import torch
        shifted = self.uniform_attention(2)
        unshifted = self.uniform_attention(0)
        inputs = torch.zeros(1, 8, 8, 2)
        inputs[:, 3, 3] = 4
        self.assertGreater(shifted(inputs)[0, 4, 4, 0].item(), 0)
        self.assertEqual(unshifted(inputs)[0, 4, 4, 0].item(), 0)
        inputs.zero_()
        inputs[:, 7, 7] = 100
        self.assertEqual(shifted(inputs)[0, 0, 0, 0].item(), 0)

    def test_both_branches_shapes_parameters_and_backward(self):
        import torch
        from deepfake_detection.avfakenet import DSTNet, classification_loss
        parameter_counts = []
        for modality, pixels, grid, final_grid in (("visual", 128, 43, 6), ("audio", 175, 59, 8)):
            torch.manual_seed(7)
            model = DSTNet(self.profile(), modality)
            logits, shapes = model(torch.rand(1, 3, pixels, pixels), return_shapes=True)
            self.assertEqual(shapes["patch_embedding"], [1, grid, grid, 64])
            self.assertEqual(shapes["merge_3"], [1, final_grid, final_grid, 512])
            self.assertEqual(shapes["logits"], [1, 2])
            loss = classification_loss(logits, torch.tensor([1]), .1)
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            parameter_counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(parameter_counts[0], parameter_counts[1])

    def test_loss_matches_explicit_bce_and_handles_extreme_logits(self):
        import torch
        from torch.nn import functional as F
        from deepfake_detection.avfakenet import classification_loss
        logits = torch.tensor([[.2, -.4], [-.3, .8]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1])
        targets = F.one_hot(labels, 2).double() * .9 + .05
        actual = classification_loss(logits, labels, .1)
        expected = F.binary_cross_entropy(logits.softmax(-1), targets)
        torch.testing.assert_close(actual, expected)
        extreme = torch.tensor([[10000., -10000.]], requires_grad=True)
        loss = classification_loss(extreme, torch.tensor([1]), .1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_profile_rejects_missing_and_incompatible_settings(self):
        from deepfake_detection.avfakenet import validate_profile
        for key, value in (("attention_heads", [3, 4, 8]), ("shift_size", 7),
                           ("mlp_multipliers", [4, 4, 2, 2])):
            profile = copy.deepcopy(self.profile())
            profile["architecture"][key] = value
            with self.assertRaises(ValueError):
                validate_profile(profile)
        profile = self.profile()
        del profile["architecture"]["dropout"]
        with self.assertRaises(ValueError):
            validate_profile(profile)

    def test_preparation_shapes_range_and_missing_audio(self):
        import numpy as np
        import torch
        from deepfake_detection.avfakenet_preprocessing import prepare_audio, prepare_face
        profile = self.profile()
        face = prepare_face(np.full((40, 60, 3), 255, dtype=np.uint8), profile)
        self.assertEqual(tuple(face.shape), (3, 128, 128))
        self.assertTrue(torch.equal(face, torch.ones_like(face)))
        time = np.arange(16000, dtype=np.float32) / 16000
        waveform = .5 * np.sin(2 * np.pi * 440 * time)
        mel = prepare_audio(waveform, 16000, profile)
        self.assertEqual(tuple(mel.shape), (3, 175, 175))
        self.assertTrue(torch.isfinite(mel).all())
        self.assertTrue(((mel >= 0) & (mel <= 1)).all())
        torch.testing.assert_close(mel[0], mel[1])
        stereo = np.stack((waveform, waveform), axis=1)
        torch.testing.assert_close(prepare_audio(stereo, 16000, profile), mel)
        resampled = prepare_audio(waveform[::2], 8000, profile)
        self.assertEqual(tuple(resampled.shape), (3, 175, 175))
        self.assertTrue(torch.isfinite(resampled).all())
        for bad in (np.array([], dtype=np.float32), np.zeros(16000, dtype=np.float32)):
            with self.assertRaises(ValueError):
                prepare_audio(bad, 16000, profile)

    def test_inference_votes_reset_between_videos(self):
        import torch
        from deepfake_detection.avfakenet import DSTNet, predict_prepared_video
        profile = self.profile()
        visual, audio = DSTNet(profile, "visual").eval(), DSTNet(profile, "audio").eval()
        faces, mel = torch.rand(1, 3, 128, 128), torch.rand(1, 3, 175, 175)
        # Set known classifiers so this tests branch wiring, not random accuracy.
        with torch.no_grad():
            for model in (visual, audio):
                model.output_block[-1].weight.zero_()
                model.output_block[-1].bias.copy_(torch.tensor([2., -2.]))
            visual.output_block[-1].bias.copy_(torch.tensor([-2., 2.]))
        first = predict_prepared_video(visual, audio, faces, mel)
        self.assertEqual(first["prediction"], 1)
        with torch.no_grad():
            visual.output_block[-1].bias.copy_(torch.tensor([2., -2.]))
        second = predict_prepared_video(visual, audio, faces, mel)
        self.assertEqual(second["prediction"], 0)
        self.assertIsNone(second["score"])
        self.assertEqual(predict_prepared_video(visual, audio, None, mel)["status"], "UNASSESSED")

    def test_training_saves_reloadable_profiled_checkpoint(self):
        import torch
        from deepfake_detection.avfakenet import DSTNet
        from deepfake_detection.avfakenet_training import fit_branch
        torch.manual_seed(9)
        model = DSTNet(self.profile(), "visual")
        train = [{"images": torch.rand(1, 3, 128, 128), "visual_label": torch.tensor([0])}]
        validation = [{"images": torch.rand(1, 3, 128, 128), "visual_label": torch.tensor([1])}]
        with tempfile.TemporaryDirectory() as tmp:
            output = _fk_paths.relocate((Path(tmp) / "smoke"), __file__)
            result = fit_branch(model, train, validation, output, epochs=1)
            self.assertEqual(result["epochs_completed"], 1)
            checkpoint = torch.load(_fk_paths.relocate_value((_fk_paths.relocate((output / "best.pt"), __file__)), __file__), weights_only=True)
            self.assertEqual(checkpoint["profile_sha256"], model.profile_fingerprint)
            reloaded = DSTNet(self.profile(), "visual").eval()
            reloaded.load_state_dict(checkpoint["model_state"])
            with torch.inference_mode():
                torch.testing.assert_close(model(validation[0]["images"]), reloaded(validation[0]["images"]))
            with self.assertRaises(FileExistsError):
                fit_branch(model, train, validation, output, epochs=1)
