"""Frozen per-window FK EfficientNet-B0/Wav2Vec2 feature adapters.

No whole-file audio reuse: every window is normalized/encoded separately.
Timestamp centers follow the selected Wav2Vec2 convolutional receptive field.
"""

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


from pathlib import Path

import torch
from torch import nn

from .artifacts import require_recorded, verified_files
from .cli import load_json


def convolution_centers(samples, kernels, strides, sample_rate):
    if (type(samples) is not int or samples <= 0 or not kernels or len(kernels) != len(strides)
            or type(sample_rate) is not int or sample_rate <= 0
            or any(type(v) is not int or v < 1 for v in (*kernels, *strides))):
        raise ValueError('invalid convolution timestamp mapping')
    receptive, jump = 1, 1
    for kernel, stride in zip(kernels, strides):
        receptive += (kernel - 1) * jump
        jump *= stride
    if samples < receptive:
        raise ValueError('audio shorter than encoder receptive field')
    length = (samples - receptive) // jump + 1
    return ((torch.arange(length, dtype=torch.float64) * jump + (receptive - 1) / 2) / sample_rate).float()


class FrozenFKEncoders(nn.Module):
    def __init__(self, visual_model, audio_model, *, image_mean, image_std, normalize_audio=True):
        super().__init__()
        self.visual_model, self.audio_model = visual_model, audio_model
        self.register_buffer('image_mean', torch.tensor(image_mean).reshape(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(image_std).reshape(1, 3, 1, 1))
        if not torch.isfinite(self.image_mean).all() or not torch.isfinite(self.image_std).all() or not (self.image_std > 0).all():
            raise ValueError('invalid image normalization')
        self.normalize_audio = normalize_audio
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @classmethod
    def from_local(cls, visual_directory, visual_manifest, audio_directory, audio_manifest):
        from safetensors.torch import load_file
        from torchvision.models import efficientnet_b0
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        visual_meta = verified_files(visual_directory, visual_manifest)
        audio_meta = verified_files(audio_directory, audio_manifest)
        if visual_meta['architecture'] != 'torchvision-efficientnet-b0' or audio_meta['architecture'] != 'transformers-wav2vec2-base':
            raise ValueError('FK requires EfficientNet-B0 and Wav2Vec2 Base artifacts')
        require_recorded(visual_meta, 'model.safetensors', 'preprocessor_config.json')
        require_recorded(audio_meta, 'model.safetensors', 'config.json', 'preprocessor_config.json')
        visual = efficientnet_b0(weights=None)
        visual.load_state_dict(load_file(_fk_paths.relocate_value((str(_fk_paths.relocate((Path(visual_directory) / 'model.safetensors'), __file__))), __file__)), strict=True)
        preparation = load_json(_fk_paths.relocate((Path(visual_directory) / 'preprocessor_config.json'), __file__))
        if preparation['image_size'] != 224:
            raise ValueError('FK requires 224x224 prepared face images')
        audio, info = Wav2Vec2Model.from_pretrained(
            _fk_paths.relocate_value((audio_directory), __file__), local_files_only=True, use_safetensors=True, output_loading_info=True,
        )
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Wav2Vec2 weights did not load exactly')
        if audio.config.hidden_size != 768 or getattr(audio.config, 'add_adapter', False):
            raise ValueError('expected 768-dimensional Wav2Vec2 Base without temporal adapters')
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(_fk_paths.relocate_value((audio_directory), __file__), local_files_only=True)
        if extractor.sampling_rate != 16000:
            raise ValueError('FK requires 16kHz audio')
        result = cls(visual, audio, image_mean=preparation['mean'], image_std=preparation['std'],
                     normalize_audio=extractor.do_normalize)
        result.artifacts = {'visual': visual_meta, 'audio': audio_meta}
        return result

    @torch.inference_mode()
    def encode_window(self, faces, waveform, visual_times):
        if (faces.shape != (50, 3, 224, 224) or waveform.shape != (32000,)
                or visual_times.shape != (50,) or not all(
                    x.is_floating_point() and torch.isfinite(x).all() for x in (faces, waveform, visual_times))):
            raise ValueError('one complete window requires 50 RGB faces and 32000 mono samples')
        if ((faces < 0) | (faces > 1)).any() or ((visual_times < 0) | (visual_times >= 2)).any():
            raise ValueError('face pixels or relative feature-center timestamps outside valid range')
        normalized = (faces - self.image_mean) / self.image_std
        visual = self.visual_model.features(normalized).mean(dim=(-2, -1))
        samples = waveform
        if self.normalize_audio:
            samples = (samples - samples.mean()) / torch.sqrt(samples.var(unbiased=False) + 1e-7)
        audio = self.audio_model(input_values=samples.unsqueeze(0), return_dict=True).last_hidden_state
        config = self.audio_model.config
        times = convolution_centers(len(waveform), config.conv_kernel, config.conv_stride, 16000)
        if audio.shape[1] != len(times):
            raise ValueError('encoder output length disagrees with declared timestamp mapping')
        # Ordinary detached tensors can safely enter the trainable GRU/head graph.
        with torch.inference_mode(False):
            return {'visual_features': visual.unsqueeze(0).cpu().clone(), 'audio_features': audio.cpu().clone(),
                    'visual_times': visual_times.cpu().clone(), 'audio_times': times.clone()}
