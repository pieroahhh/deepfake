"""Explicit backbone adapters and local-only, checksum-verified weight loading.

A constructible architecture is not evidence that its weights are pretrained.
Production loaders require manifests; tests label small random artifacts synthetic.
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
from torch.nn import functional as F

from .artifacts import require_recorded, verified_files
from .cli import load_json


def prepare_vivit(rgb_frames, config):
    """Declared direct-resize preparation; caller supplies source-sampled frames.

    This is a reconstruction choice, not an inferred Scenic checkpoint recipe.
    """
    import numpy as np
    frames = np.asarray(rgb_frames)
    if (frames.ndim != 4 or frames.shape[0] != config['frames'] or frames.shape[-1] != 3
            or min(frames.shape[1:3]) < 1 or frames.dtype != np.uint8):
        raise ValueError('expected exactly the configured number of uint8 RGB frames')
    if (config['input_normalization'] != 'rgb_unit_interval_to_minus_one_plus_one'
            or config['resize'] != 'bilinear_stretch_antialias_true'):
        raise ValueError('unsupported explicit ViViT preparation convention')
    pixels = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255
    pixels = F.interpolate(pixels, size=(config['image_size'], config['image_size']),
                           mode='bilinear', align_corners=False, antialias=True)
    return (pixels * 2 - 1).unsqueeze(0)


class FactorizedViViT(nn.Module):
    """Tubelets -> independent spatial encoders -> temporal cls-token encoder.

    This is a PyTorch reconstruction, not a converted Scenic model. Checkpoints
    must target this exact architecture; joint-attention HF ViViT is incompatible.
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        size, frames = config['image_size'], config['frames']
        temporal, height, width = config['tubelet']
        dimension = config['dimension']
        if (any(type(v) is not int or v <= 0 for v in
                (size, frames, temporal, height, width, dimension, config['heads'],
                 config['spatial_depth'], config['temporal_depth'], config['mlp_ratio']))
                or frames % temporal or size % height or size % width
                or dimension % config['heads']):
            raise ValueError('invalid factorized ViViT dimensions')
        self.feature_dimension = dimension
        self.grid_time = frames // temporal
        spatial_tokens = (size // height) * (size // width)
        self.tubelets = nn.Conv3d(3, dimension, (temporal, height, width), stride=(temporal, height, width))
        self.spatial_cls = nn.Parameter(torch.zeros(1, 1, dimension))
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, dimension))
        self.spatial_positions = nn.Parameter(torch.zeros(1, spatial_tokens + 1, dimension))
        self.temporal_positions = nn.Parameter(torch.zeros(1, self.grid_time + 1, dimension))
        self.dropout = nn.Dropout(config['dropout'])

        def encoder(depth):
            approximate=config.get('gelu_approximate','none')
            if approximate not in ('none','tanh'):
                raise ValueError('unsupported GELU approximation')
            activation=(lambda value:F.gelu(value,approximate='tanh')) if approximate=='tanh' else 'gelu'
            layer = nn.TransformerEncoderLayer(
                dimension, config['heads'], dim_feedforward=dimension * config['mlp_ratio'],
                dropout=config['dropout'], activation=activation, batch_first=True,
                norm_first=True, layer_norm_eps=config['layer_norm_epsilon'],
            )
            result = nn.TransformerEncoder(
                layer, depth, norm=nn.LayerNorm(dimension, eps=config['layer_norm_epsilon']),
                enable_nested_tensor=False,
            )
            # TransformerEncoder clones layers; initialize separately rather than
            # leaving every layer with identical initial attention/linear weights.
            for module in result.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(module.in_proj_weight)
            return result

        self.spatial_encoder = encoder(config['spatial_depth'])
        self.temporal_encoder = encoder(config['temporal_depth'])
        for parameter in (self.spatial_cls, self.temporal_cls, self.spatial_positions, self.temporal_positions):
            nn.init.normal_(parameter, std=.02)

    def forward(self, pixels):
        config = self.config
        if (pixels.ndim != 5 or tuple(pixels.shape[1:]) !=
                (config['frames'], 3, config['image_size'], config['image_size'])
                or pixels.shape[0] == 0 or not pixels.is_floating_point()
                or not torch.isfinite(pixels).all()):
            raise ValueError('expected finite [batch, configured_frames, 3, height, width]')
        # Inputs have already been normalized to the checkpoint's declared scale.
        patches = self.tubelets(pixels.transpose(1, 2))
        batch, channels, times, _, _ = patches.shape
        spatial = patches.permute(0, 2, 3, 4, 1).reshape(batch * times, -1, channels)
        spatial = torch.cat((self.spatial_cls.expand(batch * times, -1, -1), spatial), dim=1)
        spatial = self.spatial_encoder(self.dropout(spatial + self.spatial_positions))[:, 0]
        temporal = spatial.reshape(batch, times, channels)
        temporal = torch.cat((self.temporal_cls.expand(batch, -1, -1), temporal), dim=1)
        return self.temporal_encoder(self.dropout(temporal + self.temporal_positions))[:, 0]

    @classmethod
    def from_local(cls, directory, manifest, *, require_pretrained=True):
        from safetensors.torch import load_file
        metadata = verified_files(directory, manifest, require_pretrained=require_pretrained)
        if metadata['architecture'] != 'factorized-vivit-reconstruction-v1':
            raise ValueError('checkpoint is not for the factorized reconstruction')
        require_recorded(metadata, 'config.json', 'model.safetensors')
        model = cls(load_json(_fk_paths.relocate((Path(directory) / 'config.json'), __file__)))
        model.load_state_dict(load_file(_fk_paths.relocate_value((str(_fk_paths.relocate((Path(directory) / 'model.safetensors'), __file__))), __file__)), strict=True)
        model.artifact = metadata
        return model


class ASTBackend(nn.Module):
    """Explicit cls or mean(cls,distillation) features, not AudioSet class scores."""

    def __init__(self, model, *, token):
        super().__init__()
        if token not in ('cls', 'mean_cls_distillation'):
            raise ValueError('AST token convention must be explicit')
        self.model, self.token = model, token
        self.feature_dimension = model.config.hidden_size

    def forward(self, values):
        if (values.ndim != 3 or tuple(values.shape[1:]) !=
                (self.model.config.max_length, self.model.config.num_mel_bins)
                or values.shape[0] == 0 or not values.is_floating_point()
                or not torch.isfinite(values).all()):
            raise ValueError('invalid prepared AST feature shape or values')
        outputs = self.model(input_values=values, return_dict=True)
        sequence = outputs.last_hidden_state
        return sequence[:, 0] if self.token == 'cls' else (sequence[:, 0] + sequence[:, 1]) / 2

    @classmethod
    def from_local(cls, directory, manifest, *, token, require_pretrained=True):
        from transformers import ASTModel
        metadata = verified_files(directory, manifest, require_pretrained=require_pretrained)
        if metadata['architecture'] != 'transformers-ast-base':
            raise ValueError('unexpected AST artifact architecture')
        require_recorded(metadata, 'config.json', 'model.safetensors', 'preprocessor_config.json')
        model, info = ASTModel.from_pretrained(
            _fk_paths.relocate_value((directory), __file__), local_files_only=True, use_safetensors=True, output_loading_info=True,
        )
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('AST checkpoint did not load exactly; partial/random initialization rejected')
        if (model.config.patch_size, model.config.frequency_stride, model.config.time_stride) != (16, 10, 10):
            raise ValueError('AST patches must be 16x16 with stride10 for this comparator')
        result = cls(model, token=token)
        result.artifact = metadata
        return result


def prepare_ast(waveforms, sample_rate, directory, manifest):
    """Use the checksummed checkpoint's feature extractor, never unrelated defaults."""
    from transformers import ASTFeatureExtractor
    metadata = verified_files(directory, manifest)
    require_recorded(metadata, 'preprocessor_config.json')
    extractor = ASTFeatureExtractor.from_pretrained(_fk_paths.relocate_value((directory), __file__), local_files_only=True)
    if sample_rate != extractor.sampling_rate:
        raise ValueError('resample audio explicitly before AST feature extraction')
    return extractor(waveforms, sampling_rate=sample_rate, return_tensors='pt')['input_values']


class AVHubertBackend(nn.Module):
    """Differentiable adapter for an already loaded official AVHubertModel.

    Run in its compatible environment. Both modalities and equal, unpadded
    feature/video time lengths are required; missing modalities are not zero-filled.
    """

    def __init__(self, model, dimension=768):
        super().__init__()
        if not callable(getattr(model, 'extract_finetune', None)):
            raise ValueError('official core model with extract_finetune required')
        self.model, self.feature_dimension = model, dimension

    def forward(self, source):
        if set(source) != {'audio', 'video'} or any(source[k] is None for k in source):
            raise ValueError('paired audio filterbanks and lip video are required')
        audio, video = source['audio'], source['video']
        if (audio.ndim != 3 or video.ndim != 5 or audio.shape[0] != video.shape[0]
                or audio.shape[-1] != video.shape[2] or audio.shape[0] == 0
                or video.shape[1] != 1 or video.shape[2] == 0
                or not all(value.is_floating_point() and torch.isfinite(value).all() for value in (audio, video))):
            raise ValueError('expected aligned audio[B,F,T] and grayscale lips[B,1,T,H,W]')
        features, padding = self.model.extract_finetune(source=source, padding_mask=None, mask=False)
        if padding is not None and padding.any():
            raise ValueError('padded AV-HuBERT output cannot enter the unmasked temporal head')
        if (features.ndim != 3 or features.shape[0] != audio.shape[0]
                or features.shape[1] != video.shape[2] or features.shape[2] != self.feature_dimension
                or not torch.isfinite(features).all()):
            raise ValueError('unexpected AV-HuBERT feature shape or values')
        return features
