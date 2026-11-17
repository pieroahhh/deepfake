"""Versioned, paper-guided DST-Net reconstruction; not author-verified code.

Inputs are prepared RGB face or Mel images [batch, 3, height, width]. Two
independent instances represent the visual and audio branches. The profile
and audit distinguish paper facts from the dimensions and policies chosen here.
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


import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from deepfake_detection.baselines import avfakenet_decision


def load_profile(path):
    """Load the explicit profile; do not fill in missing architecture settings."""
    profile = json.loads(_fk_paths.relocate((Path(path)), __file__).read_text())
    return validate_profile(profile)


def validate_profile(profile):
    """Reject incompatible structural settings before allocating a network."""
    if profile.get("profile_id") != "avfakenet-reconstruction-v1":
        raise ValueError("unsupported AVFakeNet reconstruction profile")
    architecture = profile["architecture"]
    required = {
        "input_channels", "input_dense_widths", "patch_size", "embedding_dimension",
        "stage_depths", "attention_heads", "window_size", "shift_size",
        "mlp_multipliers", "output_dense_widths", "dropout", "layer_norm_epsilon",
        "relative_position_bias", "qkv_bias",
    }
    if set(architecture) != required:
        raise ValueError("missing or unexpected architecture settings")
    if architecture["input_channels"] != 3 or architecture["patch_size"] != 3:
        raise ValueError("v1 requires RGB inputs and 3x3 patches")
    if architecture["embedding_dimension"] != 64:
        raise ValueError("v1 requires initial embedding dimension 64")
    for name, count in (("input_dense_widths", 3), ("output_dense_widths", 2),
                        ("mlp_multipliers", 4), ("attention_heads", 3)):
        values = architecture[name]
        if not isinstance(values, list) or len(values) != count or any(
            type(value) is not int or value <= 0 for value in values
        ):
            raise ValueError(f"invalid {name}")
    if architecture["stage_depths"] != [2, 2, 2]:
        raise ValueError("v1 has three stages, each with a window/shifted-window pair")
    if architecture["mlp_multipliers"][-1] != 1:
        raise ValueError("MLP output must match its residual dimension")
    window, shift = architecture["window_size"], architecture["shift_size"]
    if type(window) is not int or type(shift) is not int or not 0 < shift < window:
        raise ValueError("require integer 0 < shift_size < window_size")
    if not 0 <= architecture["dropout"] < 1 or not 0 < architecture["layer_norm_epsilon"] < 1:
        raise ValueError("invalid dropout or normalization epsilon")
    for stage, heads in enumerate(architecture["attention_heads"]):
        if (64 * 2 ** stage) % heads:
            raise ValueError("stage dimension must be divisible by attention heads")
    for name in ("relative_position_bias", "qkv_bias"):
        if type(architecture[name]) is not bool:
            raise ValueError(f"{name} must be Boolean")
    if profile["preprocessing"]["visual_pixels"] != 128 or profile["preprocessing"]["audio_pixels"] != 175:
        raise ValueError("unexpected branch image sizes")
    if profile["decision"]["tie_break"] not in ("real", "fake", "unassessed"):
        raise ValueError("explicit majority-vote tie policy required")
    if profile["decision"]["fusion"] != "hard_OR" or profile["decision"]["frame_class_tie"] != "argmax_first_class_real":
        raise ValueError("unsupported reconstruction decision settings")
    preparation = profile["preprocessing"]
    implemented = {
        "mel_rendering": "fixed_db_grayscale_replicated_rgb",
        "db_reference": "per_clip_max_power", "normalization": "unit_interval",
        "resize_interpolation": "bilinear", "audio_mono": "mean_channels_before_resampling",
        "face_input": "caller_supplied_RGB_face_crop", "frame_sampling": "not_implemented",
    }
    if any(preparation[key] != value for key, value in implemented.items()):
        raise ValueError("unsupported preparation convention")
    if preparation["db_floor"] >= 0 or preparation["minimum_peak_power"] <= 0:
        raise ValueError("invalid spectrogram energy settings")
    training = profile["training"]
    if (training["optimizer"] != "AdamW" or training["scheduler"] != "none"
            or training["loss"] != "mean_two_output_softmax_bce_uniform_label_smoothing"
            or training["early_stopping_metric"] != "validation_accuracy"
            or training["checkpoint_ties"] != "keep_first_strict_accuracy_improvement"):
        raise ValueError("unsupported training convention")
    for name in ("maximum_epochs", "batch_size", "early_stopping_patience"):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    # Also reject NaN/Infinity anywhere in the recorded profile.
    profile_hash(profile)
    return profile


def profile_hash(profile):
    """Commit to every setting, including preprocessing and decision policies."""
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _partition(images, window):
    """[batch, height, width, channels] -> [batch*windows, tokens, channels]."""
    batch, height, width, channels = images.shape
    return images.reshape(batch, height // window, window, width // window, window, channels).permute(
        0, 1, 3, 2, 4, 5
    ).reshape(-1, window * window, channels)


def _reverse(windows, window, batch, height, width):
    return windows.reshape(batch, height // window, width // window, window, window, -1).permute(
        0, 1, 3, 2, 4, 5
    ).reshape(batch, height, width, -1)


class WindowAttention(nn.Module):
    """Local attention with cyclic-shift boundary and padding masks.

    Padded keys cannot affect valid outputs. Artificial cyclic wraparound is
    blocked by region IDs, while actual neighboring windows can communicate.
    """

    def __init__(self, dimension, heads, window, shift, dropout, relative_bias, qkv_bias):
        super().__init__()
        self.heads, self.window, self.shift = heads, window, shift
        self.scale = (dimension // heads) ** -0.5
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=qkv_bias)
        self.projection = nn.Linear(dimension, dimension)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.relative_bias = (
            nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads)) if relative_bias else None
        )
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
        coords = coords.flatten(1)
        offsets = coords[:, :, None] - coords[:, None, :]
        offsets = offsets.permute(1, 2, 0) + window - 1
        index = offsets[..., 0] * (2 * window - 1) + offsets[..., 1]
        self.register_buffer("relative_index", index, persistent=False)

    def _layout(self, height, width, device):
        window = self.window
        padded_h = height + (-height) % window
        padded_w = width + (-width) % window
        # A single-window axis need not be shifted.
        shift_h = self.shift if height > window else 0
        shift_w = self.shift if width > window else 0
        valid = torch.zeros(1, padded_h, padded_w, 1, device=device, dtype=torch.bool)
        valid[:, :height, :width] = True
        valid = torch.roll(valid, (-shift_h, -shift_w), dims=(1, 2))
        valid = _partition(valid, window).squeeze(-1)

        regions = torch.zeros(1, padded_h, padded_w, 1, device=device, dtype=torch.long)
        slices_h = (slice(0, -window), slice(-window, -shift_h), slice(-shift_h, None)) if shift_h else (slice(None),)
        slices_w = (slice(0, -window), slice(-window, -shift_w), slice(-shift_w, None)) if shift_w else (slice(None),)
        for row, h_slice in enumerate(slices_h):
            for col, w_slice in enumerate(slices_w):
                regions[:, h_slice, w_slice] = row * len(slices_w) + col
        regions = _partition(regions, window).squeeze(-1)
        allowed = (regions[:, :, None] == regions[:, None, :]) & valid[:, None, :]
        # Invalid queries get a finite self-only row, then are zeroed on output.
        diagonal = torch.eye(window * window, device=device, dtype=torch.bool)
        allowed = torch.where(valid[:, :, None], allowed, diagonal)
        return padded_h, padded_w, shift_h, shift_w, allowed, valid

    def forward(self, images):
        batch, height, width, channels = images.shape
        padded_h, padded_w, shift_h, shift_w, allowed, valid = self._layout(height, width, images.device)
        padded = F.pad(images, (0, 0, 0, padded_w - width, 0, padded_h - height))
        shifted = torch.roll(padded, (-shift_h, -shift_w), dims=(1, 2))
        tokens = _partition(shifted, self.window)
        count, length, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(count, length, 3, self.heads, channels // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query @ key.transpose(-2, -1)) * self.scale
        if self.relative_bias is not None:
            bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
            scores = scores + bias.unsqueeze(0)
        scores = scores.masked_fill(~allowed.repeat(batch, 1, 1).unsqueeze(1), float("-inf"))
        weights = self.attention_dropout(scores.softmax(-1))
        output = (weights @ value).transpose(1, 2).reshape(count, length, channels)
        output = self.output_dropout(self.projection(output))
        output = output * valid.repeat(batch, 1).unsqueeze(-1)
        output = _reverse(output, self.window, batch, padded_h, padded_w)
        output = torch.roll(output, (shift_h, shift_w), dims=(1, 2))
        return output[:, :height, :width]


class DenseMLP(nn.Module):
    """Four dense layers; ReLU only after the first two, dropout after all four."""

    def __init__(self, dimension, multipliers, dropout):
        super().__init__()
        widths = [dimension * multiplier for multiplier in multipliers]
        layers = []
        previous = dimension
        for index, width in enumerate(widths):
            layers.append(nn.Linear(previous, width))
            if index < 2:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous = width
        if previous != dimension:
            raise ValueError("MLP output must match its residual dimension")
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)


class SwinBlock(nn.Module):
    def __init__(self, dimension, heads, shift, architecture):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=architecture["layer_norm_epsilon"])
        self.norm2 = nn.LayerNorm(dimension, eps=architecture["layer_norm_epsilon"])
        self.attention = WindowAttention(
            dimension, heads, architecture["window_size"], shift,
            architecture["dropout"], architecture["relative_position_bias"], architecture["qkv_bias"],
        )
        self.mlp = DenseMLP(dimension, architecture["mlp_multipliers"], architecture["dropout"])

    def forward(self, inputs):
        inputs = inputs + self.attention(self.norm1(inputs))
        return inputs + self.mlp(self.norm2(inputs))


class PatchMerging(nn.Module):
    """Pad odd spatial axes, concatenate 2x2 neighbors, normalize, reduce 4C->2C."""

    def __init__(self, dimension, epsilon):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dimension, eps=epsilon)
        self.reduction = nn.Linear(4 * dimension, 2 * dimension, bias=False)

    def forward(self, inputs):
        height, width = inputs.shape[1:3]
        inputs = F.pad(inputs, (0, 0, 0, width % 2, 0, height % 2))
        neighbors = torch.cat((inputs[:, 0::2, 0::2], inputs[:, 1::2, 0::2],
                               inputs[:, 0::2, 1::2], inputs[:, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(neighbors))


class DSTNet(nn.Module):
    """One branch; returns logits for class 0=genuine and class 1=fake."""

    def __init__(self, profile, modality):
        super().__init__()
        validate_profile(profile)
        if modality not in ("visual", "audio"):
            raise ValueError("modality must be visual or audio")
        # Round-trip to avoid caller mutations changing recorded provenance.
        self.profile = json.loads(json.dumps(profile, allow_nan=False))
        self.profile_fingerprint = profile_hash(self.profile)
        self.modality = modality
        self.pixels = profile["preprocessing"][f"{modality}_pixels"]
        architecture = profile["architecture"]
        self.patch_size = architecture["patch_size"]
        layers = []
        previous = architecture["input_channels"]
        for width in architecture["input_dense_widths"]:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.input_block = nn.Sequential(*layers)
        dimension = architecture["embedding_dimension"]
        self.patch_embedding = nn.Linear(self.patch_size ** 2 * previous, dimension)
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        for heads in architecture["attention_heads"]:
            self.stages.append(nn.Sequential(
                SwinBlock(dimension, heads, 0, architecture),
                SwinBlock(dimension, heads, architecture["shift_size"], architecture),
            ))
            self.merges.append(PatchMerging(dimension, architecture["layer_norm_epsilon"]))
            dimension *= 2
        layers = []
        for width in architecture["output_dense_widths"]:
            layers.extend((nn.Linear(dimension, width), nn.ReLU(), nn.Dropout(architecture["dropout"])))
            dimension = width
        layers.append(nn.Linear(dimension, 2))
        self.output_block = nn.Sequential(*layers)

    def forward(self, images, *, return_shapes=False):
        expected = (3, self.pixels, self.pixels)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected or images.shape[0] == 0:
            raise ValueError(f"expected a nonempty batch of {expected} images")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("images must be finite floating-point tensors")
        if ((images < 0) | (images > 1)).any():
            raise ValueError("prepared image values must lie in [0, 1]")
        shapes = {"input": list(images.shape)}
        features = self.input_block(images.permute(0, 2, 3, 1))
        shapes["input_dense"] = list(features.shape)
        patch = self.patch_size
        height, width = features.shape[1:3]
        features = F.pad(features, (0, 0, 0, (-width) % patch, 0, (-height) % patch))
        batch, height, width, channels = features.shape
        patches = features.reshape(batch, height // patch, patch, width // patch, patch, channels)
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, height // patch, width // patch, patch * patch * channels
        )
        features = self.patch_embedding(patches)
        shapes["patch_embedding"] = list(features.shape)
        for index, (stage, merge) in enumerate(zip(self.stages, self.merges), 1):
            features = stage(features)
            shapes[f"stage_{index}"] = list(features.shape)
            features = merge(features)
            shapes[f"merge_{index}"] = list(features.shape)
        pooled = features.mean(dim=(1, 2))
        shapes["pooled"] = list(pooled.shape)
        logits = self.output_block(pooled)
        shapes["logits"] = list(logits.shape)
        return (logits, shapes) if return_shapes else logits


def classification_loss(logits, labels, smoothing):
    """Chosen BCE convention: mean over both softmax outputs and samples.

    Smooth one-hot targets toward [0.5, 0.5]. Log-softmax and the complementary
    class log probability avoid log(0) even with extreme finite logits.
    This is our resolution of the paper's ambiguous loss description.
    """
    if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] == 0:
        raise ValueError("expected nonempty two-class logits")
    if labels.shape != logits.shape[:1] or labels.dtype != torch.long:
        raise ValueError("expected one integer class per image")
    if not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("invalid logits or labels")
    if not 0 <= smoothing < 1:
        raise ValueError("label smoothing must lie in [0, 1)")
    targets = F.one_hot(labels, num_classes=2).to(logits.dtype)
    targets = targets * (1 - smoothing) + smoothing / 2
    log_probabilities = logits.log_softmax(-1)
    return -(targets * log_probabilities + (1 - targets) * log_probabilities.flip(-1)).mean()


@torch.inference_mode()
def predict_prepared_video(visual_model, audio_model, faces, mel_image, *, frame_batch_size=16):
    """Hard native-style decisions on prepared inputs using trained branch models.

    This helper cannot establish whether weights are trained. It never supplies
    a fabricated continuous fused score. Pass None for a missing input.
    """
    if visual_model.modality != "visual" or audio_model.modality != "audio":
        raise ValueError("expected visual and audio branches in that order")
    if visual_model.profile_fingerprint != audio_model.profile_fingerprint:
        raise ValueError("branch reconstruction profiles must match")
    if visual_model.training or audio_model.training:
        raise ValueError("call eval() on both branches before inference")
    if type(frame_batch_size) is not int or frame_batch_size < 1:
        raise ValueError("frame_batch_size must be a positive integer")
    tie_break = visual_model.profile["decision"]["tie_break"]
    if faces is None or mel_image is None or len(faces) == 0:
        return avfakenet_decision([], None, tie_break=tie_break)
    if mel_image.shape != (1, 3, audio_model.pixels, audio_model.pixels):
        raise ValueError("one Mel image is required for one video")
    votes = []
    for chunk in faces.split(frame_batch_size):
        votes.extend(visual_model(chunk).argmax(-1).tolist())
    audio_prediction = audio_model(mel_image).argmax(-1).item()
    return avfakenet_decision(votes, audio_prediction, tie_break=tie_break)
