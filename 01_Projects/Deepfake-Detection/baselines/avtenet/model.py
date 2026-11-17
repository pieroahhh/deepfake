"""AVTENet feature-fusion components, not a complete pretrained ensemble.

Actual factorized ViViT, AST, and AV-HuBERT backends are deliberately external.
No generic/random transformer is silently substituted for those pretrained models.
See configs/avtenet.components-v1.json and the paper audit for chosen dimensions.
"""

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from deepfake_detection.cli import load_json

EXPERTS = ("visual", "audio", "audiovisual")


def load_component_profile(path):
    profile = load_json(path)
    if profile["profile_id"] not in ("avtenet-ff-avhubert-components-v1","avtenet-hf-vivit-joint-adaptation-v1"):
        raise ValueError("unsupported AVTENet component profile")
    expected_variant=('feature_fusion_hf_joint_vivit_ast_avhubert' if profile['profile_id']=='avtenet-hf-vivit-joint-adaptation-v1'
                      else 'feature_fusion_vivit_ast_avhubert')
    if profile["variant"] != expected_variant:
        raise ValueError("only the AV-HuBERT feature-fusion variant is implemented")
    dimensions = profile["expert_feature_dimensions"]
    if set(dimensions) != set(EXPERTS) or any(type(dimensions[k]) is not int or dimensions[k] < 1 for k in EXPERTS):
        raise ValueError("explicit positive dimensions required for all three experts")
    head = profile["temporal_head"]
    if head["kernels"] != [7, 5, 3] or head["convolution_groups_per_block"] != 2 or head["blocks"] != 1:
        raise ValueError("unsupported temporal head structure")
    if head["input_dimension"] != 768 or head["branch_channels"] != 256 or head["residual_channels"] != 768:
        raise ValueError("v1 explicitly resolves the residual channel mismatch to 768")
    if head["pooling"] != "unweighted_mean_no_padded_sequences" or head["padding"] != "symmetric_same":
        raise ValueError("unsupported temporal preparation/pooling policy")
    if not 0 <= head["dropout"] < 1 or head["prelu_parameters"] != 1:
        raise ValueError("invalid temporal regularization settings")
    fusion = profile["fusion"]
    if (fusion["feature_order"] != list(EXPERTS) or fusion["input_dimension"] != sum(dimensions.values())
            or fusion["output_dimension"] != 1 or not fusion["experts_frozen"]
            or fusion["classifier"] != "single_linear_to_real_logit"):
        raise ValueError("unsupported feature fusion settings")
    if dimensions["audiovisual"] != head["residual_channels"]:
        raise ValueError("AVN feature dimension must match temporal head output")
    fingerprint(profile)  # Reject nonfinite profile values.
    return profile


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def expert_target(expert, *, visual_label, audio_label, video_label):
    """External 1=fake labels -> native 1=real targets; preserve unknowns."""
    if expert not in EXPERTS:
        raise ValueError("unknown expert")
    labels = (visual_label, audio_label, video_label)
    if any(value is not None and (type(value) is not int or value not in (0, 1)) for value in labels):
        raise ValueError("labels must be 0, 1 or None")
    if video_label == 0 and 1 in (visual_label, audio_label):
        raise ValueError("genuine video contradicts a manipulated modality")
    if visual_label is not None and audio_label is not None and video_label is not None:
        if video_label != int(bool(visual_label or audio_label)):
            raise ValueError("contradictory video and modality labels")
    label = {"visual": visual_label, "audio": audio_label, "audiovisual": video_label}[expert]
    return None if label is None else 1 - label


def _validate_vectors(features, dimension):
    if (features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != dimension
            or not features.is_floating_point() or not torch.isfinite(features).all()):
        raise ValueError(f"expected nonempty finite [batch, {dimension}] features")
    return features


class VectorExpert(nn.Module):
    """Linear expert head over an explicitly supplied vector-producing backend.

    The backend must return the declared penultimate vector, not class scores.
    Selecting and loading an actual ViViT/AST checkpoint is outside this wrapper.
    """

    def __init__(self, backend, feature_dimension):
        super().__init__()
        self.backend = backend
        self.feature_dimension = feature_dimension
        self.classifier = nn.Linear(feature_dimension, 1)

    def extract_features(self, inputs):
        return _validate_vectors(self.backend(inputs), self.feature_dimension)

    def forward(self, inputs):
        return self.classifier(self.extract_features(inputs)).squeeze(-1)


class MultiScaleTemporalHead(nn.Module):
    """Two multiscale conv groups and a residual projection, then mean pooling.

    Inputs: full-length, aligned [batch, time, 768] AV-HuBERT outputs, with no
    padded time positions. Three 256-channel branches yield 768, explicitly
    choosing 768 instead of the paper figure's inconsistent 756 notation.
    """

    def __init__(self, profile):
        super().__init__()
        settings = profile["temporal_head"]
        self.input_dimension = settings["input_dimension"]
        self.feature_dimension = settings["residual_channels"]
        self.groups = nn.ModuleList()
        previous = self.input_dimension
        for _ in range(settings["convolution_groups_per_block"]):
            branches = nn.ModuleList()
            for kernel in settings["kernels"]:
                branches.append(nn.Sequential(
                    nn.Conv1d(previous, settings["branch_channels"], kernel, padding=kernel // 2),
                    nn.BatchNorm1d(settings["branch_channels"], eps=settings["batch_norm_epsilon"],
                                   momentum=settings["batch_norm_momentum"]),
                    nn.PReLU(settings["prelu_parameters"]),
                ))
            self.groups.append(branches)
            previous = self.feature_dimension
        self.dropout = nn.Dropout(settings["dropout"])
        self.residual = nn.Conv1d(self.input_dimension, self.feature_dimension, 1)

    def forward(self, sequence):
        if (sequence.ndim != 3 or sequence.shape[-1] != self.input_dimension
                or sequence.shape[0] == 0 or sequence.shape[1] == 0
                or not sequence.is_floating_point() or not torch.isfinite(sequence).all()):
            raise ValueError("expected finite nonempty [batch, time, 768] unpadded features")
        if self.training and sequence.shape[0] * sequence.shape[1] < 2:
            raise ValueError("temporal BatchNorm training requires at least two samples across batch/time")
        inputs = sequence.transpose(1, 2)
        features = inputs
        for branches in self.groups:
            features = self.dropout(torch.cat([branch(features) for branch in branches], dim=1))
        return (features + self.residual(inputs)).mean(-1)


class AudioVisualExpert(nn.Module):
    """Attach the declared temporal head to an actual AV-HuBERT backend adapter.

    The supplied backend is responsible for returning [batch,time,768] features
    from paired lips/filterbanks and for rejecting padding or missing modalities.
    """

    def __init__(self, backend, profile):
        super().__init__()
        self.backend = backend
        self.temporal_head = MultiScaleTemporalHead(profile)
        self.feature_dimension = self.temporal_head.feature_dimension
        self.classifier = nn.Linear(self.feature_dimension, 1)

    def extract_features(self, inputs):
        return self.temporal_head(self.backend(inputs))

    def forward(self, inputs):
        return self.classifier(self.extract_features(inputs)).squeeze(-1)


class FeatureFusion(nn.Module):
    """One linear classifier on concatenated penultimate expert vectors."""

    def __init__(self, profile):
        super().__init__()
        self.dimensions = dict(profile["expert_feature_dimensions"])
        self.classifier = nn.Linear(sum(self.dimensions.values()), 1)

    def forward(self, features):
        if set(features) != set(EXPERTS):
            raise ValueError("all three named penultimate feature tensors are required")
        vectors = [_validate_vectors(features[name], self.dimensions[name]) for name in EXPERTS]
        if len({value.shape[0] for value in vectors}) != 1:
            raise ValueError("expert batches must match")
        return self.classifier(torch.cat(vectors, dim=-1)).squeeze(-1)


class FrozenExpertFusion(nn.Module):
    """Stage-two wrapper: keep experts frozen AND in evaluation mode.

    Expert checkpoints must already be selected/fixed. This class enforces
    freezing but cannot certify checkpoint training or dataset provenance.
    """

    def __init__(self, experts, profile):
        super().__init__()
        if set(experts) != set(EXPERTS):
            raise ValueError("visual, audio and audiovisual experts are required")
        for name, expert in experts.items():
            if not hasattr(expert, "extract_features") or expert.feature_dimension != profile["expert_feature_dimensions"][name]:
                raise ValueError(f"incompatible {name} expert feature interface")
        self.experts = nn.ModuleDict(experts)
        self.fusion = FeatureFusion(profile)
        for expert in self.experts.values():
            expert.requires_grad_(False)
            for parameter in expert.parameters():
                parameter.grad = None
            expert.eval()

    def train(self, mode=True):
        super().train(mode)
        for expert in self.experts.values():
            expert.eval()  # Prevent BatchNorm statistics/dropout changes in stage two.
        return self

    def forward(self, inputs):
        if set(inputs) != set(EXPERTS):
            raise ValueError("explicit inputs required for all three experts")
        with torch.no_grad():
            features = {name: expert.extract_features(inputs[name]) for name, expert in self.experts.items()}
        return self.fusion(features)


def native_binary_loss(real_logits, external_labels):
    """Stable scalar BCE, converting common 1=fake targets to native 1=real."""
    if (real_logits.ndim != 1 or real_logits.numel() == 0
            or external_labels.shape != real_logits.shape
            or external_labels.dtype != torch.long
            or not ((external_labels == 0) | (external_labels == 1)).all()
            or not torch.isfinite(real_logits).all()):
        raise ValueError("expected finite real logits and one binary external target per sample")
    return F.binary_cross_entropy_with_logits(real_logits, 1 - external_labels.to(real_logits.dtype))


def fake_scores(real_logits):
    """The scalar classifier estimates native real; external fake is its complement."""
    if not real_logits.is_floating_point() or not torch.isfinite(real_logits).all():
        raise ValueError("real logits must be finite floating-point values")
    return (-real_logits).sigmoid()


def cache_header(profile, checkpoint_hashes, *, preprocessing_sha256, split_sha256):
    """Bind feature-cache identity to all expert weights, preprocessing and split."""
    if set(checkpoint_hashes) != set(EXPERTS):
        raise ValueError("checkpoint hashes required for all three experts")
    hashes = [*checkpoint_hashes.values(), preprocessing_sha256, split_sha256]
    if any(not isinstance(value, str) or len(value) != 64
           or any(character not in '0123456789abcdef' for character in value) for value in hashes):
        raise ValueError("cache provenance requires lowercase SHA-256 digests")
    return {"schema": "avtenet-feature-cache-v1", "profile_sha256": fingerprint(profile),
            "expert_checkpoint_sha256": dict(checkpoint_hashes),
            "preprocessing_sha256": preprocessing_sha256, "split_sha256": split_sha256}


def verify_cache_header(actual, expected):
    """Expected provenance must be independently derived from current artifacts."""
    if fingerprint(actual) != fingerprint(expected):
        raise ValueError("stale or incompatible AVTENet feature cache")
