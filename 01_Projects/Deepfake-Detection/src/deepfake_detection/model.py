"""FK neural path over cached features, not a raw-media inference pipeline.

Each batch contains usable two-second windows. Visual and audio features may
have different sequence lengths; feature-center timestamps align both to bins.
Module names are retained so existing state dictionaries remain compatible.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .core import Decision, finite


class EvidenceModel(nn.Module):
    """Combine visual, audio, cross-modal, and temporal evidence per bin."""

    WINDOW_SECONDS = 2.0

    def __init__(self, visual_dim=1280, audio_dim=768, hidden=128, quality_dim=2,
                 dimension=128, decision=None, disabled_evidence=(), uniform_weights=False):
        super().__init__()
        self.decision = decision or Decision()
        self.quality_dim = quality_dim
        if any(name not in ('av', 'temporal') for name in disabled_evidence) or len(set(disabled_evidence)) != len(disabled_evidence):
            raise ValueError('disabled evidence may contain av and/or temporal once')
        if type(uniform_weights) is not bool:
            raise ValueError('uniform_weights must be Boolean')
        self.disabled_evidence = tuple(disabled_evidence)
        self.uniform_weights = uniform_weights
        self.gru = nn.GRU(visual_dim, hidden, batch_first=True)
        self.visual = nn.Linear(hidden, dimension)
        self.audio = nn.Linear(audio_dim, dimension)
        self.vq, self.vk, self.vo, self.aq, self.ak, self.ao = [
            nn.Linear(dimension, dimension, bias=False) for _ in range(6)
        ]
        self.av = nn.Linear(4 * dimension, dimension)
        self.dv = nn.Linear(dimension, dimension)
        self.da = nn.Linear(dimension, dimension)
        self.temporal = nn.Linear(4 * dimension, dimension)
        self.summary = nn.Linear(4 * dimension, dimension)
        self.reliability = nn.Sequential(
            nn.Linear(dimension + quality_dim, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 4),
        )
        self.heads = nn.ModuleList([nn.Linear(dimension, 1) for _ in range(4)])
        if 'av' in self.disabled_evidence:
            for module in (self.vq, self.vk, self.vo, self.aq, self.ak, self.ao, self.av, self.heads[2]):
                module.requires_grad_(False)
        if 'temporal' in self.disabled_evidence:
            for module in (self.dv, self.da, self.temporal, self.heads[3]):
                module.requires_grad_(False)
        if self.uniform_weights:
            self.summary.requires_grad_(False)
            self.reliability.requires_grad_(False)

    def pool(self, sequence, timestamps):
        """Mean features in bins; reject missing bins rather than invent data.

        sequence: [windows, feature centers, channels]. timestamps: shared
        [feature centers] or per-window [windows, feature centers], in seconds.
        """
        if sequence.ndim != 3:
            raise ValueError("features must have shape [windows, time, channels]")
        batch, length, _ = sequence.shape
        if timestamps.ndim == 1:
            timestamps = timestamps.unsqueeze(0).expand(batch, -1)
        if (
            timestamps.shape != (batch, length)
            or timestamps.device != sequence.device
            or not torch.isfinite(timestamps).all()
            or not ((timestamps >= 0) & (timestamps < self.WINDOW_SECONDS)).all()
        ):
            raise ValueError("invalid feature-center timestamps")

        indices = torch.floor(
            timestamps * self.decision.bins / self.WINDOW_SECONDS
        ).long()
        # A small bin-membership matrix pools all channels in one batched product.
        membership = F.one_hot(indices, num_classes=self.decision.bins)
        membership = membership.transpose(1, 2).to(sequence.dtype)
        counts = membership.sum(-1, keepdim=True)
        if (counts == 0).any():
            raise ValueError("empty feature bin; reject upstream rather than silently fill")
        return membership.bmm(sequence) / counts

    @staticmethod
    def _hybrid(scores, k, peak_weight):
        """Blend the maximum with the top-k mean along the last axis."""
        top_scores = scores.topk(k, dim=-1, sorted=True).values
        return peak_weight * top_scores[..., 0] + (1 - peak_weight) * top_scores.mean(-1)

    def forward(self, visual_features, audio_features, visual_times, audio_times, quality):
        """Return bin/window scores and their four signed evidence contributions."""
        if visual_features.ndim != 3 or audio_features.ndim != 3:
            raise ValueError("features must have shape [windows, time, channels]")
        batch = visual_features.shape[0]
        if batch == 0 or audio_features.shape[0] != batch:
            raise ValueError("visual and audio batches must match and be nonempty")
        if quality.shape != (batch, self.quality_dim):
            raise ValueError("invalid quality shape")
        if not all(torch.isfinite(x).all() for x in (visual_features, audio_features, quality)):
            raise ValueError("nonfinite model input")

        # No hidden state is passed: the GRU resets for each window.
        states, _ = self.gru(visual_features)
        visual = self.visual(self.pool(states, visual_times))
        audio = self.audio(self.pool(audio_features, audio_times))
        bins, dimension = visual.shape[1:]
        positions = torch.arange(bins, device=visual.device)
        outside_neighborhood = (positions[:, None] - positions[None, :]).abs() > 1

        def attend(query, key, value):
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(dimension)
            weights = scores.masked_fill(outside_neighborhood, float("-inf")).softmax(-1)
            return weights @ value

        cross_modal = torch.zeros_like(visual)
        if 'av' not in self.disabled_evidence:
            visual_context = attend(self.vq(visual), self.ak(audio), self.ao(audio))
            audio_context = attend(self.aq(audio), self.vk(visual), self.vo(visual))
            cross_modal = self.av(torch.cat((
                (visual - visual_context).abs(), visual * visual_context,
                (audio - audio_context).abs(), audio * audio_context,
            ), dim=-1))

        temporal = torch.zeros_like(visual)
        if 'temporal' not in self.disabled_evidence:
            visual_change = self.dv(visual[:, 1:] - visual[:, :-1])
            audio_change = self.da(audio[:, 1:] - audio[:, :-1])
            temporal = self.temporal(torch.cat((
                visual_change, audio_change,
                (visual_change - audio_change).abs(), visual_change * audio_change,
            ), dim=-1))
            temporal = torch.cat((torch.zeros_like(visual[:, :1]), temporal), dim=1)

        evidence = (visual, audio, cross_modal, temporal)
        if self.uniform_weights:
            reliability = torch.ones_like(visual[..., :1]).expand(-1, -1, 4)
        else:
            summary = self.summary(torch.cat(evidence, dim=-1))
            quality_per_bin = quality[:, None, :].expand(-1, bins, -1)
            reliability = self.reliability(torch.cat((quality_per_bin, summary), dim=-1)).sigmoid()
        active = torch.tensor([True, True, 'av' not in self.disabled_evidence,
                               'temporal' not in self.disabled_evidence], device=visual.device)
        weight_logits = reliability / self.decision.reliability_temperature
        alpha = weight_logits.masked_fill(~active, float('-inf')).softmax(-1)
        logits = torch.cat([head(value) for head, value in zip(self.heads, evidence)], dim=-1)
        contributions = alpha * logits
        bin_scores = contributions.sum(-1).sigmoid()
        window_scores = self._hybrid(bin_scores, self.decision.top_k, self.decision.peak_weight)
        return {
            "bin_scores": bin_scores,
            "window_scores": window_scores,
            "alpha": alpha,
            "contributions": contributions,
            "reliability": reliability,
            "evidence_logits": logits,
            "temporal_evidence": temporal,
        }

    def aggregate_video(self, window_scores):
        """Aggregate usable windows from exactly one video, retaining gradients."""
        if window_scores.ndim != 1 or window_scores.numel() == 0:
            raise ValueError("one nonempty video's usable windows required")
        k = max(1, math.ceil(self.decision.top_ratio * window_scores.numel()))
        return self._hybrid(window_scores, k, self.decision.video_peak_weight)


def _supervised_bce(scores, labels):
    """Sum only known targets; None never becomes a real-class target."""
    if any(label is not None and (type(label) is not int or label not in (0, 1))
           for label in labels):
        raise ValueError("supervision targets must be 0, 1 or None")
    known = torch.tensor(
        [label is not None for label in labels], dtype=torch.bool, device=scores.device
    )
    targets = scores.new_tensor([0 if label is None else label for label in labels])
    # Empty selections sum to differentiable zero, with no per-label GPU calls.
    return F.binary_cross_entropy(scores[known], targets[known], reduction="sum")


def multilevel_loss(outputs, video_probability, video_label, supervision,
                    window_weight=1.0, bin_weight=1.0):
    """Sum video, known-window and known-bin BCE terms (Eq. 52)."""
    if type(video_label) is not int or video_label not in (0, 1):
        raise ValueError("video label must be 0 or 1")
    if finite(window_weight) < 0 or finite(bin_weight) < 0:
        raise ValueError("loss weights must be nonnegative")
    windows, bins = outputs["window_scores"], outputs["bin_scores"]
    labels = supervision["windows"]
    if windows.ndim != 1 or bins.ndim != 2 or bins.shape[0] != len(windows):
        raise ValueError("invalid window/bin score shapes")
    if len(labels) != len(windows):
        raise ValueError("supervision/window count mismatch")
    if any(len(row["bins"]) != bins.shape[1] for row in labels):
        raise ValueError("supervision/bin count mismatch")

    loss = window_weight * _supervised_bce(windows, [row["window"] for row in labels])
    loss = loss + bin_weight * _supervised_bce(
        bins.reshape(-1), [label for row in labels for label in row["bins"]]
    )
    if supervision["video_eligible"]:
        loss = loss + F.binary_cross_entropy(
            video_probability, video_probability.new_tensor(float(video_label))
        )
    return loss
