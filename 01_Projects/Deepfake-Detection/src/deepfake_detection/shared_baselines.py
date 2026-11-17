"""Four shared-encoder controls using the primary benchmark's common manifest."""
import torch
from torch import nn
from .core import Decision
from .model import EvidenceModel


class SharedBaseline(nn.Module):
    WINDOW_SECONDS = 2.
    pool = EvidenceModel.pool
    _hybrid = staticmethod(EvidenceModel._hybrid)
    aggregate_video = EvidenceModel.aggregate_video

    def __init__(self, model_kind, visual_dim=1280, audio_dim=768, hidden=128,
                 dimension=128, quality_dim=2, decision=None):
        super().__init__()
        if model_kind not in ('visual_only', 'audio_only', 'concatenation', 'late_fusion'):
            raise ValueError('unsupported shared baseline')
        self.model_kind = model_kind
        self.decision = decision or Decision()
        self.quality_dim = quality_dim
        self.spec = dict(model_kind=model_kind, visual_dim=visual_dim, audio_dim=audio_dim,
                         hidden=hidden, dimension=dimension, quality_dim=quality_dim)
        if model_kind != 'audio_only':
            self.gru = nn.GRU(visual_dim, hidden, batch_first=True)
            self.visual = nn.Linear(hidden, dimension)
            self.visual_head = nn.Linear(dimension, 1)
        if model_kind != 'visual_only':
            self.audio = nn.Linear(audio_dim, dimension)
            self.audio_head = nn.Linear(dimension, 1)
        if model_kind == 'concatenation':
            self.head = nn.Linear(2*dimension, 1)
            del self.visual_head, self.audio_head

    def forward(self, visual_features, audio_features, visual_times, audio_times, quality):
        if not all(torch.isfinite(v).all() for v in (visual_features, audio_features)):
            raise ValueError('nonfinite baseline features')
        visual = audio = None
        if self.model_kind != 'audio_only':
            visual = self.visual(self.pool(self.gru(visual_features)[0], visual_times))
        if self.model_kind != 'visual_only':
            audio = self.audio(self.pool(audio_features, audio_times))
        streams = []
        if self.model_kind == 'concatenation':
            bins = self.head(torch.cat((visual, audio), -1)).squeeze(-1).sigmoid()
        else:
            if visual is not None:
                streams.append(self.visual_head(visual).squeeze(-1).sigmoid())
            if audio is not None:
                streams.append(self.audio_head(audio).squeeze(-1).sigmoid())
            bins = torch.stack(streams).mean(0)
        result = {'bin_scores': bins, 'window_scores': self._hybrid(bins, self.decision.top_k, self.decision.peak_weight)}
        if self.model_kind == 'late_fusion':
            result['stream_bins'] = torch.stack(streams)
        return result
