"""Source-time intervals, assessment and supervision (manuscript Eqs. 12–52)."""
from dataclasses import asdict, dataclass
import math


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def probability(value):
    finite(value)
    if not 0 <= value <= 1:
        raise ValueError("probability must lie in [0, 1]")
    return value


def interval(value):
    if len(value) != 2:
        raise ValueError("interval requires two endpoints")
    start, end = map(finite, value)
    if not 0 <= start < end:
        raise ValueError("interval must satisfy 0 <= start < end")
    return (start, end)


def union(intervals):
    result = []
    for start, end in sorted(interval(x) for x in intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap(left, right):
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def coverage(duration, intervals):
    finite(duration)
    if duration <= 0:
        raise ValueError("duration must be positive")
    assessed = union((max(0, a), min(duration, b)) for a, b in intervals
                     if max(0, a) < min(duration, b))
    uncovered, cursor = [], 0.0
    for start, end in assessed:
        if start > cursor:
            uncovered.append((cursor, start))
        cursor = end
    if cursor < duration:
        uncovered.append((cursor, duration))
    return {"assessed": assessed, "unassessed": uncovered,
            "covered_duration": sum(b - a for a, b in assessed)}


@dataclass(frozen=True)
class Decision:
    bins: int = 10
    top_k: int = 3
    peak_weight: float = 0.5
    top_ratio: float = 0.2
    video_peak_weight: float = 0.5
    window_threshold: float = 0.5
    video_threshold: float = 0.5
    reliability_temperature: float = 1.0

    def __post_init__(self):
        if type(self.bins) is not int or type(self.top_k) is not int or not 1 <= self.top_k <= self.bins:
            raise ValueError("require integer 1 <= top_k <= bins")
        for name in ("peak_weight", "top_ratio", "video_peak_weight", "window_threshold", "video_threshold"):
            probability(getattr(self, name))
        if self.top_ratio == 0 or finite(self.reliability_temperature) <= 0:
            raise ValueError("ratio and temperature must be positive")

    def to_dict(self):
        return asdict(self)


def hybrid(scores, k, weight):
    if not scores or not 1 <= k <= len(scores):
        raise ValueError("invalid top-k aggregation")
    probability(weight)
    selected = sorted((probability(x) for x in scores), reverse=True)[:k]
    return weight * selected[0] + (1 - weight) * math.fsum(selected) / k


def video_score(scores, decision):
    if not scores:
        return None
    k = max(1, math.ceil(decision.top_ratio * len(scores)))
    return hybrid(scores, k, decision.video_peak_weight)


def sigmoid(value):
    finite(value)
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    e = math.exp(value)
    return e / (1 + e)


def windows(duration, speech, *, has_audio=True, face_ok=None, length=2.0, stride=1.0, speech_min=0.5):
    """face_ok is an explicit per-window quality decision, never inferred here."""
    if finite(duration) <= 0 or finite(length) <= 0 or finite(stride) <= 0:
        raise ValueError("duration, length and stride must be positive")
    probability(speech_min)
    speech = union(speech)
    n = math.ceil(duration / stride)
    if face_ok is None or len(face_ok) != n or any(type(x) is not bool for x in face_ok):
        raise ValueError("explicit Boolean face quality decisions required for every window")
    output = []
    for i in range(n):
        start, end = i * stride, i * stride + length
        ratio = sum(overlap((start, end), s) for s in speech) / length if has_audio else 0.0
        reasons = []
        if end > duration:
            reasons.append("incomplete_window")
        if not has_audio:
            reasons.append("missing_audio")
        if ratio < speech_min:
            reasons.append("insufficient_speech")
        if not face_ok[i]:
            reasons.append("unusable_face_track")
        output.append(dict(window_id=i, interval=[start, end], status="UNASSESSED" if reasons else "USABLE",
                           reasons=reasons, speech_coverage=ratio))
    return output


def supervision(video_label, complete, manipulated, assessed_windows, bins=10):
    """None means unavailable supervision; labels never broadcast from a fake clip."""
    if type(video_label) is not int or video_label not in (0, 1) or type(complete) is not bool:
        raise ValueError("invalid label/completeness")
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be positive integer")
    edits = union(manipulated)
    if video_label == 0 and edits:
        raise ValueError("real video cannot contain manipulation intervals")
    usable = [interval(w) for w in assessed_windows]
    eligible = bool(usable) and (video_label == 0 or (complete and any(overlap(w, e) > 0 for w in usable for e in edits)))
    # A genuine video has known-negative windows even without temporal edits.
    temporal_known = complete or video_label == 0
    labels = []
    for start, end in usable:
        width = (end - start) / bins
        labels.append({"window": int(any(overlap((start, end), e) > 0 for e in edits)) if temporal_known else None,
                       "bins": [int(any(overlap((start+b*width, start+(b+1)*width), e) > 0 for e in edits))
                                if temporal_known else None for b in range(bins)]})
    return {"video_eligible": eligible, "windows": labels}
