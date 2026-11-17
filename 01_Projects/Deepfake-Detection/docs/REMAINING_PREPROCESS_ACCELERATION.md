# Further preprocessing acceleration

Applied while the continuation was executing FK robustness. The active FK worker,
its code/configuration and completed outputs were not interrupted or modified.

## AVFakeNet: unchanged modality inputs

For robustness only, exact compressed stream/configuration/timeline identity is
checked against the original source. Audio-only derivatives can reuse the
original hash-checked visual array or original visual preparation exclusion.
Video-only derivatives can reuse the original hash-checked Mel array or original
audio preparation exclusion. Source duration, labels and original source hashes
must also match. If identity cannot be established, normal raw preparation runs.

This can avoid repeated MTCNN/tracking/crop work on the 275 audio-change cases and
Mel work on the 200 video-change cases. The classifiers still use the same frame
batch of eight and the same inputs/voting logic. Previously excluded branches
remain excluded; reuse must not silently improve coverage.

Clean predictions were already reused with source/checkpoint verification.
Profiling rows have no robustness condition, so they cannot enter this shortcut.
Measured profiling continues to include raw preparation for every pass.

One runnable test checks exact array reuse, changed-array rejection, changed-stream
fallback and raw profiling selection. Real retained-input checks are recorded in
`runs/remaining-supplement-002/avfakenet-modality-reuse-checks.json`. One sampled
real audio branch was originally excluded for incomplete coverage; the same
exclusion is intentionally retained.

## AVTENet: bounded CPU decode prefetch

Visual and audiovisual raw inference now enable the existing one-chunk CPU decode
lookahead. Detection/tracking and model execution remain ordered on the main
thread. The retained AV-HuBERT runtime was checked to support this option.

Seven existing tests passed, covering real CFR/VFR frame parity, overlap/tails,
gaps, crops, request ordering, cleanup on errors and bounded worker/lookahead.
These tests verify behavior; no total-runtime speedup percentage is claimed.

Profiling measures this actual optimized pipeline and still performs one warmup
and three timed repetitions. Model settings and precision are unchanged. Keep
the previously completed FK profiling results and explain per-method execution
settings when comparing measurements.

## Scope and operational behavior

AVFakeNet is imported only after the current FK stage, and the AVTENet study
script is launched later. Changes therefore apply automatically to future stages.
The active worker is not restarted. GPU-heavy model jobs remain sequential on
the single 8 GiB GPU to avoid contention/OOM and corrupting isolated profiling.
Actual overall wall-time savings remain unmeasured.
