# Optional CPU video decode overlap

The frozen AVTENet inference CLI accepts `--prefetch-video-decode`. It is disabled
by default. With the flag enabled, one CPU worker decodes the next chunk while
the calling thread detects faces and crops the current chunk. Lookahead is at
most one chunk, with at most one additional second of sampled RGB frames held
ahead of the consumer. Memory depends on source resolution.

Frame timestamps, frame counts, crop/tracking rules, model weights, precision,
batch size, threshold and output order are unchanged. Face detection and model
execution remain on the calling thread. Worker exceptions are delivered in order;
closing an iterator waits for an active decode before its reader is reused.

The option is available in `scripts/run_avtenet_frozen_test.py`. Add the flag to a
future frozen-inference command after benchmarking it on representative clips.
The automatic continuation controller does not enable it. Existing commands
retain their previous behavior.

CPU tests cover exact decoded pixels and timestamps across constant/variable
frame rates, overlapping/backwards/tail requests, source gaps, exact face/lip
crops, real overlap, bounded lookahead and thread cleanup after failures.
They also cover existing cache validation, input/RNG order and CPU training parity.
Run `test_decode_prefetch.py` with `src`, the repository root and `tests` on the
Python import path. Tests do not establish an end-to-end speedup.

Before production use, measure full throughput and peak memory in an idle GPU
slot using the same clips and selected checkpoints; compare prepared inputs,
features and failure decisions. Validate the pinned AV-HuBERT Docker runtime
separately. Leave the flag disabled if it gives no measurable benefit.

## Deployment on Machine A, 16 November 2026

Applied to the main `Deepfake-Detection` repository only. The active run uses the
separate `avh-a/code` checkout and continues with its existing code. Its controller,
selected checkpoints, completed results, inputs and caches were not changed.
The initial main-repository deployment did not restart the active process.
The user subsequently authorized a safe pause, validation and resume. The active
checkout received the opt-in code after its partial feature store was backed up.
Seven pinned-runtime decode tests and 15 exact saved-feature comparisons passed.
With warmup excluded, the small A/B benchmark measured 18.945 seconds without
prefetch and 17.797 seconds with it (about 6.4% higher throughput on that sample).
The resume controller enables prefetch and skips all completed stages. No
Docker/WSL service restart or resource-allocation change was needed.

Pre-change source backups and verification evidence are stored outside the repo
at `../avh-a/main-repo-before-decode-prefetch-001`. The original isolated candidate
and patch are in `../avh-a/optimization-candidate-001`.
Live continuation evidence, backups and an English handoff are in
`../avh-a/live-decode-prefetch-001`.
