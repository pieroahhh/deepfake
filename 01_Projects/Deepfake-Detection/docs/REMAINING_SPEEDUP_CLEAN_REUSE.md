# Verified clean-reference reuse for remaining robustness runs

Added on 17 November 2026 while `video_mean` was training. The active aggregation
runner and all completed models, results and caches were left unchanged.

The robustness population has 575 method/clip cases: 100 clean, 400 compression
derivatives and 75 genuine audio-delay stress cases. AVFakeNet and AVTENet can
reuse their original clean references on those same 100 source files.

- AVFakeNet verifies retained prediction/checkpoint identity, source SHA256 and
  labels before returning the original prediction, including null assessments.
- Each AVTENet expert verifies its original completed feature-store checksum,
  source identity and valid window keys before reading retained clean features.
  Fusion still uses the locked fusion checkpoint and the original threshold.
- Changed/compressed/delayed files undergo new preparation and inference.
- Profiling remains raw inference with one warmup and three timed passes. It
  never invokes clean-reference reuse. Reused robustness timings are null,
  so they cannot be misrepresented as newly measured inference runtimes.

This removes 100/575 (17.4%) raw cases from the two affected robustness adapters;
it is not a measured 17.4% reduction in total job time. The benefit depends on
source duration, usable windows and each model's preparation cost. No new
wall-clock estimate is claimed before measurement.

Two checks pass: original expert vector preservation with wrong-source/window
rejection, and retaining null clean predictions with modified-source rejection.
The actual retained AVFakeNet model identity was also checked against its
completed training identity. Future AVTENet stage identities include the modified
runner script digest; no existing AVTENet study outputs were present at edit time.

These helpers are imported/launched later by the already running supplement
queue. The current training worker does not need to be interrupted or restarted.
