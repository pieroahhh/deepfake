# AVFakeNet: current recovery scope and implementation map

Updated 17 November 2026. This supersedes the initial recommendation to restart
both entire networks from random weights. The full validation pilot is ON HOLD.
Original checkpoints, predictions and diagnostic reports remain unchanged.

## Machine A update: input bundle verified and paired diagnostics completed

The attempt-003 inputs are now available locally in
`C:\Users\teeku\Downloads\LANDrop\return-b-recovery-probe-inputs-001`.
Machine A verified all 19 payload checksums, labels and checkpoint bindings, and
independently reproduced the pooled grouped-probe BA values: visual 68.75%, audio
53.125%. See `runs/avfakenet-recovery-input-audit-002/REVIEW.md`.

Machine A subsequently completed paired raw-versus-standardized head diagnostics
with the same head initialization and 200-update budget. Raw heads stayed at 50%
tiny-training BA. Standardized heads reached 88.28% visual / 87.50% audio and passed
the head-training loss/BA targets. Grouped head-fold BA was 75.00% / 65.63%.
These are training-only diagnostics with a previously trained backbone, not
held-out results. Original end-to-end gates and the full pilot remain on hold.
See `runs/avfakenet-head-scaling-001/REPORT.md` for evidence and the next boundary.

## Original visual experiment specification

The operator reported grouped-source-family tiny-set linear-probe balanced
accuracy of 68.8% for visual and 53.1% for audio after 200 updates. These values
were initially reported diagnostics; they have now been reproduced as noted above.

Use the **visual feature extractor from recovery attempt 003**, not the original
failed visual epoch-1 checkpoint. Reinitialize only the existing classifier
`DSTNet.output_block`, including its upstream linear layers before the dead ReLU.
Keep its architecture, activations, class semantics and loss unchanged.

Freeze `input_block`, `patch_embedding`, `stages` and `merges`. Keep the feature
extractor in evaluation mode, with only `output_block` in training mode. Setting
`requires_grad=False` alone does not disable dropout. Optimize classifier
parameters only for exactly 200 updates in a new attempt directory.

Use the exact frozen examples, targets, preprocessing, source hashes and fold
assignments from the earlier probe. If reporting comparison with grouped-fold
68.8%, preserve the same out-of-fold evaluation; do not compare in-sample training
accuracy against the grouped-fold figure. Keep sanity training and grouped-fold
results separately named.

Check feature-extractor tensor hashes before and after, and verify that classifier
weights changed, loss/gradients are finite, activations are alive and predictions
respond to examples. Record sample, parent checkpoint and configuration hashes.
Do not substitute another checkpoint or recreate a different sample set if the
required attempt-003 artifacts are absent.

## Implementation map

| Location | Required change |
|---|---|
| `baselines/avfakenet/model.py` | Optionally expose pooled feature extraction separately from `output_block` without changing numerical operations; support head reset/freeze explicitly. Do not first replace ReLU with another activation. |
| `baselines/avfakenet/pipeline.py` | Add a separately named classifier-refit mode with exact frozen samples, 200 updates and classifier-only optimizer parameters. Leave the original full-training path available. |
| `baselines/common/training.py` | Add sustained health telemetry; current gradient diagnostics record only the first update of each split. Keep metric selection direction explicit if balanced accuracy is added. |
| AVFakeNet evaluation path | Export raw per-branch probabilities and confusion/coverage independently. Keep the original hard-OR fused score policy; branch AUC and fused hard-vote AUC are different quantities. |
| `baselines/avfakenet/preprocessing.py` | Audit exact whole-clip Mel inputs against annotated source intervals before changing preparation. Any windowed-input experiment must have a new version and temporal targets. |
| `baselines/common/media.py` | Later review whole-video rejection and the face helper's unnecessary lip requirement for this baseline. Do not silently relax shared FK/AVTENet eligibility or pad absent evidence. |
| `src/deepfake_detection/baselines.py` | Retain original majority-vote and OR rules as a reference. A localized Top-K/window variant needs a separate declared method and validation-only selection. |

## Correction to the earlier diagnosis wording

The current `baselines/common/training.py` already applies
`clip_grad_norm_(..., schedule.get('gradient_clip_norm', 1.), ...)` before the
optimizer step. **It is not correct to claim that this current trainer lacks
gradient clipping.** The returned original bundle does not bind the exact source
revision used on B, so an absence of clipping in that original run has not been
established. Earlier diagnostic wording that lists "without ... clipping" as a
training risk is superseded by this clarification.

Inspect and record the actual training source and pre/post-clipping gradient
norms. Do not add a duplicate clipping operation or treat its absence as the
confirmed cause. The observed dead classifier path and nearly invariant audio
probabilities remain verified findings; their initiating training cause is not
uniquely proven.

## Audio work remains inspection-only

Inspect annotated fake-audio training clips with the exact recovery preparation
profile. Measure edited duration, resized Mel columns and STFT support; show
whole-clip input and an annotation-aligned diagnostic zoom. Geometry/appearance
alone cannot prove that a manipulation artifact is discriminative.

If edits are diluted, test a separately versioned windowed-Mel preparation on
training data with source-time targets and source-family grouping. If meaningful
input variation survives preparation but representations remain weak, investigate
the audio model/optimization next. Do not fit a threshold to a constant score or
start a full audio retrain before deciding which component to investigate.

## Release gate

The full validation pilot stays on hold until both branches pass their declared
training sanity checks. A visual-only success or a completed process does not
release that gate. Recovery remains exploratory because the old held-out results
have already been inspected.

Machine A now has the recovery attempt-003 weights and exact frozen probe sample
pack. Keep using those verified artifacts; do not replace them with the old
diagnostic bundle. No full-pilot release is implied by a head-only target pass.
