# Remaining experiments: prepared execution package

Prepared on 17 November 2026. This package does not start training, raw-media
inference, robustness transforms or a second GPU workload. The active AVFakeNet
held-out evaluation continues separately. Existing results/checkpoints remain
unchanged. JSON protocol files are sealed by `SHA256SUMS.json`; version the package
rather than editing those files after results become visible.

## Recommended order

1. Finish `runs/avfakenet-heldout-002` and inspect the three-model comparison.
2. Use the existing FK localization results and four shared controls.
3. Run paired source-family bootstrap on the final common assessed predictions.
4. Generate and verify real source-bound FK forensic records.
5. After GPU is free, retrain `full`, `no_av`, `no_temporal`, `uniform_weights`.
6. Run aggregation variants and fair runtime/memory profiling.
7. Run robustness, then cross-dataset transfer when the data/protocol are ready.

These follow-ups are exploratory because the existing held-out results have
already been inspected. They are not fresh confirmatory experiments.

## What is already available

- Full FK and Visual-only, Audio-only, Concatenation, Late fusion held-out results
  exist. Do not rerun their expensive preparation merely to make a comparison.
- `localization-summary.json` summarizes the existing FK source-aligned timeline
  adapter, using the already selected threshold. It keeps unassessed manipulated
  time in the full-duration denominator. It is not a new localization model.
- `profiles/` contains eight model configs and explicit validation grids. Epoch
  budget is five, seed 7, matching the original FK head-training budget. Other
  training settings come from the actual recorded FK run.
- `cache-preflight.json` records access as seen by the restricted tool process.
  `native-cache-preflight.json`, when present, records a read-only probe under the
  normal Windows user. A successful first-cache probe is not a full cache audit;
  the trainer still verifies every loaded cache during training.

The native-user preflight has now **PASSED**. The existing trusted FK training
cache manifest contains **2,912 train / 485 validation entries**. This is the
same cached population used by the recorded FK run; it must not be described as
3,000/500 usable videos merely because the raw attempted manifest has those
counts. Preserve the original preparation/exclusion accounting. The preflight
read and verified one cache from each split and started no GPU work.

Localization analysis has also been generated in
`runs/remaining-offline-001/localization`, including summary JSON, a readable
report and visually checked four-example PNG. It uses the existing frozen
threshold and includes a missed visual manipulation example; examples were not
chosen for successful predictions. This run is an analysis of retained outputs,
not a new raw-media inference pass.

## Common setup

Open ordinary PowerShell. All following commands run from this repository:

```powershell
Set-Location -LiteralPath 'C:\Users\teeku\Desktop\fk\Deepfake-Detection'
$Py = Join-Path (Get-Location) '.venv-gpu\Scripts\python.exe'
```

## 1. Localization: existing outputs, CPU only

```powershell
& $Py scripts/analyze_remaining_offline.py --task localization --output '.\runs\remaining-offline-001\localization'
```

This produces an aggregate duration-weighted summary and four timeline examples
selected by a fixed metadata/hash rule, not prediction success. Red spans are
annotations, gray spans are unassessed, and the dashed line is the existing
validation-selected threshold. Report assessed-region precision/recall/F1/IoU,
total time coverage, manipulated-time coverage and detected share of all edited
time separately. Native AVFakeNet/AVTENet localization is “not provided.”

If this output already exists, inspect it instead of rerunning into the same
folder. Use a new output name for a distinct analysis.

## 2. Paired confidence intervals: after held-out completes

```powershell
& $Py scripts/analyze_remaining_offline.py --task bootstrap --output '.\runs\remaining-offline-001\bootstrap'
```

Uses the existing reporting implementation, 2,000 source-family resamples, seed 7,
paired across FK/default AVTENet/current AVFakeNet on their common assessed clips.
Report differences in BA/F1 and their 95% intervals. These intervals measure test
population sampling, not independent training-seed variability. The calibrated
AVTENet row remains a separate post-inspection supplementary analysis.

## 3. Retrained ablations: GPU after the active test

First check read access without starting training:

```powershell
& $Py scripts/run_prepared_ablation.py --variant full --preflight-only
```

Then start **one** variant at a time:

```powershell
& $Py scripts/run_prepared_ablation.py --variant full
& $Py scripts/run_prepared_ablation.py --variant no_av
& $Py scripts/run_prepared_ablation.py --variant no_temporal
& $Py scripts/run_prepared_ablation.py --variant uniform_weights
```

Do not paste all four as unattended sequential commands unless you intend all
four runs. The launcher refuses training before the current held-out completion
marker exists. Each variant trains from a matched seeded initialization using
the original frozen encoder cache. It does not fine-tune or recompute encoders.
Results go to a fresh `runs/remaining-ablations-001/<variant>/attempt-001`.

The fresh `full` run is a matched reference for the current code/environment.
It does not replace the completed historical FK checkpoint. Selection uses
validation only; the runner does not evaluate test automatically.

Optional aggregation runs:

```powershell
& $Py scripts/run_prepared_ablation.py --variant window_max
& $Py scripts/run_prepared_ablation.py --variant window_mean
& $Py scripts/run_prepared_ablation.py --variant video_max
& $Py scripts/run_prepared_ablation.py --variant video_mean
```

Their grids lock the axis being studied: window max, whole-window mean, video
max, or whole-video mean. Other aggregation settings may use the documented
validation grid. Selected fixed parameters are asserted after calibration.
Test-time channel zeroing is a sensitivity intervention, not this retrained
ablation. Independent seeds 17/27 remain optional future runs, not claimed results.

The prepared launcher does not provide whole-plan or automatic crash resume.
Keep interrupted outputs/checkpoints. The underlying trainer has a resume API,
but continuation into a fresh attempt and matching selection-grid metadata must
be prepared explicitly before resuming; do not overwrite an old run.

## 4. Forensic record verification

`forensic-plan.json` fixes metadata-selected source examples and case types.
The intended procedure is:

1. Run the actual `infer-fk` raw-media path with the selected checkpoint, verified
   local assets and matching measured-quality profile to generate a real record.
2. Retain the original commitment independently of the writable record copy.
3. Use `deepfake_detection.cli verify --record ... --trusted-commitment ...
   --source ... --artifact-root ...` on the unmodified record.
4. Modify separate copies: source bytes, one derived artifact, a contribution,
   timestamp, threshold and coverage field. Verify expected rejection and retain
   expected/observed outcomes and logs.
5. Include no usable windows, partial coverage, no audio, short media and a
   near-threshold software fixture. Synthetic fixtures prove software behavior,
   not real-media predictive performance.

The bulk real-record case runner is **not yet implemented in this package**.
Real record/commitment paths must be obtained from actual raw inference, not
fabricated from classification summaries. Do not claim a completed forensic
experiment from this plan alone. Integrity/reconstruction does not prove that
the media is authentic or the classifier correct.

## 5. Robustness: transforms prepared, bulk evaluation still required

`robustness-manifest.json`: 100 same held-out clips, 25 per manipulation type,
chosen by metadata and a fixed seed/hash. `robustness-plan.json` declares clean,
video CRF 28/38 and audio bitrate 64/32 kbps conditions. The real-only AV-delay
study at 0.08/0.16/0.32 seconds is separate synchronization stress, not a new fake
ground-truth class. Prepared-input blur/noise helpers are not raw end-to-end tests.

Existing transform CLI, example syntax after supplying a source path:

```powershell
& $Py -m deepfake_detection.cli corrupt-media --input 'PATH_TO_SOURCE.mp4' --output '.\runs\robustness-001\clip-001-crf28' --kind video_compression --strength 28
```

This creates an immutable derivative and transform metadata. Before bulk scoring,
implement/review derivative manifests, timestamp/label preservation, and all-model
inference adapters. Use the **identical derived files** for every method, frozen
models/thresholds, and report coverage at each severity. Select severity using
train/validation review; any post-held-out choice remains exploratory. No
transcoding or robustness inference has been started by this preparation step.

## 6. Computational cost

`profiling-manifest.json`: 12 same clips, three per type, selected without looking
at prediction outcomes. `profiling-plan.json` specifies one warmup and three
timed repetitions. Run after the GPU is free, synchronizing CUDA around timers.

Separate decode/hash/preparation, neural inference, FK record/verification and
total raw-file-to-output time. Report cold/warm cache separately, median seconds
per clip and source-second, peak allocated/reserved VRAM, process RSS and failures.
Cached-head smoke measurements and historical concurrent wall times are not
substitutes. The standardized three-method profiler remains **to be implemented**.

## 7. Transfer and reproducibility

FakeAVCeleb transfer is deferred until access, labels and cross-dataset source
overlap are audited. Do not download it or start inference under this preparation
step. After all analyses, retain environments, code/asset/checkpoint hashes,
manifests, commands, exclusions and per-clip outputs. Keep original and calibrated
threshold analyses separate and disclose reconstruction/adaptation choices.

## Verification performed for this package

Three targeted tests passed: aggregation-grid constraints, balanced deterministic
metadata selection, and localization accounting that does not hide unassessed
positive duration. All 1,000 stored timeline IDs/durations/thresholds were checked
against the existing held-out manifest and FK decision. None of these checks
starts training or changes the active evaluation's code/inputs.
