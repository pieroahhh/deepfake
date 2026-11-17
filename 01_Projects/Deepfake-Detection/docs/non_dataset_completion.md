# Preparation completed without the research dataset

Work performed on branch `margin`, 12 November 2026. Final local validation: **78 tests passed, zero skipped**; the retained generated-video record verifies with **zero reconstruction error**. This is a software-readiness
report, not a completed model evaluation or a claim of pretrained accuracy.

## Implemented and exercised locally

| Deliverable | What exists and what was checked |
| --- | --- |
| Source-to-record FK inference | `inference.py`, `fk infer-fk`: source copy/hash → source-time preparation → supplied encoders → FK → committed record → independent verification |
| Float32 record export | Signed contributions retained; canonical scores reconstructed in binary64; raw/canonical differences and any threshold crossing recorded; score verification tolerance stays strict |
| Full integration smoke | Generated colorbars/tone through the default FK architecture, with explicitly synthetic detector/VAD/encoder providers; verified record and retained fixture weights |
| Retrainable ablations | No AV, no temporal channel, uniform evidence weights; disabled weights renormalized, branches omitted from computation and frozen; configuration/checkpoint provenance retains variant |
| Aggregation variants | Separate window max/mean and video max/mean profiles, preserving verifier-compatible aggregation equations |
| Experiment register | Eight FK variants × three seeds plus two primary baselines × three seeds = 30 planned runs; every run explicitly NOT_RUN |
| Prepared-input robustness | Seeded Gaussian image noise, JPEG, blur, downsample/upsample; audio noise with measured SNR and no hidden clipping |
| Raw-media robustness | Immutable FFmpeg compression, blur, downsampling, AAC bitrate and audio-delay derivatives; source/derivative hashes, stream metadata, exact arguments and label policies |
| Statistical reporting | Per-method assessment and common-assessed populations, ROC-AUC/AP/trapezoidal PR-AUC only with complete justified scores; hard-only AVFakeNet retains null continuous metrics |
| Grouped uncertainty | Paired source-family resampling, observed metric differences, percentile intervals and explicit undefined-replicate counts; separate from training-seed variability |
| Localization adapter | Endpoint-partitioned overlap averaging; duration-weighted assessed precision/recall/F1/IoU; uncovered manipulation coverage separately reported |
| Profiling | Warm-up, CPU/CUDA synchronization boundaries, median/p90/p95, sample timings, declared units, parameter storage and accurately labeled memory counters |
| Conversion tooling | Explicit Flax-array mapping with dense/QKV/convolution layout transformations, complete coverage/shape checks, safe tensor serialization |
| Parity tooling | Matching input/revision/checkpoint metadata and intermediate-stage comparisons; deliberately wrong tensor order fails |
| Repeatable check bundle | `scripts/prepare_without_dataset.sh`: tests, readiness, generated-video record, example CSV/JSON reports, provisional run plans and synthetic head-only profile |

The conversion tool marks artifacts `converted_unverified`. Production loaders
reject them by default. A synthetic tensor conversion round trip is not proof
that the actual official ViViT checkpoint has been converted correctly.

Ablations must be retrained with matched protocol/selection budgets. Applying a
mask to a trained full-model checkpoint remains a sensitivity intervention and
must not be reported as a retrained ablation. Frozen branch modules remain in
state dictionaries for compatibility; profiling distinguishes total stored from
trainable parameters. Uniform weights preserve the same assessment gate.

The raw-video command requires actual local encoder assets and an FK checkpoint.
The only supplied quality mapping is explicitly named
`availability-speech-v1-provisional`; it is not a calibrated blur/occlusion model.

## Actual local evidence

Later asset update: five successful downloads are now available, checksummed and
runtime exercised in `runs/pretrained-runtime-checks-001/runtime-checks.json`:
AST, EfficientNet-B0, Wav2Vec2 Base, Silero and YuNet. The AST feature dimension
is 768; FK visual/audio outputs are [1,50,1280]/[1,99,768], and their synthetic
record verifies with zero reconstruction error. Silero processed silence and
YuNet processed a blank image. No positive real-face or trained-head accuracy
was tested. ViViT returned HTTP403; AV-HuBERT failed expired-certificate
validation. The acquisition-attempt table below is historical.

Matplotlib is now installed as well: `runs/checks-with-downloaded-assets-001/`
completed all 78 tests and rendered the synthetic confusion-matrix SVGs.

Pass the real registry explicitly rather than using the empty example:

```bash
bash scripts/prepare_without_dataset.sh runs/checks-with-assets-001 \
  --assets checkpoints/pretrained-001/assets.local.json
python -m deepfake_detection.pretrained_checks \
  --assets checkpoints/pretrained-001/assets.local.json \
  --output runs/pretrained-runtime-checks-002
```

The synthetic bundle verifies registry checksums and local mechanics. The
separate pretrained check runs actual models on synthetic signals; neither
workflow certifies a real-face video or final experiment readiness.

The newest check bundle is `runs/non-dataset-checks-003/` (ignored by Git):

- `tests.log`: complete local test run.
- `synthetic-video/generated-colorbars-and-tone.mkv`: generated source, not a human recording.
- `synthetic-video/fixture.pt`: untrained synthetic fixture model/encoder state.
- `synthetic-video/inference/record.json`: default-architecture float32 result.
- `synthetic-video/inference/trusted-commitment.txt`: separately retained commitment.
- `synthetic-video/inference/verification.json`: valid, zero reconstruction error.
- `example-report/metrics.csv` and `metrics.json`: synthetic reporting examples.
- `head-only-profile.json`: measured synthetic cached-feature head timing, not
  pretrained inference or raw-video latency.
- `experiments/plan.json`: planned runs, not training results.

Rerun with a fresh directory:

```bash
bash scripts/prepare_without_dataset.sh runs/non-dataset-checks-004
```

## Asset acquisition and runtime attempts

Actual attempts were made; missing assets were not replaced with random weights.

| Attempt | Observed outcome |
| --- | --- |
| Official torchvision EfficientNet-B0 URL | Failed DNS resolution; no weights installed |
| Official MIT AST Hugging Face model | Failed DNS resolution; no AudioSet weights installed |
| Official Facebook Wav2Vec2 Base Hugging Face model | Failed DNS resolution; no Base weights installed |
| Official Silero GitHub revision/artifact | Failed DNS resolution; no JIT installed |
| Official OpenCV YuNet GitHub revision/artifact | Failed DNS resolution; no ONNX detector installed |
| Official Scenic FE and AV-HuBERT LRS3 Base checkpoint acquisition | Failed DNS resolution; neither native checkpoint was downloaded |
| Workspace-local uv/Python bootstrap | Package acquisition failed; Python 3.8 and AV-HuBERT environment remain unavailable |
| Matplotlib installation | No package could be acquired; numeric reports work, plots are explicitly marked NOT_RENDERED |

Attempt logs: `runs/pretrained-acquisition-attempt-001/acquisition.json`,
`runs/hf-acquisition-attempt-001/acquisition.json`, and
`runs/native-backbone-acquisition-attempt-001/acquisition.json`. The checked-in example
asset registry intentionally remains null rather than pointing at failed folders.

To attempt the five directly supported assets and two native checkpoint candidates
on a network-enabled host:

```bash
source .venv/bin/activate
python scripts/acquire_pretrained.py --output checkpoints/pretrained-001
```

The downloader resolves mutable provider refs to immutable revisions before
fetching, retains source/checksum metadata, extracts base models from AST/Wav2Vec2
wrappers, checks exact local reload behavior and publishes an asset registry only
for successful assets. Acquisition/conversion execution on real weights remains
unverified here. This is a script ready to exercise, not a promise every current
upstream model/configuration loads unchanged. Its provenance records the selected
asset; it does not establish that AVTENet's authors used that exact checkpoint.
Native Scenic/AV-HuBERT files remain marked as requiring conversion/runtime checks;
a successful download does not mark those backends ready for inference.

Legacy bootstrap command:

```bash
bash scripts/bootstrap_avhubert.sh
```

It keeps tools, downloaded Python and legacy dependencies inside ignored workspace
folders, leaves the main `.venv` intact, and delegates to the existing pinned-tree
setup. It still requires network access and runtime validation. AV-HuBERT license,
checkpoint/task dictionaries and paired-input preparation must accompany the actual
checkpoint. The research dataset is not needed for this runtime validation.

## ViViT source discovery and remaining parity gate

The [official model zoo](https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/README.md)
identifies a **ViViT-B/16x2 FE, Kinetics-400** checkpoint at
`https://storage.googleapis.com/scenic-bucket/vivit/kinetics_400/vivit_base_16x2_fe/checkpoint`.
This is a candidate compatible family, not proof of the exact AVTENet checkpoint.

The [current official FE configuration](https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/configs/kinetics400/vivit_base_factorised_encoder.py)
uses 32 frames, 768 dimensions, 12 spatial/4 temporal layers and 12 heads. These
match our macro architecture choices. Exact numerical equivalence still depends
on tensor mapping, activation implementation, normalization, token/position layout
and input preparation. For example, its resize/crop recipe differs from our
explicit provisional direct-resize preparation. Do not claim parity from matching
layer counts alone. The moving upstream source is not a pinned checkpoint revision.

`checkpoint_conversion.convert_factorized_arrays` requires an exported named-array
checkpoint and a complete reviewed map. It does not guess the actual Scenic
training-state format or key paths. The synthetic map tests validate conversion
operations only. Actual named-array extraction, map review and independent
reference outputs still require the downloaded checkpoint and compatible backend.

`validation_tools.compare_backend_exports` compares named intermediate tensors
with explicit provenance and tolerances. A passing operator-supplied export cannot
by itself authenticate how reference tensors were produced; preserve their exact
upstream code, source checkpoint and shared input bytes independently.

## Commands now available

```bash
fk prepare-experiments --model-config configs/ours.provisional.json --output runs/experiment-plan-001
fk profile-smoke --output runs/fk-head-profile-001.json --iterations 20 --threads 2
fk report --input /path/to/predictions-and-provenance.json --output runs/report-001 --reference FK
fk corrupt-media --input /path/to/consented-video.mp4 --output runs/degraded-001 --kind video_compression --strength 28
```

Report input contains `methods` (method → prediction records) and `provenance`
with `protocol_id` and `data_kind` (`synthetic` or `research`). Each prediction
contains video ID, label, assessment status, prediction, score and source family
when grouped bootstrap is requested. Never use fabricated scores for a hard-only method.

When actual encoder/detector/VAD assets and a trained FK checkpoint exist:

```bash
fk infer-fk --input /path/to/consented-video.mp4 --duration 3.0 \
  --checkpoint /path/to/selected.pt --assets /path/to/assets.local.json \
  --quality-profile availability-speech-v1-provisional --output runs/video-check-001
```

The duration must correspond to the source timeline. No sample real-person video
was supplied or downloaded, no camera/microphone was accessed, and no real-face
or real-Silero execution is claimed. Generated colorbars with fixed geometry do
not establish detector quality, tracking identity stability or model accuracy.

## What still needs the research dataset

- Actual metadata/annotation/identity/source-family audit and release fingerprint.
- Frozen common train/validation/test manifests and modality-correct loaders.
- Pilot resource measurements, augmentation recipes and quality calibration.
- Training/fine-tuning all methods, full validation selection and protocol freeze.
- Detection/localization/robustness/ablation comparisons and final figures.

These research steps remain pending. Separately, the network/runtime/recording
blocks above remain open even though they do not inherently require that dataset.
