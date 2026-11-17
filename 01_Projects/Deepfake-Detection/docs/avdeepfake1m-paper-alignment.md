# AV-Deepfake1M paper alignment protocol

AV-Deepfake1M (the 2024 release) is the primary benchmark. FakeAVCeleb remains
an optional external transfer evaluation. LAV-DF is a legacy adapter only.
Use official train/val/test splits and source-family audits. Dataset access is not bundled.
The adapter follows the [official metadata schema](https://github.com/ControlNet/AV-Deepfake1M#metadata-structure)
and [2024 challenge protocol](https://deepfakes1m.github.io/2024/details).
Use the original release, not AV-Deepfake1M++. Media access is managed through
the [official dataset page](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M).
Do not infer identity-disjointness from source-family checks alone.

## Reproducible Windows commands

To recreate the checked CPU environment, use `uv venv --python 3.11 .venv` and
`uv pip install --python .venv/Scripts/python.exe -r requirements-paper-alignment.lock.txt`.
For FFmpeg/FFprobe, install `static-ffmpeg==3.0` into an isolated tools directory,
call its `get_or_fetch_platform_executables_else_raise()` function and copy the
two Windows executables into `.venv/Scripts/`. The checked binaries are FFmpeg 8.
The runtime resolver supports both Windows `.exe` names and POSIX installations.

Run from the repository root with `.venv/Scripts/python.exe`. The local checked
asset registry is `checkpoints/paper-alignment-001/assets.local.json`. Model files,
datasets and generated runs are ignored by Git. Supply the extracted AV-Deepfake1M
directory with `train_metadata.json`, `val_metadata.json`, `train/`, `val/` and,
when available, `test_files.txt` and `test/`. This adapter targets full temporal
annotations, rather than the limited-label classification challenge.

```powershell
$fkPython = '.venv/Scripts/python.exe'
& $fkPython -m unittest discover -s tests -v
& $fkPython -m deepfake_detection.cli prepare-avdeepfake1m --dataset-root AV-Deepfake1M --output data/avdf1m-study --source-overlap-policy keep-official
& $fkPython -m deepfake_detection.cli fit-quality --manifest data/avdf1m-study/manifest.json --dataset-root AV-Deepfake1M --assets checkpoints/paper-alignment-001/assets.local.json --output data/avdf1m-study/quality.json
& $fkPython -m deepfake_detection.cli cache-fk --manifest data/avdf1m-study/manifest.json --dataset-root AV-Deepfake1M --assets checkpoints/paper-alignment-001/assets.local.json --quality-normalization data/avdf1m-study/quality.json --output runs/cache-v2 --device auto
& $fkPython -m deepfake_detection.cli train-fk --manifest runs/cache-v2/training-manifest.json --model-config configs/ours.provisional.json --training-config configs/fk.training-v1.json --output runs/fk-v2 --calibrate --device auto
& $fkPython -m deepfake_detection.cli evaluate-fk --manifest runs/cache-v2/training-manifest.json --checkpoint runs/fk-v2/selected.pt --output runs/fk-v2/validation.json --split validation --device auto
& $fkPython -m deepfake_detection.cli infer-fk --input example.mp4 --duration 3 --checkpoint runs/fk-v2/selected.pt --assets checkpoints/paper-alignment-001/assets.local.json --quality-profile measured-quality-v2 --output runs/evidence-v2
& $fkPython -m deepfake_detection.cli verify --record runs/evidence-v2/record.json --trusted-commitment runs/evidence-v2/trusted-commitment.txt --source example.mp4 --artifact-root runs/evidence-v2
```

Preparation uses all train/val families by default. For a bounded pilot, specify
`--train-clips 15000 --validation-clips 2500`; these are caps on complete families,
not the full benchmark. Durations come from media timestamps, not an assumed FPS.
The genuine source ID and each fake's original reference determine source families.
Modality-specific intervals are preserved; unavailable ones remain unknown.

`test_files.txt` has no labels. It becomes a separate `test-inference-files.json`,
never a supervised cache or metrics population. Only supply `--test-metadata PATH`
when actual labeled official test metadata is available; then test evaluation is
possible. The validation command above is a development diagnostic on the same
split used for selection, not held-out test performance. Blind test files can be
processed individually using `infer-fk`; the inventory is not an official submission export.

Use a fresh output path for each run. `cache-fk --resume` verifies completed clips.
Training resumes from `last.pt` into a new run directory; `selected.pt` is the
validation-selected inference artifact. The default training cap is 50 epochs,
patience 5; explicitly freeze any study-specific alternative before test scoring.

For the shared controls and FK ablations:

```powershell
& $fkPython -m deepfake_detection.cli prepare-experiments --model-config configs/ours.provisional.json --output runs/plan-v2
& $fkPython -m deepfake_detection.cli run-experiments --plan runs/plan-v2/plan.json --manifest runs/cache-v2/training-manifest.json --training-config configs/fk.training-v1.json --output runs/experiments-v2 --device auto
```

The runner trains/calibrates FK variants and four shared controls. AVFakeNet and
AVTENet remain separately planned external reconstructions with their own inputs;
they are not executed by this shared-cache runner. Test scoring is a separate step.

## Quality and compatibility

`measured-quality-v2` retains the model's two-dimensional quality interface.
Visual quality is the mean of face availability, detection confidence, tracking
IoU and normalized Laplacian variance. Audio quality averages speech coverage,
one minus clipping fraction, and normalized estimated SNR. Detector confidence
and landmark/track failures are proxies; there is no independent occlusion model.

Blur and SNR use training-only 5th/95th percentiles. Constant or unavailable
measurements map to 0.5, never best quality. SNR requires at least 100 ms each
of speech and non-speech, positive noise power, and speech power above noise.
It estimates `(speech power - noise power) / noise power`; this is not a clean
reference SNR. Reasons for unavailable SNR remain in measurements.

Profile hashes bind training population, detector/encoder assets and bounds.
New cache manifests use v2 and retain the full profile; legacy caches remain
readable. Never reuse an old checkpoint with v2 quality values: rebuild caches
and retrain. Omitting `--quality-normalization` explicitly retains the legacy
availability/speech summary and float audio inputs. New v2 PCM16 round-trip inputs
are shared with saved audio; legacy records retain exact float inputs in NPZ and
identify the PCM16 WAV as a derivative.

## Selection and controls

Validation-only grid: K={1,3,5}; lambda/lambda_v={0,0.5,1}; eta={0.1,0.2,0.5};
tau_r={0.25,0.5,1}. Select video balanced accuracy, then window balanced accuracy
for tau_win. Ties prefer the L1-nearest original aggregation settings, then grid
order; thresholds prefer the original threshold, then the lower value. All
candidates are saved. These choices are declared reconstruction choices, not
numeric settings attributed to the paper. Selection with either class absent
fails explicitly. Test data never select parameters.

Visual-only uses the frozen visual features, GRU and projection; audio-only uses
audio projection. Concatenation adds a scalar head to concatenated projected bins.
Late fusion averages per-bin probabilities from independently supervised visual
and audio branches. All use the same usable windows and hierarchical reduction.
Each model has its own calibration and typed checkpoint. Late-fusion branches
receive independent losses; validation selects their paired checkpoint jointly.

Window evaluation uses positive-duration any-overlap, half-open intervals and
known labels only. Unassessed videos/windows remain in coverage denominators.
Timeline mean-of-overlapping-bin metrics are explicitly supplemental, including
unassessed manipulated time. Legacy assessments lacking duration are reported
as missing timeline metadata rather than assigned guessed durations.

## Forensic schema and manuscript edits

v2 records bind source metadata and derived NPZ crops/features of preparation,
PCM16 audio, preparation metadata and a representative PNG for each usable window.
The NPZ contains prepared pixels/audio/timestamps, not neural encoder embeddings.
Artifact verification must be requested with `--artifact-root`; without it the
verifier clearly reports `NOT_REQUESTED`. Keep the trusted FAC separately.

Metadata inspection uses PyAV/libavformat, recorded in the record, rather than
invoking FFprobe. VFR is detected from decoded PTS deltas with one-timebase-tick
tolerance. Missing acquisition time and source context are null, not invented.
Caption/transcript and perturbation are `NOT_RUN` and never affect scores.

Keep manuscript Section III-E's AV-Deepfake1M-primary claim. State the exact frozen
population and whether a subset or the full release was used. Describe official
splits and independently checked family separation. Document these
quality formulas, calibration grid/tie rules, baseline definitions and metadata
backend in the implementation details. Keep the original PDF unchanged until
the manuscript authors incorporate these edits. Report empirical results only
after real-data training and evaluation; synthetic tests prove mechanics only.

## Real-release adapter corrections (14 November 2026)

Canonicalize repeated slashes in original source IDs, while retaining strict media
path validation. Fake-only source families use the release's
`id/video/clip/real.mp4` reference layout to identify the VoxCeleb2 source; this
does not create a genuine example or require missing original media.

Use `--source-overlap-policy keep-official` for the user's current protocol:
retain all official train/validation clips without caps or overlap exclusion.
This supersedes the earlier drop-train decision. Source-family/content overlap
is reported rather than rejected. The policy follows the population through
quality fitting, caching, training, resume, and evaluation. Save
`source-overlaps.json` and `audit.json`; exclusions are empty. Use a new
full-official-001 output and refit quality/rebuild caches for this population.
The earlier filtered pilot remains a historical artifact. Test inventories have
no labels and remain inference-only unless actual test annotations are provided.

Conflicts between modality type and modality intervals are masked as unknown
supervision for that modality. Original conflicting intervals remain in each row,
with counts and policy in the audit. Combined labels/intervals retain their own
strict validation. Do not edit the downloaded annotation files to pass checks.

Duration bounds: intersect internally validated annotations with observed media
duration. Preserve exact source intervals and overruns in per-row
annotation_bounds_adjustments and report counts in audit.json. Keep every video;
if no fake time remains observable, keep the video label and mark temporal
supervision incomplete. Original annotation files remain unchanged. Preparation
prints progress every 1,000 clips; it has no resume support.
