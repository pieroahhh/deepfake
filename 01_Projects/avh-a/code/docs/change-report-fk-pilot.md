# FK pilot code change report

Snapshot of the Windows checkout on 12 November 2026. The comparison starts at the teammate branch checkout. The scope includes local source, documentation, and test changes plus the pilot artifacts those changes produced.

## 1. Git state

- Current branch and upstream are `feature/model-infrastructure-validation` and `origin/feature/model-infrastructure-validation`.
- The checkout base and current `HEAD` are both `1a99238ad966d602481eb6874636c83ecabd61a3`, **Fix resumable AV-HuBERT setup and verify legacy environment**. The reflog shows a checkout at this commit and a later return to the same commit after switching to `margin`.
- `git log --oneline 1a99238..HEAD` returned no commits. `35c56af`, which introduced inference and experiment tools, and `13bde52`, which fixed float32 weight rounding, are already in the base's ancestry.

`git status --short --branch` before adding this report showed:

```text
## feature/model-infrastructure-validation...origin/feature/model-infrastructure-validation
 M docs/dataset_access_audit.md
 M src/deepfake_detection/cli.py
 M src/deepfake_detection/commands.py
 M src/deepfake_detection/faces.py
 M src/deepfake_detection/inference.py
 M src/deepfake_detection/infrastructure.py
 M src/deepfake_detection/preparation.py
 M src/deepfake_detection/training.py
 M tests/test_infrastructure.py
?? LAV-DF.tar
?? LAV-DF/
?? src/deepfake_detection/fk_caching.py
?? src/deepfake_detection/lavdf.py
?? tests/test_lavdf.py
```

There were no staged files. This report adds `?? docs/change-report-fk-pilot.md` to that snapshot. `LAV-DF.tar` is the local dataset archive and `LAV-DF/` is the extracted data. Git ignores `.venv/`, `.cache/`, `checkpoints/`, `data/`, and `runs/`. Those directories contain the CPU and CUDA environments, downloaded weights, label manifests, feature caches, checkpoints, and forensic records.

`git diff --stat 1a99238` returned:

```text
 docs/dataset_access_audit.md             | 46 ++++++++++++++++++++++++++++++
 src/deepfake_detection/cli.py            | 30 ++++++++++++++++++--
 src/deepfake_detection/commands.py       | 39 ++++++++++++++++++--------
 src/deepfake_detection/faces.py          | 48 ++++++++++++++++++++++++++------
 src/deepfake_detection/inference.py      | 26 ++++++++---------
 src/deepfake_detection/infrastructure.py | 33 +++++++++++++++++++---
 src/deepfake_detection/preparation.py    | 32 +++++++++++++++++++--
 src/deepfake_detection/training.py       | 38 ++++++++++++++++---------
 tests/test_infrastructure.py             | 18 ++++++++++++
 9 files changed, 255 insertions(+), 55 deletions(-)
```

The three untracked source and test files add 310 lines, measured with `git diff --no-index --numstat` against `NUL` on Windows. Including them gives **12 project files, 565 insertions, and 55 deletions** before this report. The dataset archive and its extracted files are outside those counts.

| Untracked project file | Added lines |
| --- | ---: |
| `src/deepfake_detection/fk_caching.py` | 129 |
| `src/deepfake_detection/lavdf.py` | 124 |
| `tests/test_lavdf.py` | 57 |

## 2. Per-file changes

Each excerpt shows the relevant part of the working-tree diff. New files are shown as additions.

### `docs/dataset_access_audit.md`

Added instructions for the LAV-DF pilot, including how to regenerate labels and caches, preserve official splits, use the fixed crop policy, interpret unknown modality intervals, and resume into a new run directory. The previous AV-Deepfake1M audit is under its own heading. This records the local workflow needed to reproduce the pilot.

```diff
+For the LAV-DF FK pilot, run `fk cache-fk` with the verified manifest, extracted
+dataset root, and checksummed asset registry. The cache command uses the in-frame
+crop policy and records it with every cache. `fk train-fk` carries that policy into
+the checkpoint, and `fk infer-fk` uses it when generating the record.
```

### `src/deepfake_detection/cli.py`

Modified `main()` to register and dispatch `prepare-lavdf-pilot` and `cache-fk`, pass device selection into `train-fk`, and pass an optional crop override into `infer-fk`. The module description now covers dataset-specific commands. These entries make dataset preparation and CUDA training reproducible from the command line. No command or public function was removed.

```diff
+    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
+    p = commands.add_parser("cache-fk", help="build verified feature caches and a training manifest")
+    p.add_argument("--manifest", required=True)
+    p.add_argument("--dataset-root", required=True)
+    p.add_argument("--assets", required=True)
+    p.add_argument("--output", required=True)
+    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
...
+    p.add_argument("--fit-face-crops", action="store_true", default=None, help="fit boundary crops within observed pixels")
```

The first `--device` line belongs to `train-fk`. The second belongs to `cache-fk`, and the crop flag belongs to `infer-fk`.

### `src/deepfake_detection/commands.py`

Extracted `load_fk_assets()` from `infer_fk_from_assets()` to share encoder, face detector, and Silero loading between caching and inference. Modified `infer_fk_from_assets()` to use the shared quality mapping, read the checkpoint crop policy, and reject an explicit incompatible override. The inference entry point still loads the model and encoders on CPU. Sharing preprocessing prevents a training-to-inference crop mismatch.

```diff
+    recorded_crop = metadata.get('preprocessing', {}).get('face_crop_policy')
+    if recorded_crop is not None:
+        if recorded_crop not in (IN_FRAME_CROP_POLICY, STRICT_CROP_POLICY):
+            raise ValueError('unknown checkpoint face crop policy')
+        recorded_fit = recorded_crop == IN_FRAME_CROP_POLICY
+        if fit_to_frame is not None and fit_to_frame != recorded_fit:
+            raise ValueError('requested face crop policy differs from checkpoint training policy')
+        fit_to_frame = recorded_fit
```

### `src/deepfake_detection/faces.py`

Modified `aligned_crops()` and `track_and_crop()` to accept `fit_to_frame=False`. The new path checks the rotated image's observed-pixel mask, searches for a smaller centered crop when the requested square crosses the boundary, and verifies that the landmarks remain inside that fallback crop. It applies to the face and lip crops. The default strict crop and the existing detection and IoU tracking requirements still apply to callers that omit the flag. The first pilot pair and three additional families had no usable windows under the original crop check.

```diff
+        if not observed(box) and fit_to_frame:
+            lower, upper = 1, math.floor(side)
+            while lower <= upper:
+                candidate_size = (lower + upper) // 2
+                candidate = bounds(candidate_size)
+                if observed(candidate):
+                    box = candidate
+                    lower = candidate_size + 1
+                else:
+                    upper = candidate_size - 1
```

### `src/deepfake_detection/fk_caching.py`

New module with `WINDOW_LABEL_POLICY`, `_modality_target()`, and `cache_fk_manifest()`. It audits the source manifest, checks video SHA-256 hashes, loads the four assets, moves the frozen encoders to the requested device, and prepares all scheduled windows. It saves CPU safetensors for usable windows and retains measurements, labels, and exclusion reasons for every attempted window. It writes the training manifest only after processing all supplied clips. This connects the verified annotations to the existing FK trainer and captures the rejections that were previously visible only in the two-clip inspection.

```diff
+            prepared = prepare_fk_window(source, window[0], duration, detector, speech, fit_to_frame=True)
...
+            assessment['labels'] = {
+                'combined': labels['window'],
+                'audio': _modality_target(row.get('audio_manipulated_intervals'), window, labels['window']),
+                'visual': _modality_target(row.get('visual_manipulated_intervals'), window, labels['window']),
+            }
```

### `src/deepfake_detection/inference.py`

Modified `run_fk_video()` to accept `fit_to_frame=False`, call the same speech and tensor preparation helpers as caching, and pass the crop choice to each window. It writes the crop policy into forensic preprocessing metadata and records a speech preparation `ValueError` when caught. `record_from_outputs()` and the independent verifier code were already in the base. These changes let the held-out inference record identify the preprocessing used by the selected checkpoint.

```diff
+        speech = fk_source_speech(working, source_duration, vad)
+        speech_error = None
+    except ValueError as error:
+        speech, speech_error = [], str(error)
...
+    preprocessing['face_crop_policy'] = IN_FRAME_CROP_POLICY if fit_to_frame else STRICT_CROP_POLICY
+    if speech_error is not None:
+        preprocessing['speech_detection_error'] = speech_error
```

### `src/deepfake_detection/infrastructure.py`

Added `fk_device()` to resolve `auto`, `cuda`, or `cpu`. Modified `train_fk_manifest()` to move the model to the selected device and validate cache policy declarations against the new manifest before training. It passes the recorded crop and label policies into `fit_fk()`. `CachedVideoPopulation` still verifies and loads the caches. This fixes CPU-only CLI training and gives inference a training policy to inherit.

```diff
+    if manifest.get('schema') == 'fk-cache-manifest-v1':
+        preprocessing = manifest['preprocessing']
+        policy_digest = hashlib.sha256(canonical(preprocessing)).hexdigest()
...
-    )
+    ).to(device)
+    policy = {key: manifest[key] for key in ('preprocessing', 'window_label_policy') if key in manifest}
```

### `src/deepfake_detection/lavdf.py`

New module with `normalize_lavdf()` and `prepare_lavdf_pilot()`. Normalization checks unique paths, modality flags, edit counts, source-family references and split consistency. It maps `dev` to `validation`, preserves combined intervals, and excludes four source rows with intervals extending beyond their stated durations. Selection ranks eligible families by seeded SHA-256, chooses the first path per category within each family, and hashes selected media. The module converts the original LAV-DF metadata into verified pilot labels without assigning separate timings to edits whose modalities are ambiguous.

```diff
+        audio_edits = edits if audio else []
+        visual_edits = edits if visual else []
+        if audio and visual:
+            audio_edits = visual_edits = None
```

### `src/deepfake_detection/preparation.py`

Added `IN_FRAME_CROP_POLICY`, `STRICT_CROP_POLICY`, `FK_QUALITY_PROFILE`, and three shared functions. `fk_source_speech()` clips VAD interval ends to the clip duration. `encode_fk_window()` prepares the face, audio, and time tensors and sends them to the encoder device. `fk_availability_quality()` returns face and speech coverage as the two float32 quality inputs. Modified `prepare_fk_window()` to forward `fit_to_frame` and record the chosen in-frame policy when enabled. Sharing this code keeps caching and inference aligned.

```diff
+    return encoders.encode_window(faces.to(device), audio.to(device), times.to(device))
...
+    if fit_to_frame:
+        result['face_crop_policy'] = IN_FRAME_CROP_POLICY
...
-        records = track_and_crop(video['rgb'], detector)
+        records = track_and_crop(video['rgb'], detector, fit_to_frame=fit_to_frame)
```

### `src/deepfake_detection/training.py`

Modified `seed_everything()` to set the deterministic cuBLAS workspace default when requested. Modified `_rng_state()` and `_restore_rng()` to save and restore the training CUDA device's RNG and allow a CPU destination. Modified `fit_fk()` to save optimizer step count, device, GPU name, and manifest policies, and to check policies when resuming. It reconstructs the step count from history for older checkpoints. The existing CPU `map_location` and device transfer in `forward_video()` support restoring weights and optimizer state on the target device. `select_video_threshold()` was already in the base.

```diff
+        step = previous.get('step', sum(entry['training']['supervised_videos'] for entry in history))
+        resume_device = previous.get('device')
...
-            'optimizer_state': optimizer.state_dict(), 'stale_epochs': stale, 'rng': _rng_state(),
+            'optimizer_state': optimizer.state_dict(), 'scheduler_state': None,
+            'stale_epochs': stale, 'rng': _rng_state(device),
+            'device': str(device), 'gpu_name': gpu_name, **manifest_policy,
```

### `tests/test_infrastructure.py`

Added `InfrastructureTests.test_boundary_face_crop_preserves_observed_pixels`. It exercises the exact failure that blocked the pilot. It checks strict rejection, the expected face and lip output sizes under in-frame fitting, preservation of the uniform input pixels, and rejection when a required landmark cannot fit.

```diff
+        with self.assertRaisesRegex(ValueError, 'extends beyond observed source pixels'):
+            aligned_crops(image, detection)
+        face, lips = aligned_crops(image, detection, fit_to_frame=True)
+        self.assertEqual(face.shape, (224, 224, 3))
+        self.assertEqual(lips.shape, (96, 96, 3))
```

### `tests/test_lavdf.py`

New `family()` fixture helper and `LavdfTests` class with two tests. The fixtures exercise official split mapping, source grouping, exclusion of invalid intervals, hash population, and preservation of unknown modality timing. A second case rejects a cross-split source reference. These tests protect the labels and split assignments used for training.

```diff
+        self.assertEqual(both["manipulated_intervals"], [[1.2, 1.5]])
+        self.assertIsNone(both["audio_manipulated_intervals"])
+        self.assertIsNone(both["visual_manipulated_intervals"])
+        self.assertEqual(both["split"], "validation")
```

## 3. Behavior, defaults, and CLI

| Command or API | Arguments and defaults |
| --- | --- |
| `fk prepare-lavdf-pilot` | Requires `--dataset-root` and `--output`. `--train-families 8`, `--validation-families 4`, `--seed 7`. The root must contain `metadata.min.json`. |
| `fk cache-fk` | Requires `--manifest`, `--dataset-root`, `--assets`, and `--output`. `--device auto` picks CUDA if available and CPU otherwise. Explicit `cuda` requires a working CUDA build and selects `cuda:0`. This command always uses the in-frame crop policy. |
| `fk train-fk` | Existing manifest, config, and output arguments. New `--device auto` uses the same resolution rule. Existing `--epochs` defaults to the configured budget, currently 50. `--resume` defaults to no checkpoint and `--calibrate` defaults to false. |
| `train_fk_manifest()` | The Python API defaults to `device='cpu'`. The CLI explicitly passes its own `auto` default. |
| `fk infer-fk` | Existing required input, duration, checkpoint, assets, output, and `--quality-profile` arguments. The only accepted quality profile is `availability-speech-v1-provisional`. New `--fit-face-crops` defaults to `None` when omitted. An omitted flag inherits the checkpoint policy, or uses strict crops for a legacy checkpoint without policy metadata. An explicit request conflicting with a recorded policy raises an error. |
| Crop and preparation functions | `fit_to_frame=False` remains the library default in `aligned_crops()`, `track_and_crop()`, `prepare_fk_window()`, and `run_fk_video()`. |

### Crop policy flow

`cache-fk` uses `in-frame-landmark-preserving-v1` for every supplied split. It stores the name in each cache's `metadata.json → provenance.face_crop_policy`, in each cached manifest row's `cache_provenance`, in window assessments, and at `training-manifest.json → preprocessing.face_crop_policy`. The trainer checks the train and validation cache declarations and preprocessing digest. It copies the manifest preprocessing and window-label policy into `run.json` and the checkpoints. `infer_fk_from_assets()` reads `checkpoint["preprocessing"]["face_crop_policy"]` and `run_fk_video()` records the effective policy at `record.json → preprocessing.face_crop_policy`. This is the default for the new study caches and their checkpoints. Standalone legacy callers retain the strict behavior described above.

### Window and bin targets

- Windows span **2 seconds** and start every **1 second**. Intervals use half-open source seconds. Any **positive-duration** intersection with a published edit counts as fake, with minimum overlap `> 0` seconds. Endpoints that merely touch contribute zero overlap. FK applies the same rule to its ten 0.2-second bins.
- A genuine clip or a completely annotated fake clip has combined target `0` in any usable window with no edit overlap. The window stays in training if its media assessment passes. A fake clip contributes the positive video-level loss only if at least one usable window intersects an edit. Unknown combined temporal targets are masked for incompletely annotated fake clips. The LAV-DF pilot treats the accepted published combined annotation lists as complete.
- For an audio-only edit, an overlapping window has `audio=1, visual=0, combined=1`. The visual-only case reverses the modality labels. If the clip is flagged in both modalities and separate times are unknown, an overlapping window has `audio=null, visual=null, combined=1`. A known combined negative makes both modality targets zero. FK consumes the combined rule for training. The modality targets are retained for downstream consumers and checks.
- Unusable windows carry annotation labels and rejection reasons in the assessments for audit. They contribute no cached feature rows or window/bin supervision. The audio-only inspection clip has one usable negative window and two usable windows flagged fake-audio and real-video.

### Checkpoint selection and threshold calibration

Best model selection uses the first strict minimum of **mean validation video loss**. With `--calibrate`, `select_video_threshold()` evaluates balanced accuracy, the mean of sensitivity and specificity, on assessed validation videos and requires both classes. The candidates are each observed validation score plus `0`, `0.5`, and `1`. Ties favor the candidate closest to `0.5`, then the smaller candidate. `selection.json` stores the threshold, validation score, assessed count, and the source `best.pt` SHA-256. `selected.pt` copies the selected weights and writes the chosen threshold at `decision.video_threshold` plus a `validation_selection` record. `best.pt` and `last.pt` retain the configured threshold. The window threshold is unchanged.

Training and calibration instantiate only `train` and `validation` cache populations. A row with a mismatched split is rejected inside either population. The calibration function independently requires `validation` rows. A manifest may contain a top-level `test` list, whose metadata can be parsed with the JSON, but the trainer and calibrator read no test caches, video media, or predictions. `cache-fk` can process test clips when explicitly supplied. This pilot cached no test clips and selected its threshold on 16 validation clips. The held-out forensic check used a different official dev source family. Its record is a pipeline check. [Resume and selection evidence](../../../../04_Results/experiments/avtenet/fk-gpu-pilot-resumed-001/resume-checks.json)

### Checkpoint contents and compatibility

| Artifact | Contents and behavior |
| --- | --- |
| `last.pt` | `fk-training-v1` with current and best model state, optimizer state, config, model spec, decision, data signatures, epoch, history, best loss, and stale-epoch count. The pilot adds `step`, `scheduler_state: None`, `device`, `gpu_name`, and the optional manifest `preprocessing` and `window_label_policy`. The schedule is constant learning rate. |
| `rng` | Python, NumPy, and PyTorch CPU states. `cuda` is now a one-element list for the training CUDA device or an empty list for CPU. The older writer saved all visible CUDA RNG states when CUDA was available. |
| `best.pt` | Final inference checkpoint with the selected weights and `selected_epoch`. The final writer removes the optimizer state. Its `epoch` and `step` reflect completed training, so use `selected_epoch` to identify the weights. |
| `selected.pt` | Calibrated inference checkpoint derived from `best.pt` with the validation threshold and selection metadata. |
| `run.json` | Adds GPU name, source device when resuming, and the recorded policies. `metrics.json` provides completed-run status and epoch history. |

The checkpoint schema tag and model parameter format stay at `fk-training-v1`. Resume loads storage with `map_location='cpu'` and restores model and AdamW state into the target model and optimizer. It resumes at an epoch boundary, with `step` counting optimizer updates through the last completed epoch. A legacy `last.pt` can resume on its original matching config, data, model spec, and decisions. The new loader reconstructs a missing step from history and permits absent policy fields when the input manifest also has no policies. A legacy checkpoint cannot resume a run using a new manifest with different caches or explicit policies. Legacy inference checkpoints still load and use strict crops unless explicitly configured otherwise. Only same-device resume on the tested RTX 4060 was checked for bitwise equality. CPU→GPU and GPU→CPU were checked for continuity.

## 4. Data and artifact schemas

| Artifact | Fields and compatibility |
| --- | --- |
| LAV-DF source `manifest.json` and `audit.json` | New source manifest is an array of audited rows. It includes dataset and release, video ID and relative path, original source family, source hash, split and duration, combined and modality clip labels, `temporal_complete`, `manipulated_intervals`, and `audio_manipulated_intervals` and `visual_manipulated_intervals`. The audit is `lavdf-pilot-v2` and records the metadata digest, selection seed and counts, exclusions, and manifest audit. Use `lavdf-pilot-002/manifest.json` for current modality labels. The earlier pilot manifest duplicated combined times for both-modified clips and should be regenerated for modality-aware use. |
| Feature cache `metadata.json` and `features.safetensors` | `frozen-feature-cache-v1` retains its schema tag, provenance and content digests, tensor schema, and five tensors: `visual_features`, `audio_features`, `visual_times`, `audio_times`, and `quality`. The new producer adds `face_crop_policy` and `window_label_policy` inside `provenance` and records the preprocessing policy digest, source and encoder hashes, feature layers, usable intervals, and float32 precision. Typical tensor shapes for `N` usable windows are `[N,50,1280]`, `[N,99,768]`, `[N,50]`, `[N,99]`, and `[N,2]`. |
| `training-manifest.json` | New producer tags the manifest `fk-cache-manifest-v1` and adds `source_manifest_sha256`, `preprocessing`, `window_label_policy`, `test`, and `assessments` alongside the existing `train` and `validation` lists. Cached rows retain the original label metadata and add `window_intervals`, `cache`, `cache_metadata_sha256`, and `cache_provenance`. Cache paths are relative to this manifest. |
| `training-manifest.json → assessments` | Every attempted clip records ID, split, family, clip labels, speech intervals, optional speech error, elapsed time, usable count, and clip exclusion reason. Each window records its interval, status, reasons, quality measurements, source hash, detector, derivation, crop policy, and `labels.{combined,audio,visual}`. Frame timestamps, duplicate counts, and face failure details appear when available. A clip with zero usable windows stays in assessments with `no_usable_windows` and has no training cache row. |
| Forensic `record.json` | `fk-record-v1` retains its schema tag, source hash, coverage, scores, prediction, window records, and numerical export data. `preprocessing.face_crop_policy` now always records the effective policy for this inference path. `preprocessing.speech_detection_error` appears when source speech preparation raises `ValueError`. The record commitment covers these fields. |
| `verification.json` and `fk verify` | The output still contains `valid`, `max_absolute_error`, and `prediction`. The verifier checks the trusted commitment, optional original source hash, and score reconstruction. The verification result does not independently certify the crop or detector. |

The cache reader still accepts older caches with matching legacy provenance and payload digests. The trainer also accepts older manifests without the new schema tag and policy fields. A legacy cache cannot be relabeled as an in-frame study cache by editing its JSON. Regenerate features when their crop or input preparation differs, and regenerate the manifest when older labels lack the corrected modality timing. The earlier two-clip cache artifacts remain inspection evidence. The 48-clip run has its own CUDA-produced caches. [Verified source manifest](../data/lavdf-pilot-002/manifest.json) · [Training manifest](../../../../04_Results/experiments/avtenet/lavdf-fk-cache-001/training-manifest.json) · [Forensic record](../../../../04_Results/experiments/avtenet/fk-pilot-heldout-001/record.json)

## 5. Rejected windows

I grouped the saved `assessments[].windows[]` rows by clip-level modality flags and window-level combined targets. A rejection is any status other than `USABLE`. Each of the 104 rejected windows in this run carries exactly one reason, so reason counts sum to rejected window counts. The denominators include every scheduled window unless the table explicitly limits them to full windows. All 48 clips have usable caches. [Window assessments](../../../../04_Results/experiments/avtenet/lavdf-fk-cache-001/training-manifest.json) · [Cache totals](../../../../04_Results/experiments/avtenet/lavdf-fk-cache-001/cache-checks.json)

| Clip-level class | Clips | Attempted windows | Usable | Incomplete tail | Face track | Speech | Rejected total | Full windows rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Genuine | 12 | 103 | 78 | 24 | 1 | 0 | 25 | 1 / 79 (1.27%) |
| Audio-only fake | 12 | 110 | 84 | 24 | 1 | 1 | 26 | 2 / 86 (2.33%) |
| Visual-only fake | 12 | 103 | 76 | 24 | 3 | 0 | 27 | 3 / 79 (3.80%) |
| Audio and visual fake | 12 | 111 | 85 | 24 | 2 | 0 | 26 | 2 / 87 (2.30%) |
| **Total** | **48** | **427** | **323** | **96** | **7** | **1** | **104** | **8 / 331 (2.42%)** |

The incomplete tails account for 96 of 104 rejections, exactly two scheduled tail windows per clip in this pilot. After removing those tails, seven of 252 full windows in fake clips were rejected (2.78%) and one of 79 in genuine clips was rejected (1.27%). The visual-only clips had the highest observed full-window rejection rate. There are only eight full-window failures, so these rates are descriptive.

| Combined annotation for the window | Attempted | Usable | Incomplete tail | Face track | Speech | Rejected total | Full windows rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No edit overlap (`0`) | 328 | 231 | 95 | 2 | 0 | 97 | 2 / 233 (0.86%) |
| Edit overlap (`1`) | 99 | 92 | 1 | 5 | 1 | 7 | 6 / 98 (6.12%) |

All eight full-window failures occurred in train. Train had 272 attempted windows, 200 usable, and 72 rejected. Validation had 155 attempted, 123 usable, and 32 incomplete tails. The more direct comparison for localization is the window-level table. Full windows that intersected an edit were rejected more often in this pilot, so the larger run should report assessment coverage by window label and manipulation category.

The modality target view counts each window once per modality. The audio and visual totals each reconcile to 104 and should be read independently.

| Modality window target | Attempted | Incomplete tail | Face track | Speech | Rejected total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Audio genuine (`0`) | 357 | 96 | 4 | 0 | 100 |
| Audio fake (`1`) | 34 | 0 | 1 | 1 | 2 |
| Audio unknown (`null`) | 36 | 0 | 2 | 0 | 2 |
| Visual genuine (`0`) | 362 | 95 | 3 | 1 | 99 |
| Visual fake (`1`) | 29 | 1 | 2 | 0 | 3 |
| Visual unknown (`null`) | 36 | 0 | 2 | 0 | 2 |

## 6. Tests

| Added test | What it asserts |
| --- | --- |
| `InfrastructureTests.test_boundary_face_crop_preserves_observed_pixels` | The strict crop rejects a boundary-crossing detection. The in-frame crop returns 224 × 224 face and 96 × 96 lip images with the original uniform pixel value. It rejects a fallback crop that cannot retain a required landmark. |
| `LavdfTests.test_pilot_keeps_official_splits_and_modality_labels` | Train and dev stay separate, dev maps to validation, selected media have hashes, an out-of-duration edit is excluded, audio-only labels and intervals are correct, and both-modified modality intervals remain null. Test metadata are present in the fixture, and test media are absent. |
| `LavdfTests.test_cross_split_original_is_rejected` | A derivative whose original points to a different official split raises a validation error. |

The full suite was rerun for this report with the CPU environment. The command and summary were:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& .\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

```text
Ran 81 tests in 14.010s

OK
```

The suite includes existing tests for strict crop geometry, synthetic video inference and verification, cache integrity, training from a trusted manifest, deterministic CPU resume, and validation-only threshold selection. The added code paths also have recorded real-data checks:

- CUDA cached all 48 clips. The checks confirmed usable labels for one audio-only, one visual-only, and one both-modified clip and preserved the unknown separate timings. Representative face crops were inspected. [Cache checks](../../../../04_Results/experiments/avtenet/lavdf-fk-cache-001/cache-checks.json) · [Visual clip checks](../../../../04_Results/experiments/avtenet/fk-pilot-inspection-001/visual-clip-checks.json)
- The four-epoch RTX 4060 run resumed after epoch 2 and matched the continuous run in parameters, optimizer tensors, history, CUDA RNG, and 128 steps. CPU→GPU and GPU→CPU also advanced an epoch and preserved history and step counts. Training loss went from 35.25 to 23.11. Validation loss was lowest at epoch 2, and calibration used 16 clips. [Resume checks](../../../../04_Results/experiments/avtenet/fk-gpu-pilot-resumed-001/resume-checks.json)
- On `dev/004053.mp4`, a genuine clip from a family outside the pilot, the checkpoint crop policy was inherited and recorded. `fk verify` returned valid with zero reconstruction error. [Holdout selection](../../../../04_Results/experiments/avtenet/fk-pilot-inspection-001/forensic-holdout.json) · [Verification output](../../../../04_Results/experiments/avtenet/fk-pilot-heldout-001/verification.json)

**Coverage gaps.** The automated suite has no direct full `cache-fk` CLI test, no dedicated tests for each `--device` branch or explicit crop override conflict, and no automated real-media or CUDA resume test. The saved pilot runs cover the successful CUDA and cross-device paths. Further untested branches include speech preparation error reporting, a clip with zero usable windows, interrupted cache creation, old checkpoint migration with an absent step, and new manifest-policy mismatch handling. Crop testing uses one synthetic five-landmark fixture and a visual spot check. VAD outputs were inspected for selected windows. There is no independently annotated speech or face track assessment across the dataset, multi-host test, or RTX 3070 run.

## 7. Risks and open items

### Pilot choices and full-run constraints

- **Subset selection.** `prepare-lavdf-pilot` defaults to 8 train and 4 dev families and chooses one genuine, audio-only, visual-only, and both-modified clip per eligible family. That changes the dataset's class mixture and excludes incomplete families. Freeze a separate study subset with explicit counts and sampling rules per official split. Keep the source-family audit and document the four annotation exclusions. Source-family disjointness was verified. Identity disjointness was not independently established from the local manifest.
- **Paths and cache jobs.** The documented dataset root, output paths, and asset registry are local examples. The registry contains absolute paths from this PC. Each host needs paths resolved against its own copied assets, with checksums verified. A cache run requires a fresh destination and has no resume or merge command. A failure can leave partial caches without a completed manifest. For parallel caching, add a reliable shard and merge workflow or use separately verified manifests with an explicit merge audit. Each shard starts cache numbering at zero.
- **Preprocessing and throughput.** Window length, stride, 25 fps, input shapes, two quality inputs, and the 50% speech gate are set by the preparation and encoder code. Changing the config's descriptive values alone does not change extraction. YuNet, tracking, VAD, decoding, and cropping run on CPU. The frozen encoders run on the selected device. The measured 1.66 seconds per clip is for 48 local clips on one host, excluding model startup, and should be measured against representative lengths and storage on the other machines. The manifest records Torch and encoder device but does not capture every dependency version, driver, GPU architecture, or decoder setting.
- **Model and checkpoint policy.** Training processes all usable windows of one video per optimizer step. There is no batch-size flag, AMP, or learning-rate scheduler. Longer clips can increase peak memory on 8 GB cards. The configured defaults are AdamW at `1e-4`, up to 50 epochs, patience 5, and seed 7. The pilot used 4 total epochs. Its threshold `0.14605498313903809` came from 16 validation clips and should be recalibrated for the frozen paper subset. The default video and window thresholds are both `0.5` before calibration. `run.json` is written with `STARTED` and remains a configuration snapshot. Read `metrics.json` or the process exit status to confirm completion.
- **Policy enforcement.** Training validates the cache policies against the manifest and inference inherits the saved crop policy. The inference path verifies the supplied asset files against their registry but does not compare their digests with training-time asset digests. Pin the same registry content across hosts and strengthen this check before collecting paper results. Mixed cache jobs with different `encoder_device` or Torch version have different policy digests. Verify consistency before combining them.

### Section III-E internal baselines

The cached EfficientNet-B0 and Wav2Vec2 features, usable window intervals, crop policy, and audited splits can be reused for video-only, audio-only, and plain concatenation fusion baselines. The branch's existing FK ablation settings do not implement these three baselines as specified. Each baseline still needs its model, training and validation selection logic, and shared comparison reporting. Decide whether unimodal models use the common FK eligibility mask or have separate coverage reported. Compare models on the same clip IDs and declare any common-assessed analysis. Use the shared clip-level detection target where the comparison calls for video authenticity. Keep modality targets separate for any modality-specific analysis. For both-modified clips, evaluate localization on the combined timeline because separate modality intervals are unknown. LAV-DF's caption score is unavailable. AVFakeNet and AVTENet remain published references, with their original datasets and hardware identified.

### Windows and CUDA portability

- The verified host used Windows, Python 3.14.5, PyTorch `2.15.0+cu132`, CUDA runtime 13.2, and NVIDIA driver `616.92`. The installed wheel reported `sm_86`, which is the RTX 3070's compute capability. This confirms architecture coverage for that wheel. Driver, memory headroom, and actual runtime still need checking on each host. Run a CUDA tensor operation, a real encoder window, and a training/resume smoke test on each. [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus) · [CUDA 13.2 compatibility notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)
- The tested CUDA environment lives at `.cache/fk-cuda-venv` and the full unit suite uses the existing CPU `.venv`. Recreate environments for the host's OS and wheel ABI. Repoint local asset registry paths, ensure FFmpeg and FFprobe resolve, and check imports of PyAV, OpenCV, and Safetensors. Some media tests look for an extensionless `ffmpeg` beside the Python executable. This Windows setup provides that name and `ffmpeg.exe`. Silero's `torch.jit.load` worked but emitted a Python 3.14 support warning. Validate it on every target environment.
- `--device auto` can fall back to CPU. Use `--device cuda` and check recorded `device` and GPU name for GPU measurements. The CLI selects one visible GPU at index 0. Deterministic mode sets a default `CUBLAS_WORKSPACE_CONFIG=:4096:8` if the environment does not already define it. GPU resume equality was measured on the same host and software stack. Cross-device resume is operational and can produce different numerical results.

## Summary

The verified label adapter, fixed in-frame preprocessing, cache format, and FK training and forensic verification paths are ready for a larger dry run with the same pinned assets and policies. Before committing to the paper run, freeze the subset and evaluation rules, test the other hosts with real clips and resume checks, make parallel caching and registry relocation reliable, and implement the Section III-E baselines against the shared caches. Track assessment coverage by label and recalibrate on the frozen validation split before opening the test results.
