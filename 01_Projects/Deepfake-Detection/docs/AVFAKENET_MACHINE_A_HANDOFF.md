# AVFakeNet recovery on Machine A

## Current state

### Unexpected process exit and checkpoint recovery

The user reported an issue after the Visual worker stopped without a Python
traceback. The last saved status still said RUNNING at 392,392/586,130 images
(66.95%), but the training process no longer existed and the progress timestamp
was stale. The exit cause has not been established; do not label it as a model,
CUDA-memory or Windows failure without evidence.

The last checkpoint loaded successfully, matched the run identity, contained
finite model tensors, and recorded epoch 1 / cursor 392,392. The prefetched runner
was restarted from this checkpoint, including optimizer and RNG state. Its brief
`prepare` stage rechecks existing cached arrays rather than redetecting faces.
The latest PID is in `prefetch-launcher.json`. Exit callbacks now record an exit
code/signal if observed by the launcher runtime. The watcher warns after 180
seconds of stale RUNNING status; restart the watcher to load that change.

### Training input prefetch enabled

The homogeneous-cache one-epoch run reached Visual training. CPU thread counts
1, 2 and 4 had nearly identical compute timing; removing repeated GPU input/loss
validation synchronization also gave no measured speedup and was **not enabled**.
Those exploratory paths remain inactive.

The selected optimization changes only `Inputs`: a single CPU worker prepares
one batch ahead, with up to 1,024 read-only mapped files rather than 32. Model
forward, loss, FP32 precision, per-step finiteness/gradient checks, batch size,
optimizer and exact sample order remain the original runner's implementations.

On the same 50 real batches, including five warmup batches, measured time for
45 batches fell from **4.3821 s to 3.8359 s** (about **14.2% higher throughput**).
Every output logit and all final model weights matched bit for bit. The benchmark
ran alongside live training, so the actual speedup is indicative rather than a
guaranteed end-to-end gain. Evidence:
`runs/avfakenet-training-prefetch-benchmark-001/report.json`.

Two additional tests passed for unchanged forward/loss/gradient mathematics in
the exploratory path and exact prefetched input/label equality, including resume
and switching to validation. Only the input-loader optimization was applied.

The live worker was paused at a saved batch boundary and resumed using
`scripts/resume_avfakenet_prefetched.py`. No epoch restart or completed-cache
rebuild was required. This wrapper verifies original runner hashes and benchmark
equivalence, then installs only the prefetched loader. It records its provenance
in `runs/avfakenet-one-epoch-001/input-prefetch-*.json`. Logs and PID are in
`prefetch-*.log`, `prefetch-*.err.log`, and `prefetch-launcher.json` in that run.

If this specific training process is paused, remove the acknowledged pause
request and run `resume_avfakenet_prefetched.py` with `.venv-gpu\Scripts\python.exe`
from the repository root. The existing watcher continues to follow this run.

### Complete Machine B cache received

Source: `C:\Users\teeku\Downloads\LANDrop\prepared-cache-machb` contains all
3,500 paired NPZ/JSON cache entries and the original identity. Full payload and
source hashes were verified against the shared population. Architecture,
preprocessing, decision rules and detector provenance match the recovery pilot.

The overlap audit compared 814 clips already prepared on A. It found 748 Mel
arrays with numerical differences, 361 visual arrays with pixel differences,
and five eligibility/file-set differences. On the first eight examples, Mel
differences were approximately 1.2e-7–1.94e-7 at maximum. Face differences were
larger and were not dismissed as floating-point noise. The cause was not uniquely
established. **The two caches were not mixed and no A record was overwritten.**

The original preparation worker was safely paused at 814 completed records.
The first import audit intentionally refused to fill A's missing records because
the exact-equality requirement failed. Its full report is in
`runs/avfakenet-normalized-pilot-001/machine-b-cache-import-*/report.json`.

The selected continuation instead uses **Machine B's pixels for every clip**,
packed without numerical conversion into a separate homogeneous directory:
`runs/avfakenet-machine-b-prepared-001`. Labels are regenerated from the same
manifest using cached source timestamps. Original NPZs and A's outputs remain
read-only. This reuses the original B preparation provenance and its eligibility
outcomes; it is not a claim of bitwise parity with A's preparation.

`scripts/pack_machine_b_cache.py --then-train` packs and verifies all entries,
then automatically starts the one-epoch runner on the same 3,000/500 population.
The pilot identity records the homogeneous cache's provenance. Two targeted tests
passed for timestamp labels, preserved array values, absent rejected modalities,
and rejection of contradictory cache records.

During packing, progress is in
`runs/avfakenet-machine-b-prepared-001/progress.json` and logs are
`machine-b-direct-*.log` / `.err.log` in the original pilot directory. Afterwards
the existing watcher follows `runs/avfakenet-one-epoch-001/progress.json`.

This eliminates further face detection preparation if packing completes; it does
not guarantee that full-network training and final evaluation meet the deadline.
Training is one additional epoch per branch from the successful tiny-sanity
parents. Held-out evaluation remains pending validation review.

Packing completed successfully for all 3,500 clips. The original B source files
passed the post-pack checksum check. The one-epoch runner has started and is
verifying the packed records/media before training; this `prepare` phase now
reuses completed arrays rather than detecting faces again. Original A preparation
is intentionally left paused. The direct background Python workflow is recorded
in `machine-b-launcher.json`; its main log is `machine-b-direct-*.log`.

### Two-hour deadline: one-epoch continuation

The user now requires the same comparison population and explicitly accepts a
reduced epoch budget. The next training stage has been changed to a **separate
one-epoch adaptation** in `runs/avfakenet-one-epoch-001`. Preparation stays in the
original run, with all completed cache records retained. The new runner requires
the complete 3,000/500 cache, verifies it, and does not silently use only the clips
prepared earliest. Training inputs, frame rate, labels and native decision rules
are unchanged. Only the pilot epoch budget changes from three to one; the earlier
tiny warmup/recovery steps still need to be disclosed.

The previous three-epoch choice came from the returned Machine B recovery
configuration (`pilot_epochs: 3`), not a paper requirement. At the time of the
deadline request, measured remaining preparation alone was about 3.5 hours.
**One epoch cannot guarantee a complete held-out comparison within two hours.**
It reduces subsequent training and epoch-validation work; it does not remove
initial face preparation or later held-out preparation/evaluation.

The local Machine B transfer contains only the tiny probe inputs, not a full
prepared cache. Reusing a compatible full cache from B could save preparation
time, but source IDs, hashes, timestamps, labels, profile and detector provenance
must be verified first. Its availability is pending a user response.

Latest continuation launcher while preparing:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\start_avfakenet_deadline.ps1'
```

Use only after an acknowledged pause with the pause request removed. It uses the
same six-worker accelerator with the Windows progress-sharing fix, then dispatches
to `run_avfakenet_one_epoch.py` with the original prepared-cache path. It does not
edit the original pilot or its identities. `next-run.json` records the handoff.
The existing watcher follows the new output when its progress file appears;
restart the watcher to load this behavior. Held-out evaluation remains manual
after reviewing validation, not automatically added to meet the deadline.

The one-epoch runner passed syntax/CLI checks, and a targeted dispatcher test
verified that dataset and parent sanity inputs are preserved, the cache is reused,
and only the next runner/output are redirected. Original-vs-one-epoch source
differences were inspected before launching.

### Windows progress sharing fix

The six-worker preparation process stopped after its status update failed with
WinError 5 while replacing `progress.json`. The traceback locates the failure
in progress reporting, not data preparation or model computation. A temporary
Windows reader lock is consistent with this error; the specific locking process
was not identified.

The restarted worker verified and reused **372 completed cache records** (the
last displayed count was 367; five in-flight clips completed while the executor
was shutting down). No records or historical outputs were deleted.

The launchers now use `scripts/run_avfakenet_resilient.py`, an explicitly logged
I/O-only runtime wrapper. It leaves the original hashed preparation/training
sources unchanged and installs a JSON writer that uses unique temporary files,
retries permission/sharing failures for up to two seconds, and skips only a
still-locked `progress.json` update. Metadata-write failures still raise errors.
When preparation finishes, the original pilot is automatically continued through
the same wrapper so its progress updates have the same protection.

The watcher now opens the progress file with Windows read/write/delete sharing,
allowing atomic replacement while it reads. Restart an already-open watcher to
load this fix. The command is unchanged. A skipped progress update may leave the
display briefly stale; worker logs continue reporting progress.

Four targeted tests passed, including a real Windows file handle held without
delete sharing, transient sharing failures, recovery on the next update, and
ensuring metadata failures are not suppressed. `io-wrapper-*.json` records the
wrapper code hashes and policy for each invocation. This fix does not change
data preprocessing, numerical precision, labels, model architecture or training
parameters.

### Latest resource tuning: six workers

The user authorized using the machine's available resources for maximum useful
throughput. Production was paused safely at **170/3,500** clips. An uncontended
benchmark regenerated the same 12 completed training clips, spanning durations
4.84–24.676 seconds and both successful and rejected preparation outcomes.

| Workers (two CPU threads each) | Wall time for 12 clips | Lowest observed available RAM |
|---|---:|---:|
| 2 | 90.07 s | 11.15 GiB |
| 3 | 72.47 s | 10.21 GiB |
| 4 | 65.23 s | 9.34 GiB |
| **6 (selected)** | **63.28 s** | **7.73 GiB** |

All prepared array hashes and record contents matched the completed reference
cache in every configuration. The reference files remained unchanged. Six workers
delivered approximately **42.3% higher throughput / 29.7% less elapsed time** than
two on this benchmark. This is the fastest measured configuration among these
candidates, not proof of a universal hardware maximum. Four workers already
produced a GPU utilization observation of 95%; adding workers gave diminishing
returns. One short CPU-only decoder test suite overlapped part of the two-worker
measurement; benchmark timings are indicative, not laboratory-isolated results.

Production now uses `scale_avfakenet_prepare.py` with **six separate workers**,
two Torch CPU threads each, persistent decoder and bounded decode prefetch.
It reuses all 170 original completed records, and will hand back to the unchanged
pilot runner after preparation. The training batch size, optimizer, numerical
precision, labels, sampling and validation protocol are unchanged by resource
tuning. Training is not run concurrently with the preparation workers.

The existing watcher remains valid and shows `prepare_scaled`, completed count,
fresh-work rate and phase ETA. The new evidence and logs are:

- `runs/avfakenet-worker-scaling-001/report.json`
- `runs/avfakenet-normalized-pilot-001/scaled-launcher.json`
- `scaled-preparation-*.json` (execution metadata and protected original hashes)
- `scaled-*.log` / `scaled-*.err.log`

Seven existing decode-prefetch tests passed, covering exact frame/timestamp
equivalence, variable frame rate, source gaps, exception cleanup and bounded
lookahead. The original pilot code and its hashed dependencies were not edited.

To resume **scaled preparation** after an acknowledged pause, remove only the
pause request and use this latest launcher (supersedes the older two-worker one):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\start_avfakenet_scaled_prepare.ps1'
```

Once preparation is complete and the original worker is training/validating,
use `start_avfakenet_normalized_pilot.ps1` to resume that phase instead.

### Preparation acceleration now active

The original worker was safely paused at a completed-clip boundary. A separate
accelerator now prepares **two clips concurrently**, with two CPU threads per
worker, persistent video decoding and one-chunk decode prefetch. All **83**
existing completed cache records and their payload hashes were verified and
preserved. The original pilot runner and its recorded dependencies are unchanged.

Benchmark: six already-completed training clips, including available preparation
success/failure categories, were regenerated in a separate directory. All record
contents and array-file SHA-256 hashes matched the original cache for each arm:

| Preparation mode | Six-clip wall time |
|---|---:|
| Sequential reference | 94.49 seconds |
| Two workers | 48.62 seconds |
| Two workers + persistent decode/prefetch | 44.65 seconds |

This is approximately **2.12x throughput** in that small benchmark, not a promise
that all remaining preparation or training will speed up by this factor. The
original worker was running during the benchmark, so timings include competing
load. Do not use the first few resumed clips for a reliable ETA.

The unchanged watcher command below now shows `prepare_parallel`, exact completed
count and fresh-work ETA. Each worker owns its detector and decoder. The same
pilot OS lock prevents the accelerator and original worker from writing together.
On completion, both preparation workers exit and the accelerator automatically
resumes the original pilot, which rechecks every cache payload and uses the same
manifest ordering, sampling, training and validation procedure.

Acceleration evidence: `runs/avfakenet-preparation-benchmark-001/report.json`.
Active execution/protected hashes: `runs/avfakenet-normalized-pilot-001/parallel-preparation.json`.
Logs: `parallel-*.log` and `parallel-*.err.log`. Two additional targeted tests
passed for identical arrays/labels/rejection reasons, read-only cache reuse, and
rejection of changed source files.

If paused **during parallel preparation**, wait for `PAUSED`, remove only
`pause.request`, then resume with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\start_avfakenet_parallel_prepare.ps1'
```

If the accelerator has completed and the phase is training or validation, use
the original continuation launcher documented below instead. Completed artifacts
remain reusable in either path. No completed training result is recomputed by
this acceleration change.

Both branches passed the **full-network tiny training sanity** checks in
`runs/avfakenet-normalized-full-tiny-004`. A new, separately versioned validation
pilot has started in `runs/avfakenet-normalized-pilot-001`.

The worker is preparing the shared **3,000 training / 500 validation clips**.
After preparation it automatically trains visual and audio for three epochs
each, selects each branch by validation balanced accuracy, and performs native
video-level validation using majority-frame vote and hard OR. It **does not run
held-out evaluation automatically**.

This is an AVFakeNet recovery **adaptation**, not an unchanged paper reproduction.
No original FK, AVTENet or historical AVFakeNet results/checkpoints were modified.
Do not present tiny sanity results as final comparison scores.

## Evidence and experimental sequence

All four full-network experiments used the exact returned attempt-003 backbone,
a freshly reset existing classifier, a fixed training-fitted feature normalizer,
200 updates, seed 7, class-balanced batches of eight, AdamW, the existing loss,
global gradient clipping at 1.0, and classifier learning rate 1e-4.

| Attempt | Feature dropout | Backbone LR | Visual train BA | Audio train BA | Both gates |
|---|---|---:|---:|---:|---|
| 001 | On | 1e-6 | 67.97% | 50.00% | Fail |
| 002 | Off | 1e-6 | 67.97% | 50.00% | Fail |
| 003 | Off | 1e-7 | 82.81% | 79.69% | Fail |
| 004 | Off | 1e-8 | **85.94%** | **87.50%** | **Pass** |

The two lower-rate candidates were recorded before running them in
`runs/avfakenet-normalized-study-001/plan.json`. All outcomes are retained.

Attempt 004 reduced visual loss by **41.43%** and audio loss by **24.78%**.
Both passed the original numerical targets of BA >= 80% and loss reduction >=
20%, plus 200 completed updates, nonconstant scores, active gradients through
the extractor and classifier, updated parameters in every extractor group,
and a classifier that was not entirely dead. The normalizer stayed fixed.
Disabling feature dropout uses evaluation mode for those modules but does not
freeze their parameters. All model parameter groups remain trainable.

The very small backbone LR means this is conservative fine-tuning, primarily
learning the head. It does not prove that the original training setup is fixed.

A fixed-input dropout probe found mean absolute feature changes of roughly
2.37 training standard deviations for visual and 12.80 for audio. This supports
investigating stochastic feature variation and normalization sensitivity; it
does not uniquely establish the cause of all historical failures.

## Verified before launching the pilot

- All returned bundle hashes and completed sanity checkpoint hashes matched.
- Both saved attempt-004 models were reloaded with strict parameter matching.
- Predictions were recomputed on the exact tiny examples and matched the saved
  confusion matrices and probability values within 1e-5.
- Every saved optimizer parameter state recorded 200 steps.
- Six targeted tests passed for normalization/gradient flow, disabling feature
  dropout without freezing weights, invalid scales, class-balanced epoch
  sampling that retains all training examples, and cached image/label loading.
- Preparation was safely paused after a completed clip and resumed in a hidden
  background process. Already completed cache records were verified and reused.

See `runs/avfakenet-normalized-pilot-001/sanity-verification.json` and the four
sanity directories for the underlying evidence.

An additional checkpoint-resume test passed after launch: replaying the saved
sampler cursor, AdamW state and RNG reproduced the remaining dropout-enabled
training updates bit for bit in the CPU test. Seven targeted tests now pass in
total. The live background worker was observed advancing to 20/3,500 prepared
clips after the handoff, with no worker error log output at that check.

## Pilot protocol

- Parent weights: both full-network sanity attempt-004 checkpoints. The extra
  attempt-003 and sanity training on the tiny training subset must be disclosed.
- Normalizer: fixed statistics from the exact tiny training examples; inherited
  in the checkpoint, never fitted on validation or held-out data.
- Training: backbone LR 1e-8, classifier LR 1e-4; feature dropout off, classifier
  dropout unchanged. Three additional epochs per branch.
- Sampling: use every majority-class prepared training example once per epoch;
  shuffled minority examples repeat to match its count. Every usable known
  training example participates. This change must also be disclosed.
- Preprocessing: the existing 25-Hz face sampling, whole-clip Mel preparation,
  timestamp labels, and eligibility checks remain unchanged. A failure in one
  modality is recorded independently during preparation.
- Selection: maximize validation balanced accuracy, keeping the earliest strict
  tie. Visual selection is at annotated-frame level; audio selection is at clip
  level. These metrics are distinct from the final video-level fused result.
- Native validation: majority-frame vote, explicit unassessed ties, then hard OR
  with the audio prediction. Fused continuous AUC stays unavailable; no invented
  fused probability or test-selected threshold is added.
- The shared population retains the user's `keep-official` overlap policy.

Preparation stores uncompressed, memory-mapped arrays to avoid repeatedly
decoding videos and detecting faces across epochs. It is still expensive because
each source frame must be prepared initially. The initial five-clip run suggested
hours rather than minutes; that sample is too small for a reliable total ETA.
Cache-reuse rates immediately after resume are not representative of new work.

## Watch progress

Run from any PowerShell directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\watch_avfakenet_pilot.ps1'
```

The watcher displays phase, completed/total work, percentage, approximate **phase**
ETA, estimated phase completion time and age of the latest progress. It suppresses
the misleading initial cache-replay ETA. Ctrl+C stops only the watcher.

The authoritative files are:

- `runs/avfakenet-normalized-pilot-001/progress.json`
- `runs/avfakenet-normalized-pilot-001/launcher.json` (launcher PID; the venv
  redirector may have a separate Python child process)
- `worker-*.log` and `worker-*.err.log` in the same directory

## Pause and resume

Create `runs/avfakenet-normalized-pilot-001/pause.request` to pause at a safe
boundary. During preparation the current clip finishes and its cache record is
committed. During training the worker saves model, optimizer, RNG, exact sampler
cursor and epoch. Checkpoints are also saved about every 60 seconds. Validation
can be replayed from the completed training boundary.

After the status becomes `PAUSED`, remove that one pause file and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\start_avfakenet_normalized_pilot.ps1'
```

The starter is a continuation launcher for this already-verified run. It does
not start a new protocol. An OS file lock prevents two workers from running the
same pilot concurrently. Resume rejects changed code/configuration/input hashes.
Do not edit the active runner or its recorded dependencies while this pilot runs.

## After completion

Inspect `summary.json`, `visual/metrics.json`, `audio/metrics.json`, preparation
failure reasons, and `validation-predictions.jsonl`. Check both branch validation
BA/AUC and native video-level coverage/recall/specificity; a branch gate pass alone
does not establish good native fused performance on localized manipulations.

Only then decide whether this adaptation is ready for a locked held-out run.
Preserve and report all historical results separately. No GitHub commit or push
was performed in this continuation.
