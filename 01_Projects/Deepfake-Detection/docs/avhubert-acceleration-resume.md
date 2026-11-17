# Machine A: accelerated AV-HuBERT, progress and resume

Updated 15 November 2026 (Bangkok). The isolated run is at
`C:\Users\teeku\Desktop\fk\avh-a`. It trains the audiovisual expert for one
epoch on the frozen 3,000 training / 500 validation clips, then extracts its
selected expert features. Machine C can continue its visual work independently.
Fusion still needs C's verified audio and visual artifacts.

## Resource settings and measured preparation speed

WSL now permits 20 GB RAM, all 12 logical processors, and 4 GB swap. RAM is a
limit, not a reservation. The host has 32 GB RAM and an RTX 3070 with 8 GB VRAM.
Giving every remaining byte to WSL can make Windows page; a larger limit alone
does not accelerate model arithmetic. Keep other large workloads closed.
The previous `.wslconfig` is backed up as `avh-a/wslconfig.before-acceleration`.

The training process uses eight PyTorch/OpenMP/MKL CPU threads. A small real
preparation benchmark measured 12.87 / 10.13 / 10.04 / 11.09 seconds at
4 / 6 / 8 / 12 threads respectively. Eight was fastest in that sample; this is
not proof of a universal optimum. WSL still exposes all 12 processors.

Enabled equivalent-input optimizations:

- Reuse sequential video decoding across overlapping windows.
- Reuse MTCNN detections for identical RGB frames within a clip.
- Read ahead up to four already prepared CPU cache entries.
- Reuse the prepared input cache from the original attempt.

On six real windows, reference preparation took 18.59 seconds and optimized
preparation took 9.58 seconds. All usable raw arrays and model inputs matched
exactly; one failed window produced the same failure in both paths. This is a
preparation benchmark, not an end-to-end training speedup claim. Cached early
windows are faster than unseen windows, so initial ETA is not representative.
The model, precision, labels, input order, seed, and one-epoch budget are unchanged.
The CPU handles decoding, preprocessing and cache reads while CUDA trains the
model. Missing raw windows are still prepared on the main thread; this is not a
fully parallel multi-process media loader.

## Watch progress from any PowerShell directory

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\avh-a\watch.ps1'
```

The watcher prints the stage, epoch, split, processed/total windows, usable
training steps, speed, split ETA, and latest checkpoint step. It refreshes every
10 seconds. Ctrl+C stops the watcher only; training continues in the background.
Progress is updated after a window finishes, so a long operation can delay it.
ETA covers the current split only, not the subsequent validation/features stage.
Processed windows include preparation failures; failures never enter loss metrics.

For a single snapshot, append `-Once`. To read the detailed stage log:

```powershell
Get-Content 'C:\Users\teeku\Desktop\fk\avh-a\code\runs\worker-c\avtenet\logs\audiovisual-train.log' -Tail 20 -Wait
```

## Pause safely

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\avh-a\pause.ps1'
```

Wait until the watcher reports `PAUSED` before closing Docker or rebooting.
The current window finishes, an atomic checkpoint is written, and the controller
exits without starting feature extraction. During feature extraction it commits
the SQLite cache before pausing.

## Resume after pause, reboot, or a resolved failure

Open Docker Desktop and wait for its engine to be ready, then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\avh-a\start.ps1'
```

This removes the pause request and launches the audiovisual-only queue. It rejects
an already running controller/container. It picks the newest checkpoint-bearing
attempt, including when a later attempt failed before saving anything, and writes
into a new attempt directory. Do not delete older attempts or the prepared cache.
No automatic restart loop is installed: relaunch explicitly after a failure.

Training saves `last.pt` after its first step and approximately every 120 seconds
thereafter, plus at epoch completion or a requested pause. An abrupt power loss
replays work since the latest completed checkpoint (normally about two minutes,
plus the current operation/checkpoint write). Resume restores model weights,
Adam state, RNG state, split metrics and the source/window cursor. It does not
retrain all earlier windows. Partial epoch checkpoints have `epoch: 0` and a
`partial.epoch: 1` section while epoch one is in progress; this is intentional.

`best.pt` is selected only after validation completes. It is expected to be absent
during the first training epoch. Do not use an unfinished `last.pt` as a selected
evaluation checkpoint. Each AV-HuBERT optimizer checkpoint is about 1.31 GB, so
retaining many attempts consumes disk space. Checkpoint writes replace the latest
file atomically within an attempt instead of accumulating one file per interval.

Partial feature caches created by this version have a producer record. On relaunch,
the controller checks the selected checkpoint and population, snapshots committed
features, preserves the interrupted folder, and verifies feature parity before
continuing. Legacy/unidentified partial caches are rejected for manual diagnosis.

## Verification and provenance

- 39 targeted host tests passed: preparation equivalence, training/resume,
  checkpoint identity, feature recovery, and comparator workflows.
- Eight targeted tests passed in the pinned Torch 1.13 container. Mid-epoch resume
  exactly reproduced a continuous CPU run's model, optimizer, RNG and history
  after interruptions in train/validation and at split boundaries.
- The actual AV-HuBERT CUDA run saved a step-one checkpoint, paused, and resumed
  at step two in a new attempt. GPU floating point training is not claimed to be
  bitwise reproducible across machines.
- `attempt-001` is preserved from the original interrupted run; it had no completed
  epoch checkpoint. `attempt-002` is the real safe-pause verification checkpoint.
  The resumed run starts in `attempt-003`; check status for subsequent attempts.
- `avh-a/acceleration-source.json` records deployed source hashes and before hashes.
  Original changed source files are in `avh-a/source-before-acceleration`.
  `bundle.json` / `source-bundle.json` describe the original verified transfer,
  not the modified runtime source. Original data and assets remain unchanged.
- Equivalent input optimizations came from Machine C commit
  `19d4595a965db0c747c0ee5a8d13096ba4c0ff90`; mid-epoch resume, pause, monitoring
  and owned feature recovery were added locally. The source changes are also
  copied to the main repository and have not been committed or pushed.

Frozen population SHA-256:
`3c0955b7daaeed0d22b0efe68d06ee78146a8f43c88a5c3df28e1db3e13f5fea`.
Shared index SHA-256:
`6eb102d507559f5ef22f16f8da348a58b30a6771ce3e5bdca82473f9881c9d34`.
Container image: `fk-avhubert:cuda117-v1` (Torch 1.13.1+cu117).
Host controller: the main repository's `.venv-gpu` (Torch 2.2.2+cu121).
The cross-checkout GPU queue remains in `Deepfake-Detection/.cache`.

The input cache needs at least 5 GiB free before it writes a new entry. All 30,421
potential train/validation windows have an estimated 16.7 GiB uncompressed raw
payload; compressed storage varies. Windows must stay awake and Docker must stay
running. Do not overwrite this folder with Machine C's continuation bundle;
validate and merge its completed expert artifacts separately.
