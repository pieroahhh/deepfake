# Four shared-encoder controls on Machine A

This workflow trains Visual-only, Audio-only, Concatenation, and Late fusion
sequentially using the existing FK frozen-feature cache. Each control uses seed 7
and up to 5 epochs, matching the current FK run. Best weights are selected by
validation loss; each control then gets its own validation-selected aggregation
parameters and thresholds. All four checkpoints are frozen before test scoring.

## Commands

Run in PowerShell. Each command is one line:

```powershell
Set-Location -LiteralPath 'C:\Users\teeku\Desktop\fk\Deepfake-Detection'
& '.\.venv-gpu\Scripts\python.exe' scripts/run_shared_controls.py --check
```

Continue only after `PRECHECK PASSED`:

```powershell
& '.\.venv-gpu\Scripts\python.exe' scripts/run_shared_controls.py --run
```

The script verifies manifest/label separation, cache policy compatibility and
the FK reference checkpoint. Training reads and checks each feature file, and
the resulting control data signatures must exactly match FK before calibration.
It runs one model at a time on the GPU. Feature extraction and quality fitting
are reused. Media, encoder weights and new dataset downloads are not needed.

All model-specific outputs are under `runs/shared-controls-001/<model_kind>/`.
Each model has a configuration, training attempts, selected checkpoint,
validation report and held-out test report. The original caches and FK reports
are reused in place; test-cache views contain absolute local cache references,
so rerun the workflow with local paths if relocating to a different computer.

## Progress and recovery

The active console shows the model, epoch, train/validation clip progress about
every 10 seconds, and losses at completed epochs. Initial signature validation
reads the feature cache before epoch progress appears. Calibration also reads
validation features and searches its parameter grid; it can be quiet temporarily.

From another PowerShell window in the repository:

```powershell
Get-Content '.\runs\shared-controls-001\workflow.log' -Tail 20 -Wait
```

Pressing Ctrl+C in this log-viewing window only stops the viewer. In the training
window Ctrl+C stops the workflow. Rerun the same `--run` command to continue.
Completed models are reused; interrupted training resumes from the last completed
epoch into a new attempt. Work since that epoch is repeated. A completed training
budget with interrupted calibration is recovered without retraining. Interrupted
evaluation restarts that evaluation from the shared cache.

Do not edit source code, configs, inputs or change Git revisions during a run:
the workflow binds input and source hashes and rejects mismatched restarts.
File/OS/GPU locks are released when a process exits; do not delete active locks.
If a cache file reports Access denied, run the command in the same Windows account
and PowerShell environment that successfully ran the FK test; do not rebuild caches
or disable provenance checks merely to bypass the error.

## Results and interpretation

Wait for `FINISHED`, then read:

```powershell
Get-Content '.\runs\shared-controls-001\summary.json'
```

The summary contains separate video metrics, decision settings and checkpoint
hashes for all four controls. Compare them with `runs/heldout-001/fk/summary.json`.
This is the same local held-out subset of 1,000 unused official validation clips;
it is not official benchmark test performance. Retain coverage in comparisons.

The controls intentionally share FK's usable windows: training has 2,912 usable
clips out of 3,000, validation 485 out of 500, and held-out test 980 out of 1,000.
Even unimodal controls use this joint assessment population. Therefore these
results compare classifiers under common coverage; they do not measure independent
image-only or sound-only preprocessing coverage.

Visual-only uses visual features with a GRU and projection. Audio-only uses audio
features and projection. Concatenation joins projected visual/audio features.
Late fusion averages probabilities from independently supervised visual/audio
branches in one paired checkpoint. Their training losses are independent; the
paired checkpoint and final decision settings are selected using validation.

This limited run uses one seed and the fixed five-epoch budget. It does not run
FK component ablations, degradation studies, or AVFakeNet/AVTENet.
