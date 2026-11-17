# Remaining experiments: precision fix and continuation

The original supplement stopped at `loading_frozen_fk`. Completed primary,
shared-control, core-ablation and all four aggregation results are preserved.
The 475 robustness derivative files are also complete.

## Diagnosed cause

The raw FK adapter disabled cuDNN TF32. Historical FK cache creation used
PyTorch's default cuDNN TF32 setting (enabled). This changed encoder outputs.
In isolated diagnostic `runs/remaining-fk-parity-diagnosis-002`, restoring
cuDNN TF32 produced exactly matching cached visual/audio features, timestamps
and quality on all four metadata-selected cases. Predictions matched, and
maximum raw canonical score difference was 9.202083628423452e-08, below the
original 5e-6 check. The threshold/tolerance was not relaxed.

`FrozenFK` now explicitly uses matmul TF32 disabled, cuDNN TF32 enabled and
cuDNN benchmark disabled. The continuation switches cuDNN TF32 off again before
the AVFakeNet phase, matching its retained pilot configuration.

## Start the pending work

The continuation has **not been started** as of this handoff. Automatic tool
approval failed because the account usage limit was reached, not because the
action was determined unsafe. No new continuation task was registered.

Run in an ordinary PowerShell terminal (keep the terminal open):

```powershell
& 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\.venv-gpu\Scripts\python.exe' 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\continue_remaining_after_precision_fix.py'
```

The runner redirects stdout/stderr to `runs/remaining-supplement-002/worker-*.log`;
the terminal may therefore remain quiet. Check progress in another terminal:

```powershell
& 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\.venv-gpu\Scripts\python.exe' 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\watch_remaining_all.py'
```

Restart an already-open watcher so it reads the continuation pointer. Once the
new runner starts, `runs/remaining-supplement-active.json` points to the new run.
Do not start the old `FK-Remaining-Supplement-001` task: its identity intentionally
binds the previous helper code. Do not rerun completed training/aggregation.

The new runner verifies protected historical input digests and the completed
diagnostic, then performs FK parity, real forensic checks, profiling and
robustness; AVFakeNet parity/profiling/robustness; and AVTENet expert/fusion
evaluation. It writes into a fresh namespace and preserves old failed logs.
Raw runtime stages can reveal additional errors; a successful diagnosis is not
a claim that the entire pending queue has completed.

## Report

`runs/report-0900-001/REPORT.md` contains completed results, including all four
aggregation variants. `build_deadline_report.py` can refresh it without GPU work.
The report refresher was configured to stop at 09:00 Bangkok time. FakeAVCeleb
remains deferred by the user; no independent training-seed results are claimed.

Checks performed: Python compilation of both continuation/helper scripts, two
targeted mutation/timeline checks, and inspection of all four diagnostic results.
