# Active next experiments

The previous three-model held-out comparison is complete and remains unchanged.
The user authorized continuing the next prepared experiments.

## Completed analysis

`runs/remaining-offline-001/bootstrap` contains paired source-family bootstrap
results for FK/default-threshold AVTENet/current AVFakeNet: **2,000 resamples**
over **415 common assessed clips / families**, seed 7. Every requested BA/F1
resample was defined. This measures evaluation-population sampling uncertainty,
not independent training-seed variability, and remains exploratory. The
calibrated AVTENet row is intentionally kept out of this primary bootstrap.

Files: `metrics.json`, `metrics.csv`, and three confusion SVGs. A matplotlib font
cache write warning did not stop report/figure generation. These statistical
intervals do not resolve the weak AVFakeNet reconstruction or unequal budgets.

The earlier localization summary and checked examples remain in
`runs/remaining-offline-001/localization`.

## Sequential GPU queue

Windows user task: `FK-Remaining-Core-Ablations-001`.
Runner: `scripts/run_remaining_queue.py`.
Output: `runs/remaining-ablations-001`.

Order: `full`, `no_av`, `no_temporal`, `uniform_weights`. Each has the same
**five-epoch maximum**, seed 7, original cached training/validation population,
optimizer/settings and predeclared validation-selection grid. Early stopping
follows the existing configured patience. No encoder feature recomputation or
raw-media preparation is performed.

The fresh Full run is the matched reference for these ablations; it does not
replace the historical FK checkpoint. Native Windows cache access was verified.
The underlying manifest contains 2,912 train / 485 validation cached entries,
retaining the original exclusion accounting. Epoch-boundary checkpoints and
per-run timings/selection outputs are saved by the existing trainer.

The queue publishes variant, phase, epoch and clip counts. Initialization includes
full cache/signature checks and can have no clip progress initially. It uses a
native user-level scheduled task because tool-launched worker trees previously
disappeared unexpectedly. Logs are `queue-*.log` in the queue output directory.

## Automatic evaluation after training

Windows user task: `FK-Remaining-Ablation-Evaluation-001`.
Runner: `scripts/evaluate_completed_ablations.py`.
Output: `runs/remaining-ablation-evaluation-001`.

This task waits without GPU work until `queue-complete.json` exists. It stops
waiting if the training queue reports FAILED/PAUSED. Once all four models are
ready, it records selected checkpoint hashes before held-out inference, verifies
the same training data signatures and all 1,000 held-out IDs/labels, and reuses
the original FK test feature caches. Each derived evaluation manifest explicitly
rebinds the context to the new checkpoint/training-run hashes and rebases unchanged
cache paths. It never deletes or edits the original cache/manifest.

The existing evaluator runs each selected model and writes video/window/timeline
results. A new paired report compares the four ablations against the newly
trained Full reference with 2,000 source-family resamples. No thresholds or model
selection are fitted using test results. Nonetheless, this is exploratory because
the same held-out population was inspected earlier; it is not a fresh confirmatory
test. Every predeclared variant is reported rather than only favorable outcomes.

Two tests passed for retaining labels/cache hashes and original manifest contents
when rebinding evaluation context, and rejecting a changed training population.

## Watch

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\watch_remaining_experiments.ps1'
```

It shows the current variant/phase, epoch, completed clips and number of finished
variants. After the queue completes it follows the ablation evaluation task.
Ctrl+C stops only the watcher. Raw progress files are `progress.json` in each
output directory. Timings from the first epoch should be used before estimating
the total queue duration; calibration and evaluation have separate costs.

## Pause/resume boundaries

Creating `runs/remaining-ablations-001/pause.request` pauses the queue **between
variants only**. It does not interrupt or save the current epoch immediately.
Do not kill a running worker unless necessary: the existing trainer checkpoints
at epoch boundaries. The queue does not silently overwrite an interrupted variant.
An interrupted run requires an explicit new-attempt resume through the trainer's
matching checkpoint/config/RNG API. Completed variant directories are preserved.

## Still pending

Aggregation ablations, real source-bound forensic-record experiments, standardized
end-to-end profiling, robustness derivatives/evaluation and FakeAVCeleb transfer
remain outside this active queue. Their plans are in `experiments/remaining-001`.
Do not claim they were executed merely because configuration files exist.
