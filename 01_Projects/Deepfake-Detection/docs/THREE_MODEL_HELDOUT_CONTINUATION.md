# Locked preliminary three-model comparison

## Latest execution update

The original tool-launched process tree disappeared without a traceback while
status remained RUNNING at 204 clips. The exact exit cause is not established.
The evaluation was relaunched through a **user-level Windows Task Scheduler**
task named `FK-AVFakeNet-Heldout-002`, independent of the tool's process tree.
It resumed completed cache records and safely paused at 234/1,000 for execution
benchmarks. No completed cache was deleted or overwritten.

Current task action: `.venv-gpu\Scripts\pythonw.exe` running
`scripts/tune_avfakenet_heldout.py`. Logs are `scheduled-*.log` in the held-out
run. `execution-tuning-*.json` records code hashes and execution settings.
The original locked evaluator, model checkpoints and comparison source remain
unchanged; this separate wrapper only prepares missing cache records and changes
the execution batch size for inference before returning to the original evaluator.

### Preparation

On the same 12 completed held-out clips (duration-spread selection, no selection
by prediction outcome), six workers took **51.403 s**, eight **51.232 s**.
Every array-file hash and preparation record matched the completed reference.
Eight workers were the fastest measured but the difference is only ~0.3%; it
should not be represented as a meaningful preparation speedup. Both settings
use two Torch CPU threads per worker and the same bounded decoder/prefetch.

Evidence: `runs/avfakenet-heldout-preparation-benchmark-001/report.json`.

### Inference

On four validation clips, using the same first up-to-192 frames of each, the
batch benchmark measured:

| Batch size | Best of two measured passes |
|---|---:|
| 8 | 3.3911 s |
| 16 | 2.7541 s |
| 32 | 2.8161 s |
| 64 | 2.7236 s |

Batch 64 was selected by execution time: ~19.7% lower elapsed time / ~24.5%
higher throughput than batch 8 in this benchmark. These timings included a
competing preparation process and do not guarantee a full-run speedup.

FP32 logits are **not bitwise identical** across batch sizes: maximum observed
batch-64 difference was ~0.000195, while every validation frame argmax matched.
The execution wrapper accepts <=0.001 validation-logit difference and repeats
near-tie chunks (absolute class-logit gap <=0.001) with the original aligned
batch size of eight. This is a numerical guard, not a new model threshold.
Precision remains FP32 with TF32 disabled; weights, majority vote, audio inference
and hard OR are unchanged. This does not constitute a proof of numerical equality
for every unseen input; the batch-size execution difference is explicitly logged.

A separate integration check on four **complete** validation clips found exact
agreement in native clip decisions, visual vote fractions and audio probabilities.
Two tests passed for original-batch near-tie recomputation and preserving the
original preparation-rejection path. Evidence:
`runs/avfakenet-inference-batch-benchmark-001/{report.json,native-parity.json}`.

### Metrics and figures

These already use CPU and existing retained predictions. No new fitting or GPU
work was added. All comparison denominators, thresholds and source hashes remain
the locked original procedure. Tables and figures are generated automatically
after inference. Do not claim that these short phases received a measured speedup.

The existing watcher command still works. If this latest wrapper is paused,
wait for PAUSED, remove only `pause.request`, then start the existing Windows
task again. Its action is not the older tool-launched supervisor command.

The user authorized proceeding with the current AVFakeNet adaptation despite
its weak validation and low coverage. Do not change preprocessing, checkpoints,
sampling or thresholds to improve the held-out outcome. Preserve historical
results and distinguish this adaptation from the original AVFakeNet paper.

## Active workflow

`scripts/evaluate_avfakenet_locked_heldout.py --supervise` is running in the
background. Output: `runs/avfakenet-heldout-002`.

1. Verify and freeze the completed one-epoch AVFakeNet branch checkpoints.
2. Verify the same local held-out manifest used by FK/AVTENet: 1,000 clips,
   SHA-256 `c60bc02bf696360c3dfce8eca2a9a895b84db864e51255639e1529e23893c5b5`.
   Check ID, source-family and content-hash separation from the shared 3,500
   training/validation population.
3. Prepare held-out clips using six workers with the existing strict face/audio
   rules, persistent decoding and bounded prefetch. Eligibility is not relaxed.
   Per-clip cache records are checked and reusable on resume.
4. Load the selected frozen branch models in evaluation mode. Infer using the
   existing batch size of eight and FP32; retain majority vote, explicit ties and
   hard OR. Save each prediction separately, permitting clip-boundary resume.
5. Write all 1,000 predictions, metrics, coverage and protected-source checks.
6. Automatically run `scripts/compare_three_heldout.py`, creating tables and
   PNG/PDF figures in `runs/three-model-heldout-002`.

This is a local held-out set sampled from unused official validation, not the
dataset's official test. The workflow does no fitting or threshold selection.
Hardware-dependent preparation differences between B's training cache and A's
held-out preparation remain a declared limitation.

## FK file access resolved

The user copied the unreadable original `runs/heldout-001/fk/test.json` into
`C:\Users\teeku\Desktop\fk\fk-heldout-copy.json`. The copy is readable and has
1,000 per-video entries. Its confusion counts, accuracy, BA, F1 and AUC must match
the original readable FK summary before comparison output is accepted. All
comparison rows must also match the exact held-out IDs and labels.

## Comparison interpretation

The table/figure contains three models and one supplementary threshold row:

- FK, using its existing frozen decisions.
- AVTENet adaptation, original threshold 0.5.
- The current one-epoch normalized AVFakeNet adaptation.
- AVTENet validation-calibrated threshold, clearly labeled exploratory because
  its calibration followed inspection of the original held-out results.

Report each model's coverage and metrics on its own assessed clips, then report
paired metrics on the intersection of assessed clips. Report the intersection
size and clip IDs. This selected subset does not eliminate missing-data bias for
the full population. AVFakeNet has no continuous fused score; its fused ROC AUC
must remain N/A rather than being invented from the two hard branch decisions.

Disclose unequal training budgets: FK five epochs, AVTENet stages one epoch,
AVFakeNet one pilot epoch plus its separately documented tiny recovery warmups.
Do not use these exploratory adaptations to claim verified superiority over the
authors' original methods.

## Validation and locks

- Original completed training artifact hashes are checked before inference and
  after evaluation, including both selected checkpoint files.
- `locked-before-inference.json` records identities, decisions and protected
  inputs before the first held-out inference.
- A real validation clip passed decision parity against the existing native
  `predict_prepared_video` helper; evidence: `validation-parity-check.json`.
- Two targeted tests passed for common-assessed denominators, coverage, and
  rejecting duplicate IDs or mismatched labels.
- An OS lock prevents duplicate workers in this run. The optional supervisor
  retries an unexpected process exit up to three times using verified completed
  cache/predictions. Explicit FAILED or PAUSED status does not trigger retries.
- Progress updates use the Windows sharing-retry writer. Metadata failures are
  fatal; an unavailable progress display alone does not abort computation.

## Watch, pause and resume

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\scripts\watch_avfakenet_pilot.ps1' -RunRoot 'C:\Users\teeku\Desktop\fk\Deepfake-Detection\runs\avfakenet-heldout-002'
```

The existing watcher also follows `next-run.json` from the previous training run
to this held-out run. Restart an already-open watcher to load the latest logic.

Create `runs/avfakenet-heldout-002/pause.request` for a safe pause. Preparation
drains in-flight clips; evaluation pauses between clips. After `PAUSED`, remove
only that pause request and resume from the repository root:

```powershell
& '.\.venv-gpu\Scripts\python.exe' -u scripts/evaluate_avfakenet_locked_heldout.py --supervise
```

Inspect `progress.json`, `worker-*.log`, `worker-*.err.log`, `launcher.json` and
`supervisor.json`. After completion, inspect `summary.json` and
`runs/three-model-heldout-002/COMPARISON.md`, `comparison.json`, `comparison.png`
and `comparison.pdf`. Source-code changes require a new run identity/output.
