# Independent comparison models

[ภาษาไทย](README.th.md) · [AVFakeNet](avfakenet/README.md) · [AVTENet](avtenet/README.md)

These folders contain paper-guided reconstructions adapted to AV-Deepfake1M
(2024). They are not authenticated author implementations or pretrained deepfake
detectors. AVFakeNet starts its classifiers from scratch. AVTENet requires three
actual pretrained backbones, then task-specific expert training and fusion.

The FK workflow and its `.venv` remain separate. All models share one audited
population manifest; they do not share FK's frozen EfficientNet/Wav2Vec2 cache.
The official `keep-official` split policy is retained without removing families.
Unknown modality labels are excluded from that modality's supervision and counted
in `preparation-events.jsonl`. Media failures remain visible in evaluation coverage.

## Setup on another Windows machine

Run from the cloned repository. Install Git, Python 3.11 and uv first. The paths
below are machine-local; neither datasets nor weights are distributed by Git.

```powershell
$Repo = (git rev-parse --show-toplevel).Trim()
Set-Location -LiteralPath $Repo
$DatasetRoot = Read-Host 'Extracted AV-Deepfake1M directory'
$RunRoot = Read-Host 'Directory for experiment outputs'
$Population = Read-Host 'Completed population manifest.json path'
$env:HF_HOME = Join-Path $Repo '.cache/huggingface-comparators'
uv --cache-dir .cache/uv-baselines venv --python 3.11 .venv-baselines
uv --cache-dir .cache/uv-baselines pip install --python .venv-baselines/Scripts/python.exe -r baselines/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv --cache-dir .cache/uv-baselines pip install --python .venv-baselines/Scripts/python.exe --no-deps -e .
$Python = Join-Path $Repo '.venv-baselines/Scripts/python.exe'
& $Python -m baselines.avfakenet check
```

This recipe installs CPU wheels for integration testing. A GPU training machine
needs matching CUDA builds of the pinned Torch/Torchvision/Torchaudio versions in
its own environment. Select `--device cuda` only after checking CUDA availability.
Do not reinstall dependencies into the active FK environment.

The `check` command verifies dependencies and asset metadata; it does not claim
that task training, pretrained accuracy, or a real-video integration test passed.

## Create model-specific indices

Wait for `prepare-avdeepfake1m` to finish successfully and produce both
`manifest.json` and `audit.json`. If those files were copied from another machine,
they can be reused with that machine's extracted dataset paths.

```powershell
$AVFake = Join-Path $RunRoot 'avfakenet-001'
$AVTE = Join-Path $RunRoot 'avtenet-001'
& $Python -m baselines.avfakenet prepare --manifest $Population --output "$AVFake/index"
& $Python -m baselines.avtenet prepare --manifest $Population --output "$AVTE/index"
```

`prepare` creates a compact SQLite metadata index. It does not decode all videos
or allocate a full frame cache. No dataset files are copied or changed. Outputs
must be new directories. Omitting `--limit-per-split` includes every manifest row.
For a separate smoke run, use `--limit-per-split 8` and another output directory;
this explicit small subset is recorded and is not the full benchmark population.

Run model-specific commands from the linked guides. Every command reports errors
with a nonzero exit status. Inspect `$LASTEXITCODE` before starting the next stage.
First-frame/first-step warmup can be slow on CPU. Progress is printed every 100
training steps, and checkpoint/metrics files are updated after each full epoch.

## Checkpoints, resume and portability

- `best.pt`: first strict improvement on the stage's validation criterion.
- `last.pt`: model, optimizer, Python/NumPy/Torch RNG, history and best epoch.
- `run.json`: model, preparation, dataset hash, source-overlap policy, detector
  weights, seed, stage, schedule and asset provenance.
- `metrics.json`: epoch losses/counts and selected epoch.
- `preparation-events.jsonl`: unavailable media, unknown supervision and reasons;
  entries can repeat across epochs, so count unique IDs for population exclusions.

Resume with `--resume <old-run>/last.pt` into a **new** output directory. Keep
`best.pt` beside `last.pt`; the runner verifies their matching identity. Specify
the total epoch budget, not the number of additional epochs. Model/config/assets,
dataset index, seed, schedule and integration-step cap must match. A completed
early-stop run cannot be resumed as if its stopping state did not exist.

Indices contain relative dataset paths and can be copied unchanged. Supply a
new `--dataset-root` on the destination machine. Copy matching checkpoint files
and feature-cache metadata/payloads as a unit. Recreate environments locally;
do not copy `.venv` directories. Exact CPU resume is tested on the same runtime;
cross-version/hardware numerical equivalence is not promised.

## Evaluation and FK comparison

Evaluation writes `predictions.jsonl` and `report.json`, retaining every selected
video ID including `UNASSESSED`. Reports include coverage, TP/TN/FP/FN, precision,
recall, specificity, F1 and balanced accuracy. Undefined metrics are `null`.
AVTENet additionally reports ROC-AUC when both classes and continuous scores are
available. AVFakeNet's hard OR does not define a continuous fused score or AUC.

Pass `--compare-with <fk-evaluation.json>` to either evaluator, or provide another
comparator's `predictions.jsonl`. The report verifies matching labels, reports
coverage separately, and compares metrics on IDs assessed by both methods.
Validation metrics are selection/development results. The release's unlabeled
test list cannot produce classification metrics; a genuinely labeled test
population must be supplied for `--split test`.

## Software tests and assets

```powershell
& $Python -m unittest discover -s tests -p 'test_comparator_workflows.py' -v
```

Tests cover source overlap policy, corrupt indices, path traversal, unknown and
partial targets, whole-window boundaries, exact CPU resume, feature alignment and
coverage. Existing architecture/fusion tests remain available through legacy
imports. Synthetic fixtures never count as real pretrained-model validation.

AVTENet readiness is currently limited by the official factorized ViViT download
returning HTTP 403 during local acquisition. Its conversion/parity must pass
before visual expert training or the complete ensemble can be claimed ready.
Consult [HANDOFF](../docs/handoffs/HANDOFF.md) for the latest actual tests and remaining limits.
