# Shared local held-out population

This step prepares labeled evaluation data only. It does not run inference,
train models, fit quality normalization, or select thresholds.

The default population has 1,000 clips: 250 real, 250 visual-modified,
250 audio-modified, and 250 both-modified. This is 25% real and 75% fake;
report balanced accuracy and per-type results alongside overall accuracy.
The types come from the release metadata; any unknown modality annotations
remain unknown rather than being replaced with guessed labels.

## Selection and separation

The script reads the audited full population, using only its official validation
rows. It excludes every video ID, source family, or content SHA-256 appearing in
the existing 3,000 training / 500 validation population. Existing training and
validation data are unchanged, including their previously declared overlaps.

With seed 17, SHA-256 of `seed + NUL + video_id` ranks candidates. The lowest
ranked clip represents each remaining source family. Representatives are sorted
by rank, duplicate content hashes are removed, and the first 250 per type are
selected. Insufficient candidates cause an error. Selection does not consider
model scores or whether a model can assess a clip. There is at most one clip
per source family in the new population.

Rows retain their annotations, identifiers, relative media paths, and source
hashes. Their logical `split` becomes `test`, and `official_split` records
`validation`. This is a **local held-out subset of official validation**, not
the official AV-Deepfake1M test split. Speaker identity separation is not
established. The newly extracted official `test/` directory is not accessed.

## Prepare on Machine A

Run from the repository root. Each command below occupies one line; no PowerShell
continuation backticks are required. Python needs no extra packages for this script.
Adjust the argument paths to match your computer.

```powershell
& '.\.venv\Scripts\python.exe' scripts/prepare_heldout.py --source-manifest 'D:\Deepfake-Detection-runs\full-official-001\population\manifest.json' --used-manifest '.\runs\two-day-001\population\manifest.json' --dataset-root 'D:\AV-Deepfake1M' --output '.\runs\heldout-001\population' --clips-per-type 250 --seed 17
```

The script verifies the input manifests against their adjacent audit files,
checks the used subset's source-manifest identity, and hashes all selected media.
Progress is printed during scanning and every 100 media checks. Missing or changed
media fail preparation without substituting another clip. Only after successful
checks is the completed output directory published. An existing output is never
overwritten; verify it instead. A failed preparation can be rerun with the same
arguments if no completed output exists.

Output files:

- `manifest.json`: the 1,000 evaluation rows.
- `audit.json`: selection counts, checksums, exclusions, scope and separation checks.
- `used-manifest.json`: exact copy of the training/validation population excluded.
- `used-audit.json`: the original used-population audit, bound by checksum.

Media are referenced in place; no videos are copied. These generated files live
under ignored `runs/` and are not automatically committed or uploaded.

## Verify on any machine

```powershell
& '.\.venv\Scripts\python.exe' scripts/prepare_heldout.py --verify --dataset-root 'D:\AV-Deepfake1M' --output '.\runs\heldout-001\population'
```

For a worker using `.venv-gpu` and a repository-local dataset:

```powershell
& '.\.venv-gpu\Scripts\python.exe' scripts/prepare_heldout.py --verify --dataset-root '.\dataset' --output '.\runs\heldout-001\population'
```

Machines B and C must receive the **same four population files from A** and the
selected media at their unchanged relative paths below their own dataset root.
Their existing 3,500-clip bundles do not contain this new held-out population.
They do not need the full source manifest to run verification. Compare
`audit.json`'s `manifest_sha256` across all machines. Successful verification ends
with `VERIFIED 1000 held-out clips; zero overlap with used population`.
The verification command exits nonzero on a failure; do not print a separate,
unconditional success message in PowerShell.

## Run FK evaluation on Machine A

Run from the repository root with the same GPU environment used to create the
training cache. Defaults point to `runs/two-day-001` and the new held-out
population. All paths are overridable; see `--help` on another machine.

Check inputs without creating caches or running inference:

```powershell
& '.\.venv-gpu\Scripts\python.exe' scripts/run_fk_heldout.py --check
```

Cache the 1,000 held-out clips and evaluate the selected FK checkpoint:

```powershell
& '.\.venv-gpu\Scripts\python.exe' scripts/run_fk_heldout.py --run
```

The command checks media, checkpoint selection, the original training-cache
identity, train-fitted quality, model assets, PyTorch version and encoder device.
It combines the original population with the test rows for provenance validation,
then uses `only_split=test` to avoid re-encoding train/validation clips. It never
trains or calibrates. Progress shows completed clips, elapsed time and estimated
remaining time; the initial estimate may fluctuate. The output is under
`runs/heldout-001/fk` on the repository drive (C on Machine A), not the dataset
drive. At least 10 GiB free is required before building an incomplete cache.

Rerun the same `--run` command after interruption. It verifies and reuses completed
clips, rebuilding corrupt clips while retaining the old copies in quarantine.
If evaluation was interrupted, evaluation restarts from the completed cache.
Do not change source code, inputs or settings during caching: resume checks their
identity. The workflow and shared GPU locks prevent overlapping jobs.

Success ends with `FINISHED` and writes `test.json` (detailed results),
`summary.json` (video metrics and fixed decision settings), and `status.json`.
To display the summary:

```powershell
Get-Content '.\runs\heldout-001\fk\summary.json'
```

## Remaining comparator evaluation work

Keep selected checkpoints, quality normalization and thresholds fixed. Retain
unassessed clips in the evaluation denominator and report coverage. All methods,
including any later shared controls, must use this same population.

The comparator workflows still need compatibility changes before these new
rows can be scored with their existing checkpoints:

- AVFakeNet and AVTENet validate training-population identity when loading
  checkpoints; a new evaluation population currently fails those checks.
- The three-model comparison script currently selects validation rows.

Do not disable these checks or refit quality on held-out data. A follow-up change
must explicitly bind the new evaluation population to the original training
provenance, preserve model/preprocessing/asset checks, and add test comparison.
The FK command above is ready; no AVFakeNet/AVTENet held-out command is supplied
until that support is implemented.
