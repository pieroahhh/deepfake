# AV-Deepfake1M: after-download guide for Windows

**Current study choice:** use all clips in the official train/validation splits with `--source-overlap-policy keep-official`. Source-family and content overlap are retained and reported. After setup, use section 8 without clip caps. Section 7 is an optional pilot only. Use the fresh `full-official-001` directory; the earlier filtered pilot is a different population. Unlabeled test videos remain available for inference and are never assigned invented training labels.

[ฉบับภาษาไทย](avdeepfake1m-after-download-guide.th.md)

This guide takes you from downloaded archives to a trained FK detector, evaluation reports, and verifiable inference artifacts. It targets **AV-Deepfake1M (2024)** and the repository branch **`feature/avdeepfake1m-dataset`**. It does not target AV-Deepfake1M++.

Run the commands yourself, one section at a time. Optional paths are clearly marked; do not paste every code block into one script. All commands use **Windows PowerShell** from the repository root unless stated otherwise.

The code has software and synthetic runtime tests. Full-release data compatibility, full-scale training, GPU performance, and detection accuracy still need real-data verification. Start with the pilot before committing resources to the complete dataset.

## Contents

1. [Workflow and files](#1-workflow-and-files)
2. [Set paths and check storage](#2-set-paths-and-check-storage)
3. [Test and extract the archives](#3-test-and-extract-the-archives)
4. [Prepare the code and Python environment](#4-prepare-the-code-and-python-environment)
5. [Verify metadata and sample media](#5-verify-metadata-and-sample-media)
6. [Prepare pretrained assets](#6-prepare-pretrained-assets)
7. [Run the pilot](#7-run-the-pilot)
8. [Estimate resources and run the full study](#8-estimate-resources-and-run-the-full-study)
9. [Resume interrupted work](#9-resume-interrupted-work)
10. [Evaluate and interpret results](#10-evaluate-and-interpret-results)
11. [Run shared controls and ablations](#11-run-shared-controls-and-ablations)
12. [Run forensic inference and verify evidence](#12-run-forensic-inference-and-verify-evidence)
13. [Troubleshooting](#13-troubleshooting)
14. [Completion checklist](#14-completion-checklist)

## 1. Workflow and files

```text
Archive checks and extraction
  -> metadata/media checks + environment/assets checks
  -> prepare-avdeepfake1m: freeze the selected population
  -> fit-quality: learn normalization from training media only
  -> cache-fk: extract frozen visual/audio features
  -> train-fk --calibrate: train the detector and select on validation
  -> evaluate-fk: report predictions, metrics, and assessment coverage
  -> infer-fk + verify: create and verify source-bound evidence
```

| Item | Meaning |
| --- | --- |
| Dataset media | The original video files. Keep them unchanged. |
| Dataset metadata | Official split, source, manipulation, and temporal annotations. |
| Population manifest | This project's normalized list of selected videos, labels, families, durations, and source hashes. |
| Pretrained assets | EfficientNet-B0, Wav2Vec2 Base, Silero VAD, and YuNet. These do not constitute a trained FK detector. |
| Quality profile | Training-fitted normalization for measured visual/audio quality. |
| Feature cache | Frozen encoder outputs plus labels and provenance; reused when training heads. |
| Detector checkpoint | Learned FK head parameters, configuration, and, after calibration, selected decision parameters. |

The project preserves official splits. Source-family checks do not independently establish identity-disjointness. Do not randomly redistribute videos between train, validation, and test.

## 2. Set paths and check storage

### Start on any Windows machine

Install Git and open a PowerShell terminal **inside your local checkout** (in VS Code: open the repository folder, then open a new terminal). If you have already cloned/pulled the repository, skip the following clone block. For a new checkout, choose an existing parent directory; the target `Deepfake-Detection` directory must not already exist:

```powershell
$CloneParent = Read-Host 'Existing parent folder for the clone (full path, without quotes)'
if (-not (Test-Path -LiteralPath $CloneParent -PathType Container)) { throw 'Parent folder does not exist.' }
$CloneTarget = Join-Path (Resolve-Path -LiteralPath $CloneParent).Path 'Deepfake-Detection'
git clone --branch feature/avdeepfake1m-dataset --single-branch `
    https://github.com/tomatoesws/Deepfake-Detection.git $CloneTarget
if ($LASTEXITCODE -ne 0) { throw 'Clone failed.' }
Set-Location -LiteralPath $CloneTarget
```

The guide detects the repository root with Git, so it works with different usernames, drive letters, and checkout locations. It remains a Windows/PowerShell guide; its `.exe` paths and tool setup are not Linux/macOS instructions.

### What a pull supplies

| Supplied by Git | Prepare separately on each machine |
| --- | --- |
| Source code, configs, dependency lock, and these guides | Python `.venv`, FFmpeg/FFprobe, 7-Zip, dataset media/metadata, pretrained files, and trained checkpoints/cache/results |

Each machine needs its own Python environment and valid local asset paths. Do not copy `.venv` from another computer. Dataset access/download terms still apply to that machine/user. To continue an existing experiment, transfer its complete artifacts separately and follow the resume compatibility checks in section 9; pulling code alone cannot resume it.

### Set local paths once per PowerShell session

The next block asks for three **full local paths**, without surrounding quotes. For example, use `E:\Datasets\AV-Deepfake1M-download`, `E:\Datasets\AV-Deepfake1M`, and `E:\FK-runs` if E: is your storage drive. Spaces in paths are supported. Extraction/run directories can be new. If media is already extracted, supply its real directory and skip section 3. Keep the run directory outside the repository so generated artifacts do not appear as source changes.

```powershell
$ErrorActionPreference = 'Stop'
$Repo = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'Open PowerShell inside the cloned repository first.' }
$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $Repo
if (-not (Test-Path -LiteralPath 'pyproject.toml') -or
    -not (Test-Path -LiteralPath 'src/deepfake_detection/cli.py')) {
    throw 'This is not the Deepfake-Detection repository root.'
}

function Read-LocalPath([string]$Prompt) {
    $value = (Read-Host $Prompt).Trim().Trim('"')
    if ([string]::IsNullOrWhiteSpace($value)) { throw 'A path is required.' }
    if ($value -notmatch '^[A-Za-z]:[\\/]' -and
        $value -notmatch '^\\\\[^\\]+\\[^\\]+') {
        throw 'Use a full drive or UNC path.'
    }
    [System.IO.Path]::GetFullPath($value)
}
$Archives = Read-LocalPath 'Downloaded archive folder (full path)'
$Dataset = Read-LocalPath 'Extracted dataset folder (full path)'
$RunRoot = Read-LocalPath 'Experiment output folder outside the repository (full path)'
$Pilot = Join-Path $RunRoot 'pilot-001'
$Full = Join-Path $RunRoot 'full-official-001'
$SevenZipCommand = Get-Command 7z.exe -ErrorAction SilentlyContinue
$SevenZip = if ($SevenZipCommand) { $SevenZipCommand.Source } else {
    Join-Path $env:ProgramFiles '7-Zip\7z.exe'
}
$fkPython = Join-Path $Repo '.venv\Scripts\python.exe'
$Device = 'auto'

# PowerShell 5 does not automatically stop on native nonzero exit codes.
function Invoke-Python {
    $input | & $fkPython @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}
function Invoke-7Zip {
    & $SevenZip @args
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed or warned: exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root,
    @{Name='UsedGiB';Expression={[math]::Round($_.Used / 1GB, 1)}},
    @{Name='FreeGiB';Expression={[math]::Round($_.Free / 1GB, 1)}}
```

The archive download alone is roughly 404 GB for the previously inspected release. This is a planning estimate, not a current disk-space check or a guarantee of extracted size. You need room for **archives + extracted media + feature caches + run artifacts**. Use archive listings and pilot measurements for your machine. Keep additional headroom for temporary files and failed/restarted runs.

If space is tight, process one split at a time. Only remove its archive parts after successful extraction and validation, and only if you can download them again. This guide does not automatically delete archives. A complete labeled training workflow requires both train and validation media; blind test media can be deferred.

Keep large artifacts on your chosen storage drive. The Python environment remains in `.venv` under your local repository, wherever you cloned it. Direct future package/model download caches to your chosen run root for this shell:

```powershell
$env:UV_CACHE_DIR = Join-Path $RunRoot 'download-caches\uv'
$env:HF_HOME = Join-Path $RunRoot 'download-caches\huggingface'
$env:TORCH_HOME = Join-Path $RunRoot 'download-caches\torch'
```

These variables do not relocate files already downloaded elsewhere.

## 3. Test and extract the archives

Install [7-Zip from its official site](https://7-zip.org/) if `$SevenZip` does not exist. The [official dataset card](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M) specifies extracting the multipart archive with `7z`, starting from `.zip.001`.

Keep all parts for each split together. For the previously inspected snapshot, the sequences ended at `train.zip.254`, `val.zip.020`, and `test.zip.112`. Use the file listing for your downloaded revision as the source of truth; a file count alone does not prove completeness.

```powershell
if (-not (Test-Path -LiteralPath $SevenZip)) { throw 'Install 7-Zip first.' }
Get-ChildItem -LiteralPath (Join-Path $Archives 'train') -Filter 'train.zip.*' |
    Select-Object Name, Length

Invoke-7Zip t (Join-Path $Archives 'train\train.zip.001')
Invoke-7Zip t (Join-Path $Archives 'val\val.zip.001')
# Optional: needed only when you will use the blind test media.
Invoke-7Zip t (Join-Path $Archives 'test\test.zip.001')
```

Each test must finish without CRC, missing-volume, or unexpected-end errors. Tests may take a long time because they read the archive contents. Do not concatenate the parts manually or extract every numbered part separately.

### Choose the correct extraction destination

Inspect internal paths first:

```powershell
Invoke-7Zip l (Join-Path $Archives 'train\train.zip.001')
Invoke-7Zip l (Join-Path $Archives 'val\val.zip.001')
```

If member paths already start with `train/` or `val/`, extract into `$Dataset`:

```powershell
Invoke-7Zip x (Join-Path $Archives 'train\train.zip.001') "-o$Dataset"
Invoke-7Zip x (Join-Path $Archives 'val\val.zip.001') "-o$Dataset"
```

If member paths start directly with video IDs/subdirectories and have **no split prefix**, use this alternative instead:

```powershell
Invoke-7Zip x (Join-Path $Archives 'train\train.zip.001') "-o$(Join-Path $Dataset 'train')"
Invoke-7Zip x (Join-Path $Archives 'val\val.zip.001') "-o$(Join-Path $Dataset 'val')"
```

Apply the same listing-and-destination rule to `test.zip.001` when needed. Use only one extraction method per split. Avoid a result such as `train\train\...`. Do not silently overwrite a previous extraction after an error; inspect the failing paths first.

Copy the plain official metadata JSON files from the download root:

```powershell
foreach ($name in @('train_metadata.json', 'val_metadata.json', 'test_files.txt')) {
    $source = Join-Path $Archives $name
    $target = Join-Path $Dataset $name
    if (Test-Path -LiteralPath $source) {
        if (Test-Path -LiteralPath $target) {
            if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash) {
                throw "Existing metadata differs: $target"
            }
        } else {
            Copy-Item -LiteralPath $source -Destination $target
        }
    }
}
```

If you only downloaded compressed metadata, list and extract those archives separately to obtain the actual JSON files. Do not rename a ZIP to `.json` or assume an archive's contents from its filename.

Required layout:

```text
<your $Dataset folder>\
  train_metadata.json
  val_metadata.json
  train\
    <paths corresponding to train metadata>
  val\
    <paths corresponding to val metadata>
  test_files.txt                  # optional, unlabeled inventory
  test\                          # optional for train/validation work
    <paths corresponding to test_files.txt>
```

## 4. Prepare the code and Python environment

### Existing checkout

```powershell
Set-Location -LiteralPath $Repo
git status --short
if ($LASTEXITCODE -ne 0) { throw 'Git status failed.' }
git branch --show-current
if ($LASTEXITCODE -ne 0) { throw 'Cannot read the current branch.' }
```

The expected branch is `feature/avdeepfake1m-dataset`. Before changing branches or pulling, commit your intentional changes or stash them. Preserve existing stashes. If the working tree is clean:

```powershell
git fetch origin
if ($LASTEXITCODE -ne 0) { throw 'Fetch failed.' }
git switch feature/avdeepfake1m-dataset
if ($LASTEXITCODE -ne 0) { throw 'Branch switch failed.' }
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull failed; inspect the divergence/error.' }
```

### Environment on a new machine

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) before the tool check below. Clone first using section 2 if needed. All commands below assume `$Repo` was detected there. Create `.venv` on this machine even if you pulled an existing experiment's code:

```powershell
Set-Location -LiteralPath $Repo
Get-Command git, uv -ErrorAction Stop
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not available. To recreate the checked CPU environment, run the following only when creating or intentionally restoring an environment. Do not reinstall the CPU lock over a GPU environment you intend to keep.

```powershell
if (-not (Test-Path -LiteralPath $fkPython)) {
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
uv pip install --python $fkPython -r requirements-paper-alignment.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
```

The lock includes an editable installation of this repository. Keep the working directory at `$Repo`. Activation is unnecessary because the guide calls the interpreter explicitly.

### Device and import checks

```powershell
@'
import sys, torch, torchvision, transformers, av, cv2, soundfile, safetensors
print(sys.executable)
print(sys.version)
print('torch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
'@ | Invoke-Python -
Invoke-Python -m deepfake_detection.cli --help
```

The previously checked environment uses CPU PyTorch. `--device auto` selects CUDA only if `torch.cuda.is_available()` is true; otherwise it uses CPU. To require a GPU and fail instead of falling back, set `$Device = 'cuda'` after confirming CUDA works.

For an NVIDIA GPU environment, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for Windows and a compatible CUDA build. Apply its package choices to **this interpreter**, using `uv pip install --python $fkPython ...` with the selector's packages/index. Keep torch and torchvision compatible. A separately installed CUDA toolkit alone does not turn a CPU torch wheel into a CUDA build. Re-run import/runtime tests after changing packages, and save a separate environment snapshot instead of claiming it matches the CPU lock.

### FFmpeg and FFprobe

If `.venv\Scripts\ffmpeg.exe` and `ffprobe.exe` already exist, verify them below. Otherwise, the previously checked setup can be recreated with an isolated tools installation:

```powershell
uv pip install --python $fkPython --target .cache/ffmpeg-tools 'static-ffmpeg==3.0'
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg helper installation failed.' }
@'
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.cache/ffmpeg-tools').resolve()))
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise
ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
destination = Path(sys.executable).parent
for source, name in [(ffmpeg, 'ffmpeg.exe'), (ffprobe, 'ffprobe.exe')]:
    target = destination / name
    if target.exists():
        raise FileExistsError(f'Inspect existing executable before replacing it: {target}')
    shutil.copy2(source, target)
    print(target)
'@ | Invoke-Python -
```

This helper downloads executables. If one executable already exists, resolve that partial installation before running the copy block. Alternatively, use an existing trusted FFmpeg installation on PATH; the project's resolver supports both locations.

```powershell
@'
import subprocess
from deepfake_detection.media import media_tool
for name in ('ffmpeg', 'ffprobe'):
    subprocess.run([media_tool(name), '-version'], check=True)
'@ | Invoke-Python -
Invoke-Python -m unittest discover -s tests -v
```

The recorded implementation handoff reports 97 passing tests. Use your current test output as the result for your checkout. Passing tests establish software behavior, not real-data accuracy.

Record the actual environment before processing data:

```powershell
$Setup = Join-Path $RunRoot ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $Setup | Out-Null
git rev-parse HEAD | Set-Content -LiteralPath (Join-Path $Setup 'git-commit.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cannot record code revision.' }
git status --short | Set-Content -LiteralPath (Join-Path $Setup 'git-status.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cannot record working-tree state.' }
uv pip freeze --python $fkPython | Set-Content -LiteralPath (Join-Path $Setup 'requirements-actual.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cannot record installed packages.' }
```

## 5. Verify metadata and sample media

Run this small check before the more expensive population preparation. It validates the metadata type and split field, resolves sample paths using the project's adapter, and opens three media files per split. It does not validate every video or every annotation.

```powershell
@'
import json
import sys
from pathlib import Path
from deepfake_detection.avdeepfake1m import media_path, media_duration
root = Path(sys.argv[1]).resolve(strict=True)
for split in ('train', 'val'):
    path = root / f'{split}_metadata.json'
    records = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(records, list) and records, f'Empty/invalid metadata: {path}'
    assert all(r.get('split') == split for r in records), f'Wrong split: {path}'
    print(split, 'metadata rows:', len(records))
    for row in records[:3]:
        source = (root / media_path(row['file'], split)).resolve(strict=True)
        source.relative_to(root)
        print(source, 'duration_seconds:', media_duration(source))
'@ | Invoke-Python - $Dataset
```

If paths fail, compare the `file` field with the extracted directory tree. The adapter supports paths with or without the official split prefix. Do not rewrite annotation paths or durations to hide an extraction problem.

## 6. Prepare pretrained assets

### Reuse verified local assets

If this machine already has verified assets, set `$Assets` to their local registry. The default below matches the fresh-download directory used by this guide; `False` means you should use the download step next:

```powershell
$Assets = Join-Path $RunRoot 'pretrained-001\assets.local.json'
Test-Path -LiteralPath $Assets
```

Git does not include these model files. The registry contains absolute paths, so copying only its JSON to another machine is insufficient. The original development registry, if supplied separately, is not an automatic dependency of this guide. Prefer downloading the four assets locally below. To reuse transferred assets, copy their complete directories and integrity manifests, then update the registry's artifact `directory`/`manifest`, `silero.path`, and `detector.path` to real paths on this machine. Preserve file hashes/revisions and verify them with the checks below. Do not share local registry paths as if they were portable configuration.

### Download assets only if needed

Choose a new directory; the downloader will not overwrite an existing acquisition:

```powershell
$AssetDir = Join-Path $RunRoot 'pretrained-001'
Invoke-Python scripts/acquire_pretrained.py --output $AssetDir `
    --assets efficientnet_b0 wav2vec2_base silero yunet
$Assets = Join-Path $AssetDir 'assets.local.json'
```

Expected outputs include `assets.local.json`, `acquisition.json`, and four asset directories with downloaded files and integrity manifests. All four requested entries in `acquisition.json` must be `ACQUIRED_AND_CHECKED`. A failed acquisition retains diagnostic files; retry in a new directory after resolving the cause.

### Verify checksums and runtime

```powershell
Invoke-Python -m deepfake_detection.cli infrastructure --assets $Assets `
    --output (Join-Path $Setup 'infrastructure.json')
Invoke-Python scripts/check_fk_runtime.py --assets $Assets `
    --output (Join-Path $Setup 'runtime.json')
```

The infrastructure report also describes optional backbones for other methods. Missing AST, factorized ViViT, or AV-HuBERT does not block this four-asset FK path. Its top-level status is conservative and does not automatically become READY. Inspect `efficientnet_b0`, `wav2vec2_base`, `silero`, and `detector`, then require the runtime report to say `PASSED`.

The runtime script uses real pretrained assets with synthetic inputs and an untrained FK head. It runs its tensors on CPU even if it reports CUDA availability; it is not a GPU benchmark or an accuracy test. The pilot checks actual data processing on your selected device.

## 7. Run the pilot

Use this pilot to expose path, annotation, quality, memory, and calibration problems before the full run. The 15,000/2,500 caps are bounded development populations, not a claim that they run quickly on CPU. Smaller caps are possible, but must fit complete source families and provide both classes.

Do not pre-create `population`, `cache-v2`, or `fk-smoke` below. Their commands create new output directories. If `pilot-001` contains an earlier attempt, inspect it and use `pilot-002` rather than overwriting artifacts.

### 7.1 Freeze the pilot population

Annotation bounds: the release includes timestamps slightly or substantially beyond actual media duration. The adapter intersects them with the observed interval, retains exact source intervals and overrun in `annotation_bounds_adjustments`, and reports adjusted-clip counts in the audit. No video is dropped. If no annotated fake time remains inside the clip, the video label stays unchanged and temporal supervision becomes incomplete/unknown. Preparation prints progress every 1,000 clips (with percentage for uncapped populations); it still has no resume option.

The real release contains repeated slashes in original IDs and references to real media that was not distributed. The adapter resolves these source IDs without changing downloaded metadata. The agreed `keep-official` policy retains all rows in their original splits, including overlapping families/content. `source-overlaps.json` and `audit.json` report overlaps; `source-overlap-exclusions.json` is empty. The policy is preserved in manifests, caches, checkpoints, resume checks, and evaluation reports. Conflicting modality annotations remain masked as unknown and retained for audit. The older `drop-train` mode is still available explicitly but is not the current study choice.

```powershell
Invoke-Python -m deepfake_detection.cli prepare-avdeepfake1m `
    --dataset-root $Dataset --output (Join-Path $Pilot 'population') `
    --train-clips 15000 --validation-clips 2500 --seed 7 --source-overlap-policy keep-official
```

This reads official metadata, resolves real/fake source families, checks labels and intervals, obtains durations from media timestamps, and hashes selected files. Preparation still loads full train/validation metadata even with caps. It selects a deterministic prefix of whole families; counts can be below the caps, and too-small caps can fail.

Outputs:

- `population\manifest.json`: selected labeled train/validation videos.
- `population\audit.json`: metadata hashes, population audit, selection settings, and test-label status.
- `population\test-inference-files.json`: created only when an unlabeled `test_files.txt` is present. These files are not added to supervised training.

Inspect counts and the audit:

```powershell
Get-Content -LiteralPath (Join-Path $Pilot 'population\audit.json')
@'
import json, sys
from collections import Counter
rows = json.load(open(sys.argv[1], encoding='utf-8'))
print('split/label counts:', Counter((r['split'], r['label']) for r in rows))
print('source families:', len({r['source_family'] for r in rows}))
print('total source hours:', sum(r['duration'] for r in rows) / 3600)
'@ | Invoke-Python - (Join-Path $Pilot 'population\manifest.json')
```

Check that train and validation are nonempty and contain real (`0`) and fake (`1`) videos. Unknown modality-specific intervals remain unknown; do not manufacture them. Preparation failures must be resolved before fitting quality.

### 7.2 Fit measured quality on training media

```powershell
Invoke-Python -m deepfake_detection.cli fit-quality `
    --manifest (Join-Path $Pilot 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --output (Join-Path $Pilot 'quality.json')
```

Expected: `quality.json`, with schema `measured-quality-v2` and `fit_split` equal to `train`. The profile binds the selected training population and assets. It uses training-only blur/SNR percentile bounds; unavailable or constant measurements map to a neutral value. It does not fit on validation or test.

This stage decodes media and measures face/audio quality. It prints clips/windows completed, speed, elapsed time and ETA, and saves each completed window to `<output>.work/state.sqlite`. Missing SNR is allowed when speech/non-speech evidence is unavailable. There is no `--device` option. See section 9 for `--resume`.

### 7.3 Cache frozen encoder features

```powershell
$CacheTimer = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Python -m deepfake_detection.cli cache-fk `
    --manifest (Join-Path $Pilot 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --quality-normalization (Join-Path $Pilot 'quality.json') `
    --output (Join-Path $Pilot 'cache-v2') --device $Device
$CacheTimer.Stop()
$CacheTimer.Elapsed.ToString() | Set-Content -LiteralPath (Join-Path $Pilot 'cache-elapsed.txt')
```

Expected: `cache-run.json`, `caches\...`, and a finalized `training-manifest.json`. The latter uses schema `fk-cache-manifest-v2` and retains assessments for unusable clips/windows. A clip with no usable windows is not silently converted into a real prediction.

Inspect usable class counts and assessment coverage before training:

```powershell
@'
import json, sys
from collections import Counter
m = json.load(open(sys.argv[1], encoding='utf-8'))
assert m['schema'] == 'fk-cache-manifest-v2'
for split in ('train', 'validation', 'test'):
    print(split, 'usable videos:', len(m[split]), 'labels:', Counter(r['label'] for r in m[split]))
    assessments = [a for a in m['assessments'] if a['split'] == split]
    print('all assessed-for-usability videos:', len(assessments))
    print('window statuses:', Counter(w['status'] for a in assessments for w in a['windows']))
    if split in ('train', 'validation'):
        assert {r['label'] for r in m[split]} == {0, 1}, f'Both usable classes required: {split}'
print('usable validation window labels:', Counter(
    w['labels']['combined'] for a in m['assessments'] if a['split'] == 'validation'
    for w in a['windows'] if w['status'] == 'USABLE'))
'@ | Invoke-Python - (Join-Path $Pilot 'cache-v2\training-manifest.json')
```

Calibration requires both classes among known, usable validation video labels and window labels. Investigate unexpectedly high unusable coverage before increasing the dataset size.

### 7.4 Train for two epochs and calibrate

```powershell
Invoke-Python -m deepfake_detection.cli train-fk `
    --manifest (Join-Path $Pilot 'cache-v2\training-manifest.json') `
    --model-config configs/ours.provisional.json `
    --training-config configs/fk.training-v1.json `
    --output (Join-Path $Pilot 'fk-smoke') --epochs 2 --calibrate --device $Device
```

This is a **pipeline check**, not the final experiment. The frozen encoders were used during caching; this stage learns the detector head using complete videos and all their usable windows. The code does not expose a generic batch-size, mixed-precision, or distributed-training CLI switch.

| Output | Purpose |
| --- | --- |
| `run.json` | Configuration, device, population signatures, and provenance recorded at startup. Its `STARTED` value alone does not indicate final success/failure. |
| `last.pt` | Last completed epoch, optimizer state, RNG state, and history for resuming. |
| `best.pt` | Best validation-loss model; finalized as an inference checkpoint after training. |
| `metrics.json` | Completed training summary and per-epoch history. |
| `timing.json` | Training and validation time per epoch. |
| `selection.json` | Validation calibration candidates and selected decision settings. |
| `selected.pt` | Best model weights with validation-selected aggregation and thresholds. Use this for calibrated evaluation/inference. |

Calibration runs after training. It can take substantial additional time and memory. Training loss finishing successfully does not guarantee calibration completed: check for both `selected.pt` and `selection.json`.

### 7.5 Evaluate the pilot validation split

```powershell
Invoke-Python -m deepfake_detection.cli evaluate-fk `
    --manifest (Join-Path $Pilot 'cache-v2\training-manifest.json') `
    --checkpoint (Join-Path $Pilot 'fk-smoke\selected.pt') `
    --split validation --device $Device `
    --output (Join-Path $Pilot 'validation.json')
```

Success means the report is generated, the split is correct, metrics/coverage are understandable, and no provenance checks fail. Low accuracy can still be a valid result. Do not tune expectations by deleting difficult videos. This is selected-validation performance, not held-out test performance.

## 8. Estimate resources and run the full study

### Measure before scaling

```powershell
$PilotCacheBytes = (Get-ChildItem -LiteralPath (Join-Path $Pilot 'cache-v2') -Recurse -File |
    Measure-Object -Property Length -Sum).Sum
'Pilot cache GiB: {0:N2}' -f ($PilotCacheBytes / 1GB)
Get-Content -LiteralPath (Join-Path $Pilot 'cache-elapsed.txt')
Get-Content -LiteralPath (Join-Path $Pilot 'fk-smoke\timing.json')
Get-PSDrive -PSProvider FileSystem
```

Estimate storage and time using source duration or usable-window counts, not only video counts:

```text
estimated full cache size ~= pilot cache size * (full windows / pilot windows)
estimated full cache time ~= pilot cache time * (full source duration / pilot source duration)
estimated epoch time ~= pilot epoch time * (full usable windows / pilot usable windows)
```

These are rough extrapolations. Decode failures, face detection, video lengths, disk throughput, GPU memory, and calibration costs can change the ratio. Add practical headroom and measure again after a larger pilot. No full-dataset runtime, RAM, VRAM, or disk requirement has been validated for this implementation. Quality fitting and validation calibration retain collections in RAM, and the calibration search may become a separate bottleneck.

### Prepare all labeled train/validation data

Use a fresh full-run population. Omit clip caps:

```powershell
Invoke-Python -m deepfake_detection.cli prepare-avdeepfake1m `
    --dataset-root $Dataset --output (Join-Path $Full 'population') --seed 7 --source-overlap-policy keep-official
```

Repeat the population audit checks from the pilot using `$Full`. Preparation reads/hashes every selected media file; a long run is expected. Freeze the code, assets, configuration, and environment before caching.

### Fit a full-training profile, cache, and train

```powershell
Invoke-Python -m deepfake_detection.cli fit-quality `
    --manifest (Join-Path $Full 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets --output (Join-Path $Full 'quality.json')

Invoke-Python -m deepfake_detection.cli cache-fk `
    --manifest (Join-Path $Full 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --quality-normalization (Join-Path $Full 'quality.json') `
    --output (Join-Path $Full 'cache-v2') --device $Device

Invoke-Python -m deepfake_detection.cli train-fk `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --model-config configs/ours.provisional.json `
    --training-config configs/fk.training-v1.json `
    --output (Join-Path $Full 'fk-main') --calibrate --device $Device

Invoke-Python -m deepfake_detection.cli evaluate-fk `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --checkpoint (Join-Path $Full 'fk-main\selected.pt') `
    --split validation --device $Device --output (Join-Path $Full 'validation.json')
```

Run each command separately and inspect its results before the next. Apply the pilot cache coverage/class checks to the full cache as well.

The checked training configuration uses AdamW, learning rate `0.0001`, weight decay `0.0001`, maximum **50 epochs**, early-stopping patience **5**, and seed **7**. Best weights are selected by mean validation video loss; subsequent calibration selects aggregation/threshold settings on validation. Record any changed settings before final test evaluation.

Do not reuse the pilot quality profile or its cached features as the full-population profile/cache. The training-population fingerprint differs. Do not report LAV-DF-trained or pilot checkpoints as fully trained AV-Deepfake1M results.

## 9. Resume interrupted work

### Quality fitting: same JSON output path

```powershell
Invoke-Python -m deepfake_detection.cli fit-quality `
    --manifest (Join-Path $Full 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --output (Join-Path $Full 'quality.json') --resume --progress-seconds 10
```

Omit `--resume` for the first run. Keep the entire `quality.json.work` directory;
each completed window and each clip's VAD are committed there. An interruption
repeats at most the current uncommitted window. Progress updates at processing
boundaries, normally every 10 seconds; hashing/VAD/decoding can delay an update.
ETA becomes meaningful after measuring several windows.

Resume verifies manifest, assets, runtime, relevant implementation files and
saved measurement hashes. It also checks source hashes as clips are visited.
Keep those inputs unchanged; use a new output when intentionally changing them.
Percentiles remain exact over all usable training windows. Measurements stay on
disk; the work directory can grow substantially. Only one writer is allowed.

A process started before resume support has no saved state. Stop it with Ctrl+C
and start the updated command once without `--resume`; its earlier in-memory
progress cannot be imported. Do not repeat extraction or population preparation.
The final JSON is written after fitting and can be verified with `--resume`.

After reopening PowerShell, restore the variables/functions in section 2, set the working directory to `$Repo`, and restore `$Assets` to the registry you actually used. Keep the same code and environment for continuation.

### Feature caching: same output directory

```powershell
Invoke-Python -m deepfake_detection.cli cache-fk `
    --manifest (Join-Path $Full 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --quality-normalization (Join-Path $Full 'quality.json') `
    --output (Join-Path $Full 'cache-v2') --resume --device $Device
```

The command verifies completed clips and resumes incomplete work. Source manifest, assets, profile, code provenance, torch version, and encoder device must match the original run. Changing `auto` from a CPU environment to a CUDA environment can invalidate continuation. Do not edit provenance JSON to bypass the checks. Use a fresh cache for an intentionally changed pipeline.

### Training: last checkpoint, new output directory

```powershell
Invoke-Python -m deepfake_detection.cli train-fk `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --model-config configs/ours.provisional.json `
    --training-config configs/fk.training-v1.json `
    --resume (Join-Path $Full 'fk-main\last.pt') `
    --output (Join-Path $Full 'fk-main-resumed-001') --calibrate --device $Device
```

Resume uses the last completed epoch; an interrupted partial epoch is repeated. The epoch limit is the **total target epoch number**, not a number of additional epochs. Model/configuration/population compatibility is checked. Keep the original configuration and device for a reproducible continuation. Use the resumed directory's `selected.pt` for subsequent evaluation.

Do not resume from `best.pt` or `selected.pt`. A checkpoint that already reached the requested budget or early stopping is rejected. The CLI currently has no standalone calibration-only command: if training finished but calibration failed, retain the checkpoint and error log; do not rename `best.pt` to `selected.pt` or force resume past an early stop. Calibration-only recovery would require a separate code change, or a fresh training/calibration run after fixing the cause.

Dataset population preparation has no resume switch; restart it into a fresh path if needed. Quality fitting supports the per-window resume above. Preserve failed outputs and logs until you understand the failure. The multi-experiment runner has no whole-plan resume switch; retain completed runs instead of blindly rerunning into the same directory.

## 10. Evaluate and interpret results

### Validation diagnostics

The commands above evaluate validation, which also selected the checkpoint and decision settings. Label these results **validation diagnostics**. They are useful for development but are not an unbiased held-out test estimate.

Inspect `video_metrics`, `window_metrics`, `window_coverage_by_label`, `timeline_coverage`, and individual `videos`/`windows`. Classification metrics are conditional on assessed cases; coverage must be reported alongside them. ROC AUC, average precision, and other metrics can be `null` when the population does not support their calculation. An unassessed clip is not a negative prediction.

Window labels use positive-duration overlap with annotated manipulation intervals. Timeline metrics are supplemental and combine overlapping predictions. The evaluator's classification average precision and timeline outputs are not the official temporal-localization AP/AR benchmark.

### Actual labeled test metadata: optional alternative population

Only use this route if you possess genuinely labeled official test metadata compatible with the adapter. `test_files.txt` cannot be substituted. Add test metadata when preparing a new study population, before fitting/caching/training:

```powershell
$Labeled = Join-Path $RunRoot 'full-labeled-test-001'
$TestMetadata = Join-Path $Dataset 'test_metadata.json'
if (-not (Test-Path -LiteralPath $TestMetadata)) { throw 'Actual labeled test metadata is required.' }
Invoke-Python -m deepfake_detection.cli prepare-avdeepfake1m `
    --dataset-root $Dataset --output (Join-Path $Labeled 'population') `
    --test-metadata $TestMetadata --seed 7 --source-overlap-policy keep-official
```

For this alternative, set `$Full = $Labeled` and run the quality/cache/train steps in section 8 using the new population. The fitter still uses training only, and training/calibration still use train/validation only. Do not mix this manifest with an old run's profile or manually append test records to an existing cache manifest.

After freezing the protocol and obtaining `selected.pt`, score the held-out test once for final reporting:

```powershell
Invoke-Python -m deepfake_detection.cli evaluate-fk `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --checkpoint (Join-Path $Full 'fk-main\selected.pt') `
    --split test --device $Device --output (Join-Path $Full 'test.json')
```

Do not use test results to fit quality bounds, select thresholds, choose epochs, or tune architecture. Correcting an evaluation bug should be documented distinctly from tuning on test outcomes.

### Unlabeled official test media

`test_files.txt` supplies paths, not labels. The adapter writes a separate inference inventory. Train and validate normally, then run inference on selected test files using section 12. Local accuracy/F1/AUC cannot be calculated without ground truth. There is no implemented bulk official-submission exporter in this path; follow the [official dataset/challenge documentation](https://github.com/ControlNet/AV-Deepfake1M) for an applicable evaluation route. Do not substitute validation labels or fabricate all-real labels.

## 11. Run shared controls and ablations

Do this after the single FK run works and the study budget is agreed:

```powershell
Invoke-Python -m deepfake_detection.cli prepare-experiments `
    --model-config configs/ours.provisional.json --output (Join-Path $Full 'experiment-plan')

Get-Content -LiteralPath (Join-Path $Full 'experiment-plan\plan.json')

Invoke-Python -m deepfake_detection.cli run-experiments `
    --plan (Join-Path $Full 'experiment-plan\plan.json') `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --training-config configs/fk.training-v1.json `
    --output (Join-Path $Full 'experiments') --device $Device
```

The generated plan uses seeds 7, 17, and 27. It contains four shared controls (visual-only, audio-only, concatenation, late fusion), eight FK variants, and separate external-method entries. The shared runner trains/calibrates **36 runs** and skips the six external AVFakeNet/AVTENet entries, which require their own inputs/runners. It runs sequentially and does not score test data automatically. `completed.json` records finished runs and skipped methods when the plan completes.

For a runner smoke check, use `--epochs 2` and a separate output directory with the pilot cache. Do not label smoke results as final multi-seed results.

Inspect each run's selected decision parameters before interpreting an ablation. The current calibration search can change pooling parameters initialized by the `window_max`, `window_mean`, `video_max`, and `video_mean` profiles. Do not claim those runs retained fixed max/mean pooling without checking their `selection.json`; enforcing fixed pooling through calibration requires implementation/protocol changes.

Evaluate selected checkpoints individually with `evaluate-fk`, keeping split, population, and reporting rules consistent. Shared-baseline checkpoints are accepted by evaluation but not by FK's evidence-decomposition inference command.

## 12. Run forensic inference and verify evidence

Use a real input file and a trained **FK** `selected.pt`. The example measures duration from the source rather than copying an arbitrary three-second example.

```powershell
$InputVideo = Read-LocalPath 'Input video to analyze (full path)'
if (-not (Test-Path -LiteralPath $InputVideo -PathType Leaf)) { throw 'Input video does not exist.' }
$Checkpoint = Join-Path $Full 'fk-main\selected.pt'
$Evidence = Join-Path $Full 'evidence-example-001'
$Duration = @'
import sys
from deepfake_detection.avdeepfake1m import media_duration
print(media_duration(sys.argv[1]))
'@ | Invoke-Python - $InputVideo

Invoke-Python -m deepfake_detection.cli infer-fk `
    --input $InputVideo --duration $Duration --checkpoint $Checkpoint `
    --assets $Assets --quality-profile measured-quality-v2 --output $Evidence

Invoke-Python -m deepfake_detection.cli verify `
    --record (Join-Path $Evidence 'record.json') `
    --trusted-commitment (Join-Path $Evidence 'trusted-commitment.txt') `
    --source $InputVideo --artifact-root $Evidence
```

For a pilot demonstration, set `$Checkpoint` to `$Pilot\fk-smoke\selected.pt` and clearly label it a pilot detector. Use the same asset versions as training. Do not manually substitute a legacy quality profile. `infer-fk` currently has no device flag; do not append `--device` to it.

The output includes the record and commitment, plus per-usable-window prepared NPZ data, PCM16 audio, preparation metadata, and a representative PNG. The NPZ contains prepared pixels/audio/timestamps, not cached neural embeddings. Source metadata is collected through PyAV/libavformat in this path. Optional caption/transcript and perturbation analyses remain `NOT_RUN`.

The verifier must check the source, record, and derived artifacts. Without `--artifact-root`, artifact verification is `NOT_REQUESTED`. Read the returned verification result; the mere existence of JSON or a zero exit code is not a substitute for checking the report. A clip with no usable windows may produce no per-window artifacts and an unassessed result.

Retain a copy of the trusted commitment in a separate, trusted location when the record is first produced. If both a record and its only commitment are replaced together, comparing them alone cannot establish integrity. Successful hash verification establishes consistency with retained evidence, not scientific correctness of the model prediction.

## 13. Troubleshooting

| Symptom | What to check or do |
| --- | --- |
| Missing volume, CRC error, or unexpected end of archive | Verify all parts of that split are present; retry the affected download, then rerun `7z t`. |
| Metadata exists but video paths fail | Inspect internal archive paths and extraction depth; check for `train\train`. Use the original official metadata. |
| Annotation exceeds duration or labels disagree | Check release/schema and media integrity. Preserve the failing entry for investigation; inspect bounds adjustments; do not edit downloaded annotations or relabel clips to pass. |
| No complete family fits a cap | Increase the cap and prepare into a fresh directory. Selection never splits a family just to hit a number. |
| Both classes required for calibration | Inspect usable validation video/window labels and exclusion reasons. Enlarge/reprepare the pilot when necessary and rebuild its profile/cache. |
| Most clips/windows unusable | Check sample decoding, face detections, crop policy, and speech intervals. Report exclusions; do not silently relax gates for the final run. |
| Missing model file or checksum mismatch | Check absolute registry paths and artifact manifests. Reacquire or restore the original assets instead of changing hashes. |
| Infrastructure top-level status is not READY | Inspect the four FK entries and run `check_fk_runtime.py`; optional-method assets are reported separately. |
| CUDA requested but unavailable | Check the interpreter, torch wheel, compatible driver, and `torch.cuda.is_available()`. Use CPU explicitly if intended. |
| GPU out of memory | Close competing jobs; inspect the longest video's usable-window count. Whole-video training can require significant memory. CPU can help diagnosis but does not prove full-scale feasibility. There is no batch-size switch to reduce here. |
| RAM grows during fitting/calibration | Reduce pilot scope to diagnose. Full-scale optimization/sharding is not implemented by these instructions. Preserve logs and measure peak usage before another full attempt. |
| Disk full | Recheck your selected storage volume, archive/extraction overlap, and cache estimates. Keep source data and completed artifacts recoverable; do not broadly delete run folders. |
| Output already exists | Use a new output name, except for supported cache `--resume`. Most artifacts deliberately refuse overwrite. |
| Quality-profile/population mismatch | Use the profile fitted from that exact training manifest and assets. Rebuild after intentional population changes. |
| Cache resume provenance mismatch | Restore matching code/environment/device or start a new cache. Do not patch manifest hashes. |
| Training resume refused | Use `last.pt`, matching configuration/population, a fresh output directory, and an unfinished total epoch budget. |
| `selected.pt` absent after training | Calibration may have failed. Inspect logs and both-class checks; there is no standalone calibration-recovery CLI yet. |
| Test split empty or no labels | The normal public test inventory is unlabeled. Use validation diagnostics or obtain actual labeled test metadata; do not invent labels. |
| GitHub Desktop reports a lock file | Close active Git operations and inspect the exact lock path. Remove a stale lock only after confirming no Git process owns it; do not delete all lock files or `.git`. |

When requesting help, retain the exact command, complete error/traceback, code commit, device/package versions, and relevant audit/run/selection JSON. Avoid sending large private media or credentials in logs.

## 14. Completion checklist

- [ ] Archive tests passed and extraction paths match official metadata.
- [ ] Train/validation media checks passed; test labels were not fabricated.
- [ ] Code commit, working-tree state, package versions, device, and configuration were recorded.
- [ ] Four FK assets passed integrity/runtime checks; their manifests and registry were retained.
- [ ] Pilot preparation, quality fitting, caching, two-epoch training/calibration, and validation evaluation completed.
- [ ] Full-run storage/time/RAM feasibility was assessed from measurements.
- [ ] Full population audit, source hashes, metadata hashes, and normalized manifest were retained.
- [ ] Full quality profile was fitted only on the full training population.
- [ ] Cache manifest, feature files, assessment denominators, and exclusion reasons were retained together.
- [ ] Training configuration, `last.pt`, `best.pt`, `selected.pt`, metrics, timing, and selection report were retained.
- [ ] Validation diagnostics and held-out test results were clearly distinguished.
- [ ] Baseline/ablation claims match the executed runs and selected decision parameters.
- [ ] Forensic records and artifacts were verified, with trusted commitments retained independently.
- [ ] Experimental results describe the actual dataset population, coverage, seeds, and limitations.

Keep datasets and generated artifacts out of Git. Share code/configuration/documentation through Git and transfer large assets separately with hashes. Back up the complete run directories: manifests alone are not enough to reconstruct missing cached tensors or source media.

For method details, read the [paper alignment protocol](avdeepfake1m-paper-alignment.md). For previously verified implementation status and limitations, read [HANDOFF.md](handoffs/HANDOFF.md).
