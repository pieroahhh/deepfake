# Start here: Machine C — complete AVTENet adaptation

Use the **Ryzen 5 5600X / RTX 4060 8 GB / 32 GB RAM** desktop. This machine runs
the entire AVTENet experiment independently while A runs FK and B runs AVFakeNet.
The three experts run sequentially on this single GPU; this is not distributed training.

The visual backbone is the approved **Google joint-attention ViViT adaptation**,
not the original factorized ViViT reproduction. Keep that description in reports.

## 1. Copy the C bundle and check SSD space

Copy the entire **Machine-C-bundle** folder to this machine's SSD, preserving:

```text
Machine-C-bundle/
  START_HERE_THIRD_MACHINE.md
  bundle.json
  avhubert-cuda117-v1.tar
  dataset/train/...
  dataset/val/...
  code/
    scripts/machine_c.ps1
    runs/two-day-001/population/...
    runs/two-day-001/avtenet/shared-index/...
    checkpoints/vivit-joint-adaptation-002/...
    checkpoints/comparators-ast-001/ast/...
    checkpoints/comparators-native-001/avhubert_native/...
```

The transfer is about **5.2 GiB**, including all three pretrained experts and the
CUDA image. No full AV-Deepfake1M download is needed. The frozen 3,500 clips,
manifest and index are identical to A/B. Do not resample or recreate them.

Plan for **200 GiB of free SSD space** before the run, including environments,
Docker storage, prepared visual/lip/audio caches, and resumable checkpoints. This
is a conservative allocation, not a measured final size; lossless cache compression
depends on the media. A 416 GB SSD can work if it actually has sufficient free
space. The cache stops before the drive becomes full, with a 5 GiB free-space
guard, but that guard does not reserve space for later checkpoints.

## 2. Prerequisites

Install an NVIDIA driver, Python **3.11**, **uv**, and **Docker Desktop with
Linux containers / WSL2 GPU support**. Start Docker Desktop and wait for its
engine. On this 32 GB machine, begin with a Docker/WSL memory allowance around
**12 GB** and leave memory for Windows. Close large applications and prevent
sleep. Internet is needed for dependencies and the first MTCNN download.

## 3. Set up the host environment

Open PowerShell in the copied `code` folder, adjusting the example path:

```powershell
Set-Location -LiteralPath 'D:\Machine-C-bundle\code'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_c.ps1 -Setup
& '.\.venv-gpu\Scripts\python.exe' scripts/verify_second_machine.py --bundle '..' --role C
& '.\.venv-gpu\Scripts\python.exe' scripts/check_worker_cuda.py
```

Expected: `Bundle verified: 3500 clips`, the RTX 4060 device name, and a passed
CUDA arithmetic check. Fix checksum errors by recopying the affected files.
Do not edit the inventory to bypass verification or use CPU training as a fallback.

## 4. Load the supplied CUDA container

```powershell
docker load --input '..\avhubert-cuda117-v1.tar'
```

If no image archive was supplied, build the pinned runtime instead:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_c.ps1 -BuildContainer
```

The tag must be `fk-avhubert:cuda117-v1`. AV-HuBERT uses isolated Linux Python
3.8 / Torch 1.13.1+cu117; AST/ViViT use host Python 3.11 / Torch 2.2.2+cu121.
The launcher checks actual CUDA execution on both runtimes before training.

## 5. Launch the complete model

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_c.ps1 -Run
```

Run one copy. It verifies the bundle, configures all machine-local model paths,
then executes:

| Order | Stage | Budget / result |
| --- | --- | --- |
| 1 | AST audio expert | 1 epoch; extract train/validation features |
| 2 | Google ViViT visual expert | 1 epoch; extract train/validation features |
| 3 | AV-HuBERT audiovisual expert | 1 epoch in Docker; extract train/validation features |
| 4 | Merge expert feature caches | Verify shared index, source hashes and source-time alignment |
| 5 | Frozen-expert feature fusion | 1 epoch |
| 6 | Validation evaluation | All 500 IDs, including unassessed cases and coverage |
| 7 | Result export | `return-c-to-machine-a` |

Seed is 7. ViViT uses SDPA attention and gradient checkpointing for the 8 GB
GPU. Preparation and feature extraction can be substantial work even with only
one training epoch. Completion within 48 hours is not guaranteed, and one epoch
does not establish convergence. No full-subset runtime has been measured on C yet.

## 6. Monitor and resume

In a second PowerShell window in `code`:

```powershell
Get-Content '.\runs\worker-c\avtenet\status.json'
Get-Content '.\runs\worker-c\avtenet\logs\audio-train.log' -Tail 20 -Wait
nvidia-smi
```

Follow the stage's current `log` path from status.json. Feature extraction may
have less frequent output than training. Ctrl+C in a log viewer only stops viewing.

If interrupted, preserve `runs/worker-c` and rerun `machine_c.ps1 -Run` after
fixing the cause. Completed experts/features are reused. Training resumes at the
last completed epoch in a new attempt; unfinished epoch work repeats. Keep
`best.pt` with `last.pt`. Do not change profiles, libraries, paths, code or media
mid-resume. An unfinished feature-cache/merge/evaluation directory must be kept
under another name before retrying that stage; rename only after the job stops.

For CUDA OOM, close other GPU applications and keep the failure log. Do not
start multiple experts or change model/profile settings silently. For disk-space
errors, preserve the cache and coordinate recovery before moving the experiment;
moving paths changes workflow identity. Monitor Docker's Windows-host storage too.

## 7. Send the completed result back

After successful export, copy the entire:

```text
Machine-C-bundle/return-c-to-machine-a/
```

It includes the final predictions/report, selected checkpoints for all three
experts and fusion, their run/metric provenance, and `return.json` with role C,
population identity and SHA-256 hashes. No expert features need to cross machines:
fusion and evaluation have already happened on C. Keep the feature/prepared
caches and optimizer resume checkpoints locally until the results are accepted.

If export reports an existing return folder, preserve that previous export under
another name before requesting another. A verifies both B and C returns before
the final three-model comparison; see `code/docs/three-machine-guide.md`.
