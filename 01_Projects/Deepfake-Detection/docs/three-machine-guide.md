# Three-machine experiment handoff

This supersedes the earlier two-machine work split. Each machine owns one model;
all use the same frozen **3,000 train / 500 validation** clips, seed 7, official
split, keep-official policy and population SHA-256
`3c0955b7daaeed0d22b0efe68d06ee78146a8f43c88a5c3df28e1db3e13f5fea`.
The shared AVTENet index is copied byte-for-byte. Different preprocessing and
assessment coverage remain explicit; validation diagnostics are not held-out test results.

| Machine | Hardware | Assignment |
| --- | --- | --- |
| A | Ryzen 5600, RTX 3070 8 GB, RAM 32 GB | FK quality → cache → 5 training epochs → calibration → evaluation; final comparison |
| B | i7-13700HX, RTX 4060 8 GB, RAM 16 GB | AVFakeNet visual/audio, 3 epochs each, batch 8; evaluation |
| C | Ryzen 5600X, RTX 4060 8 GB, RAM 32 GB | Complete AVTENet adaptation: AST, ViViT, AV-HuBERT, fusion, 1 epoch per stage; evaluation |

The machines work in parallel, with one CUDA task at a time on each GPU. C's
32 GB RAM helps the container/preparation workflow; the existing SDPA and
gradient checkpointing path supports the selected 8 GB configuration. This does
not establish throughput or guarantee a two-day finish. AVTENet remains the
explicit Google joint-attention ViViT adaptation, not exact paper reproduction.

## Transfer and launch

Send **Machine-B-bundle** to B and **Machine-C-bundle** to C. Do not use the old
Second-machine-bundle launcher, which assigned additional AV-HuBERT work to B.
GitHub supplies source/docs; these bundles supply media, manifest/audit/index,
and C's weights/container image. Recreate environments rather than copying venvs.

Follow [B's English checklist](START_HERE_SECOND_MACHINE.md) or
[C's English checklist](START_HERE_THIRD_MACHINE.md). B needs about 80 GiB free;
reserve about 200 GiB on C conservatively for larger caches. These are planning
allocations; monitor actual space and the Docker host drive. Only C needs Docker.

For a new A launch, from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_a.ps1 -Stage FK -DatasetRoot 'D:\AV-Deepfake1M'
```

**A is already running: do not launch another copy.** The old outer queue was
detached separately so it cannot start AST/ViViT after FK. Its FK subprocess
continues in the same output directory, retaining quality state. See
`runs/two-day-001/job.json`, `three-machine-transition.json` and
`experiment/status.json`. Do not stop unrelated processes or clear saved work.

B uses `machine_b.ps1 -Setup` then `-Run`. C uses `machine_c.ps1 -Setup`, loads
the supplied image with `docker load`, then uses `-Run`. Both verify role-specific
bundle hashes and actual CUDA first. Every epoch budget is unchanged. No model
waits for another machine's training before starting.

## Return and compare on A

B returns **return-b-to-machine-a**; C returns **return-c-to-machine-a**.
Return bundles contain selected checkpoints/run provenance, predictions/metrics,
and a role/population-bound checksum inventory. C completes fusion itself, so no
feature-cache transfer is needed. Keep prepared caches and optimizer state locally.

After FK and both workers have finished, copy both returns to A and adjust paths:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_a.ps1 -Stage Compare -ReturnedB 'E:\return-b-to-machine-a' -ReturnedC 'E:\return-c-to-machine-a'
```

Import validates role, population and all file hashes before copying. An identical
already-verified import is reused; different or partial imported results are
rejected. The comparator requires identical validation IDs and labels and reports
full-population, pairwise and common-assessed metrics with coverage. Output:
`runs/two-day-001/three-model-comparison.json`. An existing comparison is preserved;
choose a new output or retain it under another name for a deliberate rerun.

## Export fresh bundles on A

Use new output folders. Existing folders are never overwritten by the exporter:

```powershell
& '.\.venv-gpu\Scripts\python.exe' scripts/export_second_machine.py --role B --manifest runs/two-day-001/population/manifest.json --dataset-root 'D:\AV-Deepfake1M' --index runs/two-day-001/avtenet/shared-index --output '..\Machine-B-bundle'
& '.\.venv-gpu\Scripts\python.exe' scripts/export_second_machine.py --role C --manifest runs/two-day-001/population/manifest.json --dataset-root 'D:\AV-Deepfake1M' --index runs/two-day-001/avtenet/shared-index --output '..\Machine-C-bundle' --include-container-image
```

`--container-archive` can reuse the already-exported CUDA tar to avoid another
Docker save. B's bundle omits all AVTENet weights and the image. C receives only
verified consumed AST/ViViT files, AV-HuBERT checkpoint/license/notice and the
optional image, excluding upstream download caches and local registries.
