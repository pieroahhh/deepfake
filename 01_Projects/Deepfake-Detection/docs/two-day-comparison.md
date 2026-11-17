# Two-day GPU subset comparison

[Thai operational guide](two-day-comparison.th.md)

The approved experiment runs FK and AVFakeNet on the identical AV-Deepfake1M
subset: 3,000 training and 500 validation clips, seed 7. Selection preserves
proportions by official split and modification category using seeded SHA-256
ranking of clip IDs. Original labels, annotations, source families and
`keep-official` remain intact. This is clip-level sampling: it does not imply
whole-family inclusion or identity-disjoint validation. The frozen subset audit
records five source families shared across official splits. No test labels are
used. Report results as a limited subset development comparison.

The existing machine has an RTX 3070 8 GiB. `.venv-gpu` isolates Torch
2.2.2+cu121/Torchvision 0.17.2+cu121 from the original CPU environments. The
official PyTorch previous-version CUDA 12.1 wheel recipe was used. See
`baselines/requirements.txt` for other pinned direct dependencies.

`baselines.common.subset` creates reproducible, proportional subsets from an
already audited population using bounded-memory streaming. It requires the
matching source audit. `scripts/run_two_day.py` orchestrates train-only quality
fitting, verified GPU FK caches, five FK epochs with validation calibration,
validation evaluation, and three epochs for each AVFakeNet branch (batch 8).
It then writes full coverage metrics and a comparison on mutually assessed IDs.

AVFakeNet reuses a lossless prepared-input cache across branches and epochs,
retaining the original 25 fps, crop dimensions and Mel conversion. This avoids
repeating MTCNN for each epoch. FK retains its 2-second windows and 1-second
stride, EfficientNet-B0/Wav2Vec2 frozen encoders and trainable FK head. Epoch
budgets are feasibility choices, not evidence of convergence or equal tuning.

Run from the repository using the CUDA environment:

```powershell
& '.\.venv-gpu\Scripts\python.exe' scripts/run_two_day.py `
    --manifest '.\runs\two-day-001\population\manifest.json' `
    --dataset-root 'D:\AV-Deepfake1M' `
    --assets '.\checkpoints\paper-alignment-001\assets.local.json' `
    --output '.\runs\two-day-001\experiment'
```

Logs live under `experiment/logs`, current state under `status.json`, and final
summary under `comparison.json`. Repeating the same command resumes incomplete
quality/cache work; training uses the last completed epoch in a new attempt
directory. Attempts without an epoch checkpoint restart with their logs retained.
The runner refuses changed inputs, concurrent writers, or incomplete calibration
after a finished FK budget. Keep the dataset, assets, code and environment fixed
during a run. Evaluation coverage accounts for unusable clips instead of silently
removing them. A partial evaluation directory must be preserved under another
name before retrying evaluation.

AVTENet is explicitly NOT_RUN: official factorized ViViT acquisition returned
403 and no converted checkpoint passed independent official-reference parity.
AST/AV-HuBERT integration checks do not establish complete AVTENet readiness.

Two days is an experiment budget, not a completion guarantee. Check timings and
disk usage before extending epochs or population size. Preserve the older full
quality work; an updated subset does not reuse its population-bound profile.
