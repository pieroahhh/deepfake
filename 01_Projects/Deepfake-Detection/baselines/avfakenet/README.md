# AVFakeNet on AV-Deepfake1M

[ภาษาไทย](README.th.md) · [Environment and shared workflow](../README.md)

This folder owns DST-Net, image/audio preprocessing, streaming data adapters and
the runnable comparator. Legacy imports under `deepfake_detection.avfakenet*`
remain supported. `legacy_training.py` preserves the old prepared-batch API;
new dataset runs use `pipeline.py` and the common resumable trainer.

## Train the two branches

For repeated epochs, add `--prepared-cache <shared-directory>` to both branch
training commands. This stores detected face crops as lossless uint8 arrays,
their source timestamps, and the prepared Mel tensor once per video. Both
branches share the cache. Population, profile and detector identity plus payload
checksums are verified before reuse. No sampling rate/resolution is reduced.
Keep this directory on a disk with sufficient room; it contains model inputs,
not the original videos. Corrupt cache entries fail explicitly.

Use `$Python`, `$DatasetRoot`, `$RunRoot` and `$AVFake` from the shared setup.
First run the commands below with `--epochs 1 --max-steps 1` and distinct output
directories for integration testing. Omit these flags for the configured run.
An integration-only checkpoint retains that flag in evaluation reports.

```powershell
& $Python -m baselines.avfakenet train --index "$AVFake/index" --dataset-root $DatasetRoot --stage visual --output "$AVFake/visual" --device cpu --seed 7
& $Python -m baselines.avfakenet train --index "$AVFake/index" --dataset-root $DatasetRoot --stage audio --output "$AVFake/audio" --device cpu --seed 7
```

Both DST-Net classifiers train from scratch. The profile uses AdamW, learning
rate 0.001, weight decay 0.0001, batch size 16, label smoothing 0.1, maximum 100
epochs, and patience 5 on validation accuracy. Reconstruction architecture and
loss choices are inherited from the existing audited reconstruction profile.

Visual inputs use MTCNN face tracking, geometric eye alignment and 128×128 RGB
crops sampled at 25 fps. The tracker persists across decoding chunks. Supervision
uses the source timestamp of each sampled frame with half-open fake intervals.
An unknown visual interval does not become a genuine-frame label.

Audio inputs use the complete observed video interval at 16 kHz, a 175-bin Mel
image resized to 175×175, and audio-specific whole-video targets. Missing or
incomplete audio is reported as a preparation failure, not silently padded.
Both branches use no stochastic data augmentation in this declared adaptation.
This differs from an exact reproduction of the original augmentation recipe.

Source files are checked against population hashes when read. Full-resolution
RGB is decoded in chunks; transformed examples are batched without holding the
whole dataset in memory. The current decoder scans from the start for each
requested chunk, so long-video preprocessing favors correctness over seeking speed.

## Evaluate

```powershell
& $Python -m baselines.avfakenet evaluate --index "$AVFake/index" --dataset-root $DatasetRoot --visual-checkpoint "$AVFake/visual/best.pt" --audio-checkpoint "$AVFake/audio/best.pt" --split validation --output "$AVFake/evaluation" --device cpu
```

Visual predictions vote across all sampled face frames. The final decision is
visual OR audio. Visual ties or missing modalities are unassessed; there is no
invented continuous fused score. Add `--compare-with <FK-report.json>` for a
video-ID-aligned comparison. Read coverage and excluded-reason logs alongside metrics.

## Resume example

```powershell
& $Python -m baselines.avfakenet train --index "$AVFake/index" --dataset-root $DatasetRoot --stage visual --output "$AVFake/visual-resumed" --resume "$AVFake/visual/last.pt" --epochs 100 --device cpu --seed 7
```

The old run needs its matching `best.pt`. For a capped smoke run, the same
`--max-steps` value must also be supplied. A smoke checkpoint is not a shortcut
to a full-run checkpoint: use a fresh run for the actual experiment.

## Local validation

Actual MTCNN and both DST-Net branches completed a one-step train/validation
check on real AV-Deepfake1M clips. Saved checkpoints were reloaded for video
evaluation. This validates the software path only; the two-video integration
sample cannot estimate detection accuracy. See `runs/comparator-real-smoke-001`
on the development machine and the repository HANDOFF for evidence locations.
