# AVTENet feature-fusion comparator

Latest: the user-approved **joint-attention Google ViViT adaptation** is now
available with `avtenet.hf-joint-adaptation-v1.json`. Its strict checkpoint reload
and a complete two-clip real-media AST/ViViT/AV-HuBERT → caches → fusion →
evaluation check passed. Use [the two-machine guide](../../docs/two-machine-guide.md).
The factorized-encoder acquisition limitation below still applies to that
separate original reconstruction profile. Adaptation results must be named as such.

[ภาษาไทย](README.th.md) · [Shared setup](../README.md)

The code exposes visual (factorized ViViT), audio (AST), audiovisual (AV-HuBERT
plus temporal head), and final fusion stages. The runnable pipeline is a declared
AV-Deepfake1M adaptation of the existing audited reconstruction.

**Readiness:** AST and AV-HuBERT have completed one-step training/validation,
checkpoint reload and feature caching on real clips. AV-HuBERT gradients were
verified in both the backbone and temporal head in the isolated Linux runtime.
The official factorized ViViT checkpoint download returned HTTP 403. A compatible
checkpoint and conversion parity are therefore still required for the complete
ensemble. Missing pretrained weights fail explicitly. There is no random-weight
fallback and no substitution with joint-attention ViViT.

## Obtain and register assets

```powershell
& $Python -m baselines.avtenet.acquire --output checkpoints/avtenet-assets-001
$Assets = Join-Path $Repo 'checkpoints/avtenet-assets-001/baseline-assets.local.json'
& $Python -m baselines.avtenet check --assets $Assets
```

Acquisition preserves successes when another asset fails and returns exit code 2
for a failed download. Its report distinguishes downloaded weights from verified
runtime readiness. The registry records machine-local paths. Update the visual
entry only after conversion succeeds, and set AV-HuBERT's repository to its pinned
Linux checkout (`/opt/av_hubert` in the container).

Chosen backbone candidates:

- Scenic ViViT-B/16×2 FE, Kinetics-400, 32 frames, 768 features.
- `MIT/ast-finetuned-audioset-10-10-0.4593`, using the cls token and the checkpoint's
  own checksummed feature extractor, with 16×16 patches and stride 10.
- Official AV-HuBERT Base LRS3 clean pretraining iteration 5, 768 features.

These choices are not claims that the AVTENet authors used these exact files.
Record source revisions and SHA-256 in each run. AV-HuBERT preprocessing is
checked against settings saved in the native checkpoint.

`python -m baselines.avtenet.convert_vivit --help` exposes explicit tensor-map
conversion followed by parity on at least two full-sized reference inputs.
The reference NPZ contains `inputs[N,32,3,224,224]` and `features[N,768]` from the
official Scenic backend on the same source checkpoint. Its manifest records
`backend: official-scenic-factorized-vivit`, source URL, 40-character revision,
source checkpoint hash, arrays hash and reference hash. The array NPZ and mapping
use the existing `explicit-flax-tensor-map-v1` conversion contract. They must cover
every model tensor and declare all unused source tensors. Parity uses rtol 1e-4
and atol 1e-5. Failed parity leaves the artifact unverified. No conversion is
claimed until those source exports can actually be obtained and exercised.

## Train visual and audio experts

Use the index produced by the shared guide. These commands require their actual
verified assets; visual training remains blocked until ViViT is available.

```powershell
& $Python -m baselines.avtenet train --index "$AVTE/index" --dataset-root $DatasetRoot --assets $Assets --stage visual --output "$AVTE/visual" --device cpu --seed 7
& $Python -m baselines.avtenet train --index "$AVTE/index" --dataset-root $DatasetRoot --assets $Assets --stage audio --output "$AVTE/audio" --device cpu --seed 7
```

Expert targets use positive-duration overlap with that modality's annotations in
each 2-second view. Views have stride 1 second plus an end-aligned last view.
ViViT uses 32 face frames; AV-HuBERT uses 50 paired lip frames and 16-kHz audio.
Clips shorter than two seconds and unavailable inputs remain reported unassessed.
ViViT direct-resize face crops and MTCNN geometric lip alignment are explicit
adaptations, not the original Kinetics/LRS3 data preparation.

The provisional schedule uses Adam, 50 epochs maximum, patience 5 on validation
loss, one prepared view per step, and no stochastic augmentation. Learning rates
are visual 1e-4, audio 1e-5, audiovisual 0.002, fusion 0.002. Unknown targets do
not participate in loss. Native real logits are converted to external fake
probabilities by `sigmoid(-logit)`.

## Linux AV-HuBERT worker

The worker fine-tunes the backbone and temporal head in one process. A frozen
cross-process feature extractor cannot replace this training stage.

The development machine uses Docker Desktop's Linux backend. Build the pinned
CPU image from PowerShell:

```powershell
docker build --build-arg AVHUBERT_COMMIT=258fb50e155134eec2c4b49c2ae8de267075fd18 -t fk-avhubert:cpu-v1 baselines/avtenet/runtime
```

Make a **separate Linux registry**, keeping the Windows registry intact. Set the
AV-HuBERT repository to `/opt/av_hubert`; use `/workspace/...` for weights under
the repository and `/runs/...` for files under `$RunRoot`. Other absolute Windows
paths cannot be used inside Linux. Copy the Linux registry into `$RunRoot` as
`assets.docker.local.json`.

```powershell
& ./baselines/avtenet/runtime/invoke-docker.ps1 -DatasetRoot $DatasetRoot -RunRoot $RunRoot -CommandArguments @('train', '--index', '/runs/avtenet-001/index', '--dataset-root', '/dataset', '--assets', '/runs/assets.docker.local.json', '--stage', 'audiovisual', '--output', '/runs/avtenet-001/audiovisual', '--device', 'cpu', '--seed', '7')
```

The launcher mounts the repository and dataset read-only and mounts run outputs
writable. Its image is CPU-only for integration testing. It does not configure
GPU training. On a Linux GPU machine, reproduce the pinned legacy source and
compatible CUDA Torch environment before starting a full run.

For an existing WSL2 Linux environment, `runtime/invoke-wsl.ps1` accepts the distro,
Linux Python executable, Linux repository path and an argument array. Paths passed
to it must already be Linux paths. The original `scripts/setup_avhubert.sh` is a
separate environment recipe; Docker is the locally exercised isolation approach.

## Selected-expert feature caches and fusion

After all three experts have selected checkpoints, extract each one's features:

```powershell
& $Python -m baselines.avtenet cache-features --index "$AVTE/index" --dataset-root $DatasetRoot --assets $Assets --stage visual --checkpoint "$AVTE/visual/best.pt" --output "$AVTE/visual-features"
& $Python -m baselines.avtenet cache-features --index "$AVTE/index" --dataset-root $DatasetRoot --assets $Assets --stage audio --checkpoint "$AVTE/audio/best.pt" --output "$AVTE/audio-features"
& ./baselines/avtenet/runtime/invoke-docker.ps1 -DatasetRoot $DatasetRoot -RunRoot $RunRoot -CommandArguments @('cache-features', '--index', '/runs/avtenet-001/index', '--dataset-root', '/dataset', '--assets', '/runs/assets.docker.local.json', '--stage', 'audiovisual', '--checkpoint', '/runs/avtenet-001/audiovisual/best.pt', '--output', '/runs/avtenet-001/audiovisual-features')
& $Python -m baselines.avtenet merge-features --visual "$AVTE/visual-features" --audio "$AVTE/audio-features" --audiovisual "$AVTE/audiovisual-features" --output "$AVTE/fusion-features"
& $Python -m baselines.avtenet train --index "$AVTE/index" --dataset-root $DatasetRoot --stage fusion --feature-cache "$AVTE/fusion-features" --output "$AVTE/fusion" --device cpu --seed 7
```

Cache payloads and metadata are checksummed. Merge requires identical population,
preparation and component profile, and joins by video ID plus exact source-time
interval. All three expert checkpoints are bound into the fusion identity.
Only complete three-expert views enter fusion; evaluation retains videos with no
complete view in its coverage denominator. The cache does not hold raw RGB.
Feature extraction and merging require new output directories; they currently
restart after interruption. Training itself supports epoch-boundary resume.

## Evaluate the ensemble

```powershell
& $Python -m baselines.avtenet evaluate --index "$AVTE/index" --dataset-root $DatasetRoot --checkpoint "$AVTE/fusion/best.pt" --feature-cache "$AVTE/fusion-features" --split validation --output "$AVTE/evaluation" --device cpu
```

The video score is the mean of usable window fake probabilities, with threshold
0.5. Reports retain usable/expected view counts, population provenance and
integration-only status. Add `--compare-with <FK-report.json>` for matched-ID
comparison. Use the shared resume instructions for every expert and fusion stage.
