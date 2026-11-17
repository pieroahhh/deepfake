# Dataset-independent model infrastructure

## Current comparator workflow (14 November 2026)

Latest three-machine setup is documented in
[the Thai guide](three-machine-guide.th.md). The explicitly approved AVTENet
joint-attention ViViT adaptation has a downloaded, checksummed Google backbone
and a completed small real-media train/cache/fusion/evaluation check. Its loader
uses no pooler, matching the original classifier's backbone. Factorized ViViT
remains a separate unavailable-checkpoint profile; adaptation results are labeled.

The independent [AVFakeNet](../baselines/avfakenet/README.md) and
[AVTENet](../baselines/avtenet/README.md) workflows now live under `baselines/`.
They provide audited population indices, streamed real-media inputs, resumable
training, evaluation and matched-ID comparison. Existing imports remain available.
Use [the current setup guide](../baselines/README.md) for Windows paths and Linux
AV-HuBERT isolation; the older examples below are historical infrastructure notes.

Actual MTCNN/AVFakeNet real-media one-step training, checkpoint reload and video
evaluation passed. AST pretrained extraction, one-step fine-tuning and selected
feature caching also passed. AV-HuBERT Linux one-step training, checkpoint reload
and selected-expert feature caching passed on real clips, with backbone gradients.
The official Scenic factorized ViViT checkpoint returned HTTP 403 during local
download, so complete pretrained AVTENet readiness and conversion parity remain
unverified. No full dataset training or benchmark accuracy is claimed.

Implemented on 12 November 2026. This guide separates working local mechanics
from pretrained assets and runtime integrations that have not been obtained.
No dataset training, pretrained detection result, or final benchmark freeze is claimed.

Follow-up: the [non-dataset preparation report](non_dataset_completion.md) covers
source-to-record orchestration, retrainable ablations, degradation/reporting tools,
native checkpoint conversion/parity helpers and actual acquisition attempts. The
software path now runs on generated media with synthetic providers. Real pretrained
and real-face validation are still blocked by absent assets/runtime/recording.

## Implemented and locally exercised

| Area | Code | Validation |
| --- | --- | --- |
| Factorized ViViT | `backbones.FactorizedViViT` | Small real spatial/temporal transformer, finite gradients, strict safetensors reload |
| AST | `backbones.ASTBackend` | Actual installed Transformers AST, both token conventions, exact local checkpoint loading |
| AV-HuBERT interface | `backbones.AVHubertBackend` | Synthetic official-interface fixture, gradients, missing/misaligned input rejection |
| FK encoders | `encoders.FrozenFKEncoders` | Tiny actual Wav2Vec2 plus a synthetic visual fixture; frozen state, per-window output/gradient contracts |
| Timestamped decoding | `media` | Generated FFV1/PCM media, nearest-PTS sampling, late-start audio preserved as missing coverage |
| Face/lip geometry | `faces` | Synthetic landmark/detection fixtures, alignment/crop dimensions and missing-frame handling |
| Window preparation | `preparation` | Generated media through all decode/crop/audio gates, 50 faces, 32,000 samples, PCM16 derivation hashes |
| Speech intervals | `vad` | Minimum duration, merging, padding and source bounds with supplied probabilities |
| Frozen feature caches | `feature_cache` | Round trip, changed provenance and corrupted payload rejection |
| FK training | `training` | Whole-video gradients, split/family/hash checks, bit-exact CPU resume and reload |
| Manifest workflow | `infrastructure.train_fk_manifest` | Checksummed cache metadata, actual training and validation-only threshold selection |
| AVTENet stages | `avtenet_training` | Expert training/checkpoint selection, unknown modality-target exclusion; fusion freezing also tested |

These checks use synthetic data and small random networks where stated. The
actual architecture classes run, but no test labels those weights as pretrained.
The tested AST checkpoint is generated locally from a tiny random configuration,
not downloaded AudioSet weights. No learned detector or actual AV-HuBERT model
was exercised.

## Still external or unverified

Update after user-side downloads: AST, EfficientNet-B0, Wav2Vec2 Base, Silero and
YuNet are now present in `checkpoints/pretrained-001/`. Their checksums and actual
synthetic-input runtime checks passed; see `runs/pretrained-runtime-checks-001/runtime-checks.json`.
FK encoder outputs were [1,50,1280] and [1,99,768], followed by a random FK head
and a record with zero verification error. This is runtime compatibility, not
real-face validation or trained classification. The original missing-asset list
below records the earlier state; ViViT and AV-HuBERT remain unresolved.

- Actual pretrained weights for factorized ViViT, AST, EfficientNet-B0,
  Wav2Vec2 Base, AV-HuBERT and Silero; detector/landmark assets.
- A validated conversion/parity check from an official factorized ViViT
  checkpoint to our PyTorch architecture. The installed Hugging Face joint
  space-time ViViT is not silently substituted. Our factorized implementation
  can load an exact matching state dictionary, not an arbitrary Scenic checkpoint.
- The AV-HuBERT legacy environment: this machine has no Python 3.8 runtime,
  conda/container runtime, or working shell DNS. The setup check reports NOT_READY.
  Its install recipe, actual checkpoint loading and log-filterbank conversion
  have not been executed. Package compatibility is not certified by writing a script.
- Real detector performance, primary-track identity stability, canonical LRS3
  landmark alignment and real-world AV sync. Geometric crop tests do not validate
  learned detection or identity disjointness.
- Calibrated quality thresholds, dataset-specific adapters/augmentation, actual
  training populations, GPU profiling, robustness runs and final protocol freeze.

## Check the environment

```bash
cd /home/kali/Desktop/Deepfake_detection
source .venv/bin/activate
python -m unittest discover -s tests -v
fk infrastructure --assets configs/assets.example.json \
  --output runs/infrastructure-readiness-002.json
bash scripts/setup_avhubert.sh --check
```

Use new output paths: experiment artifacts are protected from accidental overwrite.
The example asset registry intentionally has null paths. Copy it to a local
configuration and fill in actual source/revision/checksum metadata when weights
are available. A checksum proves correspondence to the supplied manifest; it
cannot independently establish authorship, license, pretraining or conversion parity.
The readiness report never upgrades untested checksums to verified inference.

## Backbone conventions and artifact loading

The provisional architecture profile is `configs/backbones.reconstruction-v1.json`.
Its factorized ViViT has 32 frames, tubelets 2×16×16, dimension 768, 12 spatial
layers and 4 temporal layers. Those exact widths/depths are our choices, not
settings verified from AVTENet author code. Input is `[B,T,3,H,W]`, normalized to
[-1,1]. The architecture processes each tubelet's spatial tokens independently,
then processes their class vectors with a temporal transformer and class token.
`prepare_vivit` applies the explicitly chosen bilinear stretch resize with
antialiasing and scales uint8 RGB to [-1,1]. It requires the configured frame count;
sampling and clip duration must be supplied explicitly, not inferred from the model.

AST features explicitly choose `cls` (paper-guided) or
`mean_cls_distillation` (the upstream AST pooling convention). Loaders reject
partial loading, random/synthetic manifests by default and incompatible patch
geometry. Never silently replace a missing pretrained checkpoint with a fresh model.

Each local model directory needs an independently retained manifest with:

- `schema: model-artifact-v1`, `source`, immutable `revision`, `architecture`,
  `weights_status: pretrained`, and a mapping `files` of relative file names to
  lowercase SHA-256 digests.
- Factorized ViViT: architecture `factorized-vivit-reconstruction-v1`,
  `config.json`, and an exact matching `model.safetensors`.
- AST: architecture `transformers-ast-base`, `config.json`, `model.safetensors`,
  `preprocessor_config.json`, saved as the base `ASTModel` without a task head.
- EfficientNet-B0: architecture `torchvision-efficientnet-b0`, full torchvision
  state dictionary in `model.safetensors`, and `preprocessor_config.json` with
  `image_size:224`, `mean` and `std` from the selected weight recipe.
- Wav2Vec2: architecture `transformers-wav2vec2-base`, base-model
  `model.safetensors`, `config.json`, and `preprocessor_config.json`. The loader
  requires 768 output channels, no extra temporal adapter, and 16kHz preparation.

State-dictionary names are checked strictly. These loaders do not convert foreign
framework checkpoints or certify that checkpoint conversion preserved outputs.
No executable remote model code or network download is enabled by these loaders.

FK EfficientNet features are global averages of the feature map before the
classification head. Wav2Vec2 runs separately on each complete two-second mono
window. Its normalization is per-window according to the selected feature
extractor. Feature-center time is `(receptive_field-1)/2 + index*total_stride`,
divided by sample rate, treating sample zero as time zero. The common Base
convolution configuration produces 99 centers from 32,000 samples, not an assumed
100. No whole-file audio encoding is substituted for this operation.

## Media preparation and assessment

`media.probe(path)` returns FFprobe stream metadata without modifying media.
`sample_video` retains source PTS, sampling error and duplicate counts. It samples
the first video stream; `aligned_audio` selects the first audio stream and averages
channels arithmetically before resampling. Both subtract the same container start,
preserving relative AV offsets. Audio has a per-sample coverage mask. Incomplete
coverage is rejected by the FK preparation gate. Audio samples outside observed
coverage are never counted as valid merely because storage was zero-initialized.
Decoding currently scans from stream start and retains a bounded requested region;
keyframe seeking is a future optimization requiring equivalent-output checks.
Display rotation and unusual edit-list behavior have not been certified.

`prepare_fk_window(path,start,source_duration,detector,source_speech_intervals)`
requires explicit source duration, detector and whole-source VAD intervals. It
returns USABLE arrays or UNASSESSED reasons. No-speech and incomplete windows
remain unassessed. It records coverage, RMS, clipping and blur measurements;
those measurements are not calibrated perceptual-quality thresholds. It does not
invent the two normalized quality features needed by the model; that mapping
must be chosen and validated explicitly.

Detector adapters include MTCNN (caller-supplied facenet-pytorch object), YuNet
(local checksummed ONNX file) and dlib68 (optional dlib plus local predictor).
YuNet and dlib are declared adaptations and must not be represented as AVFakeNet's
MTCNN. One unambiguous face initializes a track. Later associations require a
single IoU-qualified candidate; gaps are not filled with copied faces. The
geometric crop rotates eyes horizontal and centers lips from five/68 landmarks.
This is not a claim of exact official LRS3 alignment. No learned weights are bundled.

`SileroVAD` consumes a checksummed local 16kHz/512-sample JIT model, resetting its
state for the supplied waveform. The interval policy is explicit threshold 0.5,
minimum speech 250ms, merge gaps below 300ms and pad 100ms, clipped to source
bounds. This custom interval postprocessing is a reconstruction, not a claim of
bit-equivalence to every upstream Silero version. Its real JIT execution is still
pending an actual compatible artifact; the local tests exercise interval logic.

`save_prepared_window` writes lossless array storage plus PCM16 mono audio and
hashes derived files. It preserves the original and checks that source bytes did
not change during preparation. Waveform-to-PCM16 quantization/clipping is recorded.

## Frozen caches and training manifests

`write_cache(directory,tensors,provenance)` requires source, preparation, encoder,
feature-layer, window and precision metadata; it writes safetensors plus metadata.
`read_cache` checks expected provenance and payload bytes. Its metadata must itself
be trusted: the CLI requires its hash in an independently retained training manifest,
preventing a payload and its metadata from both being silently replaced.

A training-manifest row contains:

```text
video_id, source_family, source_sha256, split, label, temporal_complete,
manipulated_intervals, window_intervals, cache, cache_metadata_sha256,
cache_provenance
```

The manifest has `train` and `validation` lists; cache paths are relative to the
manifest. Cache tensors are `visual_features`, `audio_features`, `visual_times`,
`audio_times`, and `quality`. Intervals and labels remain source-time annotations,
not fake-video targets broadcast to every window. Feature timestamps within each
window are relative seconds in [0,2). All usable windows of a video remain grouped.

```bash
fk train-fk --manifest data/processed/training-manifest.json \
  --model-config configs/ours.provisional.json \
  --training-config configs/fk.training-v1.json \
  --output runs/fk-pilot-001 --calibrate
```

This command requires real prepared caches; that path is an example, not supplied
data. It rejects test rows and source/video/family leakage, seeds before model
construction, and records feature-content signatures. Each optimization step uses
the summed masked Eq.52 terms over one video. Only epoch reporting averages
per-video losses. Videos with no usable supervised terms cannot silently drive
optimization or checkpoint selection.

`last.pt` includes optimizer and Python/NumPy/Torch/CUDA RNG states. Resume with
`--resume runs/fk-pilot-001/last.pt` into a new run directory using unchanged
architecture, decision settings, training profile and cache contents. The CPU
test establishes exact uninterrupted/resumed parameter equality on its fixture;
it does not establish cross-hardware or cross-version bitwise equality.
The manifest workflow reads one video's cache at a time, keeping dataset-size
feature storage off RAM. All usable windows from that video still remain together.
`best.pt` is inference-only and restores the first strict minimum validation loss.
With `--calibrate`, balanced-accuracy threshold selection uses assessed validation
videos of both classes, writes `selection.json`, and publishes `selected.pt`.
Only the video threshold is selected by this command; other decision settings
remain provisional until a declared validation search/freeze. No test data is used.

Training settings in `fk.training-v1.json` are provisional reconstruction choices:
AdamW 1e-4, weight decay 1e-4, cap 50 epochs, patience 5, gradient norm clip 1,
loss weights 1, seed 7. They are not attributed to the manuscript.

For AVTENet use `fit_avtenet_stage` with explicit prepared input records and
`configs/avtenet.training-v1.json`. VN/AN/AVN targets come from their own modality
labels; unknown targets are excluded and counted. Experts train independently,
then fusion accepts either the frozen-expert wrapper or explicit cached vectors.
The runner selects validation loss and checkpoints each stage. Unlike FK's runner,
it does not implement exact resume; its schedule is a declared reconstruction.
No video-level result may be inferred from a clip-level metric without a pinned
aggregation policy. The AV-HuBERT differentiable adapter must run in-process in
a compatible environment to fine-tune; the extraction subprocess has no gradients.

## Primary sources inspected

- [Official ViViT/Scenic](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit):
  architecture/checkpoint discovery; no conversion parity established.
- [Official AST implementation](https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py):
  class/distillation token distinction. Installed Transformers source was also
  inspected for current forward signatures and pooling behavior.
- [Official AV-HuBERT model API](https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert.py)
  and [dataset conventions](https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert_dataset.py):
  paired modality inputs, alignment, crop/normalization settings. The actual
  immutable checkout and matching checkpoint are still required.
- [Official Silero VAD](https://github.com/snakers4/silero-vad) and
  [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet):
  artifact/provider discovery, not locally executed learned models.
