# AVFakeNet reconstruction v1

This profile is a runnable, untrained paper-guided reconstruction. It is not
verified author code, an exact reproduction, or evidence of detection accuracy.
It resolves the implementation gaps identified in the
[paper audit](avfakenet_audit.md) explicitly rather than claiming the paper
specified the chosen values.

## Files and execution

- `configs/avfakenet.reconstruction-v1.json`: full architecture, preparation,
  training and decision profile. Its canonical SHA-256 identifies the settings.
- `src/deepfake_detection/avfakenet.py`: DST-Net branch, profile validation,
  stable classification loss and prepared-video decision integration.
- `src/deepfake_detection/avfakenet_preprocessing.py`: RGB crop and waveform conversion.
- `src/deepfake_detection/avfakenet_training.py`: separate-branch training runner.
- `src/deepfake_detection/avfakenet_smoke.py`: synthetic shape/gradient report.
- `tests/test_avfakenet.py`: attention, padding, loss, preparation, inference and checkpoint checks.

Run the synthetic CPU check, using a new directory:

```bash
source .venv/bin/activate
python -m deepfake_detection.avfakenet_smoke \
  --profile configs/avfakenet.reconstruction-v1.json \
  --output runs/avfakenet-v1-smoke-002
python -m unittest discover -s tests -v
```

The report includes parameter counts and a complete tensor shape trace. Random
weights and inputs are used; no accuracy or latency claim is generated.

## Paper facts and our choices

Paper facts retain their page references in `docs/specification_ledger.csv`.
Everything in the right-hand column below is a reconstruction decision.

| Component | Chosen v1 behavior and reason |
| --- | --- |
| Input dense layers | Per-pixel channel projections 3→32→64→64, each with ReLU. Preserve spatial layout for subsequent patches and avoid a huge whole-image flattening layer. |
| Patch boundaries | Pad bottom/right with zeros after dense processing to multiples of 3. Preserve image content rather than crop. Project flattened 3×3×64 patches to 64. |
| Stages | Three groups, one W-MHSA/SW-MHSA pair each. Follow text for shifted attention despite Figure 5 labeling both blocks W-MHSA. |
| Dimensions | 64/128/256 within groups; merge after every group, including the final one, yielding 512 channels. Figure 4 depicts three merges. |
| Attention | 2/4/8 heads, 7×7 windows, shift 3, trainable relative-position bias initialized to zero, QKV bias, scaling by inverse square root of head dimension. |
| Padding masks | Internal window padding is excluded as attention keys. Cyclic wraparound is masked by region IDs. Invalid query outputs are zeroed before reversing/cropping. |
| Four-dense MLP | C→4C→4C→2C→C; ReLU after first two, dropout after all four. No DenseNet-style concatenation assumed. |
| Merge | Zero-pad odd axes, concatenate 2×2 neighbors, LayerNorm, bias-free linear projection 4C→2C. Boundary zeros are part of this declared policy. |
| Output | Average spatial tokens, 512→256→128→2; ReLU and dropout on hidden layers. Return logits; softmax/argmax belongs in loss/inference. |
| Regularization | Dropout 0.1, LayerNorm epsilon 1e-5, ordinary PyTorch Linear defaults, LayerNorm defaults; no stochastic depth, extra output normalization or pretrained initialization. |
| Face preparation | Caller supplies a detected uint8 RGB crop; bilinear resize to 128² with align_corners=False and no antialias, scale by 255. |
| Audio preparation | Mix channels by mean, resample to 16 kHz using soxr_hq, periodic Hann window, centered constant-padded STFT, power 2, Slaney Mel scale/normalization, 0–8 kHz. |
| Mel rendering | Per-clip maximum power dB reference, floor −80 dB, amin 1e-10. Scale [−80,0] to [0,1], bilinear resize with align_corners=False and no antialias, replicate grayscale into three channels. This does not claim to reproduce the paper's colored images. |
| Missing audio | Reject empty/nonfinite or peak Mel power ≤1e-10. This is an energy gate, not speech detection. |
| Loss | Mean BCE over the two softmax probabilities, with one-hot targets smoothed toward uniform by 0.1. Evaluate through complementary log-softmax probabilities to avoid saturation. |
| Training extras | AdamW betas (.9,.999), epsilon 1e-8, no scheduler, maximum 100 epochs; keep first strict validation-accuracy improvement; stop after 5 non-improving epochs. The epoch cap and these extra settings are not paper facts. |
| Decisions | Frame probability ties choose argmax's first class (genuine); majority-vote ties are UNASSESSED; final hard OR has no continuous fused score. |

The mean two-output softmax BCE convention is mathematically equivalent to
smoothed categorical cross entropy for exactly two complementary classes. We
implement its explicit BCE expression and test it against direct BCE on
non-saturated probabilities. That equivalence does not establish which loss
reduction the authors actually used.

## Tensor shapes

Internal image features use NHWC; prepared inputs use NCHW. N is batch size.

| Location | Visual branch | Audio branch |
| --- | --- | --- |
| Input | N×3×128×128 | N×3×175×175 |
| Dense input | N×128×128×64 | N×175×175×64 |
| Patch embedding / stage 1 | N×43×43×64 | N×59×59×64 |
| Merge 1 / stage 2 | N×22×22×128 | N×30×30×128 |
| Merge 2 / stage 3 | N×11×11×256 | N×15×15×256 |
| Merge 3 | N×6×6×512 | N×8×8×512 |
| Average pooling | N×512 | N×512 |
| Class logits | N×2 | N×2 |

Both branches have 6,770,366 parameters each in this profile, with independent weights. The input
resolution changes intermediate storage and compute, not learned layer widths.

## Preparation and inference API

```python
from deepfake_detection.avfakenet import DSTNet, load_profile, predict_prepared_video
from deepfake_detection.avfakenet_preprocessing import prepare_audio, prepare_face

profile = load_profile('configs/avfakenet.reconstruction-v1.json')
visual = DSTNet(profile, 'visual')
audio = DSTNet(profile, 'audio')
# Before actual inference, load independently trained branch state dictionaries
# whose recorded profile SHA-256 matches this profile.
visual.eval()
audio.eval()
```

`prepare_face(rgb_crop, profile)` returns one `[3,128,128]` tensor.
`prepare_audio(normalized_pcm, sample_rate, profile)` returns one `[3,175,175]`
tensor; multichannel PCM uses shape `[samples,channels]`.
Stack face tensors and add a batch axis to the single Mel image, then call
`predict_prepared_video(visual, audio, faces, mel_image)` on the model's device.
It checks branch/profile compatibility and evaluation mode and batches face
inference. It cannot infer whether a state dictionary has actually been trained.

## Training API and provenance

`fit_branch(model, train_batches, validation_batches, output, epochs=None)`
requires re-iterable batches containing `images` and either `visual_label` or
`audio_label`. Labels are int64 0=genuine, 1=fake, with unknown labels excluded
upstream. A generic video-level label key is deliberately not accepted.

The runner uses a new directory, writes the full profile/hash and seed, records
train/validation image counts, losses and accuracies, and retains `best.pt` with
model/optimizer state, selected epoch and profile provenance. It restores the
best model in evaluation mode before returning. These are image-level branch
metrics, not final video-level fused metrics. It does not resume training or
verify group separation; the caller must supply audited disjoint loaders.
A generator that is exhausted after its first pass is not a valid loader.

Use `epochs=1` only for software integration checks. The test suite checks one
synthetic training epoch, safe checkpoint loading and identical reloaded outputs.
It does not run a dataset training experiment. The runner does not add the
paper's augmentation families: a pinned, declared training-data recipe is still
required before an experimental run.

## Remaining integration work

MTCNN implementation/weights, multi-face policy, crop margins, video decoding,
frame selection, official metadata/modality-label adapters, disjoint source-family
loaders, augmentation, real-media failure handling and final video-level
assessment are not supplied by this implementation. They cannot be inferred from
a successful synthetic check. No detector/checkpoint/dataset download occurs.
AVTENet now has audited temporal/fusion components and external expert interfaces,
but no integrated pretrained experts; see `avtenet_audit.md`.
