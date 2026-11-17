# AVTENet audit and component reconstruction

Audit date: 12 November 2026. Primary variant: **AVTENet feature fusion with
factorized ViViT, AST, and AV-HuBERT**, not the alternative AST/ViViT AVN and
not score fusion or majority voting.

Infrastructure update: factorized ViViT code, AST and AV-HuBERT adapters,
strict local loaders, preparation helpers and stage-training runners are now
implemented. See [the infrastructure guide](../model_infrastructure.md).
Actual pretrained checkpoints, conversion parity and the legacy environment
remain unavailable/unverified; the evidence limitations in this audit still apply.

## Source version and boundaries

The supplied `AVTENet.pdf` has 17 pages and the title *AVTENet: A Human-Cognition-
Inspired Audio-Visual Transformer-Based Ensemble Network for Video Deepfake
Detection*, by Ammarah Hashmi and colleagues. SHA-256:

```text
be7eda054e2824aaf86c291e575a0ccc3c14c7f6d3445f6ab84a476a1381f197
```

The [arXiv record](https://arxiv.org/abs/2310.13103) distinguishes the October
2023 v1 from the July 2025 v2 with the human-cognition title and links the journal
DOI. The local title matches that later title, but title matching does not prove
binary equivalence to a downloaded arXiv or journal PDF. This audit uses the
fingerprinted local copy; it does not silently mix numerical settings from versions.

Read the local Sections III.A–D, IV.A–B, Tables II–III and the parameter/runtime
paragraph on p14. Page numbers below are one-based PDF pages. Rendered p6 with
MuPDF and visually inspected Figure 4, including the diagram's two parallel
convolution groups and the 756-channel residual label.

## Evidence-backed specification

| Component | Local source evidence | Unresolved requirements |
| --- | --- | --- |
| VN | p5 III.A: factorized spatial and temporal ViViT; Kinetics pretraining; fine-tune backbone and linear head; temporal cls token | Exact backbone/checkpoint, tubelet shape, depth, frame count, crop and clip aggregation |
| AN | p5 III.B: AudioSet-pretrained AST; fine-tune backbone and head; 16×16 patches overlapping by six; cls representation | Checkpoint, spectrogram dimensions, normalization, cls versus upstream distilled-token conventions |
| AVN | p6 III.C: paired lip images/log-filterbanks; LRS3-pretrained AV-HuBERT; fine-tune; MS-TCN, temporal pooling, linear classifier | Model variant, checkpoint, lip alignment/crops, audio feature stacking, sequence masks and temporal head dimensions |
| Fusion | p7 III.D.4: concatenate penultimate features, train linear classifier with experts fixed | Feature dimensions, scalar versus two-class output, exact training schedule |
| Targets | p5–6 and p8 Table II: native 0=fake, 1=real; VN visual authenticity, AN audio authenticity, AVN overall authenticity | Metadata adapter must preserve unknown modality targets |
| Optimizer | p9 IV.B: Adam; VN 1e-4, AN 1e-5, AVN 0.002, fusion 0.002 | Batch sizes, duration, stopping, scheduler and other optimizer settings |
| Preparation | p8–9 IV.A.3: consecutive video-clip frames; audio 16 kHz; lips paired with log-filterbank/audio features | Actual crop/frame/audio/window alignment and failure policies |

AST patch stride 10 follows arithmetically from size 16 and overlap 6; this does
not settle its other preprocessing settings. A generic ViViT checkpoint is not
automatically the paper's factorized spatial/temporal encoder. A generic HuBERT
audio encoder is not AV-HuBERT.

## Native protocol and common-data adaptations

Pages 7–8 specify 430 training celebrities and 70 test celebrities in
FakeAVCeleb. The Table II targets are:

| Expert | Fake external class 1 | Genuine external class 0 |
| --- | --- | --- |
| VN | FvFa and FvRa | RvFa and RvRa |
| AN | RvFa and FvFa | FvRa and RvRa |
| AVN and final fusion | FvFa, FvRa, RvFa | RvRa |

Table III reports VN 35,618 samples, AN 18,669 and AVN 36,411. The p8 discussion
adds 16,949 real VoxCeleb1 examples to VN and 17,742 to AVN, whereas AN uses the
FakeAVCeleb categories. Do not give every expert the same native training count
or silently add this external task-training data in the common-data experiment.
Pretraining data and additional task-training samples are different disclosures.

Our grouped AV-Deepfake1M protocol is a separate adaptation. It needs real
metadata, modality labels, partial-edit policies and common video IDs. Unknown
modality targets remain unknown even if a video is known fake. The implemented
`expert_target` converts validated external labels to native real-class targets.

## Architecture discrepancies

Figure 4 shows two groups of three parallel Conv1D branches with kernels 7, 5,
and 3, each at 256 output channels, with BatchNorm and PReLU. Each concatenation
is followed by dropout 0.2. The residual projection is labeled Conv1D(756,1).
Three 256-channel branches concatenate to 768, so the depicted residual addition
cannot be dimensionally inferred without a declared resolution. The paper cites
references [65] and [66] for temporal-model detail; their implementations and any
exact equivalence to this configuration remain unaudited.

The p14 parameter paragraph reports VN 114.75M, AVN 129.31M, AN 87.73M and
2.89M in the decision module. A scalar linear classifier on three 768-dimensional
vectors has 2,305 parameters, not 2.89M. Even a two-class linear head would have
4,610. Those 768-dimensional vectors are our component-profile assumption, not
established paper dimensions. We will not introduce an undocumented large MLP
merely to force agreement with the reported total.

The paper also reports 0.55s per video on an RTX 2080 Ti, explicitly excluding
pre/post-processing. That is neither local timing nor full video-pipeline timing.

## Public source search

- The [first author's GitHub profile](https://github.com/ammarahhashmi) links
  AVTENet as a publication. The inspected profile did not identify a dedicated
  AVTENet implementation.
- The author's [earlier ensemble repository](https://github.com/ammarahhashmi/Multimodal-Forgery-Detection-Using-Ensemble-Learning)
  is explicitly for the APSIPA 2022 work. Its visible files and README do not
  establish AVTENet source code. It must not be imported as if it were AVTENet.
- The [coauthor's paper list](https://github.com/sahibzadaadil/Awesome-Multimodal-Deepfakes-Detection)
  links AVTENet as a paper, not an authenticated implementation.
- [Official AST source](https://github.com/YuanGongND/ast) is identifiable as the
  backbone authors' implementation and lists a BSD-3-Clause license. This does
  not establish which checkpoint/extraction setting AVTENet used.
- [Official AV-HuBERT source](https://github.com/facebookresearch/av_hubert) is
  archived and documents a Python 3.8 environment, a Fairseq submodule and its
  own license. This is a separate compatibility/licensing task; do not install
  those legacy dependencies blindly into the current Python 3.13 environment.

No dedicated AVTENet author implementation/checkpoint was verified in the
inspected sources. That is not proof none exists. No source repository was
cloned, no third-party code imported, and no checkpoint or dataset downloaded.
Exact commits, checkpoint hashes and model compatibility remain unverified.

## Implemented component profile

`configs/avtenet.components-v1.json` has status
`PARTIAL_RECONSTRUCTION_NO_PRETRAINED_EXPERTS`. Its chosen values are:

- Assume 768-dimensional expert vectors and AV-HuBERT sequence features.
- One temporal block with the two depicted convolution groups; symmetric same
  padding, 256 channels per branch, concatenate to 768; residual 1×1 projection
  to **768**; mean over full-length unpadded sequences. BatchNorm epsilon 1e-5,
  momentum 0.1; a shared scalar PReLU per branch. Ordinary PyTorch defaults for
  Conv/Linear initialization and bias; BatchNorm defaults otherwise. No extra
  activation after the final residual addition. These are reconstruction choices.
- Concatenate VN, AN, AVN vectors in that order. One 2304→1 linear classifier
  emits a native **real** logit. External fake probability is sigmoid(-real_logit).
- Scalar BCE-with-logits uses native real targets; fusion training converts the
  project convention 1=fake to the native 1=real target explicitly.

`src/deepfake_detection/avtenet.py` implements the temporal head, linear expert
wrappers with an external-backend interface, feature fusion, stage-two expert
freezing, modality target conversion and cache provenance helpers. Wrappers do
not implement or load ViViT, AST or AV-HuBERT themselves. Frozen experts remain
in evaluation mode even when the fusion classifier trains, preventing dropout
and BatchNorm-statistic changes. Cached-feature headers bind to the profile,
all three selected expert checkpoint hashes, preprocessing and split hashes.
Header checks are provenance compatibility checks, not hashes of feature content
or proof of video-ID alignment. A complete cache store is not implemented.

`tests/test_avtenet.py` checks convolution shape/gradients, label directions,
linear fusion shape/order, exact expert-state preservation during a fusion
training step, and cache invalidation. The expert backends in these tests are
explicit synthetic fixtures. They are never represented as pretrained experts.

## Next integration steps

1. Select and pin a genuinely factorized ViViT implementation/checkpoint and the
   AST checkpoint. Trace their exact penultimate-feature extraction, including
   any AST distilled-token discrepancy, and reproduce their preparation settings.
2. Set up a separate compatible AV-HuBERT environment and choose its exact
   LRS3 checkpoint. Record commit, license, checksum, feature dimension and
   executable paired lip/log-filterbank extraction. Do not substitute audio-only HuBERT.
3. Finalize frame/clip sampling, lip tracking and AV alignment, and implement
   audited modality-specific training loaders and a documented training budget.
4. Fine-tune each real expert independently; select/freeze checkpoints using
   validation data; only then generate feature caches with video-ID alignment
   and content hashes and train the final fusion classifier.
5. Run one real video through all experts and fuse it before beginning full
   experiments. Current synthetic tests do not satisfy this integration gate.
