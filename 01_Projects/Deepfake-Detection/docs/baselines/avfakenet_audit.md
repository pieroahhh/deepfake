# AVFakeNet reconstruction audit

Audit date: 12 November 2026. Scope: the supplied final published PDF, its
architecture figures, current decision adapter, and public source discovery.
This is a specification audit, not a reconstructed network or a result reproduction.

Implementation update: a separate [reconstruction-v1 guide](avfakenet_reconstruction_v1.md)
now documents the chosen settings and implemented DST-Net, preparation helpers,
branch runner and synthetic checks. The original paper gaps below remain gaps
in author evidence; they have not become paper-specified values. The readiness
discussion at the end records the state at this audit, before that implementation.

## Source and method

Hafsa Ilyas, Ali Javed, Khalid Mahmood Malik, *AVFakeNet*, Applied Soft Computing
136 (2023), 110124. [Published DOI](https://doi.org/10.1016/j.asoc.2023.110124).

Local source: `AVFakeNet.pdf`, 16 pages. SHA-256:

```text
b761456da20989dcad126af3b6be7a5127085b108a2978ea57bbccc4ff4eb0e8
```

Page references below are one-based PDF pages, which match printed page numbers.
The local PDF is the implementation evidence. Existing project plans and web
summaries are not substitutes for it. The SSRN preprint is a separate version;
its contents were not substituted for the local final publication.

## What was checked

Read Sections 4.1–4.2.3, 5.2, the FakeAVCeleb protocol and augmentation discussion
in 5.4.1, Tables 2–3, and the data-availability statement. Figures 3–5 were
extracted from PDF image objects and visually inspected, not inferred from OCR.
Figure extraction used pypdf; no full PDF or figure copies were added to Git.

The evidence ledger is [specification_ledger.csv](../specification_ledger.csv).
The [audit configuration](../../configs/avfakenet.audit.json) captures numerical
facts and explicit nulls. It is intentionally not a runnable training config.

## Established pipeline

1. Extract frames and detect faces using MTCNN (p5, Section 4.1).
2. Resize faces to 128 by 128 (p8, Section 5.2).
3. Generate a librosa Mel spectrogram using FFT 2048, hop 512, and 175 Mel bins;
   apply `power_to_db` (p5, Section 4.1).
4. Store spectrograms as three-channel images (p10, audio-only discussion) and
   resize inputs to 175 by 175 (p8, Section 5.2).
5. Train separate DST-Net visual and audio classifiers, from scratch
   (p5, Section 4.1; p8, Section 5.2).
6. Majority-vote face-frame predictions within each video. Declare fake if that
   visual decision OR the audio decision is fake (p5, Equations 1–2; p6, Figure 3).

Three channels are established; the exact color mapping is not. We cannot
silently assume replicated grayscale, RGB colormap, or contemporary defaults.
Similarly, the paper does not establish the sample rate required to interpret
FFT and hop sizes in seconds.

## Architecture: confirmed structure, unresolved dimensions

| Part | Established | Still required |
| --- | --- | --- |
| Input | Three dense layers before patch extraction; p6, 4.2.1 | Widths, activation, tensor axis, retained spatial layout |
| Patches | Non-overlapping 3 by 3; initial embedding 64; p6, 4.2.2 | Padding/cropping and embedding implementation |
| Stages | Figure 4 depicts three consecutive-block groups and three merges | Exact repeat counts, channel growth and reduction details |
| Attention | Text: window attention then shifted-window attention; p7 | Heads, window/shift sizes, positional bias and masking |
| Residuals | Pre-normalization with residuals around attention and MLP; p7 | Epsilon and implementation details |
| MLP | Four dense layers; first two with ReLU/dropout, last two with dropout; p7 | Widths and dropout rates |
| Pooling | 1D global average pooling after Swin; p6 | Exact tensor layout after final merge |
| Output | Two ReLU dense layers with dropout, then two-output softmax; p7 | Hidden widths and dropout rates |

Figure 5 labels both attention blocks W-MHSA, whereas the text explicitly says
that the second uses SW-MHSA. That discrepancy must be recorded. Following the
text is a defensible reconstruction choice, not author-code verification.

Neither input size is divisible by 3: 128 mod 3 = 2, and 175 mod 3 = 1.
That arithmetic demonstrates why patch-boundary behavior must be specified; it
does not demonstrate which policy the authors used. Dense layers before patching
also prevent a complete numeric shape trace until their tensor interpretation is
settled. A stock Swin model is not established as equivalent to this architecture.

## Training and protocol

Pages 8–9, Section 5.2, specify AdamW, learning rate 0.001, batch size 16,
weight decay 0.0001, label smoothing 0.1, and validation-accuracy early stopping
with patience 5. The model is trained from scratch.

The paper describes binary cross entropy but also a two-output softmax head.
These do not identify target encoding, smoothing, and reduction unambiguously.
One must specify those before training; neither substituting cross entropy nor
using sigmoid BCE can be called an exact author-verified implementation.

Maximum epochs, scheduler, AdamW betas/epsilon, seed, minimum improvement, and
checkpoint tie handling were not established by this audit. Augmentation families
are listed on p10, but the numeric strengths, sampling probabilities and complete
composition recipe are not established.

Table 2 labels FakeAVCeleb splits 80:20 but gives FaFv counts 8,753/2,081 and
RaFv counts 7,841/1,866, which are not exactly 80:20. Preserve reported counts
as reported; do not invent exact original membership from a ratio. Pages 10 and
12 describe different native branch-training experiments: FaFv versus RaRv, and
RaFv/RaRv for visual with FaRv/RaRv for audio. These are not one universal
training checkpoint pair.

Our proposed AV-Deepfake1M common-data protocol is an experimental adaptation.
It requires audited modality labels and an explicit policy for partial edits;
video-level fake labels must not automatically label a genuine modality fake.
Do not use test data to select reconstruction settings. Any native-protocol
reproduction belongs in a separate run from the common-data comparison.

## Decisions and metrics

The existing `avfakenet_decision` adapter implements per-call frame voting and
hard OR, requires a tie policy, and returns no continuous score. Its policy for
missing inputs is a project assessment adaptation. It does not run DST-Net.

The paper reports AUC in Table 3. Equations 1–2 establish hard decisions but do
not establish a unique continuous fused score. We therefore cannot claim that
an average or maximum of branch probabilities reproduces the published AUC.
Native hard-decision evaluation can report confusion counts, sensitivity,
specificity, balanced accuracy and F1. Any continuous extension must be separately
named. No paper accuracy or latency is a measured result of this repository.

The PDF references Algorithm 1 on p5, but the text extractor did not expose a
standalone algorithm listing. This audit does not attribute loop placement or
counter-reset behavior to unseen pseudocode. The adapter's per-video reset is
instead a necessary implementation behavior already tested in the repository.

## Author-code search and limits

Public web searches included `AVFakeNet github`, `AVFakeNet Ilyas code
implementation`, `AVFakeNet source code`, author-name/GitHub combinations, and
the final DOI with `code`. No authenticated author repository or checkpoint
was identified in the returned material. This does not prove none exists.

Primary publication locations checked:

- [Published DOI](https://doi.org/10.1016/j.asoc.2023.110124): retrieval through
  the browsing tool failed; publisher supplements were not verified.
- [Author-listed SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4216427):
  search returned its bibliographic record; direct retrieval returned HTTP 403.
- Local p15 data-availability statement points to public datasets; it does not
  provide an implementation repository. No GitHub code link was found in the
  local PDF's extracted text or URI annotations.

Search snippets from unrelated papers can attach their own code links beside
AVFakeNet citations. Such links were not accepted as AVFakeNet author code.
No candidate code was imported, no license was assumed, and no author was contacted.

## Implementation order and readiness

The decision adapter is available. The DST-Net network, preprocessing and training
are not implemented. Exact reconstruction readiness is not established.

Next, define a versioned reconstruction profile that explicitly supplies the
unresolved dimensions, attention settings, padding, loss and preprocessing above.
Keep paper facts separate from chosen values. The profile must have no hidden
fallbacks, and its identity must appear in every run.

Then implement the three-dense input, custom four-dense MLP, attention/merging
stages and output head. Validate tensor shapes at both input resolutions,
shifted-window masks, parameter counts, finite gradients, branch-specific targets
and two-successive-video vote independence. Numerical parameter-count checks
validate our profile; they cannot establish agreement with an unknown author
parameter count. Add preprocessing only with its pinned settings and provenance.

A synthetic forward/backward run validates software mechanics. One actual video
through both branches validates integration. Neither establishes published
accuracy; that requires controlled dataset runs and a frozen evaluation protocol.
