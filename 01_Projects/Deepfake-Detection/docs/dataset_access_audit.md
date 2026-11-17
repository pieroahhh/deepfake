# Dataset access audit

## Local LAV-DF pilot

Use the extracted dataset's `metadata.min.json` to build the pilot for FK and the Section III-E internal baselines:

```powershell
.\.venv\Scripts\fk.exe prepare-lavdf-pilot --dataset-root .\LAV-DF --output .\data\lavdf-pilot-002
```

Use a fresh output directory for each run. `manifest.json` preserves the official
train and dev splits, maps dev to validation, and hashes the selected clips.
Each selected family contributes its genuine clip and one variant in each of
the audio-only, video-only, and audio-visual categories. The default seed is 7,
with eight training families and four validation families. `audit.json` records
the metadata digest, selection settings, and excluded annotations. Clips with
invalid manipulation intervals are excluded without changing their timestamps.
The pilot uses no test media. The dataset authors describe the generation and
split protocol in the [LAV-DF paper](https://research.monash.edu/files/522960745/509050212_oa.pdf).

The shared manifest takes clip-level audio and video labels from `modify_audio`
and `modify_video`. Its combined intervals preserve the endpoints in `fake_periods`.
For a clip flagged in only one modality, those intervals apply to that modality.
An unmodified modality has an empty interval list. For clips flagged in both
modalities, the separate interval fields are `null` because the metadata supplies
one combined list. FK training also requires a cache manifest.

For the LAV-DF FK pilot, run `fk cache-fk` with the verified manifest, extracted
dataset root, and checksummed asset registry. The cache command uses the in-frame
crop policy and records it with every cache. `fk train-fk` carries that policy into
the checkpoint, and `fk infer-fk` uses it when generating the record.

```powershell
fk cache-fk --manifest data/lavdf-pilot-002/manifest.json --dataset-root LAV-DF --assets checkpoints/fk-assets-001/assets.local.json --device cuda --output runs/lavdf-fk-cache-001
fk train-fk --manifest runs/lavdf-fk-cache-001/training-manifest.json --model-config configs/ours.provisional.json --training-config configs/fk.training-v1.json --device cuda --epochs 2 --output runs/fk-gpu-pilot-001
```

Any positive-duration overlap with an annotated edit marks a usable window as
fake. With complete annotations, a window outside every edit is genuine.
Modality targets use their published or unambiguously attributable intervals.
Within a combined edit whose modality timings are unknown, the separate labels
remain `null`. The cache manifest retains rejected windows and their reasons.
Resume from `last.pt` into a fresh output directory with a larger total epoch
budget. Moving between CPU and GPU preserves training state and progress, with
device-specific numerical differences possible.

## AV-Deepfake1M access audit

Checked 11 November 2026. Status: documentation and public file listing reviewed; actual annotation records and media have NOT been obtained or audited. Gate G1 remains incomplete.

## Selected candidate

The project plans propose the original AV-Deepfake1M validation release as a source pool for a custom grouped research split. AV-Deepfake1M++ is a separate release and must not be substituted silently.

## Official evidence

- [Official repository](https://github.com/ControlNet/AV-Deepfake1M): documents `file`, `original`, `split`, `modify_type`, `audio_model`, `fake_segments`, `audio_fake_segments`, `visual_fake_segments`, `video_frames`, and `audio_frames`. The documented schema has no explicit subject-ID field. `original` has different meanings for real and fake records, so family resolution must follow references rather than blindly grouping this field.
- [Dataset card](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M): access requires reviewing/accepting the conditions through an account. The card displays 404 GB for the repository overall; this is not the validation size or extracted working-storage requirement.
- [Public file listing](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M/tree/main): displays `val_metadata.json` at 19.1 MB and `val_metadata.zip` at 63.3 MB. Their contents/equivalence are unverified. Metadata and media downloads are gated. Do not assume the ZIP is interchangeable with the JSON.
- [2024 task rules](https://deepfakes1m.github.io/2024/details): classification training restricts labels to video level; temporal localization permits interval labels. Our proposed temporal supervision must be reported under our research protocol, not as compliance with the restricted classification track.

## Local access limits

No authenticated dataset session has been established in this work. A shell request to the public Hugging Face listing API failed with DNS resolution error. Browser-based documentation retrieval succeeded. No metadata or videos were downloaded, no terms were accepted, and no credentials were requested or examined.

## Next executable sequence

1. User checks account access at the official dataset page and reviews the linked terms. Do not put access tokens in chat or committed files.
2. Obtain `val_metadata.json` through authorized access, retaining release/revision, retrieval date, source URL, byte size and SHA-256. Inspect its actual records before committing to the source pool.
3. Audit required fields, missing/unknown annotations, modality categories, interval validity/units, duplicate paths, original references and source-family grouping. Verify how paths map to identities; unknown identity must remain unknown. Do not claim complete annotations or subject-disjointness from field names alone.
4. Inspect the validation media layout and download granularity. Measure archive and extracted size before choosing how to retrieve pilot videos. The public listing alone does not establish selective-download feasibility.
5. Freeze provisional group assignments and deterministic sampling. Select pilot IDs only from training/validation groups; do not run test inference.
6. Download selected pilot media on a suitable machine, hash it, inspect stream metadata and verify annotation alignment. Measure storage, preprocessing and model feasibility before final common manifest freeze.

## Still unknown

Approved account access; pinned release revision; actual record/category/identity counts; annotation completeness; identity mapping; validation media bytes; selective retrieval support; extraction peak storage; physical-host GPU availability. The 6k/1k/2k targets in the sprint plan remain planning targets.

No full-media download or model-package installation has been performed. The current VM inventory is in `docs/machine_inventory.md`.
