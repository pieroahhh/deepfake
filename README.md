# Research source code for supervisor review

This branch contains the current deepfake detection implementation and the
source used for baseline evaluation, internal controls, forensic records, and
runtime measurements. The existing workspace layout is preserved so its path
adapters can find both native runtimes after cloning.

## Start reading here

| Location | Contents |
| --- | --- |
| [model.py](01_Projects/Deepfake-Detection/src/deepfake_detection/model.py) | Evidence decomposition, reliability weighting, and localized scores |
| [core.py](01_Projects/Deepfake-Detection/src/deepfake_detection/core.py) | Temporal windows, assessment coverage, and hierarchical aggregation |
| [preparation.py](01_Projects/Deepfake-Detection/src/deepfake_detection/preparation.py) | Face/speech preparation and usable-window gates |
| [encoders.py](01_Projects/Deepfake-Detection/src/deepfake_detection/encoders.py) | Frozen EfficientNet-B0 and Wav2Vec2 feature extraction |
| [training.py](01_Projects/Deepfake-Detection/src/deepfake_detection/training.py) | Training, checkpoint selection, and resume |
| [calibration.py](01_Projects/Deepfake-Detection/src/deepfake_detection/calibration.py) | Validation-only aggregation and threshold selection |
| [inference.py](01_Projects/Deepfake-Detection/src/deepfake_detection/inference.py) and [forensics.py](01_Projects/Deepfake-Detection/src/deepfake_detection/forensics.py) | Source-to-record inference, commitments, and verification |
| [study.py](01_Projects/Deepfake-Detection/src/deepfake_detection/study.py) and [reporting.py](01_Projects/Deepfake-Detection/src/deepfake_detection/reporting.py) | Detection/localization evaluation and reporting |
| [baselines](01_Projects/Deepfake-Detection/baselines) | AVFakeNet and AVTENet implementation/configuration |
| [AVTENet runtime](01_Projects/avh-a/code) | Retained native AVTENet runtime sources, including AV-HuBERT integration |
| [experiment scripts](01_Projects/Deepfake-Detection/scripts) | Held-out evaluation, internal controls, and experiment orchestration |
| [profiling tools](06_Tools/workspace-index) | Training/extraction scaling, stage profiling, and comparison-figure generation |
| [tests](01_Projects/Deepfake-Detection/tests) | Automated checks and synthetic test fixtures |

## Installation and reading commands

Use Python 3.11. From the clone root:

```bash
python -m pip install -e './01_Projects/Deepfake-Detection[infrastructure,reports]'
python -m deepfake_detection.cli --help
python -m unittest discover -s 01_Projects/Deepfake-Detection/tests -p test_foundations.py -v
```

For the retained GPU dependency versions, see the requirements lock files in
`01_Projects/Deepfake-Detection/`. AV-HuBERT uses a separate legacy Linux/Docker
environment; see `01_Projects/Deepfake-Detection/environments/avhubert/` and
`01_Projects/avh-a/code/baselines/avtenet/runtime/`.

Model inference and experiment scripts require separately provisioned datasets,
pretrained weights, selected checkpoints, manifests, and experiment state.
Acquire authorized data/assets and configure paths before running them. Keep
`workspace-layout.json` and `06_Tools/workspace/` with the project. Some retained
orchestration scripts are specific to the original multi-machine experiment;
they are provided for inspection, not as unattended launch instructions.

## Experimental interpretation

- AVFakeNet and AVTENet are the local evaluated implementations, with adaptation
  details in their configuration files and documentation.
- The retained development split has five shared source families between
  training and validation. The held-out population is drawn from unused official
  validation data; it is not an official-test or verified identity-disjoint claim.
- Ours uses validation-selected operating thresholds; AVTENet retains 0.5.
- Common-assessed external comparison and internal controls use different
  cohorts (415 and 980 clips). Processing availability is reported separately.
- Prepared-window timing probes and five-clip profiling have different timing
  boundaries; neither establishes full-training convergence cost.
- Record verification requires an independently retained trusted commitment.

This branch ships source, configuration, dependency specifications, tests, and
technical guides. Web demo assets, LaTeX/manuscripts, PDFs, datasets, model
weights, caches, and generated result artifacts are excluded. Synthetic fixtures
inside automated tests are software checks, not experimental evidence.

`SOURCE_MANIFEST.json` records hashes of copied original files. Research source
files are copied without algorithm changes. The branch is prepared on
2026-11-17 from the working source tree; no training or experiment rerun is part
of preparing this submission.
