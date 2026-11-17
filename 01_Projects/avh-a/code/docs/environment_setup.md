# Local environment setup

Update: the user subsequently completed package installation. All preparation
package imports, FFmpeg/FFprobe execution, and `pip check` passed. All 34 tests
passed with zero skips after correcting the float32 loss assertion tolerance.
Installed versions are recorded in `requirements-local.lock.txt`. This verifies
the local preparation stack, not full baseline or GPU training readiness.

Installation attempted on 12 November 2026. Created `.venv` successfully.
Package downloads failed because this execution session could not resolve
`download.pytorch.org` or `pypi.org`. No third-party packages were installed.
This is a network failure, not evidence of incompatible package versions.

Run from the project directory in a terminal with internet access:

```bash
bash scripts/setup_environment.sh
source .venv/bin/activate
```

The script installs CPU PyTorch, TorchVision, TorchAudio, the project, the
preprocessing packages in `requirements-preprocessing.txt`, FFmpeg and FFprobe.
It checks imports and package consistency, runs all tests, fails on skipped
tests, and writes `requirements-local.lock.txt` only after verification passes.
The script has not yet been verified end to end because downloads are blocked.

The current model consumes cached features. Installing these packages does not
implement raw-media preprocessing, supply encoder checkpoints, or train a model.
Dataset access and checkpoint selection remain unresolved in the project plans.
Legacy AV-HuBERT/AVTENet dependencies require a separate source and compatibility
audit and environment, as specified in the experiment plan. They are not included
in this local preparation environment. No NVIDIA device is exposed to this VM;
GPU training requires a suitable host or remote environment and matching drivers.
