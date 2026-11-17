# Isolated AV-HuBERT environment

## Partial-install repair update

Native-build update: dependency installation has progressed to compiling Fairseq,
where the managed Python 3.8 `Python.h` requires a missing `crypt.h`. On Kali/Debian,
`libcrypt-dev` provides this system header. The setup script now preprocesses
`Python.h` before installing/building and reports missing development headers
early. Install `libcrypt-dev` from the host terminal and rerun bootstrap; no
Python environment deletion or source-header patch is needed. Later compilation
and actual imports still require verification after the system header is present.

Python 3.8.20, Torch 1.13.1+cpu and upstream preprocessing requirements are now
installed locally. The existing upstream checkout is
`258fb50e155134eec2c4b49c2ae8de267075fd18`, with Fairseq submodule
`afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`. Fairseq/Hydra/OmegaConf imports
have not yet passed: installation is incomplete, not a clock issue.

`scripts/bootstrap_avhubert.sh` and `scripts/setup_avhubert.sh` now resume an
existing compatible environment instead of refusing it. They preserve the
selected source revisions and refuse to reset modified/different checkouts.
`constraints.txt` pins legacy Cython/Hydra/OmegaConf and packaging choices.
The Fairseq editable build uses those installed build dependencies with
`--no-build-isolation` and limits parallel compilation. These are compatibility
choices that still need an actual successful build/runtime check.

Resume from a network-enabled terminal:

```bash
bash scripts/bootstrap_avhubert.sh
```

This script performs the final checks automatically. For a separate diagnosis:

```bash
bash scripts/setup_avhubert.sh --check
```

The check now reports individual dependency imports, native extensions and
AV-HuBERT registration. It does not certify checkpoint execution. The local
repair attempt reached dependency acquisition but failed DNS resolution while
fetching Cython; no Fairseq installation success is claimed. The earlier status
notes below are historical.

Status: NOT CREATED OR RUNTIME VERIFIED on this machine. Python 3.8 is absent,
and shell DNS resolution fails. The main Python 3.13 `.venv` was left intact.

A workspace-local `scripts/bootstrap_avhubert.sh` now attempts uv installation,
managed Python 3.8 acquisition and immutable-commit resolution before setup.
The actual local attempt failed at package acquisition; no legacy environment was
created. Partial tools live in ignored `.venv-tools`, not the main environment.

The [official project](https://github.com/facebookresearch/av_hubert) documents
Python 3.8 and its Fairseq submodule. The setup script uses an explicitly supplied
immutable upstream commit and initializes that tree's pinned submodule. Its CPU
Torch 1.13.1/Pip 23.3.2 pins are our compatibility starting point, not a verified
author training environment. Upstream package constraints, wheel availability and
actual checkpoint loading still need runtime validation. No GPU environment has
been configured by this recipe.

Check the prerequisite:

```bash
bash scripts/setup_avhubert.sh --check
```

On a host with Python 3.8 and network access, set `AVHUBERT_COMMIT` to the full,
verified upstream SHA and optionally `AVHUBERT_PYTHON` to its Python executable,
then run `bash scripts/setup_avhubert.sh`. The script creates `.venv-avhubert`
and `external/av_hubert`, refuses to overwrite them, installs the isolated
requirements and records actual versions only after import/package checks pass.
No upstream SHA has been invented or silently resolved from a moving branch.

The bridge is:

1. Prepare source-aligned PCM and lip crops with the modern preparation code.
2. In the legacy environment, run `scripts/prepare_avhubert_input.py` with an
   explicit checkpoint-compatible settings JSON. It requires crop size, image
   mean/std, audio normalization flag, stack order, alignment-policy description,
   and all filterbank arguments. These must be obtained from the actual model
   task configuration; do not copy generic normalization defaults blindly.
3. Run `scripts/avhubert_worker.py --help` for required inputs. Supply the pinned
   repo/commit, local checkpoint/SHA-256, paired input NPZ, settings JSON and a new
   output directory. It verifies the checkout/checkpoint and checks preparation
   against saved task settings before running the official core model.

Example preparation command after actual inputs/settings exist:

```bash
.venv-avhubert/bin/python scripts/prepare_avhubert_input.py \
  --prepared runs/prepared-video/window-000/prepared.npz \
  --pcm16 runs/prepared-video/window-000/audio-16k-pcm16.wav \
  --settings /path/to/verified-checkpoint-preparation.json \
  --output runs/prepared-avhubert-input.npz
```

The worker expects a core model exposing `extract_finetune`; it does not guess
how to unwrap arbitrary ASR/CTC/seq2seq checkpoints. Some Fairseq checkpoints may
also depend on task dictionaries or original task data; those dependencies must
be audited with the actual checkpoint. The worker has not run a real checkpoint.
It is a frozen feature-extraction subprocess; fine-tuning must run the
differentiable adapter and AVTENet head in the same compatible process. No
cross-process gradient or completed AVN training is claimed.
