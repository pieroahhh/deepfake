#!/usr/bin/env bash
# Resume an isolated legacy installation without overwriting or resetting sources.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--check" ]]; then
  if [[ -x .venv-avhubert/bin/python ]]; then
    .venv-avhubert/bin/python scripts/check_avhubert_environment.py
  else
    printf 'NOT_READY: isolated .venv-avhubert environment has not been created.\n'
    exit 2
  fi
  exit 0
fi
if [[ -x .venv-avhubert/bin/python ]]; then
  legacy_python="$PWD/.venv-avhubert/bin/python"
else
  legacy_python="${AVHUBERT_PYTHON:-python3.8}"
fi
"$legacy_python" -c 'import sys; assert sys.version_info[:2] == (3,8), "Python 3.8 required"'
"$legacy_python" - <<'PY'
import os
import shlex
import subprocess
import sys
import sysconfig

compiler = shlex.split(os.environ.get('CC') or sysconfig.get_config_var('CC') or 'cc')
flags = shlex.split(os.environ.get('CPPFLAGS', '')) + shlex.split(os.environ.get('CFLAGS', ''))
try:
    result = subprocess.run(
        compiler + flags + ['-x', 'c', '-fsyntax-only', '-I', sysconfig.get_path('include'), '-'],
        input='#include <Python.h>\n', text=True, capture_output=True,
    )
except OSError as error:
    print('Native build prerequisite failed: ' + str(error), file=sys.stderr)
    raise SystemExit(2)
if result.returncode:
    print(result.stderr, file=sys.stderr)
    print('Python 3.8 development-header check failed before package/build changes.', file=sys.stderr)
    if 'crypt.h' in result.stderr:
        print('On Kali/Debian install: sudo apt install libcrypt-dev', file=sys.stderr)
    print('Then rerun: bash scripts/bootstrap_avhubert.sh', file=sys.stderr)
    raise SystemExit(2)
PY
if [[ -z "${AVHUBERT_COMMIT:-}" && -d external/av_hubert/.git ]]; then
  AVHUBERT_COMMIT="$(git -C external/av_hubert rev-parse HEAD)"
fi
: "${AVHUBERT_COMMIT:?Set AVHUBERT_COMMIT to an independently verified 40-character upstream commit.}"
[[ "$AVHUBERT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { printf 'Invalid commit SHA\n'; exit 2; }
if [[ ! -e .venv-avhubert ]]; then "$legacy_python" -m venv .venv-avhubert; fi
mkdir -p external
if [[ ! -e external/av_hubert ]]; then
  git clone --no-checkout https://github.com/facebookresearch/av_hubert.git external/av_hubert
  git -C external/av_hubert checkout --detach "$AVHUBERT_COMMIT"
fi
[[ "$(git -C external/av_hubert rev-parse HEAD)" == "$AVHUBERT_COMMIT" ]] || {
  printf 'Existing AV-HuBERT checkout differs from requested commit; no checkout/reset performed.\n'; exit 2;
}
git -C external/av_hubert diff --exit-code
git -C external/av_hubert diff --cached --exit-code
fairseq_commit="$(git -C external/av_hubert ls-tree HEAD fairseq | awk '{print $3}')"
if [[ ! -f external/av_hubert/fairseq/setup.py ]]; then
  git -C external/av_hubert submodule update --init --recursive
fi
[[ "$(git -C external/av_hubert/fairseq rev-parse HEAD)" == "$fairseq_commit" ]] || {
  printf 'Fairseq submodule differs from the pinned tree; no reset performed.\n'; exit 2;
}
git -C external/av_hubert/fairseq diff --exit-code
git -C external/av_hubert/fairseq diff --cached --exit-code
export PIP_CACHE_DIR="$PWD/.cache/pip-avhubert"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MAX_JOBS=1
constraints="$PWD/environments/avhubert/constraints.txt"
.venv-avhubert/bin/python -m pip install --retries 1 --timeout 15 -c "$constraints" pip setuptools wheel
.venv-avhubert/bin/python -m pip install --retries 1 --timeout 15 torch==1.13.1+cpu torchaudio==0.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv-avhubert/bin/python -m pip install --retries 1 --timeout 15 -c "$constraints" numpy Cython hydra-core omegaconf PyYAML cffi regex sacrebleu tqdm
.venv-avhubert/bin/python -m pip install --retries 1 --timeout 15 -c "$constraints" -r external/av_hubert/requirements.txt
# Use the explicitly prepared Cython/NumPy/toolchain, not a fresh isolated build
# with incompatible latest versions. Dependencies were resolved above.
.venv-avhubert/bin/python -m pip install --no-build-isolation --no-deps -e external/av_hubert/fairseq
.venv-avhubert/bin/python -m pip check
.venv-avhubert/bin/python scripts/check_avhubert_environment.py --output environments/avhubert/runtime-checks.installed.json
.venv-avhubert/bin/python -m pip freeze > environments/avhubert/requirements.installed.txt
git -C external/av_hubert rev-parse HEAD > environments/avhubert/upstream.installed.txt
printf 'Legacy environment imports/native extensions verified; actual checkpoint extraction remains separate.\n'
