#!/usr/bin/env bash
# Workspace-local tooling/Python; never replace .venv or system Python.
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="$PWD/.cache/pip-tools"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_PYTHON_BIN_DIR="$PWD/.python-bin"
if [[ -x .venv-avhubert/bin/python ]]; then
  AVHUBERT_PYTHON="$PWD/.venv-avhubert/bin/python"
elif ! command -v python3.8 >/dev/null 2>&1; then
  if [[ ! -e .venv-tools ]]; then python3 -m venv .venv-tools; fi
  .venv-tools/bin/python -m pip install --no-cache-dir --retries 0 --timeout 10 'uv>=0.9,<1'
  .venv-tools/bin/python -m uv python install 3.8
  AVHUBERT_PYTHON="$(.venv-tools/bin/python -m uv python find 3.8 --managed-python)"
else
  AVHUBERT_PYTHON="$(command -v python3.8)"
fi
export AVHUBERT_PYTHON
if [[ -z "${AVHUBERT_COMMIT:-}" ]]; then
  if [[ -d external/av_hubert/.git ]]; then
    AVHUBERT_COMMIT="$(git -C external/av_hubert rev-parse HEAD)"
  else
    AVHUBERT_COMMIT="$(.venv/bin/python - <<'PY'
from deepfake_detection.asset_download import _github_revision
print(_github_revision('facebookresearch/av_hubert','main'))
PY
)"
  fi
fi
export AVHUBERT_COMMIT
bash scripts/setup_avhubert.sh
