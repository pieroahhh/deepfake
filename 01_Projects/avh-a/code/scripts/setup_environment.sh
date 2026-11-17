#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
export PIP_CACHE_DIR="$PWD/.cache/pip"
.venv/bin/python -m pip install --upgrade pip setuptools wheel
# This VM exposes no NVIDIA device. GPU machines need their own CUDA setup.
.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-preprocessing.txt
.venv/bin/python -m pip install -e .
# Download both tools and expose them inside the activated environment.
.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess
import static_ffmpeg
static_ffmpeg.add_paths()
import shutil
for name in ('ffmpeg', 'ffprobe'):
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f'{name} unavailable after static-ffmpeg setup')
    target = Path('.venv/bin') / name
    if not target.exists():
        target.symlink_to(Path(executable).resolve())
    subprocess.run([str(target), '-version'], check=True, capture_output=True)
PY
.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
import importlib
for name in ('torch', 'torchvision', 'torchaudio', 'numpy', 'scipy', 'pandas',
             'sklearn', 'cv2', 'librosa', 'soundfile', 'transformers', 'timm',
             'huggingface_hub', 'av', 'pytest'):
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'import OK'))
import unittest
suite = unittest.defaultTestLoader.discover('tests')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit('Setup verification failed: failed or skipped tests')
PY
.venv/bin/python -m pip freeze > requirements-local.lock.txt
printf 'Setup verified. Activate with: source .venv/bin/activate\n'
