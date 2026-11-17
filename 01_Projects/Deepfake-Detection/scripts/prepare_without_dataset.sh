#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
: "${1:?Usage: bash scripts/prepare_without_dataset.sh runs/new-check-bundle [--assets path/to/assets.local.json]}"
bundle_output="$1"
shift
.venv/bin/python -m deepfake_detection.offline_checks --output "$bundle_output" "$@"
