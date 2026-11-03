#!/usr/bin/env bash
# Start the geometry service (scene/event memory: reconstruction, registration, rendering).
# Configuration comes from the environment; see geometry/.env.example. Typical use:
#   GEOMETRY_PYTHON=/path/to/env/bin/python GEOMETRY_MODEL_ROOT=/path/to/models ./run_geometry_service.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export GEOMETRY_BACKBONES_DIR="${GEOMETRY_BACKBONES_DIR:-$ROOT/backbones}"
export GEOMETRY_PYTHON="${GEOMETRY_PYTHON:-$(command -v python3)}"
export GEOMETRY_HOT_ROOT="${GEOMETRY_HOT_ROOT:-/var/tmp/geometry-service-${USER:-user}}"
export GEOMETRY_HOST="${GEOMETRY_HOST:-127.0.0.1}"
export GEOMETRY_PORT="${GEOMETRY_PORT:-18300}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
exec "${GEOMETRY_PYTHON}" -m atas_geometry "$@"
