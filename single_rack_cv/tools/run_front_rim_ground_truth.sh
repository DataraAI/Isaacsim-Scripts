#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

unset LD_LIBRARY_PATH
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset GZ_CONFIG_PATH
unset IGN_CONFIG_PATH
unset CONDA_PREFIX
unset VIRTUAL_ENV

VALIDATOR="tools/extract_front_rim_ground_truth.py"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || printf 'unknown')"

echo "[FRONT RIM VALIDATOR] commit=${COMMIT} mode=cavity-box-rayring-v3"

if grep -q 'from front_rim import extract_front_rim' "$VALIDATOR"; then
  echo "ERROR: stale validator still imports extract_front_rim." >&2
  echo "Run: git pull --ff-only origin main" >&2
  exit 2
fi

if ! grep -q '\[GROUND TRUTH DETECTION\]' "$VALIDATOR"; then
  echo "ERROR: validator diagnostics are missing; local source is stale." >&2
  echo "Run: git pull --ff-only origin main" >&2
  exit 2
fi

exec "$HOME/isaacsim/python.sh" \
  tools/extract_front_rim_ground_truth_bootstrap.py
