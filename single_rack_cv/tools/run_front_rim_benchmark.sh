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

GROUND_TRUTH="benchmarks/front_rim_ground_truth.json"

printf '[FRONT RIM BENCHMARK] mode=frozen-60-pair-v2\n'

if [[ ! -s "$GROUND_TRUTH" ]]; then
  printf '[FRONT RIM BENCHMARK] ground truth missing; generating automatically\n'
  bash tools/run_front_rim_ground_truth.sh
fi

if [[ ! -s "$GROUND_TRUTH" ]]; then
  printf 'ERROR: automatic ground-truth generation did not create %s\n' "$GROUND_TRUTH" >&2
  exit 1
fi

printf '[FRONT RIM BENCHMARK] ground truth ready: %s\n' "$GROUND_TRUTH"
exec "$HOME/isaacsim/python.sh" tools/run_front_rim_benchmark_isaac.py
