#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

unset LD_LIBRARY_PATH PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
unset GZ_CONFIG_PATH IGN_CONFIG_PATH CONDA_PREFIX VIRTUAL_ENV

printf '[ALIGNMENT STRESS] 3x3 world Y/Z grid, 3 repeats, 27 runs\n'
printf '[ALIGNMENT STRESS] child timeout=240s parent timeout=270s\n'
printf '[ALIGNMENT STRESS] qualification requires 27/27\n'

exec /usr/bin/python3 tools/run_alignment_stress.py
