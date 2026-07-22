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

DATASET="camera_output/prompt_ab_benchmark_v1/manifest.json"
GROUND_TRUTH="benchmarks/front_plane_ground_truth.json"
SUMMARY="camera_output/front_plane_benchmark/summary.json"
EXPECTED_HEIGHT=960
EXPECTED_WIDTH=1280

json_resolution_matches() {
  local path="$1"
  local key="$2"
  /usr/bin/python3 - "$path" "$key" "$EXPECTED_HEIGHT" "$EXPECTED_WIDTH" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
expected = [int(sys.argv[3]), int(sys.argv[4])]
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get(key) == expected else 1)
PY
}

printf '[FRONT-PLANE BENCHMARK] mode=front-plane-highres-v1\n'
printf '[FRONT-PLANE BENCHMARK] camera resolution: 1280x960\n'

if ! json_resolution_matches "$DATASET" "resolution_height_width"; then
  printf '[FRONT-PLANE BENCHMARK] dataset missing/stale; recapturing at 1280x960\n'
  rm -rf camera_output/prompt_ab_benchmark_v1
  "$HOME/isaacsim/python.sh" benchmarks/capture_dataset.py
fi

if ! json_resolution_matches "$DATASET" "resolution_height_width"; then
  printf 'ERROR: dataset capture did not create a valid %s\n' "$DATASET" >&2
  exit 1
fi

if ! json_resolution_matches "$GROUND_TRUTH" "camera_resolution_height_width"; then
  printf '[FRONT-PLANE BENCHMARK] ground truth missing/stale; regenerating at 1280x960\n'
  rm -f "$GROUND_TRUTH"
  "$HOME/isaacsim/python.sh" tools/generate_ground_truth.py
fi

if ! json_resolution_matches "$GROUND_TRUTH" "camera_resolution_height_width"; then
  printf 'ERROR: ground truth is invalid: %s\n' "$GROUND_TRUTH" >&2
  exit 1
fi

printf '[FRONT-PLANE BENCHMARK] high-resolution inputs ready\n'
rm -rf camera_output/front_plane_benchmark

set +e
"$HOME/isaacsim/python.sh" tools/run_benchmark_isaac.py
python_status=$?
set -e

if [[ ! -s "$SUMMARY" ]]; then
  printf 'ERROR: benchmark did not create %s (python status=%s)\n' \
    "$SUMMARY" "$python_status" >&2
  exit 1
fi

if grep -q '"QUALIFIED": true' "$SUMMARY"; then
  printf '[FRONT-PLANE BENCHMARK] QUALIFIED=true\n'
  exit 0
fi

printf '[FRONT-PLANE BENCHMARK] QUALIFIED=false\n' >&2
exit 2
