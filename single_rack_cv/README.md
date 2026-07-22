# Single-Rack Automatic Front-Opening Alignment

This project uses synchronized wrist-mounted RGB cameras to locate one RJ45 port and move a Franka ToolCenter to a 50 mm pre-insert standoff from the **physical front opening**, not the recessed dark cavity.

## Supported architecture

1. `perception.py` uses YOLOE plus dark-cavity refinement to select the same port in both eyes.
2. `front_plane.py` computes local left-right-consistent SGBM disparity around the cavity, selects the nearest coherent four-sided bezel cluster, fits a stabilized front plane, and intersects a fused cavity-center ray with that plane.
3. `live_control.py` replaces the recessed cavity observation with the automatically calculated opening-plane observation.
4. `main.py` sends only that refined observation to the translation-only stop-and-look controller and debug marker.

Runtime control is image-only. RTX/USD raycasts and ground-truth JSON are used only by offline scoring.

## Runtime

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" main.py
```

Canonical camera resolution is 1280×960. The controller keeps a fixed wrist orientation, limits each target update to 1 mm, holds position on detector/SGBM/plane-fit failure, and does not command insertion.

## Pure and structural tests

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" -m unittest -v \
  tests.test_front_plane \
  tests.test_live_control \
  tests.test_runtime_wiring \
  tests.test_benchmark \
  tests.test_ground_truth \
  tests.test_repo_cleanliness \
  tests.test_automatic_port_ground_truth \
  tests.test_alignment_stress \
  tests.test_alignment_stress_runner
```

## Qualification benchmark

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
set -o pipefail
bash tools/run_benchmark.sh 2>&1 | tee camera_output/front_plane_benchmark_console.txt
status=${PIPESTATUS[0]}
echo "benchmark exit status: $status"
cat camera_output/front_plane_benchmark/report.txt
```

The launcher automatically recaptures the 60 stereo pairs or regenerates RTX ground truth when 1280×960 resolution metadata is missing or stale.

Exit codes:

- `0`: every qualification gate passed
- `2`: benchmark completed but did not qualify
- `1`: runtime or input-generation failure

Qualification gates:

- pair success rate ≥ 95%
- track switches = 0
- radial 3D jitter ≤ 0.5 mm
- correspondence ray-gap p95 ≤ 0.5 mm
- plane-residual p95 ≤ 0.5 mm
- plane-error median ≤ 0.5 mm
- plane-error p95 ≤ 1.0 mm

## Starting-pose stress qualification

Run the benchmark first so `benchmarks/front_plane_ground_truth.json` exists and is current. The stress harness launches 27 fresh Isaac Sim processes: a 3×3 world-frame Y/Z grid at −10, 0, and +10 mm, repeated three times in deterministic shuffled order. World X and wrist orientation remain unchanged.

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
set -o pipefail
bash tools/run_alignment_stress.sh \
  2>&1 | tee camera_output/alignment_stress_console.txt
status=${PIPESTATUS[0]}
echo "alignment stress exit status: $status"
latest_dir=$(find camera_output/alignment_stress -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
cat "$latest_dir/report.txt"
```

Stress exit codes:

- `0`: all 27 isolated runs passed every gate
- `2`: the suite completed but at least one run failed qualification
- `1`: infrastructure failure or interruption

Each suite creates:

```text
camera_output/alignment_stress/<timestamp>/
  summary.json
  summary.csv
  report.txt
  runs/
    y-10_z-10_repeat-1/
      console.log
      runtime_output.txt
      child_result.json
      result.json
```

The child runtime records image-only control evidence. It never reads benchmark ground truth. The parent reads ground truth only after the child exits and scores the final ToolCenter target against the physical opening plus the 50 mm standoff.

**Kill switch:** do not start insertion work unless the latest report contains all three lines:

```text
passed_run_count=27
failed_run_count=0
QUALIFIED=True
```

Do not remove failing poses or widen limits to manufacture a pass. Inspect each failed run's `result.json` and `console.log`.

## Safety constraints

- No manual recess or depth offset.
- No fallback to the recessed cavity depth.
- No RTX, USD mesh query, or ground-truth JSON in runtime control.
- No orientation commands from vision.
- No insertion motion.
- Failed observations hold the current target and trigger reacquisition.

## Generated files

`camera_output/`, model weights, Python caches, generated ground truth, and local worktrees are ignored by Git.
