# Single-Rack Automatic Alignment and Partial Insertion

This project uses synchronized wrist-mounted RGB cameras to locate one RJ45 port, move a Franka ToolCenter to a 50 mm pre-insert standoff from the **physical front opening**, then execute a guarded two-stage motion that finishes 10 mm inside the opening.

## Supported architecture

1. `perception.py` uses YOLOE plus dark-cavity refinement to select the same port in both eyes.
2. `front_plane.py` computes local left-right-consistent SGBM disparity around the cavity, selects the nearest coherent four-sided bezel cluster, fits a stabilized front plane, and intersects a fused cavity-center ray with that plane.
3. `live_control.py` replaces the recessed cavity observation with the automatically calculated opening-plane observation.
4. `main.py` sends only that refined observation to the translation-only stop-and-look controller and debug marker.
5. `cable_runtime.py` loads the connected cable before play, enables GPU PhysX, mounts the existing rigid RJ45 plug directly to `panda_hand`, and validates the mount before YOLOE starts.
6. `insertion.py` freezes the physically settled ToolCenter frame, commands a 40 mm coarse approach followed by 20 mm fine motion, and holds on abort or completion.

Runtime visual control is image-only. RTX/USD raycasts and ground-truth JSON are used only by the offline benchmark. Perception is frozen once visual alignment completes; approach and insertion use the frozen physical ToolCenter frame.

## Cable mount

The cable asset already contains one Omni Physics deformable tail and rigid connector bodies. The tracked plug uses a direct fixed joint to `panda_hand`; the asset-authored built-in deformable attachment between the plug and tail is preserved.

The controlled ToolCenter represents the RJ45 insertion tip. Startup placement aligns that tip with the existing calibrated ToolCenter frame without changing the visual-servo orientation contract.

The cable hierarchy contains authored affine scale on both the root and plug transforms. Startup placement therefore applies a rigid world correction on top of the existing affine root transform. It must not require the root or plug matrices to be orthonormal, and it must not strip, invert, or bake away the authored cable scale.

The mount must validate for 30/30 consecutive frames before perception starts:

- RJ45 tip error ≤ 0.5 mm
- RJ45 axis error ≤ 1.0°
- fixed joint remains valid
- built-in deformable attachment remains unchanged
- GPU dynamics remains active

## Runtime

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" main.py
```

Canonical camera resolution is 1280×960. The visual servo keeps a fixed wrist orientation, uses bounded 5 mm coarse target updates, and holds position on detector/SGBM/plane-fit failure.

After visual alignment and 30/30 final ToolCenter settle frames, the runtime freezes the current pose and frozen ToolCenter +Z axis, then executes 48 commands:

- **40 mm coarse approach:** eight 5 mm commands, ending 10 mm before the physical opening
- **20 mm fine motion:** forty 0.5 mm commands, crossing the remaining 10 mm to the opening and continuing 10 mm inside the opening
- every target is computed from the frozen start pose, not accumulated from measured motion
- each command requires ≤0.3 mm physical target error for 6 consecutive frames
- each command has a 2.0 second timeout
- orientation stays fixed and perception remains frozen
- after command 48, the robot holds; it does not seat, release, or retreat

The controller holds on abort when lateral drift exceeds 0.5 mm, orientation error exceeds 1.0°, plug mount error exceeds its existing limit, Lula rejects or raises on a target, target publication fails, a command times out, or cable topology becomes invalid.

## Pure and structural tests

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" -m unittest -v \
  tests.test_front_plane \
  tests.test_live_control \
  tests.test_runtime_wiring \
  tests.test_two_stage_runtime_wiring \
  tests.test_benchmark \
  tests.test_ground_truth \
  tests.test_repo_cleanliness \
  tests.test_automatic_port_ground_truth \
  tests.test_cable_geometry \
  tests.test_affine_root_geometry \
  tests.test_scale_aware_cable_mount \
  tests.test_cable_mount_contract \
  tests.test_partial_insertion \
  tests.test_two_stage_insertion
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

## Safety constraints

- No manual recess or depth offset.
- No fallback to the recessed cavity depth.
- No RTX, USD mesh query, or ground-truth JSON in runtime control.
- No orientation commands from vision.
- No visual updates during approach or insertion.
- No full seating, connector release, or automatic retreat in this milestone.
- Failed observations hold the current target and trigger reacquisition before alignment.
- Approach or insertion failures hold the latest safe target and print the measured reason.
- No proxy connector or duplicate deformable attachment.
- No per-frame cable transform overwrite.

## Generated files

`camera_output/`, model weights, Python caches, generated ground truth, and local worktrees are ignored by Git.
