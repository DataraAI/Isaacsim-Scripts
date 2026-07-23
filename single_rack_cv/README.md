# Single-Rack Automatic Front-Opening Alignment

This project uses synchronized wrist-mounted RGB cameras to locate one RJ45 port and move a Franka ToolCenter to a 50 mm pre-insert standoff from the **physical front opening**, not the recessed dark cavity.

## Supported architecture

1. `perception.py` uses YOLOE plus dark-cavity refinement to select the same port in both eyes.
2. `front_plane.py` computes local left-right-consistent SGBM disparity around the cavity, selects the nearest coherent four-sided bezel cluster, fits a stabilized front plane, and intersects a fused cavity-center ray with that plane.
3. `live_control.py` replaces the recessed cavity observation with the automatically calculated opening-plane observation.
4. `main.py` sends only that refined observation to the translation-only stop-and-look controller and debug marker.
5. `cable_runtime.py` loads the connected cable before play, enables GPU PhysX, mounts the existing rigid RJ45 plug directly to `panda_hand`, and validates the mount before YOLOE starts.

Runtime control is image-only. RTX/USD raycasts and ground-truth JSON are used only by the offline benchmark.

## Cable mount

The cable asset already contains one Omni Physics deformable tail and rigid connector bodies. The tracked plug is fixed-jointed directly to `panda_hand`; the asset-authored plug-to-tail deformable attachment is preserved.

The cable hierarchy contains authored affine scale on both the root and plug transforms. Startup placement therefore applies a rigid world correction on top of the existing affine root transform. It must not require the root or plug matrices to be orthonormal, and it must not strip, invert, or bake away the authored cable scale.

The mount must validate for 30 consecutive frames before perception starts:

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
  tests.test_cable_geometry \
  tests.test_affine_root_geometry \
  tests.test_scale_aware_cable_mount \
  tests.test_cable_mount_contract
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
- No insertion motion.
- Failed observations hold the current target and trigger reacquisition.
- No proxy connector or duplicate deformable attachment.
- No per-frame cable transform overwrite.

## Generated files

`camera_output/`, model weights, Python caches, generated ground truth, and local worktrees are ignored by Git.

## Recovery point

The exact working repository before this hard prune is preserved on:

```text
recovery/pre-single-rack-cleanup-2026-07-22
```

Local annotated tag command:

```bash
git tag -a single-rack-working-pre-cleanup-2026-07-22 \
  recovery/pre-single-rack-cleanup-2026-07-22 \
  -m "Working 1280x960 automatic front-plane controller before hard prune"
git push origin single-rack-working-pre-cleanup-2026-07-22
```
