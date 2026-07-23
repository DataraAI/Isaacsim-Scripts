# Single-Rack Automatic Front-Opening Alignment

This project uses synchronized wrist-mounted RGB cameras to locate one RJ45 port and move a Franka-mounted RJ45 insertion tip to a 50 mm pre-insert standoff from the **physical front opening**, not the recessed dark cavity.

## Cable-mounted runtime

The supplied network-cable asset starts already mounted in the Franka hand:

- `/World/NetworkCable/E_crystal_head1_45` is the existing rigid tracked plug.
- `/World/NetworkCable/E_line_35` is the existing deformable tail.
- The asset's built-in deformable attachment between the tail and tracked plug is verified and preserved.
- `/World/CableMountFixedJoint` is a direct fixed joint from `panda_hand` to the tracked plug.
- `/World/ToolCenter` keeps its calibrated numerical transform, but physically represents the RJ45 insertion tip.
- GPU dynamics, GPU broadphase, and the TGS solver are mandatory.

The runtime does not create a proxy, duplicate the deformable attachment, move the cable every frame, grasp, release, or insert the connector.

## Supported perception and control architecture

1. `cable_runtime.py` builds the canonical rack/Franka/camera scene, enables GPU physics, mounts the cable, and blocks perception until mount validation passes.
2. `perception.py` uses YOLOE plus dark-cavity refinement to select the same port in both eyes.
3. `front_plane.py` computes local left-right-consistent SGBM disparity, selects the nearest coherent four-sided bezel cluster, fits a stabilized front plane, and intersects the fused cavity-center ray with that plane.
4. `live_control.py` replaces the recessed-cavity observation with the automatically calculated opening-plane observation.
5. `main.py` sends only that refined observation to the translation-only stop-and-look controller and debug marker.

Runtime control remains image-only. RTX/USD raycasts and ground-truth JSON are used only by offline benchmarks and asset validation.

## Inspect the cable topology

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" tools/inspect_cable_asset.py
cat camera_output/cable_asset_schema.json
```

The probe must report one Omni Physics deformable body, a rigid tracked plug, and an existing auto deformable attachment connecting those two bodies.

## Run

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" main.py
```

Before YOLOE initializes, the simulation requires **30/30** consecutive cable-mount validation frames with:

- RJ45-tip error no greater than 0.5 mm,
- connector-axis error no greater than 1 degree,
- valid direct fixed joint,
- unchanged built-in deformable attachment,
- valid deformable tail,
- active GPU dynamics.

The controller then keeps a fixed wrist orientation, limits every target update to 1 mm, holds position on detector/SGBM/plane-fit failure, and stops at the 50 mm pre-insert pose.

## Tests

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
  tests.test_cable_mount_contract

"$HOME/isaacsim/python.sh" -m py_compile \
  cable_geometry.py cable_mount.py cable_runtime.py config.py sim.py main.py \
  tools/inspect_cable_asset.py
```

## Front-plane benchmark

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
set -o pipefail
bash tools/run_benchmark.sh \
  2>&1 | tee camera_output/front_plane_benchmark_console.txt
status=${PIPESTATUS[0]}
echo "benchmark exit status: $status"
cat camera_output/front_plane_benchmark/report.txt
```

Qualification still requires at least 95% pair success, zero track switches, radial 3D jitter no greater than 0.5 mm, ray-gap p95 no greater than 0.5 mm, plane-residual p95 no greater than 0.5 mm, plane-error median no greater than 0.5 mm, and plane-error p95 no greater than 1.0 mm.

## Safety constraints and kill switch

- No manual recess or depth offset.
- No fallback to recessed-cavity depth.
- No RTX, USD mesh query, or ground-truth JSON in runtime control.
- No vision-driven orientation command.
- No insertion motion.
- No release or regrasp.
- No per-frame cable or plug transform.
- Failed observations hold the current target and trigger reacquisition.

Do not begin insertion work when the direct joint is unstable, the built-in attachment changes, the deformable tail destabilizes the arm or cameras, any mount limit fails, or nominal visual alignment no longer completes under GPU dynamics. Fix the mount or physics defect instead of weakening limits.

## Generated files

`camera_output/`, model weights, Python caches, generated ground truth, and local worktrees are ignored by Git.
