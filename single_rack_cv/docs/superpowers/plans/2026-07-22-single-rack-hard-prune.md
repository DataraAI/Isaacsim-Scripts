# Single-Rack Hard-Prune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `single_rack_cv` to one canonical 1280×960 image-only front-opening controller and one strict qualification workflow without changing verified behavior.

**Architecture:** Consolidate the final SGBM and stabilized plane-fit logic into `front_plane.py`, keep only generic stereo ray geometry in `stereo_geometry.py`, and move live observation replacement into `live_control.py`. Fold the high-resolution wrapper into `config.py` and `main.py`, collapse benchmark wrappers into one supported entry point, then delete all failed and obsolete generations only after dependency guards pass.

**Tech Stack:** Python 3.12, NumPy, OpenCV SGBM, Isaac Sim 6.0, YOLOE, Lula IK, unittest, Bash.

## Global Constraints

- Runtime remains image-only; no RTX raycast, USD mesh query, or ground-truth JSON in live control.
- Camera resolution remains 1280×960.
- Translation-only stop-and-look control remains capped at 1 mm per target step.
- Hold position on detector, SGBM, or plane-fit failure.
- No manual depth offset and no cavity-depth fallback.
- No insertion command.
- Qualification gates remain unchanged.
- `camera_output/`, model weights, caches, and generated benchmark outputs remain ignored.
- Work only on `cleanup/single-rack-hard-prune`; preserve `recovery/pre-single-rack-cleanup-2026-07-22`.

---

### Task 1: Canonical configuration and runtime entry point

**Files:**
- Modify: `single_rack_cv/config.py`
- Modify: `single_rack_cv/main.py`
- Create: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Produces `CONFIG.camera.resolution == (960, 1280)` and `CONFIG.front_plane.enabled is True`.
- `main.py` imports `refine_live_observation` from `live_control.py` and applies it before motion and debug output.

- [ ] Write structural tests that require 1280×960 defaults, automatic front-plane control enabled, refinement before `observe_visual_servo`, and hold-on-failure wiring.
- [ ] Run the tests and confirm they fail because the canonical config still uses 640×480 and old module names.
- [ ] Replace `FrontRimConfig` with a minimal `FrontPlaneConfig(enabled=True)` and set camera resolution to `(960, 1280)`.
- [ ] Update `main.py` to use `CONFIG.front_plane.enabled` and the new live-control import.
- [ ] Run the structural tests and commit.

### Task 2: Consolidated stereo geometry and front-plane estimator

**Files:**
- Create: `single_rack_cv/stereo_geometry.py`
- Create: `single_rack_cv/front_plane.py`
- Create: `single_rack_cv/tests/test_front_plane.py`

**Interfaces:**
- `stereo_geometry.triangulate_pixel_pair(left_uv, right_uv, left_camera, right_camera) -> tuple[np.ndarray, float]`.
- `front_plane.estimate_front_plane(...) -> FrontPlaneResult`.
- `front_plane.compute_local_disparity(...) -> LocalDisparityResult`.

- [ ] Port final pure tests for positive/negative/large disparity, flat rejection, four-sided support, nearest cluster selection, stable final residual, fused center ray, and synthetic plane recovery.
- [ ] Confirm tests fail because consolidated modules do not exist.
- [ ] Move only ray triangulation into `stereo_geometry.py`.
- [ ] Combine dense SGBM, strict 0.5 mm ray-gap filtering, monotonic stabilized plane fitting, and fused midpoint-ray center calculation directly into `front_plane.py` without monkey-patching another module.
- [ ] Run the pure estimator tests and commit.

### Task 3: Consolidated live-control adapter

**Files:**
- Create: `single_rack_cv/live_control.py`
- Create: `single_rack_cv/tests/test_live_control.py`

**Interfaces:**
- `apply_front_plane_result(frame, observation, desired_port_virtual_camera_usd, front_plane_result)`.
- `refine_live_observation(frame, observation, desired_port_virtual_camera_usd)`.

- [ ] Port the no-offset geometry replacement test and assert the public API has no parameter containing `offset`.
- [ ] Confirm the test fails because `live_control.py` does not exist.
- [ ] Move the final adapter and diagnostics from `live_front_plane.py`, importing `estimate_front_plane` from `front_plane.py`.
- [ ] Run live-control tests and commit.

### Task 4: One supported benchmark path

**Files:**
- Create: `single_rack_cv/benchmarks/front_plane_benchmark.py`
- Create: `single_rack_cv/tools/run_benchmark_isaac.py`
- Create: `single_rack_cv/tools/run_benchmark.sh`
- Create: `single_rack_cv/tests/test_benchmark.py`
- Modify or retain as support: `single_rack_cv/benchmarks/front_rim_benchmark.py`

**Interfaces:**
- `bash tools/run_benchmark.sh` is the only supported benchmark command.
- Exit codes remain `0=qualified`, `2=unqualified`, `1=runtime failure`.

- [ ] Add tests requiring one high-resolution mode, unchanged gates, resolution validation, automatic recapture/regeneration, and summary-driven exit status.
- [ ] Confirm tests fail because the new entry points do not exist.
- [ ] Fold high-resolution validation and strict residual postprocessing into `front_plane_benchmark.py`, importing the consolidated estimator directly.
- [ ] Create a single Isaac bootstrap and shell launcher with 1280×960 renderer settings.
- [ ] Run benchmark structural tests and commit.

### Task 5: Ground-truth and dataset tools

**Files:**
- Rename/create: `single_rack_cv/benchmarks/capture_dataset.py`
- Rename/create: `single_rack_cv/tools/generate_ground_truth.py`
- Update: `single_rack_cv/tools/run_benchmark.sh`
- Create: `single_rack_cv/tests/test_ground_truth.py`

**Interfaces:**
- Dataset manifest records `[960, 1280]`.
- Ground-truth JSON records `camera_resolution_height_width: [960, 1280]` before SimulationApp shutdown.

- [ ] Port metadata and RTX-control-forbidden tests.
- [ ] Confirm tests fail against the new file names.
- [ ] Move the current working capture and ground-truth bootstrap behavior to the canonical names.
- [ ] Update benchmark launcher references and commit.

### Task 6: Dependency guard, README, and hard deletion

**Files:**
- Create: `single_rack_cv/README.md`
- Create: `single_rack_cv/tests/test_repo_cleanliness.py`
- Modify: `.gitignore`
- Delete obsolete runtime, benchmark, test, and historical planning files listed in the approved spec.

**Interfaces:**
- Production imports may reference only canonical modules.
- README contains exact runtime, test, benchmark, recovery, gates, and safety commands.

- [ ] Add a guard test that rejects imports of `front_rim`, `front_rim_match`, `front_rim_stereo`, `front_rim_sgbm`, `front_rim_sgbm_refined`, `live_front_plane`, and `highres_config` from surviving production files.
- [ ] Add README and strengthen ignore rules for generated output, weights, caches, and local worktrees.
- [ ] Delete failed Sobel, sparse matcher, diagnostic benchmark wrappers, duplicate launchers, obsolete tests, and historical plans/specs.
- [ ] Run all pure and structural tests.
- [ ] Search the branch for legacy imports and commit.

### Task 7: Qualification and live verification

**Files:**
- Verification only.

- [ ] Run `python -m unittest -v tests.test_front_plane tests.test_live_control tests.test_runtime_wiring tests.test_benchmark tests.test_ground_truth tests.test_repo_cleanliness`.
- [ ] Run `bash tools/run_benchmark.sh`; require `QUALIFIED=true` and exit status 0.
- [ ] Run `~/isaacsim/python.sh main.py`; require `[LIVE FRONT PLANE]`, no DLSS minimum-resolution warning, marker on the opening, no insertion command, and physical tracking error within the existing tolerance.
- [ ] Compare cleanup branch against `main`, review changed files, and only then present merge options.
