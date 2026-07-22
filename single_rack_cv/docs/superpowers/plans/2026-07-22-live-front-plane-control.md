# Live Front-Plane Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the 1280×960 live visual servo automatically control to the physical RJ45 front-opening plane instead of the recessed dark cavity center.

**Architecture:** Keep YOLOE cavity detection and stereo pairing unchanged. After a valid cavity observation is selected, run the already-qualified refined local SGBM estimator on the same stationary stereo pair, replace the observation’s 3D center/range/correction with the fitted front-plane intersection, then pass that refined observation to the existing stop-and-look controller and debug marker. Enable this path only in `main_highres.py` through `highres_config.py`; the canonical 640×480 runner remains unchanged.

**Tech Stack:** Python 3, NumPy, OpenCV SGBM, Isaac Sim 6.0, existing Lula IK stop-and-look controller.

## Global Constraints

- Runtime control remains image-only; no RTX raycast, USD mesh query, or ground-truth JSON.
- No manual depth offset or fixed recess compensation.
- Keep translation-only motion, fixed wrist orientation, and 1 mm maximum target step.
- Hold position on any detector, SGBM, or geometry failure.
- Do not command insertion.
- Do not commit `camera_output/`.

---

### Task 1: Pure front-plane observation adapter

**Files:**
- Create: `single_rack_cv/live_front_plane.py`
- Create: `single_rack_cv/tests/test_live_front_plane.py`

**Interfaces:**
- Consumes: selected `StereoPortObservation`, `StereoFrame`, desired virtual-camera point, and `SGBMFrontPlaneResult`.
- Produces: `apply_front_plane_result(...) -> tuple[StereoPortObservation, LiveFrontPlaneDiagnostics]` and `refine_live_observation_to_front_plane(...)`.

- [ ] Write a failing test proving a 140 mm cavity observation is replaced by a 130 mm front-plane result with zero correction when 130 mm is desired.
- [ ] Assert the public adapter API contains no offset parameter.
- [ ] Implement geometry replacement with `dataclasses.replace`.
- [ ] Run `python -m unittest -v tests.test_live_front_plane`.

### Task 2: High-resolution live wiring

**Files:**
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/highres_config.py`
- Create: `single_rack_cv/tests/test_live_front_plane_wiring.py`

**Interfaces:**
- `main.py` conditionally calls `refine_live_observation_to_front_plane(...)` when `CONFIG.front_rim.enabled` is true.
- `highres_config.py` sets only the high-resolution config’s `front_rim.enabled=True`.

- [ ] Write structural tests proving refinement occurs before motion/debug and high-res enables the path.
- [ ] Import the adapter only after Isaac starts.
- [ ] Replace the observation before `runtime.observe_visual_servo(observation)`.
- [ ] Print cavity range, opening range, measured recess, SGBM counts, residual, and ray gap.
- [ ] Preserve the existing exception path so any refinement failure calls `runtime.note_perception_failure()` and holds position.
- [ ] Run the pure and structural tests.

### Task 3: Verification

**Files:**
- Verify only; no production changes.

- [ ] Run Python syntax checks on `live_front_plane.py`, `main.py`, and `highres_config.py`.
- [ ] Run the existing SGBM refined tests plus the two new tests.
- [ ] Run `main_highres.py` and confirm logs contain `[LIVE FRONT PLANE]` before each `[RGB STEREO SERVO]` line.
- [ ] Confirm the estimated-port sphere lies on the front opening plane and the final ToolCenter remains 50 mm from that plane.
- [ ] Confirm no insertion command is issued.
