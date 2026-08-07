# Single-Rack Dead-Code Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove legacy center-estimation code from `single_rack_cv/` that is no longer part of the final plane-rectified physical-mouth runtime, without changing the known-good runtime behavior.

**Architecture:** Preserve the production chain rooted at `main.py` and `live_control_projective.py`. Trim `live_control.py` so it contains only the shared current front-plane adapter used by production, delete the isolated superseded RGB/stereo center stack and its direct regression tests, and extend repository-cleanliness checks so those paths stay removed.

**Tech Stack:** Python 3, `unittest`, NVIDIA Isaac Sim 6.0.0 runtime wiring, Git/GitHub.

## Global Constraints

- Modify or delete files only under `single_rack_cv/`.
- Do not change the final plane-rectified width-hypothesis perception behavior.
- Do not change camera transforms, calibration constants, stereo gates, handoff, insertion offsets, motion thresholds, or safety gates.
- Do not consolidate or rename the runtime wrapper chain.
- Keep benchmarks, calibration helpers, current diagnostics, and documented standalone tools unless they are proven unreachable and unsupported.
- `main.py` must continue to import `refine_live_observation` from `live_control_projective.py`.
- `live_control_projective.py` must continue to call `outer_bezel_projective_center.estimate_outer_bezel_projective_center` and `live_control.apply_front_plane_result`.

---

### Task 1: Lock the legacy center stack out with structural tests

**Files:**
- Modify: `single_rack_cv/tests/test_repo_cleanliness.py`
- Test: `single_rack_cv/tests/test_repo_cleanliness.py`

**Interfaces:**
- Consumes: repository paths under `single_rack_cv/`.
- Produces: forbidden-path assertions for the superseded center stack.

- [ ] **Step 1: Add the following paths to `FORBIDDEN_PATHS`**

```python
"front_mouth_projective_center.py",
"lower_mouth_projective_center.py",
"stereo_center_projective.py",
"stereo_front_rim_plane.py",
"stereo_center.py",
"tests/test_front_mouth_outer_edges.py",
"tests/test_lower_mouth_projective_center.py",
"tests/test_projective_front_rim_center.py",
"tests/test_stereo_front_rim_plane.py",
"tests/test_stereo_center.py",
```

- [ ] **Step 2: Run the cleanliness test before deletion**

Run:
```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" -m unittest -v tests.test_repo_cleanliness
```
Expected: FAIL because the newly forbidden legacy paths still exist.

### Task 2: Remove the dead legacy branch from `live_control.py`

**Files:**
- Modify: `single_rack_cv/live_control.py`
- Test: `single_rack_cv/tests/test_live_control.py`
- Test: `single_rack_cv/tests/test_front_rim_plane_runtime_wiring.py`

**Interfaces:**
- Consumes: `front_plane_result` from `outer_bezel_projective_center.py` through `live_control_projective.py`.
- Produces: unchanged `LiveFrontPlaneDiagnostics`, `_replace_control_center`, and `apply_front_plane_result`.

- [ ] **Step 1: Remove the unused import**

Delete:
```python
from stereo_front_rim_plane import estimate_stereo_aperture_center
```

- [ ] **Step 2: Remove the superseded functions**

Delete only:
```python
def apply_stereo_center_result(...):
    ...

def refine_live_observation(...):
    ...
```

Keep `LiveFrontPlaneDiagnostics`, `_project_camera_local`, `_camera_error_to_world`, `_replace_control_center`, and `apply_front_plane_result` unchanged.

- [ ] **Step 3: Remove only test cases in `tests/test_live_control.py` that directly exercise the deleted legacy functions**

Preserve every test of `apply_front_plane_result` and current control-center replacement behavior.

### Task 3: Delete the isolated superseded center-estimation modules and direct tests

**Files:**
- Delete: `single_rack_cv/front_mouth_projective_center.py`
- Delete: `single_rack_cv/lower_mouth_projective_center.py`
- Delete: `single_rack_cv/stereo_center_projective.py`
- Delete: `single_rack_cv/stereo_front_rim_plane.py`
- Delete: `single_rack_cv/stereo_center.py`
- Delete: `single_rack_cv/tests/test_front_mouth_outer_edges.py`
- Delete: `single_rack_cv/tests/test_lower_mouth_projective_center.py`
- Delete: `single_rack_cv/tests/test_projective_front_rim_center.py`
- Delete: `single_rack_cv/tests/test_stereo_front_rim_plane.py`
- Delete: `single_rack_cv/tests/test_stereo_center.py`

**Interfaces:**
- Consumes: none; these files are superseded by `outer_bezel_projective_center.py` + `plane_rectified_front_lip.py`.
- Produces: no runtime interface.

- [ ] **Step 1: Delete the five superseded modules and five direct tests**

- [ ] **Step 2: Search the remaining `single_rack_cv/` tree for imports/references to the deleted modules**

Expected production result: no import of any deleted module. Historical design/plan prose may still mention old names.

### Task 4: Verify the final runtime contract

**Files:**
- Verify only: `single_rack_cv/main.py`
- Verify only: `single_rack_cv/live_control_projective.py`
- Verify only: `single_rack_cv/outer_bezel_projective_center.py`
- Verify only: runtime wrapper files

- [ ] **Step 1: Run focused pure/structural tests**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" -m unittest -v \
  tests.test_repo_cleanliness \
  tests.test_live_control \
  tests.test_front_rim_plane_runtime_wiring \
  tests.test_plane_rectified_front_lip \
  tests.test_plane_rectified_runtime_wiring \
  tests.test_front_lip_left_bezel_rejection \
  tests.test_front_lip_search_calibration \
  tests.test_visible_front_lip_calibration \
  tests.test_visible_front_lip_geometry \
  tests.test_handoff_position_hold \
  tests.test_handoff_position_hold_runtime_wiring \
  tests.test_precontact_runtime_wiring \
  tests.test_startup_geometry_settle \
  tests.test_two_stage_insertion
```
Expected: `OK`.

- [ ] **Step 2: Verify Git scope**

```bash
git diff --name-only main...HEAD
```
Expected: every changed path begins with `single_rack_cv/`.

- [ ] **Step 3: Run the workstation qualification once after merge**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" main.py 2>&1 | tee camera_output/final_cleanup_qualification.txt
```
Expected: the same width-hypothesis perception behavior and guarded insertion behavior as commit `e5274d37de6d8911aba7b6fe313c6a320f6a65be`.
