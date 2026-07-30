# Camera-Derived 6-DoF Port Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the same sub-millimeter insertion accuracy from horizontal and angled eye-in-hand views by estimating and controlling the complete RJ45 port pose from synchronized stereo RGB.

**Architecture:** Replace the cavity-centroid control point with four fitted inner-rim lines, four ordered stereo corners, and a right-handed port frame. Pass that camera-derived pose through temporal stability gates, bounded translation-and-orientation motion, a frozen 6-DoF handoff, and insertion along the frozen port normal.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Isaac Sim 6.0.0, Lula IK, USD guide geometry, `unittest`.

## Global Constraints

- No empirical X/Y/Z correction vectors.
- No rack transform, port prim, USD geometry, RTX ray hits, or benchmark ground truth in runtime control.
- One estimator/controller path for horizontal and angled views; no per-view modes or parameters.
- Keep 50 mm pre-insert standoff, existing two-stage distances, 0.5 mm lateral limit, and 1.0 degree orientation limit.
- Center error <= 0.5 mm; normal and roll/up-axis errors <= 0.5 degrees.
- Temporal center spread <= 0.5 mm; each axis spread <= 0.25 degrees; width/height spread <= 0.5 mm.
- Angled degradation relative to horizontal <= 0.25 mm center and <= 0.25 degrees per axis.
- Invalid sides, corner order, triangulation, handedness, dimensions, residual, ray gap, or reprojection must hold and reacquire; never fall back to the old centroid.
- Ground truth is benchmark-only and is read only after an estimate exists.
- Begin from the last workstation baseline that passes the pure suite and completes 48/48 insertion commands. The grip-lift and camera-compensation experiments are excluded.

---

## File Map

- Create `single_rack_cv/port_pose.py`: projective rim geometry, ordered stereo corners, right-handed port frame, ToolCenter quaternion.
- Modify `single_rack_cv/front_plane.py`: keep dense SGBM plane validation; replace cavity-centroid/bounding-box center geometry with corner geometry.
- Modify `single_rack_cv/perception.py` and `single_rack_cv/live_control.py`: carry complete port pose and desired ToolCenter pose.
- Modify `single_rack_cv/stereo_handoff.py`: temporal pose stability and bounded quaternion stepping.
- Modify `single_rack_cv/stereo_handoff_runtime.py`: full 6-DoF servo, freeze, handoff, and settle checks.
- Modify `single_rack_cv/plug_axis_insertion.py` and `single_rack_cv/angled_hand_runtime.py`: freeze camera-derived orientation and insertion axis together.
- Modify `single_rack_cv/config.py` and `single_rack_cv/debug.py`: view-independent pose gates and precise guide markers.
- Create `single_rack_cv/benchmarks/port_pose_benchmark.py`: horizontal-versus-angled benchmark-only scoring.

---

### Task 1: Restore and Snapshot the Qualified Baseline

**Files:** current `single_rack_cv` workstation tree.

**Produces:** `feature/camera-derived-port-pose` starting from code that completes 48/48 commands.

- [ ] Reverse the two experimental patches when present:

```bash
cd ~/Isaacsim-Scripts
git apply -R --check ~/Downloads/rj45_camera_to_tool_extrinsic_compensation.patch \
  && git apply -R ~/Downloads/rj45_camera_to_tool_extrinsic_compensation.patch \
  || true
git apply -R --check ~/Downloads/rj45_grip_presentation_lift.patch \
  && git apply -R ~/Downloads/rj45_grip_presentation_lift.patch \
  || true
git diff --check
```

- [ ] Run the pure suite:

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh -m unittest discover -s tests -t . -p 'test_*.py' -v
```

Expected: zero failures/errors. Reconcile the exact count before proceeding.

- [ ] Run Isaac Sim and retain the log only if mount validation passes and all 48 commands complete:

```bash
~/isaacsim/python.sh main.py 2>&1 | tee camera_output/port_pose_baseline_console.txt
```

- [ ] Commit the qualified workstation snapshot and branch from it:

```bash
cd ~/Isaacsim-Scripts
git add single_rack_cv
git commit -m 'chore: checkpoint qualified angled-hand insertion baseline'
git switch -c feature/camera-derived-port-pose
```

---

### Task 2: Fit a Perspective-Correct Inner-Rim Quadrilateral

**Files:**
- Create `single_rack_cv/port_pose.py`
- Create `single_rack_cv/tests/test_port_pose.py`

**Interface:**

```python
@dataclass(frozen=True)
class ImagePortQuadrilateral:
    corners_uv: np.ndarray          # TL, TR, BR, BL
    center_uv: np.ndarray           # diagonal intersection
    side_support_counts: tuple[int, int, int, int]


def extract_inner_rim_quadrilateral(
    mask: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
) -> ImagePortQuadrilateral: ...
```

- [ ] Write tests proving an oblique quadrilateral’s diagonal intersection is stable when asymmetric interior pixels move the mask centroid; require TL/TR/BR/BL order; reject missing side support, parallel adjacent lines, self-intersection, and degenerate area.
- [ ] Run `~/isaacsim/python.sh -m unittest -v tests.test_port_pose.ImageQuadrilateralTests`; verify failure because the module/API does not exist.
- [ ] Implement: largest refined cavity contour; classify contour pixels into top/right/bottom/left by normalized direction from the contour median; require at least 8 pixels per side; total-least-squares line fit; adjacent line intersections; convex/order validation; diagonal intersection for `center_uv`.
- [ ] Run the focused tests and commit:

```bash
git add single_rack_cv/port_pose.py single_rack_cv/tests/test_port_pose.py
git commit -m 'feat: fit perspective-correct inner port rim'
```

---

### Task 3: Reconstruct the Right-Handed 3D Port Frame

**Files:**
- Modify `single_rack_cv/port_pose.py`
- Modify `single_rack_cv/tests/test_port_pose.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PortFrame:
    center_world_m: np.ndarray
    outward_normal_world: np.ndarray
    insertion_direction_world: np.ndarray
    horizontal_world: np.ndarray
    vertical_world: np.ndarray
    corners_world_m: np.ndarray
    width_m: float
    height_m: float
    tool_orientation_wxyz: np.ndarray
    max_ray_gap_m: float
    reprojection_rms_px: float
    max_reprojection_px: float
    plane_residual_m: float


def triangulate_port_frame(
    left_corners_uv: np.ndarray,
    right_corners_uv: np.ndarray,
    left_camera,
    right_camera,
) -> PortFrame: ...
```

- [ ] Write synthetic horizontal and oblique stereo tests for the same known rectangle. Require center agreement <= 0.5 mm and axis agreement <= 0.25 degrees.
- [ ] Verify the tests fail on missing APIs.
- [ ] Triangulate TL/TR/BR/BL independently using `stereo_geometry.triangulate_pixel_pair`; collect ray gaps and reprojection errors.
- [ ] Build axes:

```python
center = corners.mean(axis=0)
horizontal = unit_vector(0.5 * ((tr + br) - (tl + bl)), 'horizontal')
vertical_raw = 0.5 * ((tl + tr) - (bl + br))
vertical = unit_vector(vertical_raw - horizontal * (vertical_raw @ horizontal), 'vertical')
insertion = unit_vector(np.cross(horizontal, vertical), 'insertion')
outward = -insertion
```

Flip horizontal/insertion/outward together when outward does not face the stereo midpoint. Recompute `vertical = normalize(cross(insertion, horizontal))`; require determinant >= 0.999999.
- [ ] Use ToolCenter local `+X=horizontal`, `+Y=vertical`, `+Z=insertion`; convert this rotation matrix to normalized `wxyz` with nonnegative `w`.
- [ ] Reject non-finite corners, width/height outside configured physical ranges, ray gap above 0.5 mm, reprojection above existing gates, or plane residual above 0.5 mm.
- [ ] Run `tests.test_port_pose` and commit:

```bash
git add single_rack_cv/port_pose.py single_rack_cv/tests/test_port_pose.py
git commit -m 'feat: reconstruct camera-derived 3D port frame'
```

---

### Task 4: Replace Centroid Geometry in the Live Front-Plane Path

**Files:**
- Modify `single_rack_cv/front_plane.py`
- Modify `single_rack_cv/perception.py`
- Modify `single_rack_cv/live_control.py`
- Modify `single_rack_cv/tests/test_front_plane.py`
- Modify `single_rack_cv/tests/test_live_control.py`

**Produces:** ordered image corners, complete port axes, and desired ToolCenter pose in each accepted observation.

- [ ] Add failing tests showing two masks with identical rim edges but different interior cutouts return the same 3D center/orientation. Add live-control tests for `desired_tool_position_world_m` and `desired_tool_orientation_wxyz`.
- [ ] Extend `FrontPlaneResult` with `left_corners_uv`, `right_corners_uv`, `horizontal_world`, `vertical_world`, `outward_normal_world`, `insertion_direction_world`, and `tool_orientation_wxyz`.
- [ ] Extend `estimate_front_plane` to accept `left_mask` and `right_mask`. Keep dense SGBM ring clustering and plane fitting as an independent validation signal.
- [ ] Replace `intersect_midpoint_ray_with_plane` and `_bbox_corners` as runtime center sources with:

```python
left_quad = extract_inner_rim_quadrilateral(left_mask, left_bbox_xywh)
right_quad = extract_inner_rim_quadrilateral(right_mask, right_bbox_xywh)
port_frame = triangulate_port_frame(
    left_quad.corners_uv,
    right_quad.corners_uv,
    left_camera,
    right_camera,
)
```

Require the corner-frame normal and dense SGBM normal to agree within 0.5 degrees. Do not translate corners to the old centroid ray.
- [ ] Extend `StereoPortObservation` with complete axes plus:

```python
desired_tool_position_world_m: np.ndarray
desired_tool_orientation_wxyz: np.ndarray
```

- [ ] In `apply_front_plane_result`, compute only:

```python
desired_tool_position = center_world + preinsert_standoff_m * outward_normal_world
desired_tool_orientation = tool_orientation_wxyz
```

The API must contain no argument whose name includes `offset`.
- [ ] Run focused tests:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_port_pose tests.test_front_plane tests.test_live_control
```

- [ ] Commit:

```bash
git add single_rack_cv/front_plane.py single_rack_cv/perception.py \
  single_rack_cv/live_control.py single_rack_cv/tests/test_front_plane.py \
  single_rack_cv/tests/test_live_control.py
git commit -m 'feat: derive live port center and axes from stereo rim corners'
```

---

### Task 5: Add Temporal Pose Stability and Bounded Orientation Motion

**Files:**
- Modify `single_rack_cv/stereo_handoff.py`
- Modify `single_rack_cv/tests/test_stereo_handoff.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PortPoseSample:
    center_world_m: np.ndarray
    desired_tool_position_world_m: np.ndarray
    outward_normal_world: np.ndarray
    insertion_direction_world: np.ndarray
    horizontal_world: np.ndarray
    vertical_world: np.ndarray
    tool_orientation_wxyz: np.ndarray
    width_m: float
    height_m: float
    plane_residual_m: float
    max_ray_gap_m: float
    reprojection_rms_px: float

@dataclass(frozen=True)
class StablePortPoseEstimate: ...
@dataclass(frozen=True)
class PoseHandoffDecision: ...

def estimate_stable_port_pose(samples, *, minimum_samples,
    maximum_center_spread_m, maximum_axis_spread_deg,
    maximum_dimension_spread_m) -> StablePortPoseEstimate | None: ...

def bounded_orientation_step(current_wxyz, target_wxyz,
    maximum_step_deg: float) -> tuple[np.ndarray, float]: ...
```

- [ ] Write failing tests: stable center/axes/dimensions accepted; stable center with unstable normal rejected; width/height spread rejected; mirrored handedness rejected; a 30-degree quaternion target produces exactly a 1-degree bounded step.
- [ ] Implement normalized-mean axes, maximum angular spread from each mean, right-handed frame rebuild, quaternion rebuild, shortest-path SLERP, and newest-three-sample selection.
- [ ] `select_recent_bounded_pose` requires translation <= 35 mm and remaining orientation <= 2 degrees before freezing.
- [ ] Run `tests.test_stereo_handoff`; preserve old bounded-translation tests.
- [ ] Commit:

```bash
git add single_rack_cv/stereo_handoff.py single_rack_cv/tests/test_stereo_handoff.py
git commit -m 'feat: add stable 6-DoF stereo handoff math'
```

---

### Task 6: Implement Frozen 6-DoF Runtime Handoff

**Files:**
- Modify `single_rack_cv/config.py`
- Modify `single_rack_cv/stereo_handoff_runtime.py`
- Modify `single_rack_cv/tests/test_stereo_handoff_runtime_wiring.py`
- Modify `single_rack_cv/tests/test_runtime_wiring.py`

**Configuration:**

```python
maximum_pose_center_spread_m: float = 0.0005
maximum_pose_axis_spread_deg: float = 0.25
maximum_pose_dimension_spread_m: float = 0.0005
max_orientation_step_deg: float = 1.0
settle_orientation_tolerance_deg: float = 0.25
maximum_handoff_orientation_deg: float = 2.0
```

- [ ] Write failing wiring tests requiring `estimate_stable_port_pose`, `bounded_orientation_step`, `_handoff_pose`, `desired_tool_orientation_wxyz`, and the log text `insertion axis: frozen camera-derived port normal`. Reject `orientation: unchanged horizontal plug`.
- [ ] Do not call the base translation-only `observe_visual_servo`. Update detection references and collect `PortPoseSample` objects directly.
- [ ] Accept a track only after three complete poses satisfy all temporal and per-frame gates.
- [ ] Before freeze, publish bounded position and quaternion steps toward the stable camera-derived desired ToolCenter pose.
- [ ] Freeze center, axes, desired ToolCenter position, desired ToolCenter quaternion, width, height, and quality metrics together.
- [ ] During handoff and final settling, require both position error <= existing settle position tolerance and orientation error <= 0.25 degrees. Use `insertion.quaternion_angular_error_deg`; do not subtract Euler angles.
- [ ] Log center spread, all axis spreads, dimension spreads, remaining translation/orientation, frozen center/normal, final position error, and final orientation error.
- [ ] Run focused runtime tests and commit:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_stereo_handoff tests.test_stereo_handoff_runtime_wiring \
  tests.test_runtime_wiring

git add single_rack_cv/config.py single_rack_cv/stereo_handoff_runtime.py \
  single_rack_cv/tests/test_stereo_handoff_runtime_wiring.py \
  single_rack_cv/tests/test_runtime_wiring.py
git commit -m 'feat: hand off camera-derived ToolCenter pose'
```

---

### Task 7: Freeze Camera-Derived Insertion Orientation and Axis

**Files:**
- Modify `single_rack_cv/plug_axis_insertion.py`
- Modify `single_rack_cv/angled_hand_runtime.py`
- Modify `single_rack_cv/stereo_handoff_runtime.py`
- Modify related insertion/runtime tests.

**Interface:**

```python
class ExplicitInsertionPoseAdapter:
    def set_pose(self, axis_world, orientation_wxyz) -> None: ...
```

- [ ] Write failing tests proving the adapter overrides `controller.axis_world` and `controller.frozen_orientation_wxyz` together at `_freeze_from`; reject missing/nonfinite/zero-length pose values. Keep the existing axis-only adapter tests green.
- [ ] Retain `ExplicitInsertionAxisAdapter` for existing callers; add `ExplicitInsertionPoseAdapter` using the same axis validation and normalized sign-stable quaternion validation.
- [ ] In `AngledHandCableRuntime`, add `_requested_insertion_pose()`. The baseline implementation returns the live plug axis and current target orientation. `_partial_insertion_sample()` calls the pose adapter before delegating.
- [ ] In `AngledHandStereoHandoffRuntime`, override `_requested_insertion_pose()` to return the frozen camera-derived insertion direction and frozen ToolCenter quaternion after handoff.
- [ ] Do not change `PartialInsertionController` distances, settle counts, drift limit, orientation limit, timeouts, or terminal holds.
- [ ] Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_plug_axis_insertion tests.test_partial_insertion \
  tests.test_two_stage_insertion tests.test_angled_hand_runtime_wiring \
  tests.test_stereo_handoff_runtime_wiring
```

- [ ] Commit:

```bash
git add single_rack_cv/plug_axis_insertion.py \
  single_rack_cv/angled_hand_runtime.py single_rack_cv/stereo_handoff_runtime.py \
  single_rack_cv/tests/test_plug_axis_insertion.py \
  single_rack_cv/tests/test_angled_hand_runtime_wiring.py \
  single_rack_cv/tests/test_stereo_handoff_runtime_wiring.py
git commit -m 'feat: freeze camera-derived insertion pose'
```

---

### Task 8: Add Precise Pose Guides and a Dual-View Camera-Only Benchmark

**Files:**
- Modify `single_rack_cv/config.py`
- Modify `single_rack_cv/debug.py`
- Create `single_rack_cv/benchmarks/port_pose_benchmark.py`
- Modify benchmark, ground-truth, and cleanliness tests.

- [ ] Add failing source-structure tests requiring a 1 mm center marker, 20 mm axes, all three estimated axes, explicit numerical gates, and benchmark-only `control_usage`; reject any `manual_port_offset` symbol.
- [ ] Set:

```python
estimated_port_marker_radius_m: float = 0.001
estimated_port_axis_length_m: float = 0.020
```

Draw red horizontal, green vertical, and blue outward-normal `UsdGeom.BasisCurves` with purpose `guide`. Debug prims must never be read by control.
- [ ] Implement `benchmarks/port_pose_benchmark.py`: load frozen horizontal and angled stereo datasets; run the normal perception/refinement path; read truth only afterward; score center, normal, horizontal, vertical, temporal spreads, dimensions, ray gap, reprojection, and plane residual; write JSON/CSV; exit 1 on any failed gate.
- [ ] Use exact constants:

```python
maximum_center_error_mm = 0.5
maximum_axis_error_deg = 0.5
maximum_center_spread_mm = 0.5
maximum_axis_spread_deg = 0.25
maximum_dimension_spread_mm = 0.5
maximum_view_degradation_center_mm = 0.25
maximum_view_degradation_axis_deg = 0.25
```

- [ ] Run benchmark/ground-truth/cleanliness tests and commit:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_benchmark tests.test_ground_truth tests.test_repo_cleanliness

git add single_rack_cv/config.py single_rack_cv/debug.py \
  single_rack_cv/benchmarks/port_pose_benchmark.py \
  single_rack_cv/tests/test_benchmark.py \
  single_rack_cv/tests/test_ground_truth.py \
  single_rack_cv/tests/test_repo_cleanliness.py
git commit -m 'test: qualify camera-derived pose across view angles'
```

---

### Task 9: Full Verification and Kill-Switch Decision

- [ ] Run the complete pure suite:

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh -m unittest discover -s tests -t . -p 'test_*.py' -v \
  2>&1 | tee camera_output/camera_derived_pose_tests.txt
```

Expected: zero failures/errors. Record the exact new count.

- [ ] Run the dual-view benchmark:

```bash
~/isaacsim/python.sh benchmarks/port_pose_benchmark.py \
  2>&1 | tee camera_output/port_pose_benchmark_console.txt
```

Expected: exit 0; both views and degradation limits pass.

- [ ] Run horizontal and angled Isaac Sim qualifications with identical estimator/controller parameters. Each must pass mount validation, settle position and orientation, complete 48/48 commands, reach approximately +10 mm, remain below 0.5 mm lateral drift and below 1 degree orientation error.
- [ ] Do not merge when any result needs an empirical correction, any center/axis gate fails, angled degradation exceeds its limit, a safety limit must be loosened, runtime reads ground truth, or either view fails 48/48.
- [ ] Update `README.md` only with measured test counts, benchmark numbers, commands, and workstation results. Do not claim physical collision validity.
- [ ] Commit verified documentation:

```bash
git add single_rack_cv/README.md \
  single_rack_cv/docs/superpowers/plans/2026-07-29-camera-derived-port-pose.md
git commit -m 'docs: record camera-derived port pose qualification'
```

## Self-Review

- Tasks 2–4 cover four per-eye rim sides, ordered corners, stereo triangulation, center, axes, handedness, and removal of the centroid fallback.
- Tasks 5–7 cover temporal pose gates, bounded 6-DoF motion, frozen pose handoff, and insertion along the frozen camera-derived normal.
- Task 8 covers precise diagnostics and benchmark-only truth scoring.
- Task 9 enforces both-view qualification and the kill switch.
- Physical rack collision enforcement remains outside this plan.
