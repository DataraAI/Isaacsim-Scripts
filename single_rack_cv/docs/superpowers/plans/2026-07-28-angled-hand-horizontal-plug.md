# Angled Hand with Horizontal RJ45 Plug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pitch the Franka hand downward by 30 degrees from the robot-right-side view while keeping the rigid RJ45 plug horizontal and preserving the qualified RGB-stereo alignment and 48-command partial insertion.

**Architecture:** Keep `/World/IK_Target` and `/World/ToolCenter` as the horizontal plug-tip frame. Encode the inverse 30 degree hand pitch in the fixed `hand_T_tool` local rotation so Lula derives a pitched `panda_hand` pose while the plug frame remains horizontal. During insertion, freeze the live PhysX plug nose axis explicitly instead of deriving travel from ToolCenter local +Z.

**Tech Stack:** Python 3, NumPy, Isaac Sim 6.0.0, USD/PhysX, Lula IK, `unittest`, existing RGB stereo + YOLOE pipeline.

## Global Constraints

- Base branch and rollback point: validated `main` at squash commit `d1ca7ad5dedb0e017eba9fd5dd731ad578a0c7a5`.
- Working branch: `feature/angled-hand-horizontal-plug`.
- Accepted pitch: exactly 30 degrees downward from the robot-right-side view.
- Direction contract: wrist higher, fingertips lower toward the port.
- Supported configured range: 0 through 45 degrees; reject negative, non-finite, and greater-than-45 values.
- Rigid plug nose/body remain horizontal; flexible cable tail remains deformable and unconstrained.
- Cameras remain rigid children of `panda_hand`; do not create independent camera pose overrides.
- Visual servo remains translation-only and live-image-only; no manual pixel/depth offsets and no RTX/USD ground-truth runtime control.
- Existing insertion remains 40 mm coarse in eight 5 mm commands plus 20 mm fine in forty 0.5 mm commands, finishing 10 mm inside the opening.
- Do not relax the existing 0.5 mm mount-tip, 1 degree plug-axis, 0.3 mm final settle, 0.5 mm lateral-drift, 1 degree hand-orientation, or 2 second step-timeout limits.
- No seating, latch engagement, release, retreat, or tail-routing changes in this milestone.
- Do not touch the user's unrelated local deletion of `.vscode/settings.json`.
- Use the existing filenames; do not add cache-avoidance version suffixes.

---

## File Structure

- Create `single_rack_cv/hand_plug_geometry.py`: pure NumPy validation, pitched local-transform construction, and geometry measurements.
- Modify `single_rack_cv/config.py`: expose one shared `hand_downward_pitch_deg` value.
- Modify `single_rack_cv/cable_runtime/__init__.py`: apply the pitched local transform before scene construction, read live plug pose/axis once, enforce startup geometry, and feed the plug axis to insertion.
- Modify `single_rack_cv/insertion.py`: accept and freeze an explicit insertion axis from the runtime sample.
- Modify `single_rack_cv/tests/test_partial_insertion.py`: update sample construction and prove explicit-axis behavior.
- Modify `single_rack_cv/tests/test_two_stage_insertion.py`: update two-stage samples to provide the explicit axis.
- Create `single_rack_cv/tests/test_hand_plug_geometry.py`: pure transform, sign, range, and horizontal-axis tests.
- Create `single_rack_cv/tests/test_angled_hand_runtime_wiring.py`: structural wiring and unchanged-safety-contract tests.
- Modify `single_rack_cv/README.md`: document the side-view convention, separate hand/plug frames, and qualification command.

---

### Task 1: Pure Hand-to-Plug Geometry Contract

**Files:**
- Create: `single_rack_cv/hand_plug_geometry.py`
- Create: `single_rack_cv/tests/test_hand_plug_geometry.py`

**Interfaces:**
- Consumes: one zero-pitch `hand_from_tool` 3x3 rotation and one configured downward pitch in degrees.
- Produces:
  - `validate_downward_hand_pitch_deg(value: float, maximum_deg: float = 45.0) -> float`
  - `compute_pitched_hand_from_tool_rotation(base_hand_from_tool: np.ndarray, downward_pitch_deg: float) -> np.ndarray`
  - `horizontal_axis_error_deg(axis_world: np.ndarray) -> float`
  - `measure_hand_plug_geometry(*, hand_position_m: np.ndarray, hand_rotation_world: np.ndarray, plug_tip_position_m: np.ndarray, plug_axis_world: np.ndarray) -> HandPlugGeometryMetrics`
  - `HandPlugGeometryMetrics(relative_pitch_deg: float, wrist_above_tip_m: float, plug_horizontal_error_deg: float, wrist_higher_fingertips_lower: bool)`

- [ ] **Step 1: Write failing tests for accepted and rejected pitch values**

```python
class PitchValidationTests(unittest.TestCase):
    def test_accepts_zero_through_forty_five_degrees(self):
        self.assertEqual(validate_downward_hand_pitch_deg(0.0), 0.0)
        self.assertEqual(validate_downward_hand_pitch_deg(30.0), 30.0)
        self.assertEqual(validate_downward_hand_pitch_deg(45.0), 45.0)

    def test_rejects_negative_nonfinite_and_above_forty_five(self):
        for value in (-0.001, 45.001, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_downward_hand_pitch_deg(value)
```

- [ ] **Step 2: Write failing transform tests**

```python
class PitchedTransformTests(unittest.TestCase):
    def test_zero_pitch_is_exact_identity_change(self):
        base = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        actual = compute_pitched_hand_from_tool_rotation(base, 0.0)
        np.testing.assert_allclose(actual, base, atol=1.0e-12)

    def test_thirty_degree_pitch_is_base_times_negative_local_x_rotation(self):
        base = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        radians = math.radians(-30.0)
        expected_delta = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(radians), -math.sin(radians)],
                [0.0, math.sin(radians), math.cos(radians)],
            ],
            dtype=np.float64,
        )
        actual = compute_pitched_hand_from_tool_rotation(base, 30.0)
        np.testing.assert_allclose(actual, base @ expected_delta, atol=1.0e-12)
```

- [ ] **Step 3: Write failing directional and horizontal tests**

```python
class GeometryMeasurementTests(unittest.TestCase):
    def test_requested_sign_is_wrist_higher_and_fingertips_lower(self):
        pitch = math.radians(30.0)
        hand_rotation = np.array(
            [
                [-0.5, 0.0, -math.cos(pitch)],
                [0.0, 1.0, 0.0],
                [math.cos(pitch), 0.0, -math.sin(pitch)],
            ],
            dtype=np.float64,
        )
        metrics = measure_hand_plug_geometry(
            hand_position_m=np.array([0.8, 0.0, 1.4]),
            hand_rotation_world=hand_rotation,
            plug_tip_position_m=np.array([0.7, 0.0, 1.3333]),
            plug_axis_world=np.array([-1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(metrics.relative_pitch_deg, 30.0, places=9)
        self.assertGreater(metrics.wrist_above_tip_m, 0.0)
        self.assertTrue(metrics.wrist_higher_fingertips_lower)
        self.assertAlmostEqual(metrics.plug_horizontal_error_deg, 0.0, places=12)

    def test_opposite_sign_fails_direction_contract(self):
        hand_rotation = np.eye(3, dtype=np.float64)
        metrics = measure_hand_plug_geometry(
            hand_position_m=np.array([0.8, 0.0, 1.2]),
            hand_rotation_world=hand_rotation,
            plug_tip_position_m=np.array([0.7, 0.0, 1.3]),
            plug_axis_world=np.array([0.0, 0.0, 1.0]),
        )
        self.assertFalse(metrics.wrist_higher_fingertips_lower)
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run:

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh -m unittest -v tests.test_hand_plug_geometry
```

Expected: `ModuleNotFoundError: No module named 'hand_plug_geometry'`.

- [ ] **Step 5: Implement the pure geometry module**

```python
#!/usr/bin/env python3
"""Pure geometry for a pitched Franka hand carrying a horizontal plug."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

_EPS = 1.0e-12


def _rotation3(value, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-9):
        raise ValueError(f"{label} must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{label} must have determinant +1")
    return matrix.copy()


def _axis3(value, *, label: str) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64).reshape(-1)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError(f"{label} must be a finite length-3 vector")
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        raise ValueError(f"{label} cannot be zero")
    return axis / norm


def validate_downward_hand_pitch_deg(
    value: float,
    maximum_deg: float = 45.0,
) -> float:
    pitch = float(value)
    maximum = float(maximum_deg)
    if not math.isfinite(pitch) or pitch < 0.0 or pitch > maximum:
        raise ValueError(
            f"downward hand pitch must be finite in [0, {maximum}], got {pitch}"
        )
    return pitch


def compute_pitched_hand_from_tool_rotation(
    base_hand_from_tool: np.ndarray,
    downward_pitch_deg: float,
) -> np.ndarray:
    base = _rotation3(base_hand_from_tool, label="base_hand_from_tool")
    pitch = math.radians(validate_downward_hand_pitch_deg(downward_pitch_deg))
    angle = -pitch
    delta = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )
    return _rotation3(base @ delta, label="pitched_hand_from_tool")


def horizontal_axis_error_deg(axis_world: np.ndarray) -> float:
    axis = _axis3(axis_world, label="axis_world")
    return math.degrees(math.asin(float(np.clip(abs(axis[2]), 0.0, 1.0))))


@dataclass(frozen=True)
class HandPlugGeometryMetrics:
    relative_pitch_deg: float
    wrist_above_tip_m: float
    plug_horizontal_error_deg: float
    wrist_higher_fingertips_lower: bool


def measure_hand_plug_geometry(
    *,
    hand_position_m: np.ndarray,
    hand_rotation_world: np.ndarray,
    plug_tip_position_m: np.ndarray,
    plug_axis_world: np.ndarray,
) -> HandPlugGeometryMetrics:
    hand_position = np.asarray(hand_position_m, dtype=np.float64).reshape(3)
    tip_position = np.asarray(plug_tip_position_m, dtype=np.float64).reshape(3)
    hand_rotation = _rotation3(hand_rotation_world, label="hand_rotation_world")
    plug_axis = _axis3(plug_axis_world, label="plug_axis_world")
    hand_forward = _axis3(hand_rotation[:, 2], label="hand_forward_axis")
    dot = float(np.clip(np.dot(hand_forward, plug_axis), -1.0, 1.0))
    relative_pitch_deg = math.degrees(math.acos(dot))
    wrist_above_tip_m = float(hand_position[2] - tip_position[2])
    sign_ok = wrist_above_tip_m > 0.0 and hand_forward[2] < plug_axis[2]
    return HandPlugGeometryMetrics(
        relative_pitch_deg=relative_pitch_deg,
        wrist_above_tip_m=wrist_above_tip_m,
        plug_horizontal_error_deg=horizontal_axis_error_deg(plug_axis),
        wrist_higher_fingertips_lower=sign_ok,
    )
```

- [ ] **Step 6: Run the geometry tests and verify they pass**

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_hand_plug_geometry
```

Expected: all tests pass.

- [ ] **Step 7: Commit the pure geometry contract**

```bash
git add single_rack_cv/hand_plug_geometry.py \
        single_rack_cv/tests/test_hand_plug_geometry.py
git commit -m "feat: define angled hand plug geometry"
```

---

### Task 2: Shared Pitch Configuration and Pitched IK Transform

**Files:**
- Modify: `single_rack_cv/config.py:177-205`
- Modify: `single_rack_cv/cable_runtime/__init__.py:53-76`
- Create: `single_rack_cv/tests/test_angled_hand_runtime_wiring.py`

**Interfaces:**
- Consumes: `compute_pitched_hand_from_tool_rotation(...)` and `CableMountConfig.hand_downward_pitch_deg`.
- Produces: a runtime `cfg.ik.tool_center_local_orientation_wxyz` whose `hand_T_tool` rotation counter-pitches the plug frame by the configured angle before `SimulationRuntime` builds the scene.

- [ ] **Step 1: Write the failing configuration test**

```python
class AngledHandConfigurationTests(unittest.TestCase):
    def test_single_shared_pitch_defaults_to_thirty_degrees(self):
        cfg = Config()
        self.assertEqual(cfg.cable_mount.hand_downward_pitch_deg, 30.0)
```

- [ ] **Step 2: Write failing structural wiring tests**

```python
class AngledHandRuntimeWiringTests(unittest.TestCase):
    def test_runtime_derives_pitched_local_tool_rotation_before_super_init(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        pitch_use = source.index("hand_downward_pitch_deg")
        super_init = source.index("super().__init__(simulation_app=simulation_app, cfg=cfg)")
        self.assertLess(pitch_use, super_init)
        self.assertIn("compute_pitched_hand_from_tool_rotation", source)
        self.assertIn("matrix_to_quaternion_wxyz", source)

    def test_cameras_are_not_independently_overridden(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("set_world_pose", source[source.find("def _camera_model"):])
        self.assertIn("self._world_from_hand_matrix()", source)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_hand_plug_geometry \
  tests.test_angled_hand_runtime_wiring
```

Expected: missing configuration field and missing runtime transform wiring.

- [ ] **Step 4: Add the shared configuration value**

Add to `CableMountConfig` immediately after `fixed_joint_path`:

```python
    # Robot-right-side view: wrist higher, fingertips lower toward the port.
    hand_downward_pitch_deg: float = 30.0
```

Do not add a second pitch value to `IKConfig`, `CameraConfig`, or `InsertionConfig`.

- [ ] **Step 5: Apply the pitched local transform before the base runtime builds the scene**

In `cable_runtime/__init__.py`, import:

```python
from hand_plug_geometry import (
    compute_pitched_hand_from_tool_rotation,
    validate_downward_hand_pitch_deg,
)
```

Before the existing `replace(...)` call, derive the new local rotation:

```python
pitch_deg = validate_downward_hand_pitch_deg(
    cfg.cable_mount.hand_downward_pitch_deg
)
base_hand_from_tool = quaternion_wxyz_to_matrix(
    np.asarray(
        cfg.ik.tool_center_local_orientation_wxyz,
        dtype=np.float64,
    )
)
pitched_hand_from_tool = compute_pitched_hand_from_tool_rotation(
    base_hand_from_tool,
    pitch_deg,
)
pitched_tool_local_orientation = tuple(
    float(value)
    for value in matrix_to_quaternion_wxyz(pitched_hand_from_tool)
)
```

Merge this into the existing immutable config replacement:

```python
cfg = replace(
    cfg,
    ik=replace(
        cfg.ik,
        tool_center_local_orientation_wxyz=pitched_tool_local_orientation,
    ),
    visual_servo=replace(
        cfg.visual_servo,
        max_target_step_m=max(
            float(cfg.visual_servo.max_target_step_m),
            0.005,
        ),
        target_settle_tolerance_m=max(
            float(cfg.visual_servo.target_settle_tolerance_m),
            0.001,
        ),
    ),
)
self._configured_hand_pitch_deg = pitch_deg
```

This preserves the world-space plug-tip target orientation while causing `tool_pose_to_hand_pose(...)` to request the pitched `panda_hand` pose.

- [ ] **Step 6: Add an exact zero-pitch compatibility test**

```python
    def test_zero_pitch_preserves_existing_local_orientation(self):
        base = quaternion_wxyz_to_matrix(
            np.asarray(Config().ik.tool_center_local_orientation_wxyz)
        )
        actual = compute_pitched_hand_from_tool_rotation(base, 0.0)
        np.testing.assert_allclose(actual, base, atol=1.0e-12)
```

- [ ] **Step 7: Run the configuration and wiring tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_hand_plug_geometry \
  tests.test_angled_hand_runtime_wiring
```

Expected: all tests pass.

- [ ] **Step 8: Commit the shared configuration and IK transform**

```bash
git add single_rack_cv/config.py \
        single_rack_cv/cable_runtime/__init__.py \
        single_rack_cv/tests/test_angled_hand_runtime_wiring.py
git commit -m "feat: pitch hand around horizontal plug frame"
```

---

### Task 3: Live Plug Geometry Validation and Startup Gate

**Files:**
- Modify: `single_rack_cv/cable_runtime/__init__.py:184-232`
- Modify: `single_rack_cv/cable_runtime.py:253-311`
- Modify: `single_rack_cv/tests/test_angled_hand_runtime_wiring.py`

**Interfaces:**
- Consumes: live PhysX plug rigid pose and live Lula `panda_hand` FK.
- Produces:
  - `_live_plug_tip_and_axis() -> tuple[np.ndarray, np.ndarray]`
  - `_live_hand_plug_geometry() -> HandPlugGeometryMetrics`
  - startup rejection when pitch sign, relative angle, horizontal plug axis, fixed joint, or deformable attachment is invalid.

- [ ] **Step 1: Write failing structural tests for one shared live plug-pose helper**

```python
    def test_runtime_has_one_live_plug_tip_and_axis_helper(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("def _live_plug_tip_and_axis", source)
        self.assertIn("def _live_hand_plug_geometry", source)
        self.assertIn("measure_hand_plug_geometry", source)

    def test_startup_geometry_keeps_existing_limits(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("max_tip_error_m", source)
        self.assertIn("max_axis_error_deg", source)
        self.assertIn("0.5", DESIGN_PATH.read_text(encoding="utf-8"))
        self.assertIn("1 degree", DESIGN_PATH.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Run the wiring test and verify it fails**

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_angled_hand_runtime_wiring
```

Expected: missing helper and geometry validation strings.

- [ ] **Step 3: Extract the live PhysX plug-tip and nose-axis calculation**

Refactor the current pose block from `_sample_mount_validation_live` into:

```python
    def _live_plug_tip_and_axis(self) -> tuple[np.ndarray, np.ndarray]:
        plug_frame = self.cable_mount.plug_frame
        if plug_frame is None:
            raise RuntimeError("Tracked plug frame is unavailable")

        plug_position, plug_orientation = self._tracked_plug_body.get_world_pose()
        plug_scale = self._tracked_plug_body.get_world_scale()
        position = to_numpy_cpu(
            plug_position,
            shape=(3,),
            label="tracked RJ45 live position",
        )
        orientation = to_numpy_cpu(
            plug_orientation,
            shape=(4,),
            label="tracked RJ45 live orientation",
        )
        scale = to_numpy_cpu(
            plug_scale,
            shape=(3,),
            label="tracked RJ45 world scale",
        )

        world_from_plug = np.eye(4, dtype=np.float64)
        world_from_plug[:3, :3] = (
            quaternion_wxyz_to_matrix(orientation) @ np.diag(scale)
        )
        world_from_plug[:3, 3] = position
        tip_world = (
            world_from_plug @ np.r_[plug_frame.tip_local_m, 1.0]
        )[:3]
        nose_world = (
            world_from_plug[:3, :3] @ plug_frame.nose_axis_local
        )
        nose_world = nose_world / np.linalg.norm(nose_world)
        return tip_world, nose_world
```

- [ ] **Step 4: Add live hand-to-plug metrics and hard gates**

Import:

```python
from hand_plug_geometry import (
    HandPlugGeometryMetrics,
    compute_pitched_hand_from_tool_rotation,
    measure_hand_plug_geometry,
    validate_downward_hand_pitch_deg,
)
```

Add:

```python
    def _live_hand_plug_geometry(self) -> HandPlugGeometryMetrics:
        if self.cable_mount is None:
            raise RuntimeError("Cable mount is unavailable")
        tip_world, plug_axis_world = self._live_plug_tip_and_axis()
        hand_position, hand_orientation = self._hand_pose_from_articulation()
        metrics = measure_hand_plug_geometry(
            hand_position_m=hand_position,
            hand_rotation_world=quaternion_wxyz_to_matrix(hand_orientation),
            plug_tip_position_m=tip_world,
            plug_axis_world=plug_axis_world,
        )
        pitch_error_deg = abs(
            metrics.relative_pitch_deg - self._configured_hand_pitch_deg
        )
        if pitch_error_deg > 0.5:
            raise RuntimeError(
                "hand-to-plug pitch error exceeded 0.5 deg: "
                f"{pitch_error_deg:.6f} deg"
            )
        if not metrics.wrist_higher_fingertips_lower:
            raise RuntimeError(
                "wrong hand pitch sign: wrist is not higher than the plug tip"
            )
        if metrics.plug_horizontal_error_deg > self.cfg.cable_mount.max_axis_error_deg:
            raise RuntimeError(
                "plug is not horizontal: "
                f"{metrics.plug_horizontal_error_deg:.6f} deg"
            )
        return metrics
```

- [ ] **Step 5: Reuse the helper in mount validation**

Replace duplicated live plug extraction inside `_sample_mount_validation_live` with:

```python
tip_world, nose_world = self._live_plug_tip_and_axis()
tool_position, tool_orientation = self._tool_pose_from_articulation()
tool_axis = quaternion_wxyz_to_matrix(tool_orientation)[:, 2]
self._live_hand_plug_geometry()
return (
    float(np.linalg.norm(tip_world - tool_position)),
    angular_error_deg(nose_world, tool_axis),
)
```

The existing base validator call remains first so fixed-joint, deformable attachment, GPU dynamics, and topology checks are not bypassed.

- [ ] **Step 6: Extend startup diagnostics without changing the gate timing**

In `cable_runtime.py::_log_startup_diagnostics`, after mount validation succeeds, call `self._live_hand_plug_geometry()` and add:

```python
geometry_status = (
    f"  configured hand pitch: {self._configured_hand_pitch_deg:.3f} deg\n"
    f"  measured hand-to-plug pitch: "
    f"{geometry.relative_pitch_deg:.6f} deg\n"
    f"  wrist above plug tip: "
    f"{geometry.wrist_above_tip_m * 1000.0:.3f} mm\n"
    f"  requested pitch sign valid: "
    f"{geometry.wrist_higher_fingertips_lower}\n"
    f"  plug horizontal error: "
    f"{geometry.plug_horizontal_error_deg:.6f} deg"
)
```

Print `geometry_status` immediately after the existing plug-tip and plug-axis measurements.

- [ ] **Step 7: Run pure and structural tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_hand_plug_geometry \
  tests.test_angled_hand_runtime_wiring \
  tests.test_runtime_wiring
```

Expected: all tests pass.

- [ ] **Step 8: Commit the startup geometry gate**

```bash
git add single_rack_cv/cable_runtime/__init__.py \
        single_rack_cv/cable_runtime.py \
        single_rack_cv/tests/test_angled_hand_runtime_wiring.py
git commit -m "feat: validate pitched hand and horizontal plug"
```

---

### Task 4: Freeze the Explicit Live Plug Insertion Axis

**Files:**
- Modify: `single_rack_cv/insertion.py:188-224,277-301`
- Modify: `single_rack_cv/cable_runtime/__init__.py:327-355`
- Modify: `single_rack_cv/tests/test_partial_insertion.py`
- Modify: `single_rack_cv/tests/test_two_stage_insertion.py`

**Interfaces:**
- Consumes: `InsertionSample.insertion_axis_world`, sourced from `_live_plug_tip_and_axis()`.
- Produces: `PartialInsertionController.axis_world` frozen from the live plug axis while `frozen_orientation_wxyz` continues to hold the pitched hand-control orientation.

- [ ] **Step 1: Update the test sample factory with an explicit axis**

In `tests/test_partial_insertion.py`:

```python
def sample(
    *,
    frame_index: int = 0,
    alignment_complete: bool = True,
    position=(0.0, 0.0, 0.0),
    orientation=(1.0, 0.0, 0.0, 0.0),
    insertion_axis=(0.0, 0.0, 1.0),
    target_error_m: float = 0.0,
    mount_tip_error_m: float = 0.0,
    mount_axis_error_deg: float = 0.0,
    fixed_joint_valid: bool = True,
    attachment_preserved: bool = True,
) -> InsertionSample:
    return InsertionSample(
        frame_index=frame_index,
        alignment_complete=alignment_complete,
        actual_position_m=np.asarray(position, dtype=np.float64),
        actual_orientation_wxyz=np.asarray(orientation, dtype=np.float64),
        insertion_axis_world=np.asarray(insertion_axis, dtype=np.float64),
        target_error_m=target_error_m,
        mount_tip_error_m=mount_tip_error_m,
        mount_axis_error_deg=mount_axis_error_deg,
        fixed_joint_valid=fixed_joint_valid,
        attachment_preserved=attachment_preserved,
    )
```

Apply the same default field to the sample helper in `tests/test_two_stage_insertion.py`.

- [ ] **Step 2: Replace the old local-+Z assumption test with an explicit-axis test**

```python
    def test_first_command_uses_explicit_plug_axis_not_hand_local_plus_z(self):
        controller = PartialInsertionController(make_limits())
        start = np.array([0.7, -0.2, 1.3])
        angle = math.radians(30.0)
        pitched_orientation = (
            math.cos(angle / 2.0),
            0.0,
            math.sin(angle / 2.0),
            0.0,
        )
        event = controller.update(
            sample(
                frame_index=10,
                position=start,
                orientation=pitched_orientation,
                insertion_axis=(-1.0, 0.0, 0.0),
            )
        )
        np.testing.assert_allclose(
            event.command.target_position_m,
            start + np.array([-0.0005, 0.0, 0.0]),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            event.command.target_orientation_wxyz,
            np.asarray(pitched_orientation),
            atol=1.0e-12,
        )
```

- [ ] **Step 3: Run insertion tests and verify they fail**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_partial_insertion \
  tests.test_two_stage_insertion
```

Expected: `InsertionSample` does not accept `insertion_axis_world` and the controller still derives axis from orientation.

- [ ] **Step 4: Add the explicit axis to `InsertionSample`**

```python
@dataclass(frozen=True)
class InsertionSample:
    frame_index: int
    alignment_complete: bool
    actual_position_m: np.ndarray
    actual_orientation_wxyz: np.ndarray
    insertion_axis_world: np.ndarray
    target_error_m: float
    mount_tip_error_m: float
    mount_axis_error_deg: float
    fixed_joint_valid: bool
    attachment_preserved: bool

    def __post_init__(self) -> None:
        ...
        object.__setattr__(
            self,
            "insertion_axis_world",
            _normalized_axis(self.insertion_axis_world),
        )
```

- [ ] **Step 5: Freeze the sample axis instead of orientation local +Z**

Replace `_freeze_from` with:

```python
    def _freeze_from(self, sample: InsertionSample) -> None:
        self.frozen_start_position_m = sample.actual_position_m.copy()
        self.frozen_orientation_wxyz = sample.actual_orientation_wxyz.copy()
        self.axis_world = sample.insertion_axis_world.copy()
        self.phase = InsertionPhase.READY
```

Keep orientation-error measurement against `frozen_orientation_wxyz`; only translation/depth uses `axis_world`.

- [ ] **Step 6: Feed the live plug axis from the runtime**

In `_partial_insertion_sample`:

```python
_, insertion_axis_world = self._live_plug_tip_and_axis()
return InsertionSample(
    frame_index=int(self.frame_index),
    alignment_complete=bool(self.visual_servo.complete),
    actual_position_m=actual_position,
    actual_orientation_wxyz=actual_orientation,
    insertion_axis_world=insertion_axis_world,
    target_error_m=target_error_m,
    mount_tip_error_m=mount_tip_error_m,
    mount_axis_error_deg=mount_axis_error_deg,
    fixed_joint_valid=self.cable_mount.fixed_joint_is_valid(),
    attachment_preserved=(
        self.cable_mount.built_in_attachment_is_preserved()
    ),
)
```

- [ ] **Step 7: Add a zero-pitch compatibility test**

```python
    def test_explicit_axis_matches_legacy_local_plus_z_at_zero_pitch(self):
        controller = PartialInsertionController(make_limits())
        event = controller.update(
            sample(
                position=(1.0, 2.0, 3.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                insertion_axis=(0.0, 0.0, 1.0),
            )
        )
        np.testing.assert_allclose(
            event.command.target_position_m,
            np.array([1.0, 2.0, 3.0005]),
            atol=1.0e-12,
        )
```

- [ ] **Step 8: Run all insertion tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_partial_insertion \
  tests.test_two_stage_insertion \
  tests.test_two_stage_runtime_wiring
```

Expected: all tests pass; the two-stage command count remains 48 and final commanded port depth remains +10 mm.

- [ ] **Step 9: Commit explicit plug-axis insertion**

```bash
git add single_rack_cv/insertion.py \
        single_rack_cv/cable_runtime/__init__.py \
        single_rack_cv/tests/test_partial_insertion.py \
        single_rack_cv/tests/test_two_stage_insertion.py
git commit -m "feat: drive insertion along live plug axis"
```

---

### Task 5: Visual-Servo Geometry and Regression Wiring

**Files:**
- Modify: `single_rack_cv/tests/test_angled_hand_runtime_wiring.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`
- Test: existing `single_rack_cv/tests/test_live_control.py`
- Test: existing `single_rack_cv/tests/test_front_plane.py`

**Interfaces:**
- Consumes: the pitched `cfg.ik.tool_center_local_orientation_wxyz` before `SimulationRuntime.__init__` computes `desired_port_virtual_camera_usd`.
- Produces: recomputed desired port observation from the pitched hand/camera extrinsics while preserving translation-only control and live camera models.

- [ ] **Step 1: Write structural tests proving the pitch reaches visual geometry**

```python
    def test_pitched_config_is_passed_to_base_runtime_before_desired_port_geometry(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertLess(
            source.index("tool_center_local_orientation_wxyz=pitched_tool_local_orientation"),
            source.index("super().__init__(simulation_app=simulation_app, cfg=cfg)"),
        )

    def test_visual_servo_remains_translation_only(self):
        sim_source = SIM_PATH.read_text(encoding="utf-8")
        self.assertIn("position=target_position + step_world_m", sim_source)
        self.assertIn("orientation=target_orientation", sim_source)

    def test_live_camera_model_still_uses_hand_fk(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        camera_section = source[source.index("def _camera_model"):]
        self.assertIn("self._world_from_hand_matrix()", camera_section)
        self.assertNotIn("ground_truth", camera_section)
        self.assertNotIn("raycast", camera_section)
```

- [ ] **Step 2: Add an unchanged-safety-contract test**

```python
    def test_insertion_distances_and_limits_are_unchanged(self):
        cfg = Config()
        self.assertEqual(cfg.insertion.total_depth_m, 0.060)
        self.assertEqual(cfg.insertion.coarse_approach_depth_m, 0.040)
        self.assertEqual(cfg.insertion.coarse_step_size_m, 0.005)
        self.assertEqual(cfg.insertion.opening_depth_m, 0.050)
        self.assertEqual(cfg.insertion.step_size_m, 0.0005)
        self.assertEqual(cfg.insertion.max_lateral_drift_m, 0.0005)
        self.assertEqual(cfg.insertion.max_orientation_error_deg, 1.0)
        self.assertEqual(cfg.cable_mount.max_tip_error_m, 0.0005)
        self.assertEqual(cfg.cable_mount.max_axis_error_deg, 1.0)
```

- [ ] **Step 3: Run structural, perception, and insertion regression tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_hand_plug_geometry \
  tests.test_angled_hand_runtime_wiring \
  tests.test_runtime_wiring \
  tests.test_live_control \
  tests.test_front_plane \
  tests.test_partial_insertion \
  tests.test_two_stage_insertion \
  tests.test_two_stage_runtime_wiring
```

Expected: all tests pass.

- [ ] **Step 4: Compile the complete package**

Run:

```bash
~/isaacsim/python.sh -m compileall -q .
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 5: Commit regression wiring**

```bash
git add single_rack_cv/tests/test_angled_hand_runtime_wiring.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "test: protect angled hand runtime wiring"
```

---

### Task 6: Operator Documentation and Workstation Qualification

**Files:**
- Modify: `single_rack_cv/README.md`
- Test output: `single_rack_cv/camera_output/angled_hand_horizontal_plug.txt` (local runtime artifact; do not commit generated output)

**Interfaces:**
- Consumes: completed implementation and all test commands.
- Produces: exact operator pull/run instructions and one qualified workstation log before merge.

- [ ] **Step 1: Update the README geometry description**

Add a section containing exactly these operational facts:

```markdown
## Angled hand with horizontal plug

The robot-right-side view is the sign convention:

- wrist higher than the fingertips
- fingertips slope downward toward the port by 30 degrees
- the rigid RJ45 plug remains horizontal
- the deformable cable tail hangs naturally
- `/World/IK_Target` and `/World/ToolCenter` remain the horizontal plug-tip frame
- Lula derives the pitched `panda_hand` pose through the fixed hand-to-tool transform
- insertion freezes the live PhysX plug nose axis, not the pitched hand axis

Both wrist cameras remain rigidly attached to `panda_hand`, so the full RGB-stereo alignment must be requalified after this geometry change.
```

- [ ] **Step 2: Document the geometry kill switch**

```markdown
Stop the run without loosening thresholds when the plug tilts with the hand, the wrist falls below the plug tip, the measured hand-to-plug pitch differs from 30 degrees by more than 0.5 degrees, plug horizontal error exceeds 1 degree, stereo correspondence persistently degrades, or insertion drift approaches 0.5 mm.
```

- [ ] **Step 3: Run the full pure/structural test set**

Run:

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh -m unittest -v \
  tests.test_hand_plug_geometry \
  tests.test_angled_hand_runtime_wiring \
  tests.test_runtime_wiring \
  tests.test_live_control \
  tests.test_front_plane \
  tests.test_partial_insertion \
  tests.test_two_stage_insertion \
  tests.test_two_stage_runtime_wiring \
  tests.test_cable_geometry \
  tests.test_scale_aware_cable_mount \
  tests.test_cable_mount_contract
```

Expected: all tests pass.

- [ ] **Step 4: Commit operator documentation**

```bash
git add single_rack_cv/README.md
git commit -m "docs: explain angled hand horizontal plug geometry"
```

- [ ] **Step 5: Push the branch for workstation testing**

```bash
git push -u origin feature/angled-hand-horizontal-plug
```

- [ ] **Step 6: Pull and verify the exact branch on the Isaac workstation**

```bash
set -e
cd ~/Isaacsim-Scripts
git fetch origin
git switch feature/angled-hand-horizontal-plug
git pull --ff-only origin feature/angled-hand-horizontal-plug
git branch --show-current
git rev-parse --short HEAD
```

Expected branch: `feature/angled-hand-horizontal-plug`.

- [ ] **Step 7: Run the geometry-qualified simulation**

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh main.py \
  2>&1 | tee camera_output/angled_hand_horizontal_plug.txt
```

- [ ] **Step 8: Check geometry before accepting vision or insertion**

The startup log must show all of these:

```text
configured hand pitch: 30.000 deg
measured hand-to-plug pitch: 30.000... deg
wrist above plug tip: positive value
requested pitch sign valid: True
plug horizontal error: <= 1.000000 deg
fixed joint valid: True
built-in attachment preserved: True
validation frames: 30/30
```

The viewport must show, from the robot-right-side view, the wrist higher than the fingertips while the rigid RJ45 plug remains level.

- [ ] **Step 9: Check full-task qualification**

The same run must also show:

```text
RGB STEREO TRACK ACQUIRED
RGB STEREO VISUAL ALIGNMENT LOCKED
RGB STEREO VISUAL SERVO COMPLETE
settled command: 48/48
PARTIAL INSERTION COMPLETE
commanded depth relative to opening: +10.000 mm
```

Acceptance limits:

- final physical ToolCenter tracking error at or below 0.3 mm
- final measured depth within 0.3 mm of +10.0 mm
- lateral drift at or below 0.5 mm
- hand-orientation error at or below 1 degree
- plug horizontal error at or below 1 degree
- no invalid fixed joint or deformable attachment
- terminal action remains hold; no seating, release, or retreat

- [ ] **Step 10: Use the kill switch on any geometry or perception regression**

Do not merge and do not increase tolerances when any of these occurs:

- plug pitches with the hand
- wrist is below the fingertips/plug tip
- relative pitch is outside 30 +/- 0.5 degrees
- persistent stereo pairing or reprojection failures appear
- visual servo does not converge cleanly
- insertion follows the hand's diagonal axis
- lateral drift grows toward or above 0.5 mm
- port-rim contact occurs before the expected opening crossing

- [ ] **Step 11: Merge only after workstation evidence passes**

Open a PR from `feature/angled-hand-horizontal-plug` to `main` and squash merge only after the accepted log proves the complete geometry, visual-servo, and insertion gates. Preserve `main` as the rollback point until then.
