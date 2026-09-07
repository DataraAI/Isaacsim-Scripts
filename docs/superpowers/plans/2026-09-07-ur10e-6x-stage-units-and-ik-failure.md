# UR10e 6× Stage Units and IK Failure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every six-arm station physically reach Cartesian waypoints and fail honestly when IK or grasping fails.

**Architecture:** Keep six-arm application geometry in metres and introduce a package-local `SixArmMotionController` that converts Cartesian targets to USD stage units at the controller boundary. The adapter opts into strict IK failure without changing shared-controller behavior; pure helpers validate units and physical grasp displacement.

**Tech Stack:** Python 3.12, NumPy, Isaac Sim 6.0, Lula kinematics, `unittest`

## Global Constraints

- Apply changed behavior only to `aayush/ur10e_6x_cable_insertions/`.
- Keep all six-arm geometry, configuration, and behavior-tree data in metres.
- Keep `detailedInsertion/cable/franka_motion_controller.py` behavior unchanged.
- Do not modify or permanently rescale `~/Desktop/Aayush_ws/DataHall_6r.usd`.
- Do not change station names, prim paths, or behavior-tree structure.
- Write each regression test first and observe its expected failure before implementation.

---

### Task 1: Pure coordinate and grasp helpers

**Files:**
- Create: `aayush/ur10e_6x_cable_insertions/runtime_support.py`
- Create: `aayush/ur10e_6x_cable_insertions/tests/test_runtime_support.py`

**Interfaces:**
- Produces: `validate_meters_per_unit(value: float) -> float`
- Produces: `meters_to_stage(position, meters_per_unit) -> np.ndarray`
- Produces: `stage_to_meters(position, meters_per_unit) -> np.ndarray`
- Produces: `physical_grasp_is_valid(*, initial_part, current_part, expected_tip, fingers, min_lift_m, max_tip_error_m, contact_rad) -> bool`

- [ ] **Step 1: Write failing unit-conversion tests**

```python
import importlib
import unittest
import numpy as np


def load_support():
    try:
        return importlib.import_module("ur10e_6x_cable_insertions.runtime_support")
    except ModuleNotFoundError as exc:
        raise AssertionError("runtime_support module is missing") from exc


def physical_grasp_is_valid(**kwargs):
    callback = getattr(load_support(), "physical_grasp_is_valid", None)
    if callback is None:
        raise AssertionError("physical_grasp_is_valid is missing")
    return callback(**kwargs)


class StageUnitTests(unittest.TestCase):
    def test_centimeter_stage_round_trip(self):
        support = load_support()
        metres = np.array([0.5783, -1.35, 3.1366])
        stage = support.meters_to_stage(metres, 0.01)
        np.testing.assert_allclose(stage, [57.83, -135.0, 313.66])
        np.testing.assert_allclose(support.stage_to_meters(stage, 0.01), metres)

    def test_meter_stage_is_identity(self):
        support = load_support()
        point = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(support.meters_to_stage(point, 1.0), point)
        np.testing.assert_allclose(support.stage_to_meters(point, 1.0), point)

    def test_invalid_stage_units_are_rejected(self):
        support = load_support()
        for value in (0.0, -0.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                support.validate_meters_per_unit(value)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_runtime_support.StageUnitTests -v
```

Expected: `FAIL` with `AssertionError: runtime_support module is missing`.

- [ ] **Step 3: Implement coordinate helpers**

```python
"""Isaac-free runtime helpers for the six-arm cable demo."""

from __future__ import annotations

import math
import numpy as np


def validate_meters_per_unit(value: float) -> float:
    mpu = float(value)
    if not math.isfinite(mpu) or mpu <= 0.0:
        raise ValueError(f"metersPerUnit must be finite and positive, got {value!r}")
    return mpu


def meters_to_stage(position, meters_per_unit: float) -> np.ndarray:
    mpu = validate_meters_per_unit(meters_per_unit)
    return np.asarray(position, dtype=np.float64) / mpu


def stage_to_meters(position, meters_per_unit: float) -> np.ndarray:
    mpu = validate_meters_per_unit(meters_per_unit)
    return np.asarray(position, dtype=np.float64) * mpu
```

- [ ] **Step 4: Run conversion tests and verify GREEN**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_runtime_support.StageUnitTests -v
```

Expected: three tests pass.

- [ ] **Step 5: Add failing physical-grasp helper tests**

```python
class PhysicalGraspTests(unittest.TestCase):
    def test_stationary_cable_fails_even_with_closed_fingers(self):
        self.assertFalse(physical_grasp_is_valid(
            initial_part=np.array([0.7, -1.35, 3.06]),
            current_part=np.array([0.7, -1.35, 3.06]),
            expected_tip=np.array([0.7, -1.35, 3.06]),
            fingers=np.array([0.82]),
            min_lift_m=0.04,
            max_tip_error_m=0.06,
            contact_rad=0.21,
        ))

    def test_lifted_cable_near_tool_passes(self):
        self.assertTrue(physical_grasp_is_valid(
            initial_part=np.array([0.7, -1.35, 3.06]),
            current_part=np.array([0.7, -1.35, 3.12]),
            expected_tip=np.array([0.7, -1.35, 3.12]),
            fingers=np.array([0.82]),
            min_lift_m=0.04,
            max_tip_error_m=0.06,
            contact_rad=0.21,
        ))
```

- [ ] **Step 6: Run grasp tests and verify RED**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_runtime_support.PhysicalGraspTests -v
```

Expected: `FAIL` with `AssertionError: physical_grasp_is_valid is missing`.

- [ ] **Step 7: Implement physical grasp helper**

```python
def physical_grasp_is_valid(
    *,
    initial_part,
    current_part,
    expected_tip,
    fingers,
    min_lift_m: float,
    max_tip_error_m: float,
    contact_rad: float,
) -> bool:
    initial = np.asarray(initial_part, dtype=np.float64).reshape(3)
    current = np.asarray(current_part, dtype=np.float64).reshape(3)
    tip = np.asarray(expected_tip, dtype=np.float64).reshape(3)
    finger_values = np.asarray(fingers, dtype=np.float64).reshape(-1)
    lifted = float(current[2] - initial[2]) >= float(min_lift_m)
    near_tip = float(np.linalg.norm(current - tip)) <= float(max_tip_error_m)
    closed = bool(finger_values.size) and (
        float(np.max(np.abs(finger_values))) >= float(contact_rad)
    )
    return bool(lifted and near_tip and closed)
```

- [ ] **Step 8: Run helper tests and full six-arm host suite**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_runtime_support -v
python3 -m unittest discover -s ur10e_6x_cable_insertions/tests -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add aayush/ur10e_6x_cable_insertions/runtime_support.py \
        aayush/ur10e_6x_cable_insertions/tests/test_runtime_support.py
git commit -m "test: define six-arm unit and grasp contracts"
```

---

### Task 2: Six-arm unit-aware strict controller

**Files:**
- Create: `aayush/ur10e_6x_cable_insertions/controller.py`
- Create: `aayush/ur10e_6x_cable_insertions/tests/test_controller.py`

**Interfaces:**
- Consumes: `meters_to_stage`, `stage_to_meters`, `validate_meters_per_unit`
- Produces: `SixArmMotionController(FrankaMotionController)`
- Produces: `current_hand_pose_meters() -> tuple[np.ndarray, np.ndarray]`
- Produces: `has_failed() -> bool`
- Produces: `failure_reason() -> str`

- [ ] **Step 1: Write a host-safe fake shared controller and failing adapter tests**

The test injects a fake `franka_motion_controller` module before importing the
package-local adapter, avoiding Isaac imports:

```python
from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class FakeSharedController:
    def __init__(self, *args, **kwargs):
        self.queued = None
        self._joint_interp_warned = False
        self._segment_failed = False
        self._current_command_index = 0
        self._command_queue = []

    def add_cartesian_waypoint(self, position, orientation, **kwargs):
        self.queued = (np.asarray(position), np.asarray(orientation), kwargs)

    def _current_hand_pose(self):
        return np.array([57.83, -135.0, 313.66]), np.array([1.0, 0.0, 0.0, 0.0])

    def _init_joint_interp_segment(self, cmd, current_joint_positions, n_dof):
        self._joint_interp_warned = bool(cmd.get("force_ik_failure"))


def load_adapter():
    fake_module = types.ModuleType("franka_motion_controller")
    fake_module.FrankaMotionController = FakeSharedController
    with patch.dict(sys.modules, {"franka_motion_controller": fake_module}):
        sys.modules.pop("ur10e_6x_cable_insertions.controller", None)
        try:
            return importlib.import_module("ur10e_6x_cable_insertions.controller")
        except ModuleNotFoundError as exc:
            raise AssertionError("six-arm controller module is missing") from exc


class SixArmControllerTests(unittest.TestCase):
    def test_cartesian_target_is_converted_to_stage_units_once(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        controller.add_cartesian_waypoint(
            np.array([0.5783, -1.35, 3.1366]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            label="descend",
        )
        np.testing.assert_allclose(controller.queued[0], [57.83, -135.0, 313.66])

    def test_measured_hand_pose_is_returned_in_metres(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        position, _ = controller.current_hand_pose_meters()
        np.testing.assert_allclose(position, [0.5783, -1.35, 3.1366])

    def test_joint_interp_ik_failure_marks_controller_failed(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        cmd = {"label": "descend", "force_ik_failure": True}
        controller._init_joint_interp_segment(cmd, np.zeros(7), 7)
        self.assertTrue(controller.has_failed())
        self.assertIn("descend", controller.failure_reason())
        self.assertEqual(controller._current_command_index, 0)
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_controller -v
```

Expected: `FAIL` with `AssertionError: six-arm controller module is missing`.

- [ ] **Step 3: Implement the six-arm adapter**

```python
"""Unit-aware strict controller used only by the six-arm demo."""

from __future__ import annotations

from franka_motion_controller import FrankaMotionController
from ur10e_6x_cable_insertions.runtime_support import (
    meters_to_stage,
    stage_to_meters,
    validate_meters_per_unit,
)


class SixArmMotionController(FrankaMotionController):
    def __init__(self, *args, meters_per_unit: float, **kwargs):
        self._meters_per_unit = validate_meters_per_unit(meters_per_unit)
        self._six_arm_failure_reason = ""
        super().__init__(*args, **kwargs)

    def add_cartesian_waypoint(self, position, orientation, **kwargs):
        return super().add_cartesian_waypoint(
            meters_to_stage(position, self._meters_per_unit),
            orientation,
            **kwargs,
        )

    def current_hand_pose_meters(self):
        position, orientation = super()._current_hand_pose()
        return stage_to_meters(position, self._meters_per_unit), orientation

    def _init_joint_interp_segment(self, cmd, current_joint_positions, n_dof):
        super()._init_joint_interp_segment(cmd, current_joint_positions, n_dof)
        if self._joint_interp_warned:
            self._segment_failed = True
            label = str(cmd.get("label", "unlabelled waypoint"))
            self._six_arm_failure_reason = f"IK failed for {label}"

    def has_failed(self) -> bool:
        return bool(self._segment_failed)

    def failure_reason(self) -> str:
        return self._six_arm_failure_reason
```

Do not override `_current_hand_pose`; the shared controller must compare queued
stage-unit goals against stage-unit FK. Six-arm diagnostics call the explicit
metre-returning method.

- [ ] **Step 4: Run adapter and full host tests**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_controller -v
python3 -m unittest discover -s ur10e_6x_cable_insertions/tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add aayush/ur10e_6x_cable_insertions/controller.py \
        aayush/ur10e_6x_cable_insertions/tests/test_controller.py
git commit -m "fix: add strict stage-unit six-arm controller"
```

---

### Task 3: Wire every station to the consistent coordinate space

**Files:**
- Modify: `aayush/ur10e_6x_cable_insertions/scene.py:414-422`
- Modify: `aayush/ur10e_6x_cable_insertions/scene.py:571-592`
- Modify: `aayush/ur10e_6x_cable_insertions/scene.py:629-646`
- Modify: `aayush/ur10e_6x_cable_insertions/tests/test_runtime.py`

**Interfaces:**
- Consumes: `SixArmMotionController(..., meters_per_unit=mpu)`
- Produces: all station controllers configured in stage units

- [ ] **Step 1: Add failing scene-wiring contract test**

```python
class SceneWiringTests(unittest.TestCase):
    def test_scene_uses_six_arm_controller_and_raw_stage_base_pose(self):
        scene_path = Path(__file__).resolve().parents[1] / "scene.py"
        source = scene_path.read_text(encoding="utf-8")
        self.assertIn("from ur10e_6x_cable_insertions.controller import SixArmMotionController", source)
        self.assertIn("SixArmMotionController(", source)
        self.assertIn("meters_per_unit=meters_per_unit(stage)", source)
        self.assertIn("kinematics.set_robot_base_pose(base_position, base_orientation)", source)
        self.assertNotIn(
            "kinematics.set_robot_base_pose(pose_to_meters(base_position, stage), base_orientation)",
            source,
        )
```

- [ ] **Step 2: Run wiring test and verify RED**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_runtime.SceneWiringTests -v
```

Expected: failure because `scene.py` still constructs `FrankaMotionController`
and converts only the base pose to metres.

- [ ] **Step 3: Wire the adapter in `scene.py`**

Replace the shared controller import with:

```python
from ur10e_6x_cable_insertions.controller import SixArmMotionController
```

Delete the now-unused `pose_to_meters` helper. In `_make_motion_controller`, use
one coordinate space for base pose, FK, and queued targets:

```python
base_position, base_orientation = robot.get_world_pose()
kinematics.set_robot_base_pose(base_position, base_orientation)
return SixArmMotionController(
    name=f"{spec.scene_name}_controller",
    robot_articulation=robot,
    task_traj_gen=trajectory_generator,
    art_kinematics=articulation_kinematics,
    gripper=robot.gripper,
    tool_offset=0.0,
    physics_dt=1.0 / 120.0,
    ee_frame=ee_frame,
    debug=True,
    meters_per_unit=meters_per_unit(stage),
)
```

`build_scene` already calls `_make_motion_controller` for every selected
station, so no station-specific branches are added.

- [ ] **Step 4: Run wiring and full host tests**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_runtime.SceneWiringTests -v
python3 -m unittest discover -s ur10e_6x_cable_insertions/tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add aayush/ur10e_6x_cable_insertions/scene.py \
        aayush/ur10e_6x_cable_insertions/tests/test_runtime.py
git commit -m "fix: use stage units for all six-arm IK"
```

---

### Task 4: Reject false grasp success and use metre diagnostics

**Files:**
- Modify: `aayush/ur10e_6x_cable_insertions/config.py:220-270`
- Modify: `aayush/ur10e_6x_cable_insertions/primitives.py:63-84`
- Modify: `aayush/ur10e_6x_cable_insertions/primitives.py:154-225`
- Modify: `aayush/ur10e_6x_cable_insertions/primitives.py:227-257`
- Modify: `aayush/ur10e_6x_cable_insertions/primitives.py:544-567`
- Modify: `aayush/ur10e_6x_cable_insertions/tests/test_runtime.py`

**Interfaces:**
- Consumes: `current_hand_pose_meters()`
- Consumes: `physical_grasp_is_valid(...)`
- Produces: `services["initial_grasp_point"]`
- Produces: honest physical grasp validation

- [ ] **Step 1: Add failing source contract test**

```python
class GraspWiringTests(unittest.TestCase):
    def test_grasp_records_initial_position_and_uses_metre_pose(self):
        primitives_path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = primitives_path.read_text(encoding="utf-8")
        self.assertIn('context.services["initial_grasp_point"] = center.copy()', source)
        self.assertIn("controller.current_hand_pose_meters()", source)
        self.assertIn("physical_grasp_is_valid(", source)
        self.assertNotIn("lifted = float(center[2]) >= block_top + 0.04", source)
```

- [ ] **Step 2: Run wiring test and verify RED**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_runtime.GraspWiringTests -v
```

Expected: failure because initial displacement and metre pose are not wired.

- [ ] **Step 3: Add a named lift threshold**

In `config.py`:

```python
GRASP_MIN_LIFT_M = 0.04
```

- [ ] **Step 4: Record the baseline during successful detection**

In `detect_grasp_part`, immediately after caching the detected point:

```python
context.services["detected_grasp_point"] = center.copy()
context.services["live_grasp_point"] = center.copy()
context.services["initial_grasp_point"] = center.copy()
```

In `queue_grasp`, require that baseline so retries cannot silently use an
absolute floor test:

```python
if context.services.get("initial_grasp_point") is None:
    context.services["initial_grasp_point"] = point.copy()
```

- [ ] **Step 5: Replace grasp validation with displacement plus tip tracking**

Import the helper:

```python
from ur10e_6x_cable_insertions.runtime_support import physical_grasp_is_valid
```

In `check_physical_grasp`, obtain current part, baseline, fingers, and the
metre-space expected tool tip:

```python
initial = context.services.get("initial_grasp_point")
if initial is None:
    print(f"[BT GRASP{_station_tag(context)}] validate missing initial grasp point")
    return False

try:
    fingers = np.asarray(
        context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
    ).reshape(-1)
    hand, quat = context.services["motion_controller"].current_hand_pose_meters()
except Exception as exc:
    print(f"[BT GRASP{_station_tag(context)}] validate state failed: {exc}")
    return False

expected_tip = np.asarray(hand, dtype=np.float64) + _quat_to_rot_matrix(quat) @ np.array(
    [0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64
)
passed = physical_grasp_is_valid(
    initial_part=initial,
    current_part=center,
    expected_tip=expected_tip,
    fingers=fingers,
    min_lift_m=cfg.GRASP_MIN_LIFT_M,
    max_tip_error_m=cfg.CABLE_IN_GRIPPER_MAX_ERR_M,
    contact_rad=cfg.ROBOTIQ_CONTACT_RAD,
)
lift_delta = float(center[2] - np.asarray(initial, dtype=np.float64)[2])
tip_err = float(np.linalg.norm(center - expected_tip))
```

Log `lift_delta`, `tip_err`, and `PASS`/`FAIL`; add `cable_held` only when
`passed` is true.

- [ ] **Step 6: Convert hold-monitor FK through the adapter**

In `cable_still_in_gripper`, replace:

```python
hand, quat = controller._current_hand_pose()
```

with:

```python
hand, quat = controller.current_hand_pose_meters()
```

This makes both `part` and `tip_expected` metres and removes the false
approximately-98-metre error.

- [ ] **Step 7: Run focused and full host tests**

Run:

```bash
cd aayush
python3 -m unittest ur10e_6x_cable_insertions.tests.test_runtime.GraspWiringTests -v
python3 -m unittest discover -s ur10e_6x_cable_insertions/tests -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add aayush/ur10e_6x_cable_insertions/config.py \
        aayush/ur10e_6x_cable_insertions/primitives.py \
        aayush/ur10e_6x_cable_insertions/tests/test_runtime.py
git commit -m "fix: require physical lift for six-arm grasp"
```

---

### Task 5: Documentation and runtime verification

**Files:**
- Modify: `aayush/ur10e_6x_cable_insertions/README.md:8-27`

**Interfaces:**
- Consumes: completed unit-aware controller and honest grasp validation
- Produces: reproducible one-station and six-station verification instructions

- [ ] **Step 1: Document expected success and failure evidence**

Add:

````markdown
## Motion verification

Validate one station before running all six:

```bash
/home/aayush/isaacsim/python.sh aayush/ur10e_6x_cable_insertions/main.py \
    --station NegativeY_Top
```

The arm must visibly reach observation and hover, descend before finger
closure, and lift the cable. Any `Joint-interp IK target failed` message must
produce behavior-tree `FAILURE`; it must never be followed by `REACHED` for the
same waypoint.
````

- [ ] **Step 2: Run all host tests**

Run:

```bash
cd aayush
python3 -m unittest discover -s ur10e_6x_cable_insertions/tests -v
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 3: Run one-station Isaac verification**

Run from the repository root:

```bash
/home/aayush/isaacsim/python.sh aayush/ur10e_6x_cable_insertions/main.py \
    --station NegativeY_Top
```

Expected visual sequence: observation move → hover → descend → fingers close →
cable lifts. Expected logs contain no IK-failure-followed-by-`REACHED` pair and
no tool-tip position tens of metres from the scene.

If IK genuinely cannot solve a target, expected behavior is immediate station
`FAILURE` with the waypoint label; do not tune targets in this task.

- [ ] **Step 4: Run all six stations**

Run:

```bash
/home/aayush/isaacsim/python.sh aayush/ur10e_6x_cable_insertions/main.py
```

Expected: every arm begins the same physical observation/hover/descend
sequence. A failure in one station does not make another station report false
success.

- [ ] **Step 5: Commit documentation**

```bash
git add aayush/ur10e_6x_cable_insertions/README.md
git commit -m "docs: add six-arm motion verification"
```
