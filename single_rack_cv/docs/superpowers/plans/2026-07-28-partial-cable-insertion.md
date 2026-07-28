# Partial Cable Insertion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the qualified RGB stereo alignment finishes, advance the pregrasped RJ45 connector exactly 10 mm along the frozen ToolCenter +Z axis in twenty 0.5 mm steps, then hold.

**Architecture:** Keep the qualified visual-servo code unchanged. Add a pure, Isaac-independent insertion state machine that owns frozen-axis geometry, step sequencing, settle counting, timeout handling, and abort decisions. The GPU-safe cable runtime facade supplies live Lula FK and live rigid-plug measurements, preflights each new target with Lula IK, and writes the accepted target to `/World/IK_Target`.

**Tech Stack:** Python 3, `dataclasses`, `enum`, NumPy, `unittest`, Isaac Sim 6.0.0, Lula IK/FK, PhysX GPU dynamics.

## Global Constraints

- Total commanded insertion depth is exactly `0.010 m`.
- Command increment is exactly `0.0005 m`, producing twenty increments.
- The insertion frame is frozen from the physically settled ToolCenter pose at visual-servo completion.
- The insertion axis is ToolCenter local `+Z` expressed in world coordinates.
- ToolCenter orientation remains fixed for the entire insertion.
- Each step requires physical target error `<= 0.0003 m` for 6 consecutive simulation frames.
- Each step times out after `2.0 s`.
- Abort when lateral drift exceeds `0.0005 m`.
- Abort when frozen-orientation error exceeds `1.0 deg`.
- Abort when live plug-tip mount error exceeds `0.0005 m`.
- Abort when live plug-axis error exceeds `1.0 deg`.
- Abort when the fixed joint or built-in deformable-tail attachment becomes invalid.
- Preflight every new target with Lula IK; do not publish an unreachable target.
- On completion or abort, hold the current target. Do not seat, release, retreat, or resume perception.
- Keep runtime image-only; do not add RTX/USD ground-truth control.

---

### Task 1: Pure frozen-axis insertion controller

**Files:**
- Create: `single_rack_cv/insertion.py`
- Test: `single_rack_cv/tests/test_partial_insertion.py`

**Interfaces:**
- Produces: `InsertionPhase`, `InsertionLimits`, `InsertionSample`, `InsertionCommand`, `InsertionEvent`, `PartialInsertionController`, `decompose_axis_motion(...)`, and `quaternion_angular_error_deg(...)`.
- Consumes: NumPy arrays in WXYZ quaternion convention.

- [ ] **Step 1: Write the failing geometry tests**

```python
from insertion import (
    InsertionLimits,
    InsertionPhase,
    InsertionSample,
    PartialInsertionController,
    decompose_axis_motion,
    quaternion_angular_error_deg,
)


def test_axis_decomposition_reports_axial_and_lateral_motion():
    axial, lateral = decompose_axis_motion(
        start_position_m=np.array([1.0, 2.0, 3.0]),
        actual_position_m=np.array([1.010, 2.0003, 3.0004]),
        axis_world=np.array([1.0, 0.0, 0.0]),
    )
    assert abs(axial - 0.010) < 1.0e-12
    assert abs(lateral - 0.0005) < 1.0e-12


def test_quaternion_error_is_sign_invariant():
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    assert quaternion_angular_error_deg(reference, -reference) == 0.0
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" -m unittest -v tests.test_partial_insertion
```

Expected: import failure because `insertion.py` does not exist.

- [ ] **Step 3: Add state-machine tests**

Cover these behaviors with separate tests:

```python
# No command before alignment completion.
controller.update(sample(alignment_complete=False))
assert controller.phase is InsertionPhase.WAITING_FOR_ALIGNMENT
assert controller.last_command is None

# First valid update freezes pose and emits exactly 0.5 mm.
command = controller.update(sample(alignment_complete=True))
assert command.step_index == 1
assert abs(command.commanded_depth_m - 0.0005) < 1.0e-12

# Six settled frames advance to the next exact frozen-axis target.
# Twenty settled increments end at 10.0 mm and COMPLETE.
# A 0.500001 mm lateral drift aborts.
# A >1 degree orientation error aborts.
# Mount, structural, IK, and timeout failures abort.
# COMPLETE and ABORTED never emit another command.
```

- [ ] **Step 4: Implement the minimal pure controller**

Implement:

```python
class InsertionPhase(str, Enum):
    WAITING_FOR_ALIGNMENT = "waiting_for_alignment"
    ADVANCING = "advancing"
    COMPLETE = "complete"
    ABORTED = "aborted"

@dataclass(frozen=True)
class InsertionLimits:
    total_depth_m: float
    step_size_m: float
    settle_tolerance_m: float
    required_settled_frames: int
    step_timeout_frames: int
    max_lateral_drift_m: float
    max_orientation_error_deg: float
    max_mount_tip_error_m: float
    max_mount_axis_error_deg: float
```

`PartialInsertionController.update(sample)` must return an `InsertionEvent` containing an optional `InsertionCommand`. Each target is `frozen_start + frozen_axis * commanded_depth`; never accumulate from the measured pose.

- [ ] **Step 5: Run the focused test module**

Run the command from Step 2. Expected: all partial-insertion tests pass.

- [ ] **Step 6: Commit**

```bash
git add single_rack_cv/insertion.py single_rack_cv/tests/test_partial_insertion.py
git commit -m "feat(insertion): add frozen-axis partial insertion controller"
```

---

### Task 2: Add exact insertion configuration and structural wiring tests

**Files:**
- Modify: `single_rack_cv/config.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Produces: `InsertionConfig` and `Config.insertion`.
- Consumes: `PartialInsertionController` limits in Task 3.

- [ ] **Step 1: Write failing config assertions**

Add a runtime-wiring test asserting:

```python
self.assertTrue(CONFIG.insertion.enabled)
self.assertEqual(CONFIG.insertion.total_depth_m, 0.010)
self.assertEqual(CONFIG.insertion.step_size_m, 0.0005)
self.assertEqual(CONFIG.insertion.settle_position_tolerance_m, 0.0003)
self.assertEqual(CONFIG.insertion.required_settled_frames, 6)
self.assertEqual(CONFIG.insertion.step_timeout_s, 2.0)
self.assertEqual(CONFIG.insertion.max_lateral_drift_m, 0.0005)
self.assertEqual(CONFIG.insertion.max_orientation_error_deg, 1.0)
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" -m unittest -v \
  tests.test_runtime_wiring.RuntimeWiringTests.test_partial_insertion_config_is_exact
```

Expected: failure because `Config` has no `insertion` field.

- [ ] **Step 3: Implement `InsertionConfig`**

Add this frozen dataclass after `VisualServoConfig`:

```python
@dataclass(frozen=True)
class InsertionConfig:
    enabled: bool = True
    total_depth_m: float = 0.010
    step_size_m: float = 0.0005
    settle_position_tolerance_m: float = 0.0003
    required_settled_frames: int = 6
    step_timeout_s: float = 2.0
    max_lateral_drift_m: float = 0.0005
    max_orientation_error_deg: float = 1.0
```

Add `insertion: InsertionConfig = field(default_factory=InsertionConfig)` to `Config`.

- [ ] **Step 4: Run focused tests**

Run `tests.test_runtime_wiring` and `tests.test_partial_insertion`. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add single_rack_cv/config.py single_rack_cv/tests/test_runtime_wiring.py
git commit -m "feat(insertion): configure 10 mm partial entry"
```

---

### Task 3: Integrate live FK, mount validation, IK preflight, and target updates

**Files:**
- Modify: `single_rack_cv/cable_runtime/__init__.py`
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Consumes: `PartialInsertionController`, `InsertionLimits`, and `InsertionSample` from Task 1; `Config.insertion` from Task 2.
- Produces: `CableMountedSimulationRuntime.update_partial_insertion()`.

- [ ] **Step 1: Write failing runtime-wiring assertions**

Assert that:

```python
runtime_source = (ROOT / "cable_runtime" / "__init__.py").read_text()
main_source = (ROOT / "main.py").read_text()
self.assertIn("PartialInsertionController", runtime_source)
self.assertIn("def update_partial_insertion", runtime_source)
self.assertIn("tool_pose_to_hand_pose", runtime_source)
self.assertIn("compute_inverse_kinematics", runtime_source)
self.assertIn("runtime.update_partial_insertion()", main_source)
self.assertLess(
    main_source.index("runtime.update_visual_servo_completion()"),
    main_source.index("runtime.update_partial_insertion()"),
)
```

- [ ] **Step 2: Run the focused test and verify RED**

Expected: missing insertion runtime symbols.

- [ ] **Step 3: Construct the controller in the GPU-safe facade**

In `__init__`, convert `step_timeout_s` to frames with `scene.physics_dt`, construct `InsertionLimits`, and initialize `PartialInsertionController`.

- [ ] **Step 4: Add live sampling**

Each frame after visual completion:

1. Read ToolCenter pose from `_tool_pose_from_articulation()`.
2. Read current target pose from `self.ik.target.get_world_pose()`.
3. Measure target error.
4. Call `_sample_mount_validation_live(self)` for live tip and axis errors while retaining structural checks.
5. Build `InsertionSample` with `alignment_complete=self.visual_servo.complete`.

Convert structural exceptions into one `ABORTED` event rather than allowing repeated main-loop warnings.

- [ ] **Step 5: Add Lula IK preflight**

Before publishing each `InsertionCommand`:

1. Convert the candidate ToolCenter pose to `panda_hand` with `tool_pose_to_hand_pose(...)`.
2. Set the current articulation base pose on Lula.
3. Call `compute_inverse_kinematics(...)` with existing IK tolerances.
4. If `success` is false, call `controller.abort("Lula IK rejected insertion target")` and leave the current target unchanged.
5. If true, write the exact frozen-axis pose to `self.ik.target.set_world_pose(...)`.

- [ ] **Step 6: Add structured diagnostics**

Log on start, every settled increment, abort, and completion. Include commanded depth, actual axial depth, lateral drift, ToolCenter target error, orientation error, plug-tip error, plug-axis error, settle count, and timeout frames.

- [ ] **Step 7: Wire the main loop**

After `runtime.update_visual_servo_completion()`, call `runtime.update_partial_insertion()` inside the existing motion exception boundary.

- [ ] **Step 8: Run focused tests**

Run:

```bash
"$HOME/isaacsim/python.sh" -m unittest -v \
  tests.test_partial_insertion \
  tests.test_runtime_wiring
```

Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add single_rack_cv/cable_runtime/__init__.py single_rack_cv/main.py single_rack_cv/tests/test_runtime_wiring.py
git commit -m "feat(insertion): execute guarded 10 mm partial insertion"
```

---

### Task 4: Update operator documentation and qualification gates

**Files:**
- Modify: `single_rack_cv/README.md`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Documents the runtime contract and workstation evidence required before merge.

- [ ] **Step 1: Replace obsolete no-insertion assertions**

Update the README wiring test to require:

```python
self.assertIn("10 mm partial insertion", source)
self.assertIn("0.5 mm steps", source)
self.assertIn("frozen ToolCenter +Z", source)
self.assertIn("holds on abort", source)
self.assertNotIn("No insertion motion", source)
```

- [ ] **Step 2: Update README**

Document:

- visual servo remains 5 mm coarse stop-and-look alignment
- insertion starts only after 30/30 final settle and mount validation
- twenty 0.5 mm commands advance 10 mm along frozen ToolCenter +Z
- perception freezes during insertion
- all abort limits and hold behavior
- add `tests.test_partial_insertion` to the canonical test command

- [ ] **Step 3: Run structural tests**

Run `tests.test_runtime_wiring`. Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add single_rack_cv/README.md single_rack_cv/tests/test_runtime_wiring.py
git commit -m "docs(insertion): document partial-entry safety contract"
```

---

### Task 5: Verify the branch and qualify on the workstation

**Files:**
- No code changes unless verification exposes a defect.

- [ ] **Step 1: Run the complete pure and structural suite**

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
  tests.test_cable_mount_contract \
  tests.test_partial_insertion
```

Expected: all tests pass.

- [ ] **Step 2: Run Isaac Sim**

```bash
cd "$HOME/Isaacsim-Scripts" || exit 1
git switch feature/partial-cable-insertion
git pull --ff-only
cd single_rack_cv || exit 1
"$HOME/isaacsim/python.sh" main.py 2>&1 | tee camera_output/partial_insertion_nominal.txt
```

- [ ] **Step 3: Check qualification output**

```bash
grep -A 14 -E \
  "VISUAL SERVO COMPLETE|PARTIAL INSERTION|ABORTED|FATAL ERROR" \
  camera_output/partial_insertion_nominal.txt
```

Passing evidence:

- prior mount and visual-servo qualification still pass
- insertion starts only after visual completion
- twenty increments settle
- final commanded and actual axial depth are 10.0 mm within tracking tolerance
- lateral drift stays `<= 0.5 mm`
- orientation error stays `<= 1.0 deg`
- mount tip and axis limits remain valid
- final phase is `COMPLETE` and target holds

- [ ] **Step 4: Apply the kill switch**

If the first run shows growing lateral error, rim collision, mount displacement, or repeated step timeout, stop. Do not loosen tolerances or increase depth. Diagnose frozen-axis direction and port geometry first.
