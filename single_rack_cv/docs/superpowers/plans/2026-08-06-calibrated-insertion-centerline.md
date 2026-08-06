# Calibrated Insertion Centerline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the guarded insertion path 0.300 mm toward world negative Y and 0.450 mm toward world negative Z while preserving the camera-derived port point, 50 mm handoff, 48-command depth schedule, and unchanged 0.5 mm deviation guard.

**Architecture:** Keep the existing insertion-only wrapper. Every generated insertion command receives the same fixed world-space offset `[0.0, -0.00030, -0.00045]` m. Recalculate only `lateral_drift_m` relative to the parallel calibrated insertion line, while preserving axial depth and all other metrics from the proven base controller. The existing controller behavior already suppresses an in-flight lateral abort while the first shifted target is still outside the translation settle radius; the step cannot settle until the calibrated-line drift is within the unchanged 0.5 mm limit.

**Tech Stack:** Python 3, NumPy, dataclasses, unittest, Isaac Sim 6.0.

## Global Constraints

- Keep `/World/EstimatedPortPoint` unchanged.
- Keep `/World/FrozenPortPoint` unchanged.
- Keep the 50 mm handoff destination unchanged.
- Keep the exact 48-command depth schedule and +10 mm terminal depth unchanged.
- Keep the 0.5 mm lateral deviation limit unchanged.
- Keep the 1 degree orientation limit unchanged.
- Apply the offset only to insertion commands.
- Use world offset `[0.0, -0.00030, -0.00045]` m.
- Reject insertion calibration offsets larger than 1.0 mm.

---

### Task 1: Specify the calibrated-line behavior

**Files:**
- Modify: `single_rack_cv/tests/test_two_stage_insertion.py`
- Modify: `single_rack_cv/tests/test_handoff_position_hold_runtime_wiring.py`

**Interfaces:**
- Consumes: `TrimmedConsecutivePoseInsertionController(limits, target_offset_world_m=...)`
- Produces: regression coverage for command positions and calibrated-line drift metrics.

- [ ] **Step 1: Change the production-offset expectation**

Use:

```python
trim_world_m = np.array([0.0, -0.00030, -0.00045])
```

Assert all 48 target positions differ from the baseline by exactly that vector, while stages and depth fields remain identical.

- [ ] **Step 2: Add a calibrated-line drift test**

Start the trimmed controller, sample the first command target, and assert:

```python
self.assertLess(event.metrics.lateral_drift_m, 1.0e-12)
```

Then add 0.00051 m in an axis-orthogonal direction, provide a settled target error, and assert the controller aborts with `lateral drift exceeded limit`.

- [ ] **Step 3: Add a first-command convergence test**

After the first shifted command is issued, provide the unshifted start position with a large target error. Assert the controller returns `waiting_for_settle`, not `aborted`. This confirms the existing in-flight recovery behavior handles the initial 0.541 mm lateral move at the 50 mm standoff.

- [ ] **Step 4: Run the focused tests and verify they fail**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_two_stage_insertion \
  tests.test_handoff_position_hold_runtime_wiring
```

Expected: FAIL because the current controller rejects a trim magnitude above 0.5 mm and measures drift from the uncalibrated line.

### Task 2: Measure drift around the calibrated insertion line

**Files:**
- Modify: `single_rack_cv/insertion_target_trim.py`

**Interfaces:**
- Produces: `TrimmedConsecutivePoseInsertionController._metrics(sample)` with calibrated-line `lateral_drift_m`.
- Preserves: base controller axial depth, port depth, target error, orientation error, mount metrics, settle counters, and timeouts.

- [ ] **Step 1: Replace the offset-versus-drift-limit validation**

Define:

```python
_MAXIMUM_INSERTION_CALIBRATION_M = 0.001
```

Reject nonfinite offsets and offsets whose norm exceeds 1.0 mm. Do not compare the intentional calibration magnitude to `max_lateral_drift_m`, because that limit now measures deviation around the calibrated line.

- [ ] **Step 2: Add calibrated-line distance calculation**

For a frozen insertion frame:

```python
calibrated_origin = self.frozen_start_position_m + self.target_offset_world_m
relative = sample.actual_position_m - calibrated_origin
calibrated_axial = float(np.dot(relative, self.axis_world))
lateral = relative - calibrated_axial * self.axis_world
calibrated_lateral_drift_m = float(np.linalg.norm(lateral))
```

- [ ] **Step 3: Override only the lateral metric**

Call `super()._metrics(sample)` and use `dataclasses.replace` to replace only `lateral_drift_m`. Leave `actual_axial_depth_m` and `actual_port_depth_m` unchanged.

- [ ] **Step 4: Run the focused tests**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_two_stage_insertion \
  tests.test_handoff_position_hold_runtime_wiring
```

Expected: PASS.

### Task 3: Install and report the larger production calibration

**Files:**
- Modify: `single_rack_cv/full_insertion_runtime.py`
- Modify: `single_rack_cv/tests/test_handoff_position_hold_runtime_wiring.py`
- Modify: PR #9 description.

**Interfaces:**
- Produces: `_INSERTION_TARGET_OFFSET_WORLD_M = np.array([0.0, -0.00030, -0.00045])`.

- [ ] **Step 1: Change the production constant**

```python
_INSERTION_TARGET_OFFSET_WORLD_M = np.array(
    [0.0, -0.00030, -0.00045],
    dtype=np.float64,
)
```

- [ ] **Step 2: Correct the startup log**

Report:

```text
insertion target line world Y: -0.300 mm
insertion target line world Z: -0.450 mm
lateral drift reference: calibrated insertion line
lateral deviation abort limit: 0.500 mm
```

Remove the old `remaining lateral budget` message because the calibration itself is no longer counted as deviation.

- [ ] **Step 3: Run the full focused suite**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_two_stage_insertion \
  tests.test_handoff_position_hold_runtime_wiring \
  tests.test_precontact_runtime_wiring \
  tests.test_startup_geometry_settle \
  tests.test_tool_goal_trim
```

Expected: PASS with zero failures.

### Task 4: Workstation live verification

**Files:**
- No code changes.

- [ ] **Step 1: Launch Isaac Sim**

```bash
~/isaacsim/python.sh main.py 2>&1 | tee camera_output/calibrated_centerline_run.txt
```

- [ ] **Step 2: Verify startup configuration**

Require the startup log to show world Y `-0.300 mm`, world Z `-0.450 mm`, and calibrated insertion-line drift reference.

- [ ] **Step 3: Verify the run**

Require handoff completion, all 48 commands, approximately +10 mm final depth, calibrated-line lateral deviation at or below 0.5 mm, orientation error at or below 1 degree, and no abort.
