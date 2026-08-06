# 2 mm Precontact Alignment Hold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the mesh-derived connector TCP from the qualified 50 mm preinsert pose to exactly 2 mm before the measured port plane, verify alignment there, and make penetration impossible in this mode.

**Architecture:** Reuse the existing frozen-axis, two-stage `PartialInsertionController`, but construct its runtime limits with `total_depth_m = opening_depth_m - 0.002`. Preserve the current 40 mm coarse stage and 0.5 mm fine stage, producing 8 coarse commands plus 16 fine commands. The controller reaches `COMPLETE` at commanded port depth `-2.000 mm`, holds the last IK target, logs final physical metrics, and cannot issue another command.

**Tech Stack:** Python 3.12, NumPy, unittest, Isaac Sim 6.0.0, OpenUSD.

## Global Constraints

- Keep the camera-derived `EstimatedPortPoint` and `FrozenPortPoint` unchanged.
- Keep the mesh-derived connector TCP unchanged.
- Preserve the 50.0 mm preinsert standoff.
- Stop at exactly 2.0 mm before the measured opening plane.
- Never issue a target at or inside the opening plane in this mode.
- Preserve 5.0 mm coarse steps through 40.0 mm total travel.
- Preserve 0.5 mm fine steps from 40.0 mm to 48.0 mm total travel.
- Preserve the 0.5 mm lateral-drift limit.
- Preserve the 1.0 degree orientation-error limit.
- Preserve all cable mount, fixed-joint, attachment, IK-preflight, and timeout aborts.
- Keep full +10 mm insertion disabled after the precontact hold completes.

---

### Task 1: Define the precontact depth policy with failing tests

**Files:**
- Create: `single_rack_cv/precontact_alignment.py`
- Create: `single_rack_cv/tests/test_precontact_alignment.py`

**Interfaces:**
- Produces: `PrecontactAlignmentPolicy`, `build_precontact_limits(base_limits, policy)`.

- [ ] Write tests asserting a 50 mm opening depth and 2 mm hold offset produce 48 mm total travel, 24 total commands, and a final commanded port depth of -2 mm.
- [ ] Write rejection tests for nonpositive hold offset, hold offset greater than the opening depth, and a capped depth not beyond the 40 mm coarse stage.
- [ ] Implement the immutable policy and limit builder without changing any safety threshold other than `total_depth_m`.
- [ ] Run `~/isaacsim/python.sh -m unittest -v tests.test_precontact_alignment`.
- [ ] Commit the pure policy and tests.

### Task 2: Wire the policy into the mesh-derived TCP runtime

**Files:**
- Modify: `single_rack_cv/connector_tcp_usd.py`
- Modify: `single_rack_cv/scale_aware_cable_mount.py`
- Modify: `single_rack_cv/cable_runtime/__init__.py`
- Modify: `single_rack_cv/tests/test_connector_tcp_runtime_wiring.py`
- Create: `single_rack_cv/tests/test_precontact_runtime_wiring.py`

**Interfaces:**
- `ScaleAwareCableMount.precontact_alignment_only: bool`
- `ScaleAwareCableMount.precontact_hold_offset_m: float`

- [ ] Replace marker-only mode with `PRECONTACT_ALIGNMENT_ONLY = True` and `PRECONTACT_HOLD_OFFSET_M = 0.002`; retain both TCP markers for diagnostics.
- [ ] Store the mode and hold offset on `ScaleAwareCableMount`.
- [ ] Build the runtime insertion controller from `build_precontact_limits(...)` when the mode is active.
- [ ] Assert static wiring retains the original opening depth, coarse depth, step sizes, lateral limit, orientation limit, and motion-validation path.
- [ ] Run focused policy and wiring tests.
- [ ] Commit runtime wiring.

### Task 3: Make the terminal hold explicit and non-penetrating

**Files:**
- Modify: `single_rack_cv/cable_runtime/__init__.py`
- Modify: `single_rack_cv/settled_stereo_handoff_runtime.py`
- Modify: `single_rack_cv/tests/test_precontact_runtime_wiring.py`

**Interfaces:**
- Produces runtime logs `PRECONTACT ALIGNMENT STARTED`, `PRECONTACT ALIGNMENT STEP SETTLED`, `PRECONTACT ALIGNMENT HOLD REACHED`, and `PRECONTACT ALIGNMENT ABORTED`.

- [ ] Use precontact-specific event labels without changing controller events.
- [ ] At completion, log commanded and actual depth relative to the opening, lateral drift, orientation error, ToolCenter tracking error, mount errors, and explicit `penetration commands: disabled`.
- [ ] Prevent orientation-hold logic from describing the motion as insertion while leaving its bounded feedback active during `ADVANCING`.
- [ ] Add tests that no completion path resets the controller or creates a second motion sequence.
- [ ] Run focused tests and commit.

### Task 4: Workstation release gate

**Files:**
- Verify only.

- [ ] Compile changed modules.
- [ ] Run focused tests.
- [ ] Run the complete test suite.
- [ ] Launch `main.py` and verify the runtime reports 24 commands total.
- [ ] Require final commanded depth `-2.000 mm` relative to the opening.
- [ ] Require final actual depth to remain negative and close to `-2.000 mm`.
- [ ] Require lateral drift <= 0.5 mm, orientation error <= 1.0 degree, and ToolCenter tracking error <= 0.3 mm after the required settled window.
- [ ] Confirm the robot holds and never logs `PARTIAL INSERTION COMPLETE` or any command with nonnegative port depth.
- [ ] Keep PR #9 draft until the workstation run satisfies every gate.
