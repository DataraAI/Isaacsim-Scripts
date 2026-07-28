# Two-Stage Cable Approach and Partial Insertion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From the qualified 50 mm visual-servo standoff, command 40 mm of coarse approach followed by 20 mm of fine motion so the RJ45 tip finishes exactly 10 mm inside the physical port opening.

**Architecture:** Preserve the qualified visual-servo pipeline. Extend the pure frozen-axis controller with two command stages: eight 5 mm coarse targets followed by forty 0.5 mm fine targets. The GPU-safe runtime continues to provide live Lula FK, live plug validation, Lula IK preflight, target publication, diagnostics, and terminal hold behavior.

**Tech Stack:** Python 3, dataclasses, enums, NumPy, unittest, Isaac Sim 6.0.0, Lula IK/FK, PhysX GPU dynamics.

## Global Constraints

- Visual-servo completion remains 50 mm in front of the physical opening.
- Freeze ToolCenter position, orientation, and local +Z axis after final visual settle.
- Coarse approach depth: exactly 0.040 m.
- Coarse command increment: exactly 0.005 m; eight commands.
- Port opening depth from frozen start: exactly 0.050 m.
- Final insertion depth past the opening: exactly 0.010 m.
- Total axial travel: exactly 0.060 m.
- Fine command increment: exactly 0.0005 m; forty commands from 40.5 mm through 60.0 mm.
- Total command count: 48.
- Every target is computed from the frozen start pose.
- Orientation remains fixed.
- Each target requires <=0.0003 m tracking error for 6 consecutive simulation frames.
- Each step times out after 2.0 s.
- Abort at >0.0005 m lateral drift, >1.0 degree orientation error, mount-limit violation, topology failure, Lula failure, or target-publication failure.
- Completion and abort both hold the latest published target. No seating, release, retreat, or resumed perception.

---

### Task 1: Lock two-stage geometry with failing pure tests

**Files:**
- Modify: `single_rack_cv/tests/test_partial_insertion.py`

**Interfaces:**
- Consumes: `InsertionLimits`, `InsertionCommand`, `PartialInsertionController`.
- Requires command stage labels `COARSE_APPROACH` and `FINE_INSERTION`.

- [ ] Replace the old 20-step assertions with tests that require:
  - first command: 5.0 mm, coarse stage
  - command 8: 40.0 mm, coarse stage
  - command 9: 40.5 mm, fine stage
  - command 28: 50.0 mm total travel and 0.0 mm relative port depth
  - command 48: 60.0 mm total travel and +10.0 mm relative port depth
  - completion only after six settled frames on command 48
- [ ] Run `python -m unittest -v tests.test_partial_insertion` and verify RED because the current controller emits 0.5 mm from command 1 and completes at 10 mm.
- [ ] Commit the failing tests.

---

### Task 2: Implement two-stage frozen-axis sequencing

**Files:**
- Modify: `single_rack_cv/insertion.py`

**Interfaces:**
- Produce `InsertionStage` with `COARSE_APPROACH` and `FINE_INSERTION`.
- Extend `InsertionLimits` with `coarse_approach_depth_m`, `coarse_step_size_m`, and `opening_depth_m`; retain `total_depth_m=0.060` and `step_size_m=0.0005` as final fine-step configuration.
- Extend commands and metrics with depth relative to the opening.

- [ ] Implement exact command generation from frozen start:
  - while current depth < 0.040 m, next depth increases by 0.005 m
  - after 0.040 m, next depth increases by 0.0005 m
  - clamp only to exact stage boundaries and final 0.060 m
- [ ] Keep all existing settle, timeout, drift, orientation, mount, and terminal-state rules.
- [ ] Run the focused pure tests and verify GREEN.
- [ ] Commit.

---

### Task 3: Configure and wire exact geometry

**Files:**
- Modify: `single_rack_cv/config.py`
- Modify: `single_rack_cv/cable_runtime/__init__.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- `InsertionConfig.total_depth_m = 0.060`
- `InsertionConfig.coarse_approach_depth_m = 0.040`
- `InsertionConfig.coarse_step_size_m = 0.005`
- `InsertionConfig.opening_depth_m = 0.050`
- `InsertionConfig.step_size_m = 0.0005`

- [ ] Add failing config and runtime-source assertions for all exact values and both stage names.
- [ ] Run focused wiring tests and verify RED.
- [ ] Pass all configuration fields into `InsertionLimits`.
- [ ] Compute total command count from the controller rather than assuming one fixed step size.
- [ ] Update startup and event logs to report total travel, depth relative to opening, and stage.
- [ ] Change visual completion message to “begin 40 mm coarse approach, then 20 mm fine entry.”
- [ ] Run focused wiring and pure tests and verify GREEN.
- [ ] Commit.

---

### Task 4: Update operator documentation

**Files:**
- Modify: `single_rack_cv/README.md`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

- [ ] Require README text for `40 mm coarse approach`, `20 mm fine motion`, `48 commands`, and `10 mm inside the opening`.
- [ ] Document eight 5 mm commands, forty 0.5 mm commands, frozen +Z, unchanged abort limits, and hold behavior.
- [ ] Run focused tests and compile all modified Python files.
- [ ] Commit.

---

### Task 5: Verification and workstation qualification

- [ ] Run the full single-rack pure/structural unittest command from README.
- [ ] Run `python -m py_compile` on `insertion.py`, `config.py`, `main.py`, and the runtime facade.
- [ ] Verify the branch is ahead of and not behind `main`.
- [ ] On the Isaac workstation, require visual-servo completion followed by eight coarse commands and forty fine commands.
- [ ] Do not merge until the run reports 60.0 mm total travel, +10.0 mm port depth, no abort, and stable hold.