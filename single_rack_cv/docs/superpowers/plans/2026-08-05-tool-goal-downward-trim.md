# ToolCenter Final Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the production RJ45 insertion path exactly 0.25 mm downward and 0.15 mm left in the camera view without moving the measured port point or changing perception, insertion depth, orientation, or safety limits.

**Architecture:** Add one pure helper that applies an explicit camera-view-left/world-negative-Y trim and world-Z downward trim to a ToolCenter goal. Keep the generic stereo handoff untouched. The production position-hold subclass intercepts the first handoff advance, trims the already-qualified ToolCenter goal once, then delegates to the proven handoff, position-hold, and insertion controllers.

**Tech Stack:** Python 3, NumPy, unittest, Isaac Sim 6.0.

## Global Constraints

- `/World/EstimatedPortPoint` and `/World/FrozenPortPoint` remain unchanged.
- Visible front-lip perception remains unchanged.
- The 48-command insertion sequence and +10 mm terminal depth remain unchanged.
- The 0.5 mm lateral and 1 degree orientation abort limits remain unchanged.
- Camera-view left maps to world negative Y in this setup.
- The production trim is exactly `Y -= 0.00015 m` and `Z -= 0.00025 m`.

---

### Task 1: Add the trim regression

**Files:**
- Create: `single_rack_cv/tests/test_tool_goal_trim.py`

- [x] Test `[0.704262, -0.192331, 1.322690]` becomes `[0.704262, -0.192481, 1.322440]` while the input remains unchanged.
- [x] Test negative or nonfinite trims fail closed.
- [x] Assert production owns the exact trim constants and the generic handoff still copies the physical opening unchanged.

### Task 2: Implement the isolated production trim

**Files:**
- Create: `single_rack_cv/tool_goal_trim.py`
- Modify: `single_rack_cv/handoff_position_hold_runtime.py`

- [x] Add `apply_tool_goal_trim(tool_goal_position_m, left_trim_m, downward_trim_m)` with finite, nonnegative validation.
- [x] Override `_advance_handoff_if_settled` only in the production position-hold runtime.
- [x] Apply the trim once before the first handoff step.
- [x] Keep `_frozen_port_point_world_m` and the generic stereo handoff unchanged.
- [x] Log both applied trims and the trimmed ToolCenter goal.
- [ ] Run the focused test and existing handoff/insertion regression suites on the Isaac Sim workstation.
