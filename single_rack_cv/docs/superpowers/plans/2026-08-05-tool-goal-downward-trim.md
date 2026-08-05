# ToolCenter Final Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the production RJ45 insertion path exactly 0.25 mm downward and 0.15 mm left in the camera view without moving the measured port point or changing perception, insertion depth, orientation, or safety limits.

**Architecture:** Add one pure helper that applies an explicit camera-view-left/world-negative-Y trim and world-Z downward trim to a ToolCenter goal. The generic stereo handoff defaults both trims to zero; the production position-hold runtime overrides them to 0.00015 m left and 0.00025 m down. Qualification still uses the unmodified camera-derived opening and unmodified 50 mm standoff, then only the frozen ToolCenter goal is trimmed.

**Tech Stack:** Python 3, NumPy, unittest, Isaac Sim 6.0.

## Global Constraints

- `/World/EstimatedPortPoint` and `/World/FrozenPortPoint` remain unchanged.
- Visible front-lip perception remains unchanged.
- The 48-command insertion sequence and +10 mm terminal depth remain unchanged.
- The 0.5 mm lateral and 1 degree orientation abort limits remain unchanged.
- Camera-view left maps to world negative Y in this setup.
- The production trim is exactly `Y -= 0.00015 m` and `Z -= 0.00025 m`.

---

### Task 1: Add the failing trim regression

**Files:**
- Create: `single_rack_cv/tests/test_tool_goal_trim.py`

- [ ] Write a test that applies the two trims to `[0.704262, -0.192331, 1.322690]` and expects `[0.704262, -0.192481, 1.322440]` while leaving the input unchanged.
- [ ] Assert negative or nonfinite trims fail closed.
- [ ] Assert production sets `_TOOL_GOAL_LEFT_TRIM_M = 0.00015` and `_TOOL_GOAL_DOWNWARD_TRIM_M = 0.00025` while the frozen opening point is still copied directly from `qualification.opening_position_m`.
- [ ] Run `~/isaacsim/python.sh -m unittest -v tests.test_tool_goal_trim` and verify it fails because the helper and production constants do not exist.

### Task 2: Implement the isolated production trim

**Files:**
- Create: `single_rack_cv/tool_goal_trim.py`
- Modify: `single_rack_cv/stereo_handoff_runtime.py`
- Modify: `single_rack_cv/handoff_position_hold_runtime.py`

- [ ] Add `apply_tool_goal_trim(tool_goal_position_m, left_trim_m, downward_trim_m)` with finite, nonnegative validation.
- [ ] Add zero-valued trim constants to the generic handoff runtime and apply the helper only after stationary qualification.
- [ ] Keep `_frozen_port_point_world_m` equal to the unmodified qualified opening.
- [ ] Set production trims to 0.00015 m left and 0.00025 m down in the position-hold runtime.
- [ ] Log both applied trims and the trimmed frozen ToolCenter goal.
- [ ] Run the focused test and the existing stereo handoff, position-hold, and two-stage insertion suites.
