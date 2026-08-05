# ToolCenter Downward Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the production RJ45 insertion path exactly 0.25 mm downward without moving the measured port point or changing perception, insertion depth, orientation, or safety limits.

**Architecture:** Add one pure helper that applies a nonnegative world-Z downward trim to a ToolCenter goal. The generic stereo handoff defaults to zero trim; the production position-hold runtime overrides the trim to 0.00025 m. Qualification still uses the unmodified camera-derived opening and 50 mm standoff, then only the frozen ToolCenter goal is trimmed.

**Tech Stack:** Python 3, NumPy, unittest, Isaac Sim 6.0.

## Global Constraints

- `/World/EstimatedPortPoint` and `/World/FrozenPortPoint` remain unchanged.
- Visible front-lip perception remains unchanged.
- The 48-command insertion sequence and +10 mm terminal depth remain unchanged.
- The 0.5 mm lateral and 1 degree orientation abort limits remain unchanged.
- The trim is exactly 0.00025 m downward in world Z.

---

### Task 1: Add the failing trim regression

**Files:**
- Create: `single_rack_cv/tests/test_tool_goal_downward_trim.py`

- [ ] Write a test that imports `apply_downward_tool_goal_trim`, applies 0.00025 m to `[0.704262, -0.192331, 1.322690]`, and expects `[0.704262, -0.192331, 1.322440]` while leaving the input unchanged.
- [ ] Assert production sets `_TOOL_GOAL_DOWNWARD_TRIM_M = 0.00025` and the frozen opening point is still copied directly from `qualification.opening_position_m`.
- [ ] Run `~/isaacsim/python.sh -m unittest -v tests.test_tool_goal_downward_trim` and verify it fails because the helper and production constant do not exist.

### Task 2: Implement the isolated production trim

**Files:**
- Modify: `single_rack_cv/stereo_handoff.py`
- Modify: `single_rack_cv/stereo_handoff_runtime.py`
- Modify: `single_rack_cv/handoff_position_hold_runtime.py`

- [ ] Add `apply_downward_tool_goal_trim(tool_goal_position_m, downward_trim_m)` with finite, nonnegative validation.
- [ ] Add `_TOOL_GOAL_DOWNWARD_TRIM_M = 0.0` to the generic handoff runtime and apply it only after stationary qualification.
- [ ] Keep `_frozen_port_point_world_m` equal to the unmodified qualified opening.
- [ ] Set `_TOOL_GOAL_DOWNWARD_TRIM_M = 0.00025` in the production position-hold runtime.
- [ ] Log the applied trim and trimmed frozen ToolCenter goal.
- [ ] Run the focused test and the existing stereo handoff, position-hold, and two-stage insertion suites.
