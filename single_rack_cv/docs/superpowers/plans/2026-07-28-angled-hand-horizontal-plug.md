# Angled Hand with Horizontal RJ45 Plug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pitch the Franka hand downward by 30 degrees from the robot-right-side view while keeping the rigid RJ45 plug at the exact validated horizontal world pose and preserving the qualified RGB-stereo alignment and 48-command partial insertion.

**Corrected architecture:** Treat the validated plug-tip world pose as immutable. Solve the new fixed `panda_hand` world position, fixed `panda_hand` world orientation, and `hand_T_tool` rotation together so recomposition reproduces the original plug pose exactly. Do not change only `hand_T_tool`: with `use_fixed_start_pose=True`, that leaves the old hand pose in place and rotates the plug frame around it.

**Tech Stack:** Python 3, NumPy, Isaac Sim 6.0.0, USD/PhysX, Lula IK, `unittest`, existing RGB stereo + YOLOE pipeline.

## Global Constraints

- Base branch and rollback point: validated `main` at squash commit `d1ca7ad5dedb0e017eba9fd5dd731ad578a0c7a5`.
- Working branch: `feature/angled-hand-horizontal-plug`.
- Accepted pitch: exactly 30 degrees downward from the robot-right-side view.
- Direction contract: wrist higher, fingertips lower toward the port.
- Palm-side axis must match the previous working pose.
- Rigid plug nose/body remain at the validated horizontal world pose; flexible cable tail remains deformable and unconstrained.
- Cameras remain rigid children of `panda_hand`; do not create independent camera pose overrides.
- Visual servo remains translation-only and live-image-only; no manual pixel/depth offsets and no RTX/USD ground-truth runtime control.
- Existing insertion remains 40 mm coarse in eight 5 mm commands plus 20 mm fine in forty 0.5 mm commands, finishing 10 mm inside the opening.
- Do not relax the existing 0.5 mm mount-tip, 1 degree plug-axis, 0.3 mm final settle, 0.5 mm lateral-drift, 1 degree orientation, or 2 second step-timeout limits.
- Palm-side error relative to the previous working pose must be at most 1 degree.
- No seating, latch engagement, release, retreat, or tail-routing changes in this milestone.
- Do not touch the user's unrelated local deletion of `.vscode/settings.json`.
- Use the existing filenames; do not add cache-avoidance version suffixes.

## Pose Solver Contract

For the canonical startup geometry, the solver must:

- preserve the existing ToolCenter position and orientation exactly
- rotate the hand forward axis 30 degrees downward from the plug axis
- use the previous working palm-side axis
- move the wrist upward/back so `world_T_hand @ hand_T_tool` reconstructs the original ToolCenter pose
- replace `IKConfig.initial_position`, `IKConfig.initial_orientation_wxyz`, and `IKConfig.tool_center_local_orientation_wxyz` together

## Verification Summary

The implementation is not mergeable until the Isaac Sim workstation run proves all of the following:

- measured hand-to-plug pitch within 30 +/- 0.5 degrees
- palm-side error at or below 1 degree
- wrist above plug tip and requested pitch sign valid
- plug horizontal error at or below 1 degree
- mount validation 30/30, fixed joint valid, built-in attachment preserved
- RGB stereo acquisition, alignment lock, and visual-servo completion
- all 48 insertion commands settle
- final commanded depth remains +10 mm inside the opening
- all existing tracking, drift, orientation, mount, topology, IK, publication, and timeout gates pass

The terminal action remains hold. This plan does not include seating, release, or retreat.
