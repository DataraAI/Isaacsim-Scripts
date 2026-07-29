# Angled Hand with Horizontal RJ45 Plug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pitch the Franka hand downward by 30 degrees from the robot-right-side view while keeping the rigid RJ45 plug horizontal and preserving the qualified RGB-stereo alignment and 48-command partial insertion.

**Architecture:** Keep `/World/IK_Target` and `/World/ToolCenter` as the horizontal plug-tip frame. Encode the 30 degree hand pitch and the previous working 180 degree palm presentation in the fixed `hand_T_tool` local rotation so Lula derives the intended `panda_hand` pose while the plug frame remains horizontal. During insertion, freeze the live PhysX plug nose axis explicitly instead of deriving travel from ToolCenter local +Z.

**Correction recorded after workstation evidence:** the first implementation had the correct downward hand axis but the palm/finger presentation was flipped by 180 degrees. The accepted geometry therefore requires both 30 degrees of downward pitch and a 180 degree `panda_hand`-local-Z palm roll. Tests must verify the palm side axis, not only the hand forward axis.

**Tech Stack:** Python 3, NumPy, Isaac Sim 6.0.0, USD/PhysX, Lula IK, `unittest`, existing RGB stereo + YOLOE pipeline.

## Global Constraints

- Base branch and rollback point: validated `main` at squash commit `d1ca7ad5dedb0e017eba9fd5dd731ad578a0c7a5`.
- Working branch: `feature/angled-hand-horizontal-plug`.
- Accepted pitch: exactly 30 degrees downward from the robot-right-side view.
- Accepted palm presentation: 180 degree hand-local-Z roll matching the previous working pose.
- Direction contract: wrist higher, fingertips lower toward the port.
- Rigid plug nose/body remain horizontal; flexible cable tail remains deformable and unconstrained.
- Cameras remain rigid children of `panda_hand`; do not create independent camera pose overrides.
- Visual servo remains translation-only and live-image-only; no manual pixel/depth offsets and no RTX/USD ground-truth runtime control.
- Existing insertion remains 40 mm coarse in eight 5 mm commands plus 20 mm fine in forty 0.5 mm commands, finishing 10 mm inside the opening.
- Do not relax the existing 0.5 mm mount-tip, 1 degree plug-axis, 0.3 mm final settle, 0.5 mm lateral-drift, 1 degree hand-orientation, or 2 second step-timeout limits.
- Palm-roll error relative to the previous working pose must be at most 1 degree.
- No seating, latch engagement, release, retreat, or tail-routing changes in this milestone.
- Do not touch the user's unrelated local deletion of `.vscode/settings.json`.
- Use the existing filenames; do not add cache-avoidance version suffixes.

## Verification summary

The implementation is not mergeable until the Isaac Sim workstation run proves all of the following:

- configured hand pitch: 30 degrees
- measured hand-to-plug pitch: within 30 +/- 0.5 degrees
- configured palm roll: 180 degrees
- measured palm-roll error: at or below 1 degree
- wrist above plug tip and requested pitch sign valid
- plug horizontal error at or below 1 degree
- mount validation 30/30, fixed joint valid, built-in attachment preserved
- RGB stereo acquisition, alignment lock, and visual-servo completion
- all 48 insertion commands settle
- final commanded depth remains +10 mm inside the opening
- all existing tracking, drift, orientation, mount, topology, IK, publication, and timeout gates pass

The terminal action remains hold. This plan does not include seating, release, or retreat.
