# Angled Hand Execution Notes

The approved visual contract is unchanged: from the robot-right-side view, the wrist is higher, the fingertips slope downward toward the port by 30 degrees, and the rigid RJ45 plug remains horizontal.

## Root cause proven by the workstation log

The first two implementations changed only `hand_T_tool`. That was the wrong control point because `IKConfig.use_fixed_start_pose=True` defines `initial_position` and `initial_orientation_wxyz` as a fixed **panda_hand pose**. `_create_ik()` then derives the ToolCenter target from that fixed hand pose.

Changing `hand_T_tool` while leaving the fixed hand pose untouched therefore rotates the plug/ToolCenter frame around the old hand pose. It does not create the requested angled hand around the existing plug. The workstation evidence exposed the mismatch:

- configured pitch: 30 degrees
- palm-side error: 48.794 degrees
- plug horizontal error during startup: 14.393 degrees
- final startup abort: wrong hand pitch sign

The passing synthetic tests were incomplete because they assumed a fixed world ToolCenter orientation that the runtime did not actually preserve.

## Corrected architecture

`compute_angled_hand_pose_preserving_tool()` now treats the validated world plug-tip pose as immutable and solves three linked values together:

1. the new fixed `panda_hand` world position
2. the new fixed `panda_hand` world orientation
3. the new `hand_T_tool` rotation

For the canonical startup pose, the solver keeps the ToolCenter at `[0.7666, -0.1375, 1.3000]`, moves the wrist to approximately `[0.882128, -0.1375, 1.3667]`, sets the hand forward axis to `[-0.866025, 0, -0.5]`, and sets the palm-side axis to `[0, -1, 0]`. Recomposition of `world_T_hand @ hand_T_tool` reproduces the original plug position and orientation to numerical precision.

There is no guessed 180-degree local palm roll in the corrected implementation.

## Preserved behavior

- Both cameras remain rigid children of `panda_hand`.
- Desired stereo geometry is recomputed from the solved hand-to-tool transform.
- `plug_axis_insertion.py` still replaces the legacy ToolCenter-local-Z direction only at the insertion freeze boundary.
- The existing 48-command insertion controller still owns all target generation, settle gates, drift checks, timeouts, aborts, and terminal hold.
- No seating, release, retreat, or threshold relaxation is included.

## Verification status

The local pure and structural suite passes 20/20 tests and Python compilation. The regression suite proves:

- exact preservation of the validated plug world position
- exact preservation of the validated plug world orientation
- 30-degree downward hand forward axis
- wrist above the plug tip
- palm-side axis matching the older working pose
- the previous local-roll composition rotates the plug by 30 degrees in world XY and is rejected

Isaac Sim workstation qualification remains mandatory before merge. The branch must demonstrate 30/30 mount validation, pitch within 30 +/- 0.5 degrees, palm-side error at or below 1 degree, plug horizontal error at or below 1 degree, stereo acquisition and lock, all 48 commands settled, and the existing insertion completion limits.
