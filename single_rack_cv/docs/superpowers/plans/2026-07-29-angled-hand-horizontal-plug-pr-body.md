## What changed

- preserves the exact validated `/World/IK_Target` and `/World/ToolCenter` plug-tip world pose
- solves a new 30-degree downward `panda_hand` position and orientation around that preserved plug pose
- replaces fixed startup hand position, fixed startup hand orientation, and `hand_T_tool` together
- matches the previous working palm side without a guessed 180-degree local roll
- keeps both wrist cameras rigidly attached to the hand
- freezes the live PhysX plug nose axis for the existing two-stage insertion controller
- preserves all qualified insertion distances, settle requirements, abort limits, and terminal hold

## Root cause of the workstation failure

`use_fixed_start_pose=True` treats `initial_position` and `initial_orientation_wxyz` as the fixed hand pose. The prior implementation changed only `hand_T_tool`, so the hand stayed in the old pose and the plug frame rotated around it. The full log showed a 48.794-degree palm-side error, 14.393-degree plug horizontal error during startup, and a final wrong-pitch-sign abort.

## Verification completed

- 20/20 local geometry, insertion-axis, and runtime-wiring tests
- Python compilation of all changed modules
- regression proves exact plug pose preservation and rejects the old local-roll composition

## Merge gate

Keep this PR draft until Isaac Sim proves 30/30 mount validation, 30 +/- 0.5 degree pitch, <=1 degree palm-side error, <=1 degree plug horizontal error, stereo lock, all 48 insertion commands, and all existing tracking/drift/orientation/topology/timeout limits.
