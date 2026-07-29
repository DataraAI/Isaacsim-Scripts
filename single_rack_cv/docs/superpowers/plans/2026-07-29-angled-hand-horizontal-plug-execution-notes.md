# Angled Hand Execution Notes

The approved visual contract is unchanged: from the robot-right-side view, the wrist is higher, the fingertips slope downward toward the port by 30 degrees, and the rigid RJ45 plug remains horizontal.

## Corrections made during implementation

1. **Pitch axis correction**

   The executable transform uses a positive rotation about **tool-local Y**. For the validated horizontal plug pose, tool-local Y is the side-view pitch axis.

2. **Palm-roll correction from workstation evidence**

   The first workstation screenshot proved that pitch magnitude and direction were correct but the palm/finger presentation was flipped by 180 degrees around `panda_hand` local +Z. The earlier tests checked only the forward axis, wrist height, and plug horizontality, so they could not detect that roll error.

   The corrected transform composes a 180-degree hand-local-Z roll before the existing 30-degree pitch transform. This leaves the hand forward axis, horizontal plug axis, and insertion direction unchanged while matching the previous working palm orientation shown by the user.

3. **Configuration isolation**

   The pitch and palm-roll settings live in `angled_hand_config.py` instead of modifying the validated global `config.py`. This keeps the existing alignment and insertion configuration unchanged and makes rollback a one-line runtime import change.

4. **Insertion-axis separation**

   `plug_axis_insertion.py` replaces the legacy ToolCenter-local-Z axis only at the controller's existing freeze boundary. The qualified `PartialInsertionController` still owns all target generation, step counts, settle gates, drift checks, timeouts, aborts, and terminal holds.

5. **Camera behavior**

   No camera pose is overwritten. Both cameras remain rigid children of `panda_hand`; desired image geometry is recomputed because the corrected hand-to-tool transform is supplied before the base runtime constructs the scene.

## Verification status

The corrected pure geometry suite explicitly proves:

- 30-degree downward hand axis is unchanged
- horizontal plug axis is unchanged
- palm side axis matches the previous working pose
- the original flipped result is detected as a 180-degree palm-roll error

Isaac Sim workstation qualification remains mandatory before merge. The branch must demonstrate 30/30 mount validation, pitch within 30 +/- 0.5 degrees, palm-roll error at or below 1 degree, plug horizontal error at or below 1 degree, stereo acquisition and lock, all 48 commands settled, and the existing insertion completion limits.
