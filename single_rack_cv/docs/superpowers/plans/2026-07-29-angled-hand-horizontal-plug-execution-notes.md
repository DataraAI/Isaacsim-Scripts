# Angled Hand Execution Notes

The approved visual contract is unchanged: from the robot-right-side view, the wrist is higher, the fingertips slope downward toward the port by 30 degrees, and the rigid RJ45 plug remains horizontal.

## Corrections made during implementation

1. **Pitch axis correction**

   The executable transform uses a positive rotation about **tool-local Y**. For the validated horizontal plug pose, tool-local Y is the side-view pitch axis. Rotating about local X would move the hand sideways rather than create the requested wrist-high/fingertips-low profile.

2. **Configuration isolation**

   The one shared pitch setting lives in `angled_hand_config.py` instead of modifying the validated global `config.py`. This keeps the existing alignment and insertion configuration byte-for-byte unchanged and makes rollback a one-line runtime import change.

3. **Insertion-axis separation**

   `plug_axis_insertion.py` replaces the legacy ToolCenter-local-Z axis only at the controller's existing freeze boundary. The qualified `PartialInsertionController` still owns all target generation, step counts, settle gates, drift checks, timeouts, aborts, and terminal holds.

4. **Camera behavior**

   No camera pose is overwritten. Both cameras remain rigid children of `panda_hand`; desired image geometry is recomputed because the pitched hand-to-tool transform is supplied before the base runtime constructs the scene.

## Verification status

The following tests passed in the isolated implementation workspace:

- `tests.test_hand_plug_geometry`: 7 tests
- `tests.test_plug_axis_insertion`: 3 tests
- `tests.test_angled_hand_runtime_wiring`: 5 tests
- Python compilation of all new modules and modified `main.py`

Isaac Sim workstation qualification remains mandatory before merge. The branch must demonstrate 30/30 mount validation, valid requested pitch sign, plug horizontal error at or below 1 degree, stereo acquisition and lock, all 48 commands settled, and the existing insertion completion limits.
