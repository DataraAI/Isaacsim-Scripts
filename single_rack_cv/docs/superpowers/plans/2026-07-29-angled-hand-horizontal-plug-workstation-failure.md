# Workstation Failure Evidence

The 2026-07-29 Isaac Sim run proved the local-roll architecture was wrong even though 19 synthetic tests passed.

Observed live values before abort:

- configured pitch: 30.000 degrees
- measured pitch: 30.000015 degrees
- palm-side error: 48.794353 degrees
- plug horizontal error: 14.393212 degrees during startup
- final fatal error: wrong hand pitch sign

The failure occurred because `IKConfig.use_fixed_start_pose=True` keeps `initial_position` and `initial_orientation_wxyz` as the fixed `panda_hand` pose. Changing only `hand_T_tool` rotated the ToolCenter/plug frame around that unchanged hand pose.

The replacement implementation preserves the original ToolCenter world pose and solves the new hand position, hand orientation, and `hand_T_tool` rotation together.
