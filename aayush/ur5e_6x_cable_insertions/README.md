# ur5e_6x_cable_insertions (hover → orient → descend → grasp)

Minimal behaviour-tree demo on `DataHall_6r_ur5e.usd`:

1. Select a station (default `NegativeY_Top`).
2. Hover: **fingertips** at Network cable XY, Z = **370** stage units,
   fingers straight down — above the cable / CableBlocks (hand ~0.16 m higher).
3. Orient + tilt: wrist to `GRASP_ORIENTATION` (60° toward +X, fingers yawed
   90° about tool Z) so pads share X/Z and straddle the cable on ±Y.
4. Descend to cable neck `E_part006_44`, tip clamped into the Left/Right
   CableBlocks X-gap so knuckles clear the supports.
5. Close the gripper and squeeze-hold on the neck.
6. Lift the cable clear of the CableBlocks (`GRASP_LIFT_CLEARANCE_M`).
7. Maneuver to the port offset: at TipLift, yaw ~90° in place, then finish
   yaw to ±180° while moving to TipOffset
   (NegativeY CW +X→−Y→−X; PositiveY CCW +X→+Y→−X), landing at
   `mating_center` with **+X** standoff `PORT_APPROACH_X_OFFSET_M`
   (station jack features from cache + live RJ45 group pose).
8. Leave the simulation running until you stop Isaac.

Debug markers live under `/World/DebugPortMarkers/<station>/` and are **hidden
by default**. Toggle that Xform’s visibility in Isaac to show them
(`DEBUG_MARKER_VISIBLE_DEFAULT` in `config.py`).

## Run

```bash
/home/aayush/isaacsim/python.sh aayush/ur5e_6x_cable_insertions/main.py \
    --station NegativeY_Top
```

Optional: `--usd /path/to/DataHall_6r_ur5e.usd`, `--headless`,
`--max-frames N` (0 = no limit).

## Notes

- Other arms stay visible and idle (no BT).
- Descend / grasp / insert are still parked; extend `task_intelligence.json` later.
- Orient timing: `ORIENT_JOINT_STEPS` / `ORIENT_SETTLE_FRAMES` in `config.py`.

## Tuning (arm / gripper stability)

All knobs live in `config.py`. Start here if the arm wobbles, sleeps, or
overshoots:

| Knob | Role | If unstable… |
|---|---|---|
| `UR5E_ARM_DRIVE_PARAMETERS` | Meter-physical angular drive (K, D, maxForce); scene ÷mpu² | Must stay gravity-capable on cm DataHall |
| `UR5E_LIVE_STIFFNESS_MULTIPLIER` / `UR5E_LIVE_DAMPING_MULTIPLIER` | Isaac articulation PD after physics ready | Prefer ↑ KD over KP (overdamped) |
| `ROBOTIQ_DRIVE_PARAMETERS` | Finger joint drive | ↑ if fingers flop while open |
| `ROBOTIQ_MIMIC_NATURAL_FREQUENCY` / `ROBOTIQ_MIMIC_DAMPING_RATIO` | Soft mimic after root strip | ↑ freq if fingers lag; ↑ ratio if they oscillate |
| `ARTICULATION_SOLVER_POSITION_ITERS` / `_VELOCITY_ITERS` | PhysX articulation solver | ↑ pos iters (e.g. 32→64) if joints look soft |
| `ARTICULATION_SLEEP_THRESHOLD` / `_STABILIZATION_THRESHOLD` | When PhysX puts the arm to sleep | ↓ sleep if tracking stalls mid-move |
| `ARTICULATION_ENABLE_SELF_COLLISIONS` | Arm/gripper self-collision | Keep `False` |
| `LINK_DISABLE_GRAVITY` | Gravity on robot links | `False` (gravity on, like ur10e_1x) |
| `LINK_LINEAR_DAMPING` / `LINK_ANGULAR_DAMPING` | Viscous damping on every link | ↑ for less shake |
| `LINK_MAX_LINEAR_VELOCITY` / `LINK_MAX_ANGULAR_VELOCITY` | Velocity clamps (stage units / rad/s) | Lower if links explode; raise if motion is clipped |
| `JOINT_FRICTION` / `JOINT_ARMATURE` | Revolute joint friction + motor inertia | Small ↑ armature helps numerics |
| `SCENE_SOLVER_TYPE` / `SCENE_ENABLE_CCD` / `SCENE_*_ITERS` / `SCENE_BOUNCE_THRESHOLD` | Global PhysX scene | Keep `TGS` + CCD on for contacts |
| `PHYSICS_DT` / `RENDERING_DT` | Sim / render step | Smaller physics dt = more stable, slower |
| `HOVER_JOINT_STEPS` / `HOVER_SETTLE_FRAMES` | Trajectory duration / settle hold | ↑ for smoother, slower moves |
| `HOVER_Z_STAGE` / `TOOL_OFFSET_M` | Tip height / tip↔hand offset | Keep tip reach ≲0.85 m from base |
| `HOVER_IK_POS_TOLERANCE_M` / `HOVER_IK_ORI_TOLERANCE_RAD` | Lula IK acceptance | Loosen if IK fails; tighten for accuracy |
