# Angled Hand with Horizontal Plug Design

## Goal

From the robot-right-side view, hold the rigid RJ45 plug horizontally while the Franka wrist is higher and the fingers slope downward toward the port by 30 degrees. Keep that geometry fixed through startup, RGB-stereo alignment, coarse approach, and partial insertion. The flexible cable tail remains free.

## Correct Frame Contract

The validated plug-tip world pose is the invariant. `/World/IK_Target` and `/World/ToolCenter` remain that horizontal plug-tip frame.

The runtime must solve these three quantities together:

1. fixed startup `panda_hand` world position
2. fixed startup `panda_hand` world orientation
3. `hand_T_tool` rotation

The solution must satisfy:

```text
world_T_hand @ hand_T_tool == validated world_T_plug_tip
```

Changing only `hand_T_tool` is invalid because `use_fixed_start_pose=True` keeps the old hand pose and derives the ToolCenter target from it. That rotates the plug frame around the old hand rather than producing the requested angled hand around the preserved plug.

## Desired Hand Frame

For a horizontal plug axis pointing toward the port:

- hand forward axis: plug axis rotated 30 degrees downward
- palm-side axis: previous working palm presentation
- wrist position: solved upward/back from the preserved plug tip using the existing 133.4 mm hand-to-tool offset

For the canonical startup pose, the expected values are approximately:

```text
preserved ToolCenter: [0.7666, -0.1375, 1.3000]
solved panda_hand:    [0.882128, -0.1375, 1.3667]
hand forward axis:   [-0.866025, 0.0, -0.5]
palm-side axis:      [0.0, -1.0, 0.0]
```

## Camera and Perception Behavior

Both wrist cameras remain rigid children of `panda_hand`. No independent camera transform is authored. Desired stereo geometry is recomputed from the solved `hand_T_tool` before scene construction. Runtime control remains live-image-only and translation-only.

## Insertion Behavior

Insertion freezes the live PhysX plug nose axis rather than assuming ToolCenter local +Z after the hand and plug frames are separated. The existing two-stage sequence is unchanged:

- 40 mm coarse approach: eight 5 mm commands
- 20 mm fine motion: forty 0.5 mm commands
- final commanded depth: 10 mm inside the opening
- terminal action: hold

No seating, release, or retreat is included.

## Validation Gates

Startup must reject the run when any of these fail:

- hand-to-plug pitch within 30 +/- 0.5 degrees
- wrist above plug tip
- requested downward sign valid
- palm-side error <= 1 degree
- plug horizontal error <= 1 degree
- mount tip error <= 0.5 mm
- fixed joint valid
- built-in deformable attachment preserved
- GPU dynamics active

The branch remains unmergeable until the workstation run also proves stereo acquisition, alignment lock, all 48 insertion commands, and every existing tracking, drift, orientation, topology, IK, publication, and timeout limit.
