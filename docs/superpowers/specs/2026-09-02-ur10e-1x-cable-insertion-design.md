# ur10e_1x_cable_insertion Design

**Date:** 2026-09-02  
**Status:** Approved  
**Location:** `aayush/ur10e_1x_cable_insertion/`

## Goal

Extend the `asset_spawn` layout with a behaviour-tree task that detects
`E_part006_44` on crystal head 45, approaches at **30°** (path + gripper tilt)
relative to the work table, physically grasps the part, and lifts the cable.

## Non-goals

- Rack / ethernet port insertion
- FixedJoint weld grasp

## Architecture

1. Reuse `asset_spawn.spawn.build_asset_spawn_scene()` (includes U-notch under head45).
2. Enable one rigid body per crystal head (Option A physical pinch).
3. Lula + `FrankaMotionController` + `tanish/behaviour_tree_insertion`.

## Tree

```text
Sequence: Grasp and lift ethernet cable head
├── Move to observation pose
├── Detect E_part006_44
├── 30° tilted hover → open → descend → close → lift
└── Confirm cable held
```

## Grasp motion

- Hover near head39 end; tool +Z along the 30° approach toward head45.
- Pinch `E_part006_44`; lift world +Z while holding.
- Pause/Stop respected (no forced `world.play()` in the tick loop).
