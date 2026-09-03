# ur10e_1x_cable_insertion Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** BT demo on `asset_spawn` that detects, 30°-grasps, and lifts `E_part006_44`.

**Architecture:** Importable `asset_spawn` scene + head physics + Lula motion + tanish BT.

**Tech Stack:** Isaac Sim, Lula, FrankaMotionController, tanish/behaviour_tree_insertion.

## Global Constraints

- Package path: `aayush/ur10e_1x_cable_insertion/`
- Reuse `asset_spawn.spawn.build_asset_spawn_scene`
- Physical pinch only (no FixedJoint)
- 30° approach with gripper aligned to path
- Respect UI Pause/Stop

---

### Task 1: Scaffold package + host test

**Files:**
- Create: `aayush/ur10e_1x_cable_insertion/{__init__,config,task_intelligence.json,README}.md` etc.
- Create: `aayush/ur10e_1x_cable_insertion/tests/test_runtime.py`

- [x] Write JSON tree + host unittest that renders the tree
- [x] Add config constants

### Task 2: Scene + primitives + main

**Files:**
- Create: `scene.py`, `primitives.py`, `main.py`
- Modify: `aayush/README.md`

- [x] Build scene with ParallelGripper + Lula + head physics
- [x] Implement detect / 30° grasp / validate
- [x] Main loop with pause/stop respect
- [x] Run host unittest
