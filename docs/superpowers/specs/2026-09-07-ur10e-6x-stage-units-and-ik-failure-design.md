# UR10e 6× Stage Units and IK Failure Design

**Date:** 2026-09-07  
**Status:** Approved  
**Location:** `aayush/ur10e_6x_cable_insertions/`

## Goal

Make all six UR10e stations physically execute observation, hover, descend,
grasp, lift, and port motions. The behavior tree must fail immediately when IK
cannot produce a motion instead of reporting an unmoved waypoint as reached.

## Known failure

`DataHall_6r.usd` uses `metersPerUnit=0.01`, while the six-arm primitives
calculate targets in metres. Runtime FK reports the end-effector position in
stage units (for example, approximately `[69, 16, 70]`), but targets are queued
as metre values (approximately `[0.5, -1.35, 3.2]`). IK therefore fails.

The shared controller currently leaves the joint-interpolation goal equal to
the starting joints after IK failure. It then advances after `joint_steps` and
the diagnostics print `REACHED`. Grasp validation can also pass a stationary
cable because it compares absolute cable height to the support height.

## Scope

- Apply changed behavior only to the six-arm demo.
- Keep all six-arm geometry, configuration, and behavior-tree data in metres.
- Keep the shared motion controller backward-compatible for other simulations.
- Do not modify or permanently rescale `DataHall_6r.usd`.
- Do not change port paths, station names, or behavior-tree structure.

## Design

### Unit boundary

Add a six-arm controller adapter around `FrankaMotionController`.

- Inputs from six-arm primitives remain in metres.
- Cartesian positions are converted to stage units exactly once when queued:
  `stage_position = metre_position / metersPerUnit`.
- The Lula base pose uses the same stage-unit coordinate space as queued
  targets and runtime FK.
- Runtime end-effector positions used by six-arm diagnostics are converted back
  to metres exactly once: `metre_position = stage_position * metersPerUnit`.
- Orientations and joint positions are unchanged.
- Reject non-positive or non-finite `metersPerUnit`.

This keeps application-level calculations readable in SI units while matching
the coordinate space observed by the articulation/Lula integration.

### Strict IK failure

The six-arm adapter marks itself failed when joint-interpolation IK fails and
stores a reason containing the waypoint label. It does not advance the command.
The existing Isaac behavior-tree adapter already checks `has_failed()` or
`_segment_failed`, so the current station returns `FAILURE` and clears its queue.
Other stations continue independently.

Diagnostics print `FAILED`, never `REACHED`, for an unsolved waypoint. Existing
shared-controller behavior remains unchanged for callers that do not use the
six-arm adapter.

### Grasp validation

At detection/queue time, store the cable grasp-part position before motion.
Validation requires all of the following:

1. The gripper is closed sufficiently.
2. The cable has risen by at least the configured lift threshold relative to
   its recorded initial position.
3. The cable tip is within the configured tolerance of the measured tool tip.

A stationary cable can no longer pass merely because it was authored above the
support floor.

### All stations

`build_scene` constructs the same adapter with the stage's
`metersPerUnit` for every selected station. No station-specific correction or
hard-coded scale is permitted.

## Error handling

- Invalid stage units fail scene construction with a clear error.
- IK failure fails only the affected station and includes its waypoint label.
- Missing initial grasp position fails grasp validation.
- Collision fallback warnings are not treated as motion success or failure;
  they are separate from this coordinate/IK correction.

## Testing

Host-side tests, written before implementation, cover:

1. Metres-to-stage and stage-to-metres conversion at `metersPerUnit=0.01`.
2. Identity conversion at `metersPerUnit=1.0`.
3. Invalid stage-unit rejection.
4. Six-arm IK initialization failure sets controller failure and does not
   advance the waypoint.
5. A stationary cable with closed fingers fails grasp validation.
6. A lifted cable near the measured tool tip passes validation.
7. Existing six-arm task/config tests continue to pass.

The final runtime check is one station (`NegativeY_Top`) before all six:
the arm must visibly move through hover and descend before finger closure, lift
the cable, and never print `REACHED` after an IK failure.

## Success criteria

1. Every selected station uses one consistent coordinate space internally.
2. The robot visibly reaches hover and descends before the close command.
3. Failed IK produces behavior-tree `FAILURE`, not action `SUCCESS`.
4. An unmoved cable cannot satisfy grasp validation.
5. The correction applies uniformly to all six station definitions without
   changing unrelated simulations.
