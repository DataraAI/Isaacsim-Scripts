# Cable Mount Roll and Forward Offset Design

**Date:** 2026-07-28  
**Branch:** `feature/pregrasped-cable-mount`

## Goal

Change the pregrasped RJ45 presentation so the connector:

- keeps its insertion nose pointed along the existing ToolCenter insertion axis,
- rolls exactly 90 degrees around that axis,
- protrudes 30 mm farther forward from `panda_hand`,
- remains fixed to the existing rigid tracked plug,
- preserves the deformable cable tail and built-in plug-to-tail attachment.

## Geometry

The current mount maps the detected RJ45 tip directly onto the existing ToolCenter frame. The new mount will instead construct a presentation frame from the current ToolCenter frame:

1. Apply a +90 degree local roll around ToolCenter local `+Z`.
2. Translate the desired RJ45 tip 30 mm farther along ToolCenter local `+Z`.
3. Use this adjusted desired tip frame for the existing one-time cable-root placement.
4. Author the fixed joint from the resulting real hand-to-plug transform.

The RJ45 nose axis remains aligned with ToolCenter local `+Z`; only connector roll and forward placement change.

## Configuration

Add explicit cable-mount presentation settings to `CableMountConfig`:

```python
presentation_roll_deg: float = 90.0
forward_tip_offset_m: float = 0.030
```

These values belong to cable presentation, not the global IK ToolCenter calibration. The existing `tool_center_local_position_m` and camera geometry remain unchanged.

## Runtime behavior

The adjustment occurs once before physics starts. There will be:

- no per-frame transform updates,
- no proxy connector,
- no additional deformable attachment,
- no change to the fixed-joint topology,
- no change to insertion direction,
- no relaxation of mount validation limits.

## Validation

Pure geometry tests will verify:

- a 90 degree roll preserves the nose axis,
- the tip moves exactly 30 mm along ToolCenter local `+Z`,
- the resulting transform is rigid and right-handed,
- zero roll and zero offset reproduce the previous placement.

Runtime structural tests will verify that the new configuration is wired into pre-play placement only.

The Isaac workstation smoke test passes only if:

- the connector visibly protrudes beyond the fingertips,
- the connector is rolled 90 degrees relative to the current presentation,
- the fixed joint remains valid,
- the built-in deformable attachment remains unchanged,
- startup validation still satisfies tip error <= 0.5 mm and axis error <= 1.0 degree.

## Kill switch

Do not keep the change if the 30 mm offset causes arm instability, excessive deformable-tail loading, camera obstruction, fixed-joint failure, or mount validation failure. Do not compensate by widening validation tolerances or teleporting the cable after physics starts.
