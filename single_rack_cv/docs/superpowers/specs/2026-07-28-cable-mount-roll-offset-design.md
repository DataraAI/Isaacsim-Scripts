# Cable Mount Roll and Forward Offset Design

**Date:** 2026-07-28  
**Branch:** `feature/pregrasped-cable-mount`  
**Status:** Approved corrected ToolCenter calibration

## Goal

Change the pregrasped RJ45 mounting pose so the connector:

- keeps the same insertion direction,
- rolls exactly 90 degrees around its insertion axis,
- protrudes 30 mm farther from `panda_hand`,
- remains the physical meaning of `/World/ToolCenter`,
- preserves the real tracked plug, direct fixed joint, deformable tail, and built-in plug-to-tail attachment.

## Correct geometry contract

The earlier presentation-only proposal was rejected because it would put the physical RJ45 tip 30 mm ahead of `/World/ToolCenter`. That would silently reduce the physical 50 mm pre-insert standoff to 20 mm and corrupt future insertion depth.

The calibrated hand-to-ToolCenter transform must instead be updated directly:

```python
tool_center_local_position_m = (
    0.0,
    0.0,
    0.1334,
)
tool_center_local_orientation_wxyz = (
    0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865475,
)
```

The previous local translation was `0.1034` m, so the new transform extends ToolCenter exactly `0.0300` m along `panda_hand` local `+Z`. The quaternion is a positive 90 degree rotation around the same local `+Z` axis.

This keeps the RJ45 nose axis unchanged while rotating its transverse orientation by 90 degrees.

## Runtime behavior

`cable_runtime.py` already computes the world ToolCenter pose from the calibrated hand-to-tool transform before loading and mounting the cable. `cable_mount.py` already maps the detected RJ45 tip directly onto that ToolCenter frame and authors the fixed joint from the resulting real hand-to-plug transform.

Therefore:

- `config.py` is the only production file that changes,
- `cable_mount.py` remains unchanged,
- no presentation-offset helper is added,
- no second connector offset exists,
- no post-play transform is authored,
- no proxy or additional deformable attachment is created,
- the visual-servo 50 mm standoff remains physically correct,
- future insertion depth remains measured from the real RJ45 tip.

Camera transforms remain unchanged because the stereo cameras are mounted to `panda_hand`, not `/World/ToolCenter`.

## Validation

Repository tests verify the exact ToolCenter position and quaternion and reject presentation-only cable-mount fields.

The Isaac workstation smoke test passes only if:

- the connector is visibly rolled 90 degrees around its nose axis,
- the connector tip protrudes 30 mm farther beyond the fingers,
- the nose still points in the same insertion direction,
- the fixed joint remains valid,
- the built-in deformable attachment remains unchanged,
- startup validation still satisfies tip error `<= 0.5 mm` and axis error `<= 1.0 degree`,
- the physical pre-insert distance remains 50 mm from the RJ45 tip.

## Kill switch

Revert the calibration if the new lever arm causes camera obstruction, arm instability, deformable-tail overload, fixed-joint failure, attachment failure, or mount-validation failure. Do not compensate by widening tolerances, adding a hidden offset, or teleporting the cable after physics starts.
