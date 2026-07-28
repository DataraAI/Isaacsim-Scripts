# Pregrasped Deformable Cable Mount Design

**Date:** 2026-07-23  
**Branch:** `feature/pregrasped-cable-mount`  
**Status:** Approved corrected topology

## Goal

Start `single_rack_cv` with the supplied network cable already mounted in the Franka hand so the runtime tests port perception and pre-insert alignment rather than grasp acquisition.

The tracked RJ45 insertion tip becomes the physical meaning of `/World/ToolCenter`. The connector remains permanently mounted for the complete run. The cable tail remains deformable.

## Asset topology established by the workstation probe

The composed asset contains:

- cable root: `/World/NetworkCable`,
- tracked rigid plug: `/World/NetworkCable/E_crystal_head1_45`,
- deformable tail: `/World/NetworkCable/E_line_35`,
- existing auto deformable attachment: `/World/NetworkCable/E_line_35/attachment`,
- second rigid plug: `/World/NetworkCable/E_crystal_head2_39`,
- second existing auto deformable attachment: `/World/NetworkCable/E_line_35/attachment_01`.

The tracked plug has `PhysicsRigidBodyAPI` and `PhysxRigidBodyAPI`. The tail has `OmniPhysicsDeformableBodyAPI`. The existing attachment already connects the deformable tail to the tracked rigid plug:

```text
attachable0 = /World/NetworkCable/E_line_35
attachable1 = /World/NetworkCable/E_crystal_head1_45
```

This topology is a startup invariant. The runtime may verify it, but must not replace, retarget, duplicate, mask, or delete the asset-authored attachment.

## Required asset paths

```python
NETWORK_CABLE_USD_PATH = (
    "/home/aayush/isaacsim_assets/Network cable 001/"
    "model_Networkcable1_69323.usd"
)
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
TRACKED_PLUG_PRIM_PATH = (
    "/World/NetworkCable/E_crystal_head1_45"
)
```

Missing files, prims, rigid-body schemas, deformable schemas, or the expected plug-to-tail attachment are fatal startup errors.

## Corrected mounting architecture

Use the existing rigid tracked plug as the mounted body.

1. Load the cable before physics starts.
2. Verify exactly one Omni Physics deformable body under the cable root.
3. Verify exactly one existing `PhysxAutoDeformableAttachmentAPI` connects that deformable body to the tracked plug.
4. Detect the RJ45 insertion tip and insertion frame automatically from composed plug geometry.
5. Move `/World/NetworkCable` once so the RJ45 tip coincides with `/World/ToolCenter` and points along ToolCenter local `+Z`.
6. Create `/World/CableMountFixedJoint` directly between the dynamically discovered `panda_hand` rigid body and the existing tracked plug rigid body.
7. Preserve both asset-authored deformable attachments unchanged.
8. Filter collisions only between the tracked plug and the Franka hand/finger rigid bodies so the permanent joint does not fight cosmetic gripper contact.
9. Preserve tracked-plug collisions against the rack, port, cable tail, and ground.
10. Keep the fixed joint active for the complete process lifetime.
11. Never overwrite cable or plug transforms after physics begins.

There is no rigid proxy, new deformable attachment, attachment mask, release mechanism, or per-frame cable teleport.

## Rejected alternatives

### Hand-mounted proxy plus a new deformable attachment

Rejected after inspecting the actual asset. The tail is already attached to the rigid plug. Adding another attachment would duplicate constraints and risk jitter, solver conflict, or an over-constrained cable end.

### Replacing the built-in attachment

Rejected because the existing attachment already connects the correct deformable and rigid bodies. Reauthoring it adds risk without adding capability.

### Moving the cable every frame

Rejected because it bypasses physics, injects energy into the deformable tail, and makes later contact behavior untrustworthy.

### Rigidly carrying the complete cable

Rejected because it destroys the deformable-tail behavior.

## ToolCenter semantics

The existing calibrated transform remains numerically unchanged:

```text
panda_hand_T_rj45_tip = panda_hand_T_toolcenter
```

The cable is mounted to the existing frame rather than moving the frame to the cable. This preserves camera geometry, Lula IK conversion, and the 50 mm pre-insert calibration.

After mounting:

- `/World/IK_Target` commands the RJ45 tip,
- `/World/ToolCenter` reports the actual RJ45 tip,
- the pre-insert distance is measured from the port opening to the RJ45 tip,
- future insertion advances this same frame along the port axis.

No second connector offset may be added elsewhere.

## Automatic plug-frame detection

A pure NumPy component computes the connector frame from composed USD geometry.

1. Compute tracked-plug bounds in plug-local coordinates.
2. Select the longest dimension as the longitudinal axis.
3. Require `longest / second_longest >= 1.5`.
4. Transform the cable-root world-bounds center into plug-local coordinates.
5. Project the cable-center vector onto the longitudinal axis.
6. Treat that direction as the cable side.
7. Treat the opposite end as the RJ45 nose.
8. Use the nose-face center as the insertion tip.
9. Align the nose axis to ToolCenter local `+Z`.
10. Align the widest transverse plug axis to ToolCenter local `+Y`.

Ambiguous axes or cable-side projections are fatal. World `-X` is not hard-coded.

## Components

### `cable_geometry.py`

Pure Python and NumPy. Owns bounds validation, axis selection, nose detection, deterministic roll, one-time root-transform calculation, angular error, and validation-window logic.

### `cable_mount.py`

Isaac/USD/PhysX integration. Owns asset loading, dynamic hand discovery, topology verification, one-time placement, direct fixed-joint creation, collision filtering, finger presentation, mount validation, and diagnostics.

### `config.py`

`CableMountConfig` contains:

- enable flag,
- asset/root/tracked-plug paths,
- fixed-joint path,
- hand and finger link names,
- axis ambiguity ratio,
- cable-side projection threshold,
- finger clearance,
- settle-frame counts,
- tip and axis tolerances.

It must not contain proxy, new-attachment, or mask paths.

### `sim.py`

Integrates cable mounting as startup infrastructure. Existing translation-only visual-servo behavior remains unchanged. No insertion path is added.

Expose:

```python
runtime.prepare_for_perception()
```

This advances simulation, updates IK, settles the mounted cable, validates the mount, and returns only after validation passes.

### `main.py`

Startup order is:

```text
create SimulationRuntime
→ runtime.prepare_for_perception()
→ initialize DebugOutputs
→ initialize YOLOE
→ enter canonical visual-servo loop
```

YOLOE initialization and RGB acquisition never occur when mount validation fails.

## GPU physics

The shared scene uses:

```text
scene device: cuda:0
GPU dynamics: enabled
broadphase: GPU
solver: TGS
physics timestep: 1/60 s initially
```

Failure to activate the required physics settings is fatal.

## Finger presentation

The fixed joint carries the connector; the fingers are cosmetic.

1. Determine the plug width aligned with ToolCenter local `+Y`.
2. Add `1.0 mm` total clearance.
3. Set both finger joints symmetrically to half the total gap.
4. Clamp to articulation limits.
5. Hold those position targets during the run.

The tracked plug is collision-filtered against the hand and fingers so cosmetic presentation cannot compete with the fixed joint.

## Startup validation

During 30 consecutive validation frames, measure every frame:

- RJ45 tip position versus ToolCenter,
- connector nose axis versus ToolCenter local `+Z`,
- direct hand-to-plug fixed-joint validity,
- tracked-plug rigid-body validity,
- deformable-tail validity,
- preservation of the existing plug-to-tail attachment targets,
- GPU-dynamics state.

Pass only when every frame satisfies:

```text
maximum tip error <= 0.5 mm
maximum axis error <= 1.0 degree
fixed joint valid
tracked plug rigid body valid
existing plug-to-tail attachment unchanged and valid
deformable tail valid
GPU dynamics enabled
```

One bad frame fails startup. No averaging hides intermittent mount motion.

## Diagnostics

Successful startup prints:

```text
[CABLE MOUNT]
  cable USD: ...
  tracked plug: /World/NetworkCable/E_crystal_head1_45
  deformable body: /World/NetworkCable/E_line_35
  preserved attachment: /World/NetworkCable/E_line_35/attachment
  plug dimensions mm: [...]
  insertion-tip local position m: [...]
  hand-to-tip translation m: [...]
  hand-to-tip orientation WXYZ: [...]
  finger total gap mm: ...
  validation frames: 30/30
  maximum tip error mm: ...
  maximum axis error deg: ...
  fixed joint: valid
  built-in attachment: preserved
  cable tail: deformable
  GPU dynamics: enabled
```

Failures print the same measured context and a specific fatal reason.

## Tests

### Pure tests

Cover X/Y/Z longitudinal axes, both cable-side signs, rotated transforms, deterministic roll, ToolCenter mapping, non-finite data, degenerate bounds, ambiguous aspect ratios, ambiguous cable-side projections, and one-bad-frame validation failure.

### Structural tests

Prove that:

- the tracked plug is used as fixed-joint body 1,
- `panda_hand` is discovered dynamically,
- the existing auto attachment is verified and not authored,
- no proxy, new deformable attachment, mask shape, release path, or insertion path exists,
- no post-play cable or plug transform exists,
- GPU dynamics is required,
- mount validation precedes YOLOE.

### Isaac workstation smoke test

Require:

- no invalid rigid-joint or deformable-attachment errors,
- the built-in attachment relationships remain unchanged,
- the connector remains mounted while the robot and cable settle,
- 30/30 validation frames pass,
- the tail visibly deforms and settles,
- nominal visual alignment completes,
- physical ToolCenter tracking error remains `<= 0.3 mm`,
- target steps remain `<= 1.0 mm`,
- no insertion occurs.

## Kill switch

Do not begin insertion or merge the feature if the fixed joint is unstable, the built-in attachment changes or fails, the cable tail destabilizes the arm/cameras, tip error exceeds `0.5 mm`, axis error exceeds `1 degree`, or nominal visual alignment fails under GPU dynamics.

Fix the asset, joint, or physics defect. Do not widen limits, duplicate attachments, rigidify the complete cable, or add per-frame transforms.