# Pregrasped Deformable Cable Mount Design

**Date:** 2026-07-23  
**Branch:** `feature/pregrasped-cable-mount`  
**Status:** Approved concept awaiting written-spec review

## Goal

Start `single_rack_cv` with the network cable already mounted in the Franka hand. The project will test port perception, pre-insert alignment, and later insertion—not grasp acquisition.

The RJ45 insertion tip becomes the physical meaning of `/World/ToolCenter`. The connector remains permanently attached for the complete run while the cable tail remains deformable.

## Non-goals

This change does not add grasping, grip-force validation, slip, release, regrasp, insertion motion, per-frame cable teleporting, or a manual connector-depth offset.

## Required asset

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

Missing files or prims are fatal startup errors.

## Chosen architecture

Use a **world-level rigid mount proxy fixed to `panda_hand`, with a deformable attachment limited to the connector geometry**.

1. Load the cable before physics starts.
2. Detect the RJ45 tip and insertion frame automatically.
3. Move the cable root once so the tip equals ToolCenter and points toward the rack.
4. Create `/World/CableMountProxy` as a hidden rigid actor.
5. Create `/World/CableMountFixedJoint` between the discovered `panda_hand` rigid body and the proxy.
6. Attach only cable vertices inside the tracked plug bounds to the proxy.
7. Keep the attachment for the entire process lifetime.
8. Never overwrite cable transforms after physics begins.

The proxy is not nested under `panda_hand`; nested rigid bodies create ambiguous PhysX ownership.

### Rejected alternatives

- Directly fixed-jointing `E_crystal_head1_45`: the prim may not be an independent rigid actor.
- Reparenting the plug: breaks the deformable asset hierarchy.
- Moving the cable every frame: bypasses physics and can inject instability.
- Rigidly carrying the complete cable: destroys the deformable-tail behavior.

## ToolCenter semantics

The existing configured transform remains numerically unchanged:

```text
panda_hand_T_rj45_tip = panda_hand_T_toolcenter
```

The cable is mounted to that frame instead of moving the frame to the cable. This preserves the existing camera geometry, Lula IK conversion, and 50 mm pre-insert calibration.

After mounting:

- `/World/IK_Target` commands the RJ45 tip.
- `/World/ToolCenter` reports the actual RJ45 tip.
- The pre-insert distance is measured from the port opening to the RJ45 tip.
- Future insertion advances this same frame along the port axis.

No connector offset may be added anywhere else.

## Automatic plug-frame detection

A pure NumPy module computes the connector frame from composed USD geometry.

### Longitudinal axis

1. Compute the tracked plug bounds in plug-local coordinates.
2. Select the longest dimension as the insertion axis.
3. Require `longest / second_longest >= 1.5` by default.
4. Abort if the ratio is smaller.

### Nose direction

1. Transform the cable-root world-bounds center into plug-local coordinates.
2. Project the vector from plug center to cable center onto the longitudinal axis.
3. The projected direction is the cable side.
4. The opposite direction is the RJ45 nose.
5. Require the projection magnitude to exceed a configured ambiguity threshold; otherwise abort.

The insertion tip is the center of the nose-side bounding-box face.

### Roll

Align the plug's widest transverse axis with ToolCenter local `+Y`. Construct the remaining axis as a right-handed basis. This makes roll deterministic.

### Final mount transform

Align:

- plug nose axis → ToolCenter local `+Z`,
- widest transverse axis → ToolCenter local `+Y`,
- detected nose-face center → ToolCenter origin.

ToolCenter local `+Z` is derived from its world transform. World `-X` is not hard-coded.

Only `/World/NetworkCable` receives this initial transform. The tracked plug child is never moved independently.

## Deformable-body discovery

Select the deformable body deterministically:

1. Walk upward from `TRACKED_PLUG_PRIM_PATH` and select the first ancestor carrying the required deformable-body API.
2. If none exists, search cable-root descendants whose world bounds contain the plug center.
3. Require exactly one valid candidate.
4. Abort on zero or multiple candidates.

The chosen prim path is printed in startup diagnostics.

## Attachment region

The proxy attachment volume is derived from the tracked plug local bounds:

- cover the complete plug bounds,
- add `0.5 mm` default padding on both transverse directions,
- add `0.5 mm` at the nose face,
- add **zero** extension beyond the cable-side longitudinal face.

This captures the connector while avoiding attachment of the cable tail beyond the plug.

The proxy collider is used for attachment generation. Collision between the proxy and Franka links is filtered out so the hidden proxy cannot fight the hand or fingers.

## Components

### `cable_geometry.py`

Pure Python/NumPy, with no Isaac, USD, PhysX, or `omni` imports.

It owns bounds validation, axis selection, nose detection, insertion-frame construction, attachment-volume calculation, and the cable-root transform onto ToolCenter.

### `cable_mount.py`

Isaac/USD/PhysX integration.

It owns asset loading, dynamic hand discovery, deformable-body discovery, geometry queries, initial cable placement, proxy creation, fixed-joint creation, deformable attachment authoring, finger positioning, mount validation, and diagnostics.

### `config.py`

Add frozen `CableMountConfig` fields for:

- enable flag,
- asset/root/plug paths,
- proxy/joint paths,
- axis ambiguity ratio,
- cable-side projection threshold,
- attachment padding,
- finger clearance,
- settle-frame counts,
- tip and axis tolerances.

Required defaults:

```text
axis ambiguity ratio: 1.5
attachment padding: 0.5 mm
finger total clearance: 1.0 mm
maximum tip error: 0.5 mm
maximum axis error: 1.0 degree
validation window: final 30 settled frames
```

### `sim.py`

Integrate cable creation as startup infrastructure. Existing visual-servo control behavior remains unchanged and no insertion path is added.

Expose:

```python
runtime.prepare_for_perception()
```

This method advances simulation, updates IK, settles the robot/cable, validates the mount, and returns only after validation passes.

### `main.py`

Initialization becomes:

```text
create SimulationRuntime
→ runtime.prepare_for_perception()
→ initialize DebugOutputs
→ initialize YOLOE
→ enter canonical visual-servo loop
```

YOLOE initialization and RGB acquisition never happen when mount validation fails.

## GPU physics

Cable mounting requires the shared scene to use GPU PhysX:

```text
scene device: cuda:0
GPU dynamics: enabled
broadphase: GPU
solver: TGS
physics timestep: unchanged initially
```

The implementation verifies the composed scene attributes after authoring them. Failure is fatal.

This intentionally changes the physics environment used by the previously validated no-cable runtime.

## Finger presentation

The proxy carries the connector; the fingers are cosmetic.

1. Determine the plug width aligned with ToolCenter local `+Y`.
2. Add `1.0 mm` total clearance.
3. Set both finger joints symmetrically to half the total gap.
4. Clamp to joint limits.
5. Hold those finger targets during the run.

The fingers must not visibly interpenetrate or squeeze the connector strongly enough to create competing contact forces.

## Startup sequence

```text
create rack and Franka
→ discover panda_hand
→ create cameras
→ create CUDA physics scene and enable GPU dynamics
→ load cable
→ discover deformable body
→ detect plug frame and attachment bounds
→ place cable root onto ToolCenter
→ create proxy and fixed joint
→ author deformable attachment
→ initialize articulation and IK
→ set finger gap
→ begin simulation
→ prepare_for_perception(): settle and validate
→ initialize YOLOE
→ begin stereo acquisition
```

## Mount validation

During the final 30 settled frames, measure every frame:

- RJ45 tip position versus ToolCenter,
- connector nose axis versus ToolCenter local `+Z`,
- proxy-to-hand fixed-joint validity,
- deformable attachment validity,
- GPU-dynamics state.

Pass only when the complete window satisfies:

```text
maximum tip error <= 0.5 mm
maximum axis error <= 1.0 degree
fixed joint valid
attachment valid
deformable body valid
GPU dynamics enabled
```

A single out-of-limit frame fails startup. No averaging hides intermittent mount motion.

## Diagnostics

Successful startup prints:

```text
[CABLE MOUNT]
  cable USD: ...
  tracked plug: /World/NetworkCable/E_crystal_head1_45
  deformable body: ...
  plug dimensions mm: [...]
  longitudinal local axis: ...
  cable-side sign: ...
  insertion-tip local position m: [...]
  attachment bounds local m: min=[...] max=[...]
  insertion axis world: [...]
  hand-to-tip translation m: [...]
  hand-to-tip orientation WXYZ: [...]
  finger total gap mm: ...
  validation frames: 30/30
  maximum tip error mm: ...
  maximum axis error deg: ...
  fixed joint: valid
  attachment: valid
  cable tail: deformable
  GPU dynamics: enabled
```

Failures print the same measured context and a specific fatal reason.

## Fail-closed conditions

Abort before perception for missing/invalid assets, non-finite or zero bounds, ambiguous axes, ambiguous nose direction, ambiguous deformable-body discovery, missing hand, invalid proxy/joint/attachment, inactive GPU dynamics, or mount validation outside tolerance.

No failure path may fall back to an unattached cable, a rigid whole cable, root teleporting, a hard-coded tip offset, or the old empty-gripper ToolCenter meaning.

## Tests

### Pure tests

Cover longitudinal axes X/Y/Z, rotated transforms, both cable-side signs, face-center calculation, deterministic roll, attachment-volume trimming, ToolCenter mapping, non-finite data, degenerate bounds, ambiguous aspect ratios, and ambiguous cable-side projections.

### Structural tests

Prove that:

- mounting and validation precede YOLOE,
- ToolCenter offset is not duplicated,
- no insertion command exists,
- no per-frame cable or plug transform exists,
- `panda_hand` and deformable body are discovered dynamically,
- GPU dynamics is required when mounting is enabled,
- mounting can be disabled only through explicit diagnostic configuration.

### Isaac workstation smoke test

Require:

- no invalid attachment-actor error,
- connector remains mounted while robot/cable settle,
- 30/30 validation frames pass,
- tail visibly deforms and settles,
- YOLOE starts only after validation,
- nominal visual alignment completes,
- physical ToolCenter tracking error remains `<= 0.3 mm`,
- target steps remain `<= 1.0 mm`,
- no insertion occurs.

## Regression boundary and kill switch

The previous no-cable nominal result does not validate the GPU/deformable runtime.

Do not begin insertion work unless the cable-mounted nominal run demonstrates:

```text
mount validation: PASS
visual alignment: COMPLETE
physical ToolCenter tracking error <= 0.3 mm
maximum target step <= 1.0 mm
no insertion command
```

If the attachment is unstable, the tail destabilizes the arm/cameras, mount limits fail, or nominal alignment no longer completes, fix the mount or physics setup. Do not widen limits or add insertion on top of an unstable frame.