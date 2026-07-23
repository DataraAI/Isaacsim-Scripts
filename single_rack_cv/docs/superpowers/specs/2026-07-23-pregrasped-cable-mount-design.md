# Pregrasped Deformable Cable Mount Design

**Date:** 2026-07-23  
**Branch:** `feature/pregrasped-cable-mount`  
**Status:** Approved design awaiting implementation plan

## Goal

Start the canonical `single_rack_cv` simulation with the network cable already held by the Franka so the project can focus on port perception, pre-insert alignment, and later insertion rather than grasp acquisition.

The RJ45 connector tip becomes the physical meaning of `/World/ToolCenter`. The connector remains permanently mounted for the entire run. The cable tail remains deformable.

## Scope

This change adds:

- the provided network-cable USD,
- automatic RJ45 insertion-tip detection,
- automatic rack-facing cable orientation,
- a permanent hand-mounted rigid proxy,
- a deformable-to-rigid connector attachment,
- GPU PhysX dynamics,
- a cosmetic symmetric finger opening around the connector,
- startup validation and diagnostics.

This change does **not** add:

- grasp acquisition,
- grip-force testing,
- connector slip,
- release or regrasp,
- insertion motion,
- a per-frame cable teleport,
- manual connector depth offsets.

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

The simulation must fail before physics starts if the USD does not exist or the tracked plug prim is missing after reference composition.

## Chosen mounting architecture

Use a **hand-mounted rigid proxy with a connector-region deformable attachment**.

1. Create a small hidden rigid proxy actor at `/World/CableMountProxy`.
2. Constrain that proxy permanently to the Franka `panda_hand` rigid body with `/World/CableMountFixedJoint`.
3. Position the network-cable root so the detected RJ45 insertion tip coincides with `/World/ToolCenter` and its insertion axis matches ToolCenter local `+Z`.
4. Attach only the deformable connector region around `E_crystal_head1_45` to the proxy.
5. Leave the cable tail governed by deformable simulation.

The proxy is a separate world-level rigid actor. It must not be created as a nested rigid body beneath `panda_hand`, because nested rigid bodies produce ambiguous PhysX ownership.

The connector attachment remains active for the complete process lifetime. There is no release mechanism.

## Why the other approaches are rejected

### Direct fixed joint to `E_crystal_head1_45`

Rejected because the tracked plug belongs to a deformable asset and may not be an independent rigid actor. Treating it as one risks invalid-actor errors and can rigidify or detach the wrong part of the cable.

### Per-frame transform overwrite

Rejected because it bypasses physics, injects energy into the deformable tail, and makes later contact behavior untrustworthy.

### Rigidly moving the complete cable

Rejected because it eliminates the cable behavior this asset was selected to represent.

## ToolCenter semantics

The existing configured ToolCenter transform relative to `panda_hand` remains numerically unchanged:

```text
panda_hand_T_toolcenter
```

The cable is mounted so the detected RJ45 insertion tip lands exactly on that frame. Therefore:

```text
panda_hand_T_rj45_tip == panda_hand_T_toolcenter
```

This preserves the existing camera geometry, Lula IK target conversion, desired 50 mm pre-insert standoff, and controller calibration.

After mounting:

- `/World/IK_Target` commands the RJ45 insertion tip.
- `/World/ToolCenter` reports the actual RJ45 insertion-tip pose.
- The 50 mm standoff is measured from the port opening to the connector tip.
- Later insertion can advance this same frame directly along the port axis.

No second connector offset may be added in perception, control, or future insertion code.

## Automatic connector geometry

A pure NumPy geometry component determines the connector insertion frame from the composed USD geometry.

### Inputs

- tracked plug local bounding box,
- tracked plug world transform,
- whole cable root world bounding-box center,
- ToolCenter target transform.

### Insertion-axis detection

1. Compute the tracked plug bounds in the plug-local frame.
2. Select the longest local dimension as the connector longitudinal axis.
3. Require the longest dimension to exceed the second-longest dimension by a configurable ambiguity ratio. Default minimum ratio: `1.5`.
4. Transform the cable-root center into the plug-local frame.
5. Project the vector from plug center to cable center onto the longitudinal axis.
6. The projected direction identifies the cable-side end.
7. The opposite end is the RJ45 nose.
8. The insertion tip is the center of the nose-side local bounding-box face.

If the longest axis or nose sign is ambiguous, startup aborts. There is no silent manual fallback.

### Mount orientation

The detected plug nose axis is aligned with ToolCenter local `+Z`.

At the canonical fixed wrist orientation, ToolCenter local `+Z` points toward the rack. The implementation must derive this from the ToolCenter transform rather than hard-code world `-X`, so the mount remains correct if the fixed wrist pose is changed later.

Roll about the insertion axis is resolved deterministically by aligning the plug's widest transverse axis with ToolCenter local `+Y`. The remaining transverse axis must form a right-handed frame.

### Mount translation

After rotation, translate the complete cable root so the detected insertion-tip point coincides with the configured ToolCenter world position at startup.

The tracked plug child is never moved independently from the cable root during placement.

## Components

### `cable_geometry.py`

Pure Python and NumPy. No Isaac, USD, PhysX, or `omni` imports.

Responsibilities:

- validate finite 3D bounds and transforms,
- select the longitudinal axis,
- determine cable side and nose side,
- calculate the local insertion-tip point,
- construct a deterministic plug insertion frame,
- calculate the rigid transform that maps the plug insertion frame onto ToolCenter,
- report ambiguity instead of guessing.

### `cable_mount.py`

Isaac/USD/PhysX integration.

Responsibilities:

- load and validate the cable USD,
- find the Franka `panda_hand` descendant dynamically,
- discover the cable's deformable body prim,
- query composed plug and cable geometry,
- call `cable_geometry.py`,
- apply the calculated root transform before simulation starts,
- create the rigid proxy and fixed joint,
- create the connector-region deformable attachment,
- configure finger positions,
- validate the mounted result after settling,
- expose mount diagnostics to `SimulationRuntime`.

### `config.py`

Add a frozen `CableMountConfig` containing:

- enabled flag,
- USD/root/tracked-plug paths,
- proxy and joint paths,
- geometry ambiguity ratio,
- attachment-region padding,
- desired cosmetic finger clearance,
- startup settle frames,
- maximum tip mounting error,
- maximum axis error.

Default required limits:

```text
maximum tip mounting error: 0.5 mm
maximum connector-axis error: 1.0 degree
```

### `sim.py`

Integrate mounting into scene construction without adding insertion behavior.

The canonical public visual-servo methods remain unchanged. Cable mounting is a startup concern, not a second controller.

## GPU dynamics

The deformable cable requires PhysX GPU dynamics.

The single-rack scene changes from CPU dynamics to the CUDA physics device and must configure:

- GPU dynamics enabled,
- GPU broadphase,
- TGS solver,
- the existing scene timestep unless a real instability proves a change is necessary.

This is an intentional global physics change. The Franka, rack collisions, ground plane, and deformable cable share the same physics scene.

The implementation must verify the composed physics-scene attributes after authoring them. Failure to activate GPU dynamics aborts startup.

## Finger presentation

The fingers do not provide the load-bearing attachment; the proxy does.

For visual consistency:

1. Use the two smaller transverse plug dimensions.
2. Choose the dimension aligned with ToolCenter local `+Y` as the required total finger gap.
3. Add a small configurable clearance, initially `1.0 mm` total.
4. Command both finger joints symmetrically to half that total gap.
5. Clamp against the Franka joint limits.

The fingers must not squeeze the attached plug hard enough to create competing contact forces or visible interpenetration.

## Startup order

```text
create rack and Franka references
→ find panda_hand
→ create cameras
→ create physics scene on CUDA
→ enable GPU dynamics
→ load cable reference
→ discover deformable body and connector geometry
→ detect RJ45 insertion frame
→ align cable root to ToolCenter
→ create rigid proxy and fixed joint
→ author connector-region deformable attachment
→ initialize articulation and IK
→ set cosmetic finger gap
→ begin simulation
→ settle robot and cable
→ validate mount
→ begin stereo visual acquisition
```

Visual acquisition must not begin before mount validation passes.

## Runtime validation

After the configured settling period, calculate:

- actual insertion-tip position,
- actual connector nose axis,
- ToolCenter position and local `+Z` axis,
- tip-to-ToolCenter position error,
- connector-axis angular error,
- attachment validity,
- deformable-body validity,
- GPU-dynamics state.

Startup passes only when:

```text
tip error <= 0.5 mm
axis error <= 1.0 degree
attachment is valid
deformable body is valid
GPU dynamics is enabled
```

On failure, stop the simulation and print the measured values. Do not start YOLOE or issue visual-servo corrections.

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
  insertion axis world: [...]
  hand-to-tip translation m: [...]
  hand-to-tip orientation WXYZ: [...]
  finger total gap mm: ...
  tip mounting error mm: ...
  axis error deg: ...
  attachment: valid
  cable tail: deformable
  GPU dynamics: enabled
```

Failures include the same fields plus a specific reason.

## Error handling

Fail closed for:

- missing cable file,
- missing tracked plug,
- zero or non-finite bounds,
- ambiguous longest axis,
- ambiguous cable-side projection,
- missing or multiple deformable bodies when one cannot be selected deterministically,
- missing `panda_hand`,
- invalid proxy rigid body,
- invalid fixed joint,
- invalid deformable attachment,
- GPU dynamics not active,
- tip or axis validation outside tolerance.

No error path may fall back to an unattached cable, root teleporting, a hard-coded tip offset, or the old empty-gripper ToolCenter meaning.

## Tests

### Pure geometry tests

Synthetic tests must cover:

- longest axis along local X, Y, and Z,
- rotated world transforms,
- cable center on either end of the plug,
- correct nose-face center,
- deterministic transverse roll,
- correct root transform onto ToolCenter,
- zero-length and non-finite bounds rejection,
- ambiguous aspect-ratio rejection,
- ambiguous cable-side projection rejection.

### Structural tests

Tests must prove:

- cable mounting happens before visual acquisition,
- ToolCenter offset is not duplicated elsewhere,
- no insertion command is introduced,
- no per-frame cable-root or plug teleport exists,
- `panda_hand` is discovered dynamically,
- GPU dynamics is enabled when cable mounting is enabled,
- cable mounting can be disabled only through explicit configuration for diagnostic comparison.

### Isaac workstation smoke test

Run the simulation and require:

- cable loads,
- no invalid deformable-attachment actor error,
- connector remains attached while the robot reaches its initial pose,
- tip error is at most 0.5 mm,
- axis error is at most 1 degree,
- cable tail visibly settles under deformable physics,
- visual acquisition begins only after mount validation,
- nominal visual alignment completes,
- no insertion occurs.

## Safety and regression boundary

Enabling GPU dynamics changes the physics environment that produced the previous nominal no-cable alignment result. That result cannot be treated as proof that the cable-mounted runtime works.

Before insertion work begins, the cable-mounted nominal run must demonstrate:

```text
mount validation: PASS
visual alignment: COMPLETE
physical ToolCenter tracking error <= 0.3 mm
maximum target step <= 1.0 mm
no insertion command
```

Do not widen mount or controller limits to force a pass.

## Kill switch

Do not implement insertion if any of these remain true:

- connector attachment is invalid or intermittently lost,
- cable tail destabilizes the arm or camera,
- tip error exceeds 0.5 mm,
- connector-axis error exceeds 1 degree,
- nominal visual alignment no longer completes under GPU dynamics.

The correct response is to fix the mount or physics setup, not add insertion on top of an unstable frame.