# ur5e_6x_cable_insertions Design

**Date:** 2026-09-13  
**Status:** Approved  
**Location:** `aayush/ur5e_6x_cable_insertions/`  
**USD:** `~/Desktop/Aayush_ws/DataHall_6r_ur5e.usd`

## Goal

Six UR5e + Robotiq 2F-85 stations each grasp their Ethernet cable, carry it to
a configurable RJ45 jack, closed-loop align crystal/port insertion features,
insert until mating planes nearly touch, release, and return to the simulation
start home joint state. Optional `--station` selects a subset of stations.

## Non-goals

- FixedJoint / weld grasp (physical pinch only)
- Live mesh re-extraction of features (use cached JSON only)
- Extracting a shared core with the UR10e packages (fork, do not refactor)

## Approach

Fork `aayush/ur10e_6x_cable_insertions/`: keep multi-station BT wiring and CLI,
swap robot/USD to UR5e, take grasp physics/friction from
`ur10e_1x_cable_insertion`, and replace pin-based insert with closed-loop
feature alignment via `aayush/insertion_features/`.

## Package layout

| File | Role |
|---|---|
| `config.py` | USD path, station specs, per-station `jack_id`, UR5e home joints, 30° grasp tilt, 1× physics constants, align/insert tolerances |
| `scene.py` | Load USD, wire selected stations, home pose, Lula/controllers, cable + robot physics |
| `primitives.py` | Observe → orient/tilt → descend/grasp/lift → via maneuver → align → insert → release → home |
| `alignment.py` | World feature transforms, residual scoring, tip nudge commands |
| `main.py` | Parallel BT tick loop; `--station` / `--usd` / `--headless` / `--max-frames` / `--json` |
| `task_intelligence.json` | Sequence covering grasp through return-home |
| `README.md` + `tests/` | Run docs; host-side config/BT/alignment tests |

## Stations

Same IDs as UR10e 6×: `NegativeY_Top`, `PositiveY_Top`, `NegativeY_Middle`,
`PositiveY_Middle`, `NegativeY_Lower`, `PositiveY_Lower`.

Each `StationSpec` maps:

- robot `/World/Robots/UR5e_<station_id>`
- cable `/World/NetworkCables/Cable_<station_id>/`
- support floor under `CableBlocks`
- switch pack for that side/height
- `jack_id` (default `jack_upper_c2` / copper `Group_14343`), configurable per station in config

CLI `--station` is repeatable; unknown IDs exit with the known list.

## Behaviour tree phases

Per station (facts in parentheses):

1. **Observe** — TCP above crystal head 39 (+Z clearance); fingers open; open axis ≈ world ±Y (`at_workspace`)
2. **Orient + tilt** — fingers on cable ±Y sides; wrist tilt **30°** down from vertical (0° = along head 45 / tool −Z)
3. **Descend + grasp** — tip to neck `E_part006_44`; avoid left/right blocks; close; lift (`cable_held`)
4. **Maneuver** — via waypoints to port offset = port mating center + `PORT_APPROACH_X_OFFSET_M` in world **+X** (`at_port_offset`)
5. **Align (closed-loop)** — each tick re-pose features and nudge until tolerances pass (`features_aligned`)
6. **Insert** — advance along port insertion axis until mating planes nearly touch (`at_port_insert`)
7. **Release** — open gripper (`cable_released`)
8. **Home** — joint-interp to start upright / gripper-down home (`at_home`)

Selected stations tick in parallel each physics frame. Pause/Stop are respected
(no forced `world.play()` in the tick loop).

## Home pose

Home is the arm joint state applied at simulation start: upright arm, gripper
pointed down, clear of cables and blocks. After release, return to that same
joint vector (open gripper remains open).

## Closed-loop feature alignment

Uses cached local features:

- `aayush/insertion_features/cable_features_local.json` → `world_features_for_head`
- `aayush/insertion_features/port_features_local.json` → `world_features_for_jack`

Each align tick:

1. Read live `world_from_head` for `E_crystal_head1_45` and `world_from_pack` for the station pack
2. Transform cached features to world metres (`meters_per_unit` for the centimetre DataHall stage)
3. Score residuals and emit a clamped tip Δp / ΔR until all pass:

| Constraint | Pass when |
|---|---|
| Latch clearance | Crystal latch keypoints have **lower world Z** than port latch keypoints (margin `LATCH_Z_MARGIN_M`) |
| Mating containment | Crystal mating corners project **inside** the port mating rectangle along port width/up axes (margin `MATING_SIDE_MARGIN_M`) |
| Axis alignment | `\|dot(crystal_axis, port_axis)\| ≥ AXIS_DOT_MIN` |
| Mating proximity (insert stop) | Distance between mating-plane points along port axis ≤ `MATING_TOUCH_GAP_M` |

Insert is gated on `features_aligned`. Align aborts after `ALIGN_MAX_FRAMES` if
not converged. Optional debug markers for mating centers / latch keypoints
(default invisible).

## Cable and robot physics

Physical pinch only — **no FixedJoint weld**.

Port 1× cable treatment onto each station cable:

- One rigid body per crystal head; mesh descendants get convexHull collision
- Grasp friction static/dynamic **0.8**, combine **max**, on fingertip pads and crystal-head meshes
- Low-friction slide on trailing head39 (and bezels if present), same as 1×
- Soft line `E_line_35` remains deformable and is rebound after physics rebuild (DataHall stage-unit handling as in UR10e 6×)

Robot link collision enabled; articulation self-collision off; gravity disabled
on robot links as in the 6× scene builder. UR5e joint drive stiffness/damping
follow the same pattern as the UR10e packages, adapted to UR5e joint names.

## Error handling

- Align timeout or IK failure → that station `FAILURE`; other arms continue
- Cable slip while `monitor_cable_hold` → abort that station (no weld fallback)
- Host tests cover station/`jack_id` config, BT facts
  (`features_aligned`, `cable_released`, `at_home`), and alignment residual
  math without Isaac Sim

## Success criteria

- `--station NegativeY_Top` (or any single station) completes observe → grasp →
  offset → feature align → insert → release → home
- All six stations can run in parallel with the same tree
- Crystal latch keypoints stay below port latch keypoints at align pass; mating
  sides contained; axes nearly collinear before insert advance
- Insert stops when mating planes nearly touch; fingers open; arm returns home
- No FixedJoint attachment is created at any point
