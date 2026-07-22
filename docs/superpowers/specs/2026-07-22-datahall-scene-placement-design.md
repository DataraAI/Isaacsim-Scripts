# Data Hall placement in ethernet cable pick-up scene

**Date:** 2026-07-22  
**Status:** Approved for planning  
**Scope:** `esha/ethernet_cable_pick_up`

## Goal

Load `DataHall_Full_01.usd` into the cable pick-up scene as a full interaction-capable environment (static collisions), placed just beyond the cable head in **+X** so it does not sit on top of the Franka, cable, or support block.

## Decisions

| Topic | Choice |
| --- | --- |
| Asset | `DataHall_Full_01.usd` |
| Role | Full interaction target (static collisions; no port/insert logic yet) |
| Placement | Config-driven offset from `cable_spawn_xy` in +X |
| Approach | SceneConfig fields + load in `_build_scene` |

## Config surface (`SceneConfig`)

Add:

- `datahall_enabled: bool = True`
- `datahall_usd_path: str` →  
  `/home/advaith/Isaacsim-assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd`
- `datahall_prim_path: str = "/World/DataHall"`
- `datahall_scale: float = 1.0` (tunable; other demos often use `2.0`)
- `datahall_yaw_deg: float = 0.0`
- `datahall_offset_from_cable_xy: tuple[float, float] = (1.5, 0.0)`
- `datahall_enable_static_collisions: bool = True`

World position:

```text
x = cable_spawn_xy[0] + datahall_offset_from_cable_xy[0]
y = cable_spawn_xy[1] + datahall_offset_from_cable_xy[1]
z = 0.0
```

With current `cable_spawn_xy = (0.74, 0.0)` and default offset `(1.5, 0.0)`, the Data Hall root sits at approximately `(2.24, 0.0, 0.0)`.

## Scene integration (`sim.py`)

Load order in `_build_scene`:

1. Ground plane + dome light  
2. Cable support (if enabled) + cable  
3. **Data Hall** (if `datahall_enabled`)  
4. Franka + hand cameras  
5. Physics / GPU dynamics / settle / IK  

Transform pattern: reuse existing `_define_xform` + `_add_reference` (same as Franka).

Scale: call `apply_datahall_scale(datahall_prim_path, datahall_scale)` when scale ≠ `1.0`, using helpers from `detailedInsertion/datahall/collision_setup.py`.

Collisions: when `datahall_enable_static_collisions` is true, call `enable_static_collisions(datahall_prim_path)` after the reference (and scale) resolve.

READY log: include USD path, computed world XY, scale, yaw, and collision flag.

Missing USD file: raise `FileNotFoundError` (same style as cable asset check) when enabled.

## Out of scope

- Port targeting / switch face alignment  
- Grasp or insert toward a network port  
- Viewport retargeting to show the hall  
- Relocating Franka or cable to match hall interior layout  
- Duplicating collision helpers into this package (import/reuse existing module)

## Success criteria

- With defaults, the hall appears past the plug in +X and does not cover the Franka/cable pick workspace.  
- Static collisions are authored under `/World/DataHall` when enabled.  
- `datahall_enabled=False` restores the previous scene behavior.  
- Offset/scale/yaw are tunable from `config.py` without code changes in `sim.py`.

## Follow-up (not this change)

Anchor the hall (or robot) so a chosen switch/port face sits near the cable head for insertion demos.
