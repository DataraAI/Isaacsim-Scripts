# UR10e Via Densification + Grip Slip Diagnostics Design

**Date:** 2026-09-05  
**Status:** Approved  
**Location:** `aayush/ur10e_1x_cable_insertion/` (+ matching via markers in `aayush/asset_spawn/`)  
**Builds on:** `2026-09-02-ur10e-1x-cable-insertion-design.md` and the staged port-via carry in `eb6953a`

## Goal

Pinpoint where the grasped cable slips out of the Robotiq fingers during the
lift→offset carry, and expose telemetry so the next sim run shows what to tune.

1. Densify approach via fractions every **0.02** from **0.75 to 0.95**.
2. Log finger and cable-head forces/torques (joint efforts), positions, and
   velocities to **console and CSV**.
3. On cable-lost abort, print **remediation hints on the console only** (CSV
   stays raw numbers).

## Non-goals

- PhysX fingertip contact-force sensors
- FixedJoint / weld grasp
- Changing default friction, close angle, or joint_steps beyond denser vias +
  logging (hints may *recommend* those changes; this pass does not apply them)
- Behaviour-tree JSON / primitive registry changes

## Context (known failure)

Prior runs showed fingers remaining closed (`fingers≈0.72`) while `tip_err`
grew past `CABLE_IN_GRIPPER_MAX_ERR_M` (0.06 m) between **via 0.82 and via
0.95** — friction grasp under large tip steps, not an open-gripper failure.
Coarse vias hide the exact slip onset; denser late vias + kinematics/effort
logs make that onset visible.

## Architecture

Keep the existing BT flow (observe → detect → grasp/lift → port vias → insert).
Add a thin diagnostics layer:

| Piece | Role |
| --- | --- |
| `config.py` | Densified `PORT_APPROACH_VIA_FRACTIONS`; CSV dir / sample knobs |
| `asset_spawn/spawn.py` | Same fraction tuple for green `Via_75`…`Via_95` markers |
| `grasp_diag_log.py` | Sample → console `[BT DIAG]` + CSV append; abort hints |
| `primitives.py` | Call logger from REACHED / periodic hold / cable-lost abort |

No `task_intelligence.json` or `main.py` primitive-map changes.

## Via densification

`PORT_APPROACH_VIA_FRACTIONS` becomes:

```text
(0.35, 0.60) + (0.75, 0.77, 0.79, …, 0.95)
```

Generated as early vias plus `np.arange(0.75, 0.95 + 1e-9, 0.02)` (or an
equivalent explicit tuple). Final offset remains `1.0` appended by
`queue_port_approach` as today. Insert vias (`PORT_INSERT_VIA_FRACTIONS`) are
unchanged.

`asset_spawn` mirrors the same list so Stage markers `Via_75`…`Via_95` match
motion waypoints.

## Sampling

**When**

- Every `[BT IK] REACHED` while `hold_gripper` is active (label = via/yaw/insert).
- Periodically during hold motion (reuse `CABLE_HOLD_CHECK_EVERY_N_FRAMES` or a
  dedicated `GRASP_DIAG_EVERY_N_FRAMES`).
- Once on cable-lost abort (final row + console hints).

**Fields per sample**

- `label`, `phase`, `frame`
- Finger joint positions, velocities, efforts (Robotiq DOFs on the articulation)
- Crystal-head / grasp-part world position; linear and angular velocity when API
  available
- Gripper tip vs cable tip error (`tip_err_m`), `in_gripper`

**Missing APIs:** write `None`/NaN for unavailable velocity/effort fields;
never abort solely because a sensor read failed. Positions + `tip_err` always
required when the stage/robot are valid.

## Outputs

**Console**

- Compact `[BT DIAG]` lines with the fields above.
- Existing `[BT GRIP]` / `[BT IK]` lines remain; diagnostics supplement them.

**CSV**

- Path: `aayush/ur10e_1x_cable_insertion/logs/grasp_diag_<timestamp>.csv`
- Raw numeric columns only (no remediation text).
- Directory gitignored (`aayush/ur10e_1x_cable_insertion/logs/`).
- If CSV open fails: print once, continue console-only.

**Abort hints (console only)**

When `_abort_cable_lost` runs, print short remediation suggestions derived from
the sample history, for example:

- Tip error spikes while fingers stay closed → raise fingertip/head friction,
  slow late vias (`joint_steps`), or reduce late tip step size.
- Finger effort collapses during carry → increase close target / squeeze-hold
  frames.
- Report the **first via label** (among denser samples) where `tip_err`
  crossed the slip threshold when history allows.

## Error handling

- Sensor/API failures → NaN fields + continue.
- CSV I/O failure → console-only fallback.
- Cable-lost abort behavior unchanged aside from extra diagnostics/hints before
  queue clear / `app.close()`.

## Testing

- Host unittest: densified via tuple includes 0.75…0.95 at 0.02 steps.
- Optional pure unit test for remediation-hint builder from synthetic tip_err
  history (Isaac-free).
- Existing tree JSON smoke test unchanged.

## Docs

- README: denser late vias, `[BT DIAG]` + CSV path, abort hints.
- This spec file.

## Success criteria

1. Running the demo produces green markers and IK labels for vias every 0.02
   from 0.75 to 0.95.
2. A CSV is written with finger/head kinematics and tip_err over the carry.
3. On slip abort, console names the late-via window and prints remediation
   hints without putting those hints in the CSV.
