# Isaacsim-Scripts

Isaac Sim scripts for QSFP module pick-and-insert on a Franka Panda inside a datacenter scene. The active workflow lives in `current/` and targets **Isaac Sim 6.0.0** with **Lula** inverse kinematics.

## Quick start

Run the main script with your Isaac Sim Python launcher:

```bash
# Linux example
~/isaac-sim-6/python.sh current/insert_at_prim_lula.py

# Windows example
python.bat current\insert_at_prim_lula.py
```

Press **Play** in the timeline. The script warms up physics, queues one insert job per port, then runs pick → transport → align → insert → release → retreat for each QSFP module.

## What it does

For each configured network-switch port:

1. Hover over a QSFP module in the pick tray
2. Descend and close the gripper
3. Lift and rotate to the insert orientation
4. Move to the port approach standoff
5. Fine lateral align at standoff
6. Insert along the port axis
7. Release, retreat, and clear the rack before the next pick

Modules are simple rigid-body proxies (`qsfp_module.py`). Port poses and insert axes come from USD connector prims via `port_frame.py`.

## Project layout

| Path | Role |
|------|------|
| `current/insert_at_prim_lula.py` | Main entry point: scene setup, controller wiring, sim loop |
| `current/franka_lula_controller.py` | Cartesian waypoint queue, closed-loop IK, gripper logic |
| `current/qsfp_insert_job.py` | Builds the 11-step job sequence for one port |
| `current/port_frame.py` | Port frame from USD prim pose (insert axis, approach goals) |
| `current/qsfp_module.py` | QSFP proxy spawn and grasp offsets |
| `current/collision_setup.py` | Data hall scale and collision enable helpers |
| `current/module_port_contact_monitor.py` | Contact sensor for insert-stop detection |

Older experiments live under `baseline/`, `attempts/`, and `cableTask/`.

## Configuration

Edit constants at the top of `current/insert_at_prim_lula.py`:

| Constant | Purpose |
|----------|---------|
| `DATAHALL_USD` | Path to the datacenter stage USD |
| `INSERT_PORT_PRIM_PATHS` | Connector prim paths to insert into (one job each) |
| `PICK_TRAY_XY`, `PICK_SPACING` | Pick tray layout for multiple modules |
| `ROBOT_BASE_POS` | Franka spawn position |
| `INSERT_LATERAL_Z_BY_CONNECTOR` | Per-connector lateral Z offset (job 0 vs 1–2 differ) |
| `POST_RESET_WARMUP_FRAMES` | Physics warmup after reset before control starts |
| `STARTUP_CAMERA_EYE`, `STARTUP_CAMERA_TARGET` | Optional fixed viewport on open |

Motion tuning is in `current/qsfp_insert_job.py` (hover height, insert step size, retreat speed, frame budgets).

## Isaac Sim 6.0 notes

This workflow was migrated from 5.1.0. Important 6.0 behaviors:

- **Startup warmup** — control starts only after `POST_RESET_WARMUP_FRAMES` settle steps so the first job matches later reruns.
- **PhysX variant** — Franka `Physics` variant is set to Physx at spawn.
- **Control order** — `apply_action()` runs before `world.step()` to avoid a one-frame lag.
- **Release timing** — per-job gripper release frames differ so seated modules are not disturbed on retreat.
- **Runtime view flush** — articulation views are flushed before pose/joint reads after reset.

IK uses `LulaKinematicsSolver` + `ArticulationKinematicsSolver` with `panda_hand` as the end-effector frame.

## Troubleshooting

**First insert fails, later ones succeed** — usually warmup or release timing; check seat logs after insert/retreat in the console.

**Pick failed — module did not lift** — gripper did not secure the module; the script retries the pick once, then restarts the full job.

**Module slips or gets knocked on pick** — increase `PICK_HOVER_OFFSET` or reduce approach speed in `qsfp_insert_job.py`.

**Port seat check failed** — inspect `Job N post-insert seat check` logs for lateral and axial tip error.

**Camera framing** — set `LOG_VIEWPORT_CAMERA_AT_STEP` to a positive step count, frame the view manually, then copy the printed `STARTUP_CAMERA_*` values into `insert_at_prim_lula.py`.

## Requirements

- Isaac Sim 6.0.0 (tested target)
- Franka Panda asset from Isaac Sim content
- Datacenter USD at the path configured in `DATAHALL_USD`
- Linux or Windows with the Isaac Sim Python environment
