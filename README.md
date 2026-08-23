# Isaacsim-Scripts

Isaac Sim scripts for block pick-and-insert with a Franka Panda using **Lula** inverse kinematics. The active workflow is in `detailedInsertion/` and targets **Isaac Sim 6.0.0**.

## Quick start

Run the main script with your Isaac Sim Python launcher:

```bash
# Linux example
~/isaac-sim-6/python.sh detailedInsertion/insert_lula.py

# Windows example
python.bat detailedInsertion\insert_lula.py
```

Press **Play** in the timeline. The script runs grasp → transit → pre-insert alignment → closed-loop horizontal insert → release.

## Project layout

| Path | Role |
|------|------|
| `detailedInsertion/insert_lula.py` | Main entry point: scene setup, task config, sim loop |
| `detailedInsertion/franka_lula_controller.py` | Cartesian waypoint queue, closed-loop IK, gripper logic |
| `tanish/behaviour_tree/` | Generated task-intelligence JSON runtime and Isaac controller adapters |
| `archive/` | Older experiments (`current/`, `baseline/`, `attempts/`, etc.) |

## Configuration

Edit `INSERT_TASKS` and `ACTIVE_TASK_INDEX` at the top of `detailedInsertion/insert_lula.py` to set block spawn position, port center, and insert axis per task.

## Requirements

- Isaac Sim 6.0.0
- Franka Panda asset from Isaac Sim content
- Linux or Windows with the Isaac Sim Python environment
