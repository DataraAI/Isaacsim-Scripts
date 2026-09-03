# asset_spawn

Minimal Isaac Sim scene that loads the DataHall rack USD.

## Run

From `Isaacsim-Scripts`:

```bash
/home/aayush/isaacsim/python.sh aayush/asset_spawn/main.py
```

Also spawns:

- `/World/WorkTable` — **visual only** (no physics/collision), suspended at `(0.42, -0.6, 1.0)`
- `/World/UR10eMount/ur10e` — UR10e with **Robotiq 2F-85**
- `/World/CableSupportBlock` — pedestal sized to span both crystal heads
- `/World/NetworkCable` — `E_crystal_head1_45` and `E_crystal_head2_39` on opposite block ends (physics/collision not enabled yet). After seating, the script does a timeline stop/play so the soft line rebinds to the heads.

Scene construction lives in [`spawn.py`](spawn.py) (`build_asset_spawn_scene`) so other demos can reuse the layout.
