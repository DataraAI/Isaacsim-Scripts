# asset_spawn

Minimal Isaac Sim scene that loads the DataHall rack USD.

## Run

From `Isaacsim-Scripts`:

```bash
/home/aayush/isaacsim/python.sh aayush/asset_spawn/main.py
```

DataHall path:

`/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Single_Rack_3x_Ethernet_Rows_2x_BakedScale1_4x1x_Switches.usd`

Prim: `/World/DataHall`

Also spawns:

- `/World/WorkTable` — fixed cuboid at `(0.42, -0.6, 1.0)`, orientation `(0, 0, -90)` deg
- `/World/UR10eMount/ur10e` — UR10e with Robotiq 2F-140 on `wrist_3_link` (assembly namespace `Gripper`)
- `/World/CableSupportBlock` — pedestal on the table (esha-style)
- `/World/NetworkCable` — plug head centered at `(0.45, -0.35)` on the support block

Physics: GPU dynamics enabled for the soft network cable. The work table stays a
fixed cuboid; the UR10e has gravity disabled so it rests on the table without
falling. Press **Play** to settle the cable on the support block.
