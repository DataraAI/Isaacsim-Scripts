# ur10e_6x_cable_insertions

Six-arm behaviour-tree demo on `~/Desktop/Aayush_ws/DataHall_6r.usd`. Each UR10e
(with a Robotiq 2F-85) detects `E_part006_44`, grasps/lifts with a **60°** wrist
tilt, then carries the held tip to that station's RJ45 **offset** and **insert**
point. Same motion as [`ur10e_1x_cable_insertion/`](../ur10e_1x_cable_insertion/).

## Run

From `Isaacsim-Scripts`:

```bash
/home/aayush/isaacsim/python.sh aayush/ur10e_6x_cable_insertions/main.py
```

Optional:

```bash
# One station only
/home/aayush/isaacsim/python.sh aayush/ur10e_6x_cable_insertions/main.py \
    --station NegativeY_Top

# Alternate USD
/home/aayush/isaacsim/python.sh aayush/ur10e_6x_cable_insertions/main.py \
    --usd ~/Desktop/Aayush_ws/DataHall_6r.usd
```

## Motion verification

Validate one station before running all six:

```bash
/home/aayush/isaacsim/python.sh aayush/ur10e_6x_cable_insertions/main.py \
    --station NegativeY_Top
```

The arm must visibly reach observation and hover, descend before finger
closure, and lift the cable. Any `Joint-interp IK target failed` message must
produce behavior-tree `FAILURE`; it must never be followed by `REACHED` for the
same waypoint.

## Stations

| Station | Robot | Side | Height | `grid_option` | `robot_loc` |
|---|---|---|---|---|---|
| `NegativeY_Top` | `/World/Robots/UR10e_NegativeY_Top` | left (−Y) | top | `AS4610_Ethernet_Row_Top_1x_Grid` | `Upper_Left` |
| `PositiveY_Top` | `/World/Robots/UR10e_PositiveY_Top` | right (+Y) | top | `AS4610_Ethernet_Row_Top_1x_Grid` | `Upper_Right` |
| `NegativeY_Middle` | `/World/Robots/UR10e_NegativeY_Middle` | left | middle | `AS4610_Ethernet_Row_Middle_1x_Grid` | `Upper_Left` |
| `PositiveY_Middle` | `/World/Robots/UR10e_PositiveY_Middle` | right | middle | `AS4610_Ethernet_Row_Middle_1x_Grid` | `Upper_Right` |
| `NegativeY_Lower` | `/World/Robots/UR10e_NegativeY_Lower` | left | bottom | `AS4610_01_1x_Grid` | `Upper_Left` |
| `PositiveY_Lower` | `/World/Robots/UR10e_PositiveY_Lower` | right | bottom | `AS4610_01_1x_Grid` | `Upper_Right` |

Port contacts:

`/World/Network_Switches/<grid_option>/<robot_loc>/AS4610_inst/AS4610_01/Switch/Net_12_Pack_no_LED_Component_03/RJ45_Group01/CopperContacts/Group_14343`

Cables live at `/World/NetworkCables/Cable_<station_id>/`.

## Debug markers

Invisible spheres under `/World/DebugPortMarkers/<station_id>/` (toggle Visibility
in the Stage panel). No rigid body, no collision:

- **yellow** `Offset` — pre-insert +0.02 m in world X
- **red** `Insert` — mid of copper pins 1907 / 1910
- **green** `Via_35`…`Via_95` and `InsertVia_45` / `InsertVia_75` — lift→offset and offset→insert waypoints

## Host-side JSON / config test

```bash
cd Isaacsim-Scripts/aayush
python3 -m unittest discover -s ur10e_6x_cable_insertions/tests -v
```
