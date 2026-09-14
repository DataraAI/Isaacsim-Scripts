# ur5e_6x_cable_insertions

Six-arm behaviour-tree demo on `~/Desktop/Aayush_ws/DataHall_6r_ur5e.usd`. Each UR5e
(with a Robotiq 2F-85) detects `E_part006_44`, grasps/lifts with a **30°** wrist
tilt, carries the held tip to that station's RJ45 **offset**, then closed-loop
**aligns and inserts** using cached crystal/port features, releases, and returns
home. Same grasp physics as [`ur10e_1x_cable_insertion/`](../ur10e_1x_cable_insertion/).

## Run

From `Isaacsim-Scripts`:

```bash
/home/aayush/isaacsim/python.sh aayush/ur5e_6x_cable_insertions/main.py
```

Optional:

```bash
# One station only
/home/aayush/isaacsim/python.sh aayush/ur5e_6x_cable_insertions/main.py \
    --station NegativeY_Top

# Alternate USD
/home/aayush/isaacsim/python.sh aayush/ur5e_6x_cable_insertions/main.py \
    --usd ~/Desktop/Aayush_ws/DataHall_6r_ur5e.usd
```

## Motion verification

Validate one station before running all six:

```bash
/home/aayush/isaacsim/python.sh aayush/ur5e_6x_cable_insertions/main.py \
    --station NegativeY_Top
```

The arm must visibly reach observation and hover, descend before finger
closure, and lift the cable. Any `Joint-interp IK target failed` message must
produce behaviour-tree `FAILURE`; it must never be followed by `REACHED` for the
same waypoint.

## Stations

| Station | Robot | Side | Height | `grid_option` | `robot_loc` |
|---|---|---|---|---|---|
| `NegativeY_Top` | `/World/Robots/UR5e_NegativeY_Top` | left (−Y) | top | `AS4610_Ethernet_Row_Top_1x_Grid` | `Upper_Left` |
| `PositiveY_Top` | `/World/Robots/UR5e_PositiveY_Top` | right (+Y) | top | `AS4610_Ethernet_Row_Top_1x_Grid` | `Upper_Right` |
| `NegativeY_Middle` | `/World/Robots/UR5e_NegativeY_Middle` | left | middle | `AS4610_Ethernet_Row_Middle_1x_Grid` | `Upper_Left` |
| `PositiveY_Middle` | `/World/Robots/UR5e_PositiveY_Middle` | right | middle | `AS4610_Ethernet_Row_Middle_1x_Grid` | `Upper_Right` |
| `NegativeY_Lower` | `/World/Robots/UR5e_NegativeY_Lower` | left | bottom | `AS4610_01_1x_Grid` | `Upper_Left` |
| `PositiveY_Lower` | `/World/Robots/UR5e_PositiveY_Lower` | right | bottom | `AS4610_01_1x_Grid` | `Upper_Right` |

Port pack (feature alignment):

`/World/Network_Switches/<grid_option>/<robot_loc>/AS4610_inst/AS4610_01/Switch/Net_12_Pack_no_LED_Component_04/RJ45_Group01`

Port contacts (debug / fallback pins):

`/World/Network_Switches/<grid_option>/<robot_loc>/AS4610_inst/AS4610_01/Switch/Net_12_Pack_no_LED_Component_04/RJ45_Group01/CopperContacts/Group_14343`

Cables live at `/World/NetworkCables/Cable_<station_id>/`.

### Jack override

Default jack is `jack_upper_c2` (copper `Group_14343`). Override per station in
`config.py` via `make_station(..., jack_id=...)` or edit the `STATIONS` tuple.
See `JACK_COPPER_GROUP` for valid IDs.

## Physics

- Grasp friction: **0.8** static/dynamic, combine mode **max** (physical pinch only)
- No `FixedJoint` / weld between cable and gripper
- Cable head45 is a dynamic rigid body held by finger friction

## Debug markers

Invisible spheres under `/World/DebugPortMarkers/<station_id>/` (toggle Visibility
in the Stage panel). No rigid body, no collision:

- **yellow** `Offset` — pre-insert +0.02 m in world X from port mating center
- **red** `Insert` — port mating center
- **green** `Via_35`…`Via_95` — lift→offset waypoints

## Host-side tests

```bash
cd Isaacsim-Scripts/aayush
python3 -m unittest discover -s ur5e_6x_cable_insertions/tests -v
```
