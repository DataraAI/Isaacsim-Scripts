# ur10e_1x_cable_insertion

Behaviour-tree demo on top of [`asset_spawn/`](../asset_spawn/): detect
`E_part006_44`, grasp/lift with a **60°** wrist tilt (from −Z toward +X) so the
gripper clears DataHall ports, then carry the held tip to the RJ45 **offset**
and on to the **insert** point. The gripper stays firmly closed for the whole
carry. Slow seating into the port body is still not implemented.

## Run

From `Isaacsim-Scripts`:

```bash
/home/aayush/isaacsim/python.sh aayush/ur10e_1x_cable_insertion/main.py
```

## Behaviour tree

```text
Sequence: Grasp, lift, and approach ethernet port
├── Move to observation pose
├── Detect E_part006_44
├── Grasp and lift (60° tilt toward +X on descend; firm close)
├── Maneuver: yaw → offset vias → insert vias (hold_gripper throughout)
└── Confirm at_port_insert
```

## Grasp tilt

`0°` = tool along world −Z; `90°` = tool along world +X. Grasp uses
`GRASP_TILT_FROM_DOWN_DEG = 60`.

## Port approach / insert

Contacts group:

`/World/DataHall/.../RJ45_Group01/CopperContacts/Group_14345`

Insert = average of pins `1907` / `1910`. Approach = insert with **X += 0.02**.
Transit yaws near the lift tip, stages tip vias to the offset, then continues
to the insert point with `hold_gripper`. Fingertip + crystal-head physics
materials use friction **0.8** with combine mode **max**.

## Host-side JSON test

```bash
cd Isaacsim-Scripts/aayush
python3 -m unittest discover -s ur10e_1x_cable_insertion/tests -v
```
