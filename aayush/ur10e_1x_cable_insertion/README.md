# ur10e_1x_cable_insertion

Behaviour-tree demo on top of [`asset_spawn/`](../asset_spawn/): detect
`E_part006_44` on crystal head 45, hover above it with the Robotiq 2F-85,
descend straight down, physically grasp, and lift.

## Run

From `Isaacsim-Scripts`:

```bash
/home/aayush/isaacsim/python.sh aayush/ur10e_1x_cable_insertion/main.py
```

Headless:

```bash
/home/aayush/isaacsim/python.sh aayush/ur10e_1x_cable_insertion/main.py --headless
```

Success marker:

```text
[BT CABLE PASS] Completed … steps in … frames
```

## Behaviour tree

```text
Sequence: Grasp and lift ethernet cable head
├── Action: Move to observation pose
├── Selector: Find grasp part on crystal head 45
├── Selector: Acquire the cable (hover above → open → descend → close → lift)
└── Condition: Confirm grasp after lift
```

Engine: [`tanish/behaviour_tree_insertion`](../../tanish/behaviour_tree_insertion/).
Motion: Lula + `detailedInsertion/cable/franka_motion_controller.py`.

## Grasp target

`/World/NetworkCable/E_crystal_head1_45/E_part006_44`

Approach: fingers pointing **down**, hover above the part, descend, side-pinch,
and lift. Uses the support block U-notch under head45. Pause/Stop in the Isaac
UI are respected.

Crystal heads get **one** rigid body each with convexHull mesh collision for a
physical pinch (no FixedJoint).

## Host-side JSON test

```bash
cd Isaacsim-Scripts/aayush
python3 -m unittest discover -s ur10e_1x_cable_insertion/tests -v
```
