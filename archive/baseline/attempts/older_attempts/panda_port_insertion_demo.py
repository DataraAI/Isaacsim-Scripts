import asyncio
import numpy as np

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
#from omni.isaac.core.utils.transforms import get_world_pose
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import XFormPrim
# Wrap the prim
#franka_prim = XFormPrim(prim_paths_expr="/World/Franka")
# Get world pose
#franka_world_poses = franka_prim.get_world_poses()
# len 2
# position is franka_world_poses[0][0]
# orientation is franka_world_poses[1][0]
#print(franka_world_poses)
#print(franka_world_poses[0][0])
#print(franka_world_poses[1][0])





# Port prim path example
# /World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_02





# ---------------------------
# Scene setup
# ---------------------------

world = World(stage_units_in_meters=1.0)

# Load Franka robot if not already in the scene
if get_prim_at_path("/World/Franka") is None:

    franka_asset = "/Isaac/Robots/Franka/franka.usd"

    add_reference_to_stage(
        usd_path=franka_asset,
        prim_path="/World/Franka"
    )

robot = Franka("/World/Franka")

world.reset()

# ---------------------------
# Cable object
# ---------------------------

cable_prim_path = "/World/Cable"

cable = get_prim_at_path(cable_prim_path)

# ---------------------------
# Port targets
# ---------------------------

port_targets = [
    "/World/ServerRack/Port1_Target",
    "/World/ServerRack/Port2_Target",
    "/World/ServerRack/Port3_Target"
]



# ---------------------------
# Helper: move EE
# ---------------------------

async def move_end_effector(target_pos, duration=2.0):

    steps = int(duration * 60)

    for _ in range(steps):

        robot.set_end_effector_position(
            np.array(target_pos)
        )

        world.step(render=True)

        await asyncio.sleep(0)



# ---------------------------
# Attach cable to gripper
# ---------------------------

def attach_cable():

    if cable is None:
        return

    # parent cable to gripper
    cable.GetPrim().GetParent().AddChild(robot.end_effector_prim)



# ---------------------------
# Main task
# ---------------------------

async def run_demo():

    world.play()

    # Move to cable
    cable_pos, _ = get_world_pose(cable_prim_path)

    await move_end_effector(cable_pos, 2.0)

    attach_cable()

    # Visit each port
    for port in port_targets:

        pos, _ = get_world_pose(port)

        print("Moving to port:", port)

        await move_end_effector(pos, 2.0)

        await asyncio.sleep(2)

    world.pause()



# Run demo
asyncio.ensure_future(run_demo())
