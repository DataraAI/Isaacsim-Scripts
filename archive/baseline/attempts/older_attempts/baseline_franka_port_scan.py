import time
import numpy as np

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver, interface_config_loader

from isaacsim.core.prims import XFormPrim


# ----------------------------
# Setup
# ----------------------------

world = World()
robot = Franka("/World/Franka")

world.reset()

# IK solver for the Franka Emika Panda (uses bundled Lula config: URDF + robot_descriptor)
kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
lula_solver = LulaKinematicsSolver(**kinematics_config)
ik_solver = ArticulationKinematicsSolver(robot, lula_solver, "panda_hand")


# ----------------------------
# Ports list
# ----------------------------

port_paths = [
"/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01",
"/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_02"
# add more ports here
]


# ----------------------------
# Helper: get port position
# ----------------------------

def get_port_position(path):
    prim = XFormPrim(path)
    pos, rot = prim.get_world_poses()
    return pos[0], rot[0]


# ----------------------------
# Move robot to pose
# ----------------------------

def move_to_pose(target_pos, target_rot):
    start = time.time()
    while time.time() - start < 10:
        base_pos, base_rot = robot.get_world_pose()
        lula_solver.set_robot_base_pose(base_pos, base_rot)
        joint_positions, success = ik_solver.compute_inverse_kinematics(
            target_pos,
            target_rot
        )
        if success:
            robot.set_joint_positions(joint_positions)
        world.step(render=True)
    return success


# ----------------------------
# Main loop
# ----------------------------

world.play()

for port in port_paths:
    print("Attempting port:", port)
    pos, rot = get_port_position(port)
    reached = move_to_pose(pos, rot)
    if reached:
        print("Reached port")
    else:
        print("Could not reach within 10 seconds")

world.pause()
