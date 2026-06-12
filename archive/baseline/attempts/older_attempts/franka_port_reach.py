import time
import numpy as np
import carb

from isaacsim import SimulationApp

# Start Isaac Sim
simulation_app = SimulationApp({"headless": False})

# from omni.isaac.core import World
from isaacsim.core.utils.extensions import get_extension_path_from_name

from isaacsim.core.api import World
# from omni.isaac.franka import Franka
import os
# from omni.isaac.motion_generation import (
#     LulaKinematicsSolver,
#     ArticulationKinematicsSolver
# )
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader
# from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.api.controllers.articulation_controller import ArticulationController

from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget
import omni.usd

from isaacsim.core.prims import XFormPrim

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

USD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"

PORT_TARGETS = [
"/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
]

EE_FRAME = "panda_hand"



# -------------------------------------------------------
# WORLD
# -------------------------------------------------------
world = World(stage_units_in_meters=1.0)
add_reference_to_stage(usd_path=USD_PATH, prim_path="/World")
stage = omni.usd.get_context().get_stage()

# robot = Franka("/World/Franka")
robot = world.scene.get_object("/World/Franka")
if not robot:
    print("Robot not found")
    exit
#my_controller = KinematicsSolver(robot)
#if not my_controller:
#    print("Controller not found")
#    exit
# articulation_controller = robot.get_articulation_controller()
articulation_controller = ArticulationController()
robot_view = Articulation(prim_paths_expr="/World/Franka") #, name="franka_panda_view")
if not articulation_controller:
    print("Articulation controller not found")
    exit
if not robot_view:
    print("robot_view not found")
    exit
articulation_controller.initialize(robot_view)


mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
print("mg_extension_path")
print(mg_extension_path)
kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
print("kinematics_config_dir")
print(kinematics_config_dir)
kinematics_solver = LulaKinematicsSolver(
    robot_description_path = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
    urdf_path = kinematics_config_dir + "/franka/lula_franka_gen.urdf"
)
if not kinematics_solver:
    print("kinematics_solver not found")
    exit
end_effector_name = "right_gripper" # EE_FRAME
articulation_kinematics_solver = ArticulationKinematicsSolver(robot, kinematics_solver, end_effector_name)
if not articulation_kinematics_solver:
    print("articulation_kinematics_solver not found")
    exit
my_controller = articulation_kinematics_solver



def get_target_pose(prim_path):
    #prim = XFormPrim(prim_paths_expr=prim_path)
    #pos, rot = prim.get_world_poses()
    #return pos[0], rot[0]
    prim = stage.GetPrimAtPath(prim_path)
    pose = omni.usd.utils.get_world_transform_matrix(prim)
    # print("Matrix Form:", pose)
    # print("Translation: ", pose.ExtractTranslation())
    q = pose.ExtractRotation().GetQuaternion()
    # print(
        # "Rotation: ", q.GetReal(), ",", q.GetImaginary()[0], ",", q.GetImaginary()[1], ",", q.GetImaginary()[2]
    # )
    pos = pose.ExtractTranslation()
    rot = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
    # print("Position: ", pos)
    # print("Rotation: ", rot)
    return np.array(pos), np.array(rot)


def reset_robot():
    robot.initialize()
    articulation_controller.reset()



reset_needed = False
port = PORT_TARGETS[0]
while simulation_app.is_running():
    world.step(render=True)
    # print(robot)
    # print(my_controller)
    # print(articulation_controller)
    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if world.is_playing():
        if reset_needed:
            world.reset()
            reset_needed = False
        #for port in PORT_TARGETS:
        print(f"Attempting target: {port}")
        print(stage.GetPrimAtPath(port))
        pos, rot = get_target_pose(port)
        #reached = attempt_reach(pos, rot)
        # start_time = time.time()
        # while time.time() - start_time < 2:
        pos[0] = pos[0] + 0.5
        actions, reached = my_controller.compute_inverse_kinematics(
            target_position=pos,
            target_orientation=rot,
        )
        print(actions)
        print(robot)
        print(my_controller)
        print(articulation_controller)
        if reached:
            articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution. No action is being taken.")
        # world.step(render=True)
        # if not reached:
            # print("Timeout — resetting robot")
            # world.reset()
        # world.pause()
        print("All targets processed")

simulation_app.close()