# Core Isaac Sim App Initialization
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Standard and Third-Party Imports
import sys
import carb
import numpy as np
import omni.usd
from pxr import PhysxSchema

# Isaac Sim Imports
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path

# Custom Controller Import
from franka_lula_controller import FrankaLulaController

# Set True to log trajectory segment stats when each segment starts.
DEBUG_TRAJ_LOG = False

# Asset Path Validation
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

# World and Environment Setup
my_world = World(stage_units_in_meters=1.0)
DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)

add_reference_to_stage(usd_path=DATAHALL_USD, prim_path="/World/DataHall")

# Robot and Gripper Setup
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
robot = add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
robot.GetVariantSet("Mesh").SetVariantSelection("Quality")

gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.005, 0.005]),
    action_deltas=np.array([0.02, 0.02]),
)

my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="my_franka",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
        position=np.array([0.45, -0.25, 1.5]),
    )
)

# Object Setup (Cube)
cube = my_world.scene.add(
    DynamicCuboid(
        name="cube",
        position=np.array([0.3, 0.3, 1.5]),
        prim_path="/World/Cube",
        scale=np.array([0.0515, 0.0515, 0.0515]),
        size=0.4,
        color=np.array([0, 0, 1]),
    )
)

_cube_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.prim)
_cube_rb_api.CreateDisableGravityAttr(True)

# Initialization and Controllers
my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
my_world.reset()

PHYSICS_DT = 1.0 / 60.0
lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
lula_kinematics = LulaKinematicsSolver(**lula_config)
task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**lula_config)
art_kinematics = ArticulationKinematicsSolver(my_franka, lula_kinematics, "panda_hand")

base_pos, base_ori = my_franka.get_world_pose()
lula_kinematics.set_robot_base_pose(base_pos, base_ori)

articulation_controller = my_franka.get_articulation_controller()

franka_controller = FrankaLulaController(
    name="franka_controller",
    robot_articulation=my_franka,
    lula_kinematics=lula_kinematics,
    task_traj_gen=task_traj_gen,
    art_kinematics=art_kinematics,
    gripper=my_franka.gripper,
    tool_offset=0.1,
    physics_dt=PHYSICS_DT,
    position_tolerance=0.005,
    cartesian_step=0.001,
    insert_velocity_scale=0.3,
)

# Helper Functions
def get_prim_pose(prim_path):
    stage = omni.usd.get_context().get_stage()
    target_prim = stage.GetPrimAtPath(prim_path)

    if not target_prim.IsValid():
        print(f"Error: No primitive found at {prim_path}")
        return None, None

    position, orientation = get_world_pose(prim_path)
    return position, orientation


def register_robot_base_pose():
    base_pos, base_ori = my_franka.get_world_pose()
    lula_kinematics.set_robot_base_pose(base_pos, base_ori)


# Targets Extraction
TARGET_PATH = "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"

target_pos, target_rot = get_prim_pose(TARGET_PATH)
print("Target Position: ", target_pos)

if target_pos is None:
    carb.log_error("Target path not found. Exiting.")
    simulation_app.close()
    sys.exit()

insert_x = target_pos[0]
insert_y = target_pos[1]
insert_z = target_pos[2]

# Command Queue Setup
# Waypoint positions are block-center goals in world coordinates.
down_ori = np.array([0, 1, 0, 0])
insert_ori = np.array([-0.7071, 0, 0.7071, 0])

# 1. Hover
franka_controller.add_cartesian_waypoint(
    position=np.array([0.3, 0.3, 1.8]), orientation=down_ori, pos_tolerance=0.05
)

# 2. Lower to grab
franka_controller.add_cartesian_waypoint(
    position=np.array([0.3, 0.3, 1.515]), orientation=down_ori, pos_tolerance=0.001
)
franka_controller.add_gripper_command(action="close")

# 3. Lift up
franka_controller.add_cartesian_waypoint(
    position=np.array([0.3, 0.3, 1.8]), orientation=down_ori, pos_tolerance=0.05
)

# 4. Move over the target
franka_controller.add_cartesian_waypoint(
    position=np.array([insert_x + 0.3, insert_y, 1.8]), orientation=down_ori, pos_tolerance=0.05
)

# 5. Rotate the wrist
franka_controller.add_cartesian_waypoint(
    position=np.array([insert_x + 0.3, insert_y, 1.8]), orientation=insert_ori, pos_tolerance=0.05
)

# 6. Align at 10 cm approach (dense linear task-space path on block center)
franka_controller.add_cartesian_waypoint(
    position=np.array([insert_x + 0.1, insert_y, insert_z]),
    orientation=insert_ori,
    pos_tolerance=0.0005,
    linear=True,
    cartesian_step=0.001,
    track_block=True,
    max_frames=1200,
)

# 7. Pure-X insertion into port (slow, hold Y/Z at segment start)
franka_controller.add_cartesian_waypoint(
    position=np.array([insert_x, insert_y, insert_z]),
    orientation=insert_ori,
    pos_tolerance=0.001,
    linear=True,
    cartesian_step=0.001,
    track_block=True,
    hold_yz=True,
    slow_motion=True,
    max_frames=2400,
)

# 8. Release
franka_controller.add_gripper_command(action="open")

# Main Simulation Loop
reset_needed = False
task_completed = False
_last_cmd_index = -1

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
        task_completed = False

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            franka_controller.reset()
            register_robot_base_pose()
            reset_needed = False
            task_completed = False
            _last_cmd_index = -1

        current_joint_pos = my_franka.get_joint_positions()
        current_block_pos, _ = cube.get_world_pose()

        actions = franka_controller.forward(
            current_joint_positions=current_joint_pos,
            current_tracked_position=current_block_pos,
        )
        articulation_controller.apply_action(actions)

        if DEBUG_TRAJ_LOG:
            dbg = franka_controller.get_traj_debug_state()
            if dbg is not None and dbg["cmd_index"] != _last_cmd_index:
                _last_cmd_index = dbg["cmd_index"]
                yz_err = current_block_pos[1:] - dbg["goal_block"][1:]
                carb.log_info(
                    f"lula segment wp={dbg['cmd_index']} actions={dbg['num_actions']} "
                    f"duration={dbg['duration_s']:.3f}s waypoints={dbg.get('num_waypoints', 'n/a')} "
                    f"fallback_ik={dbg['fallback_ik']} block_yz_err={yz_err}"
                )

        if franka_controller.is_done() and not task_completed:
            print("Done picking and placing. Task completed.")
            task_completed = True

simulation_app.close()
