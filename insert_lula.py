from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys

import carb
import carb.settings
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf, UsdLux

from franka_lula_controller import FrankaLulaController

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

world = World(stage_units_in_meters=1.0)

render_settings = carb.settings.get_settings()
render_settings.set_bool("/rtx/shadows/enabled", False)

stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(500.0)
dome.CreateColorAttr((1.0, 1.0, 1.0))

world.scene.add(
    FixedCuboid(
        name="ground",
        position=np.array([0.0, 0.0, -0.005]),
        prim_path="/World/Ground",
        scale=np.array([10.0, 10.0, 0.01]),
        size=1.0,
        color=np.array([0.95, 0.95, 0.95]),
    )
)

robot = add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    prim_path="/World/Franka",
)
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
physics = robot.GetVariantSet("Physics")
physics_names = list(physics.GetVariantNames())
if physics_names:
    physics.SetVariantSelection(
        next((name for name in physics_names if name.lower() == "physx"), physics_names[0])
    )

gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.001, 0.001]),
    action_deltas=np.array([0.02, 0.02]),
)

franka = world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="my_franka",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
    )
)

block = world.scene.add(
    DynamicCuboid(
        name="block",
        position=np.array([0.5, 0.0, 0.025]),
        prim_path="/World/Block",
        size=1.0,
        scale=np.array([0.004, 0.008, 0.050]),
        color=np.array([0, 0, 1]),
    )
)

PORT_POS = np.array([-0.7, 0.0, 0.20])

# Thin backplate facing +X (robot inserts along -X). Visual only — no collision needed.
port = world.scene.add(
    FixedCuboid(
        name="port",
        position=PORT_POS,
        prim_path="/World/Port",
        size=1.0,
        scale=np.array([0.006, 0.030, 0.055]),
        color=np.array([0.15, 0.55, 0.25]),
    )
)
world.scene.add(
    FixedCuboid(
        name="port_opening",
        position=PORT_POS + np.array([0.004, 0.0, 0.0]),
        prim_path="/World/PortOpening",
        size=1.0,
        scale=np.array([0.002, 0.018, 0.038]),
        color=np.array([0.05, 0.12, 0.08]),
    )
)

franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
world.reset()

lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
lula_kinematics = LulaKinematicsSolver(**lula_config)
task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**lula_config)
art_kinematics = ArticulationKinematicsSolver(franka, lula_kinematics, "panda_hand")
lula_kinematics.set_robot_base_pose(*franka.get_world_pose())

controller = FrankaLulaController(
    name="franka_controller",
    robot_articulation=franka,
    task_traj_gen=task_traj_gen,
    art_kinematics=art_kinematics,
    gripper=franka.gripper,
    tool_offset=0.05,
)

down_ori = np.array([0.0, 1.0, 0.0, 0.0])
insert_ori = np.array([-0.7071068, 0.0, 0.7071068, 0.0])
controller.add_cartesian_waypoint(
    position=np.array([0.5, 0.0, 0.20]),
    orientation=down_ori,
    pos_tolerance=0.05,
)
controller.add_cartesian_waypoint(
    position=np.array([0.5, 0.0, 0.04]),
    orientation=down_ori,
    pos_tolerance=0.001,
)
controller.add_gripper_command(action="open")

controller.add_gripper_command(action="close")

controller.add_cartesian_waypoint(
    position=np.array([0.5, 0.0, 0.20]),
    orientation=down_ori,
    pos_tolerance=0.001,
)

controller.add_cartesian_waypoint(
    position=np.array([0, 0.5, 0.20]),
    orientation=down_ori,
    pos_tolerance=0.001,
)


controller.add_cartesian_waypoint(
    position=np.array([-0.5, 0.0, 0.20]),
    orientation=down_ori,
    pos_tolerance=0.001,
)

controller.add_cartesian_waypoint(
    position=np.array([-0.5, 0.0, 0.20]),
    orientation=insert_ori,
    pos_tolerance=0.001,
)

controller.add_cartesian_waypoint(
    position=PORT_POS.copy(),
    orientation=insert_ori,
    pos_tolerance=0.001,
    linear=True,
    linear_step=0.001,
)


controller.add_gripper_command(action="open")


reset_needed = False

while simulation_app.is_running():
    world.step(render=True)

    if world.is_stopped() and not reset_needed:
        reset_needed = True

    if world.is_playing():
        if reset_needed:
            world.reset()
            controller.reset()
            lula_kinematics.set_robot_base_pose(*franka.get_world_pose())
            reset_needed = False
            continue

        joint_pos = franka.get_joint_positions()
        if joint_pos is None:
            continue

        franka.get_articulation_controller().apply_action(
            controller.forward(joint_pos)
        )

simulation_app.close()
