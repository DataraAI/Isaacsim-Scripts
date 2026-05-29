"""Two simultaneous Franka Lula insert pipelines for two switch panels."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys

import carb
import numpy as np
import omni.usd
from pxr import PhysxSchema

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

from franka_lula_controller import FrankaLulaController

DEBUG_TRAJ_LOG = True
NUM_BLOCKS = 4

# Tuned Y offset for QSFP prim vs visual port center.
INSERT_Y_OFFSET = -0.00145
INSERT_Y_BIAS = 0.0
INSERT_Z_OFFSET_AFTER_FIRST = 0.003

SETTLE_FRAMES = 5
CARTESIAN_STEP = 0.002
ALIGN_TOL = 0.001
INSERT_TOL = 0.001
ALIGN_MAX_FRAMES = 600
INSERT_MAX_FRAMES = 500
TRANSIT_MAX_FRAMES = 120
GRIPPER_WAIT_FRAMES = 40

PICK_X_POSITIONS = [0.22, 0.26, 0.30, 0.34]
PICK_Z = 1.5
GRASP_Z = 1.515
HOVER_Z = 1.8

ROBOT_BASE_X = 0.45
ROBOT_BASE_Z = 1.3
PANEL_A_ROBOT_Y = 0.15
PANEL_B_ROBOT_Y = -1.45
PANEL_A_PICK_Y = -0.3
PANEL_B_PICK_Y = -0.85

PANEL_A_TARGET_PATHS = [
    "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01",
    "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_02/Connector_Pair_01/QSFP_DD_Connector_A_01",
    "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_03/Connector_Pair_01/QSFP_DD_Connector_A_01",
    "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_04/Connector_Pair_01/QSFP_DD_Connector_A_01",
]

# TODO: Replace these with panel-B slot prim paths.
PANEL_B_TARGET_PATHS = [
    "/World/DataHall/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01",
    "/World/DataHall/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_02/Connector_Pair_01/QSFP_DD_Connector_A_01",
    "/World/DataHall/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_03/Connector_Pair_01/QSFP_DD_Connector_A_01",
    "/World/DataHall/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_04/Connector_Pair_01/QSFP_DD_Connector_A_01",
]

CUBE_COLORS = [
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.8, 0.2]),
    np.array([1.0, 0.2, 0.0]),
    np.array([0.9, 0.85, 0.1]),
]

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)
add_reference_to_stage(usd_path=DATAHALL_USD, prim_path="/World/DataHall")

asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


def get_prim_pose(prim_path):
    stage = omni.usd.get_context().get_stage()
    target_prim = stage.GetPrimAtPath(prim_path)
    if not target_prim.IsValid():
        carb.log_error(f"No primitive found at {prim_path}")
        return None, None
    position, orientation = get_world_pose(prim_path)
    return position, orientation


def add_pick_and_insert_job(
    controller: FrankaLulaController,
    pick_x: float,
    pick_y: float,
    insert_x: float,
    insert_y: float,
    insert_z: float,
) -> None:
    pick_pos = np.array([pick_x, pick_y, PICK_Z])
    insert_y_adj = insert_y + INSERT_Y_OFFSET

    controller.add_cartesian_waypoint(
        position=np.array([pick_pos[0], pick_pos[1], HOVER_Z]),
        orientation=down_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    controller.add_cartesian_waypoint(
        position=np.array([pick_pos[0], pick_pos[1], GRASP_Z]),
        orientation=down_ori,
        pos_tolerance=0.001,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    controller.add_gripper_command(action="close", wait_frames=GRIPPER_WAIT_FRAMES)
    controller.add_cartesian_waypoint(
        position=np.array([pick_pos[0], pick_pos[1], HOVER_Z]),
        orientation=down_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    controller.add_cartesian_waypoint(
        position=np.array([insert_x + 0.3, insert_y_adj, HOVER_Z]),
        orientation=down_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    controller.add_cartesian_waypoint(
        position=np.array([insert_x + 0.3, insert_y_adj, HOVER_Z]),
        orientation=insert_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )
    controller.add_cartesian_waypoint(
        position=np.array([insert_x + 0.1, insert_y_adj, insert_z]),
        orientation=insert_ori,
        pos_tolerance=ALIGN_TOL,
        align_yz_only=True,
        track_block=True,
        max_frames=ALIGN_MAX_FRAMES,
    )
    controller.add_cartesian_waypoint(
        position=np.array([insert_x, insert_y_adj, insert_z]),
        orientation=insert_ori,
        pos_tolerance=INSERT_TOL,
        x_only_insert=True,
        cartesian_step=CARTESIAN_STEP,
        track_block=True,
        max_frames=INSERT_MAX_FRAMES,
    )
    controller.add_gripper_command(action="open", wait_frames=GRIPPER_WAIT_FRAMES)
    controller.add_cartesian_waypoint(
        position=np.array([insert_x + 0.15, insert_y_adj, HOVER_Z]),
        orientation=insert_ori,
        pos_tolerance=0.05,
        max_frames=TRANSIT_MAX_FRAMES,
    )


def create_panel_system(
    panel_name: str,
    robot_prim_path: str,
    robot_name: str,
    cubes_prefix: str,
    robot_base_y: float,
    pick_y: float,
    target_paths: list[str],
):
    if len(target_paths) != NUM_BLOCKS:
        carb.log_error(f"{panel_name}: expected {NUM_BLOCKS} target paths, got {len(target_paths)}")
        simulation_app.close()
        sys.exit()

    for path in target_paths:
        if "REPLACE_" in path:
            carb.log_error(f"{panel_name}: target path placeholder not replaced: {path}")
            simulation_app.close()
            sys.exit()

    robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path=robot_prim_path)
    robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

    gripper = ParallelGripper(
        end_effector_prim_path=f"{robot_prim_path}/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.005, 0.005]),
        action_deltas=np.array([0.02, 0.02]),
    )

    manipulator = my_world.scene.add(
        SingleManipulator(
            prim_path=robot_prim_path,
            name=robot_name,
            end_effector_prim_path=f"{robot_prim_path}/panda_rightfinger",
            gripper=gripper,
            position=np.array([ROBOT_BASE_X, robot_base_y, ROBOT_BASE_Z]),
        )
    )
    manipulator.gripper.set_default_state(manipulator.gripper.joint_opened_positions)

    cubes = []
    for i in range(NUM_BLOCKS):
        cube = my_world.scene.add(
            DynamicCuboid(
                name=f"{cubes_prefix}_{i}",
                position=np.array([PICK_X_POSITIONS[i], pick_y, PICK_Z]),
                prim_path=f"/World/{cubes_prefix}_{i}",
                scale=np.array([0.04, 0.04, 0.04]),
                size=0.4,
                color=CUBE_COLORS[i],
            )
        )
        rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.prim)
        rb_api.CreateDisableGravityAttr(True)
        cubes.append(cube)

    lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula_kinematics = LulaKinematicsSolver(**lula_config)
    task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**lula_config)
    art_kinematics = ArticulationKinematicsSolver(manipulator, lula_kinematics, "panda_hand")
    articulation_controller = manipulator.get_articulation_controller()

    controller = FrankaLulaController(
        name=f"{robot_name}_controller",
        robot_articulation=manipulator,
        lula_kinematics=lula_kinematics,
        task_traj_gen=task_traj_gen,
        art_kinematics=art_kinematics,
        gripper=manipulator.gripper,
        tool_offset=0.1,
        physics_dt=PHYSICS_DT,
        position_tolerance=0.005,
        cartesian_step=CARTESIAN_STEP,
        settle_frames=SETTLE_FRAMES,
        insert_orientation_tolerance=0.02,
    )

    return {
        "panel_name": panel_name,
        "manipulator": manipulator,
        "controller": controller,
        "lula_kinematics": lula_kinematics,
        "articulation_controller": articulation_controller,
        "cubes": cubes,
        "pick_y": pick_y,
        "target_paths": target_paths,
        "cmd_index_to_cube": [],
        "job_insert_goals": [],
        "last_cmd_index": -1,
        "last_job_logged": -1,
        "task_completed": False,
    }


def register_robot_base_pose(panel_state: dict) -> None:
    base_pos, base_ori = panel_state["manipulator"].get_world_pose()
    panel_state["lula_kinematics"].set_robot_base_pose(base_pos, base_ori)


def build_panel_jobs(panel_state: dict) -> None:
    insert_targets = []
    for path in panel_state["target_paths"]:
        pos, _ = get_prim_pose(path)
        if pos is None:
            carb.log_error(f"{panel_state['panel_name']}: target not found: {path}")
            simulation_app.close()
            sys.exit()
        insert_targets.append(pos)
        carb.log_info(f"{panel_state['panel_name']} port {path.split('/')[-2]}: {pos}")

    controller = panel_state["controller"]
    cmd_index_to_cube = panel_state["cmd_index_to_cube"]
    job_insert_goals = panel_state["job_insert_goals"]

    for job_i in range(NUM_BLOCKS):
        target_pos = insert_targets[job_i]
        insert_x = float(target_pos[0])
        insert_y = float(target_pos[1]) + INSERT_Y_BIAS
        insert_z = float(target_pos[2])
        if job_i > 0:
            insert_z += INSERT_Z_OFFSET_AFTER_FIRST
        job_insert_goals.append(np.array([insert_x, insert_y, insert_z]))

        start_cmd = len(controller._command_queue)
        add_pick_and_insert_job(
            controller=controller,
            pick_x=PICK_X_POSITIONS[job_i],
            pick_y=panel_state["pick_y"],
            insert_x=insert_x,
            insert_y=insert_y,
            insert_z=insert_z,
        )
        end_cmd = len(controller._command_queue)
        for _ in range(start_cmd, end_cmd):
            cmd_index_to_cube.append(job_i)

        carb.log_info(
            f"{panel_state['panel_name']} job {job_i}: "
            f"cube ({PICK_X_POSITIONS[job_i]:.2f}, {panel_state['pick_y']:.2f}, {PICK_Z:.3f}) -> "
            f"port ({insert_x:.3f}, {insert_y:.3f}, {insert_z:.3f})"
        )


def active_cube_index(controller: FrankaLulaController, cmd_index_to_cube: list[int]) -> int:
    idx = min(controller._current_command_index, len(cmd_index_to_cube) - 1)
    return cmd_index_to_cube[idx]


PHYSICS_DT = 1.0 / 60.0
down_ori = np.array([0, 1, 0, 0])
insert_ori = np.array([-0.7071, 0, 0.7071, 0])

panel_a_state = create_panel_system(
    panel_name="PanelA",
    robot_prim_path="/World/Franka",
    robot_name="franka_panel_a",
    cubes_prefix="CubeA",
    robot_base_y=PANEL_A_ROBOT_Y,
    pick_y=PANEL_A_PICK_Y,
    target_paths=PANEL_A_TARGET_PATHS,
)
panel_b_state = create_panel_system(
    panel_name="PanelB",
    robot_prim_path="/World/Franka_B",
    robot_name="franka_panel_b",
    cubes_prefix="CubeB",
    robot_base_y=PANEL_B_ROBOT_Y,
    pick_y=PANEL_B_PICK_Y,
    target_paths=PANEL_B_TARGET_PATHS,
)
panel_states = [panel_a_state, panel_b_state]

my_world.reset()
for panel_state in panel_states:
    register_robot_base_pose(panel_state)
    build_panel_jobs(panel_state)

reset_needed = False

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
        for panel_state in panel_states:
            panel_state["task_completed"] = False

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            for panel_state in panel_states:
                panel_state["controller"].reset()
                register_robot_base_pose(panel_state)
                panel_state["task_completed"] = False
                panel_state["last_cmd_index"] = -1
                panel_state["last_job_logged"] = -1
            reset_needed = False

        for panel_state in panel_states:
            controller = panel_state["controller"]
            cube_i = active_cube_index(controller, panel_state["cmd_index_to_cube"])
            current_joint_pos = panel_state["manipulator"].get_joint_positions()
            current_block_pos, _ = panel_state["cubes"][cube_i].get_world_pose()

            actions = controller.forward(
                current_joint_positions=current_joint_pos,
                current_tracked_position=current_block_pos,
            )
            panel_state["articulation_controller"].apply_action(actions)

            if DEBUG_TRAJ_LOG:
                dbg = controller.get_traj_debug_state()
                if dbg is not None and dbg.get("cmd_index") != panel_state["last_cmd_index"]:
                    panel_state["last_cmd_index"] = dbg["cmd_index"]
                    if cube_i != panel_state["last_job_logged"]:
                        panel_state["last_job_logged"] = cube_i
                        goal = panel_state["job_insert_goals"][cube_i]
                        carb.log_info(
                            f"{panel_state['panel_name']} tracking cube {cube_i}, "
                            f"port goal ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f})"
                        )

            if controller.is_done() and not panel_state["task_completed"]:
                carb.log_info(f"{panel_state['panel_name']}: all four insert jobs completed.")
                panel_state["task_completed"] = True

simulation_app.close()
