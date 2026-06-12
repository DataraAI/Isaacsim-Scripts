"""Datacenter reference + FollowTarget-injected Franka; Franka/target layout applied after scene build."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb
from pxr import Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget

USD_WORLD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"
PORT_PRIM_PATH = (
    "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/"
    "SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
)

# Set before add_task: injected Franka world pose (applied after first reset when prims exist).
FRANKA_WORLD_POSITION = np.array([34.3, -100.0, 140.0], dtype=np.float64)
FRANKA_EULER_DEGREES = np.array([0.0, 0.0, -180.0], dtype=np.float64)
FRANKA_WORLD_ORIENTATION = euler_angles_to_quat(FRANKA_EULER_DEGREES, degrees=True)
FRANKA_SCALE = np.array([57.0, 57.0, 57.0], dtype=np.float64)

TARGET_LOCAL_SCALE = np.array([3.0, 3.0, 3.0], dtype=np.float64)


def world_position_only(prim_path: str) -> np.ndarray:
    """World-space XYZ of prim origin in the stage."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m.ExtractTranslation(), dtype=np.float64)


my_world = World(stage_units_in_meters=1.0)
add_reference_to_stage(USD_WORLD_PATH, "/World")

if not is_prim_path_valid(PORT_PRIM_PATH):
    carb.log_error(f"Port prim not found: {PORT_PRIM_PATH}")
    simulation_app.close()
    raise SystemExit(1)
PORT_WORLD_POSITION = world_position_only(PORT_PRIM_PATH)

# FollowTarget only creates Franka/target on reset(), so layout is applied immediately after first reset.
my_task = FollowTarget(name="follow_target_task")
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka = my_world.scene.get_object(franka_name)
my_target = my_world.scene.get_object(target_name)
my_controller = KinematicsSolver(my_franka)
articulation_controller = my_franka.get_articulation_controller()


def apply_franka_and_target_layout():
    """Franka: fixed world pose + scale. Target: port world position, default orientation, scale 3."""
    my_franka.set_local_scale(FRANKA_SCALE)
    my_franka.set_world_pose(position=FRANKA_WORLD_POSITION, orientation=FRANKA_WORLD_ORIENTATION)
    _, target_world_orientation = my_target.get_world_pose()
    my_target.set_world_pose(position=PORT_WORLD_POSITION, orientation=target_world_orientation)
    my_target.set_local_scale(TARGET_LOCAL_SCALE)


apply_franka_and_target_layout()
my_world.stop()

reset_needed = False
carb_printed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            apply_franka_and_target_layout()
            reset_needed = False
            carb_printed = False
        target_pos, target_orient = my_target.get_world_pose()
        actions, succ = my_controller.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_orient,
        )
        if succ:
            articulation_controller.apply_action(actions)
            carb_printed = False
        elif not carb_printed:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")
            carb_printed = True

simulation_app.close()
