"""Follow-target IK with datacenter USD: re-spawn Franka at the original scene Franka pose and place the target at a port."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as Rot

from isaacsim.core.api import World
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget

USD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"
PORT_PRIM_PATH = (
    "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/"
    "SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
)
LEGACY_FRANKA_PATH = "/World/Franka"
TARGET_PRIM_PATH = "/World/TargetCube"


def world_translation_orientation_scale(prim_path: str):
    """Decompose world (stage) transform of a prim into translation, quaternion (w,x,y,z), and scale."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    m = np.array(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    trans = m[:3, 3].copy()
    c0, c1, c2 = m[:3, 0], m[:3, 1], m[:3, 2]
    sx, sy, sz = np.linalg.norm(c0), np.linalg.norm(c1), np.linalg.norm(c2)
    scale = np.array([sx, sy, sz], dtype=np.float64)
    if sx > 1e-10 and sy > 1e-10 and sz > 1e-10:
        rmat = np.column_stack([c0 / sx, c1 / sy, c2 / sz])
    else:
        rmat = np.eye(3)
    q = Rot.from_matrix(rmat).as_quat()  # x, y, z, w
    orient_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return trans, orient_wxyz, scale


my_world = World(stage_units_in_meters=1.0)
add_reference_to_stage(USD_PATH, "/World")

if not is_prim_path_valid(LEGACY_FRANKA_PATH):
    carb.log_error(f"Expected Franka at {LEGACY_FRANKA_PATH} in the referenced USD.")
    simulation_app.close()
    raise SystemExit(1)
saved_translation, saved_orientation, saved_scale = world_translation_orientation_scale(LEGACY_FRANKA_PATH)
if is_prim_path_valid(PORT_PRIM_PATH):
    port_world_translation = world_translation_orientation_scale(PORT_PRIM_PATH)[0]
else:
    carb.log_warn(f"Port prim not found; target stays at task default: {PORT_PRIM_PATH}")
    port_world_translation = None

delete_prim(LEGACY_FRANKA_PATH)

my_task = FollowTarget(
    name="follow_target_task",
    franka_prim_path=LEGACY_FRANKA_PATH,
    target_prim_path=TARGET_PRIM_PATH,
)
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka = my_world.scene.get_object(franka_name)
my_target = my_world.scene.get_object(target_name)
my_controller = KinematicsSolver(my_franka)
articulation_controller = my_franka.get_articulation_controller()


def apply_franka_and_target_from_datacenter():
    """Restore Franka world pose from the removed asset; move target cube to the port (world translation)."""
    my_franka.set_local_scale(saved_scale)
    my_franka.set_world_pose(position=saved_translation, orientation=saved_orientation)
    if port_world_translation is not None:
        _, target_orient = my_target.get_world_pose()
        my_target.set_world_pose(position=port_world_translation, orientation=target_orient)


apply_franka_and_target_from_datacenter()

reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            apply_franka_and_target_from_datacenter()
            reset_needed = False
        # IK uses stage/world frame; use world pose so it stays correct if /World is transformed.
        target_pos, target_orient = my_target.get_world_pose()
        actions, succ = my_controller.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_orient,
        )
        if succ:
            articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")

simulation_app.close()
