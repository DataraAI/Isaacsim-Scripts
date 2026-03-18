"""Follow-target IK with datacenter USD.

Placement uses **world position** (prim origin in stage space), not **local translation**
(offset from parent). Orientation and scale are taken from the prim's world xform matrix.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as Rot

from omni.usd.commands import DeletePrimsCommand

from isaacsim.core.api import World
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget

USD_WORLD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"
PORT_PRIM_PATH = (
    "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/"
    "SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
)
# Franka inside the datacenter USD is an *ancestral* prim (defined only in the referenced layer).
# It cannot be removed with destructive delete; we deactivate it instead (active=false on your edit layer).
# The path /World/Franka then stays reserved by that spec, so the task Franka is spawned on a sibling path.
LEGACY_FRANKA_PRIM_PATH = "/World/Franka"
TASK_FRANKA_PRIM_PATH = "/World/Franka_IK"
TARGET_PRIM_PATH = "/World/TargetCube"


def deactivate_ancestral_prim(prim_path: str) -> None:
    """Turn off composition for a prim that lives inside a reference (same as Editor: Deactivate)."""
    DeletePrimsCommand([prim_path], destructive=False).do()


def world_xform_position_orientation_scale(prim_path: str):
    """Decompose prim's local-to-world matrix.

    - **position** (np.ndarray shape (3,)): world-space origin of the prim in the stage.
      This is not local translation (parent-relative XYZ on the xform stack).
    - **orientation** (w,x,y,z): world rotation from the orthonormalized 3x3 block.
    - **scale** (np.ndarray shape (3,)): per-axis scale from column magnitudes of that block.
    """
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    m = np.array(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    # World position of prim origin (stage frame)
    position = np.array(m[:3, 3], dtype=np.float64, copy=True)
    c0, c1, c2 = m[:3, 0], m[:3, 1], m[:3, 2]
    sx, sy, sz = np.linalg.norm(c0), np.linalg.norm(c1), np.linalg.norm(c2)
    scale = np.array([sx, sy, sz], dtype=np.float64)
    if sx > 1e-10 and sy > 1e-10 and sz > 1e-10:
        rmat = np.column_stack([c0 / sx, c1 / sy, c2 / sz])
    else:
        rmat = np.eye(3)
    q = Rot.from_matrix(rmat).as_quat()  # x, y, z, w
    orientation_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return position, orientation_wxyz, scale


def world_position_only(prim_path: str) -> np.ndarray:
    """World-space XYZ of prim origin in the stage (for prims where only position matters)."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m.ExtractTranslation(), dtype=np.float64)


my_world = World(stage_units_in_meters=1.0)
add_reference_to_stage(USD_WORLD_PATH, "/World")

if not is_prim_path_valid(LEGACY_FRANKA_PRIM_PATH):
    carb.log_error(f"Expected Franka at {LEGACY_FRANKA_PRIM_PATH} in the referenced USD.")
    simulation_app.close()
    raise SystemExit(1)
franka_world_position, franka_world_orientation, franka_world_scale = world_xform_position_orientation_scale(
    LEGACY_FRANKA_PRIM_PATH
)
if is_prim_path_valid(PORT_PRIM_PATH):
    port_world_position = world_position_only(PORT_PRIM_PATH)
else:
    carb.log_error(f"Port prim not found; target cube is not moved: {PORT_PRIM_PATH}")
    simulation_app.close()
    raise SystemExit(1)

deactivate_ancestral_prim(LEGACY_FRANKA_PRIM_PATH)
legacy = get_current_stage().GetPrimAtPath(LEGACY_FRANKA_PRIM_PATH)
if legacy.IsValid() and legacy.IsActive():
    carb.log_error(
        f"Could not deactivate {LEGACY_FRANKA_PRIM_PATH}. It may be locked or overridden in a stronger layer."
    )
    simulation_app.close()
    raise SystemExit(1)
carb.log_info(
    f"Deactivated ancestral prim {LEGACY_FRANKA_PRIM_PATH} (referenced asset). "
    f"Spawning task Franka at {TASK_FRANKA_PRIM_PATH}."
)

my_task = FollowTarget(
    name="follow_target_task",
    franka_prim_path=TASK_FRANKA_PRIM_PATH,
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
    """Franka: legacy world position + world orientation + world scale. Target: port world position + cube orient/scale."""
    my_franka.set_local_scale(franka_world_scale)
    my_franka.set_world_pose(position=franka_world_position, orientation=franka_world_orientation)
    _, cube_world_orientation = my_target.get_world_pose()
    cube_local_scale = np.array(my_target.get_local_scale(), dtype=np.float64, copy=True)
    my_target.set_world_pose(position=port_world_position, orientation=cube_world_orientation)
    my_target.set_local_scale(cube_local_scale)


apply_franka_and_target_from_datacenter()
# World.reset() starts the timeline; leave simulation stopped until the user presses Play.
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
            apply_franka_and_target_from_datacenter()
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
