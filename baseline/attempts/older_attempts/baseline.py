import asyncio
import numpy as np

from omni.isaac.core import World
from omni.isaac.franka import Franka
# from omni.isaac.core.utils.stage import add_reference_to_stage
# from omni.isaac.core.utils.prims import get_prim_at_path
#from omni.isaac.core.utils.transforms import get_world_pose
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import XFormPrim

port_targets = [
    "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_02",
    "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01",
]

world = World(stage_units_in_meters=1.0)
robot = Franka("/World/Franka")
# Register robot with World so PhysX articulation is managed correctly.
# Skipping this can cause "Illegal BroadPhaseUpdateData" when stepping.
world.scene.add(robot)
world.reset()

def get_prim_world_pose(prim_path):
    prim = XFormPrim(prim_paths_expr=prim_path)
    prim_world_poses = prim.get_world_poses()
    return prim_world_poses[0][0], prim_world_poses[1][0]

def _is_valid_position(pos):
    """Avoid sending NaN/inf to PhysX (can cause Illegal BroadPhaseUpdateData)."""
    pos = np.asarray(pos)
    return pos.shape == (3,) and np.isfinite(pos).all()


def _get_end_effector_position():
    """Current end-effector position in world frame (shape (3,))."""
    positions, _ = robot.end_effector.get_world_poses()
    pos = positions[0] if positions.shape[0] == 1 else positions
    return np.array(pos, dtype=np.float64).flatten()[:3]


async def move_end_effector(target_pos, duration=2.0):
    target_pos = np.array(target_pos, dtype=np.float64).flatten()[:3]
    if not _is_valid_position(target_pos):
        print("Skipping invalid target position:", target_pos)
        return
    steps = max(1, int(duration * 60))
    start_pos = _get_end_effector_position()
    if not _is_valid_position(start_pos):
        print("Could not read current end-effector position, moving directly to target.")
        start_pos = target_pos.copy()
    for i in range(steps):
        t = (i + 1) / steps
        interp_pos = start_pos + t * (target_pos - start_pos)
        robot.set_end_effector_position(interp_pos)
        world.step(render=True)
        await asyncio.sleep(0)

async def run_demo():
    world.play()
    # Let PhysX stabilize for a few steps before sending commands (reduces BroadPhase errors)
    for _ in range(10):
        world.step(render=True)
        await asyncio.sleep(0)
    # Visit each port
    for port in port_targets:
        pos, _ = get_prim_world_pose(port)
        print("Moving to port:", port)
        await move_end_effector(pos, 2.0)
        await asyncio.sleep(2)

    world.pause()

asyncio.ensure_future(run_demo())
