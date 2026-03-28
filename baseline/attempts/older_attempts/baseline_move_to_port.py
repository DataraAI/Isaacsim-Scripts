"""
Script for Isaac Sim Script Editor: move Franka end effector as close as possible
to a data center port when you press Play.

Usage: Run this script once in the Script Editor (Ctrl+Enter). Then press Play in the
toolbar; the arm will move to the port. The script hooks into the simulation's
post-update so it runs every frame while Play is active.
"""

import numpy as np
import omni.kit.app
import omni.timeline

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
    interface_config_loader,
)
from isaacsim.core.prims import XFormPrim

# Port prim path (Nvidia data center asset)
PORT_PRIM_PATH = "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_02"

# Approach offset (meters): move EE this far in front of the port along its -Z
APPROACH_OFFSET = 0.05
MOVE_DURATION_FRAMES = 180   # ~3 sec at 60 Hz
HOLD_DURATION_FRAMES = 300   # hold at target for ~5 sec

# -----------------------------------------------------------------------------
# One-time setup (runs when you execute the script)
# -----------------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
robot = Franka("/World/Franka")
world.reset()

kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
lula_solver = LulaKinematicsSolver(**kinematics_config)
ik_solver = ArticulationKinematicsSolver(robot, lula_solver, "panda_hand")


def get_port_pose(prim_path):
    """Return (position, orientation) in world frame. Orientation is quat (w,x,y,z)."""
    prim = XFormPrim(prim_paths_expr=prim_path)
    positions, orientations = prim.get_world_poses()
    pos = np.array(positions[0], dtype=np.float64)
    orient = np.array(orientations[0], dtype=np.float64)
    return pos, orient


def quat_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def get_target_pose_for_port(port_prim_path, offset_meters=0.0):
    """Target pose for EE: position in front of port, same orientation as port."""
    pos, quat = get_port_pose(port_prim_path)
    R = quat_to_rotation_matrix(quat)
    approach_dir = -R[:, 2]
    target_pos = pos + approach_dir * offset_meters
    return target_pos, quat


# Precompute target pose once at script load (port is in the stage)
TARGET_POS, TARGET_QUAT = get_target_pose_for_port(PORT_PRIM_PATH, APPROACH_OFFSET)
print(f"[move_to_port] Target position (world): {TARGET_POS}")

# State for the post-update callback (idle -> moving -> holding)
_state = {"phase": "idle", "frame": 0}


def _on_post_update(_event):
    """Runs every frame. When Play is active, drive the arm to the port."""
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        _state["phase"] = "idle"
        _state["frame"] = 0
        return

    phase = _state["phase"]
    frame = _state["frame"]

    if phase == "idle":
        _state["phase"] = "moving"
        _state["frame"] = 0
        print("[move_to_port] Play detected — moving to port.")
        return

    if phase == "moving":
        base_pos, base_rot = robot.get_world_pose()
        lula_solver.set_robot_base_pose(base_pos, base_rot)
        joint_positions, success = ik_solver.compute_inverse_kinematics(TARGET_POS, TARGET_QUAT)
        if success:
            robot.set_joint_positions(joint_positions)
        _state["frame"] = frame + 1
        if _state["frame"] >= MOVE_DURATION_FRAMES:
            _state["phase"] = "holding"
            _state["frame"] = 0
            print("[move_to_port] Reached target; holding.")
        return

    if phase == "holding":
        # Keep applying target so the arm holds position
        base_pos, base_rot = robot.get_world_pose()
        lula_solver.set_robot_base_pose(base_pos, base_rot)
        joint_positions, success = ik_solver.compute_inverse_kinematics(TARGET_POS, TARGET_QUAT)
        if success:
            robot.set_joint_positions(joint_positions)
        _state["frame"] = frame + 1
        # Optionally stop after hold duration; here we hold indefinitely
        if _state["frame"] >= HOLD_DURATION_FRAMES:
            _state["phase"] = "done"
        return

    # phase == "done": do nothing, arm stays where it is


# Subscribe to post-update so our logic runs every frame when you press Play.
# Keep a reference so the subscription is not garbage-collected.
_post_update_sub = omni.kit.app.get_app().get_post_update_event_stream().create_subscription_to_pop(
    _on_post_update,
    name="move_to_port"
)
print("[move_to_port] Script loaded. Press Play to move the arm to the port.")
