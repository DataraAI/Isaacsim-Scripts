"""Config for UR10e behaviour-tree cable grasp, lift, and port approach."""

from __future__ import annotations

import numpy as np

# Grasp target under crystal head 45.
GRASP_PART_PATH = "/World/NetworkCable/E_crystal_head1_45/E_part006_44"
CRYSTAL_HEAD45_PATH = "/World/NetworkCable/E_crystal_head1_45"
CABLE_ROOT_PATH = "/World/NetworkCable"
CABLE_SUPPORT_PATH = "/World/CableSupportBlock"
UR10E_PRIM_PATH = "/World/UR10eMount/ur10e"

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _rot_matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / float(np.linalg.norm(q))


def _orientation_tool_z_along(approach: np.ndarray) -> np.ndarray:
    """Tool +Z along approach; Robotiq opens along tool +X ≈ world ±Y.

    Matches the top-down observe convention (tool_x = world Y) so the fingers
    sit on opposite sides of a cable that runs roughly along world X, even when
    the wrist is tilted in the XZ plane.
    """

    tool_z = _normalize(np.asarray(approach, dtype=np.float64).reshape(3))
    # Project world +Y onto the plane perpendicular to the approach.
    y_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tool_x = y_world - float(np.dot(y_world, tool_z)) * tool_z
    if float(np.linalg.norm(tool_x)) < 1e-6:
        x_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tool_x = x_world - float(np.dot(x_world, tool_z)) * tool_z
    tool_x = _normalize(tool_x)
    tool_y = _normalize(np.cross(tool_z, tool_x))
    return _rot_matrix_to_quat_wxyz(np.column_stack((tool_x, tool_y, tool_z)))


# Grasp tilt: 0° = tool along world −Z; 90° = tool along world +X.
# 60° leans the gripper up toward +X so the wrist clears DataHall ports on descend.
GRASP_TILT_FROM_DOWN_DEG = 60.0
_tilt = np.deg2rad(GRASP_TILT_FROM_DOWN_DEG)
GRASP_APPROACH_DIR = np.array([np.sin(_tilt), 0.0, -np.cos(_tilt)], dtype=np.float64)
GRASP_ORIENTATION = _orientation_tool_z_along(GRASP_APPROACH_DIR)
OBSERVE_ORIENTATION = GRASP_ORIENTATION.copy()

# Keep the observation wrist above head39 before moving to the tilted grasp hover.
OBSERVE_Z_CLEARANCE_M = 0.32


def observation_hand_from_head39(
    head39_center: np.ndarray,
    support_y: float,
    block_top_z: float,
) -> np.ndarray:
    hand = np.asarray(head39_center, dtype=np.float64).reshape(3).copy()
    hand[1] = float(support_y)
    hand[2] = max(float(hand[2]), float(block_top_z)) + float(OBSERVE_Z_CLEARANCE_M)
    return hand


# Standoff along the tilted approach before descending into the grasp.
GRASP_HOVER_CLEARANCE_M = 0.12
GRASP_LIFT_CLEARANCE_M = 0.12
# Pinch slightly below the part center (world −Z).
GRASP_DESCEND_CLEARANCE_M = -0.003
# World-X shift from E_part006_44 center. Head45 sits on the +X end of the
# cable; positive X pinches closer to the rigid crystal head (less slip on lift).
# Keep small so fingers still drop into the support U-notch.
GRASP_X_OFFSET_M = 0.0

UR10E_LULA_NAME = "UR10e"
UR10E_EE_FRAME = "tool0"
UR10E_HOME_ARM = np.array(
    [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
    dtype=np.float64,
)
UR10E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

ROBOTIQ_CLOSED_RAD = float(np.deg2rad(70.0))  # absolute close target (no action_deltas)
ROBOTIQ_CONTACT_RAD = float(np.deg2rad(12.0))
GRASP_CLOSE_WAIT_FRAMES = 220
GRASP_SQUEEZE_HOLD_FRAMES = 80

# Tool tip offset from Lula ee_frame toward the cable (meters along tool +Z).
TOOL_OFFSET_M = 0.16

# Target RJ45 copper-contact group on the DataHall switch (runtime stage path).
PORT_CONTACTS_PATH = (
    "/World/DataHall/Network_Switches/AS4610_01_1x_Grid/Upper_Right/AS4610_inst/"
    "AS4610_01/Switch/Net_12_Pack_no_LED_Component_01/RJ45_Group01/CopperContacts/"
    "Group_14345"
)
PORT_PIN_A_NAME = "Copper_Pin_Component_1907"
PORT_PIN_B_NAME = "Copper_Pin_Component_1910"
# Pre-insert standoff: insert_point.x + this value.
PORT_APPROACH_X_OFFSET_M = 0.03
PORT_APPROACH_TOLERANCE_M = 0.01
# World-Z yaw applied after lift (0 at lift, −180 before translating to the port).
PORT_APPROACH_YAW_DEG = -180.0
# Yaw steps while still near the lift tip (orientation-only, tip barely moves).
PORT_APPROACH_YAW_STEPS = 6
# After yaw completes: tip blend fractions tip_start→tip_end (final 1.0 is always appended).
# Staged XY approach keeps each IK target close to the previous reachable pose.
PORT_APPROACH_VIA_FRACTIONS = (0.35, 0.60, 0.82, 0.95)
# Per-transition interpolation overrides. More steps reduce nominal joint speed.
PORT_APPROACH_TRANSITION_JOINT_STEPS = {
    (0.60, 0.82): 480,
    (0.82, 0.95): 480,
}


def port_approach_joint_steps(
    source_fraction: float,
    destination_fraction: float,
    *,
    is_final: bool,
) -> int:
    default_steps = 160 if is_final else 120
    transition = (float(source_fraction), float(destination_fraction))
    return int(PORT_APPROACH_TRANSITION_JOINT_STEPS.get(transition, default_steps))


# Extra world-Z on intermediate vias so the held cable clears the switch face.
PORT_APPROACH_VIA_Z_CLEARANCE_M = 0.04
# Settle at the pre-insert offset for three seconds at 120 physics frames/s.
PORT_OFFSET_HOLD_FRAMES = 360
# Final approach descends vertically; insertion then advances straight along -X.
PORT_FINAL_DESCENT_LINEAR_STEP_M = 0.002
PORT_INSERT_LINEAR_STEP_M = 0.001
PORT_LINEAR_IK_TOLERANCE_M = 0.002
PORT_INSERT_TOLERANCE_M = 0.01
PORT_APPROACH_WAYPOINTS = 8  # legacy; transit now uses yaw steps + via fractions

# Abort if grasped tip drifts this far from the tool tip (cable slipped out).
CABLE_IN_GRIPPER_MAX_ERR_M = 0.06
CABLE_HOLD_CHECK_EVERY_N_FRAMES = 5

# Fingertip / crystal-head friction (Isaac Sim closed-loop gripper tutorial defaults).
GRASP_FRICTION_STATIC = 0.8
GRASP_FRICTION_DYNAMIC = 0.8
GRASP_FRICTION_COMBINE_MODE = "max"
FINGERTIP_NAME_TOKENS = (
    "left_inner_finger",
    "right_inner_finger",
    "pad",
    "fingertip",
)
