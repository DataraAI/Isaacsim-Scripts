"""Config for UR10e behaviour-tree cable grasp + lift."""

from __future__ import annotations

import numpy as np

# Grasp target under crystal head 45.
GRASP_PART_PATH = "/World/NetworkCable/E_crystal_head1_45/E_part006_44"
CRYSTAL_HEAD45_PATH = "/World/NetworkCable/E_crystal_head1_45"
CABLE_ROOT_PATH = "/World/NetworkCable"
CABLE_SUPPORT_PATH = "/World/CableSupportBlock"
UR10E_PRIM_PATH = "/World/UR10eMount/ur10e"

# Top-down Robotiq tool with +90° yaw about world Z (wxyz).
# Fingers open along ±Y so they pinch the sides of the X-aligned crystal head.
_DOWN = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
_YAW90_Z = np.array(
    [np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)],
    dtype=np.float64,
)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


GRASP_ORIENTATION = _quat_multiply(_YAW90_Z, _DOWN)

OBSERVE_HAND = np.array([0.45, -0.20, 1.45], dtype=np.float64)
# Hover height above the grasp tip (world +Z), then descend straight down.
GRASP_HOVER_CLEARANCE_M = 0.12
GRASP_LIFT_CLEARANCE_M = 0.12
# Pinch slightly below the part center (world −Z).
GRASP_DESCEND_CLEARANCE_M = -0.003
# Shift grasp tip toward −X so fingers drop into the support U-notch
# instead of colliding with CableSupportBlock/Right.
GRASP_X_OFFSET_M = -0.015

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

ROBOTIQ_CLOSED_RAD = float(np.deg2rad(52.0))  # firmer pinch through lift
ROBOTIQ_CONTACT_RAD = float(np.deg2rad(12.0))
GRASP_CLOSE_WAIT_FRAMES = 160
GRASP_SQUEEZE_HOLD_FRAMES = 40

# Tool tip offset from Lula ee_frame toward the cable (meters along tool +Z / world −Z).
TOOL_OFFSET_M = 0.16
