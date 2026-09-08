"""Config for 6× UR10e behaviour-tree cable grasp, lift, and port approach.

Loads ``~/Desktop/Aayush_ws/DataHall_6r.usd``. Each station is one UR10e +
Robotiq 2F-85 + network cable + RJ45 target on the matching switch row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATAHALL_6R_USD = Path.home() / "Desktop/Aayush_ws/DataHall_6r.usd"

ROBOTS_SCOPE = "/World/Robots"
CABLES_SCOPE = "/World/NetworkCables"
CABLE_BLOCKS_SCOPE = "/World/CableBlocks"
NETWORK_SWITCHES_SCOPE = "/World/Network_Switches"
DEBUG_MARKER_ROOT = "/World/DebugPortMarkers"

GRASP_PART_NAME = "E_part006_44"
HEAD45_NAME = "E_crystal_head1_45"
HEAD39_NAME = "E_crystal_head2_39"

# Negative Y = left side of the rack; positive Y = right.
GRID_TOP = "AS4610_Ethernet_Row_Top_1x_Grid"
GRID_MIDDLE = "AS4610_Ethernet_Row_Middle_1x_Grid"
GRID_BOTTOM = "AS4610_01_1x_Grid"
ROBOT_LOC_NEGATIVE = "Upper_Left"
ROBOT_LOC_POSITIVE = "Upper_Right"

PORT_CONTACTS_TEMPLATE = (
    "/World/Network_Switches/{grid_option}/{robot_loc}/AS4610_inst/"
    "AS4610_01/Switch/Net_12_Pack_no_LED_Component_03/RJ45_Group01/CopperContacts/"
    "Group_14343"
)

HEIGHT_TO_GRID = {
    "top": GRID_TOP,
    "middle": GRID_MIDDLE,
    "bottom": GRID_BOTTOM,
}
SIDE_TO_ROBOT_LOC = {
    "negative": ROBOT_LOC_NEGATIVE,
    "positive": ROBOT_LOC_POSITIVE,
}


@dataclass(frozen=True)
class StationSpec:
    """One grasp+insert cell: robot, cable, support, and RJ45 contacts."""

    station_id: str
    side: str  # "negative" (left / −Y) or "positive" (right / +Y)
    height: str  # "top", "middle", "bottom"

    @property
    def grid_option(self) -> str:
        return HEIGHT_TO_GRID[self.height]

    @property
    def robot_loc(self) -> str:
        return SIDE_TO_ROBOT_LOC[self.side]

    @property
    def robot_prim_path(self) -> str:
        return f"{ROBOTS_SCOPE}/UR10e_{self.station_id}"

    @property
    def cable_root_path(self) -> str:
        return f"{CABLES_SCOPE}/Cable_{self.station_id}"

    @property
    def path45(self) -> str:
        return f"{self.cable_root_path}/{HEAD45_NAME}"

    @property
    def path39(self) -> str:
        return f"{self.cable_root_path}/{HEAD39_NAME}"

    @property
    def grasp_part_path(self) -> str:
        return f"{self.path45}/{GRASP_PART_NAME}"

    @property
    def support_floor_path(self) -> str:
        return f"{CABLE_BLOCKS_SCOPE}/Floor_{self.station_id}"

    @property
    def port_contacts_path(self) -> str:
        return PORT_CONTACTS_TEMPLATE.format(
            grid_option=self.grid_option, robot_loc=self.robot_loc
        )

    @property
    def debug_marker_root(self) -> str:
        return f"{DEBUG_MARKER_ROOT}/{self.station_id}"

    @property
    def scene_name(self) -> str:
        return f"ur10e_{self.station_id.lower()}"


def make_station(side: str, height: str) -> StationSpec:
    """Build a station. ``side`` is negative/positive; ``height`` is top/middle/bottom."""

    side_key = str(side).strip().lower()
    height_key = str(height).strip().lower()
    if height_key == "lower":
        height_key = "bottom"
    if side_key not in SIDE_TO_ROBOT_LOC:
        raise ValueError(f"side must be negative or positive, got {side!r}")
    if height_key not in HEIGHT_TO_GRID:
        raise ValueError(f"height must be top, middle, or bottom, got {height!r}")
    side_token = "NegativeY" if side_key == "negative" else "PositiveY"
    height_token = {"top": "Top", "middle": "Middle", "bottom": "Lower"}[height_key]
    return StationSpec(
        station_id=f"{side_token}_{height_token}",
        side=side_key,
        height=height_key,
    )


STATIONS: tuple[StationSpec, ...] = (
    make_station("negative", "top"),
    make_station("positive", "top"),
    make_station("negative", "middle"),
    make_station("positive", "middle"),
    make_station("negative", "bottom"),
    make_station("positive", "bottom"),
)


def stations_by_id() -> dict[str, StationSpec]:
    return {spec.station_id: spec for spec in STATIONS}


# Top-down observe pose: +90° yaw about world Z (wxyz).
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
    """Tool +Z along approach; Robotiq opens along tool +X ≈ world ±Y."""

    tool_z = _normalize(np.asarray(approach, dtype=np.float64).reshape(3))
    y_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tool_x = y_world - float(np.dot(y_world, tool_z)) * tool_z
    if float(np.linalg.norm(tool_x)) < 1e-6:
        x_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tool_x = x_world - float(np.dot(x_world, tool_z)) * tool_z
    tool_x = _normalize(tool_x)
    tool_y = _normalize(np.cross(tool_z, tool_x))
    return _rot_matrix_to_quat_wxyz(np.column_stack((tool_x, tool_y, tool_z)))


OBSERVE_ORIENTATION = _quat_multiply(_YAW90_Z, _DOWN)

# Grasp tilt: 0° = tool along world −Z; 90° = tool along world +X.
GRASP_TILT_FROM_DOWN_DEG = 60.0
_tilt = np.deg2rad(GRASP_TILT_FROM_DOWN_DEG)
GRASP_APPROACH_DIR = np.array([np.sin(_tilt), 0.0, -np.cos(_tilt)], dtype=np.float64)
GRASP_ORIENTATION = _orientation_tool_z_along(GRASP_APPROACH_DIR)

# Hover this far above E_part006_44 for the observe waypoint (meters).
OBSERVE_Z_CLEARANCE_M = 0.32

GRASP_HOVER_CLEARANCE_M = 0.12
GRASP_LIFT_CLEARANCE_M = 0.12
GRASP_DESCEND_CLEARANCE_M = -0.003
GRASP_X_OFFSET_M = 0.0
GRASP_Y_ALIGNMENT_OFFSET_M = -0.016

UR10E_LULA_NAME = "UR10e"
UR10E_EE_FRAME = "tool0"
UR10E_EE_FRAME_FALLBACK = "wrist_3_link"
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
UR10E_ARM_DRIVE_PARAMETERS = {
    "shoulder_pan_joint": (3271.49169921875, 13.085969924926758, 330.0),
    "shoulder_lift_joint": (3271.49169921875, 13.085969924926758, 330.0),
    "elbow_joint": (3271.49169921875, 13.085966110229492, 150.0),
    "wrist_1_joint": (1268.18603515625, 5.072744369506836, 56.0),
    "wrist_2_joint": (1268.18603515625, 5.070000171661377, 56.0),
    "wrist_3_joint": (1268.18603515625, 5.070000171661377, 56.0),
}
UR10E_LIVE_DAMPING_MULTIPLIER = 10.0
ROBOTIQ_DRIVE_PARAMETERS = {
    "finger_joint": (3.0, 0.00019999999494757503, 100.0),
}

ROBOTIQ_CLOSED_RAD = float(np.deg2rad(70.0))
ROBOTIQ_CONTACT_RAD = float(np.deg2rad(12.0))
GRASP_MIN_LIFT_M = 0.04
GRASP_CLOSE_WAIT_FRAMES = 220
GRASP_SQUEEZE_HOLD_FRAMES = 80

# DataHall's composed Robotiq inner-pad centers are 0.120 m from wrist_3_link.
TOOL_OFFSET_M = 0.12

PORT_PIN_A_NAME = "Copper_Pin_Component_1907"
PORT_PIN_B_NAME = "Copper_Pin_Component_1910"
PORT_APPROACH_X_OFFSET_M = 0.02
PORT_APPROACH_TOLERANCE_M = 0.04
PORT_APPROACH_YAW_DEG = -180.0
PORT_APPROACH_YAW_STEPS = 6
PORT_APPROACH_VIA_FRACTIONS = (0.35, 0.60, 0.82, 0.95)
PORT_APPROACH_VIA_Z_CLEARANCE_M = 0.04
PORT_INSERT_VIA_FRACTIONS = (0.45, 0.75)
PORT_INSERT_TOLERANCE_M = 0.035

CABLE_IN_GRIPPER_MAX_ERR_M = 0.06
CABLE_HOLD_CHECK_EVERY_N_FRAMES = 5

GRASP_FRICTION_STATIC = 5.0
GRASP_FRICTION_DYNAMIC = 5.0
GRASP_FRICTION_COMBINE_MODE = "max"
FINGERTIP_NAME_TOKENS = (
    "left_inner_finger",
    "right_inner_finger",
    "pad",
    "fingertip",
)

# Debug spheres: radius in meters (converted to stage units at spawn).
PORT_DEBUG_MARKER_SCALE_M = 0.01
PORT_MANEUVER_MARKER_SCALE_M = 0.05
DEBUG_MARKER_VISIBLE_DEFAULT = False


def debug_marker_names() -> tuple[str, ...]:
    """Prim names under each station's debug-marker xform (Offset, Insert, vias)."""

    names = ["Offset", "Insert"]
    names.extend(f"Via_{int(round(float(frac) * 100)):02d}" for frac in PORT_APPROACH_VIA_FRACTIONS)
    names.extend(
        f"InsertVia_{int(round(float(frac) * 100)):02d}" for frac in PORT_INSERT_VIA_FRACTIONS
    )
    return tuple(names)
