"""Config for 6× UR5e behaviour-tree cable grasp, lift, and port approach.

Loads ``~/Desktop/Aayush_ws/DataHall_6r_ur5e.usd``. Each station is one UR5e +
Robotiq 2F-85 + network cable + RJ45 target on the matching switch row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATAHALL_6R_UR5E_USD = Path.home() / "Desktop/Aayush_ws/DataHall_6r_ur5e.usd"
# Authored USD uses DataHall_01; keep legacy /World/DataHall as fallback.
DATAHALL_PRIM_PATH = "/World/DataHall_01"
DATAHALL_PRIM_PATH_FALLBACKS = (
    "/World/DataHall_01",
    "/World/DataHall",
)
ENABLE_DATAHALL_STATIC_COLLISIONS = True
DATAHALL_COLLISION_APPROXIMATION = "none"  # triangle mesh (static accurate)
# Flatten instanceable switch/rack meshes before CollisionAPI (proxies are read-only).
DATAHALL_DEINSTANCE_BEFORE_COLLISION = True
DATAHALL_DEINSTANCE_MAX_PASSES = 4
# Rack Front_Door (MetalMeshPanel etc.) must stay in the USD but never collide —
# authored inactive; still skip/disable so reactivation or leftover CollisionAPI
# cannot block insertion approach.
DATAHALL_DISABLE_FRONT_DOOR_COLLISION = True
DATAHALL_COLLISION_SKIP_PATH_TOKENS = (
    "Front_Door",
)

ROBOTS_SCOPE = "/World/Robots"
CABLES_SCOPE = "/World/NetworkCables"
CABLE_BLOCKS_SCOPE = "/World/CableBlocks"
NETWORK_SWITCHES_SCOPE = "/World/Network_Switches"
WORK_TABLES_SCOPE = "/World/WorkTables"
# Per-height table prims in DataHall_6r_ur5e (shared NegY/PosY per level).
WORK_TABLE_PATH_BY_HEIGHT = {
    "top": "/World/WorkTable1",
    "middle": "/World/WorkTable2",
    "bottom": "/World/WorkTable3",
}
# PhysX static colliders: switches (for insert contact) + work tables + cable
# blocks. Do NOT enable the facility rack under DataHall_01 (doors/rails/etc.
# were causing false high-wrench jams). CableBlocks already author CollisionAPI
# in the USD; listing them here re-asserts triangle meshes after de-instance.
DATAHALL_COLLISION_ROOTS = (
    NETWORK_SWITCHES_SCOPE,
    *tuple(WORK_TABLE_PATH_BY_HEIGHT.values()),
    CABLE_BLOCKS_SCOPE,
)
# Explicit rack solid the NegY arm was visually clipping through.
MANEUVER_EXTRA_OBSTACLE_PATHS: tuple[str, ...] = (
    "/World/DataHall_01/DataHall_01/DataHall_Racks/Rack_42U_01/Rack_42U_01/"
    "Rack_42RU_Rear_Door_V2_Component_01/Rack_Core",
)
DEBUG_MARKER_ROOT = "/World/DebugPortMarkers"

GRASP_PART_NAME = "E_part006_44"
HEAD45_NAME = "E_crystal_head1_45"
HEAD39_NAME = "E_crystal_head2_39"

# Negative Y = left side of the rack; positive Y = right.
GRID_TOP = "AS4610_Ethernet_Row_Top_1x_Grid"
GRID_MIDDLE = "AS4610_Ethernet_Row_Middle_1x_Grid"
GRID_BOTTOM = "AS4610_01_1x_Grid"
# Lower_* packs sit under Upper_* on each AS4610 face; use Lower_Left so the
# gripper body clears the rack mesh while targeting the same jack column.
ROBOT_LOC_NEGATIVE = "Lower_Left"
ROBOT_LOC_POSITIVE = "Upper_Right"

# Verify against the live USD at scene-build time; port_features extract used Component_04.
PACK_COMPONENT = "04"

PORT_PACK_TEMPLATE = (
    "/World/Network_Switches/{grid_option}/{robot_loc}/AS4610_inst/"
    "AS4610_01/Switch/Net_12_Pack_no_LED_Component_{component}/RJ45_Group01"
)
PORT_CONTACTS_TEMPLATE = PORT_PACK_TEMPLATE + "/CopperContacts/{copper_group}"

DEFAULT_JACK_ID = "jack_upper_c2"
JACK_COPPER_GROUP = {
    "jack_upper_c0": "Group_14341",
    "jack_upper_c1": "Group_14340",
    "jack_upper_c2": "Group_14343",
    "jack_upper_c3": "Group_14342",
    "jack_upper_c4": "Group_14344",
    "jack_upper_c5": "Group_14345",
    "jack_lower_c0": "Group_14361",
    "jack_lower_c1": "Group_14360",
    "jack_lower_c2": "Group_14358",
    "jack_lower_c3": "Group_14359",
    "jack_lower_c4": "Group_14356",
    "jack_lower_c5": "Group_14357",
}

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
    jack_id: str = DEFAULT_JACK_ID

    @property
    def grid_option(self) -> str:
        return HEIGHT_TO_GRID[self.height]

    @property
    def robot_loc(self) -> str:
        return SIDE_TO_ROBOT_LOC[self.side]

    @property
    def robot_prim_path(self) -> str:
        return f"{ROBOTS_SCOPE}/UR5e_{self.station_id}"

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
    def left_block_path(self) -> str:
        return f"{CABLE_BLOCKS_SCOPE}/Left_{self.station_id}"

    @property
    def right_block_path(self) -> str:
        return f"{CABLE_BLOCKS_SCOPE}/Right_{self.station_id}"

    @property
    def port_pack_path(self) -> str:
        return PORT_PACK_TEMPLATE.format(
            grid_option=self.grid_option,
            robot_loc=self.robot_loc,
            component=PACK_COMPONENT,
        )

    @property
    def port_contacts_path(self) -> str:
        copper = JACK_COPPER_GROUP[self.jack_id]
        return PORT_CONTACTS_TEMPLATE.format(
            grid_option=self.grid_option,
            robot_loc=self.robot_loc,
            component=PACK_COMPONENT,
            copper_group=copper,
        )

    @property
    def debug_marker_root(self) -> str:
        return f"{DEBUG_MARKER_ROOT}/{self.station_id}"

    @property
    def scene_name(self) -> str:
        return f"ur5e_{self.station_id.lower()}"


def make_station(side: str, height: str, jack_id: str = DEFAULT_JACK_ID) -> StationSpec:
    """Build a station. ``side`` is negative/positive; ``height`` is top/middle/bottom."""

    side_key = str(side).strip().lower()
    height_key = str(height).strip().lower()
    if height_key == "lower":
        height_key = "bottom"
    if side_key not in SIDE_TO_ROBOT_LOC:
        raise ValueError(f"side must be negative or positive, got {side!r}")
    if height_key not in HEIGHT_TO_GRID:
        raise ValueError(f"height must be top, middle, or bottom, got {height!r}")
    if jack_id not in JACK_COPPER_GROUP:
        raise ValueError(f"jack_id must be one of {sorted(JACK_COPPER_GROUP)}, got {jack_id!r}")
    side_token = "NegativeY" if side_key == "negative" else "PositiveY"
    height_token = {"top": "Top", "middle": "Middle", "bottom": "Lower"}[height_key]
    return StationSpec(
        station_id=f"{side_token}_{height_token}",
        side=side_key,
        height=height_key,
        jack_id=jack_id,
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


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


# Spin about tool +Z after seeding from world +X. Robotiq pads open along tool
# ±Y, so +90° puts one finger at greater world X than the other (before descend).
FINGER_OPEN_YAW_ABOUT_TOOL_Z_DEG = 90.0


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
    """Robotiq tool +Z along approach; open axis ≈ world ±X after finger yaw.

    Seed tool +X from world +X projected into the plane ⊥ approach so the
    pads straddle the neck left/right in X (one finger greater world X). Optional
    ``FINGER_OPEN_YAW`` spins about tool +Z for fine adjustment.
    """

    tool_z = _normalize(np.asarray(approach, dtype=np.float64).reshape(3))
    x_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tool_x = x_world - float(np.dot(x_world, tool_z)) * tool_z
    if float(np.linalg.norm(tool_x)) < 1e-6:
        y_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tool_x = y_world - float(np.dot(y_world, tool_z)) * tool_z
    tool_x = _normalize(tool_x)
    tool_y = _normalize(np.cross(tool_z, tool_x))
    rot = np.column_stack((tool_x, tool_y, tool_z))
    yaw = np.deg2rad(float(FINGER_OPEN_YAW_ABOUT_TOOL_Z_DEG))
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    rot = rot @ np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return _rot_matrix_to_quat_wxyz(rot)


def _quat_to_rot_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm((w, x, y, z)))
    if n > 1e-9:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# Tool (Robotiq base_link) axes in Lula EE / wrist_3 frame after mount repair.
# Scale-safe Xform measure is tutorial Z+90:
#   [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
# ``repair_robot_gripper_mount_joints`` overwrites this from the live stage.
GRIPPER_AXES_IN_EE = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def lula_orientation_from_tool(tool_quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert Robotiq-tool orientation to Lula ``wrist_3`` / ``tool0`` orientation."""

    r_tool = _quat_to_rot_matrix(tool_quat_wxyz)
    r_ee = r_tool @ GRIPPER_AXES_IN_EE.T
    return _rot_matrix_to_quat_wxyz(r_ee)


def tool_orientation_from_lula(lula_quat_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of :func:`lula_orientation_from_tool` (for FK tip estimates)."""

    r_ee = _quat_to_rot_matrix(lula_quat_wxyz)
    r_tool = r_ee @ GRIPPER_AXES_IN_EE
    return _rot_matrix_to_quat_wxyz(r_tool)


# Grasp tilt in the world YZ plane (rotation about +X), 75° from −Z.
# NegativeY: −Z → −Y; PositiveY: −Z → +Y.
#   approach = [0, ±sin(θ), −cos(θ)]
# θ=0 is straight down (−Z); θ=90° points along ±Y.
# Finger tips lie along Robotiq tool +Z, so tip world-Z is below the knuckles.
GRASP_TILT_FROM_DOWN_DEG = 60.0


def _normalize_side_key(side: str) -> str:
    side_key = str(side).strip().lower()
    if side_key in ("negative", "negativey", "-y"):
        return "negative"
    if side_key in ("positive", "positivey", "+y"):
        return "positive"
    raise ValueError(f"side must be negative or positive, got {side!r}")


def grasp_approach_dir(side: str) -> np.ndarray:
    """Unit tool approach for ``side`` (NegativeY → −Y lean, PositiveY → +Y)."""

    tilt = np.deg2rad(float(GRASP_TILT_FROM_DOWN_DEG))
    s = float(np.sin(tilt))
    c = float(np.cos(tilt))
    if _normalize_side_key(side) == "negative":
        return np.array([0.0, -s, -c], dtype=np.float64)
    return np.array([0.0, s, -c], dtype=np.float64)


def grasp_tool_orientation(side: str) -> np.ndarray:
    """Robotiq-tool wxyz for the side-keyed grasp tilt (pads open ≈ ±X)."""

    return _orientation_tool_z_along(grasp_approach_dir(side))


def grasp_orientation(side: str) -> np.ndarray:
    """Lula / wrist_3 wxyz for the side-keyed grasp tilt."""

    return lula_orientation_from_tool(grasp_tool_orientation(side))


# Defaults = NegativeY (most-used station); prefer side helpers at runtime.
GRASP_APPROACH_DIR = grasp_approach_dir("negative")
GRASP_TOOL_ORIENTATION = grasp_tool_orientation("negative")
GRASP_ORIENTATION = grasp_orientation("negative")
# Observe: gripper straight down (tool +Z = world −Z), then tilt happens in orient.
OBSERVE_APPROACH_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float64)
OBSERVE_TOOL_ORIENTATION = _orientation_tool_z_along(OBSERVE_APPROACH_DIR)
OBSERVE_ORIENTATION = lula_orientation_from_tool(OBSERVE_TOOL_ORIENTATION)


def set_gripper_axes_in_ee(axes: np.ndarray) -> None:
    """Update mount axes and refresh tool↔Lula orientations derived from them."""

    global GRIPPER_AXES_IN_EE, GRASP_TOOL_ORIENTATION, GRASP_ORIENTATION
    global OBSERVE_TOOL_ORIENTATION, OBSERVE_ORIENTATION, GRASP_APPROACH_DIR
    matrix = np.asarray(axes, dtype=np.float64).reshape(3, 3).copy()
    for col in range(3):
        matrix[:, col] = _normalize(matrix[:, col])
    GRIPPER_AXES_IN_EE = matrix
    GRASP_APPROACH_DIR = grasp_approach_dir("negative")
    GRASP_TOOL_ORIENTATION = grasp_tool_orientation("negative")
    GRASP_ORIENTATION = grasp_orientation("negative")
    OBSERVE_TOOL_ORIENTATION = _orientation_tool_z_along(OBSERVE_APPROACH_DIR)
    OBSERVE_ORIENTATION = lula_orientation_from_tool(OBSERVE_TOOL_ORIENTATION)

# Absolute fingertip / TipGrasp Z in stage units after all WorkTables raised +40.
# Offsets vs each height's cable root Z match the tuned Top station.
HOVER_Z_STAGE_BY_HEIGHT = {
    "top": 370.0,
    "middle": 253.62,
    "bottom": 137.16,
}
GRASP_TIP_Z_STAGE_BY_HEIGHT = {
    "top": 348.25,  # +1 stage unit up from 347.25
    "middle": 231.87,
    "bottom": 115.41,
}
# Defaults = Top (most-used station); prefer the BY_HEIGHT maps via helpers.
HOVER_Z_STAGE = HOVER_Z_STAGE_BY_HEIGHT["top"]
GRASP_TIP_Z_STAGE = GRASP_TIP_Z_STAGE_BY_HEIGHT["top"]


def work_table_path_for(height: str) -> str:
    key = "bottom" if height == "lower" else str(height)
    path = WORK_TABLE_PATH_BY_HEIGHT.get(key)
    if not path:
        raise ValueError(f"height must be top, middle, or bottom, got {height!r}")
    return path


def hover_z_stage_for(height: str) -> float:
    key = "bottom" if height == "lower" else str(height)
    return float(HOVER_Z_STAGE_BY_HEIGHT[key])


def grasp_tip_z_stage_for(height: str) -> float:
    key = "bottom" if height == "lower" else str(height)
    return float(GRASP_TIP_Z_STAGE_BY_HEIGHT[key])


# Legacy observe clearance (meters); kept for older helpers / tests.
OBSERVE_Z_CLEARANCE_M = 0.24


def observation_hand_from_head39(
    head39_center: np.ndarray,
    block_top_z: float,
) -> np.ndarray:
    """Place the observe TCP above head39 XY, gripper still pointed down."""

    hand = np.asarray(head39_center, dtype=np.float64).reshape(3).copy()
    hand[2] = max(float(hand[2]), float(block_top_z)) + float(OBSERVE_Z_CLEARANCE_M)
    return hand


GRASP_HOVER_CLEARANCE_M = 0.12
GRASP_LIFT_CLEARANCE_M = 0.12
# Absolute TipGrasp world-X in stage units. None → use neck bbox + relative X knobs.
GRASP_TIP_X_STAGE = None
# Relative TipGrasp world-X knobs (used when GRASP_TIP_X_STAGE is None).
GRASP_TIP_X_STAGE_DELTA = 0.0
# TipGrasp world-Y nudge in stage units (mpu=0.01 → 1.0 ≈ +1 cm toward +Y).
GRASP_TIP_Y_STAGE_DELTA = 1.0
# Legacy relative Z pads; GRASP_TIP_Z_BIAS_M is applied on top of absolute tip Z.
GRASP_DESCEND_CLEARANCE_M = -0.012
# Absolute TipGrasp Z pad (meters). Lower = pads deeper on the neck.
GRASP_TIP_Z_BIAS_M = -0.022
# After close: require left AND right fingertip pads within this gap of the neck.
GRASP_BOTH_FINGERS_CONTACT = True
GRASP_FINGER_CONTACT_MAX_GAP_M = 0.002
# Also require neck X to lie between the two pad centers (true straddle).
GRASP_REQUIRE_NECK_BETWEEN_PADS = True
# Clamp TipGrasp Y into the Left/Right CableBlock gap (was warn-only).
GRASP_CLAMP_TIP_Y_TO_BLOCK_GAP = True
# Before close: shift tip in XY so open-pad midpoint matches neck center.
GRASP_CENTER_PADS_ON_NECK = True
GRASP_PAD_CENTER_MAX_NUDGE_M = 0.012
GRASP_PAD_CENTER_JOINT_STEPS = 60
# Extra +X nudge (crystal-head side); ignored when GRASP_TIP_X_STAGE is set.
GRASP_X_OFFSET_M = 0.005
# 0 = part center — keep centered so left/right pads share the neck equally.
GRASP_TOWARD_NEG_X_FRAC = 0.0
# World-Y pad on the pinch tip (meters); prefer GRASP_TIP_Y_STAGE_DELTA for stage nudges.
GRASP_Y_ALIGNMENT_OFFSET_M = 0.0
# Keep the pinch tip in the Left/Right CableBlocks gap (meters).
GRASP_BLOCK_CLEARANCE_M = 0.008
# If True, force grasp tip X to the Network cable root X (else use neck bbox X).
# Default False: neck ``E_part006_44`` X is the pinch; cable-root X sits over Left block.
GRASP_USE_CABLE_ROOT_X = False
# High tip above the neck before the final descend (stage units → meters via HOVER_Z).
DESCEND_CLEAR_Z_STAGE = HOVER_Z_STAGE
DESCEND_JOINT_STEPS = 300
DESCEND_POS_TOLERANCE_M = 0.006
DESCEND_SETTLE_FRAMES = 30
# If measured fingertip Z is this far below the TipGrasp command, re-seek before close.
GRASP_TIP_Z_CORRECT_TOL_M = 0.004

# Fingertip / crystal-head friction (dimensionless — already stage-unit invariant).
GRASP_FRICTION_STATIC = 3.0
GRASP_FRICTION_DYNAMIC = 3.0
GRASP_FRICTION_COMBINE_MODE = "max"

# Classic upright UR family home (fallback only). Prefer the station IK home
# at work-table center X — upright joints intersect DataHall after the layout rotate.
UR5E_LULA_NAME = "UR5e"
UR5E_EE_FRAME = "tool0"
UR5E_EE_FRAME_FALLBACK = "wrist_3_link"
UR5E_HOME_ARM = np.array(
    [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0],
    dtype=np.float64,
)
UR5E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# Meter-physical angular drives (stiffness, damping, maxForce). Scene applies
# angular_drive_value_for_stage (÷ mpu²) for the cm DataHall (ur10e_6x recipe).
# Gravity ON + unitsResolve×100 needs large maxForce or the arm collapses.
UR5E_ARM_DRIVE_PARAMETERS = {
    "shoulder_pan_joint": (8.0e4, 8.0e3, 1.0e4),
    "shoulder_lift_joint": (8.0e4, 8.0e3, 1.0e4),
    "elbow_joint": (6.0e4, 6.0e3, 8.0e3),
    "wrist_1_joint": (3.0e4, 3.0e3, 3.0e3),
    "wrist_2_joint": (3.0e4, 3.0e3, 3.0e3),
    "wrist_3_joint": (2.0e4, 2.0e3, 3.0e3),
}
# Live ArticulationController: multipliers + absolute floors (Isaac often keeps
# soft default efforts even after USD DriveAPI is rewritten).
UR5E_LIVE_STIFFNESS_MULTIPLIER = 4.0
# Prefer damping over stiffness so gravity cannot sag the wrist during the
# post-grasp validate gap (controller.is_done → no command for a frame).
UR5E_LIVE_DAMPING_MULTIPLIER = 30.0
UR5E_LIVE_ARM_MIN_KP = 1.0e7
UR5E_LIVE_ARM_MIN_KD = 1.5e6
# Max joint effort after stage scale (Nm). Must exceed gravity torque under ×100.
UR5E_LIVE_ARM_MAX_EFFORT = 1.0e8
# Finger drives only (meter-physical → ÷mpu² in scene). Stronger close so pads
# meet; do not touch UR5E_ARM_* here.
ROBOTIQ_DRIVE_PARAMETERS = {
    # (stiffness, damping, maxForce) — raised so close is a firm parallel pinch.
    "finger_joint": (400.0, 40.0, 800.0),
}
# Soft-mimic finger coupling after nested ArticulationRoot strip.
# Authored USD has naturalFrequency=0 → knuckles fold under pad contact and
# leave a finger gap. High frequency ≈ rigid Robotiq kinematics (standard close).
ROBOTIQ_MIMIC_NATURAL_FREQUENCY = 250.0
ROBOTIQ_MIMIC_DAMPING_RATIO = 1.2
# DataHall leaves base_link / outer knuckles as RigidBody with no MassAPI and
# no colliders → PhysX invents negative mass, kicks them out of the arm
# articulation, then MimicJoint fails with "not part of an articulation".
ROBOTIQ_LINK_MASS_KG = {
    "base_link": 0.30,
    "left_outer_knuckle": 0.035,
    "right_outer_knuckle": 0.035,
}
ROBOTIQ_DEFAULT_LINK_MASS_KG = 0.025
ROBOTIQ_DEFAULT_DIAGONAL_INERTIA = (1.0e-4, 1.0e-4, 1.0e-4)
# Live articulation PD boost on finger_joint after physics ready (gripper only).
ROBOTIQ_LIVE_STIFFNESS_MULTIPLIER = 8.0
ROBOTIQ_LIVE_DAMPING_MULTIPLIER = 16.0
ROBOTIQ_LIVE_MIN_KP = 5.0e6
ROBOTIQ_LIVE_MIN_KD = 5.0e5
ROBOTIQ_LIVE_MAX_EFFORT = 1.0e8

# ---------------------------------------------------------------------------
# Tunable Isaac Sim / PhysX stability (edit these if the arm still wobbles)
# ---------------------------------------------------------------------------
# World / controller timestep (Hz). Match Isaac World + motion controller.
PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0

# PhysX scene (applied to /physicsScene).
SCENE_SOLVER_TYPE = "TGS"  # "TGS" (stable) or "PGS"
SCENE_ENABLE_GPU_DYNAMICS = True
SCENE_ENABLE_CCD = True
# GPU broadphase buffers (defaults are 1024; DataHall contact volume overflows).
SCENE_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY = 4096
SCENE_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY = 4096
SCENE_GPU_FOUND_LOST_PAIRS_CAPACITY = 262144
SCENE_MIN_POSITION_ITERS = 8
SCENE_MIN_VELOCITY_ITERS = 2
SCENE_BOUNCE_THRESHOLD = 0.2  # relative speed below which no bounce

# PhysX articulation root (on each UR5e ArticulationRootAPI prim).
# Keep False: self-collision between wrist↔2F-85 base can blow the mount FixedJoint
# and make the gripper "fall off" under contact / large joint forces.
ARTICULATION_ENABLE_SELF_COLLISIONS = False
# robot_gripper_joint break thresholds (PhysX). -1 = unbreakable.
GRIPPER_MOUNT_BREAK_FORCE = -1.0
GRIPPER_MOUNT_BREAK_TORQUE = -1.0
# Disable collision on Robotiq base / mount shells that overlap wrist_3 (pads stay on).
GRIPPER_DISABLE_BASE_COLLISION = True
ARTICULATION_SOLVER_POSITION_ITERS = 64  # ↑ stiffer joints under gravity
ARTICULATION_SOLVER_VELOCITY_ITERS = 8
ARTICULATION_SLEEP_THRESHOLD = 0.005  # ↓ keeps arm awake while tracking
ARTICULATION_STABILIZATION_THRESHOLD = 0.001

# PhysX rigid links (arm + gripper bodies). Stage-length units for velocity.
# Gravity on: match real sim / ur10e_1x (do not zero-g the arm).
LINK_DISABLE_GRAVITY = False
LINK_LINEAR_DAMPING = 0.2  # viscous; ↑ reduces translation shake
LINK_ANGULAR_DAMPING = 1.5  # viscous; ↑ reduces spin shake
LINK_MAX_LINEAR_VELOCITY = 300.0  # stage units/s (≈ 3 m/s at mpu=0.01)
LINK_MAX_ANGULAR_VELOCITY = 80.0  # rad/s

# Cable physics — SI / meter values matching ur10e_1x; convert lengths by mpu only.
CRYSTAL_HEAD_MASS_KG = 0.02  # MassAPI is always kilograms (not stage units)
# Soft-line damping is 1/time (not length); keep the meter-stage magnitude.
DEFORMABLE_LINEAR_DAMPING = 5000.0
DEFORMABLE_CONTACT_OFFSET_M = 0.00011
DEFORMABLE_REST_OFFSET_M = 0.00010

# PhysX revolute joints (arm + driven finger). Small friction resists drift.
JOINT_FRICTION = 0.05
JOINT_ARMATURE = 0.1  # motor inertia; ↑ helps numerical stability under gravity

# Smooth joint-space hover (120 Hz). Higher = slower / more stable.
HOVER_JOINT_STEPS = 120
HOVER_POS_TOLERANCE_M = 0.025
HOVER_SETTLE_FRAMES = 30
# Orient + tilt at the same tip (pads straddle ±X, then YZ tilt by side).
ORIENT_JOINT_STEPS = 120
ORIENT_POS_TOLERANCE_M = 0.025
ORIENT_SETTLE_FRAMES = 30
# Lula IK tolerances used by the motion controller (meters / radians).
HOVER_IK_POS_TOLERANCE_M = 0.005
HOVER_IK_ORI_TOLERANCE_RAD = 0.05

ROBOTIQ_CLOSED_RAD = float(np.deg2rad(80.0))  # firm full-close target (pads nearly meet)
ROBOTIQ_CONTACT_RAD = float(np.deg2rad(12.0))
GRASP_MIN_LIFT_M = 0.04
GRASP_CLOSE_WAIT_FRAMES = 240
GRASP_SQUEEZE_HOLD_FRAMES = 120
GRASP_RELEASE_WAIT_FRAMES = 90

# Tool tip offset from Lula ee_frame toward the pads (meters along tool +Z).
# Same magnitude as ur10e_1x; tips are further along +Z than knuckles.
TOOL_OFFSET_M = 0.16

PORT_PIN_A_NAME = "Copper_Pin_Component_1907"
PORT_PIN_B_NAME = "Copper_Pin_Component_1910"
PORT_APPROACH_X_OFFSET_M = 0.08  # +X standoff from mating (clear Mesh4680 / rack lip)
PORT_APPROACH_TOLERANCE_M = 0.04
# After lift: crystal insertion_axis (±Y) → port insertion_axis (−X).
# Short-way world-Z yaw: NegY −Y→−X ~−90° CW; PosY +Y→−X ~+90° CCW.
PORT_APPROACH_YAW_DEG = 90.0
PORT_APPROACH_YAW_STEPS = 6
# TipOffset final pose: wrist-tilt so crystal insertion_axis aligns to world −X
# (into the jack). Do **not** roll about that axis to flatten tool +Z — that
# twists the cable ~90° (latch keypoints swap Z-spread ↔ Y-spread).
PORT_OFFSET_ALIGN_AXIS_TO_X = True
PORT_OFFSET_FLATTEN_TOOL_Z = False
# Maneuver tool tilt from world −Z. Keep at grasp tilt so lift→offset is a
# pure world-Z yaw (−Y→−X); do not pitch the wrist flat (that wrecks Z ori).
MANEUVER_TOOL_TILT_FROM_DOWN_DEG = GRASP_TILT_FROM_DOWN_DEG
# Coarse TipLift→TipOffset poses (including t=0 lift). Motion uses the rest.
MANEUVER_SIMPLE_WAYPOINTS = 4
MANEUVER_SIMPLE_LIFT_TO_OFFSET = True
PORT_OFFSET_ALIGN_JOINT_STEPS = 180
PORT_OFFSET_ALIGN_MAX_FRAMES = 900
PORT_OFFSET_ALIGN_POS_TOLERANCE_M = 0.008
PORT_OFFSET_ALIGN_ORI_TOLERANCE_RAD = 0.08
# Tip-fixed ori blend for arrive→align (avoids one big joint-interp tip swing).
PORT_OFFSET_ORI_BLEND_STEPS = 12
PORT_OFFSET_ORI_BLEND_JOINT_STEPS = 40
# NegativeY: joint-space pan/lift bumps fight the clearance tip path — keep False
# so Lula cartesian tracks the waypoint tips (crystal follows markers).
PORT_NEGY_USE_BASE_SWEEP = False
PORT_NEGY_BASE_SWING_DEG = 90.0  # unused; short-way yaw comes from live axes
PORT_NEGY_BASE_WAYPOINTS = 8
PORT_NEGY_BASE_JOINT_STEPS = 160
# Mid-path clearance (rad): only used if PORT_NEGY_USE_BASE_SWEEP is re-enabled.
PORT_NEGY_CLEARANCE_SHOULDER_LIFT_DELTA = -0.45
PORT_NEGY_CLEARANCE_ELBOW_DELTA = 0.35
# TipLift → TipOffset samples (each gets its own orientation + axis marker).
PORT_APPROACH_TRANSLATE_STEPS = 8
PORT_APPROACH_TRANSLATE_Z_M = 0.03  # legacy sine bump; clearance uses VIA_Z below
# 1×-style extra Z on intermediate vias / rediscovery (meters).
PORT_APPROACH_VIA_Z_CLEARANCE_M = 0.04
PORT_MANEUVER_AXIS_LENGTH_M = 0.08
PORT_MANEUVER_AXIS_RADIUS_M = 0.003
# Plan-time USD AABB clearance (crystal/hand proxies vs switch + DataHall meshes).
MANEUVER_CRYSTAL_PROXY_HALF_EXTENTS_M = (0.040, 0.030, 0.035)
MANEUVER_HAND_PROXY_HALF_EXTENTS_M = (0.055, 0.055, 0.055)
MANEUVER_CHECK_HAND_PROXY = True
MANEUVER_CLEARANCE_MARGIN_M = 0.015
MANEUVER_CLEARANCE_SEGMENT_SAMPLES = 16
MANEUVER_OBSTACLE_MAX_EXTENT_M = 2.5  # skip hall-sized root AABBs
MANEUVER_DATAHALL_MESH_MAX_EXTENT_M = 1.5
MANEUVER_DATAHALL_MESH_MAX_COUNT = 80
MANEUVER_WORKSPACE_PAD_M = (0.35, 0.45, 0.35)
MANEUVER_VIA_FRACTIONS = (0.25, 0.50, 0.75)
MANEUVER_SEARCH_Z_CLEAR_M = (0.06, 0.10, 0.16, 0.24, 0.35, 0.45)
MANEUVER_SEARCH_X_RETRACT_M = (0.05, 0.10, 0.15, 0.22, 0.30, 0.40)
MANEUVER_SEARCH_Y_BLEND = (0.35, 0.55, 0.75, 0.90)
# Stretch-first path: TipLift → +X stretch (arm out) → align to port Y → TipOffset.
# Floor on +X beyond max(tip_lift.x, tip_offset.x) so the planner cannot pick a
# short lateral that leaves the elbow tucked.
MANEUVER_STRETCH_X_MIN_M = 0.25
MANEUVER_STRETCH_Y_BLEND = (0.25, 0.40, 0.55)  # Y progress during stretch via
# Final −X approach: duck under Mesh4679 / knuckles, then rise to tip_offset Z.
MANEUVER_APPROACH_Z_CLEAR_M = 0.015  # legacy high approach; dip path preferred
MANEUVER_APPROACH_Z_DIP_M = 0.08  # meters below tip_offset Z while sliding −X
MANEUVER_DIP_START_X_AHEAD_M = 0.10  # start duck this far +X of tip_offset
# TipOffset Z pad on copper-pin mid (meters). 0 = jack center Z.
PORT_OFFSET_Z_BIAS_M = 0.0
MANEUVER_OFFSET_ABOVE_X_RETRACT_M = 0.06
MANEUVER_POLYLINE_SAMPLES_PER_SEG = 5
# Replay verified cache tips; only rediscover when cache is missing/unverified.
MANEUVER_IGNORE_CACHED_TIPS = False
MANEUVER_REJECT_CACHED_IF_COLLIDES = True
# When verified cache ends short of the live tip_offset, append −X extension samples.
MANEUVER_EXTEND_CACHE_TO_LIVE_OFFSET = True
MANEUVER_OFFSET_EXTEND_STEP_M = 0.008
# Runtime PhysX contact logging during insert only (never aborts).
MANEUVER_ABORT_ON_DATAHALL_CONTACT = False
MANEUVER_CONTACT_COOLDOWN_S = 0.25
# After a rack hit, keep only tips with X ≥ this and push tip_offset further +X.
MANEUVER_CACHE_SAFE_TIP_X_M = 0.08
MANEUVER_CACHE_COLLISION_X_RETRACT_M = 0.10
MANEUVER_GOAL_CORRIDOR_RADIUS_M = 0.08
MANEUVER_GOAL_CORRIDOR_X_SLACK_M = 0.02
MANEUVER_CACHE_PATH = Path(__file__).resolve().parent / "maneuver_cache.json"
MANEUVER_CACHE_LOAD = True
MANEUVER_CACHE_SAVE = True
# How far past last cached progress to attempt each run (translate phase t∈[0,1]).
MANEUVER_CACHE_STEP = 0.10
HOME_CACHE_PATH = Path(__file__).resolve().parent / "home_cache.json"
HOME_CACHE_LOAD = True
HOME_CACHE_SAVE = True
PORT_FINAL_DESCENT_LINEAR_STEP_M = 0.002
PORT_LINEAR_IK_TOLERANCE_M = 0.002
PORT_INSERT_TOLERANCE_M = 0.035
LIFT_JOINT_STEPS = 240
LIFT_POS_TOLERANCE_M = 0.02
# Per yaw sample and TipOffset translate (joint-interp).
MANEUVER_YAW_JOINT_STEPS = 120
MANEUVER_OFFSET_JOINT_STEPS = 240
# NegY mid-maneuver: after densified waypoint N, swing shoulder_pan ~90° CW
# while Lula IK holds tip pose (avoids finger↔upper-arm self-collision).
MANEUVER_PAN_RECONFIG_ENABLE = True
MANEUVER_PAN_RECONFIG_AFTER_WP = 8  # 1-based index into motion_plan (skips t=0)
MANEUVER_PAN_RECONFIG_DELTA_DEG = -90.0  # CW about base +Z
MANEUVER_PAN_RECONFIG_SAMPLES = 6
MANEUVER_PAN_RECONFIG_JOINT_STEPS = 120
MANEUVER_PAN_RECONFIG_PAN_TOL_RAD = 0.35  # ~20° — reject IK that snaps pan back
# Straight-line tip tracking between clearance waypoints (meters / physics frame).
MANEUVER_LINEAR_STEP_M = 0.004
# Mid-path vias: loose enough to keep moving; final offset is tighter.
MANEUVER_VIA_POS_TOLERANCE_M = 0.06
MANEUVER_VIA_ORI_TOLERANCE_RAD = 0.35  # ~20°
MANEUVER_VIA_MAX_FRAMES = 900  # stretch segments are longer; avoid early timeout advance
MANEUVER_OFFSET_POS_TOLERANCE_M = 0.025
MANEUVER_OFFSET_ORI_TOLERANCE_RAD = 0.15  # ~9°
MANEUVER_OFFSET_MAX_FRAMES = 1600
# Postcondition: tip (and crystal mating) must be this close to TipOffset before
# the BT stamps at_port_offset / starts align→translate.
MANEUVER_AT_OFFSET_TIP_TOL_M = 0.04
MANEUVER_AT_OFFSET_CRYSTAL_TOL_M = 0.05
# From this 1-based motion_plan index onward, use precision tols so TipOffset is reached.
MANEUVER_LATE_WP_INDEX = 40
MANEUVER_LATE_POS_TOLERANCE_M = 0.008
MANEUVER_LATE_ORI_TOLERANCE_RAD = 0.08  # ~4.5°
MANEUVER_LATE_MAX_FRAMES = 1200


def port_approach_yaw_deg(side: str) -> float:
    """Legacy short-way yaw (±90°). Prefer live-axis yaw in the maneuver primitive."""

    mag = abs(float(PORT_APPROACH_YAW_DEG))
    if _normalize_side_key(side) == "negative":
        return -mag  # CW: −Y → −X
    return mag  # CCW: +Y → −X

CABLE_IN_GRIPPER_MAX_ERR_M = 0.06
CABLE_HOLD_CHECK_EVERY_N_FRAMES = 5

FINGERTIP_NAME_TOKENS = (
    "left_inner_finger",
    "right_inner_finger",
    "pad",
    "fingertip",
)

# Low-friction slide materials for bezels / switch faces / trailing head39 so the
# cable can graze and seat into the jack without snagging. Collision stays on;
# combine=min so pairs resolve to the lower μ even when head45 uses combine=max.
BEZEL_FRICTION_STATIC = 0.01
BEZEL_FRICTION_DYNAMIC = 0.005
BEZEL_FRICTION_COMBINE_MODE = "min"
DGX_BEZEL_PATH_TOKENS = ("bezel", "dgx3_bezel")
# Also bind slide μ on switch / port geometry the crystal contacts during insert.
INSERT_SLIDE_PATH_TOKENS = (
    "bezel",
    "dgx3_bezel",
    "frontplate",
    "chassis",
    "rj45",
    "networkconnector",
    "qm8700",
    "as4610",
    "ethernet",
    "jack",
)
TRAILING_HEAD_SLIDE_FRICTION = True

# Insert-diag showed trailing head39 grazing WorkTable1 — make tables as
# frictionless as possible (combine=min wins vs high-μ crystal/grasp materials).
WORKTABLE_FRICTION_STATIC = 0.0
WORKTABLE_FRICTION_DYNAMIC = 0.0
WORKTABLE_FRICTION_COMBINE_MODE = "min"
WORKTABLE_FRICTIONLESS = True

# Align + translate: match crystal↔port mating YZ via features, then advance
# along the crystal insertion axis; repeat. Cache successful tip poses.
INSERT_CACHE_PATH = Path(__file__).resolve().parent / "insert_cache.json"
INSERT_CACHE_LOAD = True
INSERT_CACHE_SAVE = True
LATCH_Z_MARGIN_M = 0.0005
MATING_SIDE_MARGIN_M = 0.00015
MATING_CENTER_YZ_TOL_M = 0.0015
LATCH_Y_MARGIN_M = 0.00015
AXIS_DOT_MIN = 0.98
INSERT_STEP_M = 0.01  # 1 cm along crystal insertion axis per translate
MATING_TOUCH_GAP_M = 0.002
ALIGN_INSERT_MAX_FRAMES = 6000
# Temporary: TipOffset YZ is trusted; insert is axis-only translates (align kept
# in code but skipped). Re-enable ALIGN_INSERT_ENABLE_YZ when ready.
ALIGN_INSERT_ENABLE_YZ = False
# Insert-step diagnostics: wrist wrench + contact impulses + friction props → JSONL.
INSERT_DIAG_ENABLE = True
INSERT_DIAG_PATH = Path(__file__).resolve().parent / "insert_diag.jsonl"
# Sample/print only when a translate IK target is reached (not every sim frame).
INSERT_DIAG_SAMPLE_EVERY_N = 0  # 0 = pose-arrival only (see tick_align_and_insert)
INSERT_DIAG_LOG_EVERY_N = 1  # print each recorded pose sample
INSERT_DIAG_FORCE_LOG_N = 5.0  # always print when |F_wrist| exceeds this
INSERT_DIAG_IMPULSE_LOG = 0.01  # always print when contact impulse exceeds this
INSERT_DIAG_CONTACT_LOG = True  # log cable↔any-mesh contacts during insert (no abort)
# Skip gripper/robot contacts so pad pinch does not flood the log. Set False to
# include those too. Ethernet/RJ45 and all other scene meshes are logged.
INSERT_CONTACT_SKIP_ROBOT = True
# Legacy flags (unused — port contacts are logged like any other mesh).
INSERT_CONTACT_IGNORE_PORT = False
INSERT_CONTACT_IGNORE_TOKENS: tuple[str, ...] = ()
# At TipOffset: require Lula IK for tip that places crystal mating at port mating
# before align→translate. If unreachable, swing shoulder_pan at fixed offset tip.
INSERT_REACH_CHECK = True
INSERT_REACH_RECONFIG_DELTAS_DEG = (-90.0, 90.0, -45.0, 45.0, -135.0, 135.0, -180.0)
INSERT_REACH_RECONFIG_SAMPLES = 6
INSERT_REACH_RECONFIG_JOINT_STEPS = 120
INSERT_REACH_RECONFIG_PAN_TOL_RAD = 0.35  # ~20°
# One-shot YZ close (full mating-center ΔYZ, capped). Old 2 mm clamp never finished.
ALIGN_YZ_MAX_M = 0.025
ALIGN_NUDGE_POS_MAX_M = 0.002  # legacy; align-yz uses ALIGN_YZ_MAX_M
ALIGN_NUDGE_ROT_MAX_RAD = 0.02
ALIGN_ROT_AFTER_YZ_ERR_M = 0.004
# Strict per-iteration motion (align / translate).
ALIGN_STEP_JOINT_STEPS = 90
ALIGN_STEP_MAX_FRAMES = 360
ALIGN_STEP_POS_TOLERANCE_M = 0.003
ALIGN_STEP_ORI_TOLERANCE_RAD = 0.05
ALIGN_STEP_LINEAR_STEP_M = 0.002
ALIGN_LOG_EVERY_N = 1
# Single next-target insert marker (diameter = crystal mating-plane hypotenuse).
ALIGN_INSERT_DEBUG_MARKERS_VISIBLE = False
ALIGN_INSERT_AXIS_LENGTH_M = 0.03
ALIGN_INSERT_AXIS_RADIUS_M = 0.00125

# Debug spheres: radius in meters (converted to stage units at spawn).
PORT_DEBUG_MARKER_SCALE_M = 0.01
PORT_MANEUVER_MARKER_SCALE_M = 0.05
OBSERVE_DEBUG_MARKER_SCALE_M = 0.02
# Hidden by default; toggle ``/World/DebugPortMarkers/<station>`` in Isaac.
DEBUG_MARKER_VISIBLE_DEFAULT = False


def debug_marker_names() -> tuple[str, ...]:
    """Prim names under each station's debug-marker xform."""

    return (
        "Cable",
        "TipHover",
        "HandHover",
        "TipGrasp",
        "HandGrasp",
        "TipLift",
        "TipYaw",
        "TipOffset",
    )
