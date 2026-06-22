# Core Isaac Sim App Initialization
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
import builtins

import carb
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
from isaacsim.core.utils.stage import add_reference_to_stage
try:
    from isaacsim.core.utils.viewports import set_camera_view
except Exception:
    set_camera_view = None
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from collision_setup import (
    apply_datahall_scale,
    enable_articulation_collisions,
    enable_static_collisions,
)
from franka_motion_controller import FrankaMotionController
from port_frame import PortFrame
from qsfp_module import (
    QSFP_GRASP_OFFSET_TO_TOP_M,
    QSFP_LENGTH_M,
    create_qsfp_module,
    grasp_tool_offset,
    gripper_closed_positions,
    pick_grasp_block_z,
)


# =============================================================================
# STABLE MULTI-QSFP INSERT
# =============================================================================
#
# Stable path strategy:
#   1. Pick a QSFP module from the table.
#   2. Measure the hand-to-module offset after grasp.
#   3. Move to a far slide lane away from the ports.
#   4. Match the final align X/Z first.
#   5. Slide horizontally in Y at the far offset.
#   6. Advance straight to the final align standoff.
#   7. Use the insert servo to crawl along the true port insertion axis.
#
# Keep this file as the baseline. Make experimental copies instead of editing this
# motion logic directly.


# =============================================================================
# CONFIG
# =============================================================================

DEBUG = False
VIEWPORT_CAMERA = True
VERBOSE_LOGS = False


def log(*args, **kwargs):
    if VERBOSE_LOGS:
        builtins.print(*args, **kwargs)


DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)
DATAHALL_SCALE = 2.0

# Sequential insert jobs.
INSERT_JOBS = [
    {
        "label": "port_0",
        "port_prim_path": (
            "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
            "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
            "pcb003636_idf_01/Connector_Quad_04/Connector_Pair_04/"
            "QSFP_DD_Connector_A_02/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
        ),
        "lateral_offset": np.array([0.0, 0.0, -0.02], dtype=np.float64),
        "pick_xy": np.array([0.30, -0.525], dtype=np.float64),
        "module_prim_path": "/World/QSFP_Module_0",
        "module_name": "qsfp_module_0",
    },
    {
        "label": "port_1",
        "port_prim_path": (
            "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
            "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
            "pcb003636_idf_01/Connector_Quad_04/Connector_Pair_01/"
            "QSFP_DD_Connector_A_01/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
        ),
        # Tune only if this port is visibly off.
        "lateral_offset": np.array([0.0, 0.0, -0.009], dtype=np.float64),
        "pick_xy": np.array([0.38, -0.525], dtype=np.float64),
        "module_prim_path": "/World/QSFP_Module_1",
        "module_name": "qsfp_module_1",
    },
    {
        "label": "port_2",
        "port_prim_path": (
            "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
            "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
            "pcb003636_idf_01/Connector_Quad_03/Connector_Pair_01/"
            "QSFP_DD_Connector_A_01/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
        ),
        # Tune only if this port is visibly off.
        "lateral_offset": np.array([0.0, 0.0, -0.009], dtype=np.float64),
        "pick_xy": np.array([0.46, -0.525], dtype=np.float64),
        "module_prim_path": "/World/QSFP_Module_2",
        "module_name": "qsfp_module_2",
    },    
    {
        "label": "port_3",
        "port_prim_path": (
            "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
            "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
            "pcb003636_idf_01/Connector_Quad_03/Connector_Pair_03/"
            "QSFP_DD_Connector_A_02/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
        ),
        # Tune only if this port is visibly off.
        "lateral_offset": np.array([0.0, 0.0, -0.02], dtype=np.float64),
        "pick_xy": np.array([0.54, -0.525], dtype=np.float64),
        "module_prim_path": "/World/QSFP_Module_3",
        "module_name": "qsfp_module_3",
    },   
]

# This local +Z convention comes from the old setup.
INSERT_LOCAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# DataHall is scaled 2x, so the ports move upward.
# Raise the table/robot base so the Franka keeps roughly the same vertical reach relationship.
TABLE_HEIGHT = 3
TABLE_THICKNESS = 0.05
ROBOT_BASE_POS = np.array([0.55, -0.15, TABLE_HEIGHT], dtype=np.float64)

# Passive second robot / block row for the two-robot workspace layout test.
# Robot 2 is spawned now, but it is not driven by the current state machine yet.
SECOND_ROBOT_BASE_POS = np.array([0.55, -1.20, TABLE_HEIGHT], dtype=np.float64)
SECOND_PICK_ROW_Y = -0.825
SECOND_ROBOT_BLOCK_COUNT = 3
SECOND_ROBOT_PRIM_PATH = "/World/Franka_2"

# Robot 2 target ports. Robot 2 is still passive in this layout script,
# but these are the jobs/ports it will use when the second state machine is added.
SECOND_INSERT_JOBS = [
    {
        "label": "robot2_port_0",
        "port_prim_path": (
            "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
            "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
            "pcb003636_idf_01/Connector_Quad_02/Connector_Pair_04/"
            "QSFP_DD_Connector_A_02/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
        ),
        "lateral_offset": np.array([0.0, 0.0, -0.02], dtype=np.float64),
        "pick_xy": np.array([0.30, SECOND_PICK_ROW_Y], dtype=np.float64),
        "module_prim_path": "/World/QSFP_Robot2_Module_0",
        "module_name": "qsfp_robot2_module_0",
    },
    {
        "label": "robot2_port_1",
        "port_prim_path": (
            "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
            "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
            "pcb003636_idf_01/Connector_Quad_02/Connector_Pair_01/"
            "QSFP_DD_Connector_A_01/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
        ),
        "lateral_offset": np.array([0.0, 0.0, -0.009], dtype=np.float64),
        "pick_xy": np.array([0.38, SECOND_PICK_ROW_Y], dtype=np.float64),
        "module_prim_path": "/World/QSFP_Robot2_Module_1",
        "module_name": "qsfp_robot2_module_1",
    },
    {
        "label": "robot2_port_2",
        "port_prim_path": (
            "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
            "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
            "pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/"
            "QSFP_DD_Connector_A_02/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
        ),
        "lateral_offset": np.array([0.0, 0.0, -0.02], dtype=np.float64),
        "pick_xy": np.array([0.46, SECOND_PICK_ROW_Y], dtype=np.float64),
        "module_prim_path": "/World/QSFP_Robot2_Module_2",
        "module_name": "qsfp_robot2_module_2",
    },
    {
        "label": "robot2_port_3",
        "port_prim_path": (
            "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
            "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
            "pcb003636_idf_01/Connector_Quad_01/Connector_Pair_04/"
            "QSFP_DD_Connector_A_02/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
        ),
        "lateral_offset": np.array([0.0, 0.0, -0.02], dtype=np.float64),
        "pick_xy": np.array([0.54, SECOND_PICK_ROW_Y], dtype=np.float64),
        "module_prim_path": "/World/QSFP_Robot2_Module_3",
        "module_name": "qsfp_robot2_module_3",
    },
]

# Runtime-selected job list. The same motion functions are reused by swapping
# this to INSERT_JOBS or SECOND_INSERT_JOBS through the robot runtime context.
active_jobs = INSERT_JOBS
active_robot_label = "robot1"

# Start robot 2 shortly after robot 1 so both are active in the same run,
# but their first pickups do not start on the exact same frame. Set to 0 if
# you want both robots to begin immediately.
ROBOT_2_START_DELAY_FRAMES = 0

PICK_XY = INSERT_JOBS[0]["pick_xy"].copy()
PICK_SURFACE_Z = TABLE_HEIGHT + QSFP_LENGTH_M / 2.0

PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 30
POST_RESET_WARMUP_FRAMES = 20

TOOL_OFFSET = grasp_tool_offset()

# Pick sequence. Same idea as old QSFP job builder.
PICK_GRASP_OFFSET_TO_TOP_M = QSFP_GRASP_OFFSET_TO_TOP_M
PICK_HOVER_CLEARANCE = 0.30
PICK_GRASP_CLEARANCE = 0.015
PICK_POS_TOL = 0.003

# Port-side staging.
# ONLY TUNE THIS:
# Distance from the port face before insertion starts.
# Bigger = line up farther away from the DataHall/port.
# Smaller = line up closer to the port.
LINEUP_PORT_OFFSET_M = 0.2

# Separate knob for the horizontal slide distance from the port.
# Bigger = the robot does the sideways/horizontal Y slide farther away from the ports.
# After the Y slide, it moves straight forward to LINEUP_PORT_OFFSET_M.
HORIZONTAL_SLIDE_PORT_OFFSET_M = 0.35

# Derived values used by the controller.
ALIGN_STANDOFF = LINEUP_PORT_OFFSET_M
PRE_ALIGN_LINEAR_STEP = 0.0025

# Table / carry-height clearance.
# ONLY TUNE THIS if the held block hits the table:
# Bigger = robot carries the block higher above the table during transit.
# Smaller = lower/faster, but more likely to hit the table.
CARRY_HEIGHT_ABOVE_TABLE_M = 0.65

TABLE_SAFE_MODULE_CENTER_Z = TABLE_HEIGHT + CARRY_HEIGHT_ABOVE_TABLE_M
TABLE_CLEAR_LINEAR_STEP = 0.003
HIGH_TRANSIT_LINEAR_STEP = 0.004

# Final insertion.
# Slightly deeper than v1. If it over-inserts, bring this back to 0.055 or 0.048.
INSERT_TIP_DEPTH_M = 0.060
# Do not waste 18+ seconds trying to remove a visually irrelevant 1-3 mm residual.
# The port mesh/proxy/origin are not calibrated tightly enough for a 1.5 mm gate.
INSERT_ALIGN_LATERAL_TOL = 0.0030
INSERT_ALIGN_HOLD_FRAMES = 3
INSERT_ALIGN_MAX_FRAMES = 240
INSERT_ALIGN_PROCEED_TOL = 0.0040
INSERT_STROKE_STEP = 0.0020
# Allow more time because the physical module moves slower than the commanded crawl near contact.
INSERT_STROKE_MAX_FRAMES = 300
# Let the commanded target get ahead of the measured module center. Without this,
# the servo only asks for 1 mm past the current pose, so once contact/friction
# resists motion it stalls instead of pushing through.
INSERT_COMMAND_LEAD_LIMIT = 0.020
INSERT_SETTLE_FRAMES = 2
INSERT_IK_POS_TOL = 0.0008
INSERT_IK_ORI_TOL = 0.01
INSERT_MAX_IK_FAILS = 60

# Keep the run open when done.
HOLD_FOR_INSPECTION = True

# Release the module after the insert servo stops.
RELEASE_AFTER_INSERT = True
RELEASE_GRIPPER_FRAMES = 70

# Practical seat criteria.
# Stop when the leading tip is past the port face and lateral error is acceptable.
SEAT_SUCCESS_TIP_AXIAL_M = 0.005
SEAT_SUCCESS_LATERAL_TOL_M = 0.003
SEAT_SUCCESS_HOLD_FRAMES = 8

# After release, pull the open gripper straight back out along the port axis.
RETREAT_AFTER_RELEASE = True
RETREAT_DISTANCE_M = 0.125

# Extra straight pullback after insertion/release.
# Increase this if the gripper still feels too close to the seated module/port.
# Example: 0.025 = 2.5 cm extra, 0.050 = 5 cm extra.
EXTRA_RETREAT_AFTER_INSERT_M = 0.125

RETREAT_LINEAR_STEP = 0.002
RETREAT_MAX_FRAMES = 260


PHASE_WAITING = "waiting"
PHASE_WARMUP = "warmup"
PHASE_PICK = "pick"
PHASE_TRANSIT = "transit"
PHASE_INSERT_SERVO = "insert_servo"
PHASE_RELEASE = "release"
PHASE_RETREAT = "retreat"
PHASE_DONE = "done"


def sep(char="-", width=72):
    return char * width


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("Cannot normalize zero-length vector")
    return np.asarray(v, dtype=np.float64) / n


def quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    return quats_to_rot_matrices(np.asarray(q_wxyz, dtype=np.float64).reshape(1, 4))[0]


def get_hand_pose():
    pos, rot = art_kinematics.compute_end_effector_pose()
    pos = np.asarray(pos, dtype=np.float64).flatten()
    rot = np.asarray(rot, dtype=np.float64)
    if rot.ndim == 3:
        rot = rot[0]
    return pos, rot


def get_module_pose():
    pos, quat = module.get_world_pose()
    pos = np.asarray(pos, dtype=np.float64).flatten()
    quat = np.asarray(quat, dtype=np.float64).flatten()
    return pos, quat


def hand_target_for_module_center(module_center: np.ndarray, orientation_wxyz: np.ndarray, offset_local: np.ndarray) -> np.ndarray:
    rot = quat_to_rot(orientation_wxyz)
    return np.asarray(module_center, dtype=np.float64) - rot @ np.asarray(offset_local, dtype=np.float64)


def compute_grasp_offset_local():
    module_pos, _ = get_module_pose()
    hand_pos, hand_rot = get_hand_pose()
    offset_world = module_pos - hand_pos
    offset_local = hand_rot.T @ offset_world

    log("\n" + sep())
    log("[GRASP OFFSET - MEASURED]")
    log(f"  module_world:  {np.round(module_pos, 5)}")
    log(f"  hand_world:    {np.round(hand_pos, 5)}")
    log(f"  offset_world:  {np.round(offset_world, 5)}")
    log(f"  offset_local:  {np.round(offset_local, 5)}")
    log(f"  magnitude_mm:  {np.linalg.norm(offset_local) * 1000.0:.2f}")
    log(sep())

    return offset_local


def check_grasp():
    fingers = my_franka.gripper.get_joint_positions()
    module_pos, _ = get_module_pose()
    if fingers is None:
        log("[GRASP CHECK] Cannot read finger positions.")
        return False
    fingers = np.asarray(fingers, dtype=np.float64).flatten()

    # QSFP module is only ~4 mm wide. If both fingers are fully closed at 1 mm,
    # the grasp probably missed. This check is intentionally loose.
    total_gap_mm = float(np.sum(fingers) * 1000.0)
    ok = total_gap_mm > 2.2

    log("\n" + sep())
    log("[GRASP CHECK]")
    log(f"  fingers_mm:      {np.round(fingers * 1000.0, 3)}")
    log(f"  total_gap_mm:    {total_gap_mm:.3f}")
    log(f"  module_center:   {np.round(module_pos, 5)}")
    log("  RESULT:          " + ("OK - contact likely" if ok else "BAD - likely missed / too closed"))
    log(sep())
    return ok


def build_work_table(world):
    margin = 0.35
    table_points = [ROBOT_BASE_POS[:2], SECOND_ROBOT_BASE_POS[:2]]
    table_points.extend(job["pick_xy"] for job in INSERT_JOBS)
    table_points.extend(job["pick_xy"] for job in SECOND_INSERT_JOBS)
    table_points = np.asarray(table_points, dtype=np.float64)

    min_xy = np.min(table_points, axis=0)
    max_xy = np.max(table_points, axis=0)
    center_xy = (min_xy + max_xy) / 2.0
    half_xy = (max_xy - min_xy) / 2.0 + margin

    position = np.array([center_xy[0], center_xy[1], TABLE_HEIGHT - TABLE_THICKNESS / 2.0], dtype=np.float64)
    scale = np.array([2.0 * half_xy[0], 2.0 * half_xy[1], TABLE_THICKNESS], dtype=np.float64)

    world.scene.add(
        FixedCuboid(
            name="work_table",
            prim_path="/World/WorkTable",
            position=position,
            scale=scale,
            size=1.0,
            color=np.array([0.55, 0.35, 0.18]),
            visible=True,
        )
    )
    log(f"[TABLE] center={np.round(position, 4)} scale={np.round(scale, 4)} top_z={TABLE_HEIGHT}")
    log(f"[TABLE] covers x={np.round([min_xy[0], max_xy[0]], 4)} y={np.round([min_xy[1], max_xy[1]], 4)} plus margin={margin}")


def build_port_frame(job_index, jobs=None, robot_position=None, robot_label="robot"):
    if jobs is None:
        jobs = active_jobs
    if robot_position is None:
        robot_position = ROBOT_BASE_POS

    job = jobs[job_index]
    prim_path = job["port_prim_path"]
    lateral_offset = job["lateral_offset"]

    if not stage.GetPrimAtPath(prim_path).IsValid():
        raise RuntimeError(f"Insert port prim not found for {robot_label} job {job_index}: {prim_path}")

    port = PortFrame.from_prim_path(
        prim_path,
        local_insert_axis=INSERT_LOCAL_AXIS,
        lateral_offset=lateral_offset,
        robot_position=robot_position,
    )
    if port is None:
        raise RuntimeError(f"Could not build PortFrame for {robot_label} job {job_index}: {prim_path}")

    log("\n" + sep("="))
    log(f"[PORT FRAME {robot_label} job={job_index}] {job['label']}")
    log(f"  prim:          {prim_path}")
    log(f"  insert_origin: {np.round(port.insert_origin, 5)}")
    log(f"  insert_axis:   {np.round(port.insert_axis, 5)}")
    log(f"  insert_rot:    {np.round(port.insert_rot, 5)}")
    log(f"  lateral_offset:{np.round(lateral_offset, 5)}")
    log(sep("="))
    return port


def set_active_job(job_index):
    global active_job_index, port_frame, module, PICK_XY, block_offset_local

    active_job_index = int(job_index)
    port_frame = port_frames[active_job_index]
    PICK_XY = active_jobs[active_job_index]["pick_xy"].copy()
    module = modules[active_job_index]

    block_offset_local = None

    log("\n" + sep("#"))
    log(f"[ACTIVE JOB {active_robot_label}] {active_job_index + 1}/{len(active_jobs)}: {active_jobs[active_job_index]['label']}")
    log(f"  pick_xy:     {np.round(PICK_XY, 5)}")
    log(f"  module_path: {active_jobs[active_job_index]['module_prim_path']}")
    log(f"  port_origin: {np.round(port_frame.insert_origin, 5)}")
    log(sep("#"))


def start_next_job_or_finish():
    next_index = active_job_index + 1
    if next_index >= len(active_jobs):
        log("\n" + sep("="))
        log("[ALL JOBS COMPLETE]")
        log(f"  completed_jobs: {len(active_jobs)}")
        log(sep("="))
        return False

    set_active_job(next_index)
    controller.clear_queue()
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    queue_pick_phase()
    return True


def apply_startup_camera_view():
    if not VIEWPORT_CAMERA or set_camera_view is None:
        return
    eye = [2.65, 2.15, 4.60]
    target = [-0.35, -0.45, 2.55]
    try:
        set_camera_view(eye=eye, target=target)
    except Exception as exc:
        carb.log_warn(f"Could not set startup camera: {exc}")


def queue_pick_phase():
    controller.clear_queue()

    grasp_center_z = pick_grasp_block_z(
        PICK_SURFACE_Z,
        offset_to_top_m=PICK_GRASP_OFFSET_TO_TOP_M,
    )
    pick_hover_z = grasp_center_z + PICK_HOVER_CLEARANCE
    pick_grasp_z = grasp_center_z + PICK_GRASP_CLEARANCE
    pick_lift_z = pick_hover_z

    pick_hover = np.array([PICK_XY[0], PICK_XY[1], pick_hover_z], dtype=np.float64)
    pick_grasp = np.array([PICK_XY[0], PICK_XY[1], pick_grasp_z], dtype=np.float64)
    pick_lift = np.array([PICK_XY[0], PICK_XY[1], pick_lift_z], dtype=np.float64)

    log("\n" + sep("="))
    log(f"[QUEUE PICK {active_robot_label} job={active_job_index}] {active_jobs[active_job_index]['label']}")
    log(f"  pick_hover: {np.round(pick_hover, 5)}")
    log(f"  pick_grasp: {np.round(pick_grasp, 5)}")
    log(f"  pick_lift:  {np.round(pick_lift, 5)}")
    log(f"  down_ori:   {np.round(port_frame.pick_down_rot, 5)}")
    log(sep("="))

    controller.add_cartesian_waypoint(
        position=pick_hover,
        orientation=port_frame.pick_down_rot,
        pos_tolerance=0.05,
        max_frames=260,
        joint_interp=True,
        joint_steps=180,
        label="pick_hover",
    )
    controller.add_cartesian_waypoint(
        position=pick_grasp,
        orientation=port_frame.pick_down_rot,
        pos_tolerance=PICK_POS_TOL,
        max_frames=260,
        joint_interp=True,
        joint_steps=140,
        label="pick_grasp",
    )
    controller.add_gripper_command(action="close", wait_frames=70)
    controller.add_cartesian_waypoint(
        position=pick_lift,
        orientation=port_frame.pick_down_rot,
        pos_tolerance=0.02,
        max_frames=260,
        joint_interp=True,
        joint_steps=180,
        hold_gripper=True,
        label="pick_lift",
    )


def queue_transit_phase(offset_local):
    controller.clear_queue()

    module_pos, _ = get_module_pose()
    insert_ori = port_frame.insert_rot

    # Stable path:
    # The horizontal slide happens away from the ports using
    # HORIZONTAL_SLIDE_PORT_OFFSET_M.
    #
    # Path:
    #   1. lift safely
    #   2. move to the far slide lane's X/Z while staying at pickup-side Y
    #   3. drop to the far slide lane Z
    #   4. slide horizontally in Y at the FAR offset from the ports
    #   5. move straight forward along the port axis to the real align point
    #   6. start normal insert servo
    safe_lift_center = module_pos.copy()
    safe_lift_center[2] = max(
        safe_lift_center[2],
        PICK_SURFACE_Z + PICK_HOVER_CLEARANCE,
        TABLE_SAFE_MODULE_CENTER_Z,
    )

    slide_center = port_frame.approach_position(HORIZONTAL_SLIDE_PORT_OFFSET_M)
    align_center = port_frame.approach_position(ALIGN_STANDOFF)

    # Same X/Z as the far slide lane, but stay at pickup-side Y first.
    xz_lineup_center = np.array(
        [slide_center[0], module_pos[1], slide_center[2]],
        dtype=np.float64,
    )

    high_xz_lineup_center = xz_lineup_center.copy()
    high_xz_lineup_center[2] = max(high_xz_lineup_center[2], TABLE_SAFE_MODULE_CENTER_Z)

    safe_lift_hand = hand_target_for_module_center(safe_lift_center, insert_ori, offset_local)
    high_xz_lineup_hand = hand_target_for_module_center(high_xz_lineup_center, insert_ori, offset_local)
    xz_lineup_hand = hand_target_for_module_center(xz_lineup_center, insert_ori, offset_local)
    slide_hand = hand_target_for_module_center(slide_center, insert_ori, offset_local)
    align_hand = hand_target_for_module_center(align_center, insert_ori, offset_local)

    log("\n" + sep("="))
    log(f"[QUEUE TRANSIT {active_robot_label} job={active_job_index}] {active_jobs[active_job_index]['label']}")
    log("  mode: far X/Z lineup, far horizontal Y slide, then straight advance to align")
    log(f"  table_top_z:                  {TABLE_HEIGHT:.5f}")
    log(f"  carry_height_above_table:     {CARRY_HEIGHT_ABOVE_TABLE_M:.5f}")
    log(f"  safe_module_center_z:         {TABLE_SAFE_MODULE_CENTER_Z:.5f}")
    log(f"  horizontal_slide_port_offset: {HORIZONTAL_SLIDE_PORT_OFFSET_M:.5f}")
    log(f"  final_lineup_port_offset:     {LINEUP_PORT_OFFSET_M:.5f}")
    log(f"  module_now:                   {np.round(module_pos, 5)}")
    log(f"  safe_lift_center:             {np.round(safe_lift_center, 5)}")
    log(f"  high_xz_lineup_center:        {np.round(high_xz_lineup_center, 5)}")
    log(f"  xz_lineup_center:             {np.round(xz_lineup_center, 5)}")
    log(f"  slide_center:                 {np.round(slide_center, 5)}")
    log(f"  align_center:                 {np.round(align_center, 5)}")
    log(f"  y_slide_distance_mm:          {(slide_center[1] - xz_lineup_center[1]) * 1000.0:.1f}")
    log(f"  straight_advance_mm:          {abs(HORIZONTAL_SLIDE_PORT_OFFSET_M - LINEUP_PORT_OFFSET_M) * 1000.0:.1f}")
    log(f"  safe_lift_hand:               {np.round(safe_lift_hand, 5)}")
    log(f"  high_xz_lineup_hand:          {np.round(high_xz_lineup_hand, 5)}")
    log(f"  xz_lineup_hand:               {np.round(xz_lineup_hand, 5)}")
    log(f"  slide_hand:                   {np.round(slide_hand, 5)}")
    log(f"  align_hand:                   {np.round(align_hand, 5)}")
    log(sep("="))

    controller.add_cartesian_waypoint(
        position=safe_lift_hand,
        orientation=insert_ori,
        target_is_hand=True,
        linear=True,
        linear_step=HIGH_TRANSIT_LINEAR_STEP,
        max_frames=520,
        pos_tolerance=0.006,
        hold_gripper=True,
        label="linear_lift_to_safe_carry_height",
    )

    controller.add_cartesian_waypoint(
        position=high_xz_lineup_hand,
        orientation=insert_ori,
        target_is_hand=True,
        linear=True,
        linear_step=HIGH_TRANSIT_LINEAR_STEP,
        max_frames=760,
        pos_tolerance=0.008,
        hold_gripper=True,
        label="linear_high_move_to_far_slide_x_pick_y",
    )

    controller.add_cartesian_waypoint(
        position=xz_lineup_hand,
        orientation=insert_ori,
        target_is_hand=True,
        linear=True,
        linear_step=TABLE_CLEAR_LINEAR_STEP,
        max_frames=500,
        pos_tolerance=0.004,
        hold_gripper=True,
        label="linear_drop_to_far_slide_xz_pick_y",
    )

    # Horizontal Y slide happens here, far from the ports.
    controller.add_cartesian_waypoint(
        position=slide_hand,
        orientation=insert_ori,
        target_is_hand=True,
        linear=True,
        linear_step=PRE_ALIGN_LINEAR_STEP,
        max_frames=900,
        pos_tolerance=0.004,
        hold_gripper=True,
        label="linear_far_horizontal_y_slide",
    )

    # Then move straight forward/back along the port insertion axis to the final align standoff.
    controller.add_cartesian_waypoint(
        position=align_hand,
        orientation=insert_ori,
        target_is_hand=True,
        linear=True,
        linear_step=PRE_ALIGN_LINEAR_STEP,
        max_frames=500,
        pos_tolerance=0.003,
        hold_gripper=True,
        label="linear_straight_advance_to_final_align",
    )

def queue_release_phase():
    controller.clear_queue()
    log("\n" + sep("="))
    log("[QUEUE RELEASE]")
    log("  Opening gripper in-place after insertion.")
    log("  Retreat will run after the fingers open.")
    log(sep("="))
    controller.add_gripper_command(action="open", wait_frames=RELEASE_GRIPPER_FRAMES)


def queue_retreat_phase():
    controller.clear_queue()

    hand_pos, _ = get_hand_pose()

    # port_frame.insert_axis points INTO the port.
    # Retreat moves opposite that axis, back toward the robot.
    retreat_dir = -normalize(port_frame.insert_axis)
    total_retreat_distance = RETREAT_DISTANCE_M + EXTRA_RETREAT_AFTER_INSERT_M
    retreat_hand = hand_pos + retreat_dir * total_retreat_distance

    log("\n" + sep("="))
    log("[QUEUE RETREAT]")
    log("  Pulling open gripper straight back along the port axis.")
    log(f"  hand_start:       {np.round(hand_pos, 5)}")
    log(f"  retreat_dir:      {np.round(retreat_dir, 5)}")
    log(f"  retreat_hand:     {np.round(retreat_hand, 5)}")
    log(f"  base_distance_mm: {RETREAT_DISTANCE_M * 1000.0:.1f}")
    log(f"  extra_distance_mm:{EXTRA_RETREAT_AFTER_INSERT_M * 1000.0:.1f}")
    log(f"  total_distance_mm:{total_retreat_distance * 1000.0:.1f}")
    log(sep("="))

    controller.add_cartesian_waypoint(
        position=retreat_hand,
        orientation=port_frame.insert_rot,
        target_is_hand=True,
        linear=True,
        linear_step=RETREAT_LINEAR_STEP,
        max_frames=RETREAT_MAX_FRAMES,
        pos_tolerance=0.006,
        label="post_release_straight_retreat",
    )


def default_insert_servo_state():
    return {
        "mode": "align",
        "frames": 0,
        "hold_frames": 0,
        "stroke_frames": 0,
        "ik_fail_count": 0,
        "target_center": None,
        "target_axial": None,
        "origin": None,
        "axis": None,
        "locked_lateral": None,
        "commanded_axial": None,
        "seat_hold_frames": 0,
    }


insert_servo = default_insert_servo_state()


def lateral_vec(position, origin, axis):
    delta = np.asarray(position, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    axis = normalize(axis)
    return delta - axis * float(np.dot(delta, axis))


def axial_coord(position, origin, axis):
    return float(np.dot(np.asarray(position, dtype=np.float64) - np.asarray(origin, dtype=np.float64), normalize(axis)))


def pos_from_axial_lateral(origin, axis, axial, lateral):
    return np.asarray(origin, dtype=np.float64) + normalize(axis) * float(axial) + np.asarray(lateral, dtype=np.float64)


def init_insert_servo():
    module_pos, _ = get_module_pose()
    module_half = QSFP_LENGTH_M / 2.0
    insert_center_goal = port_frame.center_goal_for_tip_depth(
        INSERT_TIP_DEPTH_M,
        module_half,
        module_orientation_wxyz=port_frame.insert_rot,
    )

    origin = np.asarray(port_frame.insert_origin, dtype=np.float64)
    axis = normalize(port_frame.insert_axis)
    target_axial = axial_coord(insert_center_goal, origin, axis)
    locked_lateral = lateral_vec(origin, origin, axis)

    insert_servo.update({
        "mode": "align",
        "frames": 0,
        "hold_frames": 0,
        "stroke_frames": 0,
        "ik_fail_count": 0,
        "target_center": insert_center_goal,
        "target_axial": target_axial,
        "origin": origin,
        "axis": axis,
        "locked_lateral": locked_lateral,
        "commanded_axial": axial_coord(module_pos, origin, axis),
        "seat_hold_frames": 0,
    })

    log("\n" + sep("="))
    log(f"[INSERT SERVO INIT {active_robot_label} job={active_job_index}] {active_jobs[active_job_index]['label']}")
    log(f"  actual_module:      {np.round(module_pos, 5)}")
    log(f"  port_origin:        {np.round(origin, 5)}")
    log(f"  insert_axis:        {np.round(axis, 5)}")
    log(f"  align_standoff:     {ALIGN_STANDOFF:.4f} m")
    log(f"  insert_tip_depth:   {INSERT_TIP_DEPTH_M:.4f} m")
    log(f"  target_center:      {np.round(insert_center_goal, 5)}")
    log(f"  target_axial:       {target_axial:.5f}")
    log(f"  command_lead_limit: {INSERT_COMMAND_LEAD_LIMIT * 1000.0:.1f} mm")
    log(f"  success_tip_axial:  {SEAT_SUCCESS_TIP_AXIAL_M:.5f} m")
    log(f"  success_lateral:    {SEAT_SUCCESS_LATERAL_TOL_M * 1000.0:.1f} mm")
    log(sep("="))


def insert_servo_action(joint_pos, offset_local):
    if offset_local is None:
        log("[INSERT SERVO] ERROR: missing block_offset_local; stopping this robot instead of crashing.")
        return None, True
    offset_local = np.asarray(offset_local, dtype=np.float64).reshape(-1)
    if offset_local.shape[0] != 3:
        log(f"[INSERT SERVO] ERROR: bad block_offset_local shape={offset_local.shape}; stopping this robot instead of crashing.")
        return None, True

    module_pos, _ = get_module_pose()
    origin = insert_servo["origin"]
    axis = insert_servo["axis"]
    target_axial = insert_servo["target_axial"]
    locked_lateral = insert_servo["locked_lateral"]

    current_axial = axial_coord(module_pos, origin, axis)
    current_lat = lateral_vec(module_pos, origin, axis)
    lateral_error_vec = locked_lateral - current_lat
    lateral_error = float(np.linalg.norm(lateral_error_vec))

    if insert_servo["mode"] == "align":
        insert_servo["frames"] += 1

        # Keep current depth; drive only the lateral plane onto the port line.
        desired_axial = current_axial
        target_center = pos_from_axial_lateral(origin, axis, desired_axial, locked_lateral)

        if lateral_error <= INSERT_ALIGN_LATERAL_TOL:
            insert_servo["hold_frames"] += 1
        else:
            insert_servo["hold_frames"] = 0

        if insert_servo["frames"] % 120 == 0 or insert_servo["hold_frames"] == 1:
            log(
                f"[INSERT ALIGN] frame={insert_servo['frames']} "
                f"lateral_error_mm={lateral_error * 1000.0:.3f} "
                f"hold={insert_servo['hold_frames']}/{INSERT_ALIGN_HOLD_FRAMES}"
            )

        if insert_servo["hold_frames"] >= INSERT_ALIGN_HOLD_FRAMES:
            insert_servo["mode"] = "stroke"
            insert_servo["stroke_frames"] = 0
            insert_servo["commanded_axial"] = current_axial
            log("\n" + sep())
            log("[INSERT SERVO] Alignment complete. Starting axial crawl.")
            log(f"  start_axial:  {current_axial:.5f}")
            log(f"  target_axial: {target_axial:.5f}")
            log(sep())

        if insert_servo["frames"] >= INSERT_ALIGN_MAX_FRAMES:
            log("\n" + sep())
            log("[INSERT SERVO] ALIGN MAX FRAMES REACHED")
            log(f"  module_pos:          {np.round(module_pos, 5)}")
            log(f"  lateral_error_mm:    {lateral_error * 1000.0:.3f}")

            if lateral_error <= INSERT_ALIGN_PROCEED_TOL:
                insert_servo["mode"] = "stroke"
                insert_servo["stroke_frames"] = 0
                insert_servo["commanded_axial"] = current_axial
                log("  RESULT: close enough; starting axial crawl instead of waiting forever.")
                log(f"  start_axial:         {current_axial:.5f}")
                log(f"  target_axial:        {target_axial:.5f}")
                log(sep())
            else:
                log("  RESULT: too far off; stopping for inspection.")
                log(sep())
                return None, True

    else:
        insert_servo["stroke_frames"] += 1
        sign = 1.0 if target_axial >= current_axial else -1.0
        remaining = abs(target_axial - current_axial)

        commanded_axial = insert_servo.get("commanded_axial")
        if commanded_axial is None:
            commanded_axial = current_axial

        # Do not let the command fall behind the actual module in the insertion direction.
        if sign > 0.0:
            commanded_axial = max(float(commanded_axial), current_axial)
            commanded_axial = min(
                target_axial,
                commanded_axial + INSERT_STROKE_STEP,
                current_axial + INSERT_COMMAND_LEAD_LIMIT,
            )
        else:
            commanded_axial = min(float(commanded_axial), current_axial)
            commanded_axial = max(
                target_axial,
                commanded_axial - INSERT_STROKE_STEP,
                current_axial - INSERT_COMMAND_LEAD_LIMIT,
            )

        insert_servo["commanded_axial"] = float(commanded_axial)
        target_center = pos_from_axial_lateral(origin, axis, commanded_axial, locked_lateral)

        if insert_servo["stroke_frames"] % 60 == 0:
            lead_mm = abs(commanded_axial - current_axial) * 1000.0
            log(
                f"[INSERT STROKE] frame={insert_servo['stroke_frames']} "
                f"actual_axial={current_axial:.5f}->{target_axial:.5f} "
                f"cmd_axial={commanded_axial:.5f} "
                f"lead_mm={lead_mm:.2f} "
                f"remaining_mm={remaining * 1000.0:.2f} "
                f"lateral_error_mm={lateral_error * 1000.0:.3f}"
            )

        tip_axial, tip_lateral_error, tip_pos = leading_tip_axial_and_lateral()
        practical_seated = (
            tip_axial >= SEAT_SUCCESS_TIP_AXIAL_M
            and tip_lateral_error <= SEAT_SUCCESS_LATERAL_TOL_M
        )

        if practical_seated:
            insert_servo["seat_hold_frames"] = int(insert_servo.get("seat_hold_frames", 0)) + 1
        else:
            insert_servo["seat_hold_frames"] = 0

        if insert_servo["stroke_frames"] % 60 == 0:
            log(
                f"[SEAT CHECK] tip_axial={tip_axial:.5f}m "
                f"tip_lateral_mm={tip_lateral_error * 1000.0:.3f} "
                f"hold={insert_servo['seat_hold_frames']}/{SEAT_SUCCESS_HOLD_FRAMES}"
            )

        if insert_servo["seat_hold_frames"] >= SEAT_SUCCESS_HOLD_FRAMES:
            log("\n" + sep())
            log("[INSERT SERVO] Practical seat reached. Stopping before the gripper drags the module back out.")
            log(f"  module_pos:          {np.round(module_pos, 5)}")
            log(f"  center_axial:        {current_axial:.5f}")
            log(f"  target_center_axial: {target_axial:.5f}  # diagnostic only, no longer the stop criterion")
            log(f"  tip_axial:           {tip_axial:.5f}")
            log(f"  tip_lateral_mm:      {tip_lateral_error * 1000.0:.3f}")
            log(f"  tip_pos:             {np.round(tip_pos, 5)}")
            log(sep())
            return None, True

        if remaining <= INSERT_STROKE_STEP and lateral_error <= INSERT_ALIGN_LATERAL_TOL * 2.0:
            log("\n" + sep())
            log("[INSERT SERVO] Geometric target reached.")
            log(f"  module_pos:       {np.round(module_pos, 5)}")
            log(f"  final_axial:      {current_axial:.5f}")
            log(f"  target_axial:     {target_axial:.5f}")
            log(f"  lateral_err_mm:   {lateral_error * 1000.0:.3f}")
            log(sep())
            return None, True

        if insert_servo["stroke_frames"] >= INSERT_STROKE_MAX_FRAMES:
            log("\n" + sep())
            log("[INSERT SERVO] STROKE TIMEOUT")
            log(f"  module_pos:       {np.round(module_pos, 5)}")
            log(f"  final_axial:      {current_axial:.5f}")
            log(f"  target_axial:     {target_axial:.5f}")
            log(f"  lateral_err_mm:   {lateral_error * 1000.0:.3f}")
            log(sep())
            return None, True

    target_hand = hand_target_for_module_center(target_center, port_frame.insert_rot, offset_local)
    action, success = art_kinematics.compute_inverse_kinematics(
        target_position=target_hand,
        target_orientation=port_frame.insert_rot,
        position_tolerance=INSERT_IK_POS_TOL,
        orientation_tolerance=INSERT_IK_ORI_TOL,
    )

    if not success:
        insert_servo["ik_fail_count"] += 1
        if insert_servo["ik_fail_count"] == 1 or insert_servo["ik_fail_count"] % 30 == 0:
            log(
                f"[INSERT SERVO] IK approximate/fail count={insert_servo['ik_fail_count']} "
                f"target_hand={np.round(target_hand, 5)}"
            )
        if insert_servo["ik_fail_count"] >= INSERT_MAX_IK_FAILS:
            log("[INSERT SERVO] Too many IK failures; stopping for inspection.")
            return None, True

    n_dof = int(np.asarray(joint_pos).reshape(-1).shape[0])
    return controller._with_closed_gripper(action, n_dof), False


def leading_tip_axial_and_lateral():
    module_pos, module_ori = get_module_pose()
    rot = quat_to_rot(module_ori)
    length_axis = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    half = QSFP_LENGTH_M * 0.5
    tip_a = module_pos + length_axis * half
    tip_b = module_pos - length_axis * half

    ax_a = axial_coord(tip_a, port_frame.insert_origin, port_frame.insert_axis)
    ax_b = axial_coord(tip_b, port_frame.insert_origin, port_frame.insert_axis)

    if ax_a >= ax_b:
        tip = tip_a
        tip_axial = ax_a
    else:
        tip = tip_b
        tip_axial = ax_b

    tip_lateral_error = float(
        np.linalg.norm(lateral_vec(tip, port_frame.insert_origin, port_frame.insert_axis))
    )
    return float(tip_axial), tip_lateral_error, tip


def print_insert_result():
    module_pos, module_ori = get_module_pose()
    target_center = insert_servo.get("target_center")
    if target_center is None:
        target_center = module_pos

    axial = axial_coord(module_pos, port_frame.insert_origin, port_frame.insert_axis)
    target_axial = insert_servo.get("target_axial", axial)
    lateral_error = float(np.linalg.norm(lateral_vec(module_pos, port_frame.insert_origin, port_frame.insert_axis)))
    center_error = float(np.linalg.norm(module_pos - target_center))

    tip_axial, tip_lateral_error, tip_pos = leading_tip_axial_and_lateral()
    practical_passed = (
        tip_axial >= SEAT_SUCCESS_TIP_AXIAL_M
        and tip_lateral_error <= SEAT_SUCCESS_LATERAL_TOL_M
    )

    # Keep the old strict geometric check as a diagnostic only. It is intentionally
    # not the pass/fail gate because the physical port stops the module earlier.
    strict_passed, strict_metrics = port_frame.evaluate_seat(
        module_pos,
        seat_depth=max(0.0, INSERT_TIP_DEPTH_M),
        module_orientation=module_ori,
        module_half_length=QSFP_LENGTH_M * 0.5,
        lateral_tol=0.008,
        depth_fraction=1.0,
    )

    log("\n" + sep("="))
    log(f"[HYBRID RESULT {active_robot_label} job={active_job_index}] {active_jobs[active_job_index]['label']}")
    log(f"  module_center:             {np.round(module_pos, 5)}")
    log(f"  target_center:             {np.round(target_center, 5)}  # diagnostic target")
    log(f"  center_error_mm:           {center_error * 1000.0:.2f}")
    log(f"  center_axial:              {axial:.5f}")
    log(f"  target_center_axial:       {target_axial:.5f}  # diagnostic target")
    log(f"  center_axial_error_mm:     {(axial - target_axial) * 1000.0:.2f}")
    log(f"  center_lateral_error_mm:   {lateral_error * 1000.0:.2f}")
    log(f"  leading_tip_pos:           {np.round(tip_pos, 5)}")
    log(f"  leading_tip_axial:         {tip_axial:.5f}")
    log(f"  leading_tip_lateral_mm:    {tip_lateral_error * 1000.0:.2f}")
    log(f"  practical_seat_passed:     {practical_passed}")
    log(f"  practical_tip_threshold:   {SEAT_SUCCESS_TIP_AXIAL_M:.5f} m")
    log(f"  practical_lateral_tol_mm:  {SEAT_SUCCESS_LATERAL_TOL_M * 1000.0:.2f}")
    log(f"  strict_geometric_passed:   {strict_passed}  # diagnostic only")
    log(f"  strict_geometric_metrics:  {strict_metrics}")
    log(sep("="))


# =============================================================================
# SCENE SETUP
# =============================================================================

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit(1)

my_world = World(stage_units_in_meters=1.0)
my_world.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
try:
    my_world.get_physics_context().enable_ccd(True)
except Exception:
    pass

stage = omni.usd.get_context().get_stage()

log("[SCENE] Loading DataHall...")
add_reference_to_stage(usd_path=DATAHALL_USD, prim_path="/World/DataHall")
apply_datahall_scale("/World/DataHall", DATAHALL_SCALE)

switch_collider_count = enable_static_collisions("/World/DataHall/Network_Switches", "none")
log(f"[SCENE] Static switch colliders enabled: {switch_collider_count}")

add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
    prim_path="/World/ground",
)

build_work_table(my_world)

robot = add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    prim_path="/World/Franka",
)
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
try:
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")
except Exception:
    pass
physics_variant = robot.GetVariantSet("Physics")
variant_names = list(physics_variant.GetVariantNames())
if variant_names:
    physics_variant.SetVariantSelection(next((n for n in variant_names if n.lower() == "physx"), variant_names[0]))

enable_articulation_collisions("/World/Franka")

robot_2 = add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    prim_path=SECOND_ROBOT_PRIM_PATH,
)
robot_2.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
try:
    robot_2.GetVariantSet("Mesh").SetVariantSelection("Quality")
except Exception:
    pass
physics_variant_2 = robot_2.GetVariantSet("Physics")
variant_names_2 = list(physics_variant_2.GetVariantNames())
if variant_names_2:
    physics_variant_2.SetVariantSelection(next((n for n in variant_names_2 if n.lower() == "physx"), variant_names_2[0]))

enable_articulation_collisions(SECOND_ROBOT_PRIM_PATH)

gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05], dtype=np.float64),
    joint_closed_positions=gripper_closed_positions(),
    action_deltas=np.array([0.02, 0.02], dtype=np.float64),
)

my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="my_franka",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
        position=ROBOT_BASE_POS,
    )
)

gripper_2 = ParallelGripper(
    end_effector_prim_path=f"{SECOND_ROBOT_PRIM_PATH}/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05], dtype=np.float64),
    joint_closed_positions=gripper_closed_positions(),
    action_deltas=np.array([0.02, 0.02], dtype=np.float64),
)

my_franka_2 = my_world.scene.add(
    SingleManipulator(
        prim_path=SECOND_ROBOT_PRIM_PATH,
        name="my_franka_2",
        end_effector_prim_path=f"{SECOND_ROBOT_PRIM_PATH}/panda_rightfinger",
        gripper=gripper_2,
        position=SECOND_ROBOT_BASE_POS,
    )
)

port_frames = [
    build_port_frame(i, jobs=INSERT_JOBS, robot_position=ROBOT_BASE_POS, robot_label="robot1")
    for i in range(len(INSERT_JOBS))
]
secondary_port_frames = [
    build_port_frame(
        i,
        jobs=SECOND_INSERT_JOBS,
        robot_position=SECOND_ROBOT_BASE_POS,
        robot_label="robot2",
    )
    for i in range(len(SECOND_INSERT_JOBS))
]

modules = []
for i, job in enumerate(INSERT_JOBS):
    pick_xy = job["pick_xy"]
    mod = create_qsfp_module(
        my_world,
        prim_path=job["module_prim_path"],
        name=job["module_name"],
        position=np.array([pick_xy[0], pick_xy[1], PICK_SURFACE_Z], dtype=np.float64),
        port_index=None,
    )
    modules.append(mod)
    log(f"[MODULE job={i}] spawned at pick_xy={pick_xy.tolist()} center_z={PICK_SURFACE_Z:.5f}")

secondary_modules = []
for i, job in enumerate(SECOND_INSERT_JOBS):
    pick_xy = job["pick_xy"]
    mod = create_qsfp_module(
        my_world,
        prim_path=job["module_prim_path"],
        name=job["module_name"],
        position=np.array([pick_xy[0], pick_xy[1], PICK_SURFACE_Z], dtype=np.float64),
        port_index=None,
    )
    secondary_modules.append(mod)
    log(
        f"[ROBOT 2 MODULE job={i}] spawned at pick_xy={pick_xy.tolist()} "
        f"center_z={PICK_SURFACE_Z:.5f} placeholder_port={job['label']}"
    )

active_job_index = 0
port_frame = port_frames[0]
module = modules[0]
PICK_XY = INSERT_JOBS[0]["pick_xy"].copy()

my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
my_franka_2.gripper.set_default_state(my_franka_2.gripper.joint_opened_positions)

my_world.reset()
simulation_app.update()

try:
    if hasattr(my_franka, "post_reset"):
        my_franka.post_reset()
    if hasattr(my_franka_2, "post_reset"):
        my_franka_2.post_reset()
    for mod in modules + secondary_modules:
        if hasattr(mod, "post_reset"):
            mod.post_reset()
except Exception:
    pass

apply_startup_camera_view()
simulation_app.update()

lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
kinematics_solver = LulaKinematicsSolver(**lula_config)
task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**lula_config)
art_kinematics = ArticulationKinematicsSolver(my_franka, kinematics_solver, "panda_hand")

base_pos, base_ori = my_franka.get_world_pose()
kinematics_solver.set_robot_base_pose(base_pos, base_ori)
articulation_controller = my_franka.get_articulation_controller()

controller = FrankaMotionController(
    name="hybrid_robot1_controller",
    robot_articulation=my_franka,
    task_traj_gen=task_traj_gen,
    art_kinematics=art_kinematics,
    gripper=my_franka.gripper,
    tool_offset=TOOL_OFFSET,
    physics_dt=PHYSICS_DT,
    position_tolerance=0.005,
    orientation_tolerance=0.02,
    debug=DEBUG,
)

# Robot 2 gets its own Lula solver, task generator, kinematics wrapper,
# articulation controller, and motion queue. Sharing these between robots is
# exactly how you get base-pose cross-talk and impossible-to-debug IK jumps.
lula_config_2 = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
kinematics_solver_2 = LulaKinematicsSolver(**lula_config_2)
task_traj_gen_2 = LulaTaskSpaceTrajectoryGenerator(**lula_config_2)
art_kinematics_2 = ArticulationKinematicsSolver(my_franka_2, kinematics_solver_2, "panda_hand")

base_pos_2, base_ori_2 = my_franka_2.get_world_pose()
kinematics_solver_2.set_robot_base_pose(base_pos_2, base_ori_2)
articulation_controller_2 = my_franka_2.get_articulation_controller()

controller_2 = FrankaMotionController(
    name="hybrid_robot2_controller",
    robot_articulation=my_franka_2,
    task_traj_gen=task_traj_gen_2,
    art_kinematics=art_kinematics_2,
    gripper=my_franka_2.gripper,
    tool_offset=TOOL_OFFSET,
    physics_dt=PHYSICS_DT,
    position_tolerance=0.005,
    orientation_tolerance=0.02,
    debug=DEBUG,
)


def make_robot_runtime(
    *,
    label,
    franka,
    art_kin,
    art_controller,
    motion_controller,
    jobs,
    frames,
    owned_modules,
    start_delay_frames=0,
):
    return {
        "label": label,
        "franka": franka,
        "art_kinematics": art_kin,
        "articulation_controller": art_controller,
        "controller": motion_controller,
        "jobs": jobs,
        "port_frames": frames,
        "modules": owned_modules,
        "active_job_index": 0,
        "port_frame": frames[0],
        "module": owned_modules[0],
        "pick_xy": jobs[0]["pick_xy"].copy(),
        "block_offset_local": None,
        "insert_servo": default_insert_servo_state(),
        "phase": PHASE_WAITING,
        "warmup_frames": POST_RESET_WARMUP_FRAMES + int(start_delay_frames),
        "start_delay_frames": int(start_delay_frames),
    }


def activate_robot_context(ctx):
    global my_franka, art_kinematics, articulation_controller, controller
    global active_jobs, active_robot_label
    global port_frames, modules, active_job_index, port_frame, module
    global PICK_XY, block_offset_local, insert_servo

    my_franka = ctx["franka"]
    art_kinematics = ctx["art_kinematics"]
    articulation_controller = ctx["articulation_controller"]
    controller = ctx["controller"]
    active_jobs = ctx["jobs"]
    active_robot_label = ctx["label"]
    port_frames = ctx["port_frames"]
    modules = ctx["modules"]
    active_job_index = ctx["active_job_index"]
    port_frame = ctx["port_frame"]
    module = ctx["module"]
    PICK_XY = ctx["pick_xy"].copy()
    block_offset_local = ctx["block_offset_local"]
    insert_servo = ctx["insert_servo"]


def sync_robot_context(ctx):
    ctx["active_job_index"] = active_job_index
    ctx["port_frame"] = port_frame
    ctx["module"] = module
    ctx["pick_xy"] = PICK_XY.copy()
    ctx["block_offset_local"] = block_offset_local
    ctx["insert_servo"] = insert_servo


def joint_positions_np(franka):
    joint_pos = franka.get_joint_positions()
    if joint_pos is None:
        return None
    if hasattr(joint_pos, "cpu"):
        joint_pos = joint_pos.cpu().numpy()
    return np.asarray(joint_pos, dtype=np.float64)


def reset_robot_runtime(ctx):
    activate_robot_context(ctx)
    set_active_job(0)
    controller.clear_queue()
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    ctx["phase"] = PHASE_WARMUP
    ctx["warmup_frames"] = POST_RESET_WARMUP_FRAMES + ctx["start_delay_frames"]
    sync_robot_context(ctx)


def step_robot_runtime(ctx):
    global block_offset_local

    activate_robot_context(ctx)
    phase = ctx["phase"]

    if phase == PHASE_WARMUP:
        ctx["warmup_frames"] -= 1
        if ctx["warmup_frames"] <= 0:
            queue_pick_phase()
            phase = PHASE_PICK

    elif phase == PHASE_PICK:
        joint_pos = joint_positions_np(my_franka)
        if joint_pos is not None:
            if controller.is_done():
                if not check_grasp():
                    log(f"[STOP {active_robot_label}] Grasp check failed. Leaving scene open.")
                    phase = PHASE_DONE
                else:
                    block_offset = compute_grasp_offset_local()
                    block_offset_local = block_offset
                    ctx["block_offset_local"] = block_offset
                    queue_transit_phase(block_offset)
                    phase = PHASE_TRANSIT
            else:
                action = controller.forward(joint_pos)
                articulation_controller.apply_action(action)

    elif phase == PHASE_TRANSIT:
        joint_pos = joint_positions_np(my_franka)
        if joint_pos is not None:
            if controller.is_done():
                log(f"\n[PHASE {active_robot_label}] Transit complete -> insertion servo.")
                init_insert_servo()
                phase = PHASE_INSERT_SERVO
            else:
                action = controller.forward(joint_pos)
                articulation_controller.apply_action(action)

    elif phase == PHASE_INSERT_SERVO:
        joint_pos = joint_positions_np(my_franka)
        if joint_pos is not None:
            action, done = insert_servo_action(joint_pos, ctx["block_offset_local"])
            if action is not None:
                articulation_controller.apply_action(action)
            if done:
                print_insert_result()
                if RELEASE_AFTER_INSERT:
                    queue_release_phase()
                    phase = PHASE_RELEASE
                else:
                    phase = PHASE_DONE

    elif phase == PHASE_RELEASE:
        joint_pos = joint_positions_np(my_franka)
        if joint_pos is not None:
            if controller.is_done():
                log("\n" + sep("="))
                log(f"[RELEASE COMPLETE {active_robot_label}]")
                log("  Gripper opened.")
                log(sep("="))
                if RETREAT_AFTER_RELEASE:
                    queue_retreat_phase()
                    phase = PHASE_RETREAT
                else:
                    phase = PHASE_DONE
            else:
                action = controller.forward(joint_pos)
                articulation_controller.apply_action(action)

    elif phase == PHASE_RETREAT:
        joint_pos = joint_positions_np(my_franka)
        if joint_pos is not None:
            if controller.is_done():
                log("\n" + sep("="))
                log(f"[RETREAT COMPLETE {active_robot_label}]")
                log("  Open gripper pulled back.")
                log(sep("="))
                if start_next_job_or_finish():
                    phase = PHASE_PICK
                else:
                    phase = PHASE_DONE
            else:
                action = controller.forward(joint_pos)
                articulation_controller.apply_action(action)

    elif phase == PHASE_DONE:
        if HOLD_FOR_INSPECTION:
            pass

    ctx["phase"] = phase
    sync_robot_context(ctx)


robot_runtimes = [
    make_robot_runtime(
        label="robot1",
        franka=my_franka,
        art_kin=art_kinematics,
        art_controller=articulation_controller,
        motion_controller=controller,
        jobs=INSERT_JOBS,
        frames=port_frames,
        owned_modules=modules,
        start_delay_frames=0,
    ),
    make_robot_runtime(
        label="robot2",
        franka=my_franka_2,
        art_kin=art_kinematics_2,
        art_controller=articulation_controller_2,
        motion_controller=controller_2,
        jobs=SECOND_INSERT_JOBS,
        frames=secondary_port_frames,
        owned_modules=secondary_modules,
        start_delay_frames=ROBOT_2_START_DELAY_FRAMES,
    ),
]


log("\n" + sep("="))
log("[READY] Press Play.")
log("  quiet stable scope: far horizontal-slide alignment, then straight advance and insert.")
log("  This uses measured hand->module offset after grasp and a slow axis servo for insertion.")
log("  Both robots use separate controllers, Lula solvers, state machines, and insert servos.")
log("  Robot 2 starts immediately with robot 1, then runs its three assigned ports.")
log(sep("="))


# =============================================================================
# MAIN LOOP
# =============================================================================

was_playing = False


while simulation_app.is_running():
    playing = my_world.is_playing()

    if not playing:
        was_playing = False
        my_world.step(render=True)
        continue

    if playing and not was_playing:
        log("[RUN] Play detected. Starting dual-robot warmup.")
        for runtime in robot_runtimes:
            reset_robot_runtime(runtime)
        was_playing = True

    for runtime in robot_runtimes:
        step_robot_runtime(runtime)

    my_world.step(render=True)

simulation_app.close()
