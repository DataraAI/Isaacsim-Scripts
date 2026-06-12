from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys

import carb
import carb.settings
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf, UsdLux

from franka_lula_controller import FrankaLulaController

# =============================================================================
# CONFIG
# =============================================================================

DEBUG = True
TOOL_OFFSET = 0.05

# Block dimensions from scale=[0.004, 0.008, 0.050].
# The gripper appears to be grasping across the 8mm side, so each finger
# should stop around 4mm from center when there is real contact.
BLOCK_HALF_WIDTH   = 0.002
BLOCK_HALF_DEPTH   = 0.004
BLOCK_HALF_LENGTH  = 0.025
FINGER_CONTACT_MIN = BLOCK_HALF_DEPTH - 0.0005
MAX_GRASP_ATTEMPTS = 2

# =============================================================================
# TASK CONFIG — CHANGE THESE VALUES FOR EACH BLOCK/PORT
# =============================================================================
# The goal of this section is to make the task reusable:
#   - set where the block starts
#   - set where the block center should end up
#   - keep fixed clearances/speeds as reusable policy
# Everything else is computed from those values.

INSERT_TASKS = [
    {
        "name": "demo_block_to_port_A",
        "block_spawn_position": np.array([0.5, 0.0, 0.025], dtype=np.float64),
        "port_center_position": np.array([-0.6, 0.0, 0.20], dtype=np.float64),

        # Direction the block travels during insertion.
        # Current controller/reporting assumes an X-dominant insertion axis.
        # Use [-1,0,0] for inserting toward negative X, [1,0,0] for positive X.
        "insert_axis_world": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    },

    # Add more tasks later like this:
    # {
    #     "name": "block_to_port_B",
    #     "block_spawn_position": np.array([0.45, 0.10, 0.025], dtype=np.float64),
    #     "port_center_position": np.array([-0.62, 0.08, 0.20], dtype=np.float64),
    #     "insert_axis_world": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    # },
]

ACTIVE_TASK_INDEX = 0
TASK = INSERT_TASKS[ACTIVE_TASK_INDEX]

BLOCK_SPAWN_POSITION = np.asarray(TASK["block_spawn_position"], dtype=np.float64).copy()
PORT_POSITION = np.asarray(TASK["port_center_position"], dtype=np.float64).copy()
INSERT_AXIS_WORLD = np.asarray(TASK["insert_axis_world"], dtype=np.float64).copy()
INSERT_AXIS_WORLD = INSERT_AXIS_WORLD / np.linalg.norm(INSERT_AXIS_WORLD)

if abs(float(INSERT_AXIS_WORLD[0])) < 0.9:
    raise ValueError(
        "This version expects an X-dominant insert axis. "
        "Use [-1,0,0] or [1,0,0], or refactor the X-band helpers to use full-axis projection."
    )

# Permanent offsets/policy. These stay the same across ports unless your fixture changes.
# pre-insert = port center moved backward along the insertion axis by PRE_INSERT_CLEARANCE.
PRE_INSERT_CLEARANCE = 0.02       # 2 cm away from the final port center; slow insertion begins here
TRANSIT_STAGING_CLEARANCE = 0.10  # staging point before pre-insert; joint interpolation handles the route

# Grasp/transport policy. These are hand/waypoint heights used before insertion.
GRASP_APPROACH_Z = 0.20
GRASP_DESCEND_Z_OFFSET = 0.015    # descend target z = block_spawn_z + this
TRANSIT_LIFT_Z = 0.20

# Closed-loop insertion servo. Align Y/Z first, then record the true stroke.
# Start insertion only after the block is already better than the final +/-0.5mm band.
# The tighter 0.35mm pre-alignment gives some margin for small drift during the stroke.
INSERT_ALIGN_TOL = 0.00035       # 0.35 mm Y/Z tolerance before starting measured stroke
INSERT_ALIGN_HOLD_FRAMES = 12    # require stable frames before stroke
INSERT_ALIGN_MAX_FRAMES = 3600   # fail loudly instead of starting insertion misaligned

# Final X band and slow insertion. No manual short-stop/freeze offsets:
# The servo creeps forward until the ACTUAL block center enters this final band.
FINAL_AXIS_TOL = 0.0005          # +/-0.5 mm acceptable final coordinate band
FINAL_ENDPOINT_NORM_TOL = 0.001  # final Euclidean error should be <= 1 mm
INSERT_X_SLOW_STEP = 0.000035    # 0.035 mm/frame ≈ 2.1 mm/s at 60 Hz
# Stroke correction: during the true insert, do not keep pushing X if Y/Z drifts.
# The servo pauses forward X, re-centers Y/Z, then resumes. This is what keeps
# the block from arriving at the correct X while being several mm high/sideways.
INSERT_SERVO_YZ_KP = 1.25       # stroke Y/Z proportional gain
INSERT_SERVO_YZ_KI = 0.0025     # stroke Y/Z integral gain for slow drift
INSERT_SERVO_YZ_INTEGRAL_LIMIT = 0.75
INSERT_SERVO_YZ_MAX_OVERDRIVE = 0.0060  # max 6 mm over-command around desired Y/Z line
STROKE_YZ_PAUSE_TOL = 0.00050   # pause X if Y or Z error exceeds 0.5 mm
STROKE_YZ_RESUME_TOL = 0.00028  # resume X only after Y/Z are back inside 0.28 mm
INSERT_SERVO_MAX_CORRECTION = 0.0015 # legacy clamp, kept for endpoint/older helpers
INSERT_STROKE_MAX_FRAMES = 7000      # safety cap only; pauses may increase frame count

# Pre-insert alignment is allowed to over-command Y/Z around the desired line.
# Reason: the block is not welded to the hand, so commanding only a tiny
# actual+Kp*error step was too weak and alignment stalled 1-2 mm off target.
# This is not a manual position offset; it is closed-loop error compensation.
INSERT_ALIGN_YZ_KP = 1.15
# Integral term removes the steady-state alignment error we saw around 0.55-0.60mm.
# It is automatic: the bias grows only while measured actual block Y/Z remains off target.
INSERT_ALIGN_YZ_KI = 0.004
INSERT_ALIGN_YZ_INTEGRAL_LIMIT = 0.50   # meters*frames clamp; prevents runaway windup
INSERT_ALIGN_YZ_MAX_OVERDRIVE = 0.0070  # max 7 mm over-command around desired Y/Z line
INSERT_ALIGN_X_KP = 0.35
INSERT_ALIGN_X_MAX_OVERDRIVE = 0.0020   # keep the pre-insert X near the computed clearance
X_BAND_OVERSHOOT_EPS = 0.00002          # 0.02 mm numeric cushion, not an allowed extra tolerance

# Final endpoint seating after the measured horizontal stroke. X is never pushed
# once the actual block center is inside the +/-0.5mm target band. The seat phase
# only corrects Y/Z while holding the measured X.
ENDPOINT_SEAT_TOL = 0.0005          # 0.5 mm per axis before considering stable
ENDPOINT_SEAT_HOLD_FRAMES = 8       # short seat; prefer pre-alignment over post-insert correction
ENDPOINT_SEAT_MAX_FRAMES = 40
ENDPOINT_SEAT_KP = 0.20            # very gentle Y/Z cleanup after X is in band
ENDPOINT_SEAT_MAX_CORRECTION = 0.00015 # 0.15 mm clamp; avoid shoving X while correcting Y/Z
ENABLE_ENDPOINT_YZ_SEAT = False    # safest: if pre-alignment was not good enough, fail instead of nudging inside port

# Release policy. The block is released only after the insertion endpoint passes.
# In the current demo there is no physical port/fixture supporting the block, so
# after release the block may move or fall. With a real port asset, this is where
# the port should hold the inserted object.
RELEASE_AFTER_INSERT = True
RELEASE_ONLY_IF_ENDPOINT_OK = True
RELEASE_WAIT_FRAMES = 90
MEASURE_AFTER_RELEASE = True

DOWN_ORI   = np.array([0.0, 1.0, 0.0, 0.0])
INSERT_ORI = np.array([-0.7071068, 0.0, 0.7071068, 0.0])

# =============================================================================
# PHASES
# =============================================================================

PHASE_GRASP      = 0
PHASE_TRANSIT    = 1
PHASE_PRE_INSERT = 2
PHASE_INSERT     = 3
PHASE_RELEASE    = 4
PHASE_DONE       = 5

# =============================================================================
# WORLD SETUP
# =============================================================================

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

world = World(stage_units_in_meters=1.0)

render_settings = carb.settings.get_settings()
render_settings.set_bool("/rtx/shadows/enabled", False)

stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome.CreateIntensityAttr(500.0)
dome.CreateColorAttr((1.0, 1.0, 1.0))

world.scene.add(
    FixedCuboid(
        name="ground",
        position=np.array([0.0, 0.0, -0.005]),
        prim_path="/World/Ground",
        scale=np.array([10.0, 10.0, 0.01]),
        size=1.0,
        color=np.array([0.95, 0.95, 0.95]),
    )
)

robot = add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    prim_path="/World/Franka",
)
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
physics = robot.GetVariantSet("Physics")
physics_names = list(physics.GetVariantNames())
if physics_names:
    physics.SetVariantSelection(
        next((name for name in physics_names if name.lower() == "physx"), physics_names[0])
    )

gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.001, 0.001]),
    action_deltas=np.array([0.02, 0.02]),
)

franka = world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="my_franka",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
    )
)

block = world.scene.add(
    DynamicCuboid(
        name="block",
        position=BLOCK_SPAWN_POSITION,
        prim_path="/World/Block",
        size=1.0,
        scale=np.array([0.004, 0.008, 0.050]),
        color=np.array([0, 0, 1]),
    )
)

franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
world.reset()

lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
lula_kinematics = LulaKinematicsSolver(**lula_config)
task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**lula_config)
art_kinematics = ArticulationKinematicsSolver(franka, lula_kinematics, "panda_hand")
lula_kinematics.set_robot_base_pose(*franka.get_world_pose())

controller = FrankaLulaController(
    name="franka_controller",
    robot_articulation=franka,
    task_traj_gen=task_traj_gen,
    art_kinematics=art_kinematics,
    gripper=franka.gripper,
    tool_offset=TOOL_OFFSET,
    debug=DEBUG,
)

# =============================================================================
# RUN STATISTICS
# =============================================================================

run_count         = 0
all_block_offsets = []
all_hand_errors   = []
all_block_errors  = []
all_path_y_devs   = []
all_path_z_devs   = []
all_path_actual_y_devs = []
all_path_actual_z_devs = []
all_orientation_tilts = []
all_orientation_axis_angles = []
all_endpoint_actual_errors = []
insert_path_samples = []

insert_servo = {
    'active': False,
    'mode': 'idle',
    'align_frames': 0,
    'align_stable_frames': 0,
    'stroke_step': 0,
    'seat_frames': 0,
    'seat_stable_frames': 0,
    'start_x': None,
    'align_hold_x': None,
    'align_integral_y': 0.0,
    'align_integral_z': 0.0,
    'stroke_integral_y': 0.0,
    'stroke_integral_z': 0.0,
    'stroke_paused_for_yz': False,
    'stroke_pause_count': 0,
    'x_direction': None,
    'max_x_overshoot': 0.0,
    'stopped_by_target_plane': False,
    'endpoint_ok': False,
    'insert_ran': False,
    'released': False,
    'warned_ik': False,
}

# =============================================================================
# HELPERS
# =============================================================================

def _sep(char="-", width=60):
    return char * width


def point_before_port(clearance):
    """Block-center point before the port along the approach side.

    If INSERT_AXIS_WORLD is [-1, 0, 0] and PORT_POSITION is [-0.6, 0, 0.2],
    then point_before_port(0.02) returns [-0.58, 0, 0.2].
    """
    return PORT_POSITION - INSERT_AXIS_WORLD * float(clearance)


def grasp_approach_position():
    return np.array([
        BLOCK_SPAWN_POSITION[0],
        BLOCK_SPAWN_POSITION[1],
        GRASP_APPROACH_Z,
    ], dtype=np.float64)


def grasp_descend_position():
    return np.array([
        BLOCK_SPAWN_POSITION[0],
        BLOCK_SPAWN_POSITION[1],
        BLOCK_SPAWN_POSITION[2] + GRASP_DESCEND_Z_OFFSET,
    ], dtype=np.float64)


def transit_lift_position():
    return np.array([
        BLOCK_SPAWN_POSITION[0],
        BLOCK_SPAWN_POSITION[1],
        TRANSIT_LIFT_Z,
    ], dtype=np.float64)


def transit_staging_block_center():
    return point_before_port(TRANSIT_STAGING_CLEARANCE)


def pre_insert_block_center():
    return point_before_port(PRE_INSERT_CLEARANCE)


def print_task_plan():
    print(f"\n{_sep('=')}")
    print(f"[TASK CONFIG] {TASK['name']}")
    print(f"  block_spawn_position:       {np.round(BLOCK_SPAWN_POSITION, 4)}")
    print(f"  port_center_position:       {np.round(PORT_POSITION, 4)}")
    print(f"  insert_axis_world:          {np.round(INSERT_AXIS_WORLD, 4)}")
    print(f"  grasp_approach_position:    {np.round(grasp_approach_position(), 4)}")
    print(f"  grasp_descend_position:     {np.round(grasp_descend_position(), 4)}")
    print(f"  transit_lift_position:      {np.round(transit_lift_position(), 4)}")
    print(f"  transit_staging_center:     {np.round(transit_staging_block_center(), 4)}")
    print(f"  pre_insert_block_center:    {np.round(pre_insert_block_center(), 4)}")
    print(f"  final_block_center:         {np.round(PORT_POSITION, 4)}")
    print(f"  slow_insert_step:           {INSERT_X_SLOW_STEP*1000:.3f} mm/frame")
    print(_sep('='))


def _get_hand_pose():
    pos_raw, rot_raw = art_kinematics.compute_end_effector_pose()
    pos = np.asarray(pos_raw, dtype=np.float64).flatten()
    rot = np.asarray(rot_raw, dtype=np.float64)
    if rot.ndim == 3:
        rot = rot[0]
    return pos, rot


def print_run_banner(run):
    print(f"\n{_sep('=')}")
    print(f"  RUN {run}")
    print(_sep("="))


def check_grasp():
    fingers = franka.gripper.get_joint_positions()
    if fingers is None:
        print("[GRASP] Cannot read finger positions")
        return False
    f1, f2 = float(fingers[0]), float(fingers[1])
    closed = float(franka.gripper.joint_closed_positions[0])
    print(f"\n{_sep()}")
    print("[GRASP CHECK]")
    print(f"  finger_1:      {f1*1000:.3f} mm")
    print(f"  finger_2:      {f2*1000:.3f} mm")
    print(f"  contact_min:   {FINGER_CONTACT_MIN*1000:.3f} mm")
    print(f"  total_gap:     {(f1 + f2)*1000:.3f} mm")
    print(f"  fully_closed:  {closed*1000:.1f} mm")
    ok = f1 >= FINGER_CONTACT_MIN and f2 >= FINGER_CONTACT_MIN
    if ok:
        print("  RESULT: ✓  Contact confirmed")
    else:
        if f1 <= closed + 0.0005 and f2 <= closed + 0.0005:
            print("  RESULT: ✗  Fingers at hard-stop — missed block entirely")
        else:
            print("  RESULT: ✗  Below contact threshold — check approach alignment")
    print(_sep())
    return ok


def compute_and_log_block_offset():
    """Store offset in hand LOCAL frame so it stays valid across orientation changes."""
    block_pos = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    hand_pos, hand_rot = _get_hand_pose()
    offset_world = block_pos - hand_pos
    offset_local = hand_rot.T @ offset_world
    print(f"\n{_sep()}")
    print("[GRASP OFFSET]")
    print(f"  block_world:    {np.round(block_pos, 4)}")
    print(f"  hand_world:     {np.round(hand_pos, 4)}")
    print(f"  offset_world:   {np.round(offset_world, 4)}")
    print(f"  offset_local:   {np.round(offset_local, 4)}  ← hand frame (rotation-invariant)")
    print(f"  magnitude:      {np.linalg.norm(offset_local)*1000:.2f} mm")
    all_block_offsets.append(offset_local.copy())
    if len(all_block_offsets) > 1:
        arr = np.array(all_block_offsets)
        std = np.std(arr, axis=0)
        print(f"\n  Grasp consistency across {len(all_block_offsets)} runs:")
        print(f"    std  x={std[0]*1000:.2f}mm  y={std[1]*1000:.2f}mm  z={std[2]*1000:.2f}mm")
        print(f"    max_deviation={np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1))*1000:.2f}mm")
    print(_sep())
    return offset_local


def hand_target_for_block_center(block_center_pos, target_ori, offset_local):
    """Return the hand position required to place the grasped block center.

    PORT_POSITION is defined as a block-center target. The controller ultimately
    commands the robot hand, so we convert the desired block-center point into
    the exact hand target using the measured hand-frame grasp offset.
    """
    R = quats_to_rot_matrices(np.asarray(target_ori, dtype=np.float64).reshape(1, 4))[0]
    return np.asarray(block_center_pos, dtype=np.float64) - R @ offset_local


def _block_pose_matrix():
    """Return actual block center and orientation matrix from Isaac."""
    block_pos, block_quat = block.get_world_pose()
    block_pos = np.asarray(block_pos, dtype=np.float64).flatten()
    block_quat = np.asarray(block_quat, dtype=np.float64).flatten()
    block_rot = quats_to_rot_matrices(block_quat.reshape(1, 4))[0]
    return block_pos, block_rot, block_quat


def _block_long_axis_world(block_rot):
    """The cuboid's long dimension is its local Z axis because scale Z = 0.050."""
    axis = block_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 1e-9 else axis


def sample_insert_path(offset_local):
    """Sample only the true insert stroke.

    We record BOTH:
      1) estimated block center from hand pose + grasp offset
      2) actual block center/orientation from Isaac physics

    The actual block values are what you are visually watching in Isaac Sim.
    The estimated values tell us whether the hand-frame tracking model is biased.
    """
    hand_pos, hand_rot = _get_hand_pose()
    est_block_center = hand_pos + hand_rot @ offset_local
    actual_block_center, actual_block_rot, actual_block_quat = _block_pose_matrix()
    long_axis_world = _block_long_axis_world(actual_block_rot)

    insert_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    signed_dot = float(np.clip(np.dot(long_axis_world, insert_axis), -1.0, 1.0))
    abs_dot = float(np.clip(abs(signed_dot), -1.0, 1.0))

    # Degrees away from being parallel to the intended insertion axis, ignoring sign.
    axis_angle_deg = float(np.degrees(np.arccos(abs_dot)))

    # Degrees the long axis tilts out of the horizontal XY plane.
    horizontal_tilt_deg = float(np.degrees(np.arcsin(np.clip(abs(long_axis_world[2]), 0.0, 1.0))))

    insert_path_samples.append({
        "estimated_center": est_block_center.copy(),
        "actual_center": actual_block_center.copy(),
        "long_axis": long_axis_world.copy(),
        "quat": actual_block_quat.copy(),
        "axis_angle_deg": axis_angle_deg,
        "horizontal_tilt_deg": horizontal_tilt_deg,
        "signed_dot_insert_axis": signed_dot,
    })


def measure_horizontal_insert_path():
    """Check whether the ACTUAL block moved horizontally during insertion.

    Estimated center is still printed because it helps diagnose tracking bias,
    but pass/fail is based on the actual DynamicCuboid pose from Isaac.
    """
    if len(insert_path_samples) < 2:
        print(f"\n{_sep()}")
        print("[HORIZONTAL INSERT PATH]")
        print("  Not enough samples. The insert stroke may not have run.")
        print(_sep())
        return False

    est_pts = np.array([s["estimated_center"] for s in insert_path_samples], dtype=np.float64)
    actual_pts = np.array([s["actual_center"] for s in insert_path_samples], dtype=np.float64)
    axes = np.array([s["long_axis"] for s in insert_path_samples], dtype=np.float64)
    axis_angles = np.array([s["axis_angle_deg"] for s in insert_path_samples], dtype=np.float64)
    horizontal_tilts = np.array([s["horizontal_tilt_deg"] for s in insert_path_samples], dtype=np.float64)

    est_y_dev = float(np.max(np.abs(est_pts[:, 1] - PORT_POSITION[1])))
    est_z_dev = float(np.max(np.abs(est_pts[:, 2] - PORT_POSITION[2])))

    actual_y_dev = float(np.max(np.abs(actual_pts[:, 1] - PORT_POSITION[1])))
    actual_z_dev = float(np.max(np.abs(actual_pts[:, 2] - PORT_POSITION[2])))
    actual_dx = np.diff(actual_pts[:, 0])
    actual_x_monotonic = bool(np.all(actual_dx <= 1e-5))  # insertion is negative-X
    d = float(np.sign(PORT_POSITION[0] - actual_pts[0, 0]))
    if abs(d) < 1e-9:
        d = -1.0
    near_x = PORT_POSITION[0] - d * FINAL_AXIS_TOL
    far_x = PORT_POSITION[0] + d * FINAL_AXIS_TOL
    actual_x_overshoot = np.maximum(0.0, d * (actual_pts[:, 0] - far_x))
    actual_max_x_overshoot = float(np.max(actual_x_overshoot))
    actual_final_x_in_band = bool((min(far_x, near_x) <= actual_pts[-1, 0] <= max(far_x, near_x)))
    actual_remaining_x_to_band = _x_remaining_to_band(actual_pts[-1, 0])

    max_axis_angle = float(np.max(axis_angles))
    max_horizontal_tilt = float(np.max(horizontal_tilts))
    mean_axis = np.mean(axes, axis=0)

    all_path_y_devs.append(est_y_dev)
    all_path_z_devs.append(est_z_dev)
    all_path_actual_y_devs.append(actual_y_dev)
    all_path_actual_z_devs.append(actual_z_dev)
    all_orientation_axis_angles.append(max_axis_angle)
    all_orientation_tilts.append(max_horizontal_tilt)

    print(f"\n{_sep()}")
    print("[HORIZONTAL INSERT PATH - ACTUAL BLOCK]")
    print(f"  samples:                    {len(actual_pts)}")
    print(f"  actual_max_y_deviation:     {actual_y_dev*1000:.3f} mm")
    print(f"  actual_max_z_deviation:     {actual_z_dev*1000:.3f} mm")
    print(f"  actual_x_monotonic:         {actual_x_monotonic}")
    print(f"  actual_max_x_band_overshoot:{actual_max_x_overshoot*1000:.3f} mm")
    print(f"  final_x_in_band:            {actual_final_x_in_band}")
    print(f"  remaining_x_to_band:        {actual_remaining_x_to_band*1000:.3f} mm")
    print(f"  allowed_x_band:             [{far_x:.4f}, {near_x:.4f}]")
    print(f"  actual_start_block_center:  {np.round(actual_pts[0], 4)}")
    print(f"  actual_end_block_center:    {np.round(actual_pts[-1], 4)}")

    print(f"\n  Estimated/model-based check, for debugging bias only:")
    print(f"  est_max_y_deviation:        {est_y_dev*1000:.3f} mm")
    print(f"  est_max_z_deviation:        {est_z_dev*1000:.3f} mm")
    print(f"  est_start_block_center:     {np.round(est_pts[0], 4)}")
    print(f"  est_end_block_center:       {np.round(est_pts[-1], 4)}")

    print(f"\n  Orientation / horizontality:")
    print(f"  mean_long_axis_world:       {np.round(mean_axis, 4)}")
    print(f"  start_long_axis_world:      {np.round(axes[0], 4)}")
    print(f"  end_long_axis_world:        {np.round(axes[-1], 4)}")
    print(f"  max_axis_angle_to_X:        {max_axis_angle:.3f} deg")
    print(f"  max_tilt_out_of_horizontal: {max_horizontal_tilt:.3f} deg")

    path_ok = (
        actual_y_dev <= 0.001
        and actual_z_dev <= 0.001
        and actual_x_monotonic
        and actual_max_x_overshoot <= 0.0
        and actual_final_x_in_band
    )
    # Orientation does not need to be sub-degree perfect yet. This is a practical early threshold.
    ori_ok = max_horizontal_tilt <= 2.0 and max_axis_angle <= 5.0

    if path_ok:
        print("  PATH RESULT: ✓ actual block path within 1 mm Y/Z, final X in band, no X band overshoot")
    else:
        print("  PATH RESULT: ✗ actual block path is outside 1 mm Y/Z")

    if ori_ok:
        print("  ORIENTATION RESULT: ✓ block stayed reasonably horizontal/aligned")
    else:
        print("  ORIENTATION RESULT: ✗ block orientation drifted too much")

    if len(all_path_actual_y_devs) > 1:
        print(f"\n  Repeatability across {len(all_path_actual_y_devs)} runs:")
        print(f"    actual y_dev mean={np.mean(all_path_actual_y_devs)*1000:.3f}mm  max={np.max(all_path_actual_y_devs)*1000:.3f}mm")
        print(f"    actual z_dev mean={np.mean(all_path_actual_z_devs)*1000:.3f}mm  max={np.max(all_path_actual_z_devs)*1000:.3f}mm")
        print(f"    max tilt mean={np.mean(all_orientation_tilts):.3f}deg  max={np.max(all_orientation_tilts):.3f}deg")
        print(f"    axis angle mean={np.mean(all_orientation_axis_angles):.3f}deg  max={np.max(all_orientation_axis_angles):.3f}deg")
    print(_sep())
    return path_ok and ori_ok


def check_block_still_held(offset_local, label="HOLD CHECK"):
    """Compare actual block pose to the rigid-body estimate from the hand pose."""
    block_pos = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    hand_pos, hand_rot = _get_hand_pose()
    est_block_center = hand_pos + hand_rot @ offset_local
    err = float(np.linalg.norm(block_pos - est_block_center))
    fingers = franka.gripper.get_joint_positions()

    print(f"\n{_sep()}")
    print(f"[{label}]")
    print(f"  actual_block:      {np.round(block_pos, 4)}")
    print(f"  estimated_block:   {np.round(est_block_center, 4)}")
    print(f"  block_hold_error:  {err*1000:.2f} mm")
    if fingers is not None:
        f1, f2 = float(fingers[0]), float(fingers[1])
        print(f"  finger_1:          {f1*1000:.3f} mm")
        print(f"  finger_2:          {f2*1000:.3f} mm")
        print(f"  total_gap:         {(f1 + f2)*1000:.3f} mm")
    if err > 0.015:
        print("  RESULT: ✗ block moved relative to hand — grasp slipped/dropped")
    else:
        print("  RESULT: ✓ block still matches hand-frame offset")
    print(_sep())
    return err < 0.015


def measure_insert_error(offset_local):
    hand_pos, hand_rot = _get_hand_pose()
    R_insert = quats_to_rot_matrices(INSERT_ORI.reshape(1, 4))[0]

    # PORT_POSITION is a BLOCK CENTER target. Therefore the expected hand pose
    # must be computed from the measured grasp offset, not TOOL_OFFSET.
    expected_hand = PORT_POSITION - R_insert @ offset_local

    est_block_center = hand_pos + hand_rot @ offset_local
    est_block_tip    = est_block_center + hand_rot @ np.array([0.0, 0.0, BLOCK_HALF_LENGTH])
    actual_block_center = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()

    hand_err = float(np.linalg.norm(hand_pos - expected_hand))
    est_center_err = float(np.linalg.norm(est_block_center - PORT_POSITION))
    actual_center_err = float(np.linalg.norm(actual_block_center - PORT_POSITION))

    all_hand_errors.append(hand_err)
    all_block_errors.append(est_center_err)
    all_endpoint_actual_errors.append(actual_center_err)

    print(f"\n{_sep()}")
    print("[INSERT RESULT]")
    print(f"  hand_pos:                  {np.round(hand_pos, 4)}")
    print(f"  expected_hand_pos:         {np.round(expected_hand, 4)}")
    print(f"  hand_error:                {hand_err*1000:.2f} mm  ← hand target accuracy")
    print(f"  actual_block_center:       {np.round(actual_block_center, 4)}")
    print(f"  actual_block_center_error: {actual_center_err*1000:.2f} mm  ← main endpoint metric")
    print(f"  est_block_center:          {np.round(est_block_center, 4)}")
    print(f"  est_block_tip:             {np.round(est_block_tip, 4)}")
    print(f"  est_block_center_error:    {est_center_err*1000:.2f} mm  ← model/debug metric")
    print(f"  port_pos (center):         {np.round(PORT_POSITION, 4)}")

    if len(all_endpoint_actual_errors) > 1:
        print(f"\n  Repeatability across {len(all_endpoint_actual_errors)} runs:")
        print(f"    hand          mean={np.mean(all_hand_errors)*1000:.2f}mm  "
              f"std={np.std(all_hand_errors)*1000:.2f}mm  "
              f"max={np.max(all_hand_errors)*1000:.2f}mm")
        print(f"    actual block  mean={np.mean(all_endpoint_actual_errors)*1000:.2f}mm  "
              f"std={np.std(all_endpoint_actual_errors)*1000:.2f}mm  "
              f"max={np.max(all_endpoint_actual_errors)*1000:.2f}mm")
        print(f"    est/model     mean={np.mean(all_block_errors)*1000:.2f}mm  "
              f"std={np.std(all_block_errors)*1000:.2f}mm  "
              f"max={np.max(all_block_errors)*1000:.2f}mm")
    print(_sep())

# =============================================================================
# PHASE COMMAND BUILDERS
# =============================================================================

def queue_grasp_phase():
    controller.clear_queue()
    controller.add_cartesian_waypoint(
        position=grasp_approach_position(),
        orientation=DOWN_ORI,
        pos_tolerance=0.05,
        label="approach_above",
    )
    controller.add_cartesian_waypoint(
        position=grasp_descend_position(),
        orientation=DOWN_ORI,
        pos_tolerance=0.001,
        label="descend_to_block",
    )
    controller.add_gripper_command(action="open",  wait_frames=30)
    controller.add_gripper_command(action="close", wait_frames=90)


def queue_transit_phase():
    """Reliable baseline transport computed from the active task.

    We no longer hardcode [-0.5, 0, 0.20]. The negative/positive X staging
    point is computed from the final port center and TRANSIT_STAGING_CLEARANCE.
    Joint interpolation is allowed to choose the detour automatically.
    """
    controller.clear_queue()

    staging_center = transit_staging_block_center()

    controller.add_cartesian_waypoint(
        position=transit_lift_position(),
        orientation=DOWN_ORI,
        pos_tolerance=0.001,
        hold_gripper=True,
        label="lift",
    )

    controller.add_cartesian_waypoint(
        position=staging_center,
        orientation=DOWN_ORI,
        pos_tolerance=0.003,
        joint_interp=True,
        joint_steps=260,
        max_frames=320,
        hold_gripper=True,
        label="task_staging_joint",
    )

    controller.add_cartesian_waypoint(
        position=staging_center,
        orientation=INSERT_ORI,
        pos_tolerance=0.003,
        joint_interp=True,
        joint_steps=220,
        max_frames=280,
        hold_gripper=True,
        label="task_reorient_joint",
    )


def queue_pre_insert_phase(offset_local):
    """
    Move near the pre-insert pose, but do NOT record this as insertion.

    The robot is allowed to make small non-horizontal alignment motions before
    the actual stroke. The measured insertion begins only after the closed-loop
    servo has pulled the actual block center onto the Y/Z line.
    """
    controller.clear_queue()

    pre_insert_center = pre_insert_block_center()
    final_block_center = PORT_POSITION.copy()

    pre_insert_hand = hand_target_for_block_center(
        pre_insert_center,
        INSERT_ORI,
        offset_local,
    )
    final_insert_hand = hand_target_for_block_center(
        final_block_center,
        INSERT_ORI,
        offset_local,
    )

    print(f"\n{_sep()}")
    print("[INSERT TARGETS]")
    print(f"  pre_insert_block_center: {np.round(pre_insert_center, 4)}")
    print(f"  final_block_center:      {np.round(final_block_center, 4)}")
    print(f"  pre_insert_hand:         {np.round(pre_insert_hand, 4)}")
    print(f"  final_insert_hand:       {np.round(final_insert_hand, 4)}")
    print(f"  insert_axis_world:       {np.round(INSERT_AXIS_WORLD, 4)}")
    print("  control mode:            slow closed-loop insert; stop when actual block enters final band")
    print(_sep())

    controller.add_cartesian_waypoint(
        position=pre_insert_hand,
        orientation=INSERT_ORI,
        pos_tolerance=0.001,
        joint_interp=True,
        joint_steps=140,
        max_frames=180,
        hold_gripper=True,
        target_is_hand=True,
        label="pre_insert_coarse",
    )

    controller.add_cartesian_waypoint(
        position=pre_insert_hand,
        orientation=INSERT_ORI,
        pos_tolerance=0.0005,
        linear=True,
        linear_step=0.00025,
        max_frames=360,
        hold_gripper=True,
        target_is_hand=True,
        label="pre_insert_settle",
    )


def queue_release_phase():
    """Open the gripper after the block has reached the final insert spot.

    This phase intentionally does not move the arm. It only opens the fingers
    and waits a short time so Isaac Sim can apply the gripper command.
    """
    controller.clear_queue()
    controller.add_gripper_command(action="open", wait_frames=RELEASE_WAIT_FRAMES)


def measure_post_release_pose():
    actual_block_center = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    endpoint_err_vec = actual_block_center - PORT_POSITION
    endpoint_norm = float(np.linalg.norm(endpoint_err_vec))
    print(f"\n{_sep()}")
    print("[POST-RELEASE BLOCK POSE]")
    print(f"  actual_block_center: {np.round(actual_block_center, 4)}")
    print(f"  target_port_center:  {np.round(PORT_POSITION, 4)}")
    print(f"  endpoint_error_xyz:  {np.round(endpoint_err_vec * 1000, 3)} mm")
    print(f"  endpoint_error_norm: {endpoint_norm * 1000:.3f} mm")
    print(f"  allowed_x_band:      [{_x_far_band_edge():.4f}, {_x_near_band_edge():.4f}]")
    if _x_in_target_band(actual_block_center[0]):
        print("  X RESULT: ✓ still inside final X band after release")
    else:
        print("  X RESULT: ✗ moved outside final X band after release")
    print(_sep())


def init_insert_servo():
    """Start closed-loop alignment, then the measured horizontal stroke."""
    actual_block_center = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    insert_path_samples.clear()
    insert_servo.update({
        'active': True,
        'mode': 'align',
        'align_frames': 0,
        'align_stable_frames': 0,
        'stroke_step': 0,
        'stroke_frames': 0,
        'seat_frames': 0,
        'seat_stable_frames': 0,
        'start_x': float(actual_block_center[0]),
        'align_hold_x': float(pre_insert_block_center()[0]),
        'align_integral_y': 0.0,
        'align_integral_z': 0.0,
        'stroke_integral_y': 0.0,
        'stroke_integral_z': 0.0,
        'stroke_paused_for_yz': False,
        'stroke_pause_count': 0,
        'x_direction': float(np.sign(INSERT_AXIS_WORLD[0])),
        'max_x_overshoot': 0.0,
        'stopped_by_target_plane': False,
        'endpoint_ok': False,
        'released': False,
        'warned_ik': False,
    })
    print(f"\n{_sep()}")
    print("[INSERT SERVO INIT]")
    print(f"  actual_start_block_center: {np.round(actual_block_center, 4)}")
    print(f"  align hold X:              x={pre_insert_block_center()[0]:.4f}")
    print(f"  align target Y/Z:          y={PORT_POSITION[1]:.4f}, z={PORT_POSITION[2]:.4f}")
    print(f"  final X target band:       [{_x_far_band_edge():.4f}, {_x_near_band_edge():.4f}] (±{FINAL_AXIS_TOL*1000:.1f}mm)")
    print(f"  pre-align tolerance:       ±{INSERT_ALIGN_TOL*1000:.3f} mm in Y/Z")
    print(f"  slow insert step:          {INSERT_X_SLOW_STEP*1000:.3f} mm/frame")
    print(f"  stroke Y/Z pause tol:      ±{STROKE_YZ_PAUSE_TOL*1000:.3f} mm; resume at ±{STROKE_YZ_RESUME_TOL*1000:.3f} mm")
    print("  control rule:              PI pre-align, then creep forward; pause X whenever Y/Z drifts")
    print("  NOTE: alignment frames are not counted as insertion path samples")
    print(_sep())


def _clamp_correction(value):
    return float(np.clip(value, -INSERT_SERVO_MAX_CORRECTION, INSERT_SERVO_MAX_CORRECTION))


def _clamp_endpoint_correction(value):
    return float(np.clip(value, -ENDPOINT_SEAT_MAX_CORRECTION, ENDPOINT_SEAT_MAX_CORRECTION))


def _clamp_align_yz_overdrive(value):
    return float(np.clip(value, -INSERT_ALIGN_YZ_MAX_OVERDRIVE, INSERT_ALIGN_YZ_MAX_OVERDRIVE))


def _clamp_align_integral(value):
    return float(np.clip(value, -INSERT_ALIGN_YZ_INTEGRAL_LIMIT, INSERT_ALIGN_YZ_INTEGRAL_LIMIT))


def _clamp_stroke_integral(value):
    return float(np.clip(value, -INSERT_SERVO_YZ_INTEGRAL_LIMIT, INSERT_SERVO_YZ_INTEGRAL_LIMIT))


def _clamp_stroke_yz_overdrive(value):
    return float(np.clip(value, -INSERT_SERVO_YZ_MAX_OVERDRIVE, INSERT_SERVO_YZ_MAX_OVERDRIVE))


def _clamp_align_x_overdrive(value):
    return float(np.clip(value, -INSERT_ALIGN_X_MAX_OVERDRIVE, INSERT_ALIGN_X_MAX_OVERDRIVE))


def _insert_direction():
    """Return +1 or -1 for the insertion direction along world X.

    Direction is taken from INSERT_AXIS_WORLD. Current code supports X-dominant
    insertion axes, so this returns -1 for [-1,0,0] and +1 for [1,0,0].
    """
    d = insert_servo.get('x_direction', None)
    if d is None or abs(float(d)) < 1e-9:
        start_x = insert_servo.get('start_x', None)
        if start_x is None:
            return float(np.sign(INSERT_AXIS_WORLD[0]))
        d = np.sign(float(INSERT_AXIS_WORLD[0]))
        if abs(float(d)) < 1e-9:
            d = -1.0
    return float(d)


def _x_near_band_edge():
    """Short-side edge of the acceptable X target band.

    For negative-X insertion toward -0.6000 with FINAL_AXIS_TOL=0.0005,
    this is -0.5995. Once the actual block reaches this edge, it is inside
    the acceptable band and X motion should stop.
    """
    d = _insert_direction()
    return float(PORT_POSITION[0] - d * FINAL_AXIS_TOL)


def _x_far_band_edge():
    """Overshoot-side edge of the acceptable X target band.

    For negative-X insertion toward -0.6000 with FINAL_AXIS_TOL=0.0005,
    this is -0.6005. Crossing past this edge violates the no-overshoot spec.
    """
    d = _insert_direction()
    return float(PORT_POSITION[0] + d * FINAL_AXIS_TOL)


def _x_remaining_to_target(actual_x):
    """Remaining positive distance along insertion direction before target X."""
    d = _insert_direction()
    return float(max(0.0, d * (float(PORT_POSITION[0]) - float(actual_x))))


def _x_remaining_to_band(actual_x):
    """Remaining positive distance before the actual block enters the +/-0.5mm band."""
    d = _insert_direction()
    return float(max(0.0, d * (_x_near_band_edge() - float(actual_x))))


def _x_overshoot_distance(actual_x):
    """How far the actual block center has crossed beyond the allowed X band.

    This is stricter than 'past the target plane'. For negative-X insertion,
    overshoot starts only when actual_x < PORT_POSITION[0] - FINAL_AXIS_TOL.
    """
    d = _insert_direction()
    return float(max(0.0, d * (float(actual_x) - _x_far_band_edge())))


def _x_in_target_band(actual_x):
    """True if actual X is inside the final +/-0.5mm band."""
    return _x_remaining_to_band(actual_x) <= 0.0 and _x_overshoot_distance(actual_x) <= 0.0


def _clip_x_not_past_band_entry(commanded_x):
    """Clamp commanded X so the servo never aims deeper than the near edge.

    This is not a hand-tuned offset. It is the mathematically defined entrance
    to the allowed final X band.
    """
    d = _insert_direction()
    near_x = _x_near_band_edge()
    if d < 0.0:
        return float(max(float(commanded_x), near_x))
    return float(min(float(commanded_x), near_x))


def _endpoint_status(actual_block_center):
    """Return endpoint status using the real block center.

    Endpoint pass requires:
      - X inside the +/- FINAL_AXIS_TOL band
      - Y and Z inside +/- FINAL_AXIS_TOL
      - Euclidean center error <= FINAL_ENDPOINT_NORM_TOL
      - no meaningful X-band overshoot beyond the small numeric epsilon
    """
    actual_block_center = np.asarray(actual_block_center, dtype=np.float64).flatten()
    err = actual_block_center - PORT_POSITION
    endpoint_norm = float(np.linalg.norm(err))
    overshoot_x = _x_overshoot_distance(actual_block_center[0])
    x_in_range = _x_in_target_band(actual_block_center[0]) or overshoot_x <= X_BAND_OVERSHOOT_EPS
    y_in_range = abs(err[1]) <= FINAL_AXIS_TOL
    z_in_range = abs(err[2]) <= FINAL_AXIS_TOL
    endpoint_ok = (
        x_in_range
        and y_in_range
        and z_in_range
        and endpoint_norm <= FINAL_ENDPOINT_NORM_TOL
        and overshoot_x <= X_BAND_OVERSHOOT_EPS
    )
    return endpoint_ok, err, endpoint_norm, overshoot_x


def _x_band_entry_target():
    """Return the X coordinate we are allowed to command during insertion.

    This is NOT a hand-tuned short-stop. It is the near edge of the acceptable
    final X band. For target -0.6000 and +/-0.5mm tolerance, the band is
    [-0.6005, -0.5995], so the entry target is -0.5995 for negative-X insertion.

    Commanding the band-entry edge prevents us from intentionally pushing past
    the valid range, while still forcing the actual block to keep moving until
    it reaches the band.
    """
    return _x_near_band_edge()


def _servo_target_block_center(actual_block_center):
    """Return the commanded block-center target for this servo frame.

    The important fix here is that Y/Z correction is now a true servo step:
        target = actual + Kp * (desired - actual)

    The older code used:
        target = desired + Kp * error
    which effectively over-commanded past the desired line. That can look okay
    in free motion, but during contact-style insertion it causes extra drift.
    """
    actual_block_center = np.asarray(actual_block_center, dtype=np.float64).flatten()

    if insert_servo['mode'] == 'align':
        # Pre-insert line alignment.
        #
        # The earlier "actual + Kp*error" version was too weak here: the block
        # stalled about 1-2 mm off in Y/Z and never started the real insert.
        # For alignment, we deliberately over-command around the desired line:
        #
        #     target = desired + Kp * (desired - actual)
        #
        # This is still closed-loop: every frame it recomputes the over-command
        # from the actual block pose. It is not a hand-tuned port offset.
        hold_x = insert_servo.get('align_hold_x', None)
        if hold_x is None:
            hold_x = float(pre_insert_block_center()[0])

        desired = np.array([
            float(hold_x),
            PORT_POSITION[1],
            PORT_POSITION[2],
        ], dtype=np.float64)
        err = desired - actual_block_center

        # PI alignment: proportional correction reacts immediately; integral
        # correction slowly builds an automatic bias if the block stalls just
        # outside tolerance. This fixes the repeated 0.55-0.60mm residual error
        # without using a hand-tuned port offset.
        insert_servo['align_integral_y'] = _clamp_align_integral(
            float(insert_servo.get('align_integral_y', 0.0)) + float(err[1])
        )
        insert_servo['align_integral_z'] = _clamp_align_integral(
            float(insert_servo.get('align_integral_z', 0.0)) + float(err[2])
        )

        y_overdrive = (
            INSERT_ALIGN_YZ_KP * err[1]
            + INSERT_ALIGN_YZ_KI * float(insert_servo['align_integral_y'])
        )
        z_overdrive = (
            INSERT_ALIGN_YZ_KP * err[2]
            + INSERT_ALIGN_YZ_KI * float(insert_servo['align_integral_z'])
        )

        target = np.array([
            desired[0] + _clamp_align_x_overdrive(INSERT_ALIGN_X_KP * err[0]),
            desired[1] + _clamp_align_yz_overdrive(y_overdrive),
            desired[2] + _clamp_align_yz_overdrive(z_overdrive),
        ], dtype=np.float64)
        return target, err

    if insert_servo['mode'] == 'seat':
        # Final endpoint seat: X is frozen at the actual measured X. Do not push
        # deeper and do not pull back through the port. Only make very small Y/Z
        # corrections, and only if the pre-stroke alignment was not already good
        # enough.
        desired = np.array([
            actual_block_center[0],
            PORT_POSITION[1],
            PORT_POSITION[2],
        ], dtype=np.float64)
        err = desired - actual_block_center
        correction = np.array([
            0.0,
            _clamp_endpoint_correction(ENDPOINT_SEAT_KP * err[1]),
            _clamp_endpoint_correction(ENDPOINT_SEAT_KP * err[2]),
        ], dtype=np.float64)
        return actual_block_center + correction, err

    # True insertion stroke: creep along X, but do NOT keep advancing if Y/Z
    # drift out of the allowed line. This is the key accuracy fix. The previous
    # version reached the X band while Z drifted several millimeters. That means
    # it was prioritizing insertion depth over alignment. For a port, that is
    # backwards.

    desired_line = np.array([
        actual_block_center[0],
        PORT_POSITION[1],
        PORT_POSITION[2],
    ], dtype=np.float64)
    line_err = desired_line - actual_block_center
    yz_abs = float(max(abs(line_err[1]), abs(line_err[2])))

    # Hysteresis: pause forward X when Y/Z error gets too large, and only resume
    # once the block is clearly back on the line. This avoids rapid on/off chatter.
    paused = bool(insert_servo.get('stroke_paused_for_yz', False))
    if paused:
        if yz_abs <= STROKE_YZ_RESUME_TOL:
            paused = False
    else:
        if yz_abs >= STROKE_YZ_PAUSE_TOL:
            paused = True

    insert_servo['stroke_paused_for_yz'] = paused
    if paused:
        insert_servo['stroke_pause_count'] = int(insert_servo.get('stroke_pause_count', 0)) + 1

    if _x_in_target_band(actual_block_center[0]):
        target_x = float(actual_block_center[0])  # freeze X once actually in band
    elif paused:
        target_x = float(actual_block_center[0])  # hold depth; recover Y/Z first
    else:
        d = _insert_direction()
        remaining_to_band = _x_remaining_to_band(actual_block_center[0])
        x_step = min(INSERT_X_SLOW_STEP, remaining_to_band)
        target_x = float(actual_block_center[0]) + d * x_step
        target_x = _clip_x_not_past_band_entry(target_x)

    desired = np.array([
        target_x,
        PORT_POSITION[1],
        PORT_POSITION[2],
    ], dtype=np.float64)
    err = desired - actual_block_center

    # PI Y/Z correction during the stroke. If the block slowly rises or slides
    # while being inserted, the integral term builds an automatic counter-bias.
    # This is not a manual offset; it comes only from measured block error.
    insert_servo['stroke_integral_y'] = _clamp_stroke_integral(
        float(insert_servo.get('stroke_integral_y', 0.0)) + float(err[1])
    )
    insert_servo['stroke_integral_z'] = _clamp_stroke_integral(
        float(insert_servo.get('stroke_integral_z', 0.0)) + float(err[2])
    )

    y_overdrive = (
        INSERT_SERVO_YZ_KP * err[1]
        + INSERT_SERVO_YZ_KI * float(insert_servo['stroke_integral_y'])
    )
    z_overdrive = (
        INSERT_SERVO_YZ_KP * err[2]
        + INSERT_SERVO_YZ_KI * float(insert_servo['stroke_integral_z'])
    )

    return np.array([
        target_x,
        PORT_POSITION[1] + _clamp_stroke_yz_overdrive(y_overdrive),
        PORT_POSITION[2] + _clamp_stroke_yz_overdrive(z_overdrive),
    ], dtype=np.float64), err

def insert_servo_action(joint_pos, offset_local):
    """Closed-loop insertion action based on the ACTUAL block pose.

    This is intentionally outside the queued controller because the target must
    be recomputed every frame from block.get_world_pose(). The measured path is
    the actual block path, not the hand model.
    """
    n_dof = joint_pos.shape[0]
    actual_block_center = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    target_block_center, err = _servo_target_block_center(actual_block_center)

    target_hand = hand_target_for_block_center(
        target_block_center,
        INSERT_ORI,
        offset_local,
    )

    action, success = art_kinematics.compute_inverse_kinematics(
        target_position=target_hand,
        target_orientation=INSERT_ORI,
        position_tolerance=0.0002,
        orientation_tolerance=0.03,
    )

    if not success:
        if not insert_servo['warned_ik']:
            print(f"[INSERT SERVO] IK failed once. target_hand={np.round(target_hand, 4)}")
            insert_servo['warned_ik'] = True
        return controller._with_closed_gripper(controller._hold_action(n_dof), n_dof)

    return controller._with_closed_gripper(action, n_dof)


def update_insert_servo_state(offset_local):
    """Advance align/stroke state and return True when insertion is complete."""
    actual_block_center = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()

    if insert_servo['mode'] == 'align':
        insert_servo['align_frames'] += 1
        yz_err = np.array([
            actual_block_center[1] - PORT_POSITION[1],
            actual_block_center[2] - PORT_POSITION[2],
        ], dtype=np.float64)

        if abs(yz_err[0]) <= INSERT_ALIGN_TOL and abs(yz_err[1]) <= INSERT_ALIGN_TOL:
            insert_servo['align_stable_frames'] += 1
        else:
            insert_servo['align_stable_frames'] = 0

        if insert_servo['align_frames'] % 300 == 0 and insert_servo['align_stable_frames'] == 0:
            x_err = float(actual_block_center[0] - pre_insert_block_center()[0])
            print(f"[INSERT SERVO] aligning... frame={insert_servo['align_frames']} "
                  f"x_err={x_err*1000:.3f}mm y_err={yz_err[0]*1000:.3f}mm z_err={yz_err[1]*1000:.3f}mm "
                  f"i_y={float(insert_servo.get('align_integral_y', 0.0)):.4f} "
                  f"i_z={float(insert_servo.get('align_integral_z', 0.0)):.4f}")

        if insert_servo['align_frames'] >= INSERT_ALIGN_MAX_FRAMES and insert_servo['align_stable_frames'] < INSERT_ALIGN_HOLD_FRAMES:
            print(f"\n{_sep()}")
            print("[INSERT SERVO] FAILED: alignment did not reach Y/Z tolerance")
            print(f"  actual_block_center: {np.round(actual_block_center, 4)}")
            print(f"  y_error:             {yz_err[0]*1000:.3f} mm")
            print(f"  z_error:             {yz_err[1]*1000:.3f} mm")
            print(f"  align_integral_y:    {float(insert_servo.get('align_integral_y', 0.0)):.5f}")
            print(f"  align_integral_z:    {float(insert_servo.get('align_integral_z', 0.0)):.5f}")
            print("  Refusing to start insertion from a bad Y/Z line.")
            print(_sep())
            insert_servo['endpoint_ok'] = False
            return True

        if insert_servo['align_stable_frames'] >= INSERT_ALIGN_HOLD_FRAMES:
            # Stop accumulating pre-align bias once the actual block is on the line.
            insert_servo['align_integral_y'] = 0.0
            insert_servo['align_integral_z'] = 0.0
            insert_servo['stroke_integral_y'] = 0.0
            insert_servo['stroke_integral_z'] = 0.0
            insert_servo['stroke_paused_for_yz'] = False
            insert_servo['stroke_pause_count'] = 0
            insert_servo['mode'] = 'stroke'
            insert_servo['insert_ran'] = True
            insert_servo['stroke_step'] = 0
            insert_servo['stroke_frames'] = 0
            insert_servo['start_x'] = float(actual_block_center[0])
            insert_path_samples.clear()
            print(f"\n{_sep()}")
            print("[INSERT SERVO] Starting measured horizontal stroke")
            print(f"  stroke_start_block_center: {np.round(actual_block_center, 4)}")
            print(f"  residual_y_error:          {yz_err[0]*1000:.3f} mm")
            print(f"  residual_z_error:          {yz_err[1]*1000:.3f} mm")
            print(f"  target X band:             [{_x_far_band_edge():.4f}, {_x_near_band_edge():.4f}]")
            print(_sep())
        return False

    if insert_servo['mode'] == 'stroke':
        # Stroke mode: sample the actual block path and adaptively advance X
        # until actual block X enters the +/-0.5mm target band.
        sample_insert_path(offset_local)

        overshoot_x = _x_overshoot_distance(actual_block_center[0])
        remaining_band_x = _x_remaining_to_band(actual_block_center[0])
        remaining_true_x = _x_remaining_to_target(actual_block_center[0])
        insert_servo['max_x_overshoot'] = max(insert_servo['max_x_overshoot'], overshoot_x)

        if overshoot_x > X_BAND_OVERSHOOT_EPS:
            print(f"\n{_sep()}")
            print("[INSERT SERVO] HARD STOP: X left the allowed +/-0.5mm band")
            print(f"  actual_block_center: {np.round(actual_block_center, 4)}")
            print(f"  x_band_overshoot:    {overshoot_x*1000:.3f} mm")
            print("  No corrective pullback commanded; stopping to protect the port.")
            print(_sep())
            insert_servo['endpoint_ok'] = False
            return True

        insert_servo['stroke_step'] += 1
        insert_servo['stroke_frames'] += 1

        if insert_servo.get('stroke_paused_for_yz', False) and insert_servo['stroke_frames'] % 120 == 0:
            yz_err_now = actual_block_center[[1, 2]] - PORT_POSITION[[1, 2]]
            print(f"[INSERT SERVO] X paused for Y/Z recovery: frame={insert_servo['stroke_frames']} "
                  f"x={actual_block_center[0]:.4f} y_err={yz_err_now[0]*1000:.3f}mm z_err={yz_err_now[1]*1000:.3f}mm")

        reached_x_band = _x_in_target_band(actual_block_center[0])
        stroke_timeout = insert_servo['stroke_frames'] >= INSERT_STROKE_MAX_FRAMES

        if reached_x_band:
            endpoint_ok_now, endpoint_err_vec, endpoint_norm_now, overshoot_now = _endpoint_status(actual_block_center)
            reason = "actual block entered final X band"
            insert_servo['stopped_by_target_plane'] = True
            print(f"\n{_sep()}")
            if endpoint_ok_now:
                print("[INSERT SERVO] Horizontal stroke stopped; endpoint already passes")
            else:
                print("[INSERT SERVO] Horizontal stroke stopped; starting gentle Y/Z endpoint seat")
            print(f"  reason:                    {reason}")
            print(f"  pre-seat actual_block:     {np.round(actual_block_center, 4)}")
            print(f"  remaining_x_to_band:       {remaining_band_x*1000:.3f} mm")
            print(f"  remaining_x_to_target:     {remaining_true_x*1000:.3f} mm")
            print(f"  current_x_band_overshoot:  {overshoot_x*1000:.3f} mm")
            print(f"  pre-seat endpoint_xyz:     {np.round(endpoint_err_vec*1000, 3)} mm")
            print(f"  pre-seat endpoint_error:   {endpoint_norm_now*1000:.3f} mm")
            print(f"  stroke_yz_pause_frames:    {int(insert_servo.get('stroke_pause_count', 0))}")
            if endpoint_ok_now:
                print("  NOTE: skipping seat phase to avoid disturbing X after a valid insert.")
                print(_sep())
                insert_servo['endpoint_ok'] = True
                return True
            if not ENABLE_ENDPOINT_YZ_SEAT:
                print("  NOTE: endpoint Y/Z seat is disabled; failing instead of nudging after X is inside the port.")
                print("        Fix the pre-insert alignment instead of scraping inside the port.")
                print(_sep())
                insert_servo['endpoint_ok'] = False
                return True
            print("  NOTE: seat correction freezes X and only corrects Y/Z very gently")
            print(_sep())
            insert_servo['mode'] = 'seat'
            insert_servo['seat_frames'] = 0
            insert_servo['seat_stable_frames'] = 0

        elif stroke_timeout:
            # Do NOT pretend this is success. If the actual block has not reached
            # the band, the endpoint is wrong. Stop loudly instead of freezing X
            # 20-30mm short and reporting a misleading path pass.
            print(f"\n{_sep()}")
            print("[INSERT SERVO] FAILED: stroke timed out before actual X entered target band")
            print(f"  actual_block_center:       {np.round(actual_block_center, 4)}")
            print(f"  allowed_x_band:            [{_x_far_band_edge():.4f}, {_x_near_band_edge():.4f}]")
            print(f"  remaining_x_to_band:       {remaining_band_x*1000:.3f} mm")
            print(f"  remaining_x_to_target:     {remaining_true_x*1000:.3f} mm")
            print("  This is not an overshoot problem; it means X command/progress is too weak.")
            print(_sep())
            insert_servo['endpoint_ok'] = False
            return True
        return False

    if insert_servo['mode'] == 'seat':
        insert_servo['seat_frames'] += 1
        endpoint_err_vec = actual_block_center - PORT_POSITION
        endpoint_norm = float(np.linalg.norm(endpoint_err_vec))
        overshoot_x = _x_overshoot_distance(actual_block_center[0])
        remaining_band_x = _x_remaining_to_band(actual_block_center[0])
        remaining_true_x = _x_remaining_to_target(actual_block_center[0])
        insert_servo['max_x_overshoot'] = max(insert_servo['max_x_overshoot'], overshoot_x)

        if overshoot_x > X_BAND_OVERSHOOT_EPS:
            print(f"\n{_sep()}")
            print("[ENDPOINT SEAT] HARD STOP: X left the allowed +/-0.5mm band")
            print(f"  actual_block_center: {np.round(actual_block_center, 4)}")
            print(f"  x_band_overshoot:    {overshoot_x*1000:.3f} mm")
            print("  No corrective pullback commanded; stopping to protect the port.")
            print(_sep())
            insert_servo['endpoint_ok'] = False
            return True

        endpoint_ok, endpoint_err_vec, endpoint_norm, overshoot_x = _endpoint_status(actual_block_center)

        if endpoint_ok:
            insert_servo['seat_stable_frames'] += 1
        else:
            insert_servo['seat_stable_frames'] = 0

        done = (
            insert_servo['seat_stable_frames'] >= ENDPOINT_SEAT_HOLD_FRAMES
            or insert_servo['seat_frames'] >= ENDPOINT_SEAT_MAX_FRAMES
        )

        if done:
            if insert_servo['seat_frames'] >= ENDPOINT_SEAT_MAX_FRAMES:
                print("[INSERT SERVO] Endpoint seat max frames reached.")
            print(f"\n{_sep()}")
            print("[ENDPOINT SEAT RESULT]")
            print(f"  final_actual_block_center: {np.round(actual_block_center, 4)}")
            print(f"  target_port_center:        {np.round(PORT_POSITION, 4)}")
            print(f"  allowed_x_band:            [{_x_far_band_edge():.4f}, {_x_near_band_edge():.4f}]")
            print(f"  endpoint_error_xyz:        {np.round(endpoint_err_vec * 1000, 3)} mm")
            print(f"  endpoint_error_norm:       {endpoint_norm * 1000:.3f} mm")
            print(f"  remaining_x_to_band:       {remaining_band_x * 1000:.3f} mm")
            print(f"  remaining_x_to_target:     {remaining_true_x * 1000:.3f} mm")
            print(f"  final_x_band_overshoot:    {overshoot_x * 1000:.3f} mm")
            print(f"  max_x_band_overshoot_seen: {insert_servo['max_x_overshoot'] * 1000:.3f} mm")
            insert_servo['endpoint_ok'] = bool(endpoint_ok and insert_servo['max_x_overshoot'] <= X_BAND_OVERSHOOT_EPS)
            if insert_servo['endpoint_ok']:
                print("  RESULT: ✓ endpoint within ±0.5mm per axis, no band overshoot")
            else:
                print("  RESULT: ✗ endpoint/overshoot requirement not satisfied")
            print(_sep())
            return True
        return False

    return False

# =============================================================================
# STATE
# =============================================================================

phase              = PHASE_GRASP
grasp_attempt      = 0
block_offset_local = None
reset_needed       = False

run_count += 1
print_run_banner(run_count)
print_task_plan()
queue_grasp_phase()

# =============================================================================
# MAIN LOOP
# =============================================================================

while simulation_app.is_running():
    world.step(render=True)

    if world.is_stopped() and not reset_needed:
        reset_needed = True

    if world.is_playing():
        if reset_needed:
            run_count += 1
            print_run_banner(run_count)
            print_task_plan()
            world.reset()
            controller.reset()
            lula_kinematics.set_robot_base_pose(*franka.get_world_pose())
            phase = PHASE_GRASP
            grasp_attempt = 0
            block_offset_local = None
            insert_path_samples.clear()
            insert_servo.update({
                'active': False,
                'mode': 'idle',
                'align_frames': 0,
                'align_stable_frames': 0,
                'stroke_step': 0,
                'seat_frames': 0,
                'seat_stable_frames': 0,
                'start_x': None,
                'align_hold_x': None,
                'align_integral_y': 0.0,
                'align_integral_z': 0.0,
                'stroke_integral_y': 0.0,
                'stroke_integral_z': 0.0,
                'stroke_paused_for_yz': False,
                'stroke_pause_count': 0,
                'x_direction': None,
                'max_x_overshoot': 0.0,
                'stopped_by_target_plane': False,
                'endpoint_ok': False,
                'released': False,
                'warned_ik': False,
            })
            reset_needed = False
            queue_grasp_phase()
            continue

        joint_pos = franka.get_joint_positions()
        if joint_pos is None:
            continue

        if phase == PHASE_INSERT and block_offset_local is not None:
            done = update_insert_servo_state(block_offset_local)
            if done:
                print("\n[PHASE] INSERT complete")
                if insert_servo.get('insert_ran', False):
                    measure_insert_error(block_offset_local)
                    measure_horizontal_insert_path()
                else:
                    print("\n[PHASE] Insert stroke never ran; skipping endpoint/path measurement.")
                insert_servo['active'] = False

                should_release = RELEASE_AFTER_INSERT and (
                    (not RELEASE_ONLY_IF_ENDPOINT_OK) or insert_servo.get('endpoint_ok', False)
                )

                if should_release:
                    print("\n[PHASE] → RELEASE BLOCK")
                    phase = PHASE_RELEASE
                    queue_release_phase()
                else:
                    if RELEASE_AFTER_INSERT and RELEASE_ONLY_IF_ENDPOINT_OK:
                        print("\n[PHASE] Release skipped because endpoint did not pass. Holding block for inspection.")
                    phase = PHASE_DONE
                    print("\n[PHASE] DONE — press Stop to reset and run again")
            else:
                franka.get_articulation_controller().apply_action(
                    insert_servo_action(joint_pos, block_offset_local)
                )
            continue

        if controller.is_done():

            if phase == PHASE_GRASP:
                print(f"\n[PHASE] GRASP complete (attempt {grasp_attempt + 1})")
                if check_grasp():
                    block_offset_local = compute_and_log_block_offset()
                    print("[PHASE] → TRANSIT")
                    phase = PHASE_TRANSIT
                    grasp_attempt = 0
                    queue_transit_phase()
                else:
                    grasp_attempt += 1
                    if grasp_attempt >= MAX_GRASP_ATTEMPTS:
                        print(f"\n[PHASE] Grasp failed {MAX_GRASP_ATTEMPTS} times. HALTING.")
                        break
                    print(f"[PHASE] Retrying grasp ({grasp_attempt}/{MAX_GRASP_ATTEMPTS})")
                    queue_grasp_phase()

            elif phase == PHASE_TRANSIT:
                print("\n[PHASE] TRANSIT complete")
                if block_offset_local is not None:
                    still_held = check_block_still_held(block_offset_local, label="POST-TRANSIT HOLD CHECK")
                    if not still_held:
                        print("[PHASE] Block slipped during transit. HALTING before insert.")
                        break
                print("[PHASE] → PRE-INSERT ALIGNMENT")
                phase = PHASE_PRE_INSERT
                insert_path_samples.clear()
                queue_pre_insert_phase(block_offset_local)

            elif phase == PHASE_PRE_INSERT:
                print("\n[PHASE] PRE-INSERT setup complete")
                print("[PHASE] → CLOSED-LOOP HORIZONTAL INSERT")
                phase = PHASE_INSERT
                init_insert_servo()

            elif phase == PHASE_RELEASE:
                insert_servo['released'] = True
                print("\n[PHASE] RELEASE complete — gripper opened")
                if MEASURE_AFTER_RELEASE:
                    measure_post_release_pose()
                phase = PHASE_DONE
                print("\n[PHASE] DONE — press Stop to reset and run again")

        franka.get_articulation_controller().apply_action(
            controller.forward(joint_pos)
        )

simulation_app.close()