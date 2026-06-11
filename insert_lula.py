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

# How far back from PORT_POSITION to sit before the insert stroke.
# Change this one number when the port asset defines the approach clearance.
PRE_INSERT_CLEARANCE = 0.10   # meters

# PORT_POSITION is the target BLOCK CENTER at insert.
# TODO: replace with port.get_world_pose()[0] when asset is ready.
PORT_POSITION = np.array([-0.6, 0.0, 0.20])

DOWN_ORI   = np.array([0.0, 1.0, 0.0, 0.0])
INSERT_ORI = np.array([-0.7071068, 0.0, 0.7071068, 0.0])

# =============================================================================
# PHASES
# =============================================================================

PHASE_GRASP   = 0
PHASE_TRANSIT = 1
PHASE_INSERT  = 2
PHASE_DONE    = 3

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
        position=np.array([0.5, 0.0, 0.025]),
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

# =============================================================================
# HELPERS
# =============================================================================

def _sep(char="-", width=60):
    return char * width


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
    expected_hand   = PORT_POSITION - R_insert @ np.array([0.0, 0.0, TOOL_OFFSET])
    est_block_center = hand_pos + hand_rot @ offset_local
    est_block_tip    = est_block_center + hand_rot @ np.array([0.0, 0.0, BLOCK_HALF_LENGTH])
    hand_err   = float(np.linalg.norm(hand_pos - expected_hand))
    center_err = float(np.linalg.norm(est_block_center - PORT_POSITION))
    all_hand_errors.append(hand_err)
    all_block_errors.append(center_err)
    print(f"\n{_sep()}")
    print("[INSERT RESULT]")
    print(f"  hand_pos:            {np.round(hand_pos, 4)}")
    print(f"  expected_hand_pos:   {np.round(expected_hand, 4)}")
    print(f"  hand_error:          {hand_err*1000:.2f} mm  ← trajectory accuracy")
    print(f"  est_block_center:    {np.round(est_block_center, 4)}")
    print(f"  est_block_tip:       {np.round(est_block_tip, 4)}")
    print(f"  port_pos (center):   {np.round(PORT_POSITION, 4)}")
    print(f"  block_center_error:  {center_err*1000:.2f} mm  ← overall accuracy")
    if len(all_hand_errors) > 1:
        print(f"\n  Repeatability across {len(all_hand_errors)} runs:")
        print(f"    hand   mean={np.mean(all_hand_errors)*1000:.2f}mm  "
              f"std={np.std(all_hand_errors)*1000:.2f}mm  "
              f"max={np.max(all_hand_errors)*1000:.2f}mm")
        print(f"    block  mean={np.mean(all_block_errors)*1000:.2f}mm  "
              f"std={np.std(all_block_errors)*1000:.2f}mm  "
              f"max={np.max(all_block_errors)*1000:.2f}mm")
    print(_sep())

# =============================================================================
# PHASE COMMAND BUILDERS
# =============================================================================

def queue_grasp_phase():
    controller.clear_queue()
    controller.add_cartesian_waypoint(
        position=np.array([0.5, 0.0, 0.20]),
        orientation=DOWN_ORI,
        pos_tolerance=0.05,
        label="approach_above",
    )
    controller.add_cartesian_waypoint(
        position=np.array([0.5, 0.0, 0.04]),
        orientation=DOWN_ORI,
        pos_tolerance=0.001,
        label="descend_to_block",
    )
    controller.add_gripper_command(action="open",  wait_frames=30)
    controller.add_gripper_command(action="close", wait_frames=90)


def queue_transit_phase():
    """
    Reliable baseline transport.

    Transport does not need a Cartesian-straight path. The old failing warning
    came from asking Lula to convert the edge-of-workspace neg_x_side segment.
    The blended slerp version then failed near the end because it forced a
    continuous Cartesian pose path through an awkward wrist region.

    This version keeps the route that actually picks up and carries the block,
    but uses joint-space interpolation for the transit/reorientation pieces.
    That bypasses Lula without forcing per-frame Cartesian IK through the wrist
    singularity. The final insert stroke remains linear=True in queue_insert_phase().
    """
    controller.clear_queue()

    controller.add_cartesian_waypoint(
        position=np.array([0.5, 0.0, 0.20]),
        orientation=DOWN_ORI,
        pos_tolerance=0.001,
        hold_gripper=True,
        label="lift",
    )

    controller.add_cartesian_waypoint(
        position=np.array([0.0, 0.5, 0.20]),
        orientation=DOWN_ORI,
        pos_tolerance=0.001,
        hold_gripper=True,
        label="y_detour",
    )

    # Transport: smooth joint-space move to the negative-X side.
    # No Lula. No Cartesian straight-line requirement. Keeps the grasp stable.
    controller.add_cartesian_waypoint(
        position=np.array([-0.5, 0.0, 0.20]),
        orientation=DOWN_ORI,
        pos_tolerance=0.003,
        joint_interp=True,
        joint_steps=260,
        max_frames=320,
        hold_gripper=True,
        label="neg_x_side_joint",
    )

    # Reorientation is also a transport/setup action, not the insert stroke.
    # Smooth it in joint space to avoid wrist snapping / branch flipping.
    controller.add_cartesian_waypoint(
        position=np.array([-0.5, 0.0, 0.20]),
        orientation=INSERT_ORI,
        pos_tolerance=0.003,
        joint_interp=True,
        joint_steps=220,
        max_frames=280,
        hold_gripper=True,
        label="reorient_joint",
    )


def queue_insert_phase():
    """
    linear=True: the insert stroke MUST be straight and controlled.
    Lula is not used here — per-frame IK stepping guarantees a straight line.

    PRE_INSERT_CLEARANCE positions the arm back from the port before the stroke.
    Derived from PORT_POSITION so changing one constant covers both waypoints.

    Distance = PRE_INSERT_CLEARANCE (0.10m default) at 0.001m/step = 100 frames.
    max_frames=400 gives 4x headroom.
    """
    pre_insert_pos = PORT_POSITION + np.array([PRE_INSERT_CLEARANCE, 0.0, 0.0])
    controller.clear_queue()
    controller.add_cartesian_waypoint(
        position=pre_insert_pos,
        orientation=INSERT_ORI,
        pos_tolerance=0.002,
        hold_gripper=True,
        label="pre_insert",
    )
    controller.add_cartesian_waypoint(
        position=PORT_POSITION,   # TODO: swap with port.get_world_pose()[0] when asset is ready
        orientation=INSERT_ORI,
        pos_tolerance=0.001,
        linear=True,
        linear_step=0.001,
        max_frames=400,
        hold_gripper=True,
        label="insert_stroke",
    )
    controller.add_gripper_command(action="open", wait_frames=60)

# =============================================================================
# STATE
# =============================================================================

phase              = PHASE_GRASP
grasp_attempt      = 0
block_offset_local = None
reset_needed       = False

run_count += 1
print_run_banner(run_count)
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
            world.reset()
            controller.reset()
            lula_kinematics.set_robot_base_pose(*franka.get_world_pose())
            phase = PHASE_GRASP
            grasp_attempt = 0
            block_offset_local = None
            reset_needed = False
            queue_grasp_phase()
            continue

        joint_pos = franka.get_joint_positions()
        if joint_pos is None:
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
                print("[PHASE] → INSERT")
                phase = PHASE_INSERT
                queue_insert_phase()

            elif phase == PHASE_INSERT:
                print("\n[PHASE] INSERT complete")
                measure_insert_error(block_offset_local)
                phase = PHASE_DONE
                print("\n[PHASE] DONE — press Stop to reset and run again")

        franka.get_articulation_controller().apply_action(
            controller.forward(joint_pos)
        )

simulation_app.close()