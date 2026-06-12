from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# cuMotion is used only for the macro carry to staging. Obstacle test code is removed.
try:
    import omni.kit.app

    _ext_mgr = omni.kit.app.get_app().get_extension_manager()
    for _ext_name in (
        "isaacsim.robot_motion.experimental.motion_generation",
        "isaacsim.robot_motion.cumotion",
    ):
        try:
            _ext_mgr.set_extension_enabled_immediate(_ext_name, True)
        except Exception as exc:
            print(f"[cuMotion] Could not enable {_ext_name}: {exc}")
except Exception as exc:
    print(f"[cuMotion] Extension enabling skipped: {exc}")

import sys

import carb
import carb.settings
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf, UsdLux, UsdPhysics, PhysxSchema, UsdShade

try:
    import isaacsim.robot_motion.cumotion as cu_mg
    import isaacsim.robot_motion.experimental.motion_generation as mg

    CUMOTION_AVAILABLE = True
    CUMOTION_IMPORT_ERROR = None
except Exception as exc:
    cu_mg = None
    mg = None
    CUMOTION_AVAILABLE = False
    CUMOTION_IMPORT_ERROR = exc

from franka_motion_controller import FrankaMotionController


def _sep(char="-", width=60):
    return char * width


# =============================================================================
# CONFIG — baseline without obstacle testing
# =============================================================================

DEBUG = True
TOOL_OFFSET = 0.05

BLOCK_HALF_DEPTH = 0.004
BLOCK_HALF_LENGTH = 0.025
FINGER_CONTACT_MIN = BLOCK_HALF_DEPTH - 0.0005
MAX_GRASP_ATTEMPTS = 2

TASK = {
    "name": "demo_block_to_port_A",
    "block_spawn_position": np.array([0.5, 0.0, 0.025], dtype=np.float64),
    "port_center_position": np.array([-0.6, 0.0, 0.20], dtype=np.float64),
    "insert_axis_world": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
}

BLOCK_SPAWN_POSITION = TASK["block_spawn_position"].copy()
PORT_POSITION = TASK["port_center_position"].copy()
INSERT_AXIS_WORLD = TASK["insert_axis_world"] / np.linalg.norm(TASK["insert_axis_world"])

if abs(float(INSERT_AXIS_WORLD[0])) < 0.9:
    raise ValueError("This cleaned baseline expects an X-dominant insertion axis.")

PRE_INSERT_CLEARANCE = 0.02
TRANSIT_STAGING_CLEARANCE = 0.10
GRASP_APPROACH_Z = 0.20
GRASP_DESCEND_Z_OFFSET = 0.015
TRANSIT_LIFT_Z = 0.20

# Macro carry only. No obstacle test, no payload collision proxy, no strict replan loop.
ENABLE_CUMOTION_TRANSIT = True
CUMOTION_FALLBACK_TO_JOINT_INTERP = True
CUMOTION_VISUALIZE_DEBUG_PRIMS = False
CUMOTION_DEFAULT_SAFETY_TOLERANCE = 0.035
CUMOTION_TRANSIT_MAX_VELOCITIES = np.array([0.25, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35])
CUMOTION_TRANSIT_MAX_ACCELERATIONS = np.array([0.25, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35])
CUMOTION_EXECUTION_DT = 1.0 / 60.0  # kept from the working baseline
CUMOTION_TRANSIT_MAX_EXEC_FRAMES = 900
CUMOTION_TRANSIT_FINAL_HOLD_FRAMES = 45

# Insertion tolerances and servo tuning from the successful run.
YZ_TOTAL_TOL = 0.0010
YZ_EQUAL_AXIS_COMPONENT_TOL = YZ_TOTAL_TOL / np.sqrt(2.0)
INSERT_ALIGN_TOTAL_TOL = 0.00050
INSERT_ALIGN_HOLD_FRAMES = 12
INSERT_ALIGN_MAX_FRAMES = 3600
FINAL_AXIS_TOL = 0.0005
CENTER_AXIS_TOL = 0.00025
ACTUAL_X_JITTER_TOL = 0.00020
FINAL_ENDPOINT_NORM_TOL = float(np.sqrt(YZ_TOTAL_TOL**2 + CENTER_AXIS_TOL**2))
INSERT_X_SLOW_STEP = 0.000035
INSERT_MAX_PUSH_THROUGH = 0.0040
INSERT_MAX_BACKSLIDE = 0.006
STROKE_YZ_PAUSE_TOL = 0.00070
STROKE_YZ_RESUME_TOL = 0.00040
INSERT_STROKE_MAX_FRAMES = 7000

INSERT_ALIGN_YZ_KP = 1.15
INSERT_ALIGN_YZ_KI = 0.004
INSERT_ALIGN_YZ_INTEGRAL_LIMIT = 0.50
INSERT_ALIGN_YZ_MAX_OVERDRIVE = 0.0070
INSERT_ALIGN_X_KP = 0.35
INSERT_ALIGN_X_MAX_OVERDRIVE = 0.0020

INSERT_SERVO_YZ_KP = 1.25
INSERT_SERVO_YZ_KI = 0.0025
INSERT_SERVO_YZ_INTEGRAL_LIMIT = 0.75
INSERT_SERVO_YZ_MAX_OVERDRIVE = 0.0060
X_BAND_OVERSHOOT_EPS = 0.00002

RELEASE_AFTER_INSERT = False
HOLD_AFTER_INSERT_FOR_INSPECTION = True
AUTO_CLOSE_ON_BLOCK_SLIP = False
AUTO_CLOSE_ON_CUMOTION_FAILURE = False
AUTO_CLOSE_ON_GRASP_FAILURE = False

ENABLE_PHYSICAL_GRASP_TUNING = True
BLOCK_MASS_KG = 0.0020
GRIP_STATIC_FRICTION = 3.0
GRIP_DYNAMIC_FRICTION = 2.5
GRIP_RESTITUTION = 0.0
BLOCK_SOLVER_POSITION_ITERS = 48
BLOCK_SOLVER_VELOCITY_ITERS = 12
FINGER_SOLVER_POSITION_ITERS = 48
FINGER_SOLVER_VELOCITY_ITERS = 12
GRIPPER_DRIVE_STIFFNESS = 2.0e5
GRIPPER_DRIVE_DAMPING = 2.0e4
GRIPPER_DRIVE_MAX_FORCE = 5000.0
USE_HIGH_RATE_CONTACT_SOLVER = True
PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0

DOWN_ORI = np.array([0.0, 1.0, 0.0, 0.0])
INSERT_ORI = np.array([-0.7071068, 0.0, 0.7071068, 0.0])

PHASE_GRASP = 0
PHASE_TRANSIT = 1
PHASE_PRE_INSERT = 2
PHASE_INSERT = 3
PHASE_DONE = 4


# =============================================================================
# WORLD SETUP
# =============================================================================

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

world = World(stage_units_in_meters=1.0)
if USE_HIGH_RATE_CONTACT_SOLVER:
    try:
        world.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
        print(f"[PHYSICS] simulation dt set: physics={PHYSICS_DT:.5f}s, rendering={RENDERING_DT:.5f}s")
    except Exception as exc:
        print(f"[PHYSICS] Could not set simulation dt: {exc!r}")

carb.settings.get_settings().set_bool("/rtx/shadows/enabled", False)
stage = omni.usd.get_context().get_stage()

light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
light.CreateIntensityAttr(500.0)
light.CreateColorAttr((1.0, 1.0, 1.0))

world.scene.add(FixedCuboid(
    name="ground",
    position=np.array([0.0, 0.0, -0.005]),
    prim_path="/World/Ground",
    scale=np.array([10.0, 10.0, 0.01]),
    size=1.0,
    color=np.array([0.95, 0.95, 0.95]),
))

robot = add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    prim_path="/World/Franka",
)
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
physics_variant = robot.GetVariantSet("Physics")
variant_names = list(physics_variant.GetVariantNames())
if variant_names:
    physics_variant.SetVariantSelection(next((n for n in variant_names if n.lower() == "physx"), variant_names[0]))

gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.001, 0.001]),
    action_deltas=np.array([0.02, 0.02]),
)

franka = world.scene.add(SingleManipulator(
    prim_path="/World/Franka",
    name="my_franka",
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    gripper=gripper,
))

block = world.scene.add(DynamicCuboid(
    name="block",
    position=BLOCK_SPAWN_POSITION,
    prim_path="/World/Block",
    size=1.0,
    scale=np.array([0.004, 0.008, 0.050]),
    color=np.array([0, 0, 1]),
))


# =============================================================================
# PHYSICS TUNING
# =============================================================================

def _set_attr_safe(api, create_name, get_name, value):
    try:
        attr = None
        get_fn = getattr(api, get_name, None)
        if callable(get_fn):
            attr = get_fn()
        if attr is None or not attr:
            create_fn = getattr(api, create_name, None)
            if callable(create_fn):
                try:
                    attr = create_fn(value)
                except TypeError:
                    attr = create_fn()
        if attr is not None and attr:
            attr.Set(value)
            return True
    except Exception:
        pass
    return False


def _bind_physics_material(material, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[PHYSICS] material bind skipped; missing prim: {prim_path}")
        return
    try:
        api = UsdShade.MaterialBindingAPI.Apply(prim)
        try:
            api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
        except TypeError:
            api.Bind(material)
        print(f"[PHYSICS] high-friction material bound to {prim_path}")
    except Exception as exc:
        print(f"[PHYSICS] material bind failed for {prim_path}: {exc!r}")


def _apply_mass_and_solver(prim_path, mass_kg=None, pos_iters=None, vel_iters=None):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[PHYSICS] solver tuning skipped; missing prim: {prim_path}")
        return

    if mass_kg is not None:
        try:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            ok = _set_attr_safe(mass_api, "CreateMassAttr", "GetMassAttr", float(mass_kg))
            print(f"[PHYSICS] mass {'set' if ok else 'not set'} for {prim_path}: {mass_kg:.4f} kg")
        except Exception as exc:
            print(f"[PHYSICS] mass tuning failed for {prim_path}: {exc!r}")

    try:
        rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if pos_iters is not None:
            ok = _set_attr_safe(rb_api, "CreateSolverPositionIterationCountAttr", "GetSolverPositionIterationCountAttr", int(pos_iters))
            print(f"[PHYSICS] solver position iters {'set' if ok else 'not set'} for {prim_path}: {pos_iters}")
        if vel_iters is not None:
            ok = _set_attr_safe(rb_api, "CreateSolverVelocityIterationCountAttr", "GetSolverVelocityIterationCountAttr", int(vel_iters))
            print(f"[PHYSICS] solver velocity iters {'set' if ok else 'not set'} for {prim_path}: {vel_iters}")
    except Exception as exc:
        print(f"[PHYSICS] solver tuning failed for {prim_path}: {exc!r}")


def _tune_gripper_drive(joint_path):
    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid():
        print(f"[PHYSICS] drive tuning skipped; missing joint: {joint_path}")
        return
    try:
        drive = UsdPhysics.DriveAPI.Get(prim, "linear") or UsdPhysics.DriveAPI.Apply(prim, "linear")
        _set_attr_safe(drive, "CreateStiffnessAttr", "GetStiffnessAttr", float(GRIPPER_DRIVE_STIFFNESS))
        _set_attr_safe(drive, "CreateDampingAttr", "GetDampingAttr", float(GRIPPER_DRIVE_DAMPING))
        _set_attr_safe(drive, "CreateMaxForceAttr", "GetMaxForceAttr", float(GRIPPER_DRIVE_MAX_FORCE))
        print(f"[PHYSICS] gripper drive tuned for {joint_path}")
    except Exception as exc:
        print(f"[PHYSICS] gripper drive tuning failed for {joint_path}: {exc!r}")


def configure_physical_grasp():
    if not ENABLE_PHYSICAL_GRASP_TUNING:
        return

    print("\n" + _sep())
    print("[PHYSICS] Applying physical grasp tuning — no payload attach/weld")
    material = UsdShade.Material.Define(stage, Sdf.Path("/World/GripHighFrictionPhysicsMaterial"))
    mat_prim = material.GetPrim()

    try:
        mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
        _set_attr_safe(mat_api, "CreateStaticFrictionAttr", "GetStaticFrictionAttr", float(GRIP_STATIC_FRICTION))
        _set_attr_safe(mat_api, "CreateDynamicFrictionAttr", "GetDynamicFrictionAttr", float(GRIP_DYNAMIC_FRICTION))
        _set_attr_safe(mat_api, "CreateRestitutionAttr", "GetRestitutionAttr", float(GRIP_RESTITUTION))
        try:
            physx_api = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
            _set_attr_safe(physx_api, "CreateFrictionCombineModeAttr", "GetFrictionCombineModeAttr", "max")
            _set_attr_safe(physx_api, "CreateRestitutionCombineModeAttr", "GetRestitutionCombineModeAttr", "min")
        except Exception:
            pass
        print(f"[PHYSICS] grip material created: static_friction={GRIP_STATIC_FRICTION}, dynamic_friction={GRIP_DYNAMIC_FRICTION}, restitution={GRIP_RESTITUTION}")
    except Exception as exc:
        print(f"[PHYSICS] grip material creation failed: {exc!r}")

    for path in (
        "/World/Block",
        "/World/Franka/panda_leftfinger",
        "/World/Franka/panda_rightfinger",
        "/World/Franka/panda_leftfinger/geometry",
        "/World/Franka/panda_rightfinger/geometry",
    ):
        _bind_physics_material(material, path)

    _apply_mass_and_solver("/World/Block", BLOCK_MASS_KG, BLOCK_SOLVER_POSITION_ITERS, BLOCK_SOLVER_VELOCITY_ITERS)
    for path in ("/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger"):
        _apply_mass_and_solver(path, None, FINGER_SOLVER_POSITION_ITERS, FINGER_SOLVER_VELOCITY_ITERS)
    for path in ("/World/Franka/panda_finger_joint1", "/World/Franka/panda_finger_joint2"):
        _tune_gripper_drive(path)
    print(_sep())


configure_physical_grasp()
franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
world.reset()

kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
kinematics_solver = LulaKinematicsSolver(**kinematics_config)
task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**kinematics_config)
art_kinematics = ArticulationKinematicsSolver(franka, kinematics_solver, "panda_hand")
kinematics_solver.set_robot_base_pose(*franka.get_world_pose())

controller = FrankaMotionController(
    name="franka_controller",
    robot_articulation=franka,
    task_traj_gen=task_traj_gen,
    art_kinematics=art_kinematics,
    gripper=franka.gripper,
    tool_offset=TOOL_OFFSET,
    debug=DEBUG,
)


# =============================================================================
# TASK HELPERS
# =============================================================================

def point_before_port(clearance):
    return PORT_POSITION - INSERT_AXIS_WORLD * float(clearance)


def grasp_approach_position():
    return np.array([BLOCK_SPAWN_POSITION[0], BLOCK_SPAWN_POSITION[1], GRASP_APPROACH_Z], dtype=np.float64)


def grasp_descend_position():
    return np.array([BLOCK_SPAWN_POSITION[0], BLOCK_SPAWN_POSITION[1], BLOCK_SPAWN_POSITION[2] + GRASP_DESCEND_Z_OFFSET], dtype=np.float64)


def transit_lift_position():
    return np.array([BLOCK_SPAWN_POSITION[0], BLOCK_SPAWN_POSITION[1], TRANSIT_LIFT_Z], dtype=np.float64)


def transit_staging_block_center():
    return point_before_port(TRANSIT_STAGING_CLEARANCE)


def pre_insert_block_center():
    return point_before_port(PRE_INSERT_CLEARANCE)


def print_run_banner(run):
    print(f"\n{_sep('=')}\n  RUN {run}\n{_sep('=')}")


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
    print(f"  slow_insert_step:           {INSERT_X_SLOW_STEP * 1000:.3f} mm/frame")
    print(f"  Y/Z path tolerance:         radial <= {YZ_TOTAL_TOL * 1000:.3f} mm total")
    print(f"  transit policy:             cuMotion macro carry if available; no obstacle test")
    print(f"  release policy:             release_after_insert={RELEASE_AFTER_INSERT}; hold_for_inspection={HOLD_AFTER_INSERT_FOR_INSPECTION}")
    print(_sep('='))


def _get_hand_pose():
    pos, rot = art_kinematics.compute_end_effector_pose()
    pos = np.asarray(pos, dtype=np.float64).flatten()
    rot = np.asarray(rot, dtype=np.float64)
    if rot.ndim == 3:
        rot = rot[0]
    return pos, rot


def _block_pose_matrix():
    pos, quat = block.get_world_pose()
    pos = np.asarray(pos, dtype=np.float64).flatten()
    quat = np.asarray(quat, dtype=np.float64).flatten()
    rot = quats_to_rot_matrices(quat.reshape(1, 4))[0]
    return pos, rot, quat


def hand_target_for_block_center(block_center_pos, target_ori, offset_local):
    rot = quats_to_rot_matrices(np.asarray(target_ori, dtype=np.float64).reshape(1, 4))[0]
    return np.asarray(block_center_pos, dtype=np.float64) - rot @ offset_local


def _block_long_axis_world(block_rot):
    axis = block_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 1e-9 else axis


def _get_dof_names():
    for attr in ("dof_names", "joint_names"):
        value = getattr(franka, attr, None)
        if value is not None:
            return list(value)
    for method_name in ("get_dof_names", "get_joint_names"):
        method = getattr(franka, method_name, None)
        if callable(method):
            try:
                return list(method())
            except Exception:
                pass
    return [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1", "panda_finger_joint2",
    ]


# =============================================================================
# cuMotion macro carry to staging
# =============================================================================

cumotion_transit = {
    "initialized": False,
    "available": False,
    "active": False,
    "trajectory": None,
    "elapsed": 0.0,
    "frames": 0,
    "duration": None,
    "done_hold_frames": 0,
    "controlled_joint_names": None,
    "planner": None,
    "world_binding": None,
    "last_error": None,
}


def _current_cumotion_q(joint_pos):
    names = _get_dof_names()
    return np.array([float(joint_pos[names.index(j)]) for j in cumotion_transit["controlled_joint_names"]], dtype=np.float64)


def _closed_gripper_action(cumotion_positions, joint_pos):
    names = _get_dof_names()
    full_positions = [None] * len(joint_pos)
    for i, joint_name in enumerate(cumotion_transit["controlled_joint_names"]):
        if joint_name in names and i < len(cumotion_positions):
            full_positions[names.index(joint_name)] = float(cumotion_positions[i])

    closed = np.asarray(franka.gripper.joint_closed_positions, dtype=np.float64).flatten()
    for finger_name, value in zip(["panda_finger_joint1", "panda_finger_joint2"], closed):
        if finger_name in names:
            full_positions[names.index(finger_name)] = float(value)
    return ArticulationAction(joint_positions=full_positions)


def _robot_root_world_poses():
    if hasattr(franka, "get_world_poses"):
        try:
            pos, quat = franka.get_world_poses()
            pos = np.asarray(pos, dtype=np.float64)
            quat = np.asarray(quat, dtype=np.float64)
            return pos.reshape(1, 3) if pos.ndim == 1 else pos, quat.reshape(1, 4) if quat.ndim == 1 else quat
        except Exception:
            pass
    pos, quat = franka.get_world_pose()
    return np.asarray(pos, dtype=np.float64).reshape(1, 3), np.asarray(quat, dtype=np.float64).reshape(1, 4)


def initialize_cumotion_once():
    if cumotion_transit["initialized"]:
        return cumotion_transit["available"]
    cumotion_transit["initialized"] = True

    if not ENABLE_CUMOTION_TRANSIT:
        cumotion_transit["last_error"] = "ENABLE_CUMOTION_TRANSIT=False"
        return False
    if not CUMOTION_AVAILABLE:
        cumotion_transit["last_error"] = f"cuMotion import failed: {CUMOTION_IMPORT_ERROR}"
        print(f"[cuMotion TRANSIT] unavailable: {cumotion_transit['last_error']}")
        return False

    try:
        cumotion_robot = cu_mg.load_cumotion_supported_robot("franka")
        controlled_joint_names = list(cumotion_robot.controlled_joint_names)

        obstacle_strategy = mg.ObstacleStrategy()
        obstacle_strategy.set_default_safety_tolerance(CUMOTION_DEFAULT_SAFETY_TOLERANCE)
        world_interface = cu_mg.CumotionWorldInterface(visualize_debug_prims=CUMOTION_VISUALIZE_DEBUG_PRIMS)
        world_binding = mg.WorldBinding(
            world_interface=world_interface,
            obstacle_strategy=obstacle_strategy,
            tracked_prims=[],
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        world_binding.initialize()

        planner = cu_mg.GraphBasedMotionPlanner(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=world_binding.get_world_interface(),
            graph_planner_config_filename="graph_based_motion_planner_config.yaml",
        )

        cumotion_transit.update({
            "available": True,
            "controlled_joint_names": controlled_joint_names,
            "planner": planner,
            "world_binding": world_binding,
            "last_error": None,
        })

        print(f"\n{_sep()}")
        print("[cuMotion TRANSIT] initialized — macro carry only, no obstacle test")
        print(f"  controlled joints: {controlled_joint_names}")
        print(f"  tracked obstacles: 0")
        print(_sep())
        return True
    except Exception as exc:
        cumotion_transit["last_error"] = repr(exc)
        print(f"\n{_sep()}\n[cuMotion TRANSIT] initialization failed\n  error: {exc!r}")
        if CUMOTION_FALLBACK_TO_JOINT_INTERP:
            print("  fallback: using joint-interp transit")
        print(_sep())
        return False


def plan_cumotion_to_staging(offset_local, joint_pos):
    if not initialize_cumotion_once():
        return False

    try:
        target_block = transit_staging_block_center()
        target_ori = DOWN_ORI
        target_hand = hand_target_for_block_center(target_block, target_ori, offset_local)
        q_initial = _current_cumotion_q(joint_pos)

        cumotion_transit["world_binding"].get_world_interface().update_world_to_robot_root_transforms(_robot_root_world_poses())
        cumotion_transit["world_binding"].synchronize_transforms()

        print(f"\n{_sep()}")
        print("[cuMotion TRANSIT] planning payload-safe macro carry")
        print("  policy:       carry with DOWN_ORI; pre-insert reorientation happens slowly afterward")
        print(f"  start q:      {np.round(q_initial, 4)}")
        print(f"  target block: {np.round(target_block, 4)}")
        print(f"  target hand:  {np.round(target_hand, 4)}")
        print(f"  target ori:   {np.round(target_ori, 4)}")

        path = cumotion_transit["planner"].plan_to_pose_target(q_initial, target_hand, target_ori)
        if path is None:
            raise RuntimeError("GraphBasedMotionPlanner returned no path")

        trajectory = path.to_minimal_time_joint_trajectory(
            max_velocities=CUMOTION_TRANSIT_MAX_VELOCITIES,
            max_accelerations=CUMOTION_TRANSIT_MAX_ACCELERATIONS,
            robot_joint_space=cumotion_transit["controlled_joint_names"],
            active_joints=cumotion_transit["controlled_joint_names"],
        )
        if trajectory is None:
            raise RuntimeError("Path conversion to trajectory failed")

        duration = getattr(trajectory, "duration", None)
        cumotion_transit.update({
            "active": True,
            "trajectory": trajectory,
            "elapsed": 0.0,
            "frames": 0,
            "done_hold_frames": 0,
            "duration": None if duration is None else float(duration),
            "last_error": None,
        })

        try:
            print(f"  path waypoints: {path.get_waypoints_count()}")
        except Exception:
            pass
        if duration is not None:
            print(f"  trajectory time: {float(duration):.3f} s")
        print(_sep())
        return True
    except Exception as exc:
        cumotion_transit.update({"active": False, "trajectory": None, "last_error": repr(exc)})
        print(f"\n{_sep()}\n[cuMotion TRANSIT] planning failed\n  error: {exc!r}")
        if CUMOTION_FALLBACK_TO_JOINT_INTERP:
            print("  fallback: using joint-interp transit")
        print(_sep())
        return False


def step_cumotion_transit(joint_pos):
    if not cumotion_transit["active"] or cumotion_transit["trajectory"] is None:
        return True

    cumotion_transit["frames"] += 1
    cumotion_transit["elapsed"] += CUMOTION_EXECUTION_DT

    try:
        cumotion_transit["world_binding"].get_world_interface().update_world_to_robot_root_transforms(_robot_root_world_poses())
        cumotion_transit["world_binding"].synchronize_transforms()
    except Exception:
        pass

    duration = cumotion_transit["duration"]
    query_time = min(cumotion_transit["elapsed"], duration) if duration is not None else cumotion_transit["elapsed"]
    target_state = cumotion_transit["trajectory"].get_target_state(query_time)
    if target_state is None or target_state.joints is None or target_state.joints.positions is None:
        print("[cuMotion TRANSIT] trajectory complete")
        cumotion_transit["active"] = False
        return True

    target_positions = np.asarray(target_state.joints.positions, dtype=np.float64).flatten()
    franka.get_articulation_controller().apply_action(_closed_gripper_action(target_positions, joint_pos))

    if duration is not None and cumotion_transit["elapsed"] >= duration:
        cumotion_transit["done_hold_frames"] += 1
        if cumotion_transit["done_hold_frames"] >= CUMOTION_TRANSIT_FINAL_HOLD_FRAMES:
            block_pos = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
            print(f"[cuMotion TRANSIT] final hold complete block={np.round(block_pos, 4)}")
            cumotion_transit["active"] = False
            return True

    if cumotion_transit["frames"] % 120 == 0:
        block_pos = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
        print(f"[cuMotion TRANSIT] frame={cumotion_transit['frames']} t={cumotion_transit['elapsed']:.2f}s block={np.round(block_pos, 4)}")

    if cumotion_transit["frames"] >= CUMOTION_TRANSIT_MAX_EXEC_FRAMES:
        print(f"\n{_sep()}\n[cuMotion TRANSIT] FAILED: execution timed out\n  frames: {cumotion_transit['frames']}\n  elapsed: {cumotion_transit['elapsed']:.3f} s\n{_sep()}")
        cumotion_transit["active"] = False
        return True

    return False


def start_transit_phase(offset_local, joint_pos):
    controller.clear_queue()
    if ENABLE_CUMOTION_TRANSIT and plan_cumotion_to_staging(offset_local, joint_pos):
        return "cumotion"
    if ENABLE_CUMOTION_TRANSIT and not CUMOTION_FALLBACK_TO_JOINT_INTERP:
        return "failed"
    queue_transit_phase()
    return "joint_interp"


# =============================================================================
# GRASP / MEASURE / REPORT
# =============================================================================

def check_grasp():
    fingers = franka.gripper.get_joint_positions()
    if fingers is None:
        print("[GRASP] Cannot read finger positions")
        return False
    f1, f2 = float(fingers[0]), float(fingers[1])
    closed = float(franka.gripper.joint_closed_positions[0])
    ok = f1 >= FINGER_CONTACT_MIN and f2 >= FINGER_CONTACT_MIN

    print(f"\n{_sep()}")
    print("[GRASP CHECK]")
    print(f"  finger_1:      {f1 * 1000:.3f} mm")
    print(f"  finger_2:      {f2 * 1000:.3f} mm")
    print(f"  contact_min:   {FINGER_CONTACT_MIN * 1000:.3f} mm")
    print(f"  total_gap:     {(f1 + f2) * 1000:.3f} mm")
    print(f"  fully_closed:  {closed * 1000:.1f} mm")
    print("  RESULT: ✓  Contact confirmed" if ok else "  RESULT: ✗  Grasp failed")
    print(_sep())
    return ok


def compute_and_log_block_offset():
    global block_rot_local
    block_pos, block_rot, _ = _block_pose_matrix()
    hand_pos, hand_rot = _get_hand_pose()
    offset_world = block_pos - hand_pos
    offset_local = hand_rot.T @ offset_world
    block_rot_local = hand_rot.T @ block_rot

    print(f"\n{_sep()}")
    print("[GRASP OFFSET]")
    print(f"  block_world:    {np.round(block_pos, 4)}")
    print(f"  hand_world:     {np.round(hand_pos, 4)}")
    print(f"  offset_world:   {np.round(offset_world, 4)}")
    print(f"  offset_local:   {np.round(offset_local, 4)}  ← hand frame")
    print(f"  magnitude:      {np.linalg.norm(offset_local) * 1000:.2f} mm")
    print(_sep())
    return offset_local


def check_block_still_held(offset_local, label="HOLD CHECK"):
    block_pos = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    hand_pos, hand_rot = _get_hand_pose()
    est_block = hand_pos + hand_rot @ offset_local
    err = float(np.linalg.norm(block_pos - est_block))
    fingers = franka.gripper.get_joint_positions()

    print(f"\n{_sep()}")
    print(f"[{label}]")
    print(f"  actual_block:      {np.round(block_pos, 4)}")
    print(f"  estimated_block:   {np.round(est_block, 4)}")
    print(f"  block_hold_error:  {err * 1000:.2f} mm")
    if fingers is not None:
        print(f"  finger_1:          {float(fingers[0]) * 1000:.3f} mm")
        print(f"  finger_2:          {float(fingers[1]) * 1000:.3f} mm")
        print(f"  total_gap:         {float(fingers[0] + fingers[1]) * 1000:.3f} mm")
    print("  RESULT: ✓ block still matches hand-frame offset" if err < 0.004 else "  RESULT: ✗ block moved relative to hand")
    print(_sep())
    return err < 0.004


# =============================================================================
# COMMAND QUEUES
# =============================================================================

def queue_grasp_phase():
    controller.clear_queue()
    controller.add_cartesian_waypoint(grasp_approach_position(), DOWN_ORI, pos_tolerance=0.05, label="approach_above")
    controller.add_cartesian_waypoint(grasp_descend_position(), DOWN_ORI, pos_tolerance=0.001, label="descend_to_block")
    controller.add_gripper_command(action="open", wait_frames=30)
    controller.add_gripper_command(action="close", wait_frames=90)


def queue_transit_phase():
    controller.clear_queue()
    staging = transit_staging_block_center()
    controller.add_cartesian_waypoint(transit_lift_position(), DOWN_ORI, pos_tolerance=0.001, hold_gripper=True, label="lift")
    controller.add_cartesian_waypoint(staging, DOWN_ORI, pos_tolerance=0.003, joint_interp=True, joint_steps=260, max_frames=320, hold_gripper=True, label="staging_joint")
    controller.add_cartesian_waypoint(staging, INSERT_ORI, pos_tolerance=0.003, joint_interp=True, joint_steps=420, max_frames=620, hold_gripper=True, label="reorient_joint")


def queue_pre_insert_phase(offset_local):
    controller.clear_queue()
    pre_insert_hand = hand_target_for_block_center(pre_insert_block_center(), INSERT_ORI, offset_local)
    final_hand = hand_target_for_block_center(PORT_POSITION, INSERT_ORI, offset_local)

    print(f"\n{_sep()}")
    print("[INSERT TARGETS]")
    print(f"  pre_insert_block_center: {np.round(pre_insert_block_center(), 4)}")
    print(f"  final_block_center:      {np.round(PORT_POSITION, 4)}")
    print(f"  pre_insert_hand:         {np.round(pre_insert_hand, 4)}")
    print(f"  final_insert_hand:       {np.round(final_hand, 4)}")
    print("  control mode:            align Y/Z, then slow measured X insertion")
    print(_sep())

    controller.add_cartesian_waypoint(pre_insert_hand, INSERT_ORI, pos_tolerance=0.001, joint_interp=True, joint_steps=420, max_frames=620, hold_gripper=True, target_is_hand=True, label="pre_insert_coarse")
    controller.add_cartesian_waypoint(pre_insert_hand, INSERT_ORI, pos_tolerance=0.0005, linear=True, linear_step=0.00025, max_frames=360, hold_gripper=True, target_is_hand=True, label="pre_insert_settle")


# =============================================================================
# INSERT SERVO
# =============================================================================

insert_servo = {}
insert_path_samples = []


def _clamp(value, limit):
    return float(np.clip(value, -limit, limit))


def _insert_direction():
    d = insert_servo.get("x_direction", None)
    if d is None or abs(float(d)) < 1e-9:
        d = float(np.sign(INSERT_AXIS_WORLD[0]))
    return d if abs(d) > 1e-9 else -1.0


def _x_near_band_edge():
    d = _insert_direction()
    return float(PORT_POSITION[0] - d * FINAL_AXIS_TOL)


def _x_far_band_edge():
    d = _insert_direction()
    return float(PORT_POSITION[0] + d * FINAL_AXIS_TOL)


def _x_remaining_to_band(actual_x):
    d = _insert_direction()
    return float(max(0.0, d * (_x_near_band_edge() - float(actual_x))))


def _x_remaining_to_target(actual_x):
    d = _insert_direction()
    return float(max(0.0, d * (float(PORT_POSITION[0]) - float(actual_x))))


def _x_overshoot_distance(actual_x):
    d = _insert_direction()
    return float(max(0.0, d * (float(actual_x) - _x_far_band_edge())))


def _x_in_target_band(actual_x):
    return _x_remaining_to_band(actual_x) <= 0.0 and _x_overshoot_distance(actual_x) <= 0.0


def _x_center_error(actual_x):
    return float(float(actual_x) - float(PORT_POSITION[0]))


def _x_in_center_window(actual_x):
    return abs(_x_center_error(actual_x)) <= CENTER_AXIS_TOL


def _clip_x_to_push_limit(commanded_x):
    d = _insert_direction()
    limit_x = _x_near_band_edge() + d * INSERT_MAX_PUSH_THROUGH
    return float(max(commanded_x, limit_x)) if d < 0.0 else float(min(commanded_x, limit_x))


def _endpoint_status(actual_block_center):
    err = np.asarray(actual_block_center, dtype=np.float64).flatten() - PORT_POSITION
    endpoint_norm = float(np.linalg.norm(err))
    yz_total = float(np.sqrt(err[1] ** 2 + err[2] ** 2))
    overshoot_x = _x_overshoot_distance(actual_block_center[0])
    ok = (
        _x_in_center_window(actual_block_center[0])
        and yz_total <= YZ_TOTAL_TOL
        and endpoint_norm <= FINAL_ENDPOINT_NORM_TOL
        and overshoot_x <= X_BAND_OVERSHOOT_EPS
    )
    return ok, err, endpoint_norm, overshoot_x


def init_insert_servo():
    actual = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    insert_path_samples.clear()
    insert_servo.clear()
    insert_servo.update({
        "mode": "align",
        "align_frames": 0,
        "align_stable_frames": 0,
        "align_integral_y": 0.0,
        "align_integral_z": 0.0,
        "stroke_integral_y": 0.0,
        "stroke_integral_z": 0.0,
        "stroke_paused_for_yz": False,
        "stroke_pause_count": 0,
        "stroke_frames": 0,
        "start_x": float(actual[0]),
        "align_hold_x": float(pre_insert_block_center()[0]),
        "x_direction": float(np.sign(INSERT_AXIS_WORLD[0])),
        "commanded_x": float(actual[0]),
        "max_x_overshoot": 0.0,
        "endpoint_ok": False,
        "insert_ran": False,
        "warned_ik": False,
    })

    print(f"\n{_sep()}")
    print("[INSERT SERVO INIT]")
    print(f"  actual_start_block_center: {np.round(actual, 4)}")
    print(f"  align hold X:              x={pre_insert_block_center()[0]:.4f}")
    print(f"  align target Y/Z:          y={PORT_POSITION[1]:.4f}, z={PORT_POSITION[2]:.4f}")
    print(f"  final X safety band:       [{_x_far_band_edge():.4f}, {_x_near_band_edge():.4f}]")
    print(f"  pre-align tolerance:       radial Y/Z <= {INSERT_ALIGN_TOTAL_TOL * 1000:.3f} mm")
    print(f"  slow insert step:          {INSERT_X_SLOW_STEP * 1000:.3f} mm/frame")
    print(_sep())


def _servo_target_block_center(actual):
    actual = np.asarray(actual, dtype=np.float64).flatten()

    if insert_servo["mode"] == "align":
        desired = np.array([insert_servo["align_hold_x"], PORT_POSITION[1], PORT_POSITION[2]], dtype=np.float64)
        err = desired - actual
        insert_servo["align_integral_y"] = _clamp(insert_servo["align_integral_y"] + float(err[1]), INSERT_ALIGN_YZ_INTEGRAL_LIMIT)
        insert_servo["align_integral_z"] = _clamp(insert_servo["align_integral_z"] + float(err[2]), INSERT_ALIGN_YZ_INTEGRAL_LIMIT)
        target = np.array([
            desired[0] + _clamp(INSERT_ALIGN_X_KP * err[0], INSERT_ALIGN_X_MAX_OVERDRIVE),
            desired[1] + _clamp(INSERT_ALIGN_YZ_KP * err[1] + INSERT_ALIGN_YZ_KI * insert_servo["align_integral_y"], INSERT_ALIGN_YZ_MAX_OVERDRIVE),
            desired[2] + _clamp(INSERT_ALIGN_YZ_KP * err[2] + INSERT_ALIGN_YZ_KI * insert_servo["align_integral_z"], INSERT_ALIGN_YZ_MAX_OVERDRIVE),
        ])
        return target, err

    desired_line = np.array([actual[0], PORT_POSITION[1], PORT_POSITION[2]], dtype=np.float64)
    line_err = desired_line - actual
    yz_total = float(np.sqrt(line_err[1] ** 2 + line_err[2] ** 2))

    paused = bool(insert_servo.get("stroke_paused_for_yz", False))
    if paused and yz_total <= STROKE_YZ_RESUME_TOL:
        paused = False
    elif not paused and yz_total >= STROKE_YZ_PAUSE_TOL:
        paused = True
    insert_servo["stroke_paused_for_yz"] = paused
    if paused:
        insert_servo["stroke_pause_count"] += 1

    d = _insert_direction()
    commanded_x = float(insert_servo.get("commanded_x", actual[0]))
    if not _x_in_target_band(actual[0]) and not paused:
        commanded_x = _clip_x_to_push_limit(commanded_x + d * INSERT_X_SLOW_STEP)
    insert_servo["commanded_x"] = commanded_x

    desired = np.array([commanded_x, PORT_POSITION[1], PORT_POSITION[2]], dtype=np.float64)
    err = desired - actual
    insert_servo["stroke_integral_y"] = _clamp(insert_servo["stroke_integral_y"] + float(err[1]), INSERT_SERVO_YZ_INTEGRAL_LIMIT)
    insert_servo["stroke_integral_z"] = _clamp(insert_servo["stroke_integral_z"] + float(err[2]), INSERT_SERVO_YZ_INTEGRAL_LIMIT)

    target = np.array([
        commanded_x,
        PORT_POSITION[1] + _clamp(INSERT_SERVO_YZ_KP * err[1] + INSERT_SERVO_YZ_KI * insert_servo["stroke_integral_y"], INSERT_SERVO_YZ_MAX_OVERDRIVE),
        PORT_POSITION[2] + _clamp(INSERT_SERVO_YZ_KP * err[2] + INSERT_SERVO_YZ_KI * insert_servo["stroke_integral_z"], INSERT_SERVO_YZ_MAX_OVERDRIVE),
    ])
    return target, err


def insert_servo_action(joint_pos, offset_local):
    actual = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    target_block, _ = _servo_target_block_center(actual)
    target_hand = hand_target_for_block_center(target_block, INSERT_ORI, offset_local)
    action, success = art_kinematics.compute_inverse_kinematics(
        target_position=target_hand,
        target_orientation=INSERT_ORI,
        position_tolerance=0.0002,
        orientation_tolerance=0.03,
    )
    if not success:
        if not insert_servo.get("warned_ik", False):
            print(f"[INSERT SERVO] IK failed once. target_hand={np.round(target_hand, 4)}")
            insert_servo["warned_ik"] = True
        return controller._with_closed_gripper(controller._hold_action(joint_pos.shape[0]), joint_pos.shape[0])
    return controller._with_closed_gripper(action, joint_pos.shape[0])


def sample_insert_path(offset_local):
    hand_pos, hand_rot = _get_hand_pose()
    est_center = hand_pos + hand_rot @ offset_local
    actual_center, actual_rot, actual_quat = _block_pose_matrix()
    long_axis = _block_long_axis_world(actual_rot)
    dot = float(np.clip(np.dot(long_axis, np.array([-1.0, 0.0, 0.0])), -1.0, 1.0))
    axis_angle = float(np.degrees(np.arccos(abs(dot))))
    tilt = float(np.degrees(np.arcsin(np.clip(abs(long_axis[2]), 0.0, 1.0))))
    insert_path_samples.append({
        "estimated_center": est_center.copy(),
        "actual_center": actual_center.copy(),
        "long_axis": long_axis.copy(),
        "quat": actual_quat.copy(),
        "axis_angle_deg": axis_angle,
        "horizontal_tilt_deg": tilt,
    })


def update_insert_servo_state(offset_local):
    actual = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()

    if insert_servo["mode"] == "align":
        insert_servo["align_frames"] += 1
        yz_err = actual[[1, 2]] - PORT_POSITION[[1, 2]]
        yz_norm = float(np.sqrt(yz_err[0] ** 2 + yz_err[1] ** 2))

        if yz_norm <= INSERT_ALIGN_TOTAL_TOL:
            insert_servo["align_stable_frames"] += 1
        else:
            insert_servo["align_stable_frames"] = 0

        if insert_servo["align_frames"] % 300 == 0 and insert_servo["align_stable_frames"] == 0:
            x_err = float(actual[0] - pre_insert_block_center()[0])
            print(
                f"[INSERT SERVO] aligning... frame={insert_servo['align_frames']} "
                f"x_err={x_err * 1000:.3f}mm y_err={yz_err[0] * 1000:.3f}mm "
                f"z_err={yz_err[1] * 1000:.3f}mm yz_total={yz_norm * 1000:.3f}mm "
                f"i_y={insert_servo['align_integral_y']:.4f} i_z={insert_servo['align_integral_z']:.4f}"
            )

        if insert_servo["align_frames"] >= INSERT_ALIGN_MAX_FRAMES and insert_servo["align_stable_frames"] < INSERT_ALIGN_HOLD_FRAMES:
            print(f"\n{_sep()}\n[INSERT SERVO] FAILED: alignment did not reach Y/Z tolerance")
            print(f"  actual_block_center: {np.round(actual, 4)}")
            print(f"  y_error:             {yz_err[0] * 1000:.3f} mm")
            print(f"  z_error:             {yz_err[1] * 1000:.3f} mm")
            print(f"  yz_total_error:      {yz_norm * 1000:.3f} mm")
            print(_sep())
            return True

        if insert_servo["align_stable_frames"] >= INSERT_ALIGN_HOLD_FRAMES:
            insert_servo.update({
                "mode": "stroke",
                "insert_ran": True,
                "stroke_frames": 0,
                "start_x": float(actual[0]),
                "commanded_x": float(actual[0]),
                "stroke_integral_y": 0.0,
                "stroke_integral_z": 0.0,
                "stroke_paused_for_yz": False,
                "stroke_pause_count": 0,
            })
            insert_path_samples.clear()
            print(f"\n{_sep()}")
            print("[INSERT SERVO] Starting measured horizontal stroke")
            print(f"  stroke_start_block_center: {np.round(actual, 4)}")
            print(f"  residual_y_error:          {yz_err[0] * 1000:.3f} mm")
            print(f"  residual_z_error:          {yz_err[1] * 1000:.3f} mm")
            print(f"  residual_yz_total:         {yz_norm * 1000:.3f} mm")
            print(f"  hard safety band:          [{_x_far_band_edge():.4f}, {_x_near_band_edge():.4f}]")
            print(_sep())
        return False

    sample_insert_path(offset_local)
    d = _insert_direction()
    start_x = insert_servo.get("start_x", None)
    if start_x is not None:
        backslide = float(max(0.0, -d * (float(actual[0]) - float(start_x))))
        if backslide >= INSERT_MAX_BACKSLIDE:
            print(f"\n{_sep()}\n[INSERT SERVO] FAILED: actual block moved backward during insertion")
            print(f"  backslide: {backslide * 1000:.3f} mm")
            print(_sep())
            return True

    overshoot = _x_overshoot_distance(actual[0])
    insert_servo["max_x_overshoot"] = max(insert_servo.get("max_x_overshoot", 0.0), overshoot)
    if overshoot > X_BAND_OVERSHOOT_EPS:
        print(f"\n{_sep()}\n[INSERT SERVO] HARD STOP: X left the allowed band")
        print(f"  actual_block_center: {np.round(actual, 4)}")
        print(f"  x_band_overshoot:    {overshoot * 1000:.3f} mm")
        print(_sep())
        return True

    insert_servo["stroke_frames"] += 1
    if insert_servo.get("stroke_paused_for_yz", False) and insert_servo["stroke_frames"] % 120 == 0:
        yz = actual[[1, 2]] - PORT_POSITION[[1, 2]]
        print(
            f"[INSERT SERVO] X paused for Y/Z recovery: frame={insert_servo['stroke_frames']} "
            f"x={actual[0]:.4f} y_err={yz[0] * 1000:.3f}mm z_err={yz[1] * 1000:.3f}mm "
            f"yz_total={float(np.linalg.norm(yz)) * 1000:.3f}mm"
        )

    if _x_in_center_window(actual[0]):
        endpoint_ok, endpoint_err, endpoint_norm, overshoot_now = _endpoint_status(actual)
        print(f"\n{_sep()}")
        print("[INSERT SERVO] Horizontal stroke stopped; endpoint already passes" if endpoint_ok else "[INSERT SERVO] Horizontal stroke stopped near X; endpoint failed")
        print("  reason:                    actual block reached center X target")
        print(f"  pre-seat actual_block:     {np.round(actual, 4)}")
        print(f"  remaining_x_to_band:       {_x_remaining_to_band(actual[0]) * 1000:.3f} mm")
        print(f"  remaining_x_to_target:     {_x_remaining_to_target(actual[0]) * 1000:.3f} mm")
        print(f"  center_x_error:            {_x_center_error(actual[0]) * 1000:.3f} mm")
        print(f"  current_x_band_overshoot:  {overshoot_now * 1000:.3f} mm")
        print(f"  pre-seat endpoint_xyz:     {np.round(endpoint_err * 1000, 3)} mm")
        print(f"  pre-seat endpoint_yz_total:{float(np.linalg.norm(endpoint_err[[1, 2]])) * 1000:.3f} mm")
        print(f"  pre-seat endpoint_error:   {endpoint_norm * 1000:.3f} mm")
        print(f"  stroke_yz_pause_frames:    {int(insert_servo.get('stroke_pause_count', 0))}")
        print(_sep())
        insert_servo["endpoint_ok"] = bool(endpoint_ok)
        return True

    if insert_servo["stroke_frames"] >= INSERT_STROKE_MAX_FRAMES:
        print(f"\n{_sep()}\n[INSERT SERVO] FAILED: stroke timed out before actual X reached center target")
        print(f"  actual_block_center: {np.round(actual, 4)}")
        print(_sep())
        return True

    return False


def measure_insert_error(offset_local):
    hand_pos, hand_rot = _get_hand_pose()
    expected_hand = hand_target_for_block_center(PORT_POSITION, INSERT_ORI, offset_local)
    est_center = hand_pos + hand_rot @ offset_local
    actual = np.asarray(block.get_world_pose()[0], dtype=np.float64).flatten()
    hand_err = float(np.linalg.norm(hand_pos - expected_hand))
    est_err = float(np.linalg.norm(est_center - PORT_POSITION))
    actual_err = float(np.linalg.norm(actual - PORT_POSITION))

    print(f"\n{_sep()}")
    print("[INSERT RESULT]")
    print(f"  hand_pos:                  {np.round(hand_pos, 4)}")
    print(f"  expected_hand_pos:         {np.round(expected_hand, 4)}")
    print(f"  hand_error:                {hand_err * 1000:.2f} mm")
    print(f"  actual_block_center:       {np.round(actual, 4)}")
    print(f"  actual_block_center_error: {actual_err * 1000:.2f} mm  ← main endpoint metric")
    print(f"  est_block_center:          {np.round(est_center, 4)}")
    print(f"  est_block_center_error:    {est_err * 1000:.2f} mm")
    print(f"  port_pos (center):         {np.round(PORT_POSITION, 4)}")
    print(_sep())


def measure_horizontal_insert_path():
    if len(insert_path_samples) < 2:
        print(f"\n{_sep()}\n[HORIZONTAL INSERT PATH] Not enough samples.\n{_sep()}")
        return False

    est = np.array([s["estimated_center"] for s in insert_path_samples], dtype=np.float64)
    actual = np.array([s["actual_center"] for s in insert_path_samples], dtype=np.float64)
    axes = np.array([s["long_axis"] for s in insert_path_samples], dtype=np.float64)
    axis_angles = np.array([s["axis_angle_deg"] for s in insert_path_samples], dtype=np.float64)
    tilts = np.array([s["horizontal_tilt_deg"] for s in insert_path_samples], dtype=np.float64)

    y_err = actual[:, 1] - PORT_POSITION[1]
    z_err = actual[:, 2] - PORT_POSITION[2]
    actual_y_dev = float(np.max(np.abs(y_err)))
    actual_z_dev = float(np.max(np.abs(z_err)))
    actual_yz_dev = float(np.max(np.sqrt(y_err ** 2 + z_err ** 2)))
    dx = np.diff(actual[:, 0])
    max_backtrack = float(np.max(np.maximum(0.0, dx))) if len(dx) else 0.0
    x_monotonic = max_backtrack <= ACTUAL_X_JITTER_TOL
    final_x_in_band = _x_in_target_band(actual[-1, 0])
    max_x_overshoot = float(np.max([_x_overshoot_distance(x) for x in actual[:, 0]]))

    path_ok = actual_yz_dev <= YZ_TOTAL_TOL and x_monotonic and final_x_in_band and max_x_overshoot <= 1e-9
    ori_ok = float(np.max(tilts)) <= 2.0 and float(np.max(axis_angles)) <= 5.0

    print(f"\n{_sep()}")
    print("[HORIZONTAL INSERT PATH - ACTUAL BLOCK]")
    print(f"  samples:                    {len(actual)}")
    print(f"  actual_max_y_deviation:     {actual_y_dev * 1000:.3f} mm")
    print(f"  actual_max_z_deviation:     {actual_z_dev * 1000:.3f} mm")
    print(f"  actual_max_yz_total_offset: {actual_yz_dev * 1000:.3f} mm  (limit={YZ_TOTAL_TOL * 1000:.3f} mm)")
    print(f"  actual_x_monotonic:         {x_monotonic}  (max_backtrack={max_backtrack * 1000:.3f} mm)")
    print(f"  actual_max_x_band_overshoot:{max_x_overshoot * 1000:.3f} mm")
    print(f"  final_x_in_band:            {final_x_in_band}")
    print(f"  actual_start_block_center:  {np.round(actual[0], 4)}")
    print(f"  actual_end_block_center:    {np.round(actual[-1], 4)}")
    print("\n  Estimated/model-based check, for debugging bias only:")
    print(f"  est_start_block_center:     {np.round(est[0], 4)}")
    print(f"  est_end_block_center:       {np.round(est[-1], 4)}")
    print("\n  Orientation / horizontality:")
    print(f"  mean_long_axis_world:       {np.round(np.mean(axes, axis=0), 4)}")
    print(f"  max_axis_angle_to_X:        {float(np.max(axis_angles)):.3f} deg")
    print(f"  max_tilt_out_of_horizontal: {float(np.max(tilts)):.3f} deg")
    print("  PATH RESULT: ✓ actual block path passed" if path_ok else "  PATH RESULT: ✗ actual block path failed")
    print("  ORIENTATION RESULT: ✓ block stayed reasonably horizontal/aligned" if ori_ok else "  ORIENTATION RESULT: ✗ block orientation drifted too much")
    print(_sep())
    return path_ok and ori_ok


# =============================================================================
# PHASE TRANSITIONS
# =============================================================================

def enter_inspection_hold(reason):
    global phase
    print(f"\n[PHASE] INSPECTION HOLD — {reason}")
    print("[PHASE] Simulation left open. Inspect the scene, then press Stop to reset/rerun.")
    controller.clear_queue()
    phase = PHASE_DONE


def finish_transit_phase():
    global phase
    print("\n[PHASE] TRANSIT complete")
    if block_offset_local is not None and not check_block_still_held(block_offset_local, "POST-TRANSIT HOLD CHECK"):
        if AUTO_CLOSE_ON_BLOCK_SLIP:
            return False
        enter_inspection_hold("block slipped during transit")
        return False
    print("[PHASE] → PRE-INSERT ALIGNMENT")
    phase = PHASE_PRE_INSERT
    insert_path_samples.clear()
    queue_pre_insert_phase(block_offset_local)
    return True


def reset_run_state():
    global phase, grasp_attempt, block_offset_local, block_rot_local, payload_attach_frames
    controller.reset()
    kinematics_solver.set_robot_base_pose(*franka.get_world_pose())
    phase = PHASE_GRASP
    grasp_attempt = 0
    block_offset_local = None
    block_rot_local = None
    payload_attach_frames = 0
    insert_path_samples.clear()
    insert_servo.clear()
    queue_grasp_phase()


# =============================================================================
# MAIN LOOP
# =============================================================================

phase = PHASE_GRASP
grasp_attempt = 0
block_offset_local = None
block_rot_local = None
payload_attach_frames = 0
reset_needed = False
run_count = 1

print_run_banner(run_count)
print_task_plan()
queue_grasp_phase()

while simulation_app.is_running():
    world.step(render=True)

    if world.is_stopped() and not reset_needed:
        reset_needed = True

    if not world.is_playing():
        continue

    if reset_needed:
        run_count += 1
        print_run_banner(run_count)
        print_task_plan()
        world.reset()
        reset_run_state()
        reset_needed = False
        continue

    joint_pos = franka.get_joint_positions()
    if joint_pos is None:
        continue

    if phase == PHASE_TRANSIT and cumotion_transit.get("active", False):
        if step_cumotion_transit(joint_pos):
            finish_transit_phase()
        continue

    if phase == PHASE_INSERT and block_offset_local is not None:
        if update_insert_servo_state(block_offset_local):
            print("\n[PHASE] INSERT complete")
            if insert_servo.get("insert_ran", False):
                measure_insert_error(block_offset_local)
                measure_horizontal_insert_path()
            else:
                print("\n[PHASE] Insert stroke never ran; skipping endpoint/path measurement.")

            if HOLD_AFTER_INSERT_FOR_INSPECTION:
                print("\n[PHASE] HOLDING BLOCK — release disabled for inspection.")
                print("[PHASE] Gripper remains closed. Inspect the held pre-release pose in the viewer.")
            phase = PHASE_DONE
            print("\n[PHASE] DONE — press Stop to reset and run again")
        else:
            franka.get_articulation_controller().apply_action(insert_servo_action(joint_pos, block_offset_local))
        continue

    if controller.is_done():
        if phase == PHASE_GRASP:
            print(f"\n[PHASE] GRASP complete (attempt {grasp_attempt + 1})")
            if check_grasp():
                block_offset_local = compute_and_log_block_offset()
                print("[PHASE] → TRANSIT")
                phase = PHASE_TRANSIT
                grasp_attempt = 0
                transit_mode = start_transit_phase(block_offset_local, joint_pos)
                if transit_mode == "failed":
                    if AUTO_CLOSE_ON_CUMOTION_FAILURE:
                        print("[PHASE] cuMotion transit failed and fallback is disabled. HALTING.")
                        break
                    enter_inspection_hold("cuMotion transit failed and fallback is disabled")
                    continue
                print(f"[PHASE] Transit mode: {transit_mode}")
            else:
                grasp_attempt += 1
                if grasp_attempt >= MAX_GRASP_ATTEMPTS:
                    if AUTO_CLOSE_ON_GRASP_FAILURE:
                        print(f"\n[PHASE] Grasp failed {MAX_GRASP_ATTEMPTS} times. HALTING.")
                        break
                    enter_inspection_hold(f"grasp failed {MAX_GRASP_ATTEMPTS} times")
                    continue
                print(f"[PHASE] Retrying grasp ({grasp_attempt}/{MAX_GRASP_ATTEMPTS})")
                queue_grasp_phase()

        elif phase == PHASE_TRANSIT:
            finish_transit_phase()

        elif phase == PHASE_PRE_INSERT:
            print("\n[PHASE] PRE-INSERT setup complete")
            print("[PHASE] → CLOSED-LOOP HORIZONTAL INSERT")
            phase = PHASE_INSERT
            init_insert_servo()

    franka.get_articulation_controller().apply_action(controller.forward(joint_pos))

simulation_app.close()
