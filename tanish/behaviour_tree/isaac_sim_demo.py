"""Runnable Isaac Sim integration for generated task-intelligence JSON.

Run with Isaac Sim's Python launcher, not the system Python. The bundled smoke
test is used when ``--json`` is omitted.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Generated task-intelligence JSON file")
    parser.add_argument("--headless", action="store_true", help="Run without an Isaac window")
    parser.add_argument("--max-frames", type=int, default=5000, help="Fail after this many frames")
    parser.add_argument(
        "--robot-usd",
        type=Path,
        help="Local franka.usd path (also accepted through ISAACSIM_FRANKA_USD)",
    )
    parser.add_argument(
        "--initial-fact",
        action="append",
        default=[],
        help="Add a true starting precondition; repeat for multiple facts",
    )
    return parser.parse_args()


ARGS = _parse_args()

# SimulationApp must be created before importing any other Isaac/Omniverse API.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
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
from pxr import Sdf, UsdLux, UsdPhysics, UsdShade, PhysxSchema

THIS_DIR = Path(__file__).resolve().parent
TANISH_DIR = THIS_DIR.parent
CONTROLLER_DIR = TANISH_DIR.parent / "detailedInsertion" / "cable"
for path in (TANISH_DIR, CONTROLLER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from behaviour_tree import BehaviourTreeRuntime, Status, load_task_intelligence
from behaviour_tree.isaac_adapters import controller_primitive, function_primitive
from franka_motion_controller import FrankaMotionController


DOWN_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_POSITIONS = {
    "navigate_to_workspace": np.array([0.45, 0.0, 0.50]),
    "grasp_object": np.array([0.50, 0.0, 0.20]),
    "grasp_tool": np.array([0.50, 0.0, 0.20]),
    "manipulate_object": np.array([0.40, -0.25, 0.25]),
    "trace_linear_path": np.array([0.45, 0.20, 0.35]),
    "execute_subtask": np.array([0.45, 0.0, 0.45]),
}
TARGET_PRIM_PATH = "/World/BehaviourTreeBlock"
BLOCK_SPAWN = np.array([0.50, 0.0, 0.025], dtype=np.float64)
BLOCK_SCALE = np.array([0.025, 0.008, 0.050], dtype=np.float64)
PLACE_CENTER = np.array([0.40, -0.25, 0.025], dtype=np.float64)
GRASP_APPROACH_Z = 0.20
GRASP_DESCEND_OFFSET = 0.015
FINGER_CONTACT_MIN = BLOCK_SCALE[1] * 0.5 - 0.0005


def _position(context, default_name: str) -> np.ndarray:
    raw = context.step.inputs.get("position", DEFAULT_POSITIONS[default_name])
    position = np.asarray(raw, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{context.step.name}: inputs.position must contain three finite numbers")
    return position


def _orientation(context) -> np.ndarray:
    raw = context.step.inputs.get("orientation_wxyz", DOWN_ORIENTATION)
    orientation = np.asarray(raw, dtype=np.float64).reshape(-1)
    if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
        raise ValueError(f"{context.step.name}: inputs.orientation_wxyz must contain four finite numbers")
    norm = float(np.linalg.norm(orientation))
    if norm < 1e-9:
        raise ValueError(f"{context.step.name}: orientation quaternion cannot be zero")
    return orientation / norm


def queue_move(context, primitive_name: str | None = None) -> None:
    controller = context.services["motion_controller"]
    name = primitive_name or context.step.primitive
    controller.add_cartesian_waypoint(
        position=_position(context, name),
        orientation=_orientation(context),
        max_frames=int(context.step.inputs.get("max_frames", 600)),
        pos_tolerance=float(context.step.inputs.get("position_tolerance", 0.01)),
        target_is_hand=True,
        label=context.step.name,
    )


def queue_grasp(context) -> None:
    controller = context.services["motion_controller"]
    block_position = np.asarray(
        context.services["block"].get_world_pose()[0], dtype=np.float64
    ).reshape(-1)[:3]
    orientation = _orientation(context)
    hover = np.array([block_position[0], block_position[1], GRASP_APPROACH_Z])
    target = block_position + np.array([0.0, 0.0, GRASP_DESCEND_OFFSET])
    lift = np.array([block_position[0], block_position[1], GRASP_APPROACH_Z])
    controller.add_cartesian_waypoint(
        hover, orientation, max_frames=600, pos_tolerance=0.015,
        label=f"{context.step.name}: hover",
    )
    controller.add_cartesian_waypoint(
        target, orientation, max_frames=600, pos_tolerance=0.001,
        label=f"{context.step.name}: descend",
    )
    controller.add_gripper_command(action="open", wait_frames=30)
    controller.add_gripper_command(action="close", wait_frames=45)
    controller.add_cartesian_waypoint(
        lift, orientation, max_frames=600, pos_tolerance=0.003,
        hold_gripper=True, label=f"{context.step.name}: lift",
    )


def queue_manipulate(context) -> None:
    controller = context.services["motion_controller"]
    orientation = _orientation(context)
    hover = np.array([PLACE_CENTER[0], PLACE_CENTER[1], 0.20])
    target = PLACE_CENTER + np.array([0.0, 0.0, GRASP_DESCEND_OFFSET])
    controller.add_cartesian_waypoint(
        hover, orientation, max_frames=600, pos_tolerance=0.015,
        hold_gripper=True, label=f"{context.step.name}: transit",
    )
    controller.add_cartesian_waypoint(
        target, orientation, max_frames=600, pos_tolerance=0.003,
        hold_gripper=True, label=f"{context.step.name}: lower",
    )
    controller.add_gripper_command(action="open", wait_frames=75)
    controller.add_cartesian_waypoint(
        hover, orientation, max_frames=600, pos_tolerance=0.015,
        label=f"{context.step.name}: retreat",
    )


def target_exists(context) -> bool:
    attempts = context.services.setdefault("perception_attempts", {})
    count = attempts.get(context.step.name, 0) + 1
    attempts[context.step.name] = count
    if context.step.inputs.get("fail_first_attempt") and count == 1:
        print("[BT PERCEPTION] simulated transient miss on attempt 1 (demonstrates Retry)")
        return False
    stage = context.services["stage"]
    prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    exists = bool(prim and prim.IsValid())
    print(f"[BT PERCEPTION] target prim {'found' if exists else 'missing'}: {TARGET_PRIM_PATH}")
    return exists


def check_physical_grasp(context) -> bool:
    block_position = np.asarray(
        context.services["block"].get_world_pose()[0], dtype=np.float64
    ).reshape(-1)[:3]
    fingers = np.asarray(
        context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
    ).reshape(-1)
    contact = fingers.size >= 2 and bool(np.all(fingers[:2] >= FINGER_CONTACT_MIN))
    lifted = float(block_position[2]) >= 0.10
    print(
        f"[BT GRASP CHECK] block_z={block_position[2]:.4f}m "
        f"fingers_mm={np.round(fingers[:2] * 1000.0, 2)} contact={contact} lifted={lifted}"
    )
    return contact and lifted


def block_at_goal(context) -> bool:
    block_position = np.asarray(
        context.services["block"].get_world_pose()[0], dtype=np.float64
    ).reshape(-1)[:3]
    xy_error = float(np.linalg.norm(block_position[:2] - PLACE_CENTER[:2]))
    resting = 0.015 <= float(block_position[2]) <= 0.08
    passed = xy_error <= 0.05 and resting
    print(
        f"[BT PLACE CHECK] block={np.round(block_position, 4)} "
        f"xy_error_mm={xy_error * 1000.0:.1f} resting={resting} passed={passed}"
    )
    return passed


def inspect_workspace(context) -> bool:
    robot = context.services["robot"]
    joints = robot.get_joint_positions()
    valid = joints is not None and bool(np.all(np.isfinite(np.asarray(joints))))
    print(f"[BT INSPECTION] robot joint state is {'valid' if valid else 'invalid'}")
    return valid


def _set_attr_safe(api, create_name: str, get_name: str, value) -> bool:
    try:
        getter = getattr(api, get_name, None)
        attr = getter() if callable(getter) else None
        if not attr:
            creator = getattr(api, create_name)
            try:
                attr = creator(value)
            except TypeError:
                attr = creator()
        attr.Set(value)
        return True
    except Exception:
        return False


def configure_grasp_physics(stage) -> None:
    """Apply the physical grasp tuning used by the working block demo."""

    material = UsdShade.Material.Define(stage, Sdf.Path("/World/GripPhysicsMaterial"))
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    _set_attr_safe(material_api, "CreateStaticFrictionAttr", "GetStaticFrictionAttr", 3.0)
    _set_attr_safe(material_api, "CreateDynamicFrictionAttr", "GetDynamicFrictionAttr", 2.5)
    _set_attr_safe(material_api, "CreateRestitutionAttr", "GetRestitutionAttr", 0.0)

    for path in (
        TARGET_PRIM_PATH,
        "/World/Franka/panda_leftfinger",
        "/World/Franka/panda_rightfinger",
        "/World/Franka/panda_leftfinger/geometry",
        "/World/Franka/panda_rightfinger/geometry",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            try:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            except Exception as exc:
                print(f"[BT PHYSICS] material bind warning for {path}: {exc}")

    block_prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    if block_prim and block_prim.IsValid():
        mass_api = UsdPhysics.MassAPI.Apply(block_prim)
        _set_attr_safe(mass_api, "CreateMassAttr", "GetMassAttr", 0.002)
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(block_prim)
        _set_attr_safe(
            rigid_api,
            "CreateSolverPositionIterationCountAttr",
            "GetSolverPositionIterationCountAttr",
            48,
        )
        _set_attr_safe(
            rigid_api,
            "CreateSolverVelocityIterationCountAttr",
            "GetSolverVelocityIterationCountAttr",
            12,
        )

    for joint_path in (
        "/World/Franka/panda_finger_joint1",
        "/World/Franka/panda_finger_joint2",
    ):
        prim = stage.GetPrimAtPath(joint_path)
        if prim and prim.IsValid():
            drive = UsdPhysics.DriveAPI.Get(prim, "linear") or UsdPhysics.DriveAPI.Apply(prim, "linear")
            _set_attr_safe(drive, "CreateStiffnessAttr", "GetStiffnessAttr", 2.0e5)
            _set_attr_safe(drive, "CreateDampingAttr", "GetDampingAttr", 2.0e4)
            _set_attr_safe(drive, "CreateMaxForceAttr", "GetMaxForceAttr", 5000.0)


def resolve_franka_usd() -> str:
    """Prefer an explicit/local robot asset and use the asset server last."""

    configured = ARGS.robot_usd or (
        Path(os.environ["ISAACSIM_FRANKA_USD"]).expanduser()
        if os.environ.get("ISAACSIM_FRANKA_USD") else None
    )
    local_candidates = [
        configured,
        Path.home() / "src/Re3Sim/re3sim/Collected_franka/franka.usd",
    ]
    for candidate in local_candidates:
        if candidate is not None and candidate.is_file():
            resolved = str(candidate.resolve())
            print(f"[BT DEMO] Using local Franka USD: {resolved}")
            return resolved

    assets_root = get_assets_root_path()
    if assets_root:
        return assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    raise RuntimeError(
        "Could not find a Franka asset. Pass --robot-usd /absolute/path/franka.usd "
        "or set ISAACSIM_FRANKA_USD."
    )


def build_scene():
    world = World(stage_units_in_meters=1.0)
    world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    stage = omni.usd.get_context().get_stage()

    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    light.CreateIntensityAttr(700.0)

    world.scene.add(FixedCuboid(
        name="ground",
        position=np.array([0.0, 0.0, -0.005]),
        prim_path="/World/Ground",
        scale=np.array([10.0, 10.0, 0.01]),
        size=1.0,
        color=np.array([0.18, 0.18, 0.18]),
    ))
    world.scene.add(FixedCuboid(
        name="place_goal",
        position=np.array([PLACE_CENTER[0], PLACE_CENTER[1], 0.003]),
        prim_path="/World/PlaceGoal",
        scale=np.array([0.12, 0.12, 0.006]),
        size=1.0,
        color=np.array([0.15, 0.8, 0.2]),
    ))
    block = world.scene.add(DynamicCuboid(
        name="behaviour_tree_block",
        position=BLOCK_SPAWN,
        prim_path=TARGET_PRIM_PATH,
        scale=BLOCK_SCALE,
        size=1.0,
        color=np.array([0.15, 0.35, 1.0]),
    ))

    robot_prim = add_reference_to_stage(
        usd_path=resolve_franka_usd(),
        prim_path="/World/Franka",
    )
    robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    physics = robot_prim.GetVariantSet("Physics")
    variants = list(physics.GetVariantNames())
    if variants:
        physics.SetVariantSelection(next((v for v in variants if v.lower() == "physx"), variants[0]))

    gripper = ParallelGripper(
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.001, 0.001]),
        action_deltas=np.array([0.02, 0.02]),
    )
    franka = world.scene.add(SingleManipulator(
        prim_path="/World/Franka",
        name="behaviour_tree_franka",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
    ))
    configure_grasp_physics(stage)
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    world.reset()

    config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    kinematics = LulaKinematicsSolver(**config)
    trajectory_generator = LulaTaskSpaceTrajectoryGenerator(**config)
    articulation_kinematics = ArticulationKinematicsSolver(franka, kinematics, "panda_hand")
    base_position, base_orientation = franka.get_world_pose()
    kinematics.set_robot_base_pose(base_position, base_orientation)
    controller = FrankaMotionController(
        name="behaviour_tree_controller",
        robot_articulation=franka,
        task_traj_gen=trajectory_generator,
        art_kinematics=articulation_kinematics,
        gripper=franka.gripper,
        tool_offset=0.05,
        physics_dt=1.0 / 120.0,
        debug=True,
    )
    return world, stage, franka, controller, block


def main() -> int:
    json_path = (ARGS.json or THIS_DIR / "demo_task_intelligence.json").expanduser().resolve()
    print(f"[BT DEMO] Loading: {json_path}")
    payload = load_task_intelligence(json_path)
    world, stage, franka, controller, block = build_scene()

    registry = {
        "navigate_to_workspace": controller_primitive(queue_move),
        "perceive_objects": function_primitive(target_exists),
        "grasp_object": controller_primitive(queue_grasp, validate=check_physical_grasp),
        "grasp_tool": controller_primitive(queue_grasp, validate=check_physical_grasp),
        "manipulate_object": controller_primitive(queue_manipulate, validate=block_at_goal),
        "trace_linear_path": controller_primitive(queue_move),
        "inspect_workspace": function_primitive(inspect_workspace),
        "verify_block_at_goal": function_primitive(block_at_goal),
        "execute_subtask": controller_primitive(queue_move),
    }
    tree = BehaviourTreeRuntime(
        payload,
        registry,
        initial_facts={
            "robot_ready",
            "robot_localized",
            "workspace_map_loaded",
            "camera_ready",
            *ARGS.initial_fact,
        },
        services={
            "world": world,
            "stage": stage,
            "robot": franka,
            "block": block,
            "motion_controller": controller,
            "articulation_controller": franka.get_articulation_controller(),
        },
    )

    print("\n[BT STRUCTURE]\n" + tree.render_tree() + "\n")

    world.play()
    warmup_frames = 30
    frame = 0
    result = Status.RUNNING
    while simulation_app.is_running() and frame < max(1, ARGS.max_frames):
        world.step(render=not ARGS.headless)
        if not world.is_playing():
            continue
        frame += 1
        if frame <= warmup_frames:
            continue
        result = tree.tick()
        if result in (Status.SUCCESS, Status.FAILURE):
            break

    if result is Status.SUCCESS:
        print(f"[BT DEMO PASS] Completed {tree.step_index} generated steps in {frame} frames")
        exit_code = 0
    elif result is Status.FAILURE:
        print(f"[BT DEMO FAIL] {tree.feedback}")
        exit_code = 1
    else:
        print(f"[BT DEMO FAIL] Timed out after {frame} frames; {tree.feedback}")
        exit_code = 2

    # Leave the completed pose visible briefly in GUI mode.
    for _ in range(120 if not ARGS.headless else 1):
        if not simulation_app.is_running():
            break
        world.step(render=not ARGS.headless)
    simulation_app.close()
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        simulation_app.close()
        raise
