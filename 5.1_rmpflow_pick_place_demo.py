from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time
import numpy as np
import omni.usd

from pxr import Gf, Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    ArticulationMotionPolicy,
    LulaKinematicsSolver,
    RmpFlow,
)
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_lula_kinematics_solver_config,
    load_supported_motion_policy_config,
)


USD_PATH = "C:/Users/aayus/Desktop/Jonathan_Arun_Test/frana_clean.usd"
FRANKA_PRIM_PATH = "/World/Franka_01"

# Optional: set this to a real rack port prim to place at that location.
PLACE_PORT_PRIM_PATH = None

EE_FRAME = "panda_hand"
TARGET_PRIM_PATH = "/World/RmpFlowTarget"
TARGET_CUBE_PATH = "/World/RmpFlowTarget/Cube"
PAYLOAD_PRIM_PATH = "/World/DemoPayload"
PAYLOAD_CUBE_PATH = "/World/DemoPayload/Cube"
PLACE_MARKER_PRIM_PATH = "/World/PlaceMarker"
PLACE_MARKER_CUBE_PATH = "/World/PlaceMarker/Cube"

FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

# Relative demo positions if PLACE_PORT_PRIM_PATH is not set.
PICK_OFFSET = np.array([0.45, 0.00, 0.03], dtype=np.float64)
PLACE_OFFSET = np.array([0.55, -0.20, 0.20], dtype=np.float64)
APPROACH_OFFSET = np.array([0.0, 0.0, 0.15], dtype=np.float64)
ATTACHED_PAYLOAD_OFFSET = np.array([0.0, 0.0, -0.03], dtype=np.float64)

TARGET_TOLERANCE = 0.02
GRASP_HOLD_STEPS = 30
RELEASE_HOLD_STEPS = 30


def wait_for_stage_load(num_updates: int = 120):
    for _ in range(num_updates):
        simulation_app.update()
        time.sleep(0.01)


def ensure_xform_cube(stage, prim_path: str, cube_path: str, position: np.ndarray, size: float):
    xform_prim = stage.GetPrimAtPath(prim_path)
    if not xform_prim.IsValid():
        xform_prim = stage.DefinePrim(prim_path, "Xform")

    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(size)
    set_xform_translation(stage, prim_path, position)


def set_xform_translation(stage, prim_path: str, position: np.ndarray):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    translate_ops = [
        op for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    vec = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    if translate_ops:
        translate_ops[0].Set(vec)
    else:
        xformable.AddTranslateOp().Set(vec)


def get_world_position(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(world_tf.ExtractTranslation(), dtype=np.float64)


def set_gripper(robot: SingleArticulation, width: float):
    finger_indices = np.array([robot.get_joint_index(name) for name in FINGER_JOINT_NAMES], dtype=np.int64)
    action = ArticulationAction(
        joint_positions=np.array([width, width], dtype=np.float64),
        joint_indices=finger_indices,
    )
    robot.apply_action(action)


def phase_sequence(pick_position: np.ndarray, place_position: np.ndarray):
    return [
        {"name": "pre_pick", "target": pick_position + APPROACH_OFFSET, "gripper": GRIPPER_OPEN, "attach": False},
        {"name": "pick", "target": pick_position, "gripper": GRIPPER_OPEN, "attach": False},
        {
            "name": "grasp",
            "target": pick_position,
            "gripper": GRIPPER_CLOSED,
            "attach": True,
            "hold_steps": GRASP_HOLD_STEPS,
        },
        {"name": "lift", "target": pick_position + APPROACH_OFFSET, "gripper": GRIPPER_CLOSED, "attach": True},
        {"name": "pre_place", "target": place_position + APPROACH_OFFSET, "gripper": GRIPPER_CLOSED, "attach": True},
        {"name": "place", "target": place_position, "gripper": GRIPPER_CLOSED, "attach": True},
        {
            "name": "release",
            "target": place_position,
            "gripper": GRIPPER_OPEN,
            "attach": False,
            "hold_steps": RELEASE_HOLD_STEPS,
        },
        {"name": "retreat", "target": place_position + APPROACH_OFFSET, "gripper": GRIPPER_OPEN, "attach": False},
    ]


def main():
    print(f"Opening stage: {USD_PATH}")
    omni.usd.get_context().open_stage(USD_PATH)
    wait_for_stage_load()

    stage = omni.usd.get_context().get_stage()
    world = World(stage_units_in_meters=1.0)
    world.reset()

    franka = SingleArticulation(prim_path=FRANKA_PRIM_PATH, name="franka")
    franka.initialize()

    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
    rmpflow = RmpFlow(**rmp_config)
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow)

    ik_config = load_supported_lula_kinematics_solver_config("Franka")
    ik_solver = ArticulationKinematicsSolver(franka, LulaKinematicsSolver(**ik_config), EE_FRAME)

    base_pos, base_quat = franka.get_world_pose()
    base_pos = np.array(base_pos, dtype=np.float64)

    pick_position = base_pos + PICK_OFFSET
    if PLACE_PORT_PRIM_PATH:
        place_position = get_world_position(stage, PLACE_PORT_PRIM_PATH)
    else:
        place_position = base_pos + PLACE_OFFSET

    target_orientation = euler_angles_to_quats(np.array([0.0, np.pi, 0.0], dtype=np.float64))

    print("Robot base world position:", base_pos)
    print("Pick position:", pick_position)
    print("Place position:", place_position)

    ensure_xform_cube(stage, TARGET_PRIM_PATH, TARGET_CUBE_PATH, pick_position + APPROACH_OFFSET, 0.04)
    ensure_xform_cube(stage, PAYLOAD_PRIM_PATH, PAYLOAD_CUBE_PATH, pick_position, 0.03)
    ensure_xform_cube(stage, PLACE_MARKER_PRIM_PATH, PLACE_MARKER_CUBE_PATH, place_position, 0.035)

    phases = phase_sequence(pick_position, place_position)
    phase_index = 0
    hold_counter = 0
    attached = False
    released_payload_position = pick_position.copy()
    done = False

    for _ in range(30):
        world.step(render=True)

    while simulation_app.is_running():
        step = world.get_physics_dt()
        phase = phases[phase_index]

        base_pos, base_quat = franka.get_world_pose()
        base_pos = np.array(base_pos, dtype=np.float64)
        rmpflow.set_robot_base_pose(base_pos, base_quat)

        desired_target = np.array(phase["target"], dtype=np.float64)
        set_xform_translation(stage, TARGET_PRIM_PATH, desired_target)
        rmpflow.set_end_effector_target(desired_target, target_orientation)

        arm_action = articulation_rmpflow.get_next_articulation_action(step)
        franka.apply_action(arm_action)
        set_gripper(franka, phase["gripper"])

        ee_position, _ = ik_solver.compute_end_effector_pose()
        ee_position = np.array(ee_position, dtype=np.float64)

        if phase.get("attach", False):
            attached = True
            released_payload_position = ee_position + ATTACHED_PAYLOAD_OFFSET
        elif phase["name"] == "release":
            attached = False
            released_payload_position = place_position.copy()

        if attached:
            set_xform_translation(stage, PAYLOAD_PRIM_PATH, ee_position + ATTACHED_PAYLOAD_OFFSET)
        else:
            set_xform_translation(stage, PAYLOAD_PRIM_PATH, released_payload_position)

        distance = np.linalg.norm(ee_position - desired_target)
        if distance < TARGET_TOLERANCE:
            hold_counter += 1
        else:
            hold_counter = 0

        if hold_counter >= phase.get("hold_steps", 1):
            print(f"Completed phase: {phase['name']}")
            phase_index += 1
            hold_counter = 0
            if phase_index >= len(phases):
                done = True
                phase_index = len(phases) - 1

        world.step(render=True)

        if done:
            print("Pick-and-place sequence complete. Leaving sim open.")
            break

    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        simulation_app.close()
