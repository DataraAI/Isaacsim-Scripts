from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time
import typing
import numpy as np
import omni.usd

from pxr import Gf, Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy

from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)
from isaacsim.core.prims import Articulation
from isaacsim.core.prims import SingleXFormPrim


def get_world_transform_xform(prim: Usd.Prim) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    """
    Get the local transformation of a prim using omni.usd.get_world_transform_matrix().
    See https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd/omni.usd.get_world_transform_matrix.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    return translation, rotation, scale



# USD_PATH = "C:/Users/aayus/Desktop/Jonathan_Arun_Test/frana_clean.usd"
USD_PATH = r"/home/advaith/Downloads/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
PORT_PRIM_PATH = "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"


FRANKA_PRIM_PATH = "/World/Franka"
EE_FRAME = "panda_hand"
TARGET_PRIM_PATH = "/World/FollowTarget"
TARGET_CUBE_PATH = "/World/FollowTarget/Cube"

# Start from a known reachable point in front of the robot and move only a few centimeters.
TARGET_START_OFFSET = np.array([0.35, 0.0, 0.35], dtype=np.float64)
TARGET_OFFSET_FROM_START = np.array([0.03, 0.0, 0.0], dtype=np.float64)

MOVE_STEPS = 180
SETTLE_STEPS = 60


def wait_for_stage_load(num_updates: int = 120):
    for _ in range(num_updates):
        simulation_app.update()
        time.sleep(0.01)


def create_target_cube(stage, prim_path: str, position: np.ndarray, size: float = 0.05):
    xform_prim = stage.GetPrimAtPath(prim_path)
    if not xform_prim.IsValid():
        xform_prim = stage.DefinePrim(prim_path, "Xform")

    cube = UsdGeom.Cube.Define(stage, TARGET_CUBE_PATH)
    cube.CreateSizeAttr(size)

    xformable = UsdGeom.Xformable(xform_prim)
    translate_ops = [
        op for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*position.tolist()))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))


def set_target_position(stage, prim_path: str, position: np.ndarray):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Target prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    translate_ops = [
        op for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(*position.tolist()))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position.tolist()))


def get_world_position(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    world_tf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(world_tf.ExtractTranslation(), dtype=np.float64)


def clamp_target_to_workspace(target_pos: np.ndarray, base_pos: np.ndarray, max_reach: float = 0.7):
    delta = target_pos - base_pos
    dist = np.linalg.norm(delta)
    if dist <= max_reach or dist < 1e-8:
        return target_pos
    return base_pos + (delta / dist) * max_reach


def step_follow_target(world, robot, ik_solver, lula_solver, stage, target_prim_path: str):
    base_pos, base_quat = robot.get_world_pose()
    base_pos = np.array(base_pos, dtype=np.float64)
    lula_solver.set_robot_base_pose(base_pos, base_quat)

    target_pos = get_world_position(stage, target_prim_path)
    target_pos = clamp_target_to_workspace(
        np.array(target_pos, dtype=np.float64),
        base_pos,
        max_reach=0.7,
    )

    action, success = ik_solver.compute_inverse_kinematics(target_pos)
    if success:
        robot.apply_action(action)
        return True

    print(f"IK failed for target {target_pos}")
    return False


def update(dt: float, target, rmpflow, motion_policy, robot):
    """Drive the Franka with RmpFlow so the end-effector tracks the target prim's pose."""
    base_pos, base_quat = robot.get_world_pose()
    rmpflow.set_robot_base_pose(base_pos, base_quat)

    target_position, target_orientation = target.get_world_pose()
    rmpflow.set_end_effector_target(target_position, target_orientation)

    action = motion_policy.get_next_articulation_action(dt)
    robot.apply_action(action)


def move_target_and_follow(world, robot, ik_solver, lula_solver, stage, target_prim_path, start_pos, end_pos, steps):
    success_count = 0
    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        target_pos = (1.0 - alpha) * start_pos + alpha * end_pos
        set_target_position(stage, target_prim_path, target_pos)
        if step_follow_target(world, robot, ik_solver, lula_solver, stage, target_prim_path):
            success_count += 1
        world.step(render=True)
    print(f"IK successes: {success_count}/{steps}")


def main():
    print(f"Opening stage: {USD_PATH}")
    omni.usd.get_context().open_stage(USD_PATH)
    wait_for_stage_load()
    # add_reference_to_stage(USD_PATH, "/World/Datacenter")

    stage = omni.usd.get_context().get_stage()
    world = World(stage_units_in_meters=1.0)
    world.reset()

    franka = SingleArticulation(prim_path=FRANKA_PRIM_PATH, name="franka")
    new_position = np.array([34.0 - 4.0, -100.0 + 10.0, 140.0 + 10.0])
    new_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
    Articulation(FRANKA_PRIM_PATH).set_world_poses(positions=new_position, orientations=new_orientation)
    franka.initialize()

    # config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    # lula_solver = LulaKinematicsSolver(**config)
    config = interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
    rmpflow = RmpFlow(**config)
    # print("Available frames:", lula_solver.get_all_frame_names())

    # ik_solver = ArticulationKinematicsSolver(franka, lula_solver, EE_FRAME)
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow)

    for _ in range(20):
        world.step(render=True)

    base_pos, _ = franka.get_world_pose()
    base_pos = np.array(base_pos, dtype=np.float64)
    print("Robot base world position:", base_pos)

    # target_start = base_pos + TARGET_START_OFFSET
    # target_end = target_start + TARGET_OFFSET_FROM_START
    port_prim = omni.usd.get_context().get_stage().GetPrimAtPath(PORT_PRIM_PATH)
    port_world_position = get_world_transform_xform(port_prim)[0]
    target_start = port_world_position
    target_end = port_world_position + np.array([0, 0, 2])

    print("Target start:", target_start)
    print("Target end:", target_end)

    create_target_cube(stage, TARGET_PRIM_PATH, target_start)
    target = SingleXFormPrim(prim_path=TARGET_PRIM_PATH)
    target.initialize()

    physics_dt = world.get_physics_dt() if hasattr(world, "get_physics_dt") else (1.0 / 60.0)

    while simulation_app.is_running():
        update(physics_dt, target, rmpflow, articulation_rmpflow, franka)
        world.step(render=True)

    # for _ in range(SETTLE_STEPS):
    #     step_follow_target(world, franka, ik_solver, lula_solver, stage, TARGET_PRIM_PATH)
    #     world.step(render=True)

    # print("Moving target...")
    # move_target_and_follow(
    #     world,
    #     franka,
    #     ik_solver,
    #     lula_solver,
    #     stage,
    #     TARGET_PRIM_PATH,
    #     target_start,
    #     target_end,
    #     MOVE_STEPS,
    # )

    # print("Done. Leaving sim open.")
    # while simulation_app.is_running():
    #     step_follow_target(world, franka, ik_solver, lula_solver, stage, TARGET_PRIM_PATH)
    #     world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        simulation_app.close()
