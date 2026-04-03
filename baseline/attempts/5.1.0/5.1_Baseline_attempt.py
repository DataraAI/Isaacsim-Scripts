from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time
import numpy as np
import omni.usd

from pxr import Gf, Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)


USD_PATH = "C:/Users/aayus/Desktop/Jonathan_Arun_Test/frana_clean.usd"

FRANKA_PRIM_PATH = "/World/Franka_01"
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

    stage = omni.usd.get_context().get_stage()
    world = World(stage_units_in_meters=1.0)
    world.reset()

    franka = SingleArticulation(prim_path=FRANKA_PRIM_PATH, name="franka")
    franka.initialize()

    config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula_solver = LulaKinematicsSolver(**config)
    print("Available frames:", lula_solver.get_all_frame_names())

    ik_solver = ArticulationKinematicsSolver(franka, lula_solver, EE_FRAME)

    for _ in range(20):
        world.step(render=True)

    base_pos, _ = franka.get_world_pose()
    base_pos = np.array(base_pos, dtype=np.float64)
    print("Robot base world position:", base_pos)

    target_start = base_pos + TARGET_START_OFFSET
    target_end = target_start + TARGET_OFFSET_FROM_START

    print("Target start:", target_start)
    print("Target end:", target_end)

    create_target_cube(stage, TARGET_PRIM_PATH, target_start)

    for _ in range(SETTLE_STEPS):
        step_follow_target(world, franka, ik_solver, lula_solver, stage, TARGET_PRIM_PATH)
        world.step(render=True)

    print("Moving target...")
    move_target_and_follow(
        world,
        franka,
        ik_solver,
        lula_solver,
        stage,
        TARGET_PRIM_PATH,
        target_start,
        target_end,
        MOVE_STEPS,
    )

    print("Done. Leaving sim open.")
    while simulation_app.is_running():
        step_follow_target(world, franka, ik_solver, lula_solver, stage, TARGET_PRIM_PATH)
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
