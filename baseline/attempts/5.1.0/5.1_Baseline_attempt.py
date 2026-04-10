from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time
import typing
import numpy as np
import omni.usd

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

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
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
import os

# Desired end-effector orientation in world frame (matches RMPflow tutorial target convention).
# See https://docs.isaacsim.omniverse.nvidia.com/5.1.0/manipulators/manipulators_rmpflow.html
EE_TARGET_EULER_RAD = np.array([0.0, np.pi, 0.0], dtype=np.float64)


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



USD_PATH = "C:/Users/aayus/Desktop/Jonathan_Arun_Test/frana_clean.usd"
# USD_PATH = r"/home/advaith/Downloads/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
# USD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"
# PORT_PRIM_PATH = "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
# PORT_PRIM_PATH = "/World/Equipment/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"

num_quads = 4
num_pairs = 4
num_conn_a = 2


PORT_BASE_PRIM_PATH = (
    "/World/Equipment/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base"
    "/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01"
)

PORT_PRIM_PATH_LIST = []

for q in range(1, num_quads+1):
    for p in range(1, num_pairs+1):
        for a in range(1, num_conn_a+1):
            suffix = f"/Connector_Quad_{q:02d}/Connector_Pair_{p:02d}/QSFP_DD_Connector_A_{a:02d}"
            PORT_PRIM_PATH_LIST.append(PORT_BASE_PRIM_PATH + suffix)

PORT_PRIM_PATH = PORT_PRIM_PATH_LIST[0]


# None = try FRANKA_PRIM_PATH_CANDIDATES, then scan for an articulation root matching franka/panda.
# Set to an explicit path if auto-detect picks the wrong robot or you want a fixed path.
FRANKA_PRIM_PATH: typing.Optional[str] = None
FRANKA_PRIM_PATH_CANDIDATES = (
    "/World/Franka",
    "/World/Franka_01",
    "/panda",
    "/World/panda",
)
# Must match a link name in lula_franka_gen.urdf (same as end_effector_frame_name below).
EE_FRAME = "panda_hand"
TARGET_PRIM_PATH = "/World/Cube"
# Child mesh under TARGET_PRIM_PATH (must not equal the xform path).
TARGET_CUBE_PATH = "/World/Cube/visual_cube"
TARGET_CUBE_MATERIAL_PATH = "/World/Looks/TargetCubeRed"

# Start from a known reachable point in front of the robot and move only a few centimeters.
TARGET_START_OFFSET = np.array([0.35, 0.0, 0.35], dtype=np.float64)
TARGET_OFFSET_FROM_START = np.array([0.03, 0.0, 0.0], dtype=np.float64)

MOVE_STEPS = 180
SETTLE_STEPS = 60
# Time to interpolate the cube from one port goal to the next (see goal_position_for_port_index).
PORT_SEGMENT_DURATION_SEC = 5.0


def resolve_franka_prim_path(stage: Usd.Stage) -> str:
    """Return the USD path to the Franka articulation root for this stage."""
    if FRANKA_PRIM_PATH is not None:
        prim = stage.GetPrimAtPath(FRANKA_PRIM_PATH)
        if not prim.IsValid():
            raise RuntimeError(
                f"FRANKA_PRIM_PATH is {FRANKA_PRIM_PATH!r} but that prim does not exist. "
                "Set it to None for auto-detect, or copy the path from the Stage tree."
            )
        return FRANKA_PRIM_PATH

    for path in FRANKA_PRIM_PATH_CANDIDATES:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return path

    roots: typing.List[str] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))

    hints = ("franka", "panda")
    hinted = [r for r in roots if any(h in r.lower() for h in hints)]
    if hinted:
        return sorted(hinted, key=len)[0]

    if len(roots) == 1:
        return roots[0]

    if roots:
        raise RuntimeError(
            "Could not auto-detect Franka: multiple articulation roots and none match "
            "'franka' or 'panda' in the path. Found:\n  "
            + "\n  ".join(roots)
            + "\nSet FRANKA_PRIM_PATH to the robot root path."
        )

    raise RuntimeError(
        "No Franka articulation prim found. Tried:\n  "
        + "\n  ".join(FRANKA_PRIM_PATH_CANDIDATES)
        + "\nNo prims with UsdPhysics.ArticulationRootAPI on the stage. "
        "Confirm the stage contains the robot, then set FRANKA_PRIM_PATH."
    )


def wait_for_stage_load(num_updates: int = 120):
    for _ in range(num_updates):
        simulation_app.update()
        time.sleep(0.01)


def as_gf_vec3d(position: typing.Union[np.ndarray, Gf.Vec3d, typing.Sequence[float]]) -> Gf.Vec3d:
    """Build Gf.Vec3d from numpy, Gf.Vec3d, or any length-3 sequence (e.g. port pose from get_world_transform_xform)."""
    if isinstance(position, Gf.Vec3d):
        return position
    if isinstance(position, np.ndarray):
        p = np.asarray(position, dtype=np.float64).reshape(-1)[:3]
        return Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
    return Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))


def create_target_cube(stage: Usd.Stage) -> None:
    """Place target xform at PORT_PRIM_PATH world position; red cube with edge length 2 under TARGET_CUBE_PATH."""
    print(f"Creating target cube at port: {PORT_PRIM_PATH}")
    port_prim = stage.GetPrimAtPath(PORT_PRIM_PATH)
    if not port_prim.IsValid():
        raise RuntimeError(f"Port prim not found: {PORT_PRIM_PATH}")

    world_translation = get_world_transform_xform(port_prim)[0]
    v = as_gf_vec3d(world_translation)

    xform_prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    if not xform_prim.IsValid():
        xform_prim = stage.DefinePrim(TARGET_PRIM_PATH, "Xform")

    xformable = UsdGeom.Xformable(xform_prim)
    translate_ops = [
        op for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if translate_ops:
        translate_ops[0].Set(v)
    else:
        xformable.AddTranslateOp().Set(v)

    cube = UsdGeom.Cube.Define(stage, TARGET_CUBE_PATH)
    cube.CreateSizeAttr(2.0)

    material = UsdShade.Material.Define(stage, TARGET_CUBE_MATERIAL_PATH)
    shader = UsdShade.Shader.Define(stage, f"{TARGET_CUBE_MATERIAL_PATH}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    print(f"Target xform {TARGET_PRIM_PATH} at port {PORT_PRIM_PATH} world translation {v}, cube size 2, material red")


def set_target_position(stage, prim_path: str, position: typing.Union[np.ndarray, Gf.Vec3d, typing.Sequence[float]]):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Target prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    translate_ops = [
        op for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    v = as_gf_vec3d(position)
    if translate_ops:
        translate_ops[0].Set(v)
    else:
        xformable.AddTranslateOp().Set(v)


def get_port_world_translation_np(stage: Usd.Stage, port_prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(port_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Port prim not found: {port_prim_path}")
    t = get_world_transform_xform(prim)[0]
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


def goal_position_for_port_index(stage: Usd.Stage, port_index: int) -> np.ndarray:
    """World position for the cube at PORT_PRIM_PATH_LIST[port_index]: port pose plus z-offset by parity."""
    base = get_port_world_translation_np(stage, PORT_PRIM_PATH_LIST[port_index])
    z_off = -2.0 if (port_index % 2 == 0) else 2.0
    return base + np.array([0.0, 0.0, z_off], dtype=np.float64)


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


def update(
    dt: float,
    target,
    rmpflow,
    motion_policy,
    robot,
    stage: Usd.Stage,
    motion_state: typing.MutableMapping[str, typing.Any],
    world: World,
) -> None:
    """Move the cube through port goals only while the sim is playing; RmpFlow tracks the target xform."""
    playing = world.is_playing()
    n_ports = len(PORT_PRIM_PATH_LIST)
    if playing and n_ports >= 1:
        duration = max(float(motion_state.get("segment_duration", PORT_SEGMENT_DURATION_SEC)), 1e-6)
        if n_ports == 1:
            cube_pos = goal_position_for_port_index(stage, 0)
            set_target_position(stage, TARGET_PRIM_PATH, cube_pos)
        else:
            motion_state["elapsed"] = float(motion_state.get("elapsed", 0.0)) + dt
            seg = int(motion_state.get("segment_idx", 0))
            num_segments = n_ports - 1
            seg = seg % num_segments

            p0 = goal_position_for_port_index(stage, seg)
            p1 = goal_position_for_port_index(stage, seg + 1)
            alpha = min(1.0, motion_state["elapsed"] / duration)
            cube_pos = (1.0 - alpha) * p0 + alpha * p1
            set_target_position(stage, TARGET_PRIM_PATH, cube_pos)

            if alpha >= 1.0:
                motion_state["elapsed"] = 0.0
                motion_state["segment_idx"] = (seg + 1) % num_segments

    # Same order as docs: end-effector target and base pose before get_next_articulation_action.
    # Tutorial sets orientation with euler_angles_to_quats([0, pi, 0]) so position targets are not
    # fighting a mismatched default orientation. Position = cube center in world frame.
    cube_position = get_world_position(stage, TARGET_CUBE_PATH)
    target_orientation = euler_angles_to_quats(EE_TARGET_EULER_RAD)
    rmpflow.set_end_effector_target(cube_position, target_orientation)

    rmpflow.update_world()

    base_pos, base_quat = robot.get_world_pose()
    rmpflow.set_robot_base_pose(base_pos, base_quat)

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

    franka_path = resolve_franka_prim_path(stage)
    print(f"Using Franka articulation prim: {franka_path}")

    franka = SingleArticulation(prim_path=franka_path, name="franka")
    # new_position = np.array([34.0 - 4.0, -100.0 + 10.0, 140.0 + 10.0])
    # new_position = np.array([20.0, -270.0, 140.0])
    # new_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
    # Articulation(franka_path).set_world_poses(positions=new_position, orientations=new_orientation)
    # franka.set_world_poses(positions=new_position, orientations=new_orientation)
    franka.initialize()

    # config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    # lula_solver = LulaKinematicsSolver(**config)
    # config = interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
    # rmpflow = RmpFlow(**config)
    # print("Available frames:", lula_solver.get_all_frame_names())
    mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

    # Initialize RmpFlow (see manipulators_rmpflow.html): EE frame must match the URDF link used as TCP.
    rmpflow = RmpFlow(
        robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
        urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
        rmpflow_config_path=rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml",
        end_effector_frame_name=EE_FRAME,
        maximum_substep_size=0.00334,
    )

    # ik_solver = ArticulationKinematicsSolver(franka, lula_solver, EE_FRAME)
    articulation_rmpflow = ArticulationMotionPolicy(franka, rmpflow)

    for _ in range(20):
        world.step(render=True)

    base_pos, _ = franka.get_world_pose()
    base_pos = np.array(base_pos, dtype=np.float64)
    print("Robot base world position:", base_pos)

    # target_start = base_pos + TARGET_START_OFFSET
    # target_end = target_start + TARGET_OFFSET_FROM_START
    create_target_cube(stage)
    target = SingleXFormPrim(prim_path=TARGET_PRIM_PATH)
    target.initialize()

    physics_dt = world.get_physics_dt() if hasattr(world, "get_physics_dt") else (1.0 / 60.0)

    port_motion_state: typing.Dict[str, typing.Any] = {"segment_idx": 0, "elapsed": 0.0}

    while simulation_app.is_running():
        update(physics_dt, target, rmpflow, articulation_rmpflow, franka, stage, port_motion_state, world)
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