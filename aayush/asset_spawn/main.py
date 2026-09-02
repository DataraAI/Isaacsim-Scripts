"""Load DataHall, work table, UR10e, cable support, and network cable in Isaac Sim.

    /home/aayush/isaacsim/python.sh aayush/asset_spawn/main.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin/Assets/Datacenter/"
    "Facilities/Stages/Data_Hall/"
    "DataHall_Single_Rack_3x_Ethernet_Rows_2x_BakedScale1_4x1x_Switches.usd"
)
NETWORK_CABLE_USD = (
    "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd"
)

DATAHALL_PRIM_PATH = "/World/DataHall"
WORK_TABLE_PATH = "/World/WorkTable"
UR10E_MOUNT_PATH = "/World/UR10eMount"
UR10E_PRIM_PATH = f"{UR10E_MOUNT_PATH}/ur10e"
CABLE_ROOT_PATH = "/World/NetworkCable"
CABLE_PLUG_PATH = "/World/NetworkCable/E_crystal_head1_45"
CABLE_SUPPORT_PATH = "/World/CableSupportBlock"

TABLE_POSITION = np.array([0.42, -0.6, 1.0], dtype=np.float64)
TABLE_ORIENTATION_EULER_DEG = np.array([0.0, 0.0, -90.0], dtype=np.float64)
TABLE_SIZE_XYZ = np.array([1.40, 0.90, 0.05], dtype=np.float64)
TABLE_COLOR = np.array([0.55, 0.35, 0.18], dtype=np.float64)

UR10E_POSITION = np.array([0.18, -0.085, 1.03], dtype=np.float64)

CABLE_PLUG_TARGET_XY = np.array([0.45, -0.35], dtype=np.float64)
CABLE_SUPPORT_HEIGHT_M = 0.040
CABLE_SUPPORT_COLOR = np.array([0.85, 0.45, 0.15], dtype=np.float64)
CABLE_PLUG_CLEARANCE_M = 0.004
CABLE_SUPPORT_XY_MARGIN_M = 0.012

UR10E_USD_LOCAL = Path.home() / "isaacsim_assets/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
ROBOTIQ_USD_LOCAL = Path.home() / "isaacsim_assets/Isaac/Robots/Robotiq/2F-140/Robotiq_2F_140.usd"
ROBOTIQ_USD_FALLBACK = Path(
    "/home/aayush/isaacsim/exts/isaacsim.asset.transformer.rules/data/tests/ur10e/"
    "Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd"
)
GRIPPER_ASSEMBLY_NAMESPACE = "Gripper"
GRIPPER_VARIANT_NAME = "Robotiq_2F_140"
GRIPPER_PRIM_NAME = "robotiq_2f_140"
# Tutorial Robot Assembler (URDF-imported gripper) uses robotiq_arg2f_base_link.
# Isaac bundled Gripper variant payloads use robotiq_base_link instead.
GRIPPER_ATTACH_LINK_CANDIDATES = (
    "robotiq_arg2f_base_link",
    "robotiq_base_link",
)
WRIST_LINK_NAME = "wrist_3_link"

simulation_app = SimulationApp({"headless": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

stage = omni.usd.get_context().get_stage()
if not stage.GetPrimAtPath("/World").IsValid():
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

world = World(stage_units_in_meters=1.0)
world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)


def wait_for_stage_loading() -> None:
    usd_context = omni.usd.get_context()
    stable_frames = 0
    for _ in range(3600):
        simulation_app.update()
        try:
            _message, files_loaded, total_files = usd_context.get_stage_loading_status()
            still_loading = bool(files_loaded or total_files)
        except AttributeError:
            still_loading = False
        if still_loading:
            stable_frames = 0
        else:
            stable_frames += 1
            if stable_frames >= 15:
                return
        time.sleep(0.01)


def find_descendant(root_path: str, name: str) -> str | None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    wanted = name.lower()
    for prim in Usd.PrimRange(root):
        if prim.GetName().lower() == wanted:
            return str(prim.GetPath())
    return None


def find_gripper_attach_link(root_path: str) -> str | None:
    for link_name in GRIPPER_ATTACH_LINK_CANDIDATES:
        link_path = find_descendant(root_path, link_name)
        if link_path:
            return link_path
    return None


def prim_bbox(prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = np.array(box.GetMin(), dtype=np.float64)
    maximum = np.array(box.GetMax(), dtype=np.float64)
    return minimum, maximum, 0.5 * (minimum + maximum)


def world_translation(prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    pose = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return np.array(pose.ExtractTranslation(), dtype=np.float64)


def set_world_translate(prim_path: str, translation: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    x, y, z = (float(translation[0]), float(translation[1]), float(translation[2]))
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            attr = op.GetAttr()
            if attr and str(attr.GetTypeName()) == "float3":
                op.Set(Gf.Vec3f(x, y, z))
            else:
                op.Set(Gf.Vec3d(x, y, z))
            return
    xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))


def table_top_z() -> float:
    return float(TABLE_POSITION[2] + 0.5 * TABLE_SIZE_XYZ[2])


def enable_gpu_dynamics() -> None:
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()]
    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(True).Set(True)
        api.CreateBroadphaseTypeAttr("GPU").Set("GPU")
        api.CreateSolverTypeAttr("TGS").Set("TGS")
    for scene in SimulationManager.get_physics_scenes():
        scene.set_enabled_gpu_dynamics(True)
    print("[SPAWN] PhysX GPU dynamics enabled (required for soft cable)")


def disable_gravity_on_robot(root_path: str) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
        count += 1
    print(f"[SPAWN] Disabled gravity on {count} UR10e rigid-body link(s)")


def resolve_robotiq_usd() -> str:
    for candidate in (ROBOTIQ_USD_LOCAL, ROBOTIQ_USD_FALLBACK):
        if candidate.is_file():
            path = str(candidate.resolve())
            print(f"[SPAWN] Using Robotiq 2F-140 USD: {path}")
            return path
    assets_root = get_assets_root_path()
    if assets_root:
        return assets_root + "/Isaac/Robots/Robotiq/2F-140/Robotiq_2F_140.usd"
    raise RuntimeError("Could not find Robotiq 2F-140 USD.")


def select_gripper_variant(robot_prim) -> str | None:
    if not robot_prim.HasVariantSets():
        return None
    variant_set = robot_prim.GetVariantSet(GRIPPER_ASSEMBLY_NAMESPACE)
    if not variant_set:
        return None
    names = list(variant_set.GetVariantNames())
    if not names:
        return None
    lowered = {name.lower(): name for name in names}
    for candidate in (
        GRIPPER_VARIANT_NAME,
        "Robotiq_2f_140",
        "robotiq_2f_140",
        "2F_140",
        "2F-140",
    ):
        if candidate.lower() in lowered:
            chosen = lowered[candidate.lower()]
            variant_set.SetVariantSelection(chosen)
            print(f"[SPAWN] {GRIPPER_ASSEMBLY_NAMESPACE} variant -> {chosen}")
            return chosen
    return None


def assemble_robotiq_gripper(robot_prim_path: str, wrist_path: str) -> str:
    """Attach Robotiq 2F-140 at wrist_3_link via Robot Assembler."""

    gripper_prim_path = f"{robot_prim_path}/{GRIPPER_PRIM_NAME}"
    add_reference_to_stage(usd_path=resolve_robotiq_usd(), prim_path=gripper_prim_path)
    wait_for_stage_loading()

    gripper_base_path = find_gripper_attach_link(gripper_prim_path)
    if not gripper_base_path:
        raise RuntimeError(
            f"Could not find gripper attach link {GRIPPER_ATTACH_LINK_CANDIDATES} "
            f"under {gripper_prim_path}"
        )

    from isaacsim.robot_setup.assembler import RobotAssembler

    assembler = RobotAssembler()
    assembler.begin_assembly(
        stage,
        robot_prim_path,
        wrist_path,
        gripper_prim_path,
        gripper_base_path,
        GRIPPER_ASSEMBLY_NAMESPACE,
        GRIPPER_VARIANT_NAME,
    )
    assembler.assemble()
    assembler.finish_assemble()
    for _ in range(90):
        simulation_app.update()
        time.sleep(0.01)
    wait_for_stage_loading()

    gripper_base_path = find_gripper_attach_link(robot_prim_path) or gripper_base_path
    attach_link_name = gripper_base_path.rsplit("/", 1)[-1]
    print(
        f"[SPAWN] Robot Assembler: {WRIST_LINK_NAME} -> {attach_link_name} "
        f"(namespace={GRIPPER_ASSEMBLY_NAMESPACE})"
    )
    return gripper_base_path


def attach_robotiq_gripper(robot_prim_path: str) -> tuple[str, str]:
    wrist_path = find_descendant(robot_prim_path, WRIST_LINK_NAME)
    if not wrist_path:
        raise RuntimeError(f"Could not find {WRIST_LINK_NAME} under {robot_prim_path}")

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    variant_chosen = select_gripper_variant(robot_prim)
    if variant_chosen:
        wait_for_stage_loading()
        for _ in range(120):
            gripper_base = find_gripper_attach_link(robot_prim_path)
            if gripper_base:
                attach_link_name = gripper_base.rsplit("/", 1)[-1]
                print(
                    f"[SPAWN] Gripper loaded from {GRIPPER_ASSEMBLY_NAMESPACE} variant "
                    f"({variant_chosen}) on {WRIST_LINK_NAME}; attach link={attach_link_name}"
                )
                return gripper_base, wrist_path
            simulation_app.update()
            time.sleep(0.01)
        raise RuntimeError(
            f"Gripper variant {variant_chosen} selected but no attach link "
            f"{GRIPPER_ATTACH_LINK_CANDIDATES} found under {robot_prim_path}"
        )

    gripper_base = assemble_robotiq_gripper(robot_prim_path, wrist_path)
    return gripper_base, wrist_path


def resolve_ur10e_usd() -> str:
    if UR10E_USD_LOCAL.is_file():
        path = str(UR10E_USD_LOCAL.resolve())
        print(f"[SPAWN] Using local UR10e USD: {path}")
        return path
    assets_root = get_assets_root_path()
    if assets_root:
        return assets_root + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
    raise RuntimeError("Could not find UR10e.usd.")


def select_physics_variant(robot_prim) -> None:
    if not robot_prim.HasVariantSets():
        return
    variant_set = robot_prim.GetVariantSet("Physics")
    names = list(variant_set.GetVariantNames())
    if not names:
        return
    chosen = next((name for name in names if name.lower() == "physx"), names[0])
    variant_set.SetVariantSelection(chosen)
    print(f"[SPAWN] UR10e Physics variant -> {chosen}")


def set_mount_world_translate(mount_path: str, translation: np.ndarray) -> None:
    """Author an explicit world translate on a mount Xform (shown in the Property panel)."""

    if stage.GetPrimAtPath(mount_path).IsValid():
        stage.RemovePrim(Sdf.Path(mount_path))
    xform = UsdGeom.Xform.Define(stage, Sdf.Path(mount_path))
    xformable = UsdGeom.Xformable(xform.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
    )


def apply_mount_translate(mount_path: str, translation: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(mount_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing mount prim: {mount_path}")
    value = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(value)
            return
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(value)


def finalize_ur10e_placement(robot, mount_path: str, base_link_path: str) -> None:
    """Place the robot via mount world Xform; fall back to base_link on the table."""

    apply_mount_translate(mount_path, UR10E_POSITION)
    simulation_app.update()

    mount_world = world_translation(mount_path)
    root_world = world_translation(UR10E_PRIM_PATH)
    base_min, _, base_center = prim_bbox(base_link_path)
    table_top = table_top_z()

    if float(base_min[2]) < table_top - 0.05:
        mount_pos = UR10E_POSITION.copy()
        mount_pos[2] += table_top - float(base_min[2])
        apply_mount_translate(mount_path, mount_pos)
        simulation_app.update()
        world.reset()
        base_min, _, base_center = prim_bbox(base_link_path)
        mount_world = world_translation(mount_path)
        root_world = world_translation(UR10E_PRIM_PATH)
        print(
            f"[SPAWN] UR10e mount raised so base_link sits on table: "
            f"{np.round(mount_pos, 4).tolist()}"
        )

    position, orientation = robot.get_world_pose()
    robot.set_world_pose(position=np.asarray(position, dtype=np.float64), orientation=orientation)

    print(
        f"[SPAWN] UR10e mount translate={np.round(mount_world, 4)} "
        f"articulation_root={np.round(root_world, 4)} "
        f"base_link_center={np.round(base_center, 4)} "
        f"base_link_bottom_z={base_min[2]:.4f} table_top={table_top:.4f}"
    )


def load_cable_reference() -> str:
    if stage.GetPrimAtPath(CABLE_ROOT_PATH).IsValid():
        stage.RemovePrim(Sdf.Path(CABLE_ROOT_PATH))
    print(f"[SPAWN] Referencing network cable: {NETWORK_CABLE_USD}")
    add_reference_to_stage(usd_path=NETWORK_CABLE_USD, prim_path=CABLE_ROOT_PATH)
    wait_for_stage_loading()
    plug_path = find_descendant(CABLE_ROOT_PATH, "E_crystal_head1_45") or CABLE_PLUG_PATH
    if not stage.GetPrimAtPath(plug_path).IsValid():
        raise RuntimeError(f"Cable is missing crystal head at {plug_path}")
    return plug_path


def place_cable_plug_at_xy(plug_path: str, plug_xy: np.ndarray, support_top_z: float) -> None:
    plug_min, _, plug_center = prim_bbox(plug_path)
    root_position = world_translation(CABLE_ROOT_PATH)
    desired_plug_min_z = support_top_z + CABLE_PLUG_CLEARANCE_M
    delta = np.array(
        [
            float(plug_xy[0]) - plug_center[0],
            float(plug_xy[1]) - plug_center[1],
            desired_plug_min_z - plug_min[2],
        ],
        dtype=np.float64,
    )
    set_world_translate(CABLE_ROOT_PATH, root_position + delta)
    wait_for_stage_loading()


def cable_support_size_from_bbox() -> np.ndarray:
    root_min, root_max, _ = prim_bbox(CABLE_ROOT_PATH)
    dims = root_max - root_min
    size_xy = dims[:2] + 2.0 * CABLE_SUPPORT_XY_MARGIN_M
    return np.array(
        [
            max(0.08, float(size_xy[0])),
            max(0.035, float(size_xy[1])),
            CABLE_SUPPORT_HEIGHT_M,
        ],
        dtype=np.float64,
    )


def create_cable_support_block(size_xyz: np.ndarray) -> float:
    if stage.GetPrimAtPath(CABLE_SUPPORT_PATH).IsValid():
        stage.RemovePrim(Sdf.Path(CABLE_SUPPORT_PATH))

    top_z = table_top_z()
    center = np.array(
        [
            CABLE_PLUG_TARGET_XY[0],
            CABLE_PLUG_TARGET_XY[1],
            top_z + 0.5 * size_xyz[2],
        ],
        dtype=np.float64,
    )
    world.scene.add(
        FixedCuboid(
            name="cable_support_block",
            prim_path=CABLE_SUPPORT_PATH,
            position=center,
            scale=size_xyz,
            size=1.0,
            color=CABLE_SUPPORT_COLOR,
            visible=True,
        )
    )
    block_top_z = float(top_z + size_xyz[2])
    print(
        f"[SPAWN] Cable support at {np.round(center, 4)} "
        f"size={size_xyz.tolist()} top_z={block_top_z:.4f}"
    )
    return block_top_z


def log_cable_plug(plug_path: str, block_top_z: float) -> None:
    plug_min, _, plug_center = prim_bbox(plug_path)
    print(
        f"[SPAWN] Cable plug center={np.round(plug_center, 4)} "
        f"plug_min_z={plug_min[2]:.4f} block_top={block_top_z:.4f}"
    )


light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
light.CreateIntensityAttr(800.0)

print(f"[SPAWN] Referencing DataHall: {DATAHALL_USD}")
add_reference_to_stage(usd_path=DATAHALL_USD, prim_path=DATAHALL_PRIM_PATH)
wait_for_stage_loading()
print(f"[SPAWN] DataHall loaded at {DATAHALL_PRIM_PATH}")

table_orientation = euler_angles_to_quats(np.radians(TABLE_ORIENTATION_EULER_DEG))
world.scene.add(
    FixedCuboid(
        name="work_table",
        prim_path=WORK_TABLE_PATH,
        position=TABLE_POSITION,
        orientation=table_orientation,
        scale=TABLE_SIZE_XYZ,
        size=1.0,
        color=TABLE_COLOR,
        visible=True,
    )
)
print(
    f"[SPAWN] Work table position={TABLE_POSITION.tolist()} "
    f"euler_deg={TABLE_ORIENTATION_EULER_DEG.tolist()} top_z={table_top_z():.4f}"
)

enable_gpu_dynamics()

plug_path = load_cable_reference()
place_cable_plug_at_xy(plug_path, CABLE_PLUG_TARGET_XY, table_top_z())
support_size = cable_support_size_from_bbox()
print(f"[SPAWN] Cable support size from cable bbox: {support_size.tolist()}")

block_top_z = create_cable_support_block(support_size)
place_cable_plug_at_xy(plug_path, CABLE_PLUG_TARGET_XY, block_top_z)
log_cable_plug(plug_path, block_top_z)

set_mount_world_translate(UR10E_MOUNT_PATH, UR10E_POSITION)

robot_prim = add_reference_to_stage(usd_path=resolve_ur10e_usd(), prim_path=UR10E_PRIM_PATH)
select_physics_variant(robot_prim)
wait_for_stage_loading()

base_link_path = find_descendant(UR10E_PRIM_PATH, "base_link")
if not base_link_path:
    raise RuntimeError(f"Could not find base_link under {UR10E_PRIM_PATH}")

end_effector_path, wrist_path = attach_robotiq_gripper(UR10E_PRIM_PATH)

ur10e_robot = world.scene.add(
    SingleManipulator(
        prim_path=UR10E_PRIM_PATH,
        name="ur10e",
        end_effector_prim_path=end_effector_path,
    )
)

world.reset()
finalize_ur10e_placement(ur10e_robot, UR10E_MOUNT_PATH, base_link_path)
disable_gravity_on_robot(UR10E_PRIM_PATH)

place_cable_plug_at_xy(plug_path, CABLE_PLUG_TARGET_XY, block_top_z)
log_cable_plug(plug_path, block_top_z)

print(f"[SPAWN] UR10e wrist={wrist_path} end_effector={end_effector_path}")
print("[SPAWN] Press Play to run physics. Isaac Sim will stay open until you quit.")

simulation_app.update()
while simulation_app.is_running():
    world.step(render=True)
