"""Importable builder for the asset_spawn layout.

Requires an already-created ``SimulationApp``. Call::

    bundle = build_asset_spawn_scene(simulation_app)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
CABLE_SUPPORT_PATH = "/World/CableSupportBlock"
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
TRACKED_PLUG_PRIM_PATH = f"{NETWORK_CABLE_ROOT_PATH}/E_crystal_head1_45"
OTHER_PLUG_PRIM_PATH = f"{NETWORK_CABLE_ROOT_PATH}/E_crystal_head2_39"

TABLE_POSITION = np.array([0.42, -0.6, 1.0], dtype=np.float64)
TABLE_ORIENTATION_EULER_DEG = np.array([0.0, 0.0, -90.0], dtype=np.float64)
TABLE_SIZE_XYZ = np.array([1.40, 0.90, 0.05], dtype=np.float64)
TABLE_COLOR = np.array([0.55, 0.35, 0.18], dtype=np.float64)

UR10E_POSITION = np.array([0.18, -0.085, 1.03], dtype=np.float64)

CABLE_SUPPORT_XY = np.array([0.435, -0.35], dtype=np.float64)
# Y/Z kept; X is set from the span of both crystal heads at spawn.
CABLE_SUPPORT_SIZE_M = np.array([0.22, 0.05, 0.10], dtype=np.float64)
CABLE_SUPPORT_COLOR = np.array([0.85, 0.45, 0.15], dtype=np.float64)
CABLE_SUPPORT_XY_MARGIN_M = 0.01
# Absolute world Z for /World/NetworkCable after XY seating on the block.
CABLE_ROOT_Z = 1.13
# Finger-clearance U-notch under crystal head 45 / E_part006_44 (top channel
# with a floor for balance). Must sit on the head45 end, never head39.
NOTCH_EXTRA_X_M = 0.030  # total extra beyond head X span (noticeable gap)
NOTCH_DEPTH_Z_M = 0.045  # how deep the channel is from the top (~finger room)
NOTCH_END_SHOULDER_M = 0.028  # keep a visible wall past the notch when part is at an end
NOTCH_MIN_SEGMENT_X_M = 0.012
GRASP_PART_NAME = "E_part006_44"

# Debug markers for RJ45 insert / pre-insert offset (visual only; toggle Visibility).
PORT_CONTACTS_PATH = (
    "/World/DataHall/Network_Switches/AS4610_01_1x_Grid/Upper_Right/AS4610_inst/"
    "AS4610_01/Switch/Net_12_Pack_no_LED_Component_01/RJ45_Group01/CopperContacts/"
    "Group_14345"
)
PORT_PIN_A_NAME = "Copper_Pin_Component_1907"
PORT_PIN_B_NAME = "Copper_Pin_Component_1910"
PORT_APPROACH_X_OFFSET_M = 0.02
PORT_DEBUG_MARKER_ROOT = "/World/DebugPortMarkers"
PORT_OFFSET_MARKER_PATH = f"{PORT_DEBUG_MARKER_ROOT}/PortApproachOffset"
PORT_INSERT_MARKER_PATH = f"{PORT_DEBUG_MARKER_ROOT}/PortInsert"
PORT_DEBUG_MARKER_SCALE = 0.01
# Green tip vias for the lift→offset carry (mirrors ur10e_1x_cable_insertion path).
PORT_MANEUVER_MARKER_ROOT = f"{PORT_DEBUG_MARKER_ROOT}/ManeuverVias"
PORT_MANEUVER_MARKER_SCALE = 0.05
PORT_APPROACH_VIA_FRACTIONS = (0.35, 0.60, 0.82, 0.95)
PORT_APPROACH_VIA_Z_CLEARANCE_M = 0.04
# Match ur10e_1x_cable_insertion lift-tip estimate from E_part006_44.
DEBUG_GRASP_X_OFFSET_M = 0.0
DEBUG_GRASP_DESCEND_CLEARANCE_M = -0.003
DEBUG_GRASP_LIFT_CLEARANCE_M = 0.12
# Final world X translations for the support walls (Isaac translate ops).
# Tuned for CABLE_SUPPORT_XY[0] == 0.435 (shifted −15 mm from the prior 0.45 layout).
SUPPORT_LEFT_X_M = 0.335
SUPPORT_RIGHT_X_M = 0.60621

UR10E_USD_LOCAL = Path.home() / "isaacsim_assets/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
ROBOTIQ_USD_LOCAL = Path.home() / "isaacsim_assets/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85.usd"
ROBOTIQ_USD_FALLBACK = Path(
    "/home/aayush/isaacsim/exts/isaacsim.asset.transformer.rules/data/tests/ur10e/"
    "Robotiq/2F-85/Robotiq_2F_85.usda"
)
GRIPPER_ASSEMBLY_NAMESPACE = "Gripper"
GRIPPER_VARIANT_NAME = "Robotiq_2F_85"
GRIPPER_PRIM_NAME = "Robotiq_2F_85"
# Bundled 2F-85 Gripper variant uses base_link under the Robotiq_2F_85 prim.
GRIPPER_ATTACH_LINK_CANDIDATES = (
    "robotiq_arg2f_base_link",
    "robotiq_base_link",
    "base_link",
)
WRIST_LINK_NAME = "wrist_3_link"

# Runtime globals set by build_asset_spawn_scene.
simulation_app = None  # type: ignore
stage = None  # type: ignore
world = None  # type: ignore

# Populated on first build (Isaac imports need SimulationApp first).
omni = None  # type: ignore
VisualCuboid = None  # type: ignore
SimulationManager = None  # type: ignore
euler_angles_to_quats = None  # type: ignore
add_reference_to_stage = None  # type: ignore
SingleManipulator = None  # type: ignore
get_assets_root_path = None  # type: ignore
Gf = Sdf = Usd = UsdGeom = UsdLux = UsdPhysics = PhysxSchema = None  # type: ignore

@dataclass
class AssetSpawnBundle:
    world: Any
    stage: Any
    robot: Any
    end_effector_path: str
    wrist_path: str
    base_link_path: str
    path45: str
    path39: str
    block_top_z: float


def _init_isaac_modules(app) -> None:
    global simulation_app, stage, world
    global omni, VisualCuboid, SimulationManager, euler_angles_to_quats
    global add_reference_to_stage, SingleManipulator, get_assets_root_path
    global Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

    import omni.usd as omni_usd
    import omni.timeline as omni_timeline
    import types
    omni = types.SimpleNamespace(usd=omni_usd, timeline=omni_timeline)
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import VisualCuboid as _VC
    from isaacsim.core.simulation_manager import SimulationManager as _SM
    from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats as _e2q
    from isaacsim.core.utils.stage import add_reference_to_stage as _arts
    from isaacsim.robot.manipulators import SingleManipulator as _Man
    from isaacsim.storage.native import get_assets_root_path as _gar
    from pxr import Gf as _Gf, Sdf as _Sdf, Usd as _Usd, UsdGeom as _UG
    from pxr import UsdLux as _UL, UsdPhysics as _UP, PhysxSchema as _PS

    VisualCuboid, SimulationManager = _VC, _SM
    euler_angles_to_quats = _e2q
    add_reference_to_stage, SingleManipulator, get_assets_root_path = _arts, _Man, _gar
    Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema = _Gf, _Sdf, _Usd, _UG, _UL, _UP, _PS

    simulation_app = app
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
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    for link_name in GRIPPER_ATTACH_LINK_CANDIDATES:
        wanted = link_name.lower()
        for prim in Usd.PrimRange(root):
            if prim.GetName().lower() != wanted:
                continue
            path = str(prim.GetPath())
            # Avoid matching the UR10e arm base_link when looking for 2F-85.
            if wanted == "base_link":
                lowered = path.lower()
                if not any(token in lowered for token in ("robotiq", "2f_85", "2f-85")):
                    continue
            return path
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
    print("[SPAWN] PhysX GPU dynamics enabled")


def enable_datahall_static_collisions(
    root_path: str = DATAHALL_PRIM_PATH, *, approximation_shape: str = "none"
) -> int:
    """Author static mesh colliders under DataHall so the arm cannot pass through.

    Uses triangle-mesh approximation (``none``) for static geometry — same pattern
    as ``detailedInsertion/datahall/collision_setup.py``.
    """

    from omni.physx.scripts import utils as physx_utils

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[SPAWN] DataHall collision skipped: missing {root_path}")
        return 0

    count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetMetadata("hide_in_stage_window"):
            continue
        if prim.GetAttribute("omni:no_collision"):
            continue
        is_mesh = prim.IsA(UsdGeom.Mesh)
        is_solid = (
            is_mesh
            or prim.IsA(UsdGeom.Cube)
            or prim.IsA(UsdGeom.Sphere)
            or prim.IsA(UsdGeom.Cylinder)
            or prim.IsA(UsdGeom.Capsule)
            or prim.IsA(UsdGeom.Cone)
        )
        if not is_solid and not prim.IsInstanceable():
            continue
        if is_mesh:
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if points is None or len(points) == 0:
                continue
        try:
            if prim.IsInstanceProxy():
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if is_mesh:
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_api.CreateApproximationAttr().Set(approximation_shape)
                approx = physx_utils.MESH_APPROXIMATIONS.get(approximation_shape)
                if approx is not None:
                    approx.Apply(prim)
            count += 1
        except Exception as exc:
            print(f"[SPAWN] DataHall collider skip {prim.GetPath()}: {exc}")
    print(
        f"[SPAWN] DataHall static collisions enabled on {count} prim(s) under {root_path} "
        f"(approx={approximation_shape!r})"
    )
    return count


def strip_physics_from_prim(prim_path: str) -> None:
    """Remove rigid-body / collision so a visual prim stays suspended in air."""

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    count = 0
    for child in Usd.PrimRange(prim):
        try:
            if child.HasAPI(UsdPhysics.CollisionAPI):
                child.RemoveAPI(UsdPhysics.CollisionAPI)
                count += 1
        except Exception:
            attr = child.GetAttribute("physics:collisionEnabled")
            if attr and attr.IsValid():
                attr.Set(False)
                count += 1
        try:
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
                count += 1
        except Exception:
            pass
        for attr_name in ("physics:collisionEnabled", "physics:rigidBodyEnabled"):
            attr = child.GetAttribute(attr_name)
            if attr and attr.IsValid():
                try:
                    attr.Set(False)
                except Exception:
                    pass
    print(f"[SPAWN] Stripped physics/collision from {prim_path} ({count} API change(s))")


def configure_robot_physics(root_path: str) -> None:
    """Enable link collision/physics; disable self-collision; keep gravity off.

    Gravity stays disabled so the arm holds its mount pose (work table has no
    collision and will not catch a falling robot).

    UR10e */collisions meshes are authored as triangle meshes. PhysX rejects
    those on dynamic articulation links — set convexHull (and the matching
    PhysX API) explicitly so Play does not spam fallback errors.
    """

    from omni.physx.scripts import utils as physx_utils

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return

    art_roots = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        art_roots += 1
        try:
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False).Set(
                False
            )
        except Exception:
            attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            if attr and attr.IsValid():
                attr.Set(False)

    collision_count = 0
    rigid_count = 0
    convex_count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                rb = UsdPhysics.RigidBodyAPI(prim)
                rb.CreateRigidBodyEnabledAttr(True).Set(True)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True).Set(True)
                rigid_count += 1
            except Exception:
                pass

        path_l = str(prim.GetPath()).lower()
        name_l = prim.GetName().lower()
        is_collision_prim = (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or name_l == "collisions"
            or "/collisions" in path_l
            or (prim.IsA(UsdGeom.Mesh) and "collision" in path_l)
        )
        if not is_collision_prim:
            continue

        try:
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
            collision_count += 1
        except Exception:
            pass

        # Prefer the PhysX helper when the prim is a mesh; otherwise author the
        # same APIs onto UR */collisions prims (often not reported as Mesh).
        try:
            if prim.IsA(UsdGeom.Mesh) or prim.IsInstanceable():
                physx_utils.setCollider(prim, UsdPhysics.Tokens.convexHull)
            else:
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
                for tri_api in (
                    PhysxSchema.PhysxTriangleMeshCollisionAPI,
                    PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI,
                ):
                    try:
                        if prim.HasAPI(tri_api):
                            prim.RemoveAPI(tri_api)
                    except Exception:
                        pass
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            convex_count += 1
        except Exception as exc:
            print(f"[SPAWN] convexHull skip {prim.GetPath()}: {exc}")

    print(
        f"[SPAWN] Robot physics: articulation_roots={art_roots} "
        f"rigid={rigid_count} collisions={collision_count} "
        f"convexHull={convex_count} self_collision=False gravity=off"
    )


def resolve_robotiq_usd() -> str:
    for candidate in (ROBOTIQ_USD_LOCAL, ROBOTIQ_USD_FALLBACK):
        if candidate.is_file():
            path = str(candidate.resolve())
            print(f"[SPAWN] Using Robotiq 2F-85 USD: {path}")
            return path
    assets_root = get_assets_root_path()
    if assets_root:
        return assets_root + "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85.usd"
    raise RuntimeError("Could not find Robotiq 2F-85 USD.")


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
        "Robotiq_2f_85",
        "robotiq_2f_85",
        "2F_85",
        "2F-85",
    ):
        if candidate.lower() in lowered:
            chosen = lowered[candidate.lower()]
            variant_set.SetVariantSelection(chosen)
            print(f"[SPAWN] {GRIPPER_ASSEMBLY_NAMESPACE} variant -> {chosen}")
            return chosen
    return None


def assemble_robotiq_gripper(robot_prim_path: str, wrist_path: str) -> str:
    """Attach Robotiq 2F-85 at wrist_3_link via Robot Assembler."""

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


def attach_robotiq_2f85(robot_prim_path: str) -> tuple[str, str]:
    """Ensure Robotiq 2F-85 is attached (Gripper variant, else assembler)."""

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
    """Place the robot via mount world Xform; fall back to base_link on the table top."""

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


def create_cable_support_block(size_xyz: np.ndarray | None = None) -> float:
    """Pedestal on the table top spanning both crystal heads along X."""

    if stage.GetPrimAtPath(CABLE_SUPPORT_PATH).IsValid():
        stage.RemovePrim(Sdf.Path(CABLE_SUPPORT_PATH))

    top_z = table_top_z()
    size = CABLE_SUPPORT_SIZE_M if size_xyz is None else np.asarray(size_xyz, dtype=np.float64)
    size_x, size_y, size_z = (float(v) for v in size)
    center = Gf.Vec3d(
        float(CABLE_SUPPORT_XY[0]),
        float(CABLE_SUPPORT_XY[1]),
        top_z + 0.5 * size_z,
    )

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(CABLE_SUPPORT_PATH))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr(
        [Gf.Vec3f(float(CABLE_SUPPORT_COLOR[0]), float(CABLE_SUPPORT_COLOR[1]), float(CABLE_SUPPORT_COLOR[2]))]
    )
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(center)
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(size_x, size_y, size_z))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True).Set(True)

    block_top_z = float(center[2] + 0.5 * size_z)
    print(
        f"[SPAWN] Cable support: center={np.round(np.array(center), 4)} "
        f"size_m=({size_x:.3f}, {size_y:.3f}, {size_z:.3f}) "
        f"table_top_z={top_z:.4f} block_top_z={block_top_z:.4f}"
    )
    return block_top_z


def _support_cuboid(prim_path: str, center: np.ndarray, size_xyz: np.ndarray) -> None:
    """Create one collision cuboid piece of the cable support."""

    size_x, size_y, size_z = (float(v) for v in size_xyz)
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr(
        [Gf.Vec3f(float(CABLE_SUPPORT_COLOR[0]), float(CABLE_SUPPORT_COLOR[1]), float(CABLE_SUPPORT_COLOR[2]))]
    )
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    )
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(size_x, size_y, size_z))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True).Set(True)


def cut_grasp_notch_under_part(path45: str, path39: str) -> None:
    """Cut a visible top U-channel under crystal head 45 (E_part006_44).

    Rebuilds the support as Floor + Left + Right. The channel is forced onto the
    head45 end of the block (closer to head45 than head39). Depth is enough for
    finger clearance; a floor remains for cable balance. The block is extended
    slightly past the notch so both walls stay visible.
    """

    if not stage.GetPrimAtPath(CABLE_SUPPORT_PATH).IsValid():
        print("[SPAWN] Skip notch: cable support missing")
        return
    if not stage.GetPrimAtPath(path45).IsValid():
        print(f"[SPAWN] Skip notch: head45 missing at {path45}")
        return

    part_path = find_descendant(path45, GRASP_PART_NAME)
    # Prefer E_part006_44 when present; otherwise the whole head45 prim.
    anchor_path = part_path if part_path and stage.GetPrimAtPath(part_path).IsValid() else path45

    block_min, block_max, block_center = prim_bbox(CABLE_SUPPORT_PATH)
    _amin, _amax, a_center = prim_bbox(anchor_path)
    _h45min, _h45max, c45 = prim_bbox(path45)
    c39 = None
    if stage.GetPrimAtPath(path39).IsValid():
        _h39min, _h39max, c39 = prim_bbox(path39)

    # Always center on the head45 end. If the part bbox is closer to head39
    # (bad detection), fall back to the head45 center.
    notch_center_x = float(a_center[0])
    if c39 is not None and abs(notch_center_x - float(c39[0])) < abs(notch_center_x - float(c45[0])):
        print(
            f"[SPAWN] Notch anchor {anchor_path} was closer to head39; "
            f"forcing head45 center instead"
        )
        notch_center_x = float(c45[0])
        anchor_path = path45
        _amin, _amax, a_center = prim_bbox(anchor_path)

    # Use head45 X span so the gap clearly covers the crystal head for grasping.
    head_span_x = float(_h45max[0] - _h45min[0])
    notch_half = 0.5 * (head_span_x + float(NOTCH_EXTRA_X_M))
    notch_lo = notch_center_x - notch_half
    notch_hi = notch_center_x + notch_half

    # Extend the block past the notch so both shoulders stay visible.
    x_lo = min(float(block_min[0]), notch_lo - float(NOTCH_END_SHOULDER_M))
    x_hi = max(float(block_max[0]), notch_hi + float(NOTCH_END_SHOULDER_M))
    notch_lo = max(notch_lo, x_lo + float(NOTCH_MIN_SEGMENT_X_M))
    notch_hi = min(notch_hi, x_hi - float(NOTCH_MIN_SEGMENT_X_M))
    if notch_hi <= notch_lo + 1e-4:
        print("[SPAWN] Skip notch: computed gap is empty")
        return

    notch_mid = 0.5 * (notch_lo + notch_hi)
    if c39 is not None and abs(notch_mid - float(c39[0])) < abs(notch_mid - float(c45[0])):
        print(
            f"[SPAWN] Skip notch: mid={notch_mid:.4f} still closer to head39 "
            f"(c45={c45[0]:.4f} c39={c39[0]:.4f})"
        )
        return

    size_y = float(block_max[1] - block_min[1])
    size_z = float(block_max[2] - block_min[2])
    depth = min(float(NOTCH_DEPTH_Z_M), size_z * 0.75)
    floor_z = max(size_z - depth, 0.01)
    cy = float(block_center[1])
    z0 = float(block_min[2])
    floor_cz = z0 + 0.5 * floor_z
    wall_cz = z0 + floor_z + 0.5 * depth

    wall_lo_x0, wall_lo_x1 = x_lo, float(notch_lo)
    wall_hi_x0, wall_hi_x1 = float(notch_hi), x_hi
    if wall_lo_x1 - wall_lo_x0 < float(NOTCH_MIN_SEGMENT_X_M) or wall_hi_x1 - wall_hi_x0 < float(
        NOTCH_MIN_SEGMENT_X_M
    ):
        print("[SPAWN] Skip notch: shoulder too thin after layout")
        return

    # Name walls by head, not by world ±X: Right beside head45, Left beside head39.
    mid_lo = 0.5 * (wall_lo_x0 + wall_lo_x1)
    mid_hi = 0.5 * (wall_hi_x0 + wall_hi_x1)
    if abs(mid_lo - float(c45[0])) <= abs(mid_hi - float(c45[0])):
        right_x0, right_x1 = wall_lo_x0, wall_lo_x1
        left_x0, left_x1 = wall_hi_x0, wall_hi_x1
    else:
        right_x0, right_x1 = wall_hi_x0, wall_hi_x1
        left_x0, left_x1 = wall_lo_x0, wall_lo_x1

    stage.RemovePrim(Sdf.Path(CABLE_SUPPORT_PATH))
    UsdGeom.Xform.Define(stage, Sdf.Path(CABLE_SUPPORT_PATH))

    # Continuous floor under the whole span (keeps cable balance in the gap).
    _support_cuboid(
        f"{CABLE_SUPPORT_PATH}/Floor",
        np.array([0.5 * (x_lo + x_hi), cy, floor_cz], dtype=np.float64),
        np.array([x_hi - x_lo, size_y, floor_z], dtype=np.float64),
    )
    _support_cuboid(
        f"{CABLE_SUPPORT_PATH}/Left",
        np.array([float(SUPPORT_LEFT_X_M), cy, wall_cz], dtype=np.float64),
        np.array([left_x1 - left_x0, size_y, depth], dtype=np.float64),
    )
    _support_cuboid(
        f"{CABLE_SUPPORT_PATH}/Right",
        np.array([float(SUPPORT_RIGHT_X_M), cy, wall_cz], dtype=np.float64),
        np.array([right_x1 - right_x0, size_y, depth], dtype=np.float64),
    )

    c39_x = float(c39[0]) if c39 is not None else float("nan")
    print(
        f"[SPAWN] Grasp U-notch under head45 ({anchor_path}): "
        f"notch_x=[{notch_lo:.4f}, {notch_hi:.4f}] mid={notch_mid:.4f} "
        f"width={(notch_hi - notch_lo):.4f}m depth_z={depth:.4f}m floor_z={floor_z:.4f}m "
        f"c45_x={float(c45[0]):.4f} c39_x={c39_x:.4f} "
        f"Left_x={SUPPORT_LEFT_X_M:.5f} Right_x={SUPPORT_RIGHT_X_M:.5f} "
        f"block_x=[{x_lo:.4f}, {x_hi:.4f}]"
    )


def load_network_cable() -> tuple[str, str]:
    """Reference the network cable USD. Does not enable physics or collision."""

    if stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH).IsValid():
        stage.RemovePrim(Sdf.Path(NETWORK_CABLE_ROOT_PATH))
        for _ in range(10):
            simulation_app.update()
            time.sleep(0.01)

    print(f"[SPAWN] Referencing network cable: {NETWORK_CABLE_USD}")
    add_reference_to_stage(usd_path=NETWORK_CABLE_USD, prim_path=NETWORK_CABLE_ROOT_PATH)
    wait_for_stage_loading()

    path45 = find_descendant(NETWORK_CABLE_ROOT_PATH, "E_crystal_head1_45") or TRACKED_PLUG_PRIM_PATH
    path39 = find_descendant(NETWORK_CABLE_ROOT_PATH, "E_crystal_head2_39") or OTHER_PLUG_PRIM_PATH
    for path, label in ((path45, "E_crystal_head1_45"), (path39, "E_crystal_head2_39")):
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Cable is missing {label} at {path}")
    print(
        f"[SPAWN] Network cable loaded at {NETWORK_CABLE_ROOT_PATH} "
        f"(physics/collision not enabled)"
    )
    return path45, path39


def support_size_for_both_heads(path45: str, path39: str) -> np.ndarray:
    """Block X matches the authored span of both heads so each sits on an end."""

    min45, max45, _ = prim_bbox(path45)
    min39, max39, _ = prim_bbox(path39)
    lo = np.minimum(min45, min39)
    hi = np.maximum(max45, max39)
    span = hi - lo
    size_x = float(span[0]) + 2.0 * float(CABLE_SUPPORT_XY_MARGIN_M)
    size_y = max(float(CABLE_SUPPORT_SIZE_M[1]), float(span[1]) + 2.0 * float(CABLE_SUPPORT_XY_MARGIN_M))
    size_z = float(CABLE_SUPPORT_SIZE_M[2])
    return np.array([size_x, size_y, size_z], dtype=np.float64)


def place_crystal_heads_on_block_ends(
    path45: str, path39: str, block_top_z: float
) -> None:
    """Translate only the cable root so head45 and head39 sit on opposite block ends."""

    _block_min, _block_max, block_center = prim_bbox(CABLE_SUPPORT_PATH)
    min45, max45, c45 = prim_bbox(path45)
    min39, max39, c39 = prim_bbox(path39)

    heads_mid_xy = 0.5 * (c45[:2] + c39[:2])
    root_t = world_translation(NETWORK_CABLE_ROOT_PATH)
    # XY: center heads on the block. Z: fixed world height for the cable root.
    target = np.array(
        [
            float(block_center[0]) - float(heads_mid_xy[0]) + float(root_t[0]),
            float(block_center[1]) - float(heads_mid_xy[1]) + float(root_t[1]),
            float(CABLE_ROOT_Z),
        ],
        dtype=np.float64,
    )
    set_world_translate(NETWORK_CABLE_ROOT_PATH, target)
    for _ in range(5):
        simulation_app.update()

    min45, max45, c45 = prim_bbox(path45)
    min39, max39, c39 = prim_bbox(path39)
    block_min, block_max, _ = prim_bbox(CABLE_SUPPORT_PATH)
    root_z = float(world_translation(NETWORK_CABLE_ROOT_PATH)[2])
    print(
        f"[SPAWN] Heads on block ends: "
        f"head45_center={np.round(c45, 4)} head39_center={np.round(c39, 4)} "
        f"block_x=[{block_min[0]:.4f}, {block_max[0]:.4f}] "
        f"min_z45={min45[2]:.4f} min_z39={min39[2]:.4f} "
        f"cable_root_z={root_z:.4f} block_top_z={block_top_z:.4f}"
    )


def rebind_cable_deformable() -> None:
    """Rebuild PhysX after a cable-root translate so E_line_35 reattaches.

    Translating /World/NetworkCable after the soft body is initialized leaves
    the line mesh at the old pose while the crystal heads move with the root.
    Pause → Stop → Play in the Isaac UI rebuilds the deformable from the
    current USD; mirror that here and do not translate the cable afterward.
    """

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.pause()
    timeline.stop()
    for _ in range(20):
        simulation_app.update()
        time.sleep(0.01)
    timeline.play()
    for _ in range(30):
        simulation_app.update()
        time.sleep(0.01)
    print("[SPAWN] Timeline stop/play — soft cable rebound to crystal heads")


def compute_port_debug_points() -> tuple[np.ndarray, np.ndarray, str, str] | None:
    """Insert = mid of pins 1907/1910; approach = insert with +X standoff."""

    contacts = PORT_CONTACTS_PATH
    if not stage.GetPrimAtPath(contacts).IsValid():
        print(f"[SPAWN] Port debug markers skipped: missing {contacts}")
        return None
    path_a = find_descendant(contacts, PORT_PIN_A_NAME)
    path_b = find_descendant(contacts, PORT_PIN_B_NAME)
    if not path_a or not path_b:
        print(
            f"[SPAWN] Port debug markers skipped: pins "
            f"{PORT_PIN_A_NAME}={path_a} {PORT_PIN_B_NAME}={path_b}"
        )
        return None
    _amin, _amax, ca = prim_bbox(path_a)
    _bmin, _bmax, cb = prim_bbox(path_b)
    insert = 0.5 * (ca + cb)
    approach = insert.copy()
    approach[0] += float(PORT_APPROACH_X_OFFSET_M)
    return insert, approach, path_a, path_b


def _spawn_debug_sphere(
    prim_path: str,
    center: np.ndarray,
    color_rgb: tuple[float, float, float],
    *,
    scale: float | None = None,
    visible: bool = False,
) -> None:
    """Marker sphere: invisible by default, no collision (toggle Visibility)."""

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(prim_path))
    sphere.CreateRadiusAttr(1.0)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])
    prim = sphere.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    )
    s = float(PORT_DEBUG_MARKER_SCALE if scale is None else scale)
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(s, s, s))
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
    # Markers must not affect physics / the insertion simulation.
    strip_physics_from_prim(prim_path)


def spawn_port_debug_markers() -> None:
    """Yellow = approach offset (+0.02 X); red = insert. Invisible, non-colliding."""

    resolved = compute_port_debug_points()
    if resolved is None:
        return
    insert, approach, path_a, path_b = resolved
    UsdGeom.Xform.Define(stage, Sdf.Path(PORT_DEBUG_MARKER_ROOT))
    _spawn_debug_sphere(PORT_OFFSET_MARKER_PATH, approach, (1.0, 0.92, 0.1), visible=False)
    _spawn_debug_sphere(PORT_INSERT_MARKER_PATH, insert, (0.95, 0.12, 0.12), visible=False)
    print(
        f"[SPAWN] Port debug markers (invisible; toggle Visibility in Stage):\n"
        f"  yellow offset {PORT_OFFSET_MARKER_PATH} @ {np.round(approach, 4)}\n"
        f"  red insert    {PORT_INSERT_MARKER_PATH} @ {np.round(insert, 4)}\n"
        f"  from {path_a} / {path_b}"
    )


def _port_tip_via_debug(tip_start: np.ndarray, tip_end: np.ndarray, frac: float) -> np.ndarray:
    """Same tip via rule as ur10e_1x_cable_insertion (raised Z on intermediate fracs)."""

    t = float(np.clip(frac, 0.0, 1.0))
    tip = (1.0 - t) * tip_start + t * tip_end
    if t < 1.0 - 1e-9:
        tip = tip.copy()
        tip[2] = max(float(tip[2]), float(tip_end[2])) + float(PORT_APPROACH_VIA_Z_CLEARANCE_M)
    return tip


def spawn_maneuver_via_debug_markers(path45: str) -> None:
    """Green spheres at lift tip, yaw station, and lift→offset vias (scale 0.05)."""

    resolved = compute_port_debug_points()
    if resolved is None:
        return
    _insert, approach, _path_a, _path_b = resolved

    part_path = find_descendant(path45, GRASP_PART_NAME) or path45
    try:
        _mn, _mx, part_center = prim_bbox(part_path)
    except Exception as exc:
        print(f"[SPAWN] Maneuver via markers skipped: bbox failed for {part_path}: {exc}")
        return

    tip_start = np.asarray(part_center, dtype=np.float64).copy()
    tip_start[0] += float(DEBUG_GRASP_X_OFFSET_M)
    tip_start[2] += (
        float(DEBUG_GRASP_DESCEND_CLEARANCE_M)
        + float(DEBUG_GRASP_LIFT_CLEARANCE_M)
        + abs(float(DEBUG_GRASP_DESCEND_CLEARANCE_M))
    )
    tip_end = np.asarray(approach, dtype=np.float64).copy()
    tip_yaw = tip_start.copy()
    tip_yaw[2] = max(float(tip_start[2]), float(tip_end[2])) + float(
        PORT_APPROACH_VIA_Z_CLEARANCE_M
    )

    UsdGeom.Xform.Define(stage, Sdf.Path(PORT_DEBUG_MARKER_ROOT))
    if stage.GetPrimAtPath(PORT_MANEUVER_MARKER_ROOT).IsValid():
        stage.RemovePrim(Sdf.Path(PORT_MANEUVER_MARKER_ROOT))
    UsdGeom.Xform.Define(stage, Sdf.Path(PORT_MANEUVER_MARKER_ROOT))

    green = (0.15, 0.85, 0.25)
    scale = float(PORT_MANEUVER_MARKER_SCALE)
    points: list[tuple[str, np.ndarray]] = [
        ("LiftTip", tip_start),
        ("YawStation", tip_yaw),
    ]
    for frac in PORT_APPROACH_VIA_FRACTIONS:
        tip = _port_tip_via_debug(tip_start, tip_end, float(frac))
        points.append((f"Via_{int(round(float(frac) * 100)):02d}", tip))

    print(
        f"[SPAWN] Maneuver via markers (green, scale={scale}, invisible; "
        f"toggle Visibility under {PORT_MANEUVER_MARKER_ROOT}):"
    )
    for name, tip in points:
        prim_path = f"{PORT_MANEUVER_MARKER_ROOT}/{name}"
        _spawn_debug_sphere(prim_path, tip, green, scale=scale, visible=False)
        print(f"  {name} @ {np.round(tip, 4)}")


def build_asset_spawn_scene(app) -> AssetSpawnBundle:
    """Construct the asset_spawn scene and return handles for extensions."""

    _init_isaac_modules(app)

    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    light.CreateIntensityAttr(800.0)

    print(f"[SPAWN] Referencing DataHall: {DATAHALL_USD}")
    add_reference_to_stage(usd_path=DATAHALL_USD, prim_path=DATAHALL_PRIM_PATH)
    wait_for_stage_loading()
    print(f"[SPAWN] DataHall loaded at {DATAHALL_PRIM_PATH}")
    enable_datahall_static_collisions(DATAHALL_PRIM_PATH)
    spawn_port_debug_markers()

    table_orientation = euler_angles_to_quats(np.radians(TABLE_ORIENTATION_EULER_DEG))
    world.scene.add(
        VisualCuboid(
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
    strip_physics_from_prim(WORK_TABLE_PATH)
    print(
        f"[SPAWN] Work table (visual only, no physics) position={TABLE_POSITION.tolist()} "
        f"euler_deg={TABLE_ORIENTATION_EULER_DEG.tolist()} top_z={table_top_z():.4f}"
    )

    enable_gpu_dynamics()

    path45, path39 = load_network_cable()
    block_size = support_size_for_both_heads(path45, path39)
    block_top_z = create_cable_support_block(block_size)

    set_mount_world_translate(UR10E_MOUNT_PATH, UR10E_POSITION)

    robot_prim = add_reference_to_stage(usd_path=resolve_ur10e_usd(), prim_path=UR10E_PRIM_PATH)
    select_physics_variant(robot_prim)
    wait_for_stage_loading()

    base_link_path = find_descendant(UR10E_PRIM_PATH, "base_link")
    if not base_link_path:
        raise RuntimeError(f"Could not find base_link under {UR10E_PRIM_PATH}")

    end_effector_path, wrist_path = attach_robotiq_2f85(UR10E_PRIM_PATH)

    ur10e_robot = world.scene.add(
        SingleManipulator(
            prim_path=UR10E_PRIM_PATH,
            name="ur10e",
            end_effector_prim_path=end_effector_path,
        )
    )

    configure_robot_physics(UR10E_PRIM_PATH)
    world.reset()
    finalize_ur10e_placement(ur10e_robot, UR10E_MOUNT_PATH, base_link_path)
    configure_robot_physics(UR10E_PRIM_PATH)
    strip_physics_from_prim(WORK_TABLE_PATH)

    path45, path39 = load_network_cable()
    place_crystal_heads_on_block_ends(path45, path39, block_top_z)
    cut_grasp_notch_under_part(path45, path39)
    configure_robot_physics(UR10E_PRIM_PATH)
    rebind_cable_deformable()
    finalize_ur10e_placement(ur10e_robot, UR10E_MOUNT_PATH, base_link_path)
    configure_robot_physics(UR10E_PRIM_PATH)
    strip_physics_from_prim(WORK_TABLE_PATH)
    spawn_maneuver_via_debug_markers(path45)

    print(f"[SPAWN] UR10e + Robotiq 2F-85 wrist={wrist_path} end_effector={end_effector_path}")
    return AssetSpawnBundle(
        world=world,
        stage=stage,
        robot=ur10e_robot,
        end_effector_path=end_effector_path,
        wrist_path=wrist_path,
        base_link_path=base_link_path,
        path45=path45,
        path39=path39,
        block_top_z=block_top_z,
    )
