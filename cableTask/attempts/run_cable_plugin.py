# Copyright (c) 2026 Datacenter Isaac Sim — Use Case A: Cable Plug-in
# Standalone Isaac Sim 5.1.0 orchestration script.
# Run with: <ISAAC_SIM>/python.bat scripts/run_cable_plugin.py
#
# Environment:
#   ISAAC_ASSET_ROOT      — Isaac 5.1 pack (Franka, etc.)
#   DATACENTER_ASSET_ROOT — DigitalTwin datacenter USDs/MDLs (default below)
#   REPLICATOR_CAPTURE_INTERVAL — RGB capture every N sim steps (default 10)
#   ISAAC_EXIT_ON_COMPLETE=1 — close Kit when the sim loop ends (default: keep open)
#   HIDE_PROCEDURAL_FLOOR=1 — skip /World/Ground concrete slab under the rack

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Application entry — SimulationApp MUST precede all Isaac/USD imports.
# ---------------------------------------------------------------------------
_LOCAL_ASSET_ROOT = Path(
    os.environ.get("ISAAC_ASSET_ROOT", r"C:\isaacsim_assets\Assets\Isaac\5.1")
)
_FRANKA_USD = Path(
    os.environ.get(
        "FRANKA_USD_PATH",
        r"C:\isaacsim_assets\Assets\Isaac\5.1\Isaac\Robots\FrankaRobotics\FrankaPanda\franka.usd",
    )
)

_SIM_CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "extra_args": [
        f"--/persistent/isaac/asset_root/default={_LOCAL_ASSET_ROOT.as_posix()}",
    ],
}

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(_SIM_CONFIG)

import carb  # noqa: E402

carb.settings.get_settings().set("/app/asyncRendering", False)
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/default", _LOCAL_ASSET_ROOT.as_posix()
)

# Quiet third-party shader / fabric noise that comes from DigitalTwin MDL files.
# These are non-actionable compiler diagnostics (e.g. unused MDL params, missing
# annotations, USD->Fabric string array conversion) emitted by NVIDIA's own assets.
for _channel in (
    "/log/channels/rtx.neuraylib.plugin",
    "/log/channels/omni.fabric.plugin",
    "/log/channels/usdrt.population.plugin",
    "/log/channels/omni.syntheticdata.plugin",
    "/log/channels/isaacsim.sensors.camera.camera",
    "/log/channels/omni.isaac.wheeled_robots",
    "/log/channels/omni.isaac.manipulators",
    "/log/channels/omni.isaac.motion_generation",
    "/log/channels/omni.isaac.sensor",
    "/log/channels/omni.isaac.core",
    "/log/channels/omni.replicator.isaac",
    "/log/channels/isaacsim.core.simulation_manager.plugin",
    "/log/channels/omni.graph.core.plugin",
    "/log/channels/carb.audio.context",
):
    try:
        carb.settings.get_settings().set(_channel, "error")
    except Exception:
        pass

import numpy as np  # noqa: E402
import omni  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation, XFormPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.robot_motion.motion_generation import (  # noqa: E402
    ArticulationMotionPolicy,
    PathPlannerVisualizer,
    RmpFlow,
    interface_config_loader,
)
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.sensors.physics import ContactSensor  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from isaacsim.robot_motion.motion_generation.lula import RRT as _PathPlannerCls  # noqa: E402

try:
    from isaacsim.core.utils.semantics import add_labels  # noqa: E402
except ImportError:

    def add_labels(prim, labels, instance_name="class", overwrite=True):  # type: ignore[misc]
        if prim and prim.IsValid():
            prim.CreateAttribute(
                "semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.Token
            ).Set(labels[0] if labels else "")


def _set_semantic_label(prim: Usd.Prim, semantic_label: str) -> None:
    if prim and prim.IsValid():
        add_labels(prim, labels=[semantic_label], instance_name="class")


def _set_prim_display_color(prim: Usd.Prim, rgb: Tuple[float, float, float]) -> None:
    """Visible preview color for procedural USD meshes (RTX uses displayColor)."""
    if prim and prim.IsValid():
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def resolve_assets_root_path() -> str:
    """Return local Isaac asset root (no cloud/Nucleus connectivity required)."""
    root = Path(os.environ.get("ISAAC_ASSET_ROOT", str(_LOCAL_ASSET_ROOT)))
    if not root.is_dir():
        raise RuntimeError(
            f"Local Isaac asset root not found: {root}. "
            "Download assets or set ISAAC_ASSET_ROOT to the folder containing Isaac/ and NVIDIA/."
        )
    return root.as_posix()


_FRANKA_PACK_REL_PATHS = (
    "configuration/franka_robot_schema.usd",
    "Props/panda_link0.usd",
    "Props/panda_link1.usd",
    "Props/panda_link2.usd",
    "Props/panda_link3.usd",
    "Props/panda_link4.usd",
    "Props/panda_link5.usd",
    "Props/panda_link6.usd",
    "Props/panda_link7.usd",
    "Props/panda_hand.usd",
    "Props/panda_leftfinger.usd",
    "Props/panda_rightfinger.usd",
    "Materials/Materials.usd",
)


def validate_franka_asset_pack(franka_usd: Path) -> None:
    """Ensure FrankaPanda sub-assets referenced by franka.usd exist locally."""
    pack_dir = franka_usd.parent
    missing = [rel for rel in _FRANKA_PACK_REL_PATHS if not (pack_dir / rel).is_file()]
    if missing:
        raise RuntimeError(
            f"Incomplete Franka asset pack under {pack_dir}. Missing: {', '.join(missing)}. "
            "Download Props/ and configuration/ from the Isaac Sim 5.1 FrankaPanda folder on NVIDIA CDN."
        )


def resolve_franka_usd_path() -> str:
    """Return local Franka USD path for add_reference_to_stage."""
    usd_path = Path(os.environ.get("FRANKA_USD_PATH", str(_FRANKA_USD)))
    if not usd_path.is_file():
        raise RuntimeError(
            f"Franka USD not found: {usd_path}. "
            "Download franka.usd or set FRANKA_USD_PATH to your local copy."
        )
    validate_franka_asset_pack(usd_path)
    return usd_path.as_posix()


# ---------------------------------------------------------------------------
# DigitalTwin datacenter assets (cm-authored; scaled to meters in scene)
# ---------------------------------------------------------------------------
_DATACENTER_ASSET_ROOT = Path(
    os.environ.get("DATACENTER_ASSET_ROOT", r"C:\isaacsim_assets\Assets\DigitalTwin")
)
_DATACENTER_DC = _DATACENTER_ASSET_ROOT / "Assets" / "Datacenter"
RACK_USD_PATH = _DATACENTER_DC / "Racks" / "Rack_42U_A" / "Rack_42U_A_01.usd"
PATCH_PANEL_USD_PATH = (
    _DATACENTER_DC / "Racks" / "Patch_Panels" / "Fiber_A" / "Fiber_Patch_Panel_1U_A_01.usd"
)
HIDE_PROCEDURAL_FLOOR = os.environ.get("HIDE_PROCEDURAL_FLOOR", "").lower() in (
    "1",
    "true",
    "yes",
)
DATACENTER_CM_TO_M = 0.01
# Patch panel mount offset in world meters; parent ServerRack scale remains 1.0.
PATCH_PANEL_MOUNT_OFFSET_M = (0.52, 0.0, 1.05)

_MDL_DATACENTER = _DATACENTER_ASSET_ROOT / "Materials" / "Datacenter"
MDL_METAL_ALUMINUM = _MDL_DATACENTER / "Metal_Aluminum.mdl"
MDL_MESH_GRILLE = _MDL_DATACENTER / "MeshGrille_A.mdl"
_MDL_BASE = _DATACENTER_ASSET_ROOT / "Materials" / "Base"
MDL_RUBBER = _MDL_BASE / "Rubber" / "Rubber_New_A.mdl"
MDL_CONCRETE = _MDL_BASE / "Concrete" / "Concrete_Smooth_A.mdl"

_DATACENTER_REQUIRED = (RACK_USD_PATH, PATCH_PANEL_USD_PATH)
_DATACENTER_OPTIONAL: Tuple[Path, ...] = ()


def _resolve_datacenter_asset_root() -> Tuple[Path, Path, Path]:
    """Resolve DigitalTwin root and required rack USD paths; log absolute paths."""
    root = Path(os.environ.get("DATACENTER_ASSET_ROOT", str(_DATACENTER_ASSET_ROOT))).resolve()
    dc_root = root / "Assets" / "Datacenter"
    rack_path = dc_root / "Racks" / "Rack_42U_A" / "Rack_42U_A_01.usd"
    patch_path = (
        dc_root / "Racks" / "Patch_Panels" / "Fiber_A" / "Fiber_Patch_Panel_1U_A_01.usd"
    )
    LOGGER.info("Datacenter asset root: %s", root)
    LOGGER.info("  rack USD: %s", rack_path)
    LOGGER.info("  patch panel USD: %s", patch_path)
    return root, rack_path, patch_path


def validate_datacenter_rack_pack() -> Tuple[Path, Path]:
    """Ensure rack and patch-panel USDs exist under DATACENTER_ASSET_ROOT."""
    root, rack_path, patch_path = _resolve_datacenter_asset_root()
    if not root.is_dir():
        raise RuntimeError(
            f"Datacenter asset root not found: {root}. "
            "Set DATACENTER_ASSET_ROOT to C:\\isaacsim_assets\\Assets\\DigitalTwin"
        )
    if not (root / "Assets" / "Datacenter").is_dir():
        raise RuntimeError(
            f"Missing Assets/Datacenter under {root}. "
            "Expected layout: Assets/Datacenter/Racks/Rack_42U_A/Rack_42U_A_01.usd"
        )
    missing = []
    if not rack_path.is_file():
        missing.append(f"rack: {rack_path}")
    if not patch_path.is_file():
        missing.append(f"patch panel: {patch_path}")
    if missing:
        raise RuntimeError(
            "Missing required datacenter USD assets:\n  "
            + "\n  ".join(missing)
            + f"\nSet DATACENTER_ASSET_ROOT (currently {root})."
        )
    for opt in _DATACENTER_OPTIONAL:
        if not opt.is_file():
            LOGGER.warning("Optional datacenter asset missing: %s", opt)
    return rack_path, patch_path


# ---------------------------------------------------------------------------
# Repository paths and tunables
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_PORT = REPO_ROOT / "assets" / "rack" / "target_port.usd"
ASSET_SERVER = REPO_ROOT / "assets" / "server" / "plug_tip.usd"
ASSET_CABLE = REPO_ROOT / "assets" / "cable" / "compliant_cable.usd"
OUTPUT_DIR = REPO_ROOT / "outputs"
RGB_OUTPUT_DIR = OUTPUT_DIR / "rgb"
LABELS_DIR = OUTPUT_DIR / "labels"

FRANKA_PRIM = "/World/franka_robot"
FRANKA_HAND_LINK = f"{FRANKA_PRIM}/panda_hand"
CABLE_ROOT_PRIM = "/World/CableAssembly/CableRoot"
PLUG_PRIM = "/World/CableAssembly/PlugTip"
SERVER_RACK_ROOT = "/World/ServerRack"
RACK_PRIM = f"{SERVER_RACK_ROOT}/Rack"
PATCH_PANEL_MOUNT = f"{SERVER_RACK_ROOT}/PatchPanelMount"
PATCH_PANEL_PRIM = f"{PATCH_PANEL_MOUNT}/PatchPanel"
PORT_PRIM = f"{SERVER_RACK_ROOT}/TargetPort"
PORT_PROXY_PRIM = "/World/obstacles/port_proxy"
RACK_OBSTACLE_PRIM = "/World/obstacles/rack_collision_proxy"
GRIPPER_WELD_JOINT = "/World/CableAssembly/gripper_weld"
PLUG_CASING_RELATIVE = "PlugCasing"
PORT_SLEEVE_RELATIVE = "PortSleeve"

# Populated after rack load in SceneLayout (defaults until then).
PORT_WORLD_POS = np.array([0.52, 0.0, 1.05], dtype=np.float64)
PORT_WORLD_ORIENT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # WXYZ
FRANKA_WORLD_POS = np.array([0.10, -0.20, 0.0], dtype=np.float64)
FRANKA_WORLD_ORIENT = np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float64)
KIT_DEFAULT_CAMERA_PATH = "/OmniverseKit_Persp"
# Default Kit Perspective view (from property panel — translate, rotate XYZ, lens).
KIT_DEFAULT_CAMERA_TRANSLATE = (0.93047, 1.45847, 2.13761)
# Eye offset from rack center when auto-framing (saved view minus typical rack center).
KIT_DEFAULT_CAMERA_FRAMING_OFFSET = np.array([0.63, 1.46, 1.14], dtype=np.float64)
KIT_DEFAULT_CAMERA_ROTATE_XYZ_DEG = (54.73561, 0.0, 135.0)
KIT_DEFAULT_CAMERA_FOCAL_LENGTH = 18.14756
KIT_DEFAULT_CAMERA_FOCUS_DISTANCE = 400.0
KIT_DEFAULT_CAMERA_F_STOP = 0.0
KIT_DEFAULT_CAMERA_HORIZONTAL_APERTURE = 20.955
KIT_DEFAULT_CAMERA_VERTICAL_APERTURE = 15.2908
KIT_DEFAULT_CAMERA_CLIP_NEAR = 0.01
KIT_DEFAULT_CAMERA_CLIP_FAR = 10000000.0
WRIST_CAMERA_HAPERTURE_M = 0.020955
WRIST_CAMERA_HFOV_DEG = 45.0

MAX_SIM_STEPS = 1200
MAX_RRT_WAYPOINTS = 60
REPLICATOR_CAPTURE_INTERVAL = int(os.environ.get("REPLICATOR_CAPTURE_INTERVAL", "10"))
FORCE_ABORT_N = 35.0
ALIGN_TRANS_TOL_M = 0.005
ALIGN_ROT_TOL_RAD = 0.05
SUCCESS_TRANS_TOL_M = 0.003
SUCCESS_DEPTH_M = 0.005
PRE_INSERTION_OFFSET_M = 0.08
TARGET_INSERTION_DEPTH_M = 0.025

CABLE_NUM_SEGMENTS = 10
CABLE_SEGMENT_LENGTH = 0.02
CABLE_SEGMENT_RADIUS = 0.004
CABLE_SEGMENT_MASS = 0.01
JOINT_STIFFNESS = 75.0
JOINT_DAMPING = 12.0
CABLE_LINEAR_DAMPING = 8.0
CABLE_ANGULAR_DAMPING = 8.0
ALIGN_RMP_RAMP_STEPS = 120
RRT_MAX_CSPACE_DIST = 0.01
RRT_SUBSTEPS = 5
READY_POSE_STEPS = 30
READY_POSE_JOINTS = np.array(
    [0.0, -0.65, 0.0, -2.1, 0.0, 1.75, 0.75, 0.04, 0.04], dtype=np.float64
)
CABLE_SWAY_ENABLED = True
# Physics weld from gripper to cable segment_0 fights RMPflow and explodes the chain.
USE_GRIPPER_PHYSICS_WELD = False
CABLE_ROOT_HAND_OFFSET = np.array([0.0, 0.0, -0.08], dtype=np.float64)
PLUG_OFFSET_FROM_CABLE_ROOT = np.array(
    [0.0, 0.0, CABLE_NUM_SEGMENTS * CABLE_SEGMENT_LENGTH * 0.5], dtype=np.float64
)

def _quat_rotate_wxyz(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vector by WXYZ quaternion."""
    w, x, y, z = quat_wxyz
    vx, vy, vz = vector
    ix = w * vx + y * vz - z * vy
    iy = w * vy + z * vx - x * vz
    iz = w * vz + x * vy - y * vx
    iw = -x * vx - y * vy - z * vz
    return np.array(
        [
            ix * w + iw * -x + iy * -z - iz * -y,
            iy * w + iw * -y + iz * -x - ix * -z,
            iz * w + iw * -z + ix * -y - iy * -x,
        ],
        dtype=np.float64,
    )


def _poses_are_finite(pos: np.ndarray, orient: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(pos)) and np.all(np.isfinite(orient)))


@dataclass
class SceneLayout:
    """World poses and framing derived from loaded rack geometry."""

    port_world_pos: np.ndarray = field(
        default_factory=lambda: PORT_WORLD_POS.copy()
    )
    port_world_orient: np.ndarray = field(
        default_factory=lambda: PORT_WORLD_ORIENT.copy()
    )
    franka_world_pos: np.ndarray = field(
        default_factory=lambda: FRANKA_WORLD_POS.copy()
    )
    franka_world_orient: np.ndarray = field(
        default_factory=lambda: FRANKA_WORLD_ORIENT.copy()
    )
    rack_obstacle_center: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.0, 1.0], dtype=np.float64)
    )
    rack_obstacle_scale: np.ndarray = field(
        default_factory=lambda: np.array([0.55, 0.75, 2.0], dtype=np.float64)
    )
    plug_start_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.15, 0.0, 0.12], dtype=np.float64)
    )


def _install_scene_lighting(stage: Usd.Stage) -> None:
    """Data-hall style lighting: dome IBL + aisle rect + soft sun."""
    if not stage.GetPrimAtPath("/World/DomeLight").IsValid():
        dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(380.0)
        dome.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.92, 0.94, 1.0)
        )
    if not stage.GetPrimAtPath("/World/AisleRectLight").IsValid():
        rect = stage.DefinePrim("/World/AisleRectLight", "RectLight")
        rect.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(2800.0)
        rect.CreateAttribute("inputs:width", Sdf.ValueTypeNames.Float).Set(2.5)
        rect.CreateAttribute("inputs:height", Sdf.ValueTypeNames.Float).Set(1.2)
        rect.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(1.0, 0.98, 0.95)
        )
        xf = UsdGeom.Xformable(rect)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.2, 0.0, 2.8))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-75.0, 0.0, 0.0))
    if not stage.GetPrimAtPath("/World/DistantLight").IsValid():
        sun = stage.DefinePrim("/World/DistantLight", "DistantLight")
        sun.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(420.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 48.0, 0.0))


def _install_datacenter_floor(stage: Usd.Stage, layout: SceneLayout) -> None:
    """Concrete slab under the rack aisle."""
    if HIDE_PROCEDURAL_FLOOR:
        return
    floor_path = "/World/Ground"
    if stage.GetPrimAtPath(floor_path).IsValid():
        return
    floor = UsdGeom.Cube.Define(stage, Sdf.Path(floor_path))
    floor.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(floor)
    xf.AddTranslateOp().Set(
        Gf.Vec3d(
            float(layout.rack_obstacle_center[0]),
            float(layout.rack_obstacle_center[1]),
            -0.02,
        )
    )
    xf.AddScaleOp().Set(Gf.Vec3d(4.0, 3.5, 0.04))
    _set_prim_display_color(floor.GetPrim(), (0.35, 0.36, 0.38))
    if MDL_CONCRETE.is_file():
        _bind_mdl(stage, floor_path, MDL_CONCRETE)


def _bind_mdl(stage: Usd.Stage, geom_path: str, mdl_path: Path) -> None:
    """Bind an MDL material from the DigitalTwin pack to a prim."""
    if not mdl_path.is_file():
        return
    prim = stage.GetPrimAtPath(geom_path)
    if not prim.IsValid():
        return
    safe_name = mdl_path.stem.replace(".", "_")
    mat_path = f"/World/Looks/{safe_name}"
    if not stage.GetPrimAtPath(mat_path).IsValid():
        material = UsdShade.Material.Define(stage, Sdf.Path(mat_path))
        shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{mat_path}/Shader"))
        shader.CreateIdAttr("mdlMaterial")
        shader.CreateInput("mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(mdl_path.as_posix())
        )
        shader.CreateInput("mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set(
            mdl_path.stem
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")
    material = UsdShade.Material.Get(stage, Sdf.Path(mat_path))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _set_prim_visibility(stage: Usd.Stage, prim_path: str, visible: bool) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def _apply_uniform_scale(stage: Usd.Stage, prim_path: str, scale: float) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
    if scale_ops:
        scale_ops[0].Set(Gf.Vec3f(scale, scale, scale))
    else:
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(scale, scale, scale))


def _set_or_add_translate(stage: Usd.Stage, prim_path: str, translation: Tuple[float, float, float]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*translation))
            return
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))


def _count_prim_descendants(prim_path: str, stage: Usd.Stage) -> int:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return 0
    return sum(1 for _ in Usd.PrimRange(prim)) - 1


def _wait_for_prim_descendants(
    stage: Usd.Stage,
    prim_path: str,
    min_children: int = 1,
    max_updates: int = 120,
) -> bool:
    """Pump Kit until ``prim_path`` has at least ``min_children`` descendants."""
    for _ in range(max_updates):
        if _count_prim_descendants(prim_path, stage) >= min_children:
            return True
        simulation_app.update()
    return _count_prim_descendants(prim_path, stage) >= min_children


def _make_prim_subtree_visible(stage: Usd.Stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for p in Usd.PrimRange(prim):
        UsdGeom.Imageable(p).MakeVisible()


def _disable_collisions_between_subtrees(stage: Usd.Stage, path_a: str, path_b: str) -> None:
    for source_path, target_path in ((path_a, path_b), (path_b, path_a)):
        prim = stage.GetPrimAtPath(source_path)
        if not prim.IsValid():
            continue
        filter_api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = filter_api.GetFilteredPairsRel()
        if not rel:
            rel = filter_api.CreateFilteredPairsRel()
        rel.AddTarget(Sdf.Path(target_path))


def _bbox_height_m(half_extents: np.ndarray) -> float:
    return float(2.0 * half_extents[2])


def _verify_rack_loaded(stage: Usd.Stage, rack_prim_path: str) -> None:
    """Fail fast if the rack reference did not compose any geometry."""
    if _count_prim_descendants(rack_prim_path, stage) < 1:
        root = Path(os.environ.get("DATACENTER_ASSET_ROOT", str(_DATACENTER_ASSET_ROOT)))
        raise RuntimeError(
            f"Datacenter rack prim has no descendants: {rack_prim_path}. "
            f"Check DATACENTER_ASSET_ROOT ({root}) and USD references."
        )
    center, half = _compute_prim_world_bbox(stage, rack_prim_path)
    height = _bbox_height_m(half)
    LOGGER.info(
        "Rack composed under %s; bbox center=%s half=%s height=%.3f m",
        rack_prim_path,
        center.tolist(),
        half.tolist(),
        height,
    )


def _compute_prim_world_bbox(
    stage: Usd.Stage, prim_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (center, half-extents) in world meters."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.zeros(3, dtype=np.float64), np.array([0.3, 0.3, 1.0], dtype=np.float64)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bound = cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedBox()
    mn = np.array(box.GetMin(), dtype=np.float64)
    mx = np.array(box.GetMax(), dtype=np.float64)
    center = 0.5 * (mn + mx)
    half = 0.5 * (mx - mn)
    half = np.maximum(half, np.array([0.05, 0.05, 0.05], dtype=np.float64))
    return center, half


def _compute_scene_layout_from_rack(stage: Usd.Stage, rack_prim_path: str) -> SceneLayout:
    """Derive fixed port/robot poses and obstacle extents from rack geometry."""
    center, half = _compute_prim_world_bbox(stage, rack_prim_path)
    port_pos = np.array([0.52, 0.0, 1.05], dtype=np.float64)
    port_orient = np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float64)  # face +X

    franka_pos = FRANKA_WORLD_POS.copy()
    franka_orient = FRANKA_WORLD_ORIENT.copy()

    return SceneLayout(
        port_world_pos=port_pos,
        port_world_orient=port_orient,
        franka_world_pos=franka_pos,
        franka_world_orient=franka_orient,
        rack_obstacle_center=center.copy(),
        rack_obstacle_scale=(half * 2.0 * 0.92),
        plug_start_offset=np.array([0.18, 0.0, 0.1], dtype=np.float64),
    )


def _apply_scene_layout_globals(layout: SceneLayout) -> None:
    """Sync module-level pose constants used before orchestrator holds layout."""
    global PORT_WORLD_POS, PORT_WORLD_ORIENT, FRANKA_WORLD_POS, FRANKA_WORLD_ORIENT
    PORT_WORLD_POS = layout.port_world_pos.copy()
    PORT_WORLD_ORIENT = layout.port_world_orient.copy()
    FRANKA_WORLD_POS = layout.franka_world_pos.copy()
    FRANKA_WORLD_ORIENT = layout.franka_world_orient.copy()


def _prime_renderer(frames: int = 24) -> None:
    """Let RTX load materials and warm the render pipeline before capture."""
    for _ in range(frames):
        simulation_app.update()


def _focal_length_from_hfov(hfov_deg: float, h_aperture_m: float) -> float:
    """Focal length (m) for a pinhole camera with the given horizontal FOV."""
    half_fov = np.radians(hfov_deg) * 0.5
    return float(h_aperture_m / (2.0 * np.tan(half_fov)))


def _default_viewport_camera_path() -> str:
    """Return the active Kit viewport camera, or the built-in Perspective prim."""
    return KIT_DEFAULT_CAMERA_PATH


def _apply_kit_default_camera_view(
    stage: Optional[Usd.Stage] = None,
    look_at_target: Optional[np.ndarray] = None,
) -> bool:
    """Set /OmniverseKit_Persp transform and lens to the project's default view."""
    if stage is None:
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False

    prim = stage.GetPrimAtPath(KIT_DEFAULT_CAMERA_PATH)
    if not prim.IsValid():
        LOGGER.warning("Kit perspective camera not found: %s", KIT_DEFAULT_CAMERA_PATH)
        return False

    if look_at_target is not None:
        target = np.asarray(look_at_target, dtype=np.float64).reshape(3)
        camera_translate = tuple((target + KIT_DEFAULT_CAMERA_FRAMING_OFFSET).tolist())
    else:
        camera_translate = KIT_DEFAULT_CAMERA_TRANSLATE

    xformable = UsdGeom.Xformable(prim)
    # Kit already owns translate/rotate/scale ops on Persp — update them in place.
    has_translate = has_rotate = has_scale = False
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*camera_translate))
            has_translate = True
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(*KIT_DEFAULT_CAMERA_ROTATE_XYZ_DEG))
            has_rotate = True
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
            has_scale = True
    if not has_translate:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*camera_translate))
    if not has_rotate:
        xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Vec3f(*KIT_DEFAULT_CAMERA_ROTATE_XYZ_DEG)
        )
    if not has_scale:
        xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    cam = UsdGeom.Camera(prim)
    cam.GetFocalLengthAttr().Set(KIT_DEFAULT_CAMERA_FOCAL_LENGTH)
    cam.GetFocusDistanceAttr().Set(KIT_DEFAULT_CAMERA_FOCUS_DISTANCE)
    cam.GetFStopAttr().Set(KIT_DEFAULT_CAMERA_F_STOP)
    cam.GetHorizontalApertureAttr().Set(KIT_DEFAULT_CAMERA_HORIZONTAL_APERTURE)
    cam.GetVerticalApertureAttr().Set(KIT_DEFAULT_CAMERA_VERTICAL_APERTURE)
    cam.GetClippingRangeAttr().Set(
        Gf.Vec2f(KIT_DEFAULT_CAMERA_CLIP_NEAR, KIT_DEFAULT_CAMERA_CLIP_FAR)
    )
    return True


def _reset_viewport_to_default_camera(
    stage: Optional[Usd.Stage] = None,
    look_at_target: Optional[np.ndarray] = None,
) -> None:
    """Apply the default Perspective pose/lens and make it the active viewport camera."""
    _apply_kit_default_camera_view(stage, look_at_target=look_at_target)
    try:
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(KIT_DEFAULT_CAMERA_PATH)
    except Exception as exc:
        LOGGER.debug("Could not reset viewport camera: %s", exc)


def _configure_perspective_lens(
    camera: Camera,
    hfov_deg: float = WRIST_CAMERA_HFOV_DEG,
    h_aperture_m: float = WRIST_CAMERA_HAPERTURE_M,
) -> None:
    """Set focal length and aperture in stage meters (Kit Perspective-like FOV)."""
    if hfov_deg <= 0.0 or h_aperture_m <= 0.0:
        return
    focal_m = _focal_length_from_hfov(hfov_deg, h_aperture_m)
    camera.set_focal_length(focal_m)
    camera.set_horizontal_aperture(h_aperture_m)


def _configure_rendering_rates(physics_hz: int) -> None:
    """Align Kit render/timeline rate with physics so camera interpolation has samples."""
    import omni.timeline

    settings = carb.settings.get_settings()
    settings.set_int("/app/runLoops/main/rateLimitFrequency", physics_hz)
    settings.set_float("/app/stage/timeCodesPerSecond", float(physics_hz))
    settings.set_float("/persistent/app/stage/timeCodesPerSecond", float(physics_hz))

    timeline = omni.timeline.get_timeline_interface()
    if hasattr(timeline, "set_time_codes_per_second"):
        timeline.set_time_codes_per_second(float(physics_hz))
    if hasattr(timeline, "set_target_framerate"):
        timeline.set_target_framerate(float(physics_hz))


def _start_physics_timeline(physics_hz: int) -> None:
    """Play the Kit timeline at the physics rate so sensor interpolation has samples."""
    _configure_rendering_rates(physics_hz)
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()


def _set_timeline_seconds(seconds: float) -> None:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    if hasattr(timeline, "set_current_time"):
        timeline.set_current_time(seconds)
    elif hasattr(timeline, "set_time"):
        timeline.set_time(seconds)


LOGGER = logging.getLogger("cable_plugin")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _flatten_vec3(value: Any) -> np.ndarray:
    """Flatten Isaac Sim 5.1 batch-first position arrays to shape (3,)."""
    arr = np.asarray(value, dtype=np.float64)
    return np.squeeze(arr).reshape(-1)[:3].copy()


def _flatten_quat_wxyz(value: Any) -> np.ndarray:
    """Flatten Isaac Sim 5.1 batch-first quaternion arrays to shape (4,) WXYZ."""
    arr = np.asarray(value, dtype=np.float64)
    q = np.squeeze(arr).reshape(-1)[:4].copy()
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q /= norm
    return q


def ensure_asset_usd(asset_path: Path, builder: Callable[[Usd.Stage], None]) -> Path:
    """Create and persist a USD asset if missing; return resolved path."""
    asset_path = asset_path.resolve()
    if asset_path.is_file():
        existing_stage = Usd.Stage.Open(str(asset_path))
        if existing_stage:
            if existing_stage.GetDefaultPrim().IsValid():
                LOGGER.info("Reusing cached asset: %s", asset_path)
                return asset_path
            root_children = existing_stage.GetPseudoRoot().GetChildren()
            if root_children:
                existing_stage.SetDefaultPrim(root_children[0])
                existing_stage.GetRootLayer().Save()
                LOGGER.warning("Patched missing defaultPrim for asset: %s", asset_path)
                return asset_path
        LOGGER.warning("Regenerating unreadable asset: %s", asset_path)
        asset_path.unlink(missing_ok=True)

    asset_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Usd.Stage.CreateNew(str(asset_path))
    if not scratch:
        raise OSError(f"Failed to create USD stage at {asset_path}")
    try:
        builder(scratch)
        scratch.GetRootLayer().Save()
        LOGGER.info("Generated asset: %s", asset_path)
    except Exception:
        if asset_path.is_file():
            asset_path.unlink(missing_ok=True)
        raise
    return asset_path


def _apply_sliding_material(stage: Usd.Stage, geom_path: str, material_path: str) -> None:
    prim = stage.GetPrimAtPath(geom_path)
    if not prim.IsValid():
        LOGGER.warning("Cannot bind material — prim missing: %s", geom_path)
        return
    material = UsdShade.Material.Get(stage, Sdf.Path(material_path))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def _define_d6_joint(
    stage: Usd.Stage,
    joint_path: str,
    parent_path: str,
    child_path: str,
    local_pos0: Gf.Vec3f,
    local_pos1: Gf.Vec3f,
) -> UsdPhysics.Joint:
    path = Sdf.Path(joint_path)
    if hasattr(UsdPhysics, "D6Joint"):
        joint = UsdPhysics.D6Joint.Define(stage, path)
    else:
        joint = UsdPhysics.Joint.Define(stage, path)

    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
    joint.CreateLocalPos0Attr().Set(local_pos0)
    joint.CreateLocalPos1Attr().Set(local_pos1)

    _apply_angular_joint_drive(joint.GetPrim())
    return joint


def _apply_angular_joint_drive(joint_prim: Usd.Prim) -> None:
    """Apply torsional spring-damper drives (Isaac Sim 5.1 uses UsdPhysics.DriveAPI)."""
    if hasattr(UsdPhysics, "DriveAPI"):
        for axis in ("rotX", "rotY", "rotZ"):
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, axis)
            drive_api.CreateStiffnessAttr(JOINT_STIFFNESS)
            drive_api.CreateDampingAttr(JOINT_DAMPING)
            if hasattr(drive_api, "CreateTargetPositionAttr"):
                drive_api.CreateTargetPositionAttr(0.0)
        return

    if hasattr(PhysxSchema, "PhysxJointDriveAPI"):
        drive_api = PhysxSchema.PhysxJointDriveAPI.Apply(joint_prim, "angular")
        drive_api.CreateStiffnessAttr(JOINT_STIFFNESS)
        drive_api.CreateDampingAttr(JOINT_DAMPING)
        return

    LOGGER.warning(
        "No joint drive API available; cable joints will lack angular compliance drives."
    )


def build_target_port_asset(stage: Usd.Stage) -> None:
    root = UsdGeom.Xform.Define(stage, Sdf.Path("/TargetPort"))
    cube = UsdGeom.Cube.Define(stage, Sdf.Path("/TargetPort/PortSleeve"))
    cube.CreateSizeAttr(0.032)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _set_prim_display_color(cube.GetPrim(), (0.25, 0.28, 0.32))
    _set_semantic_label(cube.GetPrim(), "target_port")
    stage.SetDefaultPrim(root.GetPrim())


def build_plug_tip_asset(stage: Usd.Stage) -> None:
    root = UsdGeom.Xform.Define(stage, Sdf.Path("/PlugTip"))
    cube = UsdGeom.Cube.Define(stage, Sdf.Path("/PlugTip/PlugCasing"))
    cube.CreateSizeAttr(0.03)
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    # Plug pose is driven kinematically from the gripper every frame
    # (see _sync_cable_assembly_to_hand). Without the kinematic flag the
    # solver treats set_world_poses as an override on a dynamic body and
    # generates unbounded contact forces on collision with the port.
    if hasattr(PhysxSchema, "PhysxRigidBodyAPI"):
        rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim())
        if hasattr(rb_api, "CreateKinematicEnabledAttr"):
            rb_api.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass_api.CreateMassAttr(0.12)
    _set_prim_display_color(cube.GetPrim(), (0.85, 0.45, 0.12))

    contact_report = PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim())
    contact_report.CreateThresholdAttr(0.0)
    _set_semantic_label(cube.GetPrim(), "plug_tip")
    stage.SetDefaultPrim(root.GetPrim())


def build_compliant_cable_asset(stage: Usd.Stage) -> None:
    parent_path = "/CableRoot"
    cable_root = UsdGeom.Xform.Define(stage, Sdf.Path(parent_path))
    previous_segment: Optional[str] = None
    half_len = CABLE_SEGMENT_LENGTH / 2.0

    for i in range(CABLE_NUM_SEGMENTS):
        segment_path = f"{parent_path}/segment_{i}"
        cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(segment_path))
        cylinder.CreateRadiusAttr(CABLE_SEGMENT_RADIUS)
        cylinder.CreateHeightAttr(CABLE_SEGMENT_LENGTH)
        cylinder.CreateAxisAttr("Z")
        _set_prim_display_color(cylinder.GetPrim(), (0.08, 0.08, 0.1))

        UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(cylinder.GetPrim())
        mass_api.CreateMassAttr(CABLE_SEGMENT_MASS)

        if hasattr(PhysxSchema, "PhysxRigidBodyAPI"):
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(cylinder.GetPrim())
            rb_api.CreateLinearDampingAttr(CABLE_LINEAR_DAMPING)
            rb_api.CreateAngularDampingAttr(CABLE_ANGULAR_DAMPING)
            if hasattr(rb_api, "CreateKinematicEnabledAttr"):
                rb_api.CreateKinematicEnabledAttr(True)

        xformable = UsdGeom.Xformable(cylinder)
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, i * CABLE_SEGMENT_LENGTH))

        if previous_segment is not None:
            joint_path = f"{parent_path}/joint_{i}"
            _define_d6_joint(
                stage,
                joint_path,
                previous_segment,
                segment_path,
                Gf.Vec3f(0.0, 0.0, half_len),
                Gf.Vec3f(0.0, 0.0, -half_len),
            )
        previous_segment = segment_path

    _set_semantic_label(stage.GetPrimAtPath(f"{parent_path}/segment_0"), "cable_segment")
    stage.SetDefaultPrim(cable_root.GetPrim())


class TaskState(Enum):
    APPROACH = auto()
    ALIGN = auto()
    INSERT = auto()
    VERIFY = auto()


class CableInsertionContext:
    """Lightweight logical state for obstacle monitors and RMP toggling."""

    def __init__(self) -> None:
        self.task_state = TaskState.APPROACH
        self.world: Optional[World] = None

    def get_logical_state(self) -> dict:
        return {"is_mating_active": self.task_state == TaskState.INSERT}


class InsertionPortObstacleMonitor:
    """Toggle port proxy obstacle in RMPflow — off during INSERT to allow sleeve entry."""

    def __init__(self, context: CableInsertionContext, port_obstacle: DynamicCuboid) -> None:
        self.context = context
        self.port_obstacle = port_obstacle
        self._active = False
        self._port_registered = False

    def is_obstacle_required(self) -> bool:
        if self.context.get_logical_state().get("is_mating_active", False):
            return False
        return True

    def enable(self) -> None:
        self._active = True

    def disable(self, rmpflow: Optional[RmpFlow] = None) -> None:
        self._active = False
        if rmpflow is not None:
            self._remove(rmpflow)

    def sync(self, rmpflow: RmpFlow) -> None:
        if not self._active:
            return
        if self.is_obstacle_required():
            self._add(rmpflow)
        else:
            self._remove(rmpflow)

    def _add(self, rmpflow: RmpFlow) -> None:
        if self._port_registered:
            return
        try:
            rmpflow.add_obstacle(self.port_obstacle)
            self._port_registered = True
        except Exception as exc:
            LOGGER.debug("Port obstacle add failed: %s", exc)

    def _remove(self, rmpflow: RmpFlow) -> None:
        if not self._port_registered:
            return
        try:
            rmpflow.remove_obstacle(self.port_obstacle)
            self._port_registered = False
        except Exception as exc:
            LOGGER.debug("Port obstacle remove failed: %s", exc)


class CablePluginOrchestrator:
    """Use Case A — programmatic cable plug-in with RRT, RMPflow, and SDG capture."""

    def __init__(self) -> None:
        self.context = CableInsertionContext()
        self.world = World(stage_units_in_meters=1.0)
        self.context.world = self.world
        self.stage = omni.usd.get_context().get_stage()

        self.articulation: Optional[SingleArticulation] = None
        self.plug: Optional[RigidPrim] = None
        self.port: Optional[XFormPrim] = None
        self.port_proxy: Optional[DynamicCuboid] = None
        self.rack_obstacle: Optional[DynamicCuboid] = None
        self.rrt: Optional[Any] = None
        self.path_visualizer: Optional[PathPlannerVisualizer] = None
        self.rmpflow: Optional[RmpFlow] = None
        self.articulation_policy: Optional[ArticulationMotionPolicy] = None
        self.port_monitor: Optional[InsertionPortObstacleMonitor] = None

        self.rrt_plan: List[Any] = []
        self.step_log: List[dict] = []
        self.contact_sensor: Optional[ContactSensor] = None
        self.wrist_camera: Optional[Camera] = None
        self.replicator_writer: Any = None
        self._contact_warned = False
        self._align_rmp_steps = 0
        self._align_start_ee_pos: Optional[np.ndarray] = None
        self._physics_unstable = False
        self._cable_root_xform: Optional[XFormPrim] = None
        self.scene_layout = SceneLayout()
        self._ready_pose_steps_left = READY_POSE_STEPS
        self._rrt_interp_action: Any = None
        self._rrt_interp_step = 0
        self._rrt_interp_start_joints: Optional[np.ndarray] = None
        self._last_finite_plug_pose: Tuple[np.ndarray, np.ndarray] = (
            PORT_WORLD_POS + np.array([0.12, 0.0, 0.18], dtype=np.float64),
            PORT_WORLD_ORIENT.copy(),
        )

        self._generate_local_assets()
        self._configure_physics_scene()
        self._assemble_scene()
        self._configure_cable_physics_mode()
        if USE_GRIPPER_PHYSICS_WELD:
            # Weld must exist before the first world.reset() when enabled.
            self._create_gripper_weld_joint()
        else:
            LOGGER.info(
                "Gripper physics weld disabled; cable will follow hand kinematically."
            )

        LOGGER.info("Structural scene assembly complete.")
        self._register_franka_articulation()
        for _ in range(12):
            simulation_app.update()
        _reset_viewport_to_default_camera(
            self.stage, look_at_target=self.scene_layout.rack_obstacle_center
        )

    def _register_franka_articulation(self) -> None:
        """Register Franka with World.scene before the first world.reset() (Isaac Sim 5.1)."""
        if self.articulation is not None:
            return
        self.articulation = self.world.scene.add(
            SingleArticulation(prim_path=FRANKA_PRIM, name="panda_manipulator")
        )

    def _generate_local_assets(self) -> None:
        try:
            ensure_asset_usd(ASSET_PORT, build_target_port_asset)
            ensure_asset_usd(ASSET_SERVER, build_plug_tip_asset)
            ensure_asset_usd(ASSET_CABLE, build_compliant_cable_asset)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write programmatic assets under {REPO_ROOT / 'assets'}"
            ) from exc

    def _configure_physics_scene(self) -> None:
        physics_scene_path = "/World/physicsScene"
        if not self.stage.GetPrimAtPath(physics_scene_path).IsValid():
            scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path(physics_scene_path))
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)

        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(
            self.stage.GetPrimAtPath(physics_scene_path)
        )
        physx_scene_api.CreateEnableCCDAttr(True)
        physx_scene_api.CreateEnableStabilizationAttr(True)
        physx_scene_api.CreateBroadphaseTypeAttr("MBP")
        physx_scene_api.CreateSolverTypeAttr("TGS")
        physx_scene_api.GetPrim().CreateAttribute(
            "physxScene:reorderArticulationContactConstraintsLast",
            Sdf.ValueTypeNames.Bool,
        ).Set(True)

        material_path = "/World/physicsMaterial_sliding"
        if not self.stage.GetPrimAtPath(material_path).IsValid():
            material = UsdShade.Material.Define(self.stage, Sdf.Path(material_path))
            mat_prim = material.GetPrim()
            mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
            mat_api.CreateStaticFrictionAttr(0.12)
            mat_api.CreateDynamicFrictionAttr(0.08)
            mat_api.CreateRestitutionAttr(0.02)
        self._sliding_material_path = material_path

        PhysicsSchemaTools.addGroundPlane(
            self.stage,
            "/World/groundPlane",
            "Z",
            100.0,
            Gf.Vec3f(0.0, 0.0, -0.5),
            Gf.Vec3f(0.5, 0.5, 0.5),
        )
        _install_scene_lighting(self.stage)

    def _load_datacenter_rack(self) -> str:
        """Reference the 42U rack and patch panel under /World/ServerRack."""
        rack_usd, patch_usd = validate_datacenter_rack_pack()
        if not self.stage.GetPrimAtPath(SERVER_RACK_ROOT).IsValid():
            UsdGeom.Xform.Define(self.stage, Sdf.Path(SERVER_RACK_ROOT))
        _set_or_add_translate(self.stage, SERVER_RACK_ROOT, (0.0, 0.0, 0.0))

        rack_ref = RACK_PRIM
        LOGGER.info("Loading 42U rack: %s", rack_usd)
        add_reference_to_stage(rack_usd.as_posix(), rack_ref)
        if not _wait_for_prim_descendants(self.stage, rack_ref, min_children=1):
            raise RuntimeError(
                f"Timed out waiting for rack USD to compose under {rack_ref}: {rack_usd}"
            )

        if not self.stage.GetPrimAtPath(PATCH_PANEL_MOUNT).IsValid():
            mount = UsdGeom.Xform.Define(self.stage, Sdf.Path(PATCH_PANEL_MOUNT))
            xf = UsdGeom.Xformable(mount)
            xf.AddTranslateOp().Set(Gf.Vec3d(*PATCH_PANEL_MOUNT_OFFSET_M))
            xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        LOGGER.info("Loading patch panel: %s", patch_usd)
        add_reference_to_stage(patch_usd.as_posix(), PATCH_PANEL_PRIM)
        _wait_for_prim_descendants(self.stage, PATCH_PANEL_PRIM, min_children=1)

        _apply_uniform_scale(self.stage, SERVER_RACK_ROOT, 1.0)
        LOGGER.info("ServerRack scale=1.0 (USD metersPerUnit handles units); warming asset composition for 60 frames.")
        for _ in range(60):
            simulation_app.update()

        _make_prim_subtree_visible(self.stage, SERVER_RACK_ROOT)
        _verify_rack_loaded(self.stage, rack_ref)
        return rack_ref

    def _assemble_scene(self) -> None:
        resolve_assets_root_path()

        rack_bbox_prim = self._load_datacenter_rack()
        self.scene_layout = _compute_scene_layout_from_rack(self.stage, rack_bbox_prim)
        _apply_scene_layout_globals(self.scene_layout)
        LOGGER.info(
            "Scene layout: port=%s franka=%s rack_bbox_center=%s",
            self.scene_layout.port_world_pos.tolist(),
            self.scene_layout.franka_world_pos.tolist(),
            self.scene_layout.rack_obstacle_center.tolist(),
        )

        _install_datacenter_floor(self.stage, self.scene_layout)

        franka_usd = resolve_franka_usd_path()
        LOGGER.info("Loading Franka from local assets: %s", franka_usd)
        add_reference_to_stage(franka_usd, FRANKA_PRIM)
        self._franka_base_xform = XFormPrim(FRANKA_PRIM, name="FrankaBasePose")
        self._franka_base_xform.set_world_poses(
            positions=np.asarray([self.scene_layout.franka_world_pos], dtype=np.float64),
            orientations=np.asarray([self.scene_layout.franka_world_orient], dtype=np.float64),
        )
        self._franka_hand_path = self._wait_for_descendant_by_name(FRANKA_PRIM, "panda_hand")
        if self._franka_hand_path is None:
            raise RuntimeError(
                f"Franka articulation did not load from {franka_usd}. "
                "Ensure the full Franka asset pack is present (meshes referenced by franka.usd)."
            )

        if not self.stage.GetPrimAtPath("/World/CableAssembly").IsValid():
            UsdGeom.Xform.Define(self.stage, Sdf.Path("/World/CableAssembly"))

        add_reference_to_stage(ASSET_PORT.as_posix(), PORT_PRIM)
        add_reference_to_stage(ASSET_CABLE.as_posix(), CABLE_ROOT_PRIM)
        add_reference_to_stage(ASSET_SERVER.as_posix(), PLUG_PRIM)

        self._port_geom_path = self._resolve_existing_prim_path(
            [
                f"{PORT_PRIM}/{PORT_SLEEVE_RELATIVE}",
                f"{PORT_PRIM}/TargetPort/{PORT_SLEEVE_RELATIVE}",
                PORT_PRIM,
            ]
        )
        self._plug_body_path = self._resolve_existing_prim_path(
            [
                f"{PLUG_PRIM}/{PLUG_CASING_RELATIVE}",
                f"{PLUG_PRIM}/PlugTip/{PLUG_CASING_RELATIVE}",
                PLUG_PRIM,
            ]
        )
        self._cable_segment0_path = self._resolve_existing_prim_path(
            [f"{CABLE_ROOT_PRIM}/segment_0", f"{CABLE_ROOT_PRIM}/CableRoot/segment_0", CABLE_ROOT_PRIM]
        )
        self._cable_last_segment_path = self._resolve_existing_prim_path(
            [
                f"{CABLE_ROOT_PRIM}/segment_{CABLE_NUM_SEGMENTS - 1}",
                f"{CABLE_ROOT_PRIM}/CableRoot/segment_{CABLE_NUM_SEGMENTS - 1}",
            ]
        )
        _disable_collisions_between_subtrees(self.stage, FRANKA_PRIM, "/World/CableAssembly")

        layout = self.scene_layout
        self.port = XFormPrim(self._port_geom_path, name="TargetPort")
        self.port.set_world_poses(
            positions=np.asarray([layout.port_world_pos], dtype=np.float64),
            orientations=np.asarray([layout.port_world_orient], dtype=np.float64),
        )

        plug_offset = layout.port_world_pos + layout.plug_start_offset
        self.plug = RigidPrim(self._plug_body_path, name="PlugTip")
        self.plug.set_world_poses(
            positions=np.asarray([plug_offset], dtype=np.float64),
            orientations=np.asarray([layout.port_world_orient.copy()], dtype=np.float64),
        )

        self._cable_root_xform = XFormPrim(CABLE_ROOT_PRIM, name="CableRoot")
        self._cable_root_xform.set_world_poses(
            positions=np.asarray(
                [
                    plug_offset
                    - np.array([0.0, 0.0, CABLE_NUM_SEGMENTS * CABLE_SEGMENT_LENGTH * 0.5])
                ],
                dtype=np.float64,
            ),
            orientations=np.asarray([layout.port_world_orient.copy()], dtype=np.float64),
        )

        obs_center = layout.rack_obstacle_center
        obs_scale = layout.rack_obstacle_scale
        self.rack_obstacle = self.world.scene.add(
            DynamicCuboid(
                prim_path=RACK_OBSTACLE_PRIM,
                name="rack_collision_proxy",
                position=obs_center.astype(np.float32),
                scale=obs_scale.astype(np.float32),
                color=np.array([0.25, 0.25, 0.28]),
            )
        )
        _set_prim_visibility(self.stage, RACK_OBSTACLE_PRIM, False)

        self.port_proxy = self.world.scene.add(
            DynamicCuboid(
                prim_path=PORT_PROXY_PRIM,
                name="port_proxy",
                position=layout.port_world_pos.astype(np.float32),
                scale=np.array([0.03, 0.03, 0.05], dtype=np.float32),
                color=np.array([0.15, 0.15, 0.18]),
            )
        )
        _set_prim_visibility(self.stage, PORT_PROXY_PRIM, False)

        _apply_sliding_material(self.stage, self._port_geom_path, self._sliding_material_path)
        _apply_sliding_material(self.stage, self._plug_body_path, self._sliding_material_path)

        self._last_finite_plug_pose = (
            plug_offset.copy(),
            layout.port_world_orient.copy(),
        )

        if MDL_MESH_GRILLE.is_file():
            _bind_mdl(self.stage, self._port_geom_path, MDL_MESH_GRILLE)
        if MDL_METAL_ALUMINUM.is_file():
            _bind_mdl(self.stage, self._plug_body_path, MDL_METAL_ALUMINUM)
        if MDL_RUBBER.is_file():
            for i in range(CABLE_NUM_SEGMENTS):
                seg = f"{CABLE_ROOT_PRIM}/segment_{i}"
                if self.stage.GetPrimAtPath(seg).IsValid():
                    _bind_mdl(self.stage, seg, MDL_RUBBER)
                    break

    def _resolve_existing_prim_path(self, candidates: List[str]) -> str:
        for candidate in candidates:
            if self.stage.GetPrimAtPath(candidate).IsValid():
                return candidate
        raise RuntimeError(f"None of the expected prim paths were found: {candidates}")

    def _find_descendant_by_name(self, root_path: str, prim_name: str) -> Optional[str]:
        root = self.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            if prim.GetName() == prim_name:
                return prim.GetPath().pathString
        return None

    def _wait_for_descendant_by_name(
        self, root_path: str, prim_name: str, max_updates: int = 180
    ) -> Optional[str]:
        for _ in range(max_updates):
            found = self._find_descendant_by_name(root_path, prim_name)
            if found is not None:
                return found
            simulation_app.update()
        return None

    def _set_cable_segment_kinematic(self, segment_index: int, enabled: bool) -> None:
        for candidate in (
            f"{CABLE_ROOT_PRIM}/segment_{segment_index}",
            f"{CABLE_ROOT_PRIM}/CableRoot/segment_{segment_index}",
        ):
            prim = self.stage.GetPrimAtPath(candidate)
            if not prim.IsValid():
                continue
            if hasattr(PhysxSchema, "PhysxRigidBodyAPI"):
                rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                if hasattr(rb_api, "CreateKinematicEnabledAttr"):
                    rb_api.CreateKinematicEnabledAttr(enabled)
            break

    def _set_cable_segments_kinematic(self, enabled: bool) -> None:
        """Set kinematic mode on every cable segment."""
        for i in range(CABLE_NUM_SEGMENTS):
            self._set_cable_segment_kinematic(i, enabled)

    def _configure_cable_physics_mode(self) -> None:
        """Root segment follows hand; tail segments sway under joint compliance."""
        if CABLE_SWAY_ENABLED:
            self._set_cable_segment_kinematic(0, True)
            for i in range(1, CABLE_NUM_SEGMENTS):
                self._set_cable_segment_kinematic(i, False)
        else:
            self._set_cable_segments_kinematic(True)

    def _sync_cable_assembly_to_hand(self) -> None:
        """Kinematic follow: move cable root and plug with the gripper (no physics weld)."""
        if USE_GRIPPER_PHYSICS_WELD:
            return
        hand_path = self._franka_hand_path
        if not self.stage.GetPrimAtPath(hand_path).IsValid():
            return
        hand_pos, hand_quat = self._get_pose_from_usd(hand_path)
        if not _poses_are_finite(hand_pos, hand_quat):
            return

        cable_pos = hand_pos + _quat_rotate_wxyz(hand_quat, CABLE_ROOT_HAND_OFFSET)
        plug_pos = cable_pos + _quat_rotate_wxyz(hand_quat, PLUG_OFFSET_FROM_CABLE_ROOT)

        if self._cable_root_xform is None:
            self._cable_root_xform = XFormPrim(CABLE_ROOT_PRIM, name="CableRoot")
        self._cable_root_xform.set_world_poses(
            positions=np.asarray([cable_pos], dtype=np.float64),
            orientations=np.asarray([hand_quat], dtype=np.float64),
        )
        if self.plug is not None:
            self.plug.set_world_poses(
                positions=np.asarray([plug_pos], dtype=np.float64),
                orientations=np.asarray([hand_quat], dtype=np.float64),
            )

    def _create_gripper_weld_joint(self) -> None:
        """FixedJoint weld — must exist before the first world.reset()."""
        if not self.stage.GetPrimAtPath(self._franka_hand_path).IsValid():
            refreshed = self._wait_for_descendant_by_name(FRANKA_PRIM, "panda_hand", max_updates=120)
            if refreshed is not None:
                self._franka_hand_path = refreshed
        hand_prim = self.stage.GetPrimAtPath(self._franka_hand_path)
        cable_prim = self.stage.GetPrimAtPath(CABLE_ROOT_PRIM)
        if not hand_prim.IsValid():
            fallback_hand = self.stage.GetPrimAtPath(FRANKA_PRIM)
            if fallback_hand.IsValid():
                LOGGER.warning(
                    "Franka hand link not found at %s; falling back weld body to %s",
                    self._franka_hand_path,
                    FRANKA_PRIM,
                )
                self._franka_hand_path = FRANKA_PRIM
                hand_prim = fallback_hand
            else:
                LOGGER.warning(
                    "Franka hand link not found and robot root invalid; skipping gripper weld."
                )
                return
        if not cable_prim.IsValid():
            raise RuntimeError(f"Cable root not found: {CABLE_ROOT_PRIM}")

        if self.stage.GetPrimAtPath(GRIPPER_WELD_JOINT).IsValid():
            return

        joint = UsdPhysics.FixedJoint.Define(self.stage, Sdf.Path(GRIPPER_WELD_JOINT))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self._franka_hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self._cable_segment0_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        LOGGER.info("Gripper weld joint created at %s (pre-reset).", GRIPPER_WELD_JOINT)

        plug_weld_path = "/World/CableAssembly/plug_cable_weld"
        if not self.stage.GetPrimAtPath(plug_weld_path).IsValid():
            if self.stage.GetPrimAtPath(self._cable_last_segment_path).IsValid():
                plug_joint = UsdPhysics.FixedJoint.Define(self.stage, Sdf.Path(plug_weld_path))
                plug_joint.CreateBody0Rel().SetTargets([Sdf.Path(self._cable_last_segment_path)])
                plug_joint.CreateBody1Rel().SetTargets([Sdf.Path(self._plug_body_path)])
                plug_joint.CreateLocalPos0Attr().Set(
                    Gf.Vec3f(0.0, 0.0, CABLE_SEGMENT_LENGTH / 2.0)
                )
                plug_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def _setup_planners(self) -> None:
        if self.articulation is None:
            self._register_franka_articulation()

        rrt_cfg = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        rmp_cfg = interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")

        self.rrt = _PathPlannerCls(**rrt_cfg)
        if hasattr(self.rrt, "set_max_iterations"):
            self.rrt.set_max_iterations(5000)

        self.path_visualizer = PathPlannerVisualizer(self.articulation, self.rrt)

        self.rmpflow = RmpFlow(**rmp_cfg)
        self.articulation_policy = ArticulationMotionPolicy(self.articulation, self.rmpflow)

        if self.rack_obstacle is not None:
            self.rrt.add_obstacle(self.rack_obstacle)
            self.rmpflow.add_obstacle(self.rack_obstacle)
        if self.port_proxy is not None:
            self.rrt.add_obstacle(self.port_proxy)

        self.port_monitor = (
            InsertionPortObstacleMonitor(self.context, self.port_proxy)
            if self.port_proxy is not None
            else None
        )

    def _ensure_articulation_physics(self, log_event: bool = False) -> bool:
        """Keep World timeline playing and PhysX handles valid for control."""
        if self.articulation is None:
            return False
        if self.articulation.handles_initialized and self.world.is_playing():
            return True
        if not self.world.is_playing():
            self.world.play()
        self.world.initialize_physics()
        phys_view = self.world.physics_sim_view
        if phys_view is not None and not self.articulation.handles_initialized:
            self.articulation.initialize(phys_view)
        if log_event and not self.articulation.handles_initialized:
            LOGGER.warning(
                "Articulation physics not ready (playing=%s).",
                self.world.is_playing(),
            )
        return self.articulation.handles_initialized

    def _warmup_simulation(self, steps: int = 8) -> None:
        """Prime PhysX and the viewport (render=True keeps the UI from freezing black)."""
        for _ in range(steps):
            self.world.step(render=True, update_fabric=True)
        self._ensure_articulation_physics()

    def _setup_sensors(self) -> None:
        sensor_hz = getattr(self, "_physics_hz", 60)
        _configure_rendering_rates(sensor_hz)
        if hasattr(rep.settings, "set_physx_timestep"):
            rep.settings.set_physx_timestep(1.0 / float(sensor_hz))

        frame_w, frame_h = 1280, 720
        # Keep wrist resolution above DLSS's 300-px minimum even with quality
        # downsampling (~50%) so we don't see the upscale warning.
        wrist_w, wrist_h = 800, 600

        # Remove legacy custom rack camera if a previous run left it on the stage.
        legacy_cam = "/World/sensors/overhead_rack_camera"
        if self.stage.GetPrimAtPath(legacy_cam).IsValid():
            self.stage.RemovePrim(Sdf.Path(legacy_cam))
            LOGGER.info("Removed legacy overhead rack camera: %s", legacy_cam)

        self.wrist_camera = Camera(
            prim_path=f"{self._franka_hand_path}/wrist_camera",
            position=np.array([0.05, 0.0, 0.04]),
            resolution=(wrist_w, wrist_h),
            frequency=sensor_hz,
            orientation=np.array([0.7071, 0.0, 0.7071, 0.0]),
        )
        self.wrist_camera.initialize()
        self.wrist_camera.set_clipping_range(0.005, 10.0)
        _configure_perspective_lens(self.wrist_camera)

        self.contact_sensor = ContactSensor(
            prim_path=f"{self._plug_body_path}/Contact_Sensor",
            name="PlugContactSensor",
            frequency=getattr(self, "_physics_hz", 60),
            min_threshold=0.0,
            max_threshold=1000.0,
            radius=-1.0,
        )
        self.contact_sensor.initialize()

        RGB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _apply_kit_default_camera_view(self.stage)
        view_camera = _default_viewport_camera_path()
        view_rp = rep.create.render_product(view_camera, (frame_w, frame_h))
        self.replicator_writer = rep.writers.get("BasicWriter")
        self.replicator_writer.initialize(
            output_dir=str(RGB_OUTPUT_DIR),
            rgb=True,
            use_common_output_dir=True,
        )
        # Default Kit Perspective only; wrist RP triggers DLSS upscale warnings.
        self.replicator_writer.attach([view_rp])
        rep.orchestrator.set_capture_on_play(False)
        _prime_renderer(24)
        self.world.step(render=True)
        _reset_viewport_to_default_camera(
            self.stage, look_at_target=self.scene_layout.rack_obstacle_center
        )
        LOGGER.info(
            "Replicator RGB -> %s every %d steps | camera %s | view %s",
            RGB_OUTPUT_DIR,
            REPLICATOR_CAPTURE_INTERVAL,
            view_camera,
            list(KIT_DEFAULT_CAMERA_TRANSLATE),
        )

    def _capture_replicator_frame(self, step: int, sim_dt: float) -> None:
        """Capture RGB without pausing the timeline (pause_timeline=True breaks PhysX)."""
        if self.replicator_writer is None:
            return
        rep.orchestrator.step(pause_timeline=False, delta_time=0.0, rt_subframes=4)
        if hasattr(rep.orchestrator, "wait_until_complete"):
            rep.orchestrator.wait_until_complete()
        self._ensure_articulation_physics()

    def _joint_targets_from_action(self, action: Any) -> Optional[np.ndarray]:
        if action is None:
            return None
        for attr in ("joint_positions", "joint_position_targets"):
            if hasattr(action, attr):
                raw = getattr(action, attr)
                if raw is not None:
                    return np.squeeze(np.asarray(raw, dtype=np.float64)).reshape(-1)
        return None

    def _match_joint_vector_length(
        self, reference: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Pad or trim ``target`` to match ``reference`` (arm planners often omit gripper)."""
        ref = reference.reshape(-1)
        tgt = target.reshape(-1)
        if tgt.shape[0] == ref.shape[0]:
            return tgt
        if tgt.shape[0] < ref.shape[0]:
            padded = ref.copy()
            padded[: tgt.shape[0]] = tgt
            return padded
        return tgt[: ref.shape[0]]

    def _plan_rrt_path(self) -> None:
        self._sync_cable_assembly_to_hand()
        pre_pos, pre_rot = self._compute_pre_insertion_target()
        z_axis = self._port_z_axis()
        wrist_target = pre_pos - self._wrist_to_plug_tip_distance_m() * z_axis

        if hasattr(self.rrt, "set_end_effector_target"):
            self.rrt.set_end_effector_target(wrist_target)
        elif hasattr(self.rrt, "compute_path"):
            joint_pos = self.articulation.get_joint_positions()
            if joint_pos is None:
                joint_pos = np.zeros(9, dtype=np.float64)
            else:
                joint_pos = np.squeeze(np.asarray(joint_pos, dtype=np.float64)).reshape(-1)
            self.rrt.compute_path(
                active_joint_positions=joint_pos,
                target_position=wrist_target,
                target_orientation=pre_rot,
            )

        if hasattr(self.rrt, "update_world"):
            self.rrt.update_world()

        try:
            self.rrt_plan = list(
                self.path_visualizer.compute_plan_as_articulation_actions(
                    max_cspace_dist=RRT_MAX_CSPACE_DIST
                )
            )
            if len(self.rrt_plan) > MAX_RRT_WAYPOINTS:
                LOGGER.warning(
                    "RRT plan has %d waypoints; keeping last %d for sim time budget.",
                    len(self.rrt_plan),
                    MAX_RRT_WAYPOINTS,
                )
                self.rrt_plan = self.rrt_plan[-MAX_RRT_WAYPOINTS:]
            LOGGER.info("RRT plan: %d articulation actions.", len(self.rrt_plan))
        except Exception as exc:
            LOGGER.warning("RRT planning failed (%s); ALIGN will use RMPflow only.", exc)
            self.rrt_plan = []

    def _matrix_to_pose_wxyz(self, matrix: Gf.Matrix4d) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array([matrix[3][0], matrix[3][1], matrix[3][2]], dtype=np.float64)
        rot = matrix.ExtractRotationQuat()
        quat = np.array(
            [
                rot.GetReal(),
                rot.GetImaginary()[0],
                rot.GetImaginary()[1],
                rot.GetImaginary()[2],
            ],
            dtype=np.float64,
        )
        return pos, _flatten_quat_wxyz(quat)

    def _get_pose_from_usd(self, prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"USD prim not found for pose query: {prim_path}")
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return self._matrix_to_pose_wxyz(mat)

    def _port_z_axis(self) -> np.ndarray:
        port_prim = self.stage.GetPrimAtPath(PORT_PRIM)
        mat = UsdGeom.Xformable(port_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        z_axis = _flatten_vec3([mat[2][0], mat[2][1], mat[2][2]])
        z_axis /= max(np.linalg.norm(z_axis), 1e-9)
        return z_axis

    def _wrist_to_plug_tip_distance_m(self) -> float:
        plug_offset_from_wrist = CABLE_ROOT_HAND_OFFSET + PLUG_OFFSET_FROM_CABLE_ROOT
        return float(np.linalg.norm(plug_offset_from_wrist))

    def _refresh_scene_prim_wrappers_after_reset(self) -> None:
        """Recreate port/plug wrappers after world.reset() so physics views stay valid."""
        layout = self.scene_layout
        self._franka_base_xform = XFormPrim(FRANKA_PRIM, name="FrankaBasePose")
        self._franka_base_xform.set_world_poses(
            positions=np.asarray([layout.franka_world_pos], dtype=np.float64),
            orientations=np.asarray([layout.franka_world_orient], dtype=np.float64),
        )

        plug_offset = layout.port_world_pos + layout.plug_start_offset
        cable_center = plug_offset - np.array(
            [0.0, 0.0, CABLE_NUM_SEGMENTS * CABLE_SEGMENT_LENGTH * 0.5], dtype=np.float64
        )

        self.port = XFormPrim(self._port_geom_path, name="TargetPort")
        self.port.set_world_poses(
            positions=np.asarray([layout.port_world_pos], dtype=np.float64),
            orientations=np.asarray([layout.port_world_orient], dtype=np.float64),
        )

        self.plug = RigidPrim(self._plug_body_path, name="PlugTip")
        self.plug.set_world_poses(
            positions=np.asarray([plug_offset], dtype=np.float64),
            orientations=np.asarray([layout.port_world_orient.copy()], dtype=np.float64),
        )

        self._cable_root_xform = XFormPrim(CABLE_ROOT_PRIM, name="CableRoot")
        self._cable_root_xform.set_world_poses(
            positions=np.asarray([cable_center], dtype=np.float64),
            orientations=np.asarray([layout.port_world_orient.copy()], dtype=np.float64),
        )
        self._configure_cable_physics_mode()

    def _compute_pre_insertion_target(self) -> Tuple[np.ndarray, np.ndarray]:
        p_port, q_port = self._get_world_pose(self.port, self._port_geom_path)
        z_axis = self._port_z_axis()
        pre_pos = p_port - PRE_INSERTION_OFFSET_M * z_axis
        return pre_pos, q_port.copy()

    def _get_world_pose(
        self, prim_wrapper: Any, usd_fallback_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if prim_wrapper is None:
            raise RuntimeError("Scene prim wrapper is None (port/plug/obstacle not initialized).")
        if hasattr(prim_wrapper, "get_world_pose"):
            pos, orient = prim_wrapper.get_world_pose()
        else:
            pos_batch, orient_batch = prim_wrapper.get_world_poses()
            pos = pos_batch[0]
            orient = orient_batch[0]
        pos = _flatten_vec3(pos)
        orient = _flatten_quat_wxyz(orient)
        if _poses_are_finite(pos, orient):
            return pos, orient
        if usd_fallback_path:
            usd_pos, usd_orient = self._get_pose_from_usd(usd_fallback_path)
            if _poses_are_finite(usd_pos, usd_orient):
                return usd_pos, usd_orient
        return pos, orient

    def evaluate_insertion_metrics(self) -> dict:
        p_plug, q_plug = self._get_world_pose(self.plug, self._plug_body_path)
        p_port, q_port = self._get_world_pose(self.port, self._port_geom_path)

        if not _poses_are_finite(p_port, q_port):
            p_port = self.scene_layout.port_world_pos.copy()
            q_port = self.scene_layout.port_world_orient.copy()
        if not _poses_are_finite(p_plug, q_plug):
            p_plug, q_plug = self._last_finite_plug_pose[0].copy(), self._last_finite_plug_pose[1].copy()
        else:
            self._last_finite_plug_pose = (p_plug.copy(), q_plug.copy())

        if not _poses_are_finite(p_plug, q_plug):
            self._physics_unstable = True
            return {
                "translation_error": float("inf"),
                "angular_error": float("inf"),
                "insertion_depth": 0.0,
                "contact_force": self._read_contact_force(),
            }

        # During ALIGN the robot targets the pre-insertion standoff (PRE_INSERTION_OFFSET_M
        # along the port's -Z axis), so the relevant translation error is plug->pre_pos.
        # INSERT/VERIFY keep measuring against the port itself.
        if self.context.task_state == TaskState.ALIGN:
            pre_pos, _ = self._compute_pre_insertion_target()
            translation_error = float(np.linalg.norm(p_plug - pre_pos))
        else:
            translation_error = float(np.linalg.norm(p_plug - p_port))

        q_inv = np.array([q_plug[0], -q_plug[1], -q_plug[2], -q_plug[3]])
        w1, x1, y1, z1 = q_inv
        w2, x2, y2, z2 = q_port
        q_err_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        angular_error_rad = float(2.0 * np.arccos(np.clip(abs(q_err_w), 0.0, 1.0)))

        port_prim = self.stage.GetPrimAtPath(PORT_PRIM)
        port_matrix = UsdGeom.Xformable(port_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        z_axis_unit = _flatten_vec3([port_matrix[2][0], port_matrix[2][1], port_matrix[2][2]])
        z_axis_unit /= max(np.linalg.norm(z_axis_unit), 1e-9)
        insertion_depth = float(np.dot(p_plug - p_port, z_axis_unit))

        return {
            "translation_error": translation_error,
            "angular_error": angular_error_rad,
            "insertion_depth": insertion_depth,
            "contact_force": self._read_contact_force(),
        }

    def _read_contact_force(self) -> float:
        if self.contact_sensor is None:
            return 0.0
        try:
            frame = self.contact_sensor.get_current_frame()
        except Exception:
            frame = None
        if not frame:
            return 0.0
        if not frame.get("in_contact", False):
            return 0.0
        force = frame.get("force")
        if force is None:
            if not self._contact_warned:
                LOGGER.warning("Contact frame missing 'force'; treating as 0 N.")
                self._contact_warned = True
            return 0.0
        return float(np.linalg.norm(_flatten_vec3(force)))

    def run_simulation_loop(self, max_steps: int = MAX_SIM_STEPS) -> dict:
        self._setup_planners()
        self.world.reset()
        self._refresh_scene_prim_wrappers_after_reset()
        sim_dt = self.world.get_physics_dt()
        self._physics_hz = max(1, int(round(1.0 / sim_dt)))
        _start_physics_timeline(self._physics_hz)
        self._warmup_simulation()
        _prime_renderer()
        if not self._ensure_articulation_physics(log_event=True):
            raise RuntimeError("Franka physics view failed to initialize after world.reset().")
        self._plan_rrt_path()
        if self.replicator_writer is None:
            self._setup_sensors()
        self._ensure_articulation_physics(log_event=True)
        LOGGER.info(
            "Simulation loop: physics_hz=%d capture_every=%d max_steps=%d",
            self._physics_hz,
            REPLICATOR_CAPTURE_INTERVAL,
            max_steps,
        )

        if self.port_monitor:
            self.port_monitor.enable()

        self.context.task_state = TaskState.APPROACH
        termination_reason = "max_steps"
        success = False

        for step in range(max_steps):
            self._ensure_articulation_physics()
            self._advance_control(sim_dt)
            capture_frame = REPLICATOR_CAPTURE_INTERVAL > 0 and (
                step % REPLICATOR_CAPTURE_INTERVAL == 0 or step == max_steps - 1
            )
            self.world.step(render=True, update_fabric=True)
            if capture_frame:
                self._capture_replicator_frame(step, sim_dt)

            metrics = self.evaluate_insertion_metrics()
            log_entry = {"step": step, "state": self.context.task_state.name, **metrics}
            self.step_log.append(log_entry)

            def _fmt_metric(value: float) -> str:
                return f"{value:.4f}" if np.isfinite(value) else "inf"

            print(
                f"Step {step:03d} | State: {self.context.task_state.name} | "
                f"TransErr: {_fmt_metric(metrics['translation_error'])} m | "
                f"RotErr: {_fmt_metric(metrics['angular_error'])} rad | "
                f"Depth: {_fmt_metric(metrics['insertion_depth'])} m | "
                f"Force: {metrics['contact_force']:.2f} N"
            )

            if self._physics_unstable:
                termination_reason = "physics_unstable"
                LOGGER.error("Physics unstable (non-finite poses); stopping at step %d.", step)
                break

            if metrics["contact_force"] > FORCE_ABORT_N:
                termination_reason = "force_threshold_exceeded"
                break

            if self._check_success(metrics):
                success = True
                termination_reason = "insertion_tolerance_met"
                break

            self._update_state_machine(metrics)

        rgb_files = list(RGB_OUTPUT_DIR.glob("**/*.png"))
        LOGGER.info(
            "RGB frames on disk: %d under %s | termination=%s",
            len(rgb_files),
            RGB_OUTPUT_DIR,
            termination_reason,
        )

        result = {
            "use_case": "cable_plugin",
            "isaac_sim_version": "5.1.0",
            "success": success,
            "termination_reason": termination_reason,
            "final_metrics": self.step_log[-1] if self.step_log else {},
            "steps": self.step_log,
        }
        self._write_labels(result)
        return result

    def _apply_ready_pose_step(self) -> None:
        if self.articulation is None or not self.articulation.handles_initialized:
            return
        start = self.articulation.get_joint_positions()
        if start is None:
            self._ready_pose_steps_left = 0
            return
        start = np.squeeze(np.asarray(start, dtype=np.float64)).reshape(-1)
        target = self._match_joint_vector_length(start, READY_POSE_JOINTS)
        alpha = 1.0 - float(self._ready_pose_steps_left) / float(READY_POSE_STEPS)
        interp = (1.0 - alpha) * start + alpha * target
        self.articulation.apply_action(ArticulationAction(joint_positions=interp))
        self._ready_pose_steps_left -= 1

    def _begin_next_rrt_action(self) -> None:
        if not self.rrt_plan or self.articulation is None:
            return
        action = self.rrt_plan.pop(0)
        target = self._joint_targets_from_action(action)
        start = self.articulation.get_joint_positions()
        if target is None or start is None:
            self.articulation.apply_action(action)
            return
        self._rrt_interp_action = action
        start_j = np.squeeze(np.asarray(start, dtype=np.float64)).reshape(-1)
        self._rrt_interp_start_joints = start_j
        self._rrt_interp_target_joints = self._match_joint_vector_length(start_j, target)
        self._rrt_interp_step = 0

    def _step_rrt_interpolation(self) -> bool:
        """Interpolate one RRT waypoint; return True when segment is complete."""
        if self._rrt_interp_start_joints is None or self._rrt_interp_target_joints is None:
            return True
        alpha = float(self._rrt_interp_step + 1) / float(RRT_SUBSTEPS)
        interp = (1.0 - alpha) * self._rrt_interp_start_joints + alpha * self._rrt_interp_target_joints
        self.articulation.apply_action(ArticulationAction(joint_positions=interp))
        self._rrt_interp_step += 1
        return self._rrt_interp_step >= RRT_SUBSTEPS

    def _advance_approach_control(self) -> None:
        if self.articulation is None or not self.articulation.handles_initialized:
            return
        if self._ready_pose_steps_left > 0:
            self._apply_ready_pose_step()
        elif self._rrt_interp_action is not None:
            if self._step_rrt_interpolation():
                self._rrt_interp_action = None
                self._rrt_interp_start_joints = None
                self._rrt_interp_target_joints = None
        elif self.rrt_plan:
            self._begin_next_rrt_action()
        if self.port_monitor and self.rmpflow:
            self.port_monitor.sync(self.rmpflow)
        self._sync_cable_assembly_to_hand()

    def _advance_control(self, sim_dt: float) -> None:
        state = self.context.task_state

        if state == TaskState.APPROACH:
            self._advance_approach_control()
            return

        pre_pos, pre_rot = self._compute_pre_insertion_target()
        z_axis = self._port_z_axis()

        if state == TaskState.ALIGN:
            self._align_rmp_steps += 1
            blend = min(1.0, self._align_rmp_steps / float(ALIGN_RMP_RAMP_STEPS))
            hand_path = self._franka_hand_path
            if self.stage.GetPrimAtPath(hand_path).IsValid():
                ee_pos, ee_rot = self._get_pose_from_usd(hand_path)
            else:
                ee_pos, ee_rot = pre_pos, pre_rot
            # Latch the blend origin on the first ALIGN tick so the linear blend
            # actually reaches pre_pos when blend hits 1.0. Re-sampling ee_pos
            # every frame here would walk the origin toward the target and
            # produce an asymptotic curve that stalls short of pre_pos.
            if self._align_start_ee_pos is None:
                self._align_start_ee_pos = ee_pos.copy()
            start_pos = self._align_start_ee_pos
            target_pos = (1.0 - blend) * start_pos + blend * pre_pos
            target_rot = pre_rot
            step_delta = target_pos - ee_pos
            step_norm = float(np.linalg.norm(step_delta))
            if step_norm > 0.012:
                target_pos = ee_pos + step_delta * (0.012 / step_norm)
        elif state == TaskState.INSERT:
            p_port, _ = self._get_world_pose(self.port, self._port_geom_path)
            target_pos = p_port - TARGET_INSERTION_DEPTH_M * z_axis
            target_rot = pre_rot
        else:
            target_pos, target_rot = pre_pos, pre_rot

        target_pos = target_pos - self._wrist_to_plug_tip_distance_m() * z_axis

        if not _poses_are_finite(target_pos, target_rot):
            return

        joint_pos = self.articulation.get_joint_positions()
        if joint_pos is None:
            LOGGER.warning("Skipping RMPflow step: articulation joint state unavailable.")
            return

        self.rmpflow.set_end_effector_target(target_pos, target_rot)
        self.rmpflow.update_world()

        base_pos, base_rot = self.articulation.get_world_pose()
        base_pos = _flatten_vec3(base_pos)
        base_rot = _flatten_quat_wxyz(base_rot)
        if not _poses_are_finite(base_pos, base_rot):
            return

        self.rmpflow.set_robot_base_pose(base_pos, base_rot)

        if self.port_monitor:
            self.port_monitor.sync(self.rmpflow)

        action = self.articulation_policy.get_next_articulation_action(sim_dt)
        self.articulation.apply_action(action)
        self._sync_cable_assembly_to_hand()

    def _update_state_machine(self, metrics: dict) -> None:
        state = self.context.task_state

        if (
            state == TaskState.APPROACH
            and not self.rrt_plan
            and self._rrt_interp_action is None
            and self._ready_pose_steps_left <= 0
        ):
            self.context.task_state = TaskState.ALIGN
            self._align_rmp_steps = 0
            self._align_start_ee_pos = None
            LOGGER.info("Transition APPROACH -> ALIGN")

        elif state == TaskState.ALIGN:
            if (
                np.isfinite(metrics["translation_error"])
                and np.isfinite(metrics["angular_error"])
                and metrics["translation_error"] < ALIGN_TRANS_TOL_M
                and metrics["angular_error"] < ALIGN_ROT_TOL_RAD
            ):
                self.context.task_state = TaskState.INSERT
                LOGGER.info("Transition ALIGN -> INSERT (port obstacle disabled)")

        elif state == TaskState.INSERT:
            if abs(metrics["insertion_depth"]) >= TARGET_INSERTION_DEPTH_M * 0.9:
                self.context.task_state = TaskState.VERIFY
                LOGGER.info("Transition INSERT -> VERIFY")

    def _check_success(self, metrics: dict) -> bool:
        if self.context.task_state != TaskState.VERIFY:
            return False
        return (
            abs(abs(metrics["insertion_depth"]) - TARGET_INSERTION_DEPTH_M) <= SUCCESS_DEPTH_M
            and metrics["translation_error"] < SUCCESS_TRANS_TOL_M
        )

    def _write_labels(self, payload: dict) -> None:
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label_path = LABELS_DIR / f"run_{stamp}.json"
        try:
            with label_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            LOGGER.info("Wrote run labels: %s", label_path)
        except OSError as exc:
            LOGGER.error("Failed to write labels: %s", exc)

    def shutdown(self) -> None:
        if self.port_monitor:
            self.port_monitor.disable(self.rmpflow)
        if self.replicator_writer is not None:
            try:
                self.replicator_writer.detach()
            except Exception:
                pass


def _exit_on_sim_complete() -> bool:
    """When True, close Kit after the simulation loop (for headless/CI)."""
    return os.environ.get("ISAAC_EXIT_ON_COMPLETE", "").lower() in ("1", "true", "yes")


def main() -> int:
    orchestrator: Optional[CablePluginOrchestrator] = None
    exit_code = 0
    try:
        orchestrator = CablePluginOrchestrator()
        result = orchestrator.run_simulation_loop()
        LOGGER.info(
            "Run finished — success=%s reason=%s",
            result.get("success"),
            result.get("termination_reason"),
        )
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if orchestrator is not None:
            orchestrator.shutdown()

    if _exit_on_sim_complete():
        simulation_app.close()
        return exit_code

    LOGGER.info(
        "Simulation complete. Isaac Sim will stay open — close the Kit window to exit."
    )
    while simulation_app.is_running():
        simulation_app.update()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
