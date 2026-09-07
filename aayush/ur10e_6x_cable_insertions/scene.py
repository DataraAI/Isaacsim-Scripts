"""Load DataHall_6r.usd and attach Lula + gripper handles for six UR10e stations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from ur10e_6x_cable_insertions import config as cfg
from ur10e_6x_cable_insertions.controller import SixArmMotionController
from ur10e_6x_cable_insertions.primitives import (
    compute_port_targets,
    meters_per_unit,
    prim_bbox,
)


@dataclass
class StationBundle:
    spec: cfg.StationSpec
    robot: Any
    motion_controller: Any
    end_effector_path: str
    grasp_part_path: str
    path45: str
    path39: str
    block_top_z: float
    observe_hand: np.ndarray
    port_contacts_path: str


@dataclass
class SceneBundle:
    world: Any
    stage: Any
    stations: list[StationBundle] = field(default_factory=list)


def wait_for_stage_loading(simulation_app) -> None:
    import omni.usd

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


def open_datahall_stage(simulation_app, usd_path: Path):
    """Replace the default stage with DataHall_6r.usd and return (stage, world)."""

    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager

    path = Path(usd_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DataHall 6-robot USD not found: {path}")

    usd_context = omni.usd.get_context()
    opened = usd_context.open_stage(str(path))
    if opened is False:
        raise RuntimeError(f"Could not open stage: {path}")
    wait_for_stage_loading(simulation_app)
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError(f"Stage is empty after opening {path}")

    mpu = meters_per_unit(stage)
    world = World(stage_units_in_meters=mpu)
    world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    _enable_gpu_dynamics(stage, SimulationManager)
    print(f"[SCENE] Opened {path} metersPerUnit={mpu}")
    return stage, world


def _enable_gpu_dynamics(stage, SimulationManager) -> None:
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()]
    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(True).Set(True)
        api.CreateBroadphaseTypeAttr("GPU").Set("GPU")
        api.CreateSolverTypeAttr("TGS").Set("TGS")
    try:
        for scene in SimulationManager.get_physics_scenes():
            scene.set_enabled_gpu_dynamics(True)
    except Exception as exc:
        print(f"[SCENE] GPU dynamics on SimulationManager warning: {exc}")
    print("[SCENE] PhysX GPU dynamics enabled")


def find_descendant(stage, root_path: str, name: str) -> str | None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    wanted = name.lower()
    for prim in Usd.PrimRange(root):
        if prim.GetName().lower() == wanted:
            return str(prim.GetPath())
    return None


def _strip_rigid_body_api(prim) -> None:
    try:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    except Exception:
        attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if attr and attr.IsValid():
            try:
                attr.Set(False)
            except Exception:
                pass


def enable_crystal_head_physics(stage, path45: str, path39: str) -> None:
    """One rigid body per crystal head; mesh descendants get convexHull collision."""

    from omni.physx.scripts import utils as physx_utils

    for head_path in (path45, path39):
        head = stage.GetPrimAtPath(head_path)
        if not head or not head.IsValid():
            print(f"[SCENE] Skip physics: missing {head_path}")
            continue
        for prim in Usd.PrimRange(head):
            if prim == head:
                continue
            _strip_rigid_body_api(prim)
        try:
            rb = UsdPhysics.RigidBodyAPI.Apply(head)
            rb.CreateRigidBodyEnabledAttr(True).Set(True)
        except Exception as exc:
            print(f"[SCENE] RigidBodyAPI failed on {head_path}: {exc}")
            continue
        try:
            mass = UsdPhysics.MassAPI.Apply(head)
            mass.CreateMassAttr(0.02).Set(0.02)
        except Exception:
            pass
        enabled_meshes = 0
        for prim in Usd.PrimRange(head):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            try:
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_api.CreateApproximationAttr().Set("convexHull")
                try:
                    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
                except Exception:
                    pass
                enabled_meshes += 1
            except Exception as exc:
                print(f"[SCENE] collision setup failed on {prim.GetPath()}: {exc}")
        try:
            physx_utils.setCollider(head, approximationShape="convexHull")
        except Exception:
            pass
        print(
            f"[SCENE] Crystal-head physics enabled at {head_path} "
            f"({enabled_meshes} mesh collision(s), single RigidBody)"
        )


def resolve_grasp_part_path(stage, spec: cfg.StationSpec) -> str:
    part = stage.GetPrimAtPath(spec.grasp_part_path)
    if part and part.IsValid():
        return spec.grasp_part_path
    found = find_descendant(stage, spec.path45, cfg.GRASP_PART_NAME)
    if found:
        return found
    raise RuntimeError(f"Missing grasp part {cfg.GRASP_PART_NAME} under {spec.path45}")


def resolve_end_effector_path(stage, robot_prim_path: str) -> str:
    gripper = stage.GetPrimAtPath(f"{robot_prim_path}/ee_link/Robotiq_2F_85")
    if gripper and gripper.IsValid():
        base = find_descendant(stage, str(gripper.GetPath()), "base_link")
        if base:
            return base
        return str(gripper.GetPath())
    ee_link = stage.GetPrimAtPath(f"{robot_prim_path}/ee_link")
    if ee_link and ee_link.IsValid():
        return str(ee_link.GetPath())
    raise RuntimeError(f"Missing Robotiq / ee_link under {robot_prim_path}")


def _ensure_physics_material(
    stage,
    material_path: str,
    *,
    static_friction: float,
    dynamic_friction: float,
    combine_mode: str,
) -> str:
    if not stage.GetPrimAtPath("/World/PhysicsMaterials").IsValid():
        UsdGeom.Xform.Define(stage, "/World/PhysicsMaterials")
    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()
    mat_api = UsdPhysics.MaterialAPI.Apply(prim)
    mat_api.CreateStaticFrictionAttr(float(static_friction)).Set(float(static_friction))
    mat_api.CreateDynamicFrictionAttr(float(dynamic_friction)).Set(float(dynamic_friction))
    mat_api.CreateRestitutionAttr(0.0).Set(0.0)
    try:
        px_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        px_mat.CreateFrictionCombineModeAttr().Set(str(combine_mode))
    except Exception as exc:
        print(f"[SCENE] PhysxMaterialAPI friction combine warning on {material_path}: {exc}")
    return material_path


def _bind_physics_material(prim, material_path: str) -> None:
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        UsdShade.Material.Get(prim.GetStage(), material_path),
        UsdShade.Tokens.strongerThanDescendants,
        "physics",
    )


def _prim_matches_finger_token(prim) -> bool:
    name = prim.GetName().lower()
    path = str(prim.GetPath()).lower()
    for token in cfg.FINGERTIP_NAME_TOKENS:
        t = token.lower()
        if t in name or t in path:
            return True
    return False


def apply_grasp_friction_materials(stage, robot_prim_path: str, path45: str, path39: str) -> None:
    mat_path = _ensure_physics_material(
        stage,
        "/World/PhysicsMaterials/fingertip_material",
        static_friction=float(cfg.GRASP_FRICTION_STATIC),
        dynamic_friction=float(cfg.GRASP_FRICTION_DYNAMIC),
        combine_mode=str(cfg.GRASP_FRICTION_COMBINE_MODE),
    )
    finger_hits = 0
    robot = stage.GetPrimAtPath(robot_prim_path)
    if robot and robot.IsValid():
        for prim in Usd.PrimRange(robot):
            if not _prim_matches_finger_token(prim):
                continue
            try:
                _bind_physics_material(prim, mat_path)
                finger_hits += 1
            except Exception as exc:
                print(f"[SCENE] fingertip material bind failed on {prim.GetPath()}: {exc}")
    head_hits = 0
    for head_path in (path45, path39):
        head = stage.GetPrimAtPath(head_path)
        if not head or not head.IsValid():
            continue
        for prim in Usd.PrimRange(head):
            if not (prim.IsA(UsdGeom.Mesh) or prim == head):
                continue
            try:
                _bind_physics_material(prim, mat_path)
                head_hits += 1
            except Exception as exc:
                print(f"[SCENE] head material bind failed on {prim.GetPath()}: {exc}")
    print(
        f"[SCENE] Grasp friction {mat_path} bound on {finger_hits} fingertip prim(s), "
        f"{head_hits} head prim(s) for {robot_prim_path}"
    )


def configure_robot_physics(stage, root_path: str) -> None:
    """Enable link collision; disable self-collision; keep gravity off."""

    from omni.physx.scripts import utils as physx_utils

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        try:
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(False).Set(
                False
            )
        except Exception:
            attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            if attr and attr.IsValid():
                attr.Set(False)
    collision_count = 0
    convex_count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                rb = UsdPhysics.RigidBodyAPI(prim)
                rb.CreateRigidBodyEnabledAttr(True).Set(True)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True).Set(True)
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
        try:
            if prim.IsA(UsdGeom.Mesh) or prim.IsInstanceable():
                physx_utils.setCollider(prim, UsdPhysics.Tokens.convexHull)
            else:
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            convex_count += 1
        except Exception as exc:
            print(f"[SCENE] convexHull skip {prim.GetPath()}: {exc}")
    print(
        f"[SCENE] Robot physics {root_path}: collisions={collision_count} "
        f"convexHull={convex_count} gravity=off"
    )


def strip_physics_from_prim(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    for child in Usd.PrimRange(prim):
        try:
            if child.HasAPI(UsdPhysics.CollisionAPI):
                child.RemoveAPI(UsdPhysics.CollisionAPI)
        except Exception:
            attr = child.GetAttribute("physics:collisionEnabled")
            if attr and attr.IsValid():
                attr.Set(False)
        try:
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        except Exception:
            pass
        for attr_name in ("physics:collisionEnabled", "physics:rigidBodyEnabled"):
            attr = child.GetAttribute(attr_name)
            if attr and attr.IsValid():
                try:
                    attr.Set(False)
                except Exception:
                    pass


def disable_robot_ros_graphs(stage, robot_prim_path: str) -> None:
    """ROS ActionGraphs would fight the Lula articulation controller."""

    for suffix in ("/ROS_ActionGraph",):
        prim = stage.GetPrimAtPath(f"{robot_prim_path}{suffix}")
        if prim and prim.IsValid():
            try:
                prim.SetActive(False)
                print(f"[SCENE] Disabled {prim.GetPath()}")
            except Exception as exc:
                print(f"[SCENE] Could not disable {prim.GetPath()}: {exc}")


def apply_ur10e_home_pose(robot, *, apply_live: bool = False) -> None:
    try:
        names = list(robot.dof_names)
    except Exception:
        names = []
    positions = np.zeros(len(names) if names else 6, dtype=np.float64)
    for i, name in enumerate(names or cfg.UR10E_ARM_JOINT_NAMES):
        if name in cfg.UR10E_ARM_JOINT_NAMES:
            positions[i] = float(cfg.UR10E_HOME_ARM[cfg.UR10E_ARM_JOINT_NAMES.index(name)])
    try:
        robot.set_joints_default_state(positions=positions)
    except Exception as exc:
        print(f"[SCENE] set_joints_default_state warning: {exc}")
    if apply_live:
        try:
            robot.set_joint_positions(positions)
        except Exception as exc:
            print(f"[SCENE] set_joint_positions warning: {exc}")


def _port_tip_via(tip_start: np.ndarray, tip_end: np.ndarray, frac: float) -> np.ndarray:
    t = float(np.clip(frac, 0.0, 1.0))
    tip = (1.0 - t) * tip_start + t * tip_end
    if t < 1.0 - 1e-9:
        tip = tip.copy()
        tip[2] = max(float(tip[2]), float(tip_end[2])) + float(cfg.PORT_APPROACH_VIA_Z_CLEARANCE_M)
    return tip


def _spawn_debug_sphere(
    stage,
    prim_path: str,
    center_m: np.ndarray,
    color_rgb: tuple[float, float, float],
    *,
    scale_m: float,
    visible: bool = False,
) -> None:
    """Invisible, non-colliding marker. ``center_m`` / ``scale_m`` are meters."""

    mpu = meters_per_unit(stage)
    inv = (1.0 / mpu) if mpu > 1e-12 else 1.0
    center = np.asarray(center_m, dtype=np.float64).reshape(3) * inv
    scale = float(scale_m) * inv
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
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale, scale, scale))
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
    strip_physics_from_prim(stage, prim_path)


def spawn_station_debug_markers(stage, spec: cfg.StationSpec, grasp_part_path: str) -> None:
    """Yellow offset, red insert, green via spheres. Invisible; no physics."""

    try:
        insert, approach, path_a, path_b = compute_port_targets(stage, spec.port_contacts_path)
    except Exception as exc:
        print(f"[SCENE] Debug markers skipped for {spec.station_id}: {exc}")
        return
    try:
        _mn, _mx, part_center = prim_bbox(stage, grasp_part_path)
    except Exception as exc:
        print(f"[SCENE] Debug via markers skipped for {spec.station_id}: {exc}")
        return

    tip_start = np.asarray(part_center, dtype=np.float64).copy()
    tip_start[0] += float(cfg.GRASP_X_OFFSET_M)
    tip_start[2] += (
        float(cfg.GRASP_DESCEND_CLEARANCE_M)
        + float(cfg.GRASP_LIFT_CLEARANCE_M)
        + abs(float(cfg.GRASP_DESCEND_CLEARANCE_M))
    )
    tip_end = np.asarray(approach, dtype=np.float64).copy()

    UsdGeom.Xform.Define(stage, Sdf.Path(cfg.DEBUG_MARKER_ROOT))
    root = spec.debug_marker_root
    if stage.GetPrimAtPath(root).IsValid():
        stage.RemovePrim(Sdf.Path(root))
    UsdGeom.Xform.Define(stage, Sdf.Path(root))
    visible = bool(cfg.DEBUG_MARKER_VISIBLE_DEFAULT)
    _spawn_debug_sphere(
        stage,
        f"{root}/Offset",
        approach,
        (1.0, 0.92, 0.1),
        scale_m=float(cfg.PORT_DEBUG_MARKER_SCALE_M),
        visible=visible,
    )
    _spawn_debug_sphere(
        stage,
        f"{root}/Insert",
        insert,
        (0.95, 0.12, 0.12),
        scale_m=float(cfg.PORT_DEBUG_MARKER_SCALE_M),
        visible=visible,
    )
    green = (0.15, 0.85, 0.25)
    via_scale = float(cfg.PORT_MANEUVER_MARKER_SCALE_M)
    via_points: list[tuple[str, np.ndarray]] = []
    for frac in cfg.PORT_APPROACH_VIA_FRACTIONS:
        tip = _port_tip_via(tip_start, tip_end, float(frac))
        via_points.append((f"Via_{int(round(float(frac) * 100)):02d}", tip))
    for frac in cfg.PORT_INSERT_VIA_FRACTIONS:
        t = float(frac)
        tip = (1.0 - t) * tip_end + t * insert
        via_points.append((f"InsertVia_{int(round(t * 100)):02d}", tip))
    for name, tip in via_points:
        _spawn_debug_sphere(
            stage, f"{root}/{name}", tip, green, scale_m=via_scale, visible=visible
        )
    print(
        f"[SCENE] Debug markers (invisible, no physics) under {root}:\n"
        f"  yellow Offset @ {np.round(approach, 4)}\n"
        f"  red Insert    @ {np.round(insert, 4)}\n"
        f"  green vias    {[n for n, _ in via_points]}\n"
        f"  from {path_a} / {path_b}"
    )


def _observe_hand_from_part(part_center: np.ndarray) -> np.ndarray:
    hand = np.asarray(part_center, dtype=np.float64).reshape(3).copy()
    hand[2] += float(cfg.OBSERVE_Z_CLEARANCE_M)
    return hand


def _attach_manipulator(world, spec: cfg.StationSpec, ee_path: str):
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path,
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0.0]),
        joint_closed_positions=np.array([cfg.ROBOTIQ_CLOSED_RAD]),
        action_deltas=None,
        use_mimic_joints=True,
    )
    try:
        world.scene.remove_object(spec.scene_name)
    except Exception:
        pass
    robot = world.scene.add(
        SingleManipulator(
            prim_path=spec.robot_prim_path,
            name=spec.scene_name,
            end_effector_prim_path=ee_path,
            gripper=gripper,
        )
    )
    try:
        robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
    except Exception as exc:
        print(f"[SCENE] gripper default state warning on {spec.station_id}: {exc}")
    return robot


def _make_motion_controller(stage, robot, spec: cfg.StationSpec, lula_config: dict):
    kinematics = LulaKinematicsSolver(**lula_config)
    trajectory_generator = LulaTaskSpaceTrajectoryGenerator(**lula_config)
    ee_frame = cfg.UR10E_EE_FRAME
    tool0 = stage.GetPrimAtPath(f"{spec.robot_prim_path}/tool0")
    if not (tool0 and tool0.IsValid()):
        ee_frame = cfg.UR10E_EE_FRAME_FALLBACK
        print(f"[SCENE] {spec.station_id}: tool0 missing; Lula ee_frame -> {ee_frame}")
    articulation_kinematics = ArticulationKinematicsSolver(robot, kinematics, ee_frame)
    base_position, base_orientation = robot.get_world_pose()
    kinematics.set_robot_base_pose(base_position, base_orientation)
    return SixArmMotionController(
        name=f"{spec.scene_name}_controller",
        robot_articulation=robot,
        task_traj_gen=trajectory_generator,
        art_kinematics=articulation_kinematics,
        gripper=robot.gripper,
        tool_offset=0.0,
        physics_dt=1.0 / 120.0,
        ee_frame=ee_frame,
        debug=True,
        meters_per_unit=meters_per_unit(stage),
    )


def build_scene(simulation_app, *, usd_path: Path | None = None, stations=None) -> SceneBundle:
    """Open DataHall_6r and return per-station robots, controllers, and markers."""

    stage, world = open_datahall_stage(simulation_app, usd_path or cfg.DATAHALL_6R_USD)
    selected = tuple(stations) if stations is not None else cfg.STATIONS

    for spec in selected:
        robot_prim = stage.GetPrimAtPath(spec.robot_prim_path)
        if not robot_prim or not robot_prim.IsValid():
            raise RuntimeError(f"Missing robot prim {spec.robot_prim_path}")
        disable_robot_ros_graphs(stage, spec.robot_prim_path)
        enable_crystal_head_physics(stage, spec.path45, spec.path39)
        apply_grasp_friction_materials(stage, spec.robot_prim_path, spec.path45, spec.path39)

    world.reset()

    robots: dict[str, Any] = {}
    ee_paths: dict[str, str] = {}
    grasp_paths: dict[str, str] = {}
    for spec in selected:
        grasp_paths[spec.station_id] = resolve_grasp_part_path(stage, spec)
        ee_paths[spec.station_id] = resolve_end_effector_path(stage, spec.robot_prim_path)
        robots[spec.station_id] = _attach_manipulator(world, spec, ee_paths[spec.station_id])
        configure_robot_physics(stage, spec.robot_prim_path)
        apply_ur10e_home_pose(robots[spec.station_id], apply_live=False)

    world.reset()

    lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
        cfg.UR10E_LULA_NAME
    )
    if lula_config is None:
        raise RuntimeError(f"No Lula config for {cfg.UR10E_LULA_NAME!r}")

    bundles: list[StationBundle] = []
    for spec in selected:
        robot = robots[spec.station_id]
        apply_ur10e_home_pose(robot, apply_live=True)
        configure_robot_physics(stage, spec.robot_prim_path)
        apply_grasp_friction_materials(stage, spec.robot_prim_path, spec.path45, spec.path39)
        controller = _make_motion_controller(stage, robot, spec, dict(lula_config))
        try:
            _mn, floor_max, _c = prim_bbox(stage, spec.support_floor_path)
            block_top_z = float(floor_max[2])
        except Exception:
            _mn, _mx, part = prim_bbox(stage, grasp_paths[spec.station_id])
            block_top_z = float(part[2]) - 0.08
        _mn, _mx, part_center = prim_bbox(stage, grasp_paths[spec.station_id])
        observe_hand = _observe_hand_from_part(part_center)
        spawn_station_debug_markers(stage, spec, grasp_paths[spec.station_id])
        print(
            f"[SCENE] {spec.station_id} robot={spec.robot_prim_path} "
            f"ee={ee_paths[spec.station_id]} grasp={grasp_paths[spec.station_id]} "
            f"port={spec.port_contacts_path} block_top_z={block_top_z:.4f} "
            f"observe={np.round(observe_hand, 4)}"
        )
        bundles.append(
            StationBundle(
                spec=spec,
                robot=robot,
                motion_controller=controller,
                end_effector_path=ee_paths[spec.station_id],
                grasp_part_path=grasp_paths[spec.station_id],
                path45=spec.path45,
                path39=spec.path39,
                block_top_z=block_top_z,
                observe_hand=observe_hand,
                port_contacts_path=spec.port_contacts_path,
            )
        )
    return SceneBundle(world=world, stage=stage, stations=bundles)
