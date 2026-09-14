"""Load the six-station UR5e DataHall and attach physics and Lula control."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import omni.timeline
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from ur5e_6x_cable_insertions import config as cfg
from ur5e_6x_cable_insertions.controller import Ur5eSixArmMotionController
from ur5e_6x_cable_insertions.primitives import meters_per_unit, prim_bbox
from ur5e_6x_cable_insertions.runtime_support import angular_drive_value_for_stage


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
    jack_id: str
    port_pack_path: str


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


def _enable_gpu_dynamics(stage, simulation_manager) -> None:
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()]
    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(True).Set(True)
        api.CreateBroadphaseTypeAttr("GPU").Set("GPU")
        api.CreateSolverTypeAttr("TGS").Set("TGS")
    try:
        for scene in simulation_manager.get_physics_scenes():
            scene.set_enabled_gpu_dynamics(True)
    except Exception as exc:
        print(f"[SCENE] GPU dynamics manager warning: {exc}")


def open_datahall_stage(simulation_app, usd_path: Path):
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager

    path = Path(usd_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"UR5e DataHall USD not found: {path}")
    context = omni.usd.get_context()
    if context.open_stage(str(path)) is False:
        raise RuntimeError(f"Could not open stage: {path}")
    wait_for_stage_loading(simulation_app)
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"Stage is empty after opening {path}")
    scale = meters_per_unit(stage)
    world = World(stage_units_in_meters=scale)
    world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    _enable_gpu_dynamics(stage, SimulationManager)
    print(f"[SCENE] Opened {path} metersPerUnit={scale}")
    return stage, world


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
            attr.Set(False)


def enable_crystal_head_physics(stage, path45: str, path39: str) -> None:
    """Give each crystal head one rigid body and convex mesh collision."""

    from omni.physx.scripts import utils as physx_utils

    for head_path in (path45, path39):
        head = stage.GetPrimAtPath(head_path)
        if not head or not head.IsValid():
            print(f"[SCENE] Skip missing crystal head {head_path}")
            continue
        for prim in Usd.PrimRange(head):
            if prim != head:
                _strip_rigid_body_api(prim)
        rigid = UsdPhysics.RigidBodyAPI.Apply(head)
        rigid.CreateRigidBodyEnabledAttr(True).Set(True)
        UsdPhysics.MassAPI.Apply(head).CreateMassAttr(0.02).Set(0.02)
        enabled_meshes = 0
        for prim in Usd.PrimRange(head):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
            try:
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            except Exception:
                pass
            enabled_meshes += 1
        try:
            physx_utils.setCollider(head, approximationShape="convexHull")
        except Exception:
            pass
        print(f"[SCENE] Crystal-head physics {head_path}: meshes={enabled_meshes}")


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
    api = UsdPhysics.MaterialAPI.Apply(prim)
    api.CreateStaticFrictionAttr(float(static_friction)).Set(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction)).Set(float(dynamic_friction))
    api.CreateRestitutionAttr(0.0).Set(0.0)
    try:
        PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr().Set(
            str(combine_mode)
        )
    except Exception as exc:
        print(f"[SCENE] friction combine warning on {material_path}: {exc}")
    return material_path


def _bind_physics_material(prim, material_path: str) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material.Get(prim.GetStage(), material_path),
        UsdShade.Tokens.strongerThanDescendants,
        "physics",
    )


def _prim_matches_finger_token(prim) -> bool:
    name = prim.GetName().lower()
    path = str(prim.GetPath()).lower()
    return any(token.lower() in name or token.lower() in path for token in cfg.FINGERTIP_NAME_TOKENS)


def apply_grasp_friction_materials(
    stage, robot_prim_path: str, path45: str, path39: str
) -> None:
    """Bind the 0.8-friction grasp material to fingertips and both heads."""

    material_path = _ensure_physics_material(
        stage,
        "/World/PhysicsMaterials/fingertip_material",
        static_friction=float(cfg.GRASP_FRICTION_STATIC),
        dynamic_friction=float(cfg.GRASP_FRICTION_DYNAMIC),
        combine_mode=str(cfg.GRASP_FRICTION_COMBINE_MODE),
    )
    robot = stage.GetPrimAtPath(robot_prim_path)
    if robot and robot.IsValid():
        for prim in Usd.PrimRange(robot):
            if _prim_matches_finger_token(prim):
                _bind_physics_material(prim, material_path)
    for head_path in (path45, path39):
        head = stage.GetPrimAtPath(head_path)
        if not head or not head.IsValid():
            continue
        for prim in Usd.PrimRange(head):
            if prim == head or prim.IsA(UsdGeom.Mesh):
                _bind_physics_material(prim, material_path)


def apply_bezel_slide_friction_materials(stage, path39: str) -> None:
    """Bind low-friction slide material to rack bezels and the trailing head."""

    material_path = _ensure_physics_material(
        stage,
        "/World/PhysicsMaterials/bezel_slide_material",
        static_friction=float(cfg.BEZEL_FRICTION_STATIC),
        dynamic_friction=float(cfg.BEZEL_FRICTION_DYNAMIC),
        combine_mode=str(cfg.BEZEL_FRICTION_COMBINE_MODE),
    )
    datahall = stage.GetPrimAtPath("/World/DataHall")
    if datahall and datahall.IsValid():
        tokens = tuple(token.lower() for token in cfg.DGX_BEZEL_PATH_TOKENS)
        for prim in Usd.PrimRange(datahall):
            text = f"{prim.GetName()} {prim.GetPath()}".lower()
            if any(token in text for token in tokens) and (
                prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                _bind_physics_material(prim, material_path)
    if cfg.TRAILING_HEAD_SLIDE_FRICTION:
        head = stage.GetPrimAtPath(path39)
        if head and head.IsValid():
            for prim in Usd.PrimRange(head):
                if prim == head or prim.IsA(UsdGeom.Mesh):
                    _bind_physics_material(prim, material_path)


def configure_cable_deformable_for_stage(stage, spec: cfg.StationSpec) -> None:
    """Scale the one-arm deformable settings for the DataHall stage units."""

    scale = meters_per_unit(stage)
    line = stage.GetPrimAtPath(f"{spec.cable_root_path}/E_line_35")
    mesh = stage.GetPrimAtPath(f"{spec.cable_root_path}/E_line_35/simulation_mesh")
    if not line or not line.IsValid() or not mesh or not mesh.IsValid():
        raise RuntimeError(f"Missing deformable cable prims for {spec.station_id}")
    line.GetAttribute("physxDeformableBody:linearDamping").Set(5000.0 * scale)
    mesh.GetAttribute("physxCollision:contactOffset").Set(0.00011 / scale)
    mesh.GetAttribute("physxCollision:restOffset").Set(0.00010 / scale)


def rebind_cable_deformables(simulation_app) -> None:
    """Rebuild PhysX after cable collision and rigid-body authoring."""

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.pause()
    timeline.stop()
    for _ in range(20):
        simulation_app.update()
    timeline.play()
    for _ in range(30):
        simulation_app.update()


def disable_robot_ros_graphs(stage, robot_prim_path: str) -> None:
    graph = stage.GetPrimAtPath(f"{robot_prim_path}/ROS_ActionGraph")
    if graph and graph.IsValid():
        graph.SetActive(False)


def configure_robot_physics(stage, root_path: str) -> None:
    """Enable robot collision, disable gravity, and disable self-collision."""

    from omni.physx.scripts import utils as physx_utils

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(
                prim
            ).CreateEnabledSelfCollisionsAttr(False).Set(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True).Set(True)
        path = str(prim.GetPath()).lower()
        if not (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.GetName().lower() == "collisions"
            or "/collisions" in path
        ):
            continue
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
        try:
            physx_utils.setCollider(prim, UsdPhysics.Tokens.convexHull)
        except Exception:
            pass


def _apply_drive_parameters(stage, root_path: str, parameters: dict[str, tuple]) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    scale = meters_per_unit(stage)
    for prim in Usd.PrimRange(root):
        values = parameters.get(prim.GetName())
        if values is None or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        stiffness, damping, max_force = (
            angular_drive_value_for_stage(value, scale) for value in values
        )
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr(float(stiffness)).Set(float(stiffness))
        drive.CreateDampingAttr(float(damping)).Set(float(damping))
        drive.CreateMaxForceAttr(float(max_force)).Set(float(max_force))


def apply_ur5e_drive_parameters(stage, root_path: str) -> None:
    _apply_drive_parameters(stage, root_path, cfg.UR5E_ARM_DRIVE_PARAMETERS)


def apply_robotiq_drive_parameters(stage, root_path: str) -> None:
    _apply_drive_parameters(stage, root_path, cfg.ROBOTIQ_DRIVE_PARAMETERS)


def apply_ur5e_home_pose(robot, *, apply_live: bool = False) -> None:
    """Set the default and optionally live UR5e arm pose."""

    try:
        names = list(robot.dof_names)
    except Exception:
        names = []
    positions = np.zeros(len(names) if names else 6, dtype=np.float64)
    for index, name in enumerate(names or cfg.UR5E_ARM_JOINT_NAMES):
        if name in cfg.UR5E_ARM_JOINT_NAMES:
            source_index = cfg.UR5E_ARM_JOINT_NAMES.index(name)
            positions[index] = float(cfg.UR5E_HOME_ARM[source_index])
    robot.set_joints_default_state(positions=positions)
    if apply_live:
        robot.set_joint_positions(positions)


def apply_live_arm_damping(robot, station_id: str) -> None:
    try:
        controller = robot.get_articulation_controller()
        kps, kds = controller.get_gains()
        kps = np.asarray(kps, dtype=np.float64)
        kds = np.asarray(kds, dtype=np.float64)
        names = list(robot.dof_names)
        indices = [names.index(name) for name in cfg.UR5E_ARM_JOINT_NAMES if name in names]
        kds[indices] *= float(cfg.UR5E_LIVE_DAMPING_MULTIPLIER)
        controller.set_gains(kps=kps, kds=kds)
    except Exception as exc:
        raise RuntimeError(f"Could not apply live damping to {station_id}") from exc


def resolve_grasp_part_path(stage, spec: cfg.StationSpec) -> str:
    part = stage.GetPrimAtPath(spec.grasp_part_path)
    if part and part.IsValid():
        return spec.grasp_part_path
    found = find_descendant(stage, spec.path45, cfg.GRASP_PART_NAME)
    if found:
        return found
    raise RuntimeError(f"Missing grasp part under {spec.path45}")


def resolve_end_effector_path(stage, robot_prim_path: str) -> str:
    gripper = stage.GetPrimAtPath(f"{robot_prim_path}/ee_link/Robotiq_2F_85")
    if gripper and gripper.IsValid():
        base = find_descendant(stage, str(gripper.GetPath()), "base_link")
        return base or str(gripper.GetPath())
    ee_link = stage.GetPrimAtPath(f"{robot_prim_path}/ee_link")
    if ee_link and ee_link.IsValid():
        return str(ee_link.GetPath())
    raise RuntimeError(f"Missing Robotiq / ee_link under {robot_prim_path}")


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
    robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
    return robot


def _make_motion_controller(stage, robot, spec: cfg.StationSpec, lula_config: dict):
    kinematics = LulaKinematicsSolver(**lula_config)
    trajectory = LulaTaskSpaceTrajectoryGenerator(**lula_config)
    ee_frame = cfg.UR5E_EE_FRAME
    tool0 = stage.GetPrimAtPath(f"{spec.robot_prim_path}/tool0")
    if not tool0 or not tool0.IsValid():
        ee_frame = cfg.UR5E_EE_FRAME_FALLBACK
    articulation_kinematics = ArticulationKinematicsSolver(robot, kinematics, ee_frame)
    base_position, base_orientation = robot.get_world_pose()
    kinematics.set_robot_base_pose(base_position, base_orientation)
    return Ur5eSixArmMotionController(
        name=f"{spec.scene_name}_controller",
        robot_articulation=robot,
        task_traj_gen=trajectory,
        art_kinematics=articulation_kinematics,
        gripper=robot.gripper,
        tool_offset=float(cfg.TOOL_OFFSET_M),
        physics_dt=1.0 / 120.0,
        ee_frame=ee_frame,
        debug=True,
        meters_per_unit=meters_per_unit(stage),
    )


def build_scene(
    simulation_app, *, usd_path: Path | None = None, stations=None
) -> SceneBundle:
    """Open the UR5e DataHall and build selected station services."""

    stage, world = open_datahall_stage(
        simulation_app, usd_path or cfg.DATAHALL_6R_UR5E_USD
    )
    selected = tuple(stations) if stations is not None else cfg.STATIONS
    for spec in selected:
        robot_prim = stage.GetPrimAtPath(spec.robot_prim_path)
        if not robot_prim or not robot_prim.IsValid():
            raise RuntimeError(f"Missing robot prim {spec.robot_prim_path}")
        port_pack = stage.GetPrimAtPath(spec.port_pack_path)
        if not port_pack or not port_pack.IsValid():
            raise RuntimeError(
                f"Missing port pack {spec.port_pack_path}; verify PACK_COMPONENT"
            )
        disable_robot_ros_graphs(stage, spec.robot_prim_path)
        enable_crystal_head_physics(stage, spec.path45, spec.path39)
        apply_grasp_friction_materials(
            stage, spec.robot_prim_path, spec.path45, spec.path39
        )
        apply_bezel_slide_friction_materials(stage, spec.path39)
        configure_cable_deformable_for_stage(stage, spec)

    rebind_cable_deformables(simulation_app)
    world.reset()
    robots: dict[str, Any] = {}
    ee_paths: dict[str, str] = {}
    grasp_paths: dict[str, str] = {}
    for spec in selected:
        grasp_paths[spec.station_id] = resolve_grasp_part_path(stage, spec)
        ee_paths[spec.station_id] = resolve_end_effector_path(stage, spec.robot_prim_path)
        robot = _attach_manipulator(world, spec, ee_paths[spec.station_id])
        robots[spec.station_id] = robot
        configure_robot_physics(stage, spec.robot_prim_path)
        apply_ur5e_drive_parameters(stage, spec.robot_prim_path)
        apply_robotiq_drive_parameters(stage, spec.robot_prim_path)
        apply_ur5e_home_pose(robot, apply_live=False)

    world.reset()
    lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
        cfg.UR5E_LULA_NAME
    )
    if lula_config is None:
        raise RuntimeError(f"No Lula config for {cfg.UR5E_LULA_NAME!r}")

    bundles: list[StationBundle] = []
    for spec in selected:
        robot = robots[spec.station_id]
        apply_ur5e_home_pose(robot, apply_live=True)
        apply_live_arm_damping(robot, spec.station_id)
        configure_robot_physics(stage, spec.robot_prim_path)
        apply_grasp_friction_materials(
            stage, spec.robot_prim_path, spec.path45, spec.path39
        )
        apply_bezel_slide_friction_materials(stage, spec.path39)
        controller = _make_motion_controller(stage, robot, spec, dict(lula_config))
        try:
            _minimum, floor_maximum, _center = prim_bbox(stage, spec.support_floor_path)
            block_top_z = float(floor_maximum[2])
        except Exception:
            _minimum, _maximum, part_center = prim_bbox(
                stage, grasp_paths[spec.station_id]
            )
            block_top_z = float(part_center[2]) - 0.08
        _minimum, _maximum, head39_center = prim_bbox(stage, spec.path39)
        observe_hand = cfg.observation_hand_from_head39(head39_center, block_top_z)
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
                jack_id=spec.jack_id,
                port_pack_path=spec.port_pack_path,
            )
        )
    return SceneBundle(world=world, stage=stage, stations=bundles)
