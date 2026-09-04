"""Scene builder: asset_spawn layout + cable-head physics + Lula motion."""

from __future__ import annotations

from dataclasses import dataclass
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
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

from asset_spawn import spawn as asset_spawn
from asset_spawn.spawn import AssetSpawnBundle, build_asset_spawn_scene
from franka_motion_controller import FrankaMotionController
from ur10e_1x_cable_insertion import config as cfg


@dataclass
class SceneBundle:
    world: Any
    stage: Any
    robot: Any
    motion_controller: Any
    end_effector_path: str
    grasp_part_path: str
    path45: str
    path39: str
    block_top_z: float


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
    """Option A: one rigid body per crystal head; meshes stay attached children.

    ``E_part006_44`` is a child of head45. Giving it its own RigidBodyAPI makes
    PhysX treat it as a separate dynamic body, so only the head roots get
    RigidBodyAPI; mesh descendants get convexHull collision only.
    """

    from omni.physx.scripts import utils as physx_utils

    for head_path in (path45, path39):
        head = stage.GetPrimAtPath(head_path)
        if not head or not head.IsValid():
            print(f"[SCENE] Skip physics: missing {head_path}")
            continue

        # Strip accidental RigidBody on descendants (e.g. E_part006_44).
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


def resolve_grasp_part_path(stage, path45: str) -> str:
    part = stage.GetPrimAtPath(cfg.GRASP_PART_PATH)
    if part and part.IsValid():
        return cfg.GRASP_PART_PATH
    # Fallback: search under head45.
    head = stage.GetPrimAtPath(path45)
    if head and head.IsValid():
        for prim in Usd.PrimRange(head):
            if prim.GetName() == "E_part006_44":
                return str(prim.GetPath())
    raise RuntimeError(f"Missing grasp part E_part006_44 under {path45}")


def _ensure_physics_material(
    stage,
    material_path: str,
    *,
    static_friction: float,
    dynamic_friction: float,
    combine_mode: str,
) -> str:
    """Create (or update) a rigid-body physics material. Returns material path."""

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
    """Raise fingertip/head friction so the pinch holds through carry (Isaac tutorial recipe).

    From NVIDIA's closed-loop gripper tutorial: rubber fingertip material with
    static/dynamic friction 0.8 and friction combine mode ``max``, bound on the
    inner-finger pads. Also bind the same recipe on crystal-head meshes so the
    contact pair resolves to the high friction value.
    """

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
            # Bind on collision-capable prims (mesh or link xform with collider kids).
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
        f"[SCENE] Grasp friction material {mat_path} "
        f"(mu_s={cfg.GRASP_FRICTION_STATIC}, mu_d={cfg.GRASP_FRICTION_DYNAMIC}, "
        f"combine={cfg.GRASP_FRICTION_COMBINE_MODE}) "
        f"bound on {finger_hits} fingertip prim(s), {head_hits} head prim(s)"
    )


def apply_ur10e_home_pose(robot, *, apply_live: bool = False) -> None:
    """Set default (and optionally live) arm joint positions to a known home."""

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
    print(f"[SCENE] UR10e home pose applied (arm rad={np.round(cfg.UR10E_HOME_ARM, 3)})")


def build_scene(simulation_app) -> SceneBundle:
    """Build asset_spawn layout, enable head physics, attach Lula motion."""

    spawn_bundle: AssetSpawnBundle = build_asset_spawn_scene(simulation_app)
    world = spawn_bundle.world
    stage = spawn_bundle.stage
    path45 = spawn_bundle.path45
    path39 = spawn_bundle.path39

    enable_crystal_head_physics(stage, path45, path39)
    apply_grasp_friction_materials(stage, cfg.UR10E_PRIM_PATH, path45, path39)
    grasp_part_path = resolve_grasp_part_path(stage, path45)

    # Rebuild physics views after adding RigidBody / collision on the cable.
    # Timeline stop/play in rebind destroys the World physics view — rebuild it.
    world.reset()
    apply_ur10e_home_pose(spawn_bundle.robot, apply_live=False)

    # Re-attach ParallelGripper on the existing manipulator for open/close commands.
    ee_path = spawn_bundle.end_effector_path
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path,
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0.0]),
        joint_closed_positions=np.array([cfg.ROBOTIQ_CLOSED_RAD]),
        # Absolute open/close only. Delta mode re-adds the close angle on every
        # "close" pulse and overshoots the joint limit (fingers chatter / open).
        action_deltas=None,
        use_mimic_joints=True,
    )

    # Replace the spawn manipulator with one that owns the gripper handle.
    try:
        world.scene.remove_object("ur10e")
    except Exception:
        pass
    robot = world.scene.add(
        SingleManipulator(
            prim_path=cfg.UR10E_PRIM_PATH,
            name="ur10e",
            end_effector_prim_path=ee_path,
            gripper=gripper,
        )
    )
    try:
        robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
    except Exception as exc:
        print(f"[SCENE] gripper default state warning: {exc}")

    world.reset()
    apply_ur10e_home_pose(robot, apply_live=True)
    asset_spawn.finalize_ur10e_placement(robot, asset_spawn.UR10E_MOUNT_PATH, spawn_bundle.base_link_path)
    asset_spawn.configure_robot_physics(cfg.UR10E_PRIM_PATH)
    # Re-bind friction after configure_robot_physics rebuilds collision authors.
    apply_grasp_friction_materials(stage, cfg.UR10E_PRIM_PATH, path45, path39)

    lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
        cfg.UR10E_LULA_NAME
    )
    if lula_config is None:
        raise RuntimeError(f"No Lula config for {cfg.UR10E_LULA_NAME!r}")
    kinematics = LulaKinematicsSolver(**lula_config)
    trajectory_generator = LulaTaskSpaceTrajectoryGenerator(**lula_config)

    ee_frame = cfg.UR10E_EE_FRAME
    tool0 = stage.GetPrimAtPath(f"{cfg.UR10E_PRIM_PATH}/tool0")
    if not (tool0 and tool0.IsValid()):
        ee_frame = "wrist_3_link"
        print(f"[SCENE] tool0 missing; Lula ee_frame -> {ee_frame}")

    articulation_kinematics = ArticulationKinematicsSolver(robot, kinematics, ee_frame)
    base_position, base_orientation = robot.get_world_pose()
    kinematics.set_robot_base_pose(base_position, base_orientation)

    controller = FrankaMotionController(
        name="ur10e_cable_insertion_controller",
        robot_articulation=robot,
        task_traj_gen=trajectory_generator,
        art_kinematics=articulation_kinematics,
        gripper=robot.gripper,
        tool_offset=0.0,
        physics_dt=1.0 / 120.0,
        ee_frame=ee_frame,
        debug=True,
    )
    print(
        f"[SCENE] Motion ready ee_frame={ee_frame} grasp_part={grasp_part_path} "
        f"block_top_z={spawn_bundle.block_top_z:.4f}"
    )
    return SceneBundle(
        world=world,
        stage=stage,
        robot=robot,
        motion_controller=controller,
        end_effector_path=ee_path,
        grasp_part_path=grasp_part_path,
        path45=path45,
        path39=path39,
        block_top_z=spawn_bundle.block_top_z,
    )
