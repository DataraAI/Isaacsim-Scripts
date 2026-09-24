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
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from ur5e_6x_cable_insertions import config as cfg
from ur5e_6x_cable_insertions.controller import Ur5eSixArmMotionController
from ur5e_6x_cable_insertions.primitives import meters_per_unit
from ur5e_6x_cable_insertions.runtime_support import angular_drive_value_for_stage


@dataclass
class StationBundle:
    spec: cfg.StationSpec
    robot: Any
    motion_controller: Any
    end_effector_path: str
    cable_root_path: str
    home_arm: np.ndarray | None = None


@dataclass
class IdleStation:
    """Unselected arm: no BT — held at work-table home every sim step."""

    spec: cfg.StationSpec
    robot: Any
    home_arm: np.ndarray


@dataclass
class SceneBundle:
    world: Any
    stage: Any
    stations: list[StationBundle] = field(default_factory=list)
    idle_stations: list[IdleStation] = field(default_factory=list)


def hold_idle_ur5e_homes(idle_stations: list[IdleStation]) -> None:
    """Unselected arms are deactivated — nothing to hold each frame."""

    return



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


def _enable_gpu_dynamics(stage, simulation_manager, world=None) -> None:
    """Apply PhysX GPU/CPU mode from ``cfg.SCENE_ENABLE_GPU_DYNAMICS``.

    When False, force CPU dynamics + MBP broadphase on the USD PhysxScene,
    SimulationManager, and (if provided) World PhysicsContext — otherwise a
    leftover GPU broadphase opinion restarts the CUDA 700 abort loop.
    """

    use_gpu = bool(cfg.SCENE_ENABLE_GPU_DYNAMICS)
    broadphase = "GPU" if use_gpu else "MBP"
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()]
    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(use_gpu).Set(use_gpu)
        api.CreateBroadphaseTypeAttr(broadphase).Set(broadphase)
        api.CreateSolverTypeAttr(str(cfg.SCENE_SOLVER_TYPE)).Set(str(cfg.SCENE_SOLVER_TYPE))
        try:
            api.CreateEnableCCDAttr(bool(cfg.SCENE_ENABLE_CCD)).Set(
                bool(cfg.SCENE_ENABLE_CCD)
            )
        except Exception:
            pass
        try:
            api.CreateMinPositionIterationCountAttr(
                int(cfg.SCENE_MIN_POSITION_ITERS)
            ).Set(int(cfg.SCENE_MIN_POSITION_ITERS))
            api.CreateMinVelocityIterationCountAttr(
                int(cfg.SCENE_MIN_VELOCITY_ITERS)
            ).Set(int(cfg.SCENE_MIN_VELOCITY_ITERS))
        except Exception:
            pass
        try:
            api.CreateBounceThresholdAttr(float(cfg.SCENE_BOUNCE_THRESHOLD)).Set(
                float(cfg.SCENE_BOUNCE_THRESHOLD)
            )
        except Exception:
            pass
        if use_gpu:
            try:
                total_cap = int(
                    getattr(cfg, "SCENE_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY", 4096)
                )
                found_agg = int(
                    getattr(cfg, "SCENE_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY", 4096)
                )
                found_pairs = int(
                    getattr(cfg, "SCENE_GPU_FOUND_LOST_PAIRS_CAPACITY", 262144)
                )
                api.CreateGpuTotalAggregatePairsCapacityAttr(total_cap).Set(total_cap)
                api.CreateGpuFoundLostAggregatePairsCapacityAttr(found_agg).Set(found_agg)
                api.CreateGpuFoundLostPairsCapacityAttr(found_pairs).Set(found_pairs)
                print(
                    f"[SCENE] GPU PhysX on: broadphase=GPU "
                    f"total={total_cap} foundLostAgg={found_agg} foundLost={found_pairs}"
                )
            except Exception as exc:
                print(f"[SCENE] GPU aggregate capacity warning: {exc}")
        else:
            print("[SCENE] CPU PhysX (SCENE_ENABLE_GPU_DYNAMICS=False, broadphase=MBP)")
    try:
        for scene in simulation_manager.get_physics_scenes():
            scene.set_enabled_gpu_dynamics(use_gpu)
    except Exception as exc:
        print(f"[SCENE] GPU dynamics manager warning: {exc}")
    if world is not None:
        try:
            pc = world.get_physics_context()
            if hasattr(pc, "enable_gpu_dynamics"):
                pc.enable_gpu_dynamics(use_gpu)
            if hasattr(pc, "set_broadphase_type"):
                pc.set_broadphase_type(broadphase)
            print(
                f"[SCENE] PhysicsContext gpu_dynamics={use_gpu} broadphase={broadphase}"
            )
        except Exception as exc:
            print(f"[SCENE] PhysicsContext GPU/CPU warning: {exc}")


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
    world.set_simulation_dt(
        physics_dt=float(cfg.PHYSICS_DT), rendering_dt=float(cfg.RENDERING_DT)
    )
    _enable_gpu_dynamics(stage, SimulationManager, world=world)
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
    """Give each crystal head one rigid body and convex mesh collision.

    Do **not** call ``setCollider`` on the head Xform: that applies
    ``CollisionAPI`` to a non-mesh rigid body and PhysX then tries a triangle
    mesh / MeshSimplification path (``Parse collision ... falling back to
    convexHull`` spam on ``E_crystal_head*``). Only Mesh children get colliders.
    """

    for head_path in (path45, path39):
        head = stage.GetPrimAtPath(head_path)
        if not head or not head.IsValid():
            print(f"[SCENE] Skip missing crystal head {head_path}")
            continue
        for prim in Usd.PrimRange(head):
            if prim != head:
                _strip_rigid_body_api(prim)
        # Head Xform must not carry its own CollisionAPI (triangle-mesh parse).
        if head.HasAPI(UsdPhysics.CollisionAPI):
            try:
                head.RemoveAPI(UsdPhysics.CollisionAPI)
            except Exception:
                pass
        if head.HasAPI(UsdPhysics.MeshCollisionAPI):
            try:
                head.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            except Exception:
                pass
        rigid = UsdPhysics.RigidBodyAPI.Apply(head)
        rigid.CreateRigidBodyEnabledAttr(True).Set(True)
        # MassAPI is always kg (same as ur10e_1x); do not scale by metersPerUnit.
        mass_kg = float(cfg.CRYSTAL_HEAD_MASS_KG)
        UsdPhysics.MassAPI.Apply(head).CreateMassAttr(mass_kg).Set(mass_kg)
        try:
            contact_report = PhysxSchema.PhysxContactReportAPI.Apply(head)
            contact_report.CreateThresholdAttr().Set(0.0)
        except Exception as exc:
            print(f"[SCENE] ContactReportAPI failed on {head_path}: {exc}")
        enabled_meshes = 0
        for prim in Usd.PrimRange(head):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            try:
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            except Exception:
                pass
            enabled_meshes += 1
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
    """Bind grasp friction material to fingertips and both heads.

    Friction coefficients are dimensionless (same numbers as meter-stage ur10e).
    """

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
    """Bind low-friction slide material to bezels, switch faces, trailing head."""

    material_path = _ensure_physics_material(
        stage,
        "/World/PhysicsMaterials/bezel_slide_material",
        static_friction=float(cfg.BEZEL_FRICTION_STATIC),
        dynamic_friction=float(cfg.BEZEL_FRICTION_DYNAMIC),
        combine_mode=str(cfg.BEZEL_FRICTION_COMBINE_MODE),
    )
    tokens = tuple(
        token.lower()
        for token in (
            *getattr(cfg, "INSERT_SLIDE_PATH_TOKENS", ()),
            *getattr(cfg, "DGX_BEZEL_PATH_TOKENS", ()),
        )
    )
    roots: list = []
    for root_path in (
        cfg.DATAHALL_PRIM_PATH,
        getattr(cfg, "NETWORK_SWITCHES_SCOPE", "/World/Network_Switches"),
    ):
        prim = stage.GetPrimAtPath(root_path)
        if prim and prim.IsValid():
            roots.append(prim)
    hits = 0
    for root in roots:
        for prim in Usd.PrimRange(root):
            text = f"{prim.GetName()} {prim.GetPath()}".lower()
            if any(token in text for token in tokens) and (
                prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                _bind_physics_material(prim, material_path)
                hits += 1
    trail_hits = 0
    if cfg.TRAILING_HEAD_SLIDE_FRICTION:
        head = stage.GetPrimAtPath(path39)
        if head and head.IsValid():
            for prim in Usd.PrimRange(head):
                if prim == head or prim.IsA(UsdGeom.Mesh):
                    _bind_physics_material(prim, material_path)
                    trail_hits += 1
    print(
        f"[SCENE] Insert/bezel slide friction "
        f"(μ_s={cfg.BEZEL_FRICTION_STATIC}, μ_d={cfg.BEZEL_FRICTION_DYNAMIC}, "
        f"combine={cfg.BEZEL_FRICTION_COMBINE_MODE}) "
        f"bound on {hits} obstacle prim(s), {trail_hits} trailing-head prim(s)"
    )


def apply_worktable_frictionless_materials(stage) -> None:
    """Bind μ≈0 physics material to work-table meshes (cable can slide freely).

    Insert contact logs showed ``Cable_*/E_crystal_head2_39/...`` hitting
    ``/World/WorkTable1``. combine=min so table μ=0 wins against high-μ heads.
    """

    if not bool(getattr(cfg, "WORKTABLE_FRICTIONLESS", True)):
        return
    material_path = _ensure_physics_material(
        stage,
        "/World/PhysicsMaterials/worktable_frictionless_material",
        static_friction=float(getattr(cfg, "WORKTABLE_FRICTION_STATIC", 0.0)),
        dynamic_friction=float(getattr(cfg, "WORKTABLE_FRICTION_DYNAMIC", 0.0)),
        combine_mode=str(getattr(cfg, "WORKTABLE_FRICTION_COMBINE_MODE", "min")),
    )
    roots: list[str] = []
    by_height = getattr(cfg, "WORK_TABLE_PATH_BY_HEIGHT", {}) or {}
    roots.extend(str(p) for p in by_height.values() if p)
    scope = str(getattr(cfg, "WORK_TABLES_SCOPE", "/World/WorkTables") or "")
    if scope:
        roots.append(scope)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq_roots = []
    for path in roots:
        if path not in seen:
            seen.add(path)
            uniq_roots.append(path)

    bound = 0
    for root_path in uniq_roots:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
                _bind_physics_material(prim, material_path)
                bound += 1
        # Also bind the root xform itself in case collisions live there.
        if root.HasAPI(UsdPhysics.CollisionAPI):
            _bind_physics_material(root, material_path)
            bound += 1
    print(
        f"[SCENE] WorkTable frictionless "
        f"(μ_s={getattr(cfg, 'WORKTABLE_FRICTION_STATIC', 0.0)}, "
        f"μ_d={getattr(cfg, 'WORKTABLE_FRICTION_DYNAMIC', 0.0)}, "
        f"combine={getattr(cfg, 'WORKTABLE_FRICTION_COMBINE_MODE', 'min')}) "
        f"bound on {bound} prim(s) under {uniq_roots}"
    )


def _resolve_instance_edit_root(stage, root_path: str) -> str:
    """If ``root_path`` is an instance proxy, lift to its owning instance root.

    Mesh133/Mesh134 live under ``…/Lower_Left/AS4610_inst/…``. ``AS4610_inst``
    itself is still a proxy; only ``Lower_Left`` (or ``Upper_Right``) is the
    editable instance. De-instancing a proxy path is a no-op.
    """

    prim = stage.GetPrimAtPath(root_path)
    if not prim or not prim.IsValid():
        return root_path
    if not prim.IsInstanceProxy():
        return root_path
    cur = prim.GetParent()
    while cur and cur.IsValid():
        path = str(cur.GetPath())
        if path in ("/", "/World", str(cfg.NETWORK_SWITCHES_SCOPE)):
            break
        if not cur.IsInstanceProxy() and (
            (hasattr(cur, "IsInstance") and cur.IsInstance()) or cur.IsInstanceable()
        ):
            print(f"[SCENE] De-instance root lifted {root_path} → {path}")
            return path
        cur = cur.GetParent()
    return root_path


def deinstance_prim_tree(
    stage, root_path: str, *, max_passes: int | None = None
) -> int:
    """Clear USD instanceable flags so mesh CollisionAPI can be authored."""

    root_path = _resolve_instance_edit_root(stage, root_path)
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[SCENE] De-instance skipped: missing {root_path}")
        return 0
    passes = (
        int(cfg.DATAHALL_DEINSTANCE_MAX_PASSES)
        if max_passes is None
        else max(1, int(max_passes))
    )
    total_changed = 0
    for _pass in range(passes):
        changed = 0
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            break
        for prim in Usd.PrimRange(root):
            try:
                if prim.IsInstanceProxy():
                    continue
                is_instance = bool(prim.IsInstance()) if hasattr(prim, "IsInstance") else False
                is_instanceable = (
                    bool(prim.IsInstanceable())
                    if hasattr(prim, "IsInstanceable")
                    else False
                )
                if is_instance or is_instanceable:
                    prim.SetInstanceable(False)
                    changed += 1
            except Exception as exc:
                print(f"[SCENE] De-instance skip {prim.GetPath()}: {exc}")
        total_changed += changed
        if changed == 0:
            break
    if total_changed:
        print(f"[SCENE] De-instanced {total_changed} prim(s) under {root_path}")
    return total_changed


def collision_roots_for_stations(selected) -> tuple[str, ...]:
    """Static collider roots: tables + cable blocks + selected AS4610 Switch trees.

    Avoids enabling CollisionAPI on the entire ``/World/Network_Switches`` tree
    (~6k triangle meshes), which overflows PhysX GPU pair buffers (CUDA 700).

    Roots include the robot-loc pack (``Lower_Left`` / ``Upper_Right``), which is
    the USD instance root — ``AS4610_inst`` alone is still an instance proxy and
    cannot receive CollisionAPI until that parent is de-instanced.
    """

    roots: list[str] = [
        *tuple(cfg.WORK_TABLE_PATH_BY_HEIGHT.values()),
        str(cfg.CABLE_BLOCKS_SCOPE),
    ]
    if not bool(getattr(cfg, "DATAHALL_COLLISION_SCOPE_TO_SELECTED", True)):
        return tuple(dict.fromkeys((*roots, *tuple(cfg.DATAHALL_COLLISION_ROOTS))))

    for spec in selected or ():
        pack = str(getattr(spec, "port_pack_path", "") or "")
        if not pack:
            continue
        # robot_loc instance root (…/Lower_Left), then AS4610_inst / Switch / pack.
        robot_loc = pack
        if "/AS4610_inst" in robot_loc:
            robot_loc = robot_loc.split("/AS4610_inst", 1)[0]
        inst = pack
        if "/AS4610_inst/" in inst:
            inst = inst.split("/AS4610_inst/", 1)[0] + "/AS4610_inst"
        switch = pack
        for marker in ("/Net_12_Pack", "/RJ45_Group"):
            if marker in switch:
                switch = switch.split(marker, 1)[0]
                break
        roots.append(robot_loc)
        roots.append(inst)
        roots.append(switch)
        if pack not in roots:
            roots.append(pack)
    if len(roots) <= 4:
        # Fallback: no station paths resolved — use authored roots.
        print(
            "[SCENE] WARN: no selected switch packs for collision scope; "
            "falling back to DATAHALL_COLLISION_ROOTS (may stress GPU PhysX)"
        )
        return tuple(dict.fromkeys((*roots, *tuple(cfg.DATAHALL_COLLISION_ROOTS))))
    return tuple(dict.fromkeys(roots))


def enable_datahall_static_collisions(
    stage,
    root_paths: tuple[str, ...] | None = None,
    *,
    approximation_shape: str | None = None,
    selected=None,
) -> int:
    """Author static mesh colliders so the arm cannot pass through the hall/rack.

    Mirrors ``asset_spawn.enable_datahall_static_collisions``: mesh approximation
    on visible solid geometry; skip invisible / no_collision / instance-proxy
    prims so switch ports are not sealed by parent hulls.
    Optionally de-instances each root first so former proxies become editable.

    When ``selected`` stations are passed (or ``DATAHALL_COLLISION_SCOPE_TO_SELECTED``),
    only those AS4610 Switch subtrees are colliders — not every Network_Switches mesh.
    """

    from omni.physx.scripts import utils as physx_utils

    if root_paths is None:
        if selected is not None or bool(
            getattr(cfg, "DATAHALL_COLLISION_SCOPE_TO_SELECTED", True)
        ):
            roots = collision_roots_for_stations(selected)
        else:
            roots = tuple(cfg.DATAHALL_COLLISION_ROOTS)
    else:
        roots = tuple(root_paths)
    # Prefer live DataHall_01 if the configured root is missing.
    resolved: list[str] = []
    for root_path in roots:
        if stage.GetPrimAtPath(root_path) and stage.GetPrimAtPath(root_path).IsValid():
            resolved.append(root_path)
            continue
        if root_path in (
            str(cfg.DATAHALL_PRIM_PATH),
            *getattr(cfg, "DATAHALL_PRIM_PATH_FALLBACKS", ()),
        ):
            for alt in getattr(cfg, "DATAHALL_PRIM_PATH_FALLBACKS", ()):
                if stage.GetPrimAtPath(alt) and stage.GetPrimAtPath(alt).IsValid():
                    resolved.append(str(alt))
                    break
        else:
            print(f"[SCENE] DataHall collision skipped: missing {root_path}")
    roots = tuple(dict.fromkeys(resolved))
    print(f"[SCENE] Static collision roots ({len(roots)}): {list(roots)}")
    approx = (
        str(cfg.DATAHALL_COLLISION_APPROXIMATION)
        if approximation_shape is None
        else str(approximation_shape)
    )
    if bool(cfg.DATAHALL_DEINSTANCE_BEFORE_COLLISION):
        for root_path in roots:
            deinstance_prim_tree(stage, root_path)

    skip_tokens = tuple(
        str(t).lower()
        for t in getattr(cfg, "DATAHALL_COLLISION_SKIP_PATH_TOKENS", ())
        if t
    )
    total = 0
    for root_path in roots:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            print(f"[SCENE] DataHall collision skipped: missing {root_path}")
            continue
        count = 0
        invisible_count = 0
        proxy_skipped = 0
        skipped_token = 0
        for prim in Usd.PrimRange(root):
            if prim.GetMetadata("hide_in_stage_window"):
                continue
            path_l = str(prim.GetPath()).lower()
            if skip_tokens and any(tok in path_l for tok in skip_tokens):
                skipped_token += 1
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                invisible_count += 1
                continue
            no_collision = prim.GetAttribute("omni:no_collision")
            if no_collision and bool(no_collision.Get()):
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
            if not is_solid:
                continue
            if is_mesh:
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if points is None or len(points) == 0:
                    continue
            try:
                if prim.IsInstanceProxy():
                    proxy_skipped += 1
                    continue
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(
                    True
                )
                # Static collider: no free rigid body — stays fixed in world.
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(
                            False
                        ).Set(False)
                    except Exception:
                        pass
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
                if is_mesh:
                    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    mesh_api.CreateApproximationAttr().Set(approx)
                    approx_api = physx_utils.MESH_APPROXIMATIONS.get(approx)
                    if approx_api is not None:
                        approx_api.Apply(prim)
                count += 1
            except Exception as exc:
                print(f"[SCENE] DataHall collider skip {prim.GetPath()}: {exc}")
        total += count
        print(
            f"[SCENE] DataHall static collisions enabled on {count} prim(s) "
            f"under {root_path} (skipped_invisible={invisible_count}, "
            f"skipped_proxy={proxy_skipped}, skipped_token={skipped_token}, "
            f"approx={approx!r})"
        )
    # Mesh4679 is a rack lip under DataHall_01 — only force it when the
    # facility root itself is in the collision set (switches-only mode skips it).
    hall_roots = {
        str(cfg.DATAHALL_PRIM_PATH),
        *tuple(str(p) for p in getattr(cfg, "DATAHALL_PRIM_PATH_FALLBACKS", ())),
    }
    if any(str(r) in hall_roots for r in roots):
        forced = _force_named_static_colliders(
            stage, name_tokens=("Mesh4679", "mesh4679"), approximation=approx
        )
        if forced:
            print(
                f"[SCENE] Forced static colliders on {forced} named Mesh4679 prim(s)"
            )
            total += forced
    else:
        print(
            "[SCENE] Skipping Mesh4679 force-colliders "
            "(DataHall facility not in DATAHALL_COLLISION_ROOTS)"
        )
    # Ethernet jack shells (Mesh133 / Mesh134) under selected switch roots.
    port_names = tuple(
        str(n)
        for n in getattr(cfg, "DATAHALL_FORCE_ETHERNET_PORT_MESHES", ())
        if n
    )
    if port_names:
        override = getattr(cfg, "DATAHALL_FORCE_ETHERNET_PORT_APPROXIMATION", None)
        port_approx = str(override) if override else approx
        switch_roots = tuple(
            r
            for r in roots
            if "Network_Switches" in str(r)
            or "AS4610" in str(r)
            or "/Switch" in str(r)
            or "RJ45" in str(r)
            or "Lower_Left" in str(r)
            or "Upper_Right" in str(r)
            or "Lower_Right" in str(r)
            or "Upper_Left" in str(r)
        )
        if not switch_roots:
            print(
                "[SCENE] WARN: no Network_Switches roots for Mesh133/Mesh134 "
                f"(collision roots={list(roots)})"
            )
        else:
            forced_ports = _force_named_static_colliders(
                stage,
                name_tokens=port_names,
                approximation=port_approx,
                root_paths=switch_roots,
                exact_name=True,
            )
            if forced_ports:
                print(
                    f"[SCENE] Ethernet port colliders ON: {forced_ports} "
                    f"prim(s) {list(port_names)} approx={port_approx!r}"
                )
                total += forced_ports
            else:
                print(
                    f"[SCENE] WARN: Mesh133/Mesh134 still have no colliders "
                    f"(still proxies? roots={list(switch_roots)})"
                )
    if bool(getattr(cfg, "DATAHALL_DISABLE_FRONT_DOOR_COLLISION", True)):
        disable_datahall_front_door_collisions(stage)
    return total


def disable_datahall_front_door_collisions(stage) -> int:
    """Keep rack ``Front_Door`` prims, but force collision off on the subtree.

    In ``DataHall_6r_ur5e.usd`` the door is authored ``active=False`` (children
    like ``MetalMeshPanel`` do not compose). We still:

    1. Find every ``Front_Door`` xform (including inactive).
    2. Temporarily activate so payloads/children compose.
    3. Set ``physics:collisionEnabled=False`` (+ ``omni:no_collision``) on the
       door and all mesh/collision descendants.
    4. Restore the prior active state (do **not** delete the door).

    This matches the historical jam at
    ``…/Rack_42U_01/…/Front_Door/MetalMeshPanel`` without removing the asset.
    """

    doors: list = []
    for prim in stage.TraverseAll():
        if prim.GetName() == "Front_Door":
            doors.append(prim)
    if not doors:
        print("[SCENE] Front_Door: none found under stage (nothing to disable)")
        return 0

    disabled_total = 0
    for door in doors:
        path = str(door.GetPath())
        was_active = bool(door.IsActive())
        activated = False
        if not was_active:
            try:
                door.SetActive(True)
                activated = True
            except Exception as exc:
                print(f"[SCENE] Front_Door activate failed {path}: {exc}")
        # Re-resolve after activation so payload children appear.
        door = stage.GetPrimAtPath(path)
        if not door or not door.IsValid():
            continue
        disabled = 0
        for prim in Usd.PrimRange(door):
            try:
                prim.CreateAttribute(
                    "omni:no_collision", Sdf.ValueTypeNames.Bool, custom=True
                ).Set(True)
            except Exception:
                pass
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
                try:
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(
                        False
                    ).Set(False)
                    disabled += 1
                except Exception as exc:
                    print(f"[SCENE] Front_Door collider skip {prim.GetPath()}: {exc}")
        disabled_total += disabled
        if activated:
            try:
                stage.GetPrimAtPath(path).SetActive(False)
            except Exception as exc:
                print(f"[SCENE] Front_Door re-deactivate failed {path}: {exc}")
        print(
            f"[SCENE] Front_Door collision OFF at {path} "
            f"(prims_disabled={disabled}, restored_active={was_active})"
        )
    return disabled_total


def _force_named_static_colliders(
    stage,
    *,
    name_tokens: tuple[str, ...],
    approximation: str,
    root_paths: tuple[str, ...] | None = None,
    exact_name: bool = False,
) -> int:
    """Enable static mesh collision on prims matching ``name_tokens``.

    When ``root_paths`` is set, only those subtrees are scanned. ``exact_name``
    matches prim name equality (case-insensitive); otherwise substring.
    """

    from omni.physx.scripts import utils as physx_utils

    tokens = tuple(t.lower() for t in name_tokens)
    count = 0

    def _iter_prims():
        if not root_paths:
            yield from stage.Traverse()
            return
        for root_path in root_paths:
            root = stage.GetPrimAtPath(root_path)
            if not root or not root.IsValid():
                continue
            yield from Usd.PrimRange(root)

    for prim in _iter_prims():
        name = prim.GetName().lower()
        if exact_name:
            if name not in tokens:
                continue
        elif not any(token in name for token in tokens):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        try:
            if prim.IsInstanceProxy():
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if points is None or len(points) == 0:
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(
                True
            )
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                try:
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False).Set(
                        False
                    )
                except Exception:
                    pass
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set(approximation)
            approx_api = physx_utils.MESH_APPROXIMATIONS.get(approximation)
            if approx_api is not None:
                approx_api.Apply(prim)
            count += 1
            print(f"[SCENE] Forced collider ON ({approximation}): {prim.GetPath()}")
        except Exception as exc:
            print(f"[SCENE] Forced collider skip {prim.GetPath()}: {exc}")
    return count



def configure_cable_deformable_for_stage(stage, spec: cfg.StationSpec) -> None:
    """Apply ur10e_1x-equivalent soft-cable params on the cm DataHall stage.

    - ``linearDamping``: 1/time — use the same magnitude as the meter-stage
      recipe (do **not** multiply by metersPerUnit; that made the line ~100×
      floppier and yanked the head out of the fingers on lift).
    - contact/rest offsets: authored in meters → divide by mpu for stage units.
    """

    mpu = meters_per_unit(stage)
    line = stage.GetPrimAtPath(f"{spec.cable_root_path}/E_line_35")
    mesh = stage.GetPrimAtPath(f"{spec.cable_root_path}/E_line_35/simulation_mesh")
    if not line or not line.IsValid() or not mesh or not mesh.IsValid():
        raise RuntimeError(f"Missing deformable cable prims for {spec.station_id}")
    damping = float(cfg.DEFORMABLE_LINEAR_DAMPING)
    contact = float(cfg.DEFORMABLE_CONTACT_OFFSET_M) / mpu
    rest = float(cfg.DEFORMABLE_REST_OFFSET_M) / mpu
    line.GetAttribute("physxDeformableBody:linearDamping").Set(damping)
    mesh.GetAttribute("physxCollision:contactOffset").Set(contact)
    mesh.GetAttribute("physxCollision:restOffset").Set(rest)
    print(
        f"[SCENE] {spec.station_id} cable deformable (ur10e-equivalent): "
        f"damping={damping:g} contact_stage={contact:g} rest_stage={rest:g} "
        f"(mpu={mpu:g})"
    )


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


def _mute_omni_graph(graph_path: str, prim=None) -> None:
    """Best-effort stop OmniGraph evaluation for ``graph_path``."""

    try:
        import omni.graph.core as og
    except Exception:
        og = None
    if og is not None:
        for fn_name in ("set_graph_enabled", "set_graph_evaluation_mode"):
            fn = getattr(og, fn_name, None)
            if fn is None:
                continue
            try:
                if fn_name == "set_graph_enabled":
                    fn(graph_path, False)
                else:
                    fn(graph_path, "Disabled")
            except Exception:
                pass
        # Drop the playback tick + controller so compute() cannot run even if
        # the graph prim is reconstituted from a referenced USD layer.
        try:
            keys = og.Controller.Keys
            og.Controller.edit(
                graph_path,
                {
                    keys.DELETE_NODES: [
                        "OnPlaybackTick",
                        "ArticulationController",
                        "PublishJointState",
                        "SubscribeJointState",
                        "ReadJointState",
                        "IsaacReadSimulationTime",
                        "OnImpulseEvent",
                    ],
                },
            )
        except Exception:
            pass
        try:
            og.Controller.edit(
                graph_path,
                {
                    og.Controller.Keys.SET_VALUES: [
                        ("state:enabled", False),
                    ]
                },
            )
        except Exception:
            pass
        try:
            delete_fn = getattr(og, "delete_graph", None) or getattr(
                og.Controller, "delete_graph", None
            )
            if delete_fn is not None:
                delete_fn(graph_path)
        except Exception:
            pass
    if prim is not None:
        try:
            attr = prim.GetAttribute("state:enabled")
            if attr and attr.IsValid():
                attr.Set(False)
        except Exception:
            pass


def _iter_robot_prims(root):
    """Yield all composed prims under ``root``, including inactive ones.

    Default ``Usd.PrimRange`` skips inactive prims, which broke ROS graph
    deletion after we deactivated ``ROS_ActionGraph`` first.
    """

    try:
        yield from Usd.PrimRange(root, Usd.PrimAllPrimsPredicate)
    except Exception:
        yield from Usd.PrimRange(root)


def _collect_ros_graph_paths(stage, robot_prim_path: str) -> list[str]:
    """Find ROS / OmniGraph paths under a robot (active or inactive)."""

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return []
    paths: list[str] = []
    direct = stage.GetPrimAtPath(f"{robot_prim_path}/ROS_ActionGraph")
    if direct and direct.IsValid():
        paths.append(str(direct.GetPath()))
    for prim in _iter_robot_prims(root):
        name = prim.GetName()
        type_name = str(prim.GetTypeName() or "")
        path = str(prim.GetPath())
        if (
            name == "ROS_ActionGraph"
            or type_name == "OmniGraph"
            or name.endswith("ActionGraph")
        ):
            paths.append(path)
    # Unique, parents before children reversed for delete (children first).
    return sorted(set(paths), key=len, reverse=True)


def _clear_articulation_controller_targets(stage, robot_prim_path: str) -> int:
    """Blank ``inputs:robotPath`` / ``inputs:targetPrim`` so OG cannot drive joints."""

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return 0
    cleared = 0
    for prim in _iter_robot_prims(root):
        path_l = str(prim.GetPath()).lower()
        name_l = prim.GetName().lower()
        if "articulationcontroller" not in name_l and "articulationcontroller" not in path_l:
            continue
        for attr_name in ("inputs:robotPath", "inputs:targetPrim"):
            attr = prim.GetAttribute(attr_name)
            if not attr or not attr.IsValid():
                continue
            try:
                type_name = str(attr.GetTypeName()).lower()
                if "array" in type_name or "token[]" in type_name or "string[]" in type_name:
                    attr.Set([])
                else:
                    attr.Set("")
                cleared += 1
            except Exception:
                try:
                    attr.Set("")
                    cleared += 1
                except Exception:
                    pass
    return cleared


def disable_robot_ros_graphs(stage, robot_prim_path: str) -> None:
    """Stop ROS / Action OmniGraphs from evaluating (BT owns joint commands)."""

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return
    if not root.IsActive():
        root.SetActive(True)

    graph_paths = _collect_ros_graph_paths(stage, robot_prim_path)
    # Mute / delete tick nodes BEFORE clearing targets — clearing alone turns
    # ``root_joint`` errors into ``No robot prim found`` spam.
    for path in graph_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and not prim.IsActive():
            try:
                prim.SetActive(True)
            except Exception:
                pass
        _mute_omni_graph(path, prim if prim and prim.IsValid() else None)

    cleared = _clear_articulation_controller_targets(stage, robot_prim_path)

    disabled = 0
    for path in graph_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        try:
            if prim.IsActive():
                prim.SetActive(False)
                disabled += 1
        except Exception:
            pass
    # Also deactivate leftover controller nodes under the robot.
    for prim in _iter_robot_prims(root):
        name_l = prim.GetName().lower()
        if name_l not in (
            "articulationcontroller",
            "onplaybacktick",
            "publishjointstate",
            "subscribejointstate",
            "readjointstate",
        ):
            continue
        try:
            if prim.IsActive():
                prim.SetActive(False)
                disabled += 1
        except Exception:
            pass
    if disabled or cleared or graph_paths:
        print(
            f"[SCENE] Disabled ROS/ActionGraph under {robot_prim_path} "
            f"(graphs={len(graph_paths)}, prims={disabled}, "
            f"controller_inputs_cleared={cleared})"
        )


def remove_robot_ros_graphs(
    stage, robot_prim_path: str, *, quiet: bool = False
) -> int:
    """Delete ROS/Action OmniGraphs under a robot; leave the robot prim active.

    Do **not** ``SetActive(False)`` the robot afterward — reactivating later
    reloads authored ``ROS_ActionGraph`` from the USD layer and OmniGraph
    immediately starts ticking against ``root_joint`` again.

    Important: collect + mute + RemovePrim while the graph is still active.
    Default ``Usd.PrimRange`` skips inactive prims, so deactivating first made
    deletion a no-op while Fabric kept evaluating the graph.
    """

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return 0
    if not root.IsActive():
        try:
            root.SetActive(True)
        except Exception:
            pass

    graph_paths = _collect_ros_graph_paths(stage, robot_prim_path)
    # Reactivate so Fabric/USD delete APIs can see the prims.
    for path in graph_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and not prim.IsActive():
            try:
                prim.SetActive(True)
            except Exception:
                pass
        _mute_omni_graph(path, prim if prim and prim.IsValid() else None)

    removed = 0
    for path in graph_paths:
        try:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            _mute_omni_graph(path, prim)
            stage.RemovePrim(Sdf.Path(path))
            # Confirm gone; if a stronger layer reconstitutes it, deactivate.
            leftover = stage.GetPrimAtPath(path)
            if leftover and leftover.IsValid():
                try:
                    leftover.SetActive(False)
                except Exception:
                    pass
                if not quiet:
                    print(
                        f"[SCENE] ROS graph still composed after RemovePrim: "
                        f"{path} (deactivated)"
                    )
            else:
                removed += 1
        except Exception as exc:
            if not quiet:
                print(f"[SCENE] remove ROS graph {path}: {exc}")

    # Clear any reconstituted ArticulationController targets as a last resort.
    _clear_articulation_controller_targets(stage, robot_prim_path)

    if (removed or graph_paths) and not quiet:
        print(
            f"[SCENE] Removed {removed}/{len(graph_paths)} ROS/ActionGraph "
            f"prim(s) under {robot_prim_path}"
        )
    return removed


def disable_all_ur5e_ros_graphs(stage) -> None:
    """Disable ROS graphs on every UR5e, including stations not selected to run."""

    robots = stage.GetPrimAtPath(cfg.ROBOTS_SCOPE)
    if not robots or not robots.IsValid():
        return
    for child in robots.GetChildren():
        if child.GetName().startswith("UR5e_"):
            disable_robot_ros_graphs(stage, str(child.GetPath()))


def remove_all_ur5e_ros_graphs(stage, selected=None) -> None:
    """Delete ROS ActionGraphs on every UR5e (selected and idle).

    When ``selected`` is provided, only those stations emit SCENE logs.
    """

    keep = None
    if selected is not None:
        keep = {spec.robot_prim_path for spec in selected}
    robots = stage.GetPrimAtPath(cfg.ROBOTS_SCOPE)
    if not robots or not robots.IsValid():
        return
    for child in robots.GetChildren():
        if child.GetName().startswith("UR5e_"):
            path = str(child.GetPath())
            quiet = keep is not None and path not in keep
            remove_robot_ros_graphs(stage, path, quiet=quiet)


def _world_rotation_matrix(prim) -> np.ndarray:
    """World rotation of ``prim``, safe under uniform hierarchy scale.

    DataHall UR5e robots author ``xformOp:scale = 100`` (meter asset on a cm
    stage). ``Gf.Matrix4d.ExtractRotation()`` mis-parses that scaled matrix and
    can be ~30–36° off; ``Gf.Transform`` factors scale out first.
    """

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    matrix = cache.GetLocalToWorldTransform(prim)
    xform = Gf.Transform()
    xform.SetMatrix(matrix)
    quat = xform.GetRotation().GetQuat()
    w = float(quat.GetReal())
    x, y, z = (float(v) for v in quat.GetImaginary())
    return cfg._quat_to_rot_matrix(np.array([w, x, y, z], dtype=np.float64))


def _world_translation(prim) -> np.ndarray:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    translate = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array(
        [float(translate[0]), float(translate[1]), float(translate[2])],
        dtype=np.float64,
    )


def repair_robot_gripper_mount_joints(stage, robot_prim_path: str | None = None) -> int:
    """Set ``robot_gripper_joint`` to the Assembler-style wrist↔base mount.

    Compare / contrast with
    https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_import_assemble_manipulator.html
    and DataHall UR10e:

    | | Tutorial / UR10e | DataHall UR5e (authored) |
    |---|---|---|
    | Attach | ``wrist_3_link`` ↔ gripper base | same via ``robot_gripper_joint`` |
    | Z+90 | Assembler button / ``ee_link`` Xform | rest pose on ``Gripper/`` Xform |
    | Fixed joint | UR10e ``ee_joint`` is **identity** | UR5e joint had a **different** 90° |
    | Articulation | single ``root_joint`` | nested Robotiq root (+ arm root) |

    After we strip the nested Robotiq root (UR10e parity), PhysX enforces the
    fixed joint. Authored ``localRot0`` ≠ the Xform mount, so the gripper
    snaps wrong. The true Xform mount (scale-safe) is a pure **Z+90°** —
    exactly the tutorial attach adjust.

    Fix: write that wrist→base relative pose into ``localPos0`` / ``localRot0``
    (``local*1`` = identity) and set ``GRIPPER_AXES_IN_EE`` from it.
    """

    roots: list = []
    if robot_prim_path:
        prim = stage.GetPrimAtPath(robot_prim_path)
        if prim and prim.IsValid():
            roots.append(prim)
    else:
        robots = stage.GetPrimAtPath(cfg.ROBOTS_SCOPE)
        if robots and robots.IsValid():
            roots.extend(
                child
                for child in robots.GetChildren()
                if child.GetName().startswith("UR5e_")
            )
    repaired = 0
    mount_axes = None
    for root in roots:
        joint_prim = None
        for prim in Usd.PrimRange(root):
            if prim.GetName() == "robot_gripper_joint":
                joint_prim = prim
                break
        if joint_prim is None:
            continue
        joint = UsdPhysics.Joint(joint_prim)
        bodies0 = list(joint.GetBody0Rel().GetTargets())
        bodies1 = list(joint.GetBody1Rel().GetTargets())
        if not bodies0 or not bodies1:
            print(f"[SCENE] Skip mount repair (missing bodies): {joint_prim.GetPath()}")
            continue
        body0 = stage.GetPrimAtPath(bodies0[0])
        body1 = stage.GetPrimAtPath(bodies1[0])
        if not body0.IsValid() or not body1.IsValid():
            continue
        r0 = _world_rotation_matrix(body0)
        r1 = _world_rotation_matrix(body1)
        t0 = _world_translation(body0)
        t1 = _world_translation(body1)
        # W1 = W0 * L0  (with L1 = identity)  =>  R_L0 = R0.T @ R1, t_L0 = R0.T @ (t1-t0)
        r_local = r0.T @ r1
        t_local = r0.T @ (t1 - t0)
        quat = cfg._rot_matrix_to_quat_wxyz(r_local)
        w, x, y, z = (float(v) for v in quat)
        joint_prim.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(float(t_local[0]), float(t_local[1]), float(t_local[2]))
        )
        joint_prim.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(
            Gf.Quatf(w, x, y, z)
        )
        joint_prim.CreateAttribute("physics:localPos1", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(0.0, 0.0, 0.0)
        )
        joint_prim.CreateAttribute("physics:localRot1", Sdf.ValueTypeNames.Quatf).Set(
            Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        )
        mount_axes = r_local
        repaired += 1
    if mount_axes is not None:
        cfg.set_gripper_axes_in_ee(mount_axes)
    return repaired


def harden_robotiq_mount_against_detach(stage, root_path: str) -> None:
    """Keep the 2F-85 attached when links graze each other or the hall.

    Self-collision between ``wrist_3`` and the Robotiq base (or a finite
    FixedJoint break force) is the usual reason the gripper "falls off".
    """

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return

    # 1) Unbreakable wrist↔gripper fixed joint.
    joint_prim = None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == "robot_gripper_joint":
            joint_prim = prim
            break
    if joint_prim is not None:
        for attr_name, value in (
            ("physics:breakForce", float(cfg.GRIPPER_MOUNT_BREAK_FORCE)),
            ("physics:breakTorque", float(cfg.GRIPPER_MOUNT_BREAK_TORQUE)),
        ):
            try:
                joint_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float).Set(
                    float(value)
                )
            except Exception as exc:
                print(f"[SCENE] {attr_name} on {joint_prim.GetPath()}: {exc}")
        try:
            PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        except Exception:
            pass
        print(
            f"[SCENE] Gripper mount hardened {joint_prim.GetPath()} "
            f"breakForce={cfg.GRIPPER_MOUNT_BREAK_FORCE} "
            f"breakTorque={cfg.GRIPPER_MOUNT_BREAK_TORQUE}"
        )

    # 2) Re-assert no articulation self-collision (wrist↔base would detach).
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        try:
            art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            art.CreateEnabledSelfCollisionsAttr(False).Set(False)
        except Exception:
            pass

    if not bool(cfg.GRIPPER_DISABLE_BASE_COLLISION):
        return

    # 3) Turn off collision on Robotiq base / housing (pads keep grasp contacts).
    #    Overlap with wrist_3 is common after the Z+90 mount repair.
    disable_tokens = (
        "base_link",
        "robotiq_base",
        "gripper_base",
        "outer_knuckle",  # proximal shells often clip wrist when closed
    )
    keep_tokens = ("pad", "fingertip", "inner_finger", "finger_tip")
    disabled = 0
    wrist_prim = None
    base_prims: list = []
    for prim in Usd.PrimRange(root):
        path_l = str(prim.GetPath()).lower()
        name_l = prim.GetName().lower()
        if "wrist_3" in name_l or name_l == "wrist_3_link":
            wrist_prim = prim
        if "robotiq" not in path_l and "gripper" not in path_l:
            continue
        if any(k in name_l or k in path_l for k in keep_tokens):
            continue
        if not any(t in name_l or t in path_l for t in disable_tokens):
            # Also disable CollisionAPI directly on the Robotiq_2F_85 xform kids
            # named exactly base_link's mesh descendants handled above.
            if name_l not in ("base_link", "robotiq_arg2f_base_link"):
                continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
            try:
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(
                    False
                ).Set(False)
                disabled += 1
            except Exception:
                pass
        if name_l in ("base_link", "robotiq_arg2f_base_link") or "base_link" in path_l:
            base_prims.append(prim)

    # 4) Explicit filtered pairs wrist_3 ↔ gripper base (belt-and-suspenders).
    if wrist_prim is not None and base_prims:
        try:
            api = UsdPhysics.FilteredPairsAPI.Apply(wrist_prim)
            rel = api.CreateFilteredPairsRel()
            targets = list(rel.GetTargets()) if rel else []
            for base in base_prims:
                path = base.GetPath()
                if path not in targets:
                    targets.append(path)
            rel.SetTargets(targets)
            print(
                f"[SCENE] Filtered wrist collisions vs {len(base_prims)} "
                f"gripper base prim(s) under {root_path}"
            )
        except Exception as exc:
            print(f"[SCENE] FilteredPairs wrist↔gripper failed: {exc}")

    if disabled:
        print(
            f"[SCENE] Disabled {disabled} Robotiq base/mount collider(s) under {root_path}"
        )


def _revolute_axis_token(prim) -> str:
    """Map ``physics:axis`` on a revolute joint to a PhysX mimic instance name."""

    axis = prim.GetAttribute("physics:axis")
    value = str(axis.Get() if axis and axis.HasAuthoredValueOpinion() else "Z").upper()
    return {
        "X": UsdPhysics.Tokens.rotX,
        "Y": UsdPhysics.Tokens.rotY,
        "Z": UsdPhysics.Tokens.rotZ,
    }.get(value, UsdPhysics.Tokens.rotZ)


def repair_robotiq_mimic_joints(stage, robot_prim_path: str | None = None) -> int:
    """After nested-root strip, re-bind PhysX mimic APIs to the joint revolute axis.

    DataHall authors ``PhysxMimicJointAPI:rotX`` on several finger joints whose
    ``physics:axis`` is ``Z``. That works while Robotiq owns its own articulation
    root, but after merging into the arm articulation PhysX logs
    ``failed to find internal joint object for PhysxMimicJointAPI``. Re-apply
    mimic on the matching rot* instance and give a non-zero natural frequency.
    """

    roots: list = []
    if robot_prim_path:
        prim = stage.GetPrimAtPath(robot_prim_path)
        if prim and prim.IsValid():
            roots.append(prim)
    else:
        robots = stage.GetPrimAtPath(cfg.ROBOTS_SCOPE)
        if robots and robots.IsValid():
            roots.extend(
                child
                for child in robots.GetChildren()
                if child.GetName().startswith("UR5e_")
            )
    repaired = 0
    mimic_axes = (
        UsdPhysics.Tokens.rotX,
        UsdPhysics.Tokens.rotY,
        UsdPhysics.Tokens.rotZ,
    )
    for root in roots:
        for prim in Usd.PrimRange(root):
            path_l = str(prim.GetPath()).lower()
            if "robotiq" not in path_l:
                continue
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            existing: dict[str, tuple[float, float, float, list]] = {}
            for axis in mimic_axes:
                if not prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                    continue
                api = PhysxSchema.PhysxMimicJointAPI(prim, axis)
                gearing = float(api.GetGearingAttr().Get() or -1.0)
                damping = float(api.GetDampingRatioAttr().Get() or 0.0)
                nat_freq = float(api.GetNaturalFrequencyAttr().Get() or 0.0)
                ref = list(api.GetReferenceJointRel().GetTargets())
                existing[axis] = (gearing, damping, nat_freq, ref)
            if not existing:
                continue
            target_axis = _revolute_axis_token(prim)
            # Prefer authored values from any instance; keep referenceJoint.
            gearing, damping, nat_freq, ref = next(iter(existing.values()))
            if target_axis in existing:
                gearing, damping, nat_freq, ref = existing[target_axis]
            for axis in list(existing):
                try:
                    prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, axis)
                except Exception:
                    pass
            api = PhysxSchema.PhysxMimicJointAPI.Apply(prim, target_axis)
            api.CreateGearingAttr(gearing).Set(gearing)
            # Authored USD often has naturalFrequency=0 → knuckles buckle and
            # leave a pad gap. Always force the configured near-rigid coupling.
            freq = float(cfg.ROBOTIQ_MIMIC_NATURAL_FREQUENCY)
            damp = float(cfg.ROBOTIQ_MIMIC_DAMPING_RATIO)
            api.CreateNaturalFrequencyAttr(freq).Set(freq)
            api.CreateDampingRatioAttr(damp).Set(damp)
            if ref:
                api.GetReferenceJointRel().SetTargets(ref)
            repaired += 1
    if repaired:
        print(
            f"[SCENE] Repaired Robotiq PhysxMimicJointAPI on {repaired} joint(s) "
            f"(freq={cfg.ROBOTIQ_MIMIC_NATURAL_FREQUENCY:g}, "
            f"dampingRatio={cfg.ROBOTIQ_MIMIC_DAMPING_RATIO:g})"
        )
    return repaired


def ensure_robotiq_link_masses(stage, robot_prim_path: str | None = None) -> int:
    """Author positive mass/inertia on Robotiq rigid links that lack both.

    ``base_link`` and the outer knuckles ship with ``PhysicsRigidBodyAPI`` but
    neither ``MassAPI`` nor colliders. PhysX then logs invalid inertia / negative
    mass, treats those links as free bodies, and MimicJoint APIs fail because the
    finger joints are no longer part of the arm articulation.
    """

    roots: list = []
    if robot_prim_path:
        prim = stage.GetPrimAtPath(robot_prim_path)
        if prim and prim.IsValid():
            roots.append(prim)
    else:
        robots = stage.GetPrimAtPath(cfg.ROBOTS_SCOPE)
        if robots and robots.IsValid():
            roots.extend(
                child
                for child in robots.GetChildren()
                if child.GetName().startswith("UR5e_")
            )
    fixed = 0
    mass_map = dict(getattr(cfg, "ROBOTIQ_LINK_MASS_KG", {}) or {})
    default_mass = float(getattr(cfg, "ROBOTIQ_DEFAULT_LINK_MASS_KG", 0.025))
    inertia = tuple(
        float(x)
        for x in getattr(cfg, "ROBOTIQ_DEFAULT_DIAGONAL_INERTIA", (1e-4, 1e-4, 1e-4))
    )
    for root in roots:
        for prim in Usd.PrimRange(root):
            path_l = str(prim.GetPath()).lower()
            if "robotiq" not in path_l:
                continue
            name = prim.GetName()
            # Xform root must never be a free rigid body after art-root strip.
            if name == "Robotiq_2F_85":
                for api in (
                    UsdPhysics.RigidBodyAPI,
                    UsdPhysics.CollisionAPI,
                    UsdPhysics.MassAPI,
                ):
                    try:
                        if prim.HasAPI(api):
                            prim.RemoveAPI(api)
                    except Exception:
                        pass
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            current = mass_api.GetMassAttr().Get() if mass_api.GetMassAttr() else None
            try:
                current_f = float(current) if current is not None else -1.0
            except Exception:
                current_f = -1.0
            if current_f > 1e-6:
                continue
            mass = float(mass_map.get(name, default_mass))
            mass_api.CreateMassAttr(mass).Set(mass)
            try:
                mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia)).Set(
                    Gf.Vec3f(*inertia)
                )
            except Exception:
                pass
            try:
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(True).Set(True)
            except Exception:
                pass
            fixed += 1
    if fixed:
        print(
            f"[SCENE] Authored MassAPI on {fixed} Robotiq link(s) "
            f"(missing mass/colliders → was breaking articulation + MimicJoint)"
        )
    return fixed


def strip_nested_gripper_articulation_roots(
    stage,
    robot_prim_path: str | None = None,
    *,
    repair_mount: bool = True,
) -> int:
    """Remove Robotiq ``ArticulationRootAPI`` so the arm ``root_joint`` owns the chain.

    ``DataHall_6r_ur5e`` authors a second articulation root on
    ``Gripper/Robotiq_2F_85``. UR10e DataHall does not. Nested roots make PhysX
    drop the arm articulation while Isaac ``JointStateSensor`` keeps querying
    ``.../root_joint`` every frame.

    ``repair_mount`` must be False after the first ``world.reset()`` — PhysX
    rejects joint local-pose updates once simulation has started.
    """

    removed = 0
    roots: list = []
    if robot_prim_path:
        prim = stage.GetPrimAtPath(robot_prim_path)
        if prim and prim.IsValid():
            roots.append(prim)
    else:
        robots = stage.GetPrimAtPath(cfg.ROBOTS_SCOPE)
        if robots and robots.IsValid():
            roots.extend(
                child
                for child in robots.GetChildren()
                if child.GetName().startswith("UR5e_")
            )
    for root in roots:
        for prim in Usd.PrimRange(root):
            path_l = str(prim.GetPath()).lower()
            if "robotiq" not in path_l and "gripper" not in path_l:
                continue
            # Never strip the arm root_joint.
            if prim.GetName() == "root_joint":
                continue
            had_root = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            if had_root:
                try:
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                    removed += 1
                    print(f"[SCENE] Removed nested articulation root {prim.GetPath()}")
                except Exception as exc:
                    print(
                        f"[SCENE] Could not strip articulation root {prim.GetPath()}: {exc}"
                    )
            # Former articulation-root Xform must not become a free rigid body
            # (PhysX logs invalid inertia / negative mass and the gripper wobbles).
            if prim.GetName() == "Robotiq_2F_85" or had_root:
                for api in (
                    UsdPhysics.RigidBodyAPI,
                    UsdPhysics.CollisionAPI,
                    UsdPhysics.MassAPI,
                ):
                    try:
                        if prim.HasAPI(api):
                            prim.RemoveAPI(api)
                    except Exception:
                        pass
                # Nested IsaacRobotAPI marks Robotiq as its own robot; after the
                # articulation root is gone PhysX can invent a free body here.
                try:
                    schemas = [str(s) for s in (prim.GetAppliedSchemas() or [])]
                    if any("IsaacRobotAPI" in s for s in schemas):
                        prim.RemoveAPI("IsaacRobotAPI")
                except Exception:
                    pass
                for attr_name in (
                    "physics:rigidBodyEnabled",
                    "physics:collisionEnabled",
                    "physics:mass",
                ):
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.IsValid():
                        try:
                            if "Enabled" in attr_name:
                                attr.Set(False)
                        except Exception:
                            pass
    if removed:
        print(
            f"[SCENE] Stripped {removed} nested Robotiq articulation root(s)"
        )
    # Always rebind mimic APIs: DataHall authors PhysxMimicJointAPI:rotX on
    # joints whose physics:axis is Z. Skipping repair (e.g. when the nested root
    # was already gone) leaves PhysX ``failed to find internal joint object``.
    repair_robotiq_mimic_joints(stage, robot_prim_path)
    # Rewrite wrist↔Robotiq fixed joint to scale-safe Xform mount (Z+90).
    # Only before physics start — after reset PhysX rejects local-pose updates.
    if repair_mount:
        repair_robot_gripper_mount_joints(stage, robot_prim_path)
    # Mass-less base_link / outer knuckles break articulation membership.
    ensure_robotiq_link_masses(stage, robot_prim_path)
    return removed


def _is_joint_state_sensor_prim(prim) -> bool:
    """True for Isaac / OmniGraph joint-state readers that query articulations."""

    name_l = prim.GetName().lower()
    path_l = str(prim.GetPath()).lower()
    type_name = str(prim.GetTypeName() or "")
    type_l = type_name.lower()
    if (
        "jointstate" in name_l
        or "joint_state" in name_l
        or "jointstatesensor" in path_l
        or "readjointstate" in name_l
        or "isaacsensor" in type_l
        or "jointstate" in type_l
        or type_name
        in (
            "IsaacJointStateSensor",
            "IsaacSensorCreateJointState",
            "IsaacSensor",
        )
    ):
        return True
    # Experimental physics JointStateSensor often stores the articulation path
    # on a string/token attr even when the prim name is generic.
    for attr in prim.GetAttributes():
        attr_name = attr.GetName().lower()
        if "articulation" not in attr_name and "robot" not in attr_name:
            continue
        try:
            val = attr.Get()
        except Exception:
            continue
        text = str(val).lower() if val is not None else ""
        if "root_joint" in text or "/robots/ur5e_" in text:
            return True
    return False


def remove_robot_joint_state_sensors(
    stage, robot_prim_path: str, *, quiet: bool = False
) -> int:
    """Remove JointStateSensor registrations for ``robot_prim_path``.

    Experimental ``JointStateSensor`` is **not** a USD prim — it is a C++ plugin
    keyed by the articulation root path (usually ``…/root_joint``). Deleting USD
    nodes alone does nothing. Call ``IJointStateSensor.remove_sensor`` and also
    drop any leftover USD/OG prims that look like joint-state readers.
    """

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return 0
    if not root.IsActive():
        try:
            root.SetActive(True)
        except Exception:
            pass

    removed = 0
    # 1) Tear down C++ sensors on known articulation-root paths.
    candidate_roots = [f"{robot_prim_path}/root_joint", robot_prim_path]
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidate_roots.append(str(prim.GetPath()))
    try:
        from isaacsim.sensors.experimental.physics.impl.extension import (
            get_joint_state_sensor_interface,
        )

        iface = get_joint_state_sensor_interface()
    except Exception:
        iface = None
    if iface is not None:
        for art_path in dict.fromkeys(candidate_roots):
            try:
                iface.remove_sensor(art_path)
                removed += 1
            except Exception:
                pass

    # 2) Best-effort USD / OmniGraph leftovers.
    to_remove: list[str] = []
    for prim in Usd.PrimRange(root):
        if _is_joint_state_sensor_prim(prim):
            to_remove.append(str(prim.GetPath()))
    for path in sorted(to_remove, key=len, reverse=True):
        try:
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(Sdf.Path(path))
                removed += 1
        except Exception as exc:
            if not quiet:
                print(f"[SCENE] remove JointStateSensor prim {path}: {exc}")
    if removed and not quiet:
        print(
            f"[SCENE] Cleared JointStateSensor for {robot_prim_path} "
            f"(cpp+usd actions={removed})"
        )
    return removed


def _root_joint_articulation_status(stage, robot_prim_path: str) -> str:
    """Describe whether ``…/root_joint`` is a live PhysX articulation root."""

    root_joint = stage.GetPrimAtPath(f"{robot_prim_path}/root_joint")
    if not root_joint or not root_joint.IsValid():
        return "missing_prim"
    if not root_joint.IsActive():
        return "inactive_prim"
    has_art_root = root_joint.HasAPI(UsdPhysics.ArticulationRootAPI)
    enabled = None
    if root_joint.HasAPI(PhysxSchema.PhysxArticulationAPI):
        attr = PhysxSchema.PhysxArticulationAPI(root_joint).GetArticulationEnabledAttr()
        if attr and attr.IsValid():
            enabled = bool(attr.Get())
    elif has_art_root:
        # ArticulationRootAPI without PhysxArticulationAPI → enabled by default.
        enabled = True
    if not has_art_root and enabled is None:
        # Some stages put ArticulationRootAPI on the robot Xform, not root_joint.
        robot = stage.GetPrimAtPath(robot_prim_path)
        if robot and robot.IsValid() and robot.HasAPI(UsdPhysics.ArticulationRootAPI):
            has_art_root = True
            if robot.HasAPI(PhysxSchema.PhysxArticulationAPI):
                attr = PhysxSchema.PhysxArticulationAPI(robot).GetArticulationEnabledAttr()
                if attr and attr.IsValid():
                    enabled = bool(attr.Get())
                else:
                    enabled = True
            else:
                enabled = True
            return f"root_on_robot enabled={enabled}"
    return f"art_root={has_art_root} enabled={enabled}"


def freeze_inactive_ur5e_robots(stage, selected) -> None:
    """Hide idle UR5es and remove them from the PhysX articulation solve.

    Idle arms used to stay ``articulationEnabled=True`` (to avoid JointStateSensor
    spam). Sensors are stripped now, and leaving 5 extra UR5e+Robotiq articulations
    (with PhysxMimicJoint) on the GPU solver is what dies with CUDA 700 during
    TipLift after grasp. Disable their articulations + colliders.
    """

    selected = tuple(selected)
    if len(selected) >= len(cfg.STATIONS):
        return
    keep = {spec.robot_prim_path for spec in selected}
    robots = stage.GetPrimAtPath(cfg.ROBOTS_SCOPE)
    if not robots or not robots.IsValid():
        return
    disabled = 0
    for child in robots.GetChildren():
        if not child.GetName().startswith("UR5e_"):
            continue
        path = str(child.GetPath())
        if path in keep:
            if not child.IsActive():
                child.SetActive(True)
            try:
                UsdGeom.Imageable(child).MakeVisible()
            except Exception:
                pass
            # Selected arm still must not run ROS (BT owns the joints).
            remove_robot_ros_graphs(stage, path)
            remove_robot_joint_state_sensors(stage, path)
            child = stage.GetPrimAtPath(path)
            for prim in Usd.PrimRange(child):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                    art.CreateArticulationEnabledAttr(True).Set(True)
            continue

        if not child.IsActive():
            try:
                child.SetActive(True)
            except Exception:
                pass
        remove_robot_ros_graphs(stage, path, quiet=True)
        remove_robot_joint_state_sensors(stage, path, quiet=True)
        child = stage.GetPrimAtPath(path)
        if not child or not child.IsValid():
            continue
        for prim in Usd.PrimRange(child):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                art.CreateArticulationEnabledAttr(False).Set(False)
                disabled += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
                try:
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(
                        False
                    ).Set(False)
                except Exception:
                    pass
        try:
            UsdGeom.Imageable(child).MakeInvisible()
        except Exception:
            pass
    if disabled:
        print(
            f"[SCENE] Disabled PhysX articulation on {disabled} idle UR5e root(s) "
            "(keeps GPU MimicJoint load on the selected arm only)"
        )


def freeze_inactive_network_cables(stage, selected) -> None:
    """Disable rigid/collision physics on non-selected ``/World/NetworkCables``.

    Authored USD enables RigidBody on every station's crystal heads. Idle cables
    still simulate and can NaN (``Invalid PhysX transform``) while the BT arm
    runs another station — freeze them like idle UR5es.
    """

    selected = tuple(selected)
    keep = {str(spec.cable_root_path) for spec in selected}
    cables = stage.GetPrimAtPath(cfg.CABLES_SCOPE)
    if not cables or not cables.IsValid():
        return
    frozen = 0
    for child in cables.GetChildren():
        path = str(child.GetPath())
        if path in keep:
            continue
        disabled_rb = 0
        disabled_col = 0
        for prim in Usd.PrimRange(child):
            try:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(
                        False
                    ).Set(False)
                    disabled_rb += 1
            except Exception:
                attr = prim.GetAttribute("physics:rigidBodyEnabled")
                if attr and attr.IsValid():
                    try:
                        attr.Set(False)
                        disabled_rb += 1
                    except Exception:
                        pass
            try:
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(
                        False
                    ).Set(False)
                    disabled_col += 1
            except Exception:
                pass
            for attr_name in (
                "physxDeformableBody:deformableEnabled",
                "physxDeformable:deformableEnabled",
            ):
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    try:
                        attr.Set(False)
                    except Exception:
                        pass
        try:
            UsdGeom.Imageable(child).MakeInvisible()
        except Exception:
            pass
        frozen += 1
        print(
            f"[SCENE] Idle cable physics OFF {path} "
            f"(rigid={disabled_rb}, colliders={disabled_col})"
        )
    if frozen:
        print(
            f"[SCENE] Froze {frozen} idle NetworkCable(s) "
            f"(kept {sorted(p.split('/')[-1] for p in keep) or '-'})"
        )


def configure_robot_physics(stage, root_path: str) -> None:
    """Enable link/gripper collision, optional self-collision, PhysX params."""

    from omni.physx.scripts import utils as physx_utils

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    self_col = bool(cfg.ARTICULATION_ENABLE_SELF_COLLISIONS)
    collision_count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            art.CreateArticulationEnabledAttr(True).Set(True)
            art.CreateEnabledSelfCollisionsAttr(self_col).Set(self_col)
            try:
                art.CreateSolverPositionIterationCountAttr(
                    int(cfg.ARTICULATION_SOLVER_POSITION_ITERS)
                ).Set(int(cfg.ARTICULATION_SOLVER_POSITION_ITERS))
                art.CreateSolverVelocityIterationCountAttr(
                    int(cfg.ARTICULATION_SOLVER_VELOCITY_ITERS)
                ).Set(int(cfg.ARTICULATION_SOLVER_VELOCITY_ITERS))
            except Exception:
                pass
            try:
                art.CreateSleepThresholdAttr(float(cfg.ARTICULATION_SLEEP_THRESHOLD)).Set(
                    float(cfg.ARTICULATION_SLEEP_THRESHOLD)
                )
                art.CreateStabilizationThresholdAttr(
                    float(cfg.ARTICULATION_STABILIZATION_THRESHOLD)
                ).Set(float(cfg.ARTICULATION_STABILIZATION_THRESHOLD))
            except Exception:
                pass
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                contact_report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                contact_report.CreateThresholdAttr().Set(0.0)
            except Exception:
                pass
            try:
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(True).Set(True)
            except Exception:
                pass
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateDisableGravityAttr(bool(cfg.LINK_DISABLE_GRAVITY)).Set(
                bool(cfg.LINK_DISABLE_GRAVITY)
            )
            try:
                rb.CreateLinearDampingAttr(float(cfg.LINK_LINEAR_DAMPING)).Set(
                    float(cfg.LINK_LINEAR_DAMPING)
                )
                rb.CreateAngularDampingAttr(float(cfg.LINK_ANGULAR_DAMPING)).Set(
                    float(cfg.LINK_ANGULAR_DAMPING)
                )
                rb.CreateMaxLinearVelocityAttr(float(cfg.LINK_MAX_LINEAR_VELOCITY)).Set(
                    float(cfg.LINK_MAX_LINEAR_VELOCITY)
                )
                rb.CreateMaxAngularVelocityAttr(
                    float(cfg.LINK_MAX_ANGULAR_VELOCITY)
                ).Set(float(cfg.LINK_MAX_ANGULAR_VELOCITY))
            except Exception:
                pass
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint):
            try:
                japi = PhysxSchema.PhysxJointAPI.Apply(prim)
                japi.CreateJointFrictionAttr(float(cfg.JOINT_FRICTION)).Set(
                    float(cfg.JOINT_FRICTION)
                )
                japi.CreateArmatureAttr(float(cfg.JOINT_ARMATURE)).Set(
                    float(cfg.JOINT_ARMATURE)
                )
            except Exception:
                pass
        path = str(prim.GetPath()).lower()
        name_l = prim.GetName().lower()
        is_collision_prim = (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or name_l == "collisions"
            or "/collisions" in path
            or (prim.IsA(UsdGeom.Mesh) and "collision" in path)
        )
        if not is_collision_prim:
            continue
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True).Set(True)
        try:
            physx_utils.setCollider(prim, UsdPhysics.Tokens.convexHull)
        except Exception:
            pass
        collision_count += 1
    print(
        f"[SCENE] Robot physics {root_path}: collisions={collision_count} "
        f"self_collisions={self_col}"
    )
    harden_robotiq_mount_against_detach(stage, root_path)
    ensure_robotiq_link_masses(stage, root_path)


def _apply_drive_parameters(stage, root_path: str, parameters: dict[str, tuple]) -> None:
    """Apply meter-physical angular drives with cm-stage compensation (÷ mpu²).

    Same convention as ``ur10e_6x``: config stores meter-stage magnitudes;
    ``angular_drive_value_for_stage`` scales them for DataHall ``mpu=0.01``.
    Skipping that scale left maxForce ~10000× too small under gravity → collapse.
    """

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    mpu = meters_per_unit(stage)
    applied: list[str] = []
    for prim in Usd.PrimRange(root):
        values = parameters.get(prim.GetName())
        if values is None or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        stiffness, damping, max_force = (float(value) for value in values)
        stiffness = angular_drive_value_for_stage(stiffness, mpu)
        damping = angular_drive_value_for_stage(damping, mpu)
        max_force = angular_drive_value_for_stage(max_force, mpu)
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr(stiffness).Set(stiffness)
        drive.CreateDampingAttr(damping).Set(damping)
        drive.CreateMaxForceAttr(max_force).Set(max_force)
        try:
            drive.CreateTargetVelocityAttr(0.0).Set(0.0)
        except Exception:
            pass
        applied.append(
            f"{prim.GetName()}(K={stiffness:.3g},D={damping:.3g},F={max_force:.3g})"
        )
    if applied:
        print(
            f"[SCENE] Stage-scaled angular drives on {root_path} "
            f"(mpu={mpu:g}, ÷mpu²): " + ", ".join(applied)
        )


def apply_ur5e_drive_parameters(stage, root_path: str) -> None:
    _apply_drive_parameters(stage, root_path, cfg.UR5E_ARM_DRIVE_PARAMETERS)


def apply_robotiq_drive_parameters(stage, root_path: str) -> None:
    _apply_drive_parameters(stage, root_path, cfg.ROBOTIQ_DRIVE_PARAMETERS)


def apply_ur5e_home_pose(
    robot,
    *,
    apply_live: bool = False,
    arm_positions: np.ndarray | None = None,
) -> None:
    """Set the default and optionally live UR5e arm pose.

    ``arm_positions`` is length-6 in ``UR5E_ARM_JOINT_NAMES`` order. When omitted,
    uses ``UR5E_HOME_ARM`` (classic upright — often collides with DataHall after
    the layout rotate; prefer the station-specific IK home at work-table X).
    """

    arm = np.asarray(
        cfg.UR5E_HOME_ARM if arm_positions is None else arm_positions,
        dtype=np.float64,
    ).reshape(-1)
    if arm.shape != (len(cfg.UR5E_ARM_JOINT_NAMES),):
        raise ValueError("arm_positions must match UR5E_ARM_JOINT_NAMES")
    try:
        names = list(robot.dof_names)
    except Exception:
        names = []
    positions = np.zeros(len(names) if names else 6, dtype=np.float64)
    for index, name in enumerate(names or cfg.UR5E_ARM_JOINT_NAMES):
        if name in cfg.UR5E_ARM_JOINT_NAMES:
            source_index = cfg.UR5E_ARM_JOINT_NAMES.index(name)
            positions[index] = float(arm[source_index])
    robot.set_joints_default_state(positions=positions)
    if apply_live:
        robot.set_joint_positions(positions)


def compute_home_arm_above_work_table(
    stage, controller, spec: cfg.StationSpec
) -> np.ndarray:
    """IK arm joints: fingertips above the work-table center X (gripper down).

    Y follows the station cable so each side has its own home. Falls back to
    ``UR5E_HOME_ARM`` if IK fails. Uses ``home_cache.json`` when enabled.
    """

    from ur5e_6x_cable_insertions import home_cache
    from ur5e_6x_cable_insertions.primitives import (
        hand_from_tip,
        home_tip_above_work_table,
        meters_per_unit,
    )
    from ur5e_6x_cable_insertions.runtime_support import meters_to_stage

    tip_m = home_tip_above_work_table(stage, spec)
    if bool(cfg.HOME_CACHE_LOAD):
        cached = home_cache.load_station(
            spec.station_id, path=cfg.HOME_CACHE_PATH
        )
        if cached is not None:
            arm = np.asarray(cached["arm_rad"], dtype=np.float64).copy()
            print(
                f"[SCENE {spec.station_id}] home from cache "
                f"tip_m={np.round(tip_m, 4)} arm_rad={np.round(arm, 3)}"
            )
            return arm

    tool_ori = np.asarray(cfg.OBSERVE_TOOL_ORIENTATION, dtype=np.float64)
    lula_ori = np.asarray(cfg.OBSERVE_ORIENTATION, dtype=np.float64)
    hand_m = hand_from_tip(tip_m, tool_ori)
    mpu = meters_per_unit(stage)
    hand_stage = meters_to_stage(hand_m, mpu)
    try:
        action, success = controller._art_kinematics.compute_inverse_kinematics(
            target_position=hand_stage,
            target_orientation=lula_ori,
            position_tolerance=float(
                getattr(controller, "_pos_tolerance", cfg.HOVER_IK_POS_TOLERANCE_M)
            ),
            orientation_tolerance=float(
                getattr(controller, "_ori_tolerance", cfg.HOVER_IK_ORI_TOLERANCE_RAD)
            ),
        )
    except Exception as exc:
        print(
            f"[SCENE {spec.station_id}] home IK above work table failed ({exc}); "
            f"using classic UR5E_HOME_ARM"
        )
        return np.asarray(cfg.UR5E_HOME_ARM, dtype=np.float64).copy()

    if not success or getattr(action, "joint_positions", None) is None:
        print(
            f"[SCENE {spec.station_id}] home IK above work table did not converge; "
            f"using classic UR5E_HOME_ARM"
        )
        return np.asarray(cfg.UR5E_HOME_ARM, dtype=np.float64).copy()

    robot = controller._robot
    names = list(getattr(robot, "dof_names", []) or [])
    jp = list(action.joint_positions)
    arm = np.asarray(cfg.UR5E_HOME_ARM, dtype=np.float64).copy()
    for i, name in enumerate(cfg.UR5E_ARM_JOINT_NAMES):
        if name not in names:
            continue
        idx = names.index(name)
        if idx < len(jp) and jp[idx] is not None:
            arm[i] = float(jp[idx])
    print(
        f"[SCENE {spec.station_id}] home above work-table X "
        f"tip_m={np.round(tip_m, 4)} arm_rad={np.round(arm, 3)}"
    )
    if bool(cfg.HOME_CACHE_SAVE):
        try:
            home_cache.save_station(
                spec.station_id,
                arm_rad=arm,
                tip_m=tip_m,
                path=cfg.HOME_CACHE_PATH,
            )
        except Exception as exc:
            print(f"[HOME CACHE] save failed {spec.station_id}: {exc}")
    return arm


def apply_live_arm_gains(robot, station_id: str) -> None:
    """Boost articulation PD gains + max efforts so gravity cannot fold the arm.

    Idempotent via ``robot._ur5e_live_gains_applied``. Call after the physics
    view is ready (not before ``world.reset`` / tensor init).

    Isaac often keeps soft default ``max_efforts`` from the authored USD
    (shoulder ~150 Nm) even after DriveAPI maxForce is rewritten — under
    gravity that saturates immediately and the arm collapses at first move.
    """

    if getattr(robot, "_ur5e_live_gains_applied", False):
        return
    try:
        controller = robot.get_articulation_controller()
        kps, kds = controller.get_gains()
        kps = np.asarray(kps, dtype=np.float64).copy()
        kds = np.asarray(kds, dtype=np.float64).copy()
        names = list(robot.dof_names)
        indices = [names.index(name) for name in cfg.UR5E_ARM_JOINT_NAMES if name in names]
        before_kp = kps[indices].copy()
        before_kd = kds[indices].copy()
        kps[indices] = np.maximum(
            kps[indices] * float(cfg.UR5E_LIVE_STIFFNESS_MULTIPLIER),
            float(cfg.UR5E_LIVE_ARM_MIN_KP),
        )
        kds[indices] = np.maximum(
            kds[indices] * float(cfg.UR5E_LIVE_DAMPING_MULTIPLIER),
            float(cfg.UR5E_LIVE_ARM_MIN_KD),
        )
        for name in cfg.ROBOTIQ_DRIVE_PARAMETERS:
            if name in names:
                fi = names.index(name)
                kps[fi] = max(
                    float(kps[fi]) * float(cfg.ROBOTIQ_LIVE_STIFFNESS_MULTIPLIER),
                    float(cfg.ROBOTIQ_LIVE_MIN_KP),
                )
                kds[fi] = max(
                    float(kds[fi]) * float(cfg.ROBOTIQ_LIVE_DAMPING_MULTIPLIER),
                    float(cfg.ROBOTIQ_LIVE_MIN_KD),
                )
        controller.set_gains(kps=kps, kds=kds)

        effort_before = None
        effort_after = None
        controlled = list(indices)
        for name in cfg.ROBOTIQ_DRIVE_PARAMETERS:
            if name in names:
                controlled.append(names.index(name))
        arm_effort = float(cfg.UR5E_LIVE_ARM_MAX_EFFORT)
        finger_effort = float(cfg.ROBOTIQ_LIVE_MAX_EFFORT)
        for setter_name in ("set_max_efforts",):
            setter = getattr(controller, setter_name, None) or getattr(
                robot, setter_name, None
            )
            getter = getattr(controller, "get_max_efforts", None) or getattr(
                robot, "get_max_efforts", None
            )
            if setter is None:
                continue
            try:
                if getter is not None:
                    efforts = np.asarray(getter(), dtype=np.float64).copy()
                else:
                    efforts = np.full(len(names), arm_effort, dtype=np.float64)
                effort_before = efforts.copy()
                for fi in indices:
                    efforts[fi] = max(float(efforts[fi]), arm_effort)
                for name in cfg.ROBOTIQ_DRIVE_PARAMETERS:
                    if name in names:
                        fi = names.index(name)
                        efforts[fi] = max(float(efforts[fi]), finger_effort)
                setter(efforts)
                effort_after = efforts
                break
            except Exception as exc:
                print(f"[SCENE] {station_id} max_efforts via {setter_name} failed: {exc}")

        robot._ur5e_live_gains_applied = True
        msg = (
            f"[SCENE] {station_id} live arm gains "
            f"KP>={cfg.UR5E_LIVE_ARM_MIN_KP:g} "
            f"({np.round(before_kp, 1)} -> {np.round(kps[indices], 1)}), "
            f"KD>={cfg.UR5E_LIVE_ARM_MIN_KD:g} "
            f"({np.round(before_kd, 1)} -> {np.round(kds[indices], 1)})"
        )
        if effort_before is not None and effort_after is not None:
            msg += (
                f", max_effort "
                f"({np.round(effort_before[indices], 1)} -> "
                f"{np.round(effort_after[indices], 1)})"
            )
        else:
            msg += " (max_efforts API unavailable)"
        print(msg)
    except Exception as exc:
        raise RuntimeError(f"Could not apply live gains to {station_id}") from exc


def apply_live_arm_damping(robot, station_id: str) -> None:
    """Backward-compatible alias for :func:`apply_live_arm_gains`."""

    apply_live_arm_gains(robot, station_id)


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


def _set_imageable_visibility(prim, mode: str) -> None:
    """``inherited`` | ``visible`` | ``invisible`` — prefer inherited for leaves."""

    if not prim or not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    try:
        if mode == "inherited":
            imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        elif mode == "visible":
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    except Exception:
        pass


def _spawn_debug_sphere(
    stage,
    prim_path: str,
    center_m: np.ndarray,
    color_rgb: tuple[float, float, float],
    *,
    scale_m: float,
    visibility: str = "inherited",
) -> None:
    """Non-colliding marker. ``center_m`` / ``scale_m`` are meters.

    Leaf markers default to ``inherited`` visibility so toggling the station
    debug root in Isaac shows/hides the whole tree. Do not author per-leaf
    invisible — that blocks the parent toggle.
    """

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
    _set_imageable_visibility(prim, visibility)
    strip_physics_from_prim(stage, prim_path)


def _rotation_from_z_direction(direction: np.ndarray) -> Gf.Quatd:
    axis = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    axis = axis / n
    rotation = Gf.Rotation(
        Gf.Vec3d(0.0, 0.0, 1.0),
        Gf.Vec3d(float(axis[0]), float(axis[1]), float(axis[2])),
    )
    quat = rotation.GetQuat()
    imag = quat.GetImaginary()
    return Gf.Quatd(
        float(quat.GetReal()),
        Gf.Vec3d(float(imag[0]), float(imag[1]), float(imag[2])),
    )


def _spawn_debug_axis_arrow(
    stage,
    prim_path: str,
    origin_m: np.ndarray,
    direction: np.ndarray,
    *,
    length_m: float,
    radius_m: float,
    color_rgb: tuple[float, float, float],
    visibility: str = "inherited",
) -> None:
    """Cylinder+cone arrow along ``direction``; lengths/radii in meters."""

    mpu = meters_per_unit(stage)
    inv = (1.0 / mpu) if mpu > 1e-12 else 1.0
    origin = np.asarray(origin_m, dtype=np.float64).reshape(3) * inv
    axis = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return
    axis = axis / n
    length = float(length_m) * inv
    radius = float(radius_m) * inv
    tip_h = 0.22 * length
    shaft_h = max(1e-4, length - tip_h)
    tip_pos = origin + length * axis
    shaft_center = origin + 0.5 * shaft_h * axis
    tip_center = tip_pos - 0.5 * tip_h * axis
    orient = _rotation_from_z_direction(axis)

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))
    UsdGeom.Xform.Define(stage, Sdf.Path(prim_path))

    shaft_path = f"{prim_path}/Shaft"
    cone_path = f"{prim_path}/Tip"
    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(shaft_path))
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(shaft_h))
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(shaft_center[0]), float(shaft_center[1]), float(shaft_center[2]))
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
    strip_physics_from_prim(stage, shaft_path)

    cone = UsdGeom.Cone.Define(stage, Sdf.Path(cone_path))
    cone.CreateRadiusAttr(float(1.8 * radius))
    cone.CreateHeightAttr(float(tip_h))
    cone.CreateAxisAttr("Z")
    cone.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])
    xform = UsdGeom.Xformable(cone.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(tip_center[0]), float(tip_center[1]), float(tip_center[2]))
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
    strip_physics_from_prim(stage, cone_path)

    for path in (prim_path, shaft_path, cone_path):
        _set_imageable_visibility(stage.GetPrimAtPath(path), visibility)


def spawn_maneuver_orientation_markers(
    stage,
    spec: cfg.StationSpec,
    waypoints: list[dict],
    *,
    visible: bool | None = None,
    branch_name: str = "Maneuver",
    tip_scale_m: float | None = None,
    touch_station_visibility: bool = True,
) -> None:
    """Tip spheres + insertion-axis arrows for each maneuver/insert waypoint.

    Prim tree: ``{debug_marker_root}/{branch_name}/Wp00..``. Leaves inherit
    visibility from the station debug root (toggle that root in Isaac).
    Set ``touch_station_visibility=False`` when adding a sibling branch (e.g.
    Insert) so an existing station-root visibility toggle is preserved.
    """

    show = cfg.DEBUG_MARKER_VISIBLE_DEFAULT if visible is None else bool(visible)
    station_root = spec.debug_marker_root
    station_prim = stage.GetPrimAtPath(station_root)
    if not station_prim.IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path(station_root))
        station_prim = stage.GetPrimAtPath(station_root)
        # New root: apply default visibility once.
        _set_imageable_visibility(
            station_prim, "visible" if show else "invisible"
        )
    root = f"{station_root}/{branch_name}"
    if stage.GetPrimAtPath(root).IsValid():
        stage.RemovePrim(Sdf.Path(root))
    UsdGeom.Xform.Define(stage, Sdf.Path(root))
    if tip_scale_m is None:
        tip_scale = float(cfg.PORT_MANEUVER_MARKER_SCALE_M) * 0.55
    else:
        tip_scale = float(tip_scale_m)
    axis_len = float(getattr(cfg, "PORT_MANEUVER_AXIS_LENGTH_M", 0.08))
    axis_rad = float(getattr(cfg, "PORT_MANEUVER_AXIS_RADIUS_M", 0.003))
    if str(branch_name).lower() == "insert":
        tip_scale = float(
            tip_scale_m
            if tip_scale_m is not None
            else getattr(cfg, "INSERT_PATH_MARKER_SCALE_M", tip_scale * 0.5)
        )
        axis_len *= 0.5
        axis_rad *= 0.5
    n = len(waypoints)
    for i, wp in enumerate(waypoints):
        # Prefer crystal mating center for insert path viz (matches jack height).
        center = wp.get("mating_center")
        if center is None:
            center = wp.get("tip")
        tip = np.asarray(center, dtype=np.float64).reshape(3)
        axis = np.asarray(
            wp.get("insertion_axis", (0.0, 0.0, -1.0)), dtype=np.float64
        ).reshape(3)
        t = float(wp.get("t", float(i) / max(1, n - 1)))
        color = (
            float(0.25 + 0.75 * t),
            float(0.85 - 0.55 * t),
            float(0.95 - 0.35 * t),
        )
        wp_root = f"{root}/Wp{i:02d}"
        UsdGeom.Xform.Define(stage, Sdf.Path(wp_root))
        _set_imageable_visibility(stage.GetPrimAtPath(wp_root), "inherited")
        _spawn_debug_sphere(
            stage,
            f"{wp_root}/Tip",
            tip,
            color,
            scale_m=tip_scale,
            visibility="inherited",
        )
        _spawn_debug_axis_arrow(
            stage,
            f"{wp_root}/InsertionAxis",
            tip,
            axis,
            length_m=axis_len,
            radius_m=axis_rad,
            color_rgb=(1.0, 0.85, 0.1),
            visibility="inherited",
        )
    _set_imageable_visibility(stage.GetPrimAtPath(root), "inherited")
    if touch_station_visibility:
        _set_imageable_visibility(
            stage.GetPrimAtPath(station_root), "visible" if show else "invisible"
        )
    print(
        f"[SCENE] {branch_name} orientation markers under {root} "
        f"({n} waypoints; toggle {station_root} in Isaac)"
    )


def crystal_mating_plane_diagonal_m(crystal) -> float:
    """Hypotenuse of the crystal mating-plane AABB (world Y×Z extents)."""

    corners = np.asarray(crystal.mating_corners, dtype=np.float64).reshape(-1, 3)
    if corners.shape[0] < 2:
        return 0.01
    dy = float(np.max(corners[:, 1]) - np.min(corners[:, 1]))
    dz = float(np.max(corners[:, 2]) - np.min(corners[:, 2]))
    diag = float(np.hypot(dy, dz))
    return diag if diag > 1e-6 else 0.01


def spawn_insert_step_marker(
    stage,
    spec: cfg.StationSpec,
    *,
    tip_cmd_m: np.ndarray,
    crystal,
    insert_dir: np.ndarray | None = None,
    visible: bool | None = None,
) -> None:
    """Single next-target sphere under ``…/AlignInsert`` (replaces prior marker).

    Diameter = hypotenuse of the crystal mating-plane corners (Y×Z AABB).
    """

    show = (
        bool(getattr(cfg, "ALIGN_INSERT_DEBUG_MARKERS_VISIBLE", False))
        if visible is None
        else bool(visible)
    )
    station_root = spec.debug_marker_root
    if not stage.GetPrimAtPath(station_root).IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path(station_root))
    root = f"{station_root}/AlignInsert"
    if stage.GetPrimAtPath(root).IsValid():
        stage.RemovePrim(Sdf.Path(root))
    UsdGeom.Xform.Define(stage, Sdf.Path(root))

    diameter = crystal_mating_plane_diagonal_m(crystal)
    radius = 0.5 * diameter
    tip = np.asarray(tip_cmd_m, dtype=np.float64).reshape(3)
    _spawn_debug_sphere(
        stage,
        f"{root}/TipCmd",
        tip,
        (1.0, 0.15, 0.7),
        scale_m=radius,
        visibility="inherited",
    )
    if insert_dir is not None:
        axis_len = float(getattr(cfg, "ALIGN_INSERT_AXIS_LENGTH_M", 0.03))
        axis_rad = float(getattr(cfg, "ALIGN_INSERT_AXIS_RADIUS_M", 0.00125))
        _spawn_debug_axis_arrow(
            stage,
            f"{root}/InsertDir",
            tip,
            insert_dir,
            length_m=axis_len,
            radius_m=axis_rad,
            color_rgb=(0.95, 0.95, 0.95),
            visibility="inherited",
        )
    _set_imageable_visibility(stage.GetPrimAtPath(root), "inherited")
    _set_imageable_visibility(
        stage.GetPrimAtPath(station_root), "visible" if show else "invisible"
    )


def spawn_align_insert_feature_markers(
    stage,
    spec: cfg.StationSpec,
    *,
    crystal,
    port,
    tip_measured_m: np.ndarray | None = None,
    tip_cmd_m: np.ndarray | None = None,
    hand_cmd_m: np.ndarray | None = None,
    insert_dir: np.ndarray | None = None,
    residual=None,
    visible: bool | None = None,
) -> None:
    """Compat wrapper: single TipCmd marker sized to crystal mating hypotenuse."""

    del tip_measured_m, hand_cmd_m, residual, port
    if tip_cmd_m is None:
        return
    spawn_insert_step_marker(
        stage,
        spec,
        tip_cmd_m=tip_cmd_m,
        crystal=crystal,
        insert_dir=insert_dir,
        visible=visible,
    )


def spawn_station_debug_markers(
    stage,
    spec: cfg.StationSpec,
    *,
    cable_m: np.ndarray | None = None,
    tip_hover_m: np.ndarray | None = None,
    hand_hover_m: np.ndarray | None = None,
    tip_grasp_m: np.ndarray | None = None,
    hand_grasp_m: np.ndarray | None = None,
    tip_lift_m: np.ndarray | None = None,
    tip_yaw_m: np.ndarray | None = None,
    tip_offset_m: np.ndarray | None = None,
    crystal_offset_m: np.ndarray | None = None,
    live_crystal_m: np.ndarray | None = None,
    visible: bool | None = None,
    touch_root_visibility: bool = True,
) -> None:
    """Spheres for targets. Default-hidden; toggle the station root in Isaac.

    Leaf markers inherit visibility from ``spec.debug_marker_root``.
    ``CrystalOffset`` is the commanded crystal mating standoff (port +X/+Z);
    ``TipOffset`` is the fingertip IK target; ``LiveCrystal`` is the measured
    mating center at plan time (often lower than CrystalOffset before rebase).
    """

    show = cfg.DEBUG_MARKER_VISIBLE_DEFAULT if visible is None else bool(visible)
    root = spec.debug_marker_root
    root_prim = stage.GetPrimAtPath(root)
    if not root_prim.IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path(root))
        root_prim = stage.GetPrimAtPath(root)
        _set_imageable_visibility(root_prim, "visible" if show else "invisible")
    scale = float(cfg.OBSERVE_DEBUG_MARKER_SCALE_M)
    via_scale = float(cfg.PORT_MANEUVER_MARKER_SCALE_M)
    markers = (
        ("Cable", cable_m, (0.2, 0.8, 1.0), scale),
        ("TipHover", tip_hover_m, (0.2, 1.0, 0.3), scale),
        ("HandHover", hand_hover_m, (1.0, 0.85, 0.1), scale),
        ("TipGrasp", tip_grasp_m, (1.0, 0.35, 0.1), scale),
        ("HandGrasp", hand_grasp_m, (0.95, 0.45, 0.85), scale),
        ("TipLift", tip_lift_m, (0.4, 0.7, 1.0), scale),
        ("TipYaw", tip_yaw_m, (0.55, 0.85, 0.2), via_scale),
        ("TipOffset", tip_offset_m, (1.0, 0.2, 0.6), scale),
        ("CrystalOffset", crystal_offset_m, (0.15, 1.0, 0.55), scale),
        ("LiveCrystal", live_crystal_m, (1.0, 0.55, 0.1), scale),
    )
    spawned: list[str] = []
    for name, center, color, marker_scale in markers:
        if center is None:
            continue
        path = f"{root}/{name}"
        _spawn_debug_sphere(
            stage,
            path,
            np.asarray(center, dtype=np.float64).reshape(3),
            color,
            scale_m=float(marker_scale),
            visibility="inherited",
        )
        spawned.append(name)
    if touch_root_visibility:
        _set_imageable_visibility(root_prim, "visible" if show else "invisible")
    if spawned:
        print(
            f"[SCENE] Debug markers under {root} "
            f"(root_visible={show if touch_root_visibility else 'unchanged'}; "
            f"toggle in Isaac): {', '.join(spawned)}"
        )

def resolve_grasp_part_path(stage, spec: cfg.StationSpec) -> str:
    part = stage.GetPrimAtPath(spec.grasp_part_path)
    if part and part.IsValid():
        return spec.grasp_part_path
    found = find_descendant(stage, spec.path45, cfg.GRASP_PART_NAME)
    if found:
        return found
    raise RuntimeError(f"Missing grasp part under {spec.path45}")


def resolve_gripper_prim_path(stage, robot_prim_path: str) -> str:
    """Robotiq asset root (for friction / joint discovery). Not for ParallelGripper.

    ``ParallelGripper`` subclasses ``SingleRigidPrim`` and must target a real
    rigid link (``wrist_3_link`` / ``base_link``). After nested-root strip,
    ``Robotiq_2F_85`` is an Xform with no RigidBodyAPI — using it as
    ``end_effector_prim_path`` crashes ``world.reset()`` with
    ``Failed to find rigid body``.
    """

    candidates = (
        f"{robot_prim_path}/Gripper/Robotiq_2F_85",
        f"{robot_prim_path}/ee_link/Robotiq_2F_85",
        f"{robot_prim_path}/Robotiq_2F_85",
    )
    for path in candidates:
        gripper = stage.GetPrimAtPath(path)
        if gripper and gripper.IsValid():
            return str(gripper.GetPath())
    found = find_descendant(stage, robot_prim_path, "Robotiq_2F_85")
    if found:
        return found
    raise RuntimeError(f"Missing Robotiq gripper under {robot_prim_path}")


def resolve_end_effector_path(stage, robot_prim_path: str) -> str:
    """TCP link for SingleManipulator / Lula tracking: prefer ``wrist_3_link``.

    Matches ur10e_1x when ``tool0`` is absent from the USD. Avoids Robotiq
    ``base_link``, which is not an articulation root on this stage.
    """

    for name in ("wrist_3_link", "ee_link", "tool0"):
        path = f"{robot_prim_path}/{name}"
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return path
    # Last resort: Robotiq base_link (has RigidBody after mass repair).
    base = find_descendant(stage, robot_prim_path, "base_link")
    if base and "robotiq" in base.lower():
        return base
    return resolve_gripper_prim_path(stage, robot_prim_path)


def _attach_manipulator(world, spec: cfg.StationSpec, ee_path: str, gripper_path: str | None = None):
    # ur10e_1x: ParallelGripper EE = wrist/tool frame (a real rigid link), not
    # the Robotiq Xform. Gripper joint names are resolved via the articulation.
    _ = gripper_path
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path,
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0.0]),
        joint_closed_positions=np.array([cfg.ROBOTIQ_CLOSED_RAD]),
        action_deltas=None,
        use_mimic_joints=True,
    )
    # Isaac ``remove_object`` on a missing name returns None from get_object and
    # logs ``No attribute prim present under the key: <scene_name>``.
    if world.scene.object_exists(spec.scene_name):
        try:
            # Registry only — never DeletePrims the DataHall robot USD reference.
            world.scene.remove_object(spec.scene_name, registry_only=True)
        except Exception as exc:
            print(
                f"[SCENE] remove_object({spec.scene_name!r}) skipped: {exc}"
            )
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


def _robot_base_pose_from_usd(stage, robot_prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """World pose of the robot prim from USD (no PhysX view required)."""

    prim = stage.GetPrimAtPath(robot_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing robot prim for base pose: {robot_prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    matrix = cache.GetLocalToWorldTransform(prim)
    translate = matrix.ExtractTranslation()
    quat = matrix.ExtractRotation().GetQuat()
    pos = np.array(
        [float(translate[0]), float(translate[1]), float(translate[2])],
        dtype=np.float64,
    )
    ori = np.array(
        [
            float(quat.GetReal()),
            float(quat.GetImaginary()[0]),
            float(quat.GetImaginary()[1]),
            float(quat.GetImaginary()[2]),
        ],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(ori))
    if n > 1e-12:
        ori = ori / n
    return pos, ori


def _make_motion_controller(stage, robot, spec: cfg.StationSpec, lula_config: dict):
    kinematics = LulaKinematicsSolver(**lula_config)
    trajectory = LulaTaskSpaceTrajectoryGenerator(**lula_config)
    # Same policy as ur10e_1x: prefer tool0, else wrist_3_link when USD lacks tool0.
    ee_frame = cfg.UR5E_EE_FRAME
    tool0_path = find_descendant(stage, spec.robot_prim_path, cfg.UR5E_EE_FRAME)
    if not tool0_path:
        ee_frame = cfg.UR5E_EE_FRAME_FALLBACK
        print(
            f"[SCENE {spec.station_id}] tool0 missing in USD; "
            f"Lula ee_frame -> {ee_frame} (ur10e_1x fallback)"
        )
    print(
        f"[SCENE {spec.station_id}] Lula end-effector frame: {ee_frame}"
        + (f" ({tool0_path})" if tool0_path else "")
    )
    articulation_kinematics = ArticulationKinematicsSolver(robot, kinematics, ee_frame)
    # Prefer live articulation pose; fall back to USD if the PhysX view was
    # invalidated by a reset / articulation USD rewrite.
    try:
        base_position, base_orientation = robot.get_world_pose()
    except Exception as exc:
        print(
            f"[SCENE {spec.station_id}] robot.get_world_pose failed ({exc}); "
            "using USD Xform for Lula base"
        )
        base_position, base_orientation = _robot_base_pose_from_usd(
            stage, spec.robot_prim_path
        )
    kinematics.set_robot_base_pose(base_position, base_orientation)
    return Ur5eSixArmMotionController(
        name=f"{spec.scene_name}_controller",
        robot_articulation=robot,
        task_traj_gen=trajectory,
        art_kinematics=articulation_kinematics,
        gripper=robot.gripper,
        tool_offset=float(cfg.TOOL_OFFSET_M),
        physics_dt=float(cfg.PHYSICS_DT),
        position_tolerance=float(cfg.HOVER_IK_POS_TOLERANCE_M),
        orientation_tolerance=float(cfg.HOVER_IK_ORI_TOLERANCE_RAD),
        ee_frame=ee_frame,
        debug=True,
        meters_per_unit=meters_per_unit(stage),
    )


def build_scene(
    simulation_app, *, usd_path: Path | None = None, stations=None
) -> SceneBundle:
    """Open the UR5e DataHall; wire + home only selected stations.

    Unselected UR5es stay composed but invisible with PhysX articulations
    **disabled** (sensors stripped) so GPU MimicJoint solve only sees the BT arm.
    ActionGraphs are deleted on every arm so OmniGraph cannot tick idle
    ``ArticulationController`` nodes after play/reset.
    """

    stage, world = open_datahall_stage(
        simulation_app, usd_path or cfg.DATAHALL_6R_UR5E_USD
    )
    selected = tuple(stations) if stations is not None else cfg.STATIONS
    # Nested Robotiq ArticulationRoot breaks the arm root_joint; strip selected only.
    for spec in selected:
        strip_nested_gripper_articulation_roots(stage, robot_prim_path=spec.robot_prim_path)
    # Re-assert mimic repair + masses after any stage compose that preceded World.
    for spec in selected:
        repair_robotiq_mimic_joints(stage, spec.robot_prim_path)
        ensure_robotiq_link_masses(stage, spec.robot_prim_path)
    remove_all_ur5e_ros_graphs(stage, selected)
    freeze_inactive_ur5e_robots(stage, selected)
    freeze_inactive_network_cables(stage, selected)
    for spec in selected:
        robot_prim = stage.GetPrimAtPath(spec.robot_prim_path)
        if not robot_prim or not robot_prim.IsValid():
            raise RuntimeError(f"Missing robot prim {spec.robot_prim_path}")
        cable = stage.GetPrimAtPath(spec.cable_root_path)
        if not cable or not cable.IsValid():
            raise RuntimeError(f"Missing cable prim {spec.cable_root_path}")

    remove_all_ur5e_ros_graphs(stage, selected)
    freeze_inactive_ur5e_robots(stage, selected)
    freeze_inactive_network_cables(stage, selected)
    if bool(cfg.ENABLE_DATAHALL_STATIC_COLLISIONS):
        enable_datahall_static_collisions(stage, selected=selected)
    for spec in selected:
        enable_crystal_head_physics(stage, spec.path45, spec.path39)
        try:
            configure_cable_deformable_for_stage(stage, spec)
        except Exception as exc:
            print(f"[SCENE] cable deformable skip {spec.station_id}: {exc}")
        apply_grasp_friction_materials(
            stage, spec.robot_prim_path, spec.path45, spec.path39
        )
        apply_bezel_slide_friction_materials(stage, spec.path39)
    apply_worktable_frictionless_materials(stage)
    world.reset()
    from isaacsim.core.simulation_manager import SimulationManager

    _enable_gpu_dynamics(stage, SimulationManager, world=world)
    # Reset can rebuild PhysX from composed USD opinions — re-strip/repair
    # *before* attaching SingleManipulator so the physics view stays valid.
    # Skip mount-joint rewrite: physics already started.
    for spec in selected:
        strip_nested_gripper_articulation_roots(
            stage, robot_prim_path=spec.robot_prim_path, repair_mount=False
        )
        repair_robotiq_mimic_joints(stage, spec.robot_prim_path)
        ensure_robotiq_link_masses(stage, spec.robot_prim_path)
    # Reset reloads authored ROS graphs from USD — strip again.
    remove_all_ur5e_ros_graphs(stage, selected)
    freeze_inactive_ur5e_robots(stage, selected)
    freeze_inactive_network_cables(stage, selected)
    robots: dict[str, Any] = {}
    ee_paths: dict[str, str] = {}
    for spec in selected:
        ee_paths[spec.station_id] = resolve_end_effector_path(stage, spec.robot_prim_path)
        gripper_path = resolve_gripper_prim_path(stage, spec.robot_prim_path)
        robot = _attach_manipulator(
            world, spec, ee_paths[spec.station_id], gripper_path
        )
        robots[spec.station_id] = robot
        configure_robot_physics(stage, spec.robot_prim_path)
        apply_ur5e_drive_parameters(stage, spec.robot_prim_path)
        apply_robotiq_drive_parameters(stage, spec.robot_prim_path)
        apply_grasp_friction_materials(
            stage, spec.robot_prim_path, spec.path45, spec.path39
        )
        apply_bezel_slide_friction_materials(stage, spec.path39)
    apply_worktable_frictionless_materials(stage)
    # Final reset builds the articulation view for attached manipulators.
    # Do NOT strip/repair mimic or articulation roots after this — that
    # invalidates the PhysX tensor view (get_world_pose / getRootTransforms).
    world.reset()
    from isaacsim.core.simulation_manager import SimulationManager

    _enable_gpu_dynamics(stage, SimulationManager, world=world)
    remove_all_ur5e_ros_graphs(stage, selected)
    freeze_inactive_ur5e_robots(stage, selected)
    freeze_inactive_network_cables(stage, selected)
    lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
        cfg.UR5E_LULA_NAME
    )
    if lula_config is None:
        raise RuntimeError(f"No Lula config for {cfg.UR5E_LULA_NAME!r}")

    bundles: list[StationBundle] = []
    for spec in selected:
        robot = robots[spec.station_id]
        # Articulation / drive USD was already applied before the final reset.
        # Do not RemoveAPI/rewrite joints here — that invalidates the PhysX view.
        apply_grasp_friction_materials(
            stage, spec.robot_prim_path, spec.path45, spec.path39
        )
        controller = _make_motion_controller(stage, robot, spec, dict(lula_config))
        home_arm = compute_home_arm_above_work_table(stage, controller, spec)
        apply_ur5e_home_pose(robot, arm_positions=home_arm, apply_live=True)
        bundles.append(
            StationBundle(
                spec=spec,
                robot=robot,
                motion_controller=controller,
                end_effector_path=ee_paths[spec.station_id],
                cable_root_path=spec.cable_root_path,
                home_arm=home_arm,
            )
        )
    apply_worktable_frictionless_materials(stage)
    remove_all_ur5e_ros_graphs(stage, selected)
    freeze_inactive_ur5e_robots(stage, selected)
    freeze_inactive_network_cables(stage, selected)
    print(
        f"[SCENE] BT stations={len(bundles)}; "
        f"inactive UR5es hidden (no sim)"
    )
    return SceneBundle(
        world=world,
        stage=stage,
        stations=bundles,
        idle_stations=[],
    )
