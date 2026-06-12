"""Shared collision authoring helpers for datacenter assets."""

from __future__ import annotations

import carb
import omni.usd
from omni.physx.scripts import utils as physx_utils
from pxr import Gf, PhysxSchema, Tf, Usd, UsdGeom, UsdPhysics


def is_collision_geometry(prim: Usd.Prim) -> bool:
    if prim.IsA(UsdGeom.Mesh):
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        return points is not None and len(points) > 0
    return (
        prim.IsA(UsdGeom.Cube)
        or prim.IsA(UsdGeom.Sphere)
        or prim.IsA(UsdGeom.Cylinder)
        or prim.IsA(UsdGeom.Capsule)
        or prim.IsA(UsdGeom.Cone)
    )


def can_author_physics(prim: Usd.Prim) -> bool:
    return prim.IsValid() and not prim.IsInstanceProxy()


def apply_mesh_collider(prim: Usd.Prim, approximation_shape: str = "none") -> bool:
    if not can_author_physics(prim):
        return False
    if prim.GetAttribute("omni:no_collision"):
        return False
    if not is_collision_geometry(prim) and not prim.IsInstanceable():
        return False
    try:
        if prim.IsInstanceable() and not prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            return True
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.CreateApproximationAttr().Set(approximation_shape)
            mesh_approx_api = physx_utils.MESH_APPROXIMATIONS.get(approximation_shape)
            if mesh_approx_api is not None:
                mesh_approx_api.Apply(prim)
        return True
    except Tf.ErrorException as exc:
        carb.log_warn(f"Skipping collider on {prim.GetPath()}: {exc}")
        return False


def enable_static_collisions(root_path: str, approximation_shape: str = "none") -> int:
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        carb.log_warn(f"Collision setup skipped: {root_path} not found")
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        if prim.GetMetadata("hide_in_stage_window"):
            continue
        if apply_mesh_collider(prim, approximation_shape):
            count += 1
    carb.log_info(
        f"Static collisions enabled on {count} prims under {root_path} ({approximation_shape})"
    )
    return count


def enable_articulation_collisions(root_path: str) -> int:
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        carb.log_warn(f"Collision setup skipped: {root_path} not found")
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
        count += 1
    carb.log_info(f"Articulation collisions enabled on {count} prims under {root_path}")
    return count


def apply_datahall_scale(datahall_prim_path: str, scale_factor: float) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(datahall_prim_path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    scale_op = xform.GetXformOp(UsdGeom.XformOp.TypeScale)
    if scale_op:
        current = scale_op.Get() or Gf.Vec3d(1.0, 1.0, 1.0)
        scale_op.Set(
            Gf.Vec3d(
                current[0] * scale_factor,
                current[1] * scale_factor,
                current[2] * scale_factor,
            )
        )
    else:
        scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(scale_factor, scale_factor, scale_factor))
