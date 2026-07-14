from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf
import trimesh
import os

USD_DIR = "/home/aayush/Desktop/bmw_animation_usds"
MESH_DIR = "/home/aayush/Desktop/proxy_meshes"
OUTPUT_USD = "/home/aayush/Desktop/bmw_grille_full_animation_with_proxy.usd"
bullet_times = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

stage = Usd.Stage.CreateNew(OUTPUT_USD)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(len(bullet_times) - 1)
stage.SetFramesPerSecond(6)

world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)

# --- Physics scene + light, once for the whole stage ---
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, -1, 0))
physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
light.CreateIntensityAttr().Set(1500.0)
light.CreateAngleAttr().Set(1.0)
UsdGeom.Xformable(light.GetPrim()).AddRotateXOp().Set(-45.0)

num_frames = len(bullet_times)
frames_with_splat = 0
frames_with_mesh = 0

for i, t in enumerate(bullet_times):
    usd_path = f"{USD_DIR}/frame_{t:04d}.usd"
    if not os.path.exists(usd_path):
        print(f"Skipping missing splat: {usd_path}")
        continue

    prim_path = f"/World/Frame_{i:04d}"
    prim = stage.OverridePrim(prim_path)
    prim.GetReferences().AddReference(usd_path, "/gaussians_0")
    frames_with_splat += 1

    # Orientation fix — applied to the splat parent, inherited by the mesh child below
    UsdGeom.Xformable(prim).AddRotateXOp().Set(180.0)

    # Visibility — flipbook logic, splat-side
    imageable = UsdGeom.Imageable(prim)
    vis_attr = imageable.GetVisibilityAttr()
    for frame in range(num_frames):
        value = UsdGeom.Tokens.inherited if frame == i else UsdGeom.Tokens.invisible
        vis_attr.Set(value, Usd.TimeCode(frame))

    # --- Per-frame collision mesh, as a child of this frame's splat prim ---
    mesh_path = f"{MESH_DIR}/proxy_mesh_{t:04d}.obj"
    if os.path.exists(mesh_path):
        tri_mesh = trimesh.load(mesh_path)
        mesh_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/CollisionProxy")
        mesh_prim.GetPointsAttr().Set([tuple(v) for v in tri_mesh.vertices])
        mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(tri_mesh.faces))
        mesh_prim.GetFaceVertexIndicesAttr().Set([int(idx) for tri in tri_mesh.faces for idx in tri])
        mesh_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
        # Only THIS frame's collider should be active while this frame is showing —
        # otherwise all 21 colliders overlap simultaneously and physics breaks
        enabled_attr = collision_api.CreateCollisionEnabledAttr()
        for frame in range(num_frames):
            enabled_attr.Set(frame == i, Usd.TimeCode(frame))

        proxy_rel = prim.CreateRelationship("proxy")
        proxy_rel.AddTarget(mesh_prim.GetPath())
        frames_with_mesh += 1
    else:
        print(f"  Warning: no mesh found for frame {t} — splat-only, no collision this frame")

# --- Test cube — dropped above frame 60's mesh bounding box as a reference point ---
ref_mesh_path = f"{MESH_DIR}/proxy_mesh_0060.obj"
if os.path.exists(ref_mesh_path):
    ref_mesh = trimesh.load(ref_mesh_path)
    bbox_min, bbox_max = ref_mesh.bounds
    drop_x = (bbox_min[0] + bbox_max[0]) / 2
    drop_z = (bbox_min[2] + bbox_max[2]) / 2
    drop_y = bbox_max[1] + 1.0

    cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
    cube.CreateSizeAttr().Set(0.2)
    UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(drop_x, drop_y, drop_z))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(1.0)
else:
    print("Warning: no reference mesh (frame 60) found — skipping test cube placement")

stage.GetRootLayer().Save()
print(f"Saved to {OUTPUT_USD}")
print(f"Frames with splat: {frames_with_splat}/{num_frames}")
print(f"Frames with collision mesh: {frames_with_mesh}/{num_frames}")
