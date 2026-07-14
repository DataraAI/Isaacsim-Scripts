from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf
import trimesh

GAUSSIAN_USD = "/home/aayush/Desktop/bmw_animation_usds/frame_0060.usd"
MESH_OBJ = "/home/aayush/Desktop/bmw_proxy_mesh_60_poisson.obj"
OUTPUT_USD = "/home/aayush/Desktop/bmw_with_collision_proxy.usd"

stage = Usd.Stage.CreateNew(OUTPUT_USD)
world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)

physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, -1, 0))
physics_scene.CreateGravityMagnitudeAttr().Set(9.81)  # meters-scale, matches your bbox numbers

light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
light.CreateIntensityAttr().Set(1500.0)
light.CreateAngleAttr().Set(1.0)
UsdGeom.Xformable(light.GetPrim()).AddRotateXOp().Set(-45.0)

# --- Gaussian splat — visual layer, WITH the upside-down fix applied ---
gaussian_prim = stage.OverridePrim("/World/GaussianSplat")
gaussian_prim.GetReferences().AddReference(GAUSSIAN_USD, "/gaussians_0")

# THE FIX: flip 180° on X — corrects the Y-up/Y-down convention mismatch
gaussian_xform = UsdGeom.Xformable(gaussian_prim)
gaussian_xform.AddRotateXOp().Set(180.0)

# --- Mesh proxy — child, NO separate rotation, inherits the fix above ---
tri_mesh = trimesh.load(MESH_OBJ)
mesh_prim = UsdGeom.Mesh.Define(stage, "/World/GaussianSplat/CollisionProxy")
mesh_prim.GetPointsAttr().Set([tuple(v) for v in tri_mesh.vertices])
mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(tri_mesh.faces))
mesh_prim.GetFaceVertexIndicesAttr().Set([int(i) for tri in tri_mesh.faces for i in tri])
mesh_prim.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())

proxy_rel = gaussian_prim.CreateRelationship("proxy")
proxy_rel.AddTarget(mesh_prim.GetPath())

# --- Test cube ---
bbox_min = tri_mesh.bounds[0]
bbox_max = tri_mesh.bounds[1]
drop_x = (bbox_min[0] + bbox_max[0]) / 2
drop_z = (bbox_min[2] + bbox_max[2]) / 2
drop_y = bbox_max[1] + 1.0

cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
cube.CreateSizeAttr().Set(0.2)
UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(drop_x, drop_y, drop_z))
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(1.0)

stage.GetRootLayer().Save()
print(f"Saved to {OUTPUT_USD}")
