# view_hand_frame.py — NO SimulationApp, NO isaacsim import
from pxr import Usd, UsdGeom, UsdLux, Gf
import trimesh
import os

# --- Config: change these to whatever frame you want to inspect ---
FRAME_NUM  = 0                                      # frame number to view
INPUT_DIR  = "/home/aayush/Desktop/hand_meshes"
OUTPUT_USD = "/home/aayush/Desktop/hand_frame_test.usd"

left_path  = f"{INPUT_DIR}/{FRAME_NUM:06d}_0.obj"
right_path = f"{INPUT_DIR}/{FRAME_NUM:06d}_1.obj"

stage = Usd.Stage.CreateNew(OUTPUT_USD)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)

# Light so it actually renders
light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
light.CreateIntensityAttr().Set(1500.0)
light.CreateAngleAttr().Set(1.0)
UsdGeom.Xformable(light.GetPrim()).AddRotateXOp().Set(-45.0)

def add_hand_mesh(stage, prim_path, obj_path, label):
    if not os.path.exists(obj_path):
        print(f"WARNING: {label} not found at {obj_path}")
        return
    mesh = trimesh.load(obj_path, force='mesh')
    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
    mesh_prim.GetPointsAttr().Set([tuple(v) for v in mesh.vertices])
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    mesh_prim.GetFaceVertexIndicesAttr().Set([int(i) for face in mesh.faces for i in face])
    mesh.fix_normals()
    mesh_prim.GetNormalsAttr().Set([tuple(n) for n in mesh.vertex_normals])
    print(f"{label}: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

add_hand_mesh(stage, "/World/LeftHand",  left_path,  "Left hand  (_0)")
add_hand_mesh(stage, "/World/RightHand", right_path, "Right hand (_1)")

stage.GetRootLayer().Save()
print(f"\nSaved to {OUTPUT_USD}")
print(f"Open in Isaac Sim with Path Tracing, press F to frame the selection")
