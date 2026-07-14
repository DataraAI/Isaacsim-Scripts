# convert_hand_frames.py — NO SimulationApp, NO isaacysim import
from pxr import Usd, UsdGeom
import trimesh
import os
import re

INPUT_DIR  = "/home/aayush/Desktop/hand_meshes"
OUTPUT_DIR = "/home/aayush/Desktop/hand_animation_usds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Naming convention: {frame:06d}_0.obj = left, {frame:06d}_1.obj = right
def left_obj_path(frame_num):
    return f"{INPUT_DIR}/{frame_num:06d}_0.obj"

def right_obj_path(frame_num):
    return f"{INPUT_DIR}/{frame_num:06d}_1.obj"

def discover_frame_numbers(input_dir):
    """
    Scan directory for files matching NNNNNN_0.obj or NNNNNN_1.obj
    and return sorted unique frame numbers.
    """
    frame_nums = set()
    pattern = re.compile(r'^(\d{6})_[01]\.obj$')
    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            frame_nums.add(int(match.group(1)))
    return sorted(frame_nums)

def write_mesh_to_prim(stage, prim_path, obj_path):
    mesh = trimesh.load(obj_path, force='mesh')
    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
    mesh_prim.GetPointsAttr().Set([tuple(v) for v in mesh.vertices])
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * len(mesh.faces))
    mesh_prim.GetFaceVertexIndicesAttr().Set([int(i) for face in mesh.faces for i in face])
    mesh.fix_normals()
    mesh_prim.GetNormalsAttr().Set([tuple(n) for n in mesh.vertex_normals])
    return mesh_prim

frame_numbers = discover_frame_numbers(INPUT_DIR)
print(f"Discovered {len(frame_numbers)} frames")
print(f"Range: {frame_numbers[0]:06d} → {frame_numbers[-1]:06d}")

frames_converted = 0
frames_skipped = 0

for frame_num in frame_numbers:
    left_path  = left_obj_path(frame_num)
    right_path = right_obj_path(frame_num)

    left_exists  = os.path.exists(left_path)
    right_exists = os.path.exists(right_path)

    if not left_exists and not right_exists:
        frames_skipped += 1
        continue

    out_path = f"{OUTPUT_DIR}/hand_frame_{frame_num:06d}.usd"

    # Resume support — skip already-converted frames
    if os.path.exists(out_path):
        frames_converted += 1
        continue

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = stage.DefinePrim("/hands", "Xform")
    stage.SetDefaultPrim(root)

    if left_exists:
        write_mesh_to_prim(stage, "/hands/left", left_path)
    else:
        print(f"  Frame {frame_num:06d}: WARNING — left hand (_0) missing")

    if right_exists:
        write_mesh_to_prim(stage, "/hands/right", right_path)
    else:
        print(f"  Frame {frame_num:06d}: WARNING — right hand (_1) missing")

    stage.GetRootLayer().Save()
    frames_converted += 1

    if frames_converted % 100 == 0:
        print(f"Progress: {frames_converted}/{len(frame_numbers)} frames converted...")

print(f"\nDone.")
print(f"Converted: {frames_converted}")
print(f"Skipped (no meshes found): {frames_skipped}")
