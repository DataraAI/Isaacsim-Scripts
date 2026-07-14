# build_hand_animation.py — NO SimulationApp, NO isaacsim import
from pxr import Usd, UsdGeom, UsdLux, Gf
import os
import re

USD_DIR    = "/home/aayush/Desktop/hand_animation_usds"
OUTPUT_USD = "/home/aayush/Desktop/hand_animation.usd"

# Auto-discover converted USDs
all_usds = sorted([
    f for f in os.listdir(USD_DIR)
    if re.match(r'^hand_frame_\d{6}\.usd$', f)
])

if not all_usds:
    raise RuntimeError(f"No hand frame USDs found in {USD_DIR} — run convert_hand_frames.py first")

frame_numbers = [int(re.findall(r'\d+', f)[0]) for f in all_usds]
num_frames = len(frame_numbers)
print(f"Found {num_frames} hand frame USDs")
print(f"Frame range: {frame_numbers[0]:06d} → {frame_numbers[-1]:06d}")

stage = Usd.Stage.CreateNew(OUTPUT_USD)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(num_frames - 1)
stage.SetFramesPerSecond(30.0)  # video frames = 30fps, adjust if different

world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)

light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
light.CreateIntensityAttr().Set(1500.0)
light.CreateAngleAttr().Set(1.0)
UsdGeom.Xformable(light.GetPrim()).AddRotateXOp().Set(-45.0)

frames_loaded = 0

for i, frame_num in enumerate(frame_numbers):
    usd_path = f"{USD_DIR}/hand_frame_{frame_num:06d}.usd"

    prim_path = f"/World/HandFrame_{i:06d}"
    prim = stage.OverridePrim(prim_path)
    prim.GetReferences().AddReference(usd_path, "/hands")

    # Efficient visibility — only 3 samples per prim instead of num_frames
    # avoids 2000*2000 = 4M set calls which would be very slow
    vis_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
    vis_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(0))
    vis_attr.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(i))
    vis_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(i + 0.5))

    frames_loaded += 1
    if frames_loaded % 100 == 0:
        print(f"Progress: {frames_loaded}/{num_frames} frames written...")

stage.GetRootLayer().Save()
print(f"\nSaved hand animation to {OUTPUT_USD}")
print(f"Frames loaded: {frames_loaded}/{num_frames}")
