# build_animation.py — NO SimulationApp, NO isaacsim import
from pxr import Usd, UsdGeom, Sdf
import os

USD_DIR = "/home/aayush/Desktop/bmw_animation_usds"
OUTPUT_USD = "/home/aayush/Desktop/bmw_grille_full_animation.usd"
bullet_times = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

stage = Usd.Stage.CreateNew(OUTPUT_USD)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(len(bullet_times) - 1)
stage.SetFramesPerSecond(6)

world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)

num_frames = len(bullet_times)
for i, t in enumerate(bullet_times):
    usd_path = f"{USD_DIR}/frame_{t:04d}.usd"
    if not os.path.exists(usd_path):
        print(f"Skipping missing: {usd_path}")
        continue
    prim_path = f"/World/Frame_{i:04d}"
    prim = stage.OverridePrim(prim_path)
    prim.GetReferences().AddReference(usd_path, "/gaussians_0")
    imageable = UsdGeom.Imageable(prim)
    vis_attr = imageable.GetVisibilityAttr()
    for frame in range(num_frames):
        value = UsdGeom.Tokens.inherited if frame == i else UsdGeom.Tokens.invisible
        vis_attr.Set(value, Usd.TimeCode(frame))

stage.GetRootLayer().Save()
print(f"Saved animated USD with {num_frames} frames to {OUTPUT_USD}")
