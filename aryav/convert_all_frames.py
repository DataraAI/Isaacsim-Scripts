import subprocess
import os

PLY_BASE = "/home/aayush/Desktop/bmw_grille_21frames/outputs/demo/lyra_dynamic/static_view_indices_fixed_5_0_1_2_3_4/lyra_dynamic_demo_generated"
OUTPUT_DIR = "/home/aayush/Desktop/bmw_animation_usds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

bullet_times = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

for t in bullet_times:
    ply_path = f"{PLY_BASE}/{t}/gaussians_orig/gaussians_0.ply"
    usd_path = f"{OUTPUT_DIR}/frame_{t:04d}.usd"
    
    if not os.path.exists(ply_path):
        print(f"Skipping missing: {ply_path}")
        continue
    
    print(f"Converting bullet time {t}...")
    subprocess.run([
        "python3", "py3dgsPlyToUsd.py",
        "--input", ply_path,
        "--output", usd_path
    ])

print("Done converting all available bullet times.")
