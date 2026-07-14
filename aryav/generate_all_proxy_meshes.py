# generate_all_proxy_meshes.py
import os
from poisson_script import gaussian_ply_to_mesh_poisson

PLY_BASE = "/home/aayush/Desktop/bmw_grille_21frames/outputs/demo/lyra_dynamic/static_view_indices_fixed_5_0_1_2_3_4/lyra_dynamic_demo_generated"
OUTPUT_DIR = "/home/aayush/Desktop/proxy_meshes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

bullet_times = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]

for t in bullet_times:
    ply_path = f"{PLY_BASE}/{t}/gaussians_orig/gaussians_0.ply"
    if not os.path.exists(ply_path):
        print(f"Skipping missing ply for frame {t}")
        continue
    out_path = f"{OUTPUT_DIR}/proxy_mesh_{t:04d}.obj"
    print(f"Frame {t}:")
    gaussian_ply_to_mesh_poisson(ply_path, out_path)

print("Done.")
