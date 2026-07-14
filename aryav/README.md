## Scripts
- `py3dgsPlyToUsd.py` — converts single .ply → .usd (visual Gaussian splat)
- `convert_all_frames.py` — batch converts all 21 .ply frames → individual .usd files
- `build_animation.py` — stitches 21 frame USDs into one animated stage
- `poisson_script.py` — generates collision proxy mesh from .ply via Poisson reconstruction
- `generate_all_proxy_meshes.py` — batch runs poisson_script.py over all 21 frames
- `combine_proxy.py` — combines Gaussian splat + collision mesh into one USD with physics
- `build_animation_with_proxy.py` — full animated stage with per-frame collision proxies
- `convert_hand_frames.py` — converts per-frame hand .obj files → USD
- `build_hand_animation.py` — stitches hand frame USDs into animated stage
- `view_hand_frames.py` — quick single-frame hand mesh viewer
## Notes
- All scripts currently hardcoded to run on 5080  machine
- Paths (`/home/aayush/Desktop/`, `~/isaacsim/python.sh`) has to be changed when using the files somewhere else
