"""Load DataHall, work table, UR10e + Robotiq 2F-85, support block, and network cable.

    /home/aayush/isaacsim/python.sh aayush/asset_spawn/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

AAYUSH_DIR = Path(__file__).resolve().parents[1]
if str(AAYUSH_DIR) not in sys.path:
    sys.path.insert(0, str(AAYUSH_DIR))

from asset_spawn.spawn import build_asset_spawn_scene  # noqa: E402

bundle = build_asset_spawn_scene(simulation_app)
print("[SPAWN] Press Play to run physics. Isaac Sim will stay open until you quit.")
simulation_app.update()
while simulation_app.is_running():
    bundle.world.step(render=True)
