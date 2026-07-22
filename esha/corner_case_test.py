from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

DATAHALL_USD = (
    "/home/advaith/Isaacsim-assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)
DATAHALL_PRIM_PATH = "/World/DataHall"

ASSET_USD = (
    "/home/advaith/Isaacsim-assets/DigitalTwin"
    "/Assets/Datacenter/Racks/Rack_42U_A/Rack_42U_A_01.usd"
)
ASSET_PRIM_PATH = "/World/Racks/Rack_42_UA"


def main() -> None:
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        return

    my_world = World(stage_units_in_meters=1.0)

    add_reference_to_stage(usd_path=DATAHALL_USD, prim_path=DATAHALL_PRIM_PATH)
    carb.log_info(f"Loaded data hall scene at {DATAHALL_PRIM_PATH}")

    add_reference_to_stage(usd_path=ASSET_USD, prim_path=ASSET_PRIM_PATH)
    carb.log_info(f"Injected asset at {ASSET_PRIM_PATH}")

    my_world.reset()

    while simulation_app.is_running():
        my_world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

