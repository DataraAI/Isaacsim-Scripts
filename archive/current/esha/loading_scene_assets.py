
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
 
import carb
import numpy as np
import omni.usd
 
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path


DATAHALL_USD = ( # path to the datahall usd file
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)

DATAHALL_PRIM_PATH = "/World/DataHall" # path to the datahall prim in the usd file

world = World(stage_units_in_meters=1.0)
add_reference_to_stage(world.scene, DATAHALL_USD, DATAHALL_PRIM_PATH)
world.reset()
carb.log_info(f"Loaded data hall scene from {DATAHALL_USD} at {DATAHALL_PRIM_PATH}")


ASSET_USD = "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin/Assets/Datacenter/Racks/Rack_42_UA/Rack_42_UA.usd"
ASSET_PRIM_PATH = "/World/Racks/Rack_42_UA"

def main() -> None:
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        return
 
    my_world = World(stage_units_in_meters=1.0) #create a new world
 
    add_reference_to_stage(usd_path=DATAHALL_USD, prim_path=DATAHALL_PRIM_PATH) #add the datahall scene to the world
    carb.log_info(f"Loaded data hall scene at {DATAHALL_PRIM_PATH}")
 
    add_reference_to_stage(usd_path=ASSET_USD, prim_path=ASSET_PRIM_PATH) #add the asset to the world
    carb.log_info(f"Injected asset at {ASSET_PRIM_PATH}")
 
    my_world.reset() #reset the world
 
