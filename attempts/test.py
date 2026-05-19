import asyncio
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

async def example():
    if World.instance():
        World.instance().clear_instance()
    world=World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()

    # add franka articulations

    asset_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    robot1 = add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka_1")
    robot1.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot1.GetVariantSet("Mesh").SetVariantSelection("Quality")
    robot2 = add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka_2")
    robot2.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot2.GetVariantSet("Mesh").SetVariantSelection("Quality")

    # batch process articulations via an Articulation
    frankas_view = Articulation(prim_paths_expr="/World/Franka_[1-2]", name="frankas_view")
    world.scene.add(frankas_view)
    await world.reset_async()
    # set root body poses
    new_positions = np.array([[-1.0, 1.0, 0], [1.0, 1.0, 0]])
    frankas_view.set_world_poses(positions=new_positions)
    # set the joint positions for each articulation
    frankas_view.set_joint_positions(np.array([[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                                                    [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]))
asyncio.ensure_future(example())