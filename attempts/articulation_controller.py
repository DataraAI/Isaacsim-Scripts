from isaacsim import SimulationApp

# Start Isaac Sim
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
import asyncio

async def robot_control_example():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()

    # Load the robot USD file
    usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    prim_path = "/World/envs/env_0/panda"
    add_reference_to_stage(usd_path, prim_path)

    # Create Articulation view for the robot
    robot_view = Articulation(prim_paths_expr=prim_path, name="franka_panda_view")

    # Create and initialize the articulation controller with the articulation view
    articulation_controller = ArticulationController()
    articulation_controller.initialize(robot_view)

    # Run simulation
    await world.play_async()

    # Get current joint positions
    current_positions = robot_view.get_joint_positions()
    print(f"Current joint positions: {current_positions}")

    # Create target positions
    target_positions = np.array([0.0, -1.5, 0.0, -2.8, 0.0, 2.8, 1.2, 0.04, 0.04])

    # Create and apply articulation action
    action = ArticulationAction(joint_positions=target_positions)
    articulation_controller.apply_action(action)

    await asyncio.sleep(5.0)  # Run for 5 seconds to reach target positions

    # Get current joint positions
    current_positions = robot_view.get_joint_positions()
    print(f"Current joint positions: {current_positions}")

    world.pause()
    #simulation_app.close()

# Run the example
asyncio.ensure_future(robot_control_example())