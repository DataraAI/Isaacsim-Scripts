from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget
import omni.kit.commands
import omni.usd

from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf
from isaacsim.core.prims import Articulation

import numpy as np

USD_PATH = r"/home/advaith/Downloads/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        # Get the world object to set up the simulation environment
        world = self.get_world()

        # Add a default ground plane to the scene for the robot to interact with
        # world.scene.add_default_ground_plane()
        add_reference_to_stage(USD_PATH, "/World/Datacenter")

        # Acquire the URDF extension interface for parsing and importing URDF files
        urdf_interface = _urdf.acquire_urdf_interface()

        # Configure the settings for importing the URDF file
        import_config = _urdf.ImportConfig()
        import_config.convex_decomp = False  # Disable convex decomposition for simplicity
        import_config.fix_base = True       # Fix the base of the robot to the ground
        import_config.make_default_prim = True  # Make the robot the default prim in the scene
        import_config.self_collision = False  # Disable self-collision for performance
        import_config.distance_scale = 57.0     # Set distance scale for the robot
        import_config.density = 0.0          # Set density to 0 (use default values)

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = extension_path + "/data/urdf/robots/franka_description/robots"
        file_name = "panda_arm_hand.urdf"

        # Parse the robot's URDF file to generate a robot model
        result, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config
        )

        # Update the joint drive parameters for better stiffness and damping
        for joint in robot_model.joints:
            robot_model.joints[joint].drive.strength = 1047.19751  # High stiffness value
            robot_model.joints[joint].drive.damping = 52.35988    # Moderate damping value

        # Import the robot onto the current stage and retrieve its prim path
        result, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_robot=robot_model,
            import_config=import_config,
        )

        # Optionally, import the robot onto a new stage and reference it in the current stage
        # (Useful for assets with textures to ensure textures load correctly)
        # dest_path = "/path/to/dest.usd"
        # result, prim_path = omni.kit.commands.execute(
        #     "URDFParseAndImportFile",
        #     urdf_path="{}/{}".format(root_path, file_name),
        #     import_config=import_config,
        #     dest_path=dest_path
        # )
        # prim_path = omni.usd.get_stage_next_free_path(
        #     self.world.scene.stage, str(current_stage.GetDefaultPrim().GetPath()) + prim_path, False
        # )
        # robot_prim = self.world.scene.stage.OverridePrim(prim_path)
        # robot_prim.GetReferences().AddReference(dest_path)
        print(omni.usd.get_world_transform_matrix(
            omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        ))


        # Initialize a predefined task for the robot (e.g., following a target)
        my_task = FollowTarget(
            name="follow_target_task",
            franka_prim_path=prim_path,  # Path to the robot's prim in the scene
            franka_robot_name="fancy_franka",  # Name for the robot instance
            target_name="target",  # Name of the target object the robot should follow
            target_position=np.array([-6, -110, 180]),
        )

        # Add the task to the simulation world
        world.add_task(my_task)
        return

    async def setup_post_load(self):
        # Set up post-load configurations, such as controllers
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        # Set Position (x, y, z) and Rotation (quaternion)
        new_position = np.array([34.0, -100.0, 140.0])
        new_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        Articulation("/panda").set_world_poses(positions=new_position, orientations=new_orientation)
        print(Articulation("/panda").get_world_poses())

        # Initialize the RMPFlow controller for the robot
        self._controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=self._franka
        )

        # Add a physics callback for simulation steps
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        # Reset the controller to its initial state
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # Perform a simulation step and compute actions for the robot
        world = self.get_world()
        observations = world.get_observations()

        # Compute actions for the robot to follow the target's position and orientation
        actions = self._controller.forward(
            target_end_effector_position=observations["target"]["position"],
            target_end_effector_orientation=observations["target"]["orientation"]
        )

        # Apply the computed actions to the robot
        self._franka.apply_action(actions)
        return

    def world_cleanup(self):
        return
