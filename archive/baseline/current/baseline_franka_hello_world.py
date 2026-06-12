from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget
import omni.kit.commands
import omni.usd

from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, Sdf, Usd
from isaacsim.core.prims import Articulation

import numpy as np

import carb

USD_PATH = r"/home/advaith/Downloads/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
# PORT_PRIM_PATH = (
#     "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/"
#     "SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
# )

PORT_PRIM_PATH = "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
# PORT_PRIM_PATH_2 = "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_02"


# /World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_02/QSFP_DD_Connector_A_02

num_quads = 4
num_pairs = 4
num_conn_a = 2


PORT_BASE_PRIM_PATH = (
    "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base"
    "/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01"
)

PORT_PRIM_PATH_LIST = []

for q in range(1, num_quads+1):
    for p in range(1, num_pairs+1):
        for a in range(1, num_conn_a+1):
            suffix = f"/Connector_Quad_{q:02d}/Connector_Pair_{p:02d}/QSFP_DD_Connector_A_{a:02d}"
            PORT_PRIM_PATH_LIST.append(PORT_BASE_PRIM_PATH + suffix)


# PORT_SUFFIX = (
#     "/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
# )



# PORT_PRIM_PATH_LIST = [
#     PORT_PRIM_PATH,
#     PORT_PRIM_PATH_2
# ]



# duration = 5.0
# time_elapsed = 0.0
def move_object(start_pos, end_pos, step_size, prim_path="/World/TargetCube", time_elapsed=0.0, duration=5.0):
    # global time_elapsed
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if time_elapsed < duration:
        time_elapsed += step_size
        fraction = min(time_elapsed / duration, 1.0)
        current_pos = start_pos + (end_pos - start_pos) * fraction
        cpgf = Gf.Vec3d(current_pos[0], current_pos[1], current_pos[2])
        # Set new position
        xform = UsdGeom.Xformable(prim)
        # xform.AddTranslateOp().Set(current_pos)
        # print("times", time_elapsed, duration)
        # print("\tstep_size", step_size)
        # print("\tsetting current pos", current_pos, cpgf)
        # print("\tpositions", start_pos, end_pos)
        prim.GetAttribute("xformOp:translate").Set(cpgf)




def set_prim_visibility_attribute(prim_path: str, value: str):
    """Set the prim visibility attribute at prim_path to value

    Args:
        prim_path (str, required): The path of the prim to modify
        value (str, required): The value of the visibility attribute
    """
    # You can reference attributes using the path syntax by appending the
    # attribute name with a leading `.`
    prop_path = f"{prim_path}.visibility"
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path(prop_path), value=value, prev=None
    )


def hide_prim(prim_path: str):
    """Hide a prim

    Args:
        prim_path (str, required): The prim path of the prim to hide
    """
    set_prim_visibility_attribute(prim_path, "invisible")


# from isaacsim.robot.manipulators.examples.franka import KinematicsSolver

import typing

def get_world_transform_xform(prim: Usd.Prim) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    """
    Get the local transformation of a prim using omni.usd.get_world_transform_matrix().
    See https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd/omni.usd.get_world_transform_matrix.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    return translation, rotation, scale



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
        # print(omni.usd.get_world_transform_matrix(
        #     omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        # ))


        port_prim = omni.usd.get_context().get_stage().GetPrimAtPath(PORT_PRIM_PATH)
        # print("Port prim coming up")
        # print(port_prim)
        # print(port_prim is None)
        port_world_position = get_world_transform_xform(port_prim)[0]
        # print(port_world_position) # (-7.750999993085862, -109.62536725783971, 178.21793722812936)
        # offset = np.array([1.75, 0, 2])
        self.offset = np.array([0, 0, 2])
        # np.array([-6, -110, 180])
        # Initialize a predefined task for the robot (e.g., following a target)
        my_task = FollowTarget(
            name="follow_target_task",
            franka_prim_path=prim_path,  # Path to the robot's prim in the scene
            franka_robot_name="fancy_franka",  # Name for the robot instance
            target_name="target",  # Name of the target object the robot should follow
            # target_prim_path="/World/TargetCube",
            target_position=np.array(port_world_position) + self.offset,
        )
        # Add the task to the simulation world
        world.add_task(my_task)
        self._currPort = 0

        return

    async def setup_post_load(self):
        # Set up post-load configurations, such as controllers
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")

        hide_prim("/World/TargetCube")
        targetCube = omni.usd.get_context().get_stage().GetPrimAtPath("/World/TargetCube")
        assert targetCube.GetAttribute("visibility").Get() == "invisible"


        # Set Position (x, y, z) and Rotation (quaternion)
        new_position = np.array([34.0 - 4.0, -100.0 + 10.0, 140.0 + 10.0])
        new_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        Articulation("/panda").set_world_poses(positions=new_position, orientations=new_orientation)
        # print(Articulation("/panda").get_world_poses())

        # Initialize the RMPFlow controller for the robot
        self._controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=self._franka
        )
        # self._controller = KinematicsSolver(
        #     self._franka
        # )
        self._timeElapsed = 0.0
        self._duration = 5.0

        # Add a physics callback for simulation steps
        if self._currPort + 1 < len(PORT_PRIM_PATH_LIST):
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
        # print("phys step orientation", observations["target"]["orientation"]) # np.array([0, 0, 1, 0])

        # Move target
        port_prim = omni.usd.get_context().get_stage().GetPrimAtPath(PORT_PRIM_PATH_LIST[self._currPort])
        port_world_position = get_world_transform_xform(port_prim)[0]
        start_pos = port_world_position + np.array([0, 0, -2 * (1 if self._currPort % 2 == 0 else -1)])

        port_prim = omni.usd.get_context().get_stage().GetPrimAtPath(PORT_PRIM_PATH_LIST[self._currPort + 1])
        port_world_position = get_world_transform_xform(port_prim)[0]
        end_pos = port_world_position + np.array([0, 0, 2 * (1 if self._currPort % 2 == 0 else -1)])
        # print("start vs end\n", "\t", start_pos, "\n\t", end_pos)

        move_object(start_pos, end_pos, step_size, time_elapsed=self._timeElapsed)
        self._timeElapsed += step_size

        # Compute actions for the robot to follow the target's position and orientation
        actions = self._controller.forward(
            target_end_effector_position=observations["target"]["position"],
            target_end_effector_orientation=observations["target"]["orientation"]
        )

        # Apply the computed actions to the robot
        self._franka.apply_action(actions)
        cube_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/TargetCube")
        cube_world_position = get_world_transform_xform(cube_prim)[0]
        carb.log_info(f"distance away so far {np.array(end_pos - cube_world_position)}")
        # np.linalg.norm(np.array(end_pos - cube_world_position)) < 0.1
        if self._timeElapsed >= self._duration:
            self._currPort += 1
            self._timeElapsed = 0.0
        return

        # actions, succ = self._controller.compute_inverse_kinematics(
        #     target_position=observations["target"]["position"],
        #     target_orientation=observations["target"]["orientation"],
        # )
        # if succ:
        #     self._franka.get_articulation_controller.apply_action(actions)
        # else:
        #     carb.log_warn("IK did not converge to a solution.  No action is being taken.")

        # return

    def world_cleanup(self):
        return
