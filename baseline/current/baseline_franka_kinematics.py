import numpy as np
import os
import carb

from isaacsim import SimulationApp

# Start Isaac Sim
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation

# Isaac Sim 5.1 renamed Articulation.handles_initialized to _is_initialized; motion_generation still expects the old name.
if not hasattr(Articulation, "handles_initialized"):
    Articulation.handles_initialized = property(lambda self: self._is_initialized)


class _ArticulationSqueezeBatchWrapper:
    """Wraps an Articulation so get_joint_positions() returns (num_dofs,) instead of (1, num_dofs).
    Isaac Sim 5.1 returns batch-first; motion_generation expects a single-robot vector, causing IndexError otherwise.
    """

    def __init__(self, articulation):
        self._articulation = articulation

    def get_joint_positions(self):
        pos = self._articulation.get_joint_positions()
        return np.asarray(pos).squeeze()

    def __getattr__(self, name):
        return getattr(self._articulation, name)


from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader

import omni.usd
from pxr import UsdGeom, Gf

# Small cube created at the port pose for the arm to point at (same pose as PORT_PRIM_PATH).
PORT_TARGET_CUBE_PRIM_PATH = "/World/port_target_cube"


USD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"

ROBOT_PRIM_PATH = "/World/Franka"

PORT_PRIM_PATH = "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"

EE_FRAME = "panda_hand"

# /World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01/QSFP_DD_Connector_01/Springs/con002228_4_15/con002228_4


def get_target_pose(prim_path, stage):
    prim = stage.GetPrimAtPath(prim_path)
    pose = omni.usd.utils.get_world_transform_matrix(prim)
    q = pose.ExtractRotation().GetQuaternion()
    pos = pose.ExtractTranslation()
    rot = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
    return np.array(pos), np.array(rot)


def create_cube_at_pose(stage, cube_prim_path, position, orientation_quat, size_meters=1.0):
    """Create a small cube at the given world position and orientation (quat w,x,y,z)."""
    xform_prim = stage.DefinePrim(cube_prim_path, "Xform")
    cube_geom = UsdGeom.Cube.Define(stage, cube_prim_path + "/cube_geom")
    cube_geom.CreateSizeAttr(size_meters)
    # Set world transform: rotation from quat (w,x,y,z) then translation
    quat = Gf.Quatd(orientation_quat[0], orientation_quat[1], orientation_quat[2], orientation_quat[3])
    rot = Gf.Rotation(quat)
    mat = Gf.Matrix4d(rot, Gf.Vec3d(position[0], position[1], position[2]))
    xformable = UsdGeom.Xformable(xform_prim)
    op = xformable.AddTransformOp()
    op.Set(mat)


class FrankaKinematicsExample():
    def __init__(self, robot_prim_path, port_prim_path, ee_frame, usd_path):
        self._robot_prim_path = robot_prim_path
        self._port_prim_path = port_prim_path
        self._ee_frame = ee_frame
        self._usd_path = usd_path
        self._world = World(stage_units_in_meters=1.0)
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None

    def load_example_assets(self):
        # Add the Franka and target to the stage

        add_reference_to_stage(self._usd_path, "/World")
        self._stage = omni.usd.get_context().get_stage()

        self._articulation = Articulation(self._robot_prim_path)

        # Register the articulation with the World so it gets initialized (PhysX handles, etc.).
        # Without this, ArticulationKinematicsSolver fails with e.g. 'handles_initialized' AttributeError.
        self._world.scene.add(self._articulation)

        #self._target = XFormPrim(self._port_prim_path)
        self._target = self._stage.GetPrimAtPath(self._port_prim_path)

        # Create a small cube at the same pose as the port; arm will target this cube.
        port_pos, port_rot = get_target_pose(self._port_prim_path, self._stage)
        create_cube_at_pose(
            self._stage,
            PORT_TARGET_CUBE_PRIM_PATH,
            np.asarray(port_pos).squeeze(),
            np.asarray(port_rot).squeeze(),
            size_meters=1.0,
        )

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target

    def setup(self):
        # Load a URDF and Lula Robot Description File for this robot:
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path = kinematics_config_dir + "/franka/lula_franka_gen.urdf"
        )

        # Kinematics for supported robots can be loaded with a simpler equivalent
        # print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        # kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        # self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())

        end_effector_name = self._ee_frame
        # Use wrapper so get_joint_positions() returns (num_dofs,) for motion_generation (Isaac Sim 5.1 returns (1, num_dofs)).
        # self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation, self._kinematics_solver, end_effector_name)
        articulation_for_ik = _ArticulationSqueezeBatchWrapper(self._articulation)
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(articulation_for_ik, self._kinematics_solver, end_effector_name)


    def update(self):
        # Use the cube (same pose as port) as IK target so we can see where the arm is pointing.
        target_position, target_orientation = get_target_pose(PORT_TARGET_CUBE_PRIM_PATH, self._stage)
        target_position = np.asarray(target_position).squeeze()
        target_orientation = np.asarray(target_orientation).squeeze()

        # Track any movements of the robot base. Lula expects 1D (3,) and (4,) not batched (1,3)/(1,4).
        robot_base_translation, robot_base_orientation = self._articulation.get_world_poses()
        robot_base_translation = np.asarray(robot_base_translation).squeeze()
        robot_base_orientation = np.asarray(robot_base_orientation).squeeze()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(target_position, target_orientation)

        if success:
            # Articulation.apply_action() in 5.1 expects batch-first; IK returns (num_dofs,) for single robot.
            to_apply = action
            if hasattr(action, "joint_positions"):
                jp = np.asarray(action.joint_positions)
                if jp.ndim == 1:
                    jp = jp[np.newaxis, :]
                to_apply = type(action)(joint_positions=jp)
            self._articulation.apply_action(to_apply)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken")

        # Unused Forward Kinematics:
        # ee_position,ee_rot_mat = articulation_kinematics_solver.compute_end_effector_pose()

    def reset(self):
        # Kinematics is stateless
        pass

# Mimic Loaded Scenario Template: setup() must run only after articulation is initialized (post_load phase).
# See: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/utilities/extension_templates_tutorial.html
franka_kinematics_example = FrankaKinematicsExample(ROBOT_PRIM_PATH, PORT_PRIM_PATH, EE_FRAME, USD_PATH)
franka_kinematics_example.load_example_assets()
franka_kinematics_example._world.reset()
# Run a few physics steps so the articulation gets fully initialized (handles, etc.) before creating the kinematics solver.
for _ in range(10):
    franka_kinematics_example._world.step(render=True)
# Keep simulation stopped so you can press Play when ready.
franka_kinematics_example._world.stop()
franka_kinematics_example.setup()

reset_needed = False
while simulation_app.is_running():
    franka_kinematics_example._world.step(render=True)
    if franka_kinematics_example._world.is_stopped() and not reset_needed:
        reset_needed = True
    if franka_kinematics_example._world.is_playing():
        if reset_needed:
            franka_kinematics_example._world.reset()
            reset_needed = False
        franka_kinematics_example.update()

simulation_app.close()
