import numpy as np
import os
import carb

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation, XFormPrim

# Isaac Sim 5.1 compatibility patch for motion_generation
if not hasattr(Articulation, "handles_initialized"):
    Articulation.handles_initialized = property(lambda self: self._is_initialized)


class _ArticulationSqueezeBatchWrapper:
    """Wraps an Articulation so get_joint_positions() returns (num_dofs,) instead of (1, num_dofs)."""

    def __init__(self, articulation):
        self._articulation = articulation

    def get_joint_positions(self):
        pos = self._articulation.get_joint_positions()
        return np.asarray(pos).squeeze()

    def __getattr__(self, name):
        return getattr(self._articulation, name)


from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
import omni.usd

USD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"

ROBOT_PRIM_PATH = "/World/Franka"
PORT_PRIM_PATH = "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"

# Keep these matched: if EE_FRAME is panda_hand, EE_PRIM_PATH should be /World/Franka/panda_hand
EE_FRAME = "panda_hand"
EE_PRIM_PATH = "/World/Franka/panda_hand"

# Safer targeting parameters
APPROACH_DISTANCE = 0.15   # meters back from the port
CONTACT_DISTANCE = 0.03    # meters in front of the port; don't try to go exactly inside it
APPROACH_Z_LIFT = 0.03     # meters above the contact line
POSITION_TOL = 0.015       # meters
STABLE_FRAMES_REQUIRED = 15

PRINT_EVERY_N_FRAMES = 30


class FrankaKinematicsExample:
    def __init__(self, robot_prim_path, port_prim_path, ee_frame, ee_prim_path, usd_path):
        self._robot_prim_path = robot_prim_path
        self._port_prim_path = port_prim_path
        self._ee_frame = ee_frame
        self._ee_prim_path = ee_prim_path
        self._usd_path = usd_path

        self._world = World(stage_units_in_meters=1.0)

        self._stage = None
        self._articulation = None
        self._port = None
        self._ee = None

        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._fixed_target_orientation = None

        self._state = "approach"   # approach -> contact -> hold
        self._stable_counter = 0
        self._frame_idx = 0

    def load_example_assets(self):
        add_reference_to_stage(self._usd_path, "/World")
        self._stage = omni.usd.get_context().get_stage()

        # Validate prims exist before wrapping
        if not self._stage.GetPrimAtPath(self._robot_prim_path).IsValid():
            raise RuntimeError(f"Robot prim path is invalid: {self._robot_prim_path}")
        if not self._stage.GetPrimAtPath(self._port_prim_path).IsValid():
            raise RuntimeError(f"Port prim path is invalid: {self._port_prim_path}")
        if not self._stage.GetPrimAtPath(self._ee_prim_path).IsValid():
            raise RuntimeError(f"EE prim path is invalid: {self._ee_prim_path}")

        self._articulation = Articulation(self._robot_prim_path)
        self._world.scene.add(self._articulation)

        self._port = XFormPrim(self._port_prim_path)
        self._ee = XFormPrim(self._ee_prim_path)

        return self._articulation, self._port

    def setup(self):
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=kinematics_config_dir + "/franka/lula_franka_gen.urdf",
        )

        print("Valid frame names:", self._kinematics_solver.get_all_frame_names())

        articulation_for_ik = _ArticulationSqueezeBatchWrapper(self._articulation)
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            articulation_for_ik,
            self._kinematics_solver,
            self._ee_frame,
        )

        # Capture the current hand orientation once and keep it fixed.
        # This avoids the unconstrained wrist spinning you were seeing.
        _, ee_rot = self._ee.get_world_pose()
        self._fixed_target_orientation = np.asarray(ee_rot).squeeze()

    def _get_robot_base_pose(self):
        base_pos, base_rot = self._articulation.get_world_poses()
        return np.asarray(base_pos).squeeze(), np.asarray(base_rot).squeeze()

    def _get_port_pose(self):
        pos, rot = self._port.get_world_pose()
        return np.asarray(pos).squeeze(), np.asarray(rot).squeeze()

    def _get_ee_pose(self):
        pos, rot = self._ee.get_world_pose()
        return np.asarray(pos).squeeze(), np.asarray(rot).squeeze()

    def _compute_targets(self):
        robot_base_pos, _ = self._get_robot_base_pose()
        port_pos, _ = self._get_port_pose()

        direction = port_pos - robot_base_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            norm = 1.0
        direction = direction / norm

        contact_target = port_pos - CONTACT_DISTANCE * direction
        approach_target = port_pos - APPROACH_DISTANCE * direction
        approach_target[2] += APPROACH_Z_LIFT

        return approach_target, contact_target

    def _apply_action_batch_safe(self, action):
        to_apply = action
        if hasattr(action, "joint_positions"):
            jp = np.asarray(action.joint_positions)
            if jp.ndim == 1:
                jp = jp[np.newaxis, :]
            to_apply = type(action)(joint_positions=jp)
        self._articulation.apply_action(to_apply)

    def update(self):
        robot_base_pos, robot_base_rot = self._get_robot_base_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_pos, robot_base_rot)

        ee_pos, ee_rot = self._get_ee_pose()
        port_pos, port_rot = self._get_port_pose()

        approach_target, contact_target = self._compute_targets()
        target_pos = approach_target if self._state == "approach" else contact_target

        # periodic diagnostics
        if self._frame_idx % PRINT_EVERY_N_FRAMES == 0:
            print("\n--- diagnostics ---")
            print("state:        ", self._state)
            print("robot base:   ", robot_base_pos)
            print("ee pos:       ", ee_pos)
            print("port pos:     ", port_pos)
            print("approach tgt: ", approach_target)
            print("contact tgt:  ", contact_target)
            print("ee->target d: ", np.linalg.norm(target_pos - ee_pos))

        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_pos,
            self._fixed_target_orientation,
            position_tolerance=POSITION_TOL,
            orientation_tolerance=0.5,  # looser than exact port alignment
        )

        if success and self._state != "hold":
            self._apply_action_batch_safe(action)
        elif not success and self._state != "hold":
            carb.log_warn("IK did not converge to a solution. No action is being taken.")

        # state transitions
        dist = np.linalg.norm(target_pos - ee_pos)
        if self._state in ("approach", "contact"):
            if dist < POSITION_TOL:
                self._stable_counter += 1
            else:
                self._stable_counter = 0

            if self._state == "approach" and self._stable_counter >= STABLE_FRAMES_REQUIRED:
                print("Reached approach target. Advancing to contact target.")
                self._state = "contact"
                self._stable_counter = 0

            elif self._state == "contact" and self._stable_counter >= STABLE_FRAMES_REQUIRED:
                print("Reached contact target. Holding position.")
                self._state = "hold"
                self._stable_counter = 0

        self._frame_idx += 1

    def reset(self):
        self._state = "approach"
        self._stable_counter = 0
        self._frame_idx = 0


franka_kinematics_example = FrankaKinematicsExample(
    ROBOT_PRIM_PATH,
    PORT_PRIM_PATH,
    EE_FRAME,
    EE_PRIM_PATH,
    USD_PATH,
)

franka_kinematics_example.load_example_assets()
franka_kinematics_example._world.reset()

# let articulation initialize fully
for _ in range(10):
    franka_kinematics_example._world.step(render=True)

franka_kinematics_example.setup()
franka_kinematics_example.reset()

# Start from the script; don't rely on UI Stop/Play
franka_kinematics_example._world.play()

while simulation_app.is_running():
    franka_kinematics_example._world.step(render=True)
    franka_kinematics_example.update()

simulation_app.close()