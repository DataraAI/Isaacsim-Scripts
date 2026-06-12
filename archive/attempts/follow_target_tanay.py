from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np

from isaacsim.core.api import World
import isaacsim.core.api.tasks as tasks
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)

# Keep this if it works on your install.
# If not, swap to:
# from isaacsim.robot.manipulators.examples.franka import Franka
from omni.isaac.franka import Franka


class KinematicsSolver:
    def __init__(self, robot, end_effector_frame_name="right_gripper"):
        self._robot = robot
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**cfg)
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            robot,
            self._kinematics_solver,
            end_effector_frame_name,
        )

    def compute_inverse_kinematics(self, target_position, target_orientation=None):
        robot_base_translation, robot_base_orientation = self._robot.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(
            robot_base_translation, robot_base_orientation
        )
        return self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation,
        )


class FollowTarget(tasks.BaseTask):
    def __init__(
        self,
        name="follow_target_task",
        target_prim_path=None,
        target_name=None,
        target_position=None,
        target_orientation=None,
        offset=None,
        franka_prim_path=None,
        franka_robot_name=None,
    ):
        super().__init__(name=name, offset=offset)

        self._target_position = (
            np.array([0.5, 0.0, 0.5], dtype=np.float64)
            if target_position is None
            else np.array(target_position, dtype=np.float64)
        )
        self._target_orientation = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if target_orientation is None
            else np.array(target_orientation, dtype=np.float64)
        )

        self._target_prim_path = target_prim_path
        self._target_name = target_name
        self._franka_prim_path = franka_prim_path
        self._franka_robot_name = franka_robot_name

        self._target = None
        self._franka = None

    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        if self._target_prim_path is None:
            self._target_prim_path = find_unique_string_name(
                initial_name="/World/TargetCube",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )

        if self._target_name is None:
            self._target_name = find_unique_string_name(
                initial_name="target",
                is_unique_fn=lambda x: scene.get_object(x) is None,
            )

        if self._franka_prim_path is None:
            self._franka_prim_path = find_unique_string_name(
                initial_name="/World/Franka",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )

        if self._franka_robot_name is None:
            self._franka_robot_name = find_unique_string_name(
                initial_name="franka",
                is_unique_fn=lambda x: scene.get_object(x) is None,
            )

        self._target = scene.add(
            VisualCuboid(
                prim_path=self._target_prim_path,
                name=self._target_name,
                position=self._target_position,
                orientation=self._target_orientation,
                size=0.04,
                color=np.array([1.0, 0.0, 0.0]),
            )
        )

        self._franka = scene.add(
            Franka(
                prim_path=self._franka_prim_path,
                name=self._franka_robot_name,
            )
        )

        self._task_objects[self._target.name] = self._target
        self._task_objects[self._franka.name] = self._franka
        self._move_task_objects_to_their_frame()

    def get_observations(self):
        target_position, target_orientation = self._target.get_world_pose()
        return {
            self._target.name: {
                "position": target_position,
                "orientation": target_orientation,
            }
        }

    def get_params(self):
        return {
            "target_name": {"value": self._target.name, "modifiable": False},
            "robot_name": {"value": self._franka.name, "modifiable": False},
        }

    def post_reset(self):
        if hasattr(self._franka, "gripper"):
            self._franka.gripper.set_joint_positions(
                self._franka.gripper.joint_opened_positions
            )


my_world = World(stage_units_in_meters=1.0)
my_task = FollowTarget(name="follow_target_task")
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]

my_franka = my_world.scene.get_object(franka_name)
my_controller = KinematicsSolver(my_franka)
articulation_controller = my_franka.get_articulation_controller()

# Start from the script instead of relying on manual Play
my_world.play()

while simulation_app.is_running():
    my_world.step(render=True)

    observations = my_world.get_observations()
    actions, succ = my_controller.compute_inverse_kinematics(
        target_position=observations[target_name]["position"],
        target_orientation=observations[target_name]["orientation"],
    )

    if succ:
        articulation_controller.apply_action(actions)
    else:
        carb.log_warn("IK did not converge to a solution. No action is being taken.")

simulation_app.close()