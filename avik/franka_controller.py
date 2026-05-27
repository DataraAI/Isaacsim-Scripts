import typing
import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper

class FrankaController(BaseController):
    """
    A waypoint controller for precise manipulation tasks.
    Executes a sequence of cartesian, joint, and gripper commands.
    Includes timeouts to prevent RMPFlow from getting stuck.
    """
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        position_tolerance: float = 0.05, 
        joint_tolerance: float = 0.05,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._command_queue = []
        self._current_command_index = 0
        
        self._pos_tolerance = position_tolerance
        self._joint_tolerance = joint_tolerance

    def add_cartesian_waypoint(self, position: np.ndarray, orientation: np.ndarray, max_frames: int = 300) -> None:
        """Moves to a point. max_frames limits how long to wait before forcing the next step."""
        self._command_queue.append({
            "type": "cartesian",
            "pos": position,
            "ori": orientation,
            "max_frames": max_frames,
            "frames_spent": 0
        })

    def add_joint_waypoint(self, joint_positions: np.ndarray, max_frames: int = 300) -> None:
        self._command_queue.append({
            "type": "joint",
            "joints": joint_positions,
            "max_frames": max_frames,
            "frames_spent": 0
        })

    def add_gripper_command(self, action: str, wait_frames: int = 60) -> None:
        self._command_queue.append({
            "type": "gripper",
            "action": action,
            "max_frames": wait_frames,
            "frames_spent": 0
        })

    def forward(
        self,
        current_joint_positions: np.ndarray,
        current_end_effector_position: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if self.is_done():
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        current_cmd = self._command_queue[self._current_command_index]

        # ---------------------------
        # Cartesian Command Logic
        # ---------------------------
        if current_cmd["type"] == "cartesian":
            current_cmd["frames_spent"] += 1
            
            if current_end_effector_position is not None:
                dist = np.linalg.norm(current_cmd["pos"] - current_end_effector_position)
                
                # Advance if we reached the distance tolerance OR if the timer runs out
                if dist < self._pos_tolerance or current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                    print(f"Finished Step {self._current_command_index} (Cartesian). Distance off: {dist:.4f}")
                    self._current_command_index += 1
                    return self.forward(current_joint_positions, current_end_effector_position)

            return self._cspace_controller.forward(
                target_end_effector_position=current_cmd["pos"],
                target_end_effector_orientation=current_cmd["ori"]
            )

        # ---------------------------
        # Gripper Command Logic
        # ---------------------------
        elif current_cmd["type"] == "gripper":
            current_cmd["frames_spent"] += 1
            
            if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                print(f"Finished Step {self._current_command_index} (Gripper).")
                self._current_command_index += 1
                return self.forward(current_joint_positions, current_end_effector_position)

            return self._gripper.forward(action=current_cmd["action"])

        # ---------------------------
        # Joint Command Logic
        # ---------------------------
        elif current_cmd["type"] == "joint":
            current_cmd["frames_spent"] += 1
            
            dist = np.linalg.norm(current_cmd["joints"] - current_joint_positions)
            if dist < self._joint_tolerance or current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                print(f"Finished Step {self._current_command_index} (Joint).")
                self._current_command_index += 1
                return self.forward(current_joint_positions, current_end_effector_position)

            return ArticulationAction(joint_positions=current_cmd["joints"])

        return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

    def reset(self) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._current_command_index = 0
        
        # Reset the spent frames for all commands so it can be re-run
        for cmd in self._command_queue:
            cmd["frames_spent"] = 0

    def is_done(self) -> bool:
        return self._current_command_index >= len(self._command_queue)