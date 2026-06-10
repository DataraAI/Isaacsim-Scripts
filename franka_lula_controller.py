import typing

import carb
import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices, rot_matrices_to_quats
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    ArticulationTrajectory,
    LulaTaskSpaceTrajectoryGenerator,
)


class FrankaLulaController(BaseController):
    """Waypoint controller using Lula task-space trajectories and IK fallback.

    Cartesian goals describe the grasped block center in world coordinates.
    Straight segments use dense task-space samples so Lula connects them with
    linear end-effector motion, then time-optimizes in joint space.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        task_traj_gen: LulaTaskSpaceTrajectoryGenerator,
        art_kinematics: ArticulationKinematicsSolver,
        gripper: Gripper,
        tool_offset: float = 0.1,
        physics_dt: float = 1.0 / 60.0,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        ee_frame: str = "panda_hand",
    ) -> None:
        super().__init__(name=name)
        self._robot = robot_articulation
        self._task_traj_gen = task_traj_gen
        self._art_kinematics = art_kinematics
        self._gripper = gripper
        self._ee_frame = ee_frame

        self._tool_offset = tool_offset
        self._physics_dt = physics_dt
        self._pos_tolerance = position_tolerance
        self._ori_tolerance = orientation_tolerance

        self._command_queue: typing.List[dict] = []
        self._current_command_index = 0
        self._action_sequence: typing.List[ArticulationAction] = []
        self._action_index = 0
        self._segment_ready = False
        self._segment_goal_hand: typing.Optional[np.ndarray] = None

        self._linear_start: typing.Optional[np.ndarray] = None
        self._linear_dir = np.zeros(3, dtype=np.float64)
        self._linear_length = 0.0
        self._linear_progress = 0.0
        self._linear_ik_warned = False

    def add_cartesian_waypoint(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        max_frames: int = 400,
        pos_tolerance: typing.Optional[float] = None,
        linear: bool = False,
        linear_step: float = 0.001,
    ) -> None:
        """Queue a Cartesian goal.

        linear=True guarantees straight-line end-effector motion: the target is
        stepped along the start->goal line by linear_step meters per frame with
        per-frame IK, instead of relying on Lula trajectory generation (whose
        failure falls back to a single IK jump with no path guarantee).
        """
        self._command_queue.append({
            "type": "cartesian",
            "pos": np.asarray(position, dtype=np.float64),
            "ori": np.asarray(orientation, dtype=np.float64),
            "max_frames": max_frames,
            "frames_spent": 0,
            "pos_tolerance": pos_tolerance,
            "linear": linear,
            "linear_step": linear_step,
        })

    def add_gripper_command(self, action: str, wait_frames: int = 60) -> None:
        self._command_queue.append({
            "type": "gripper",
            "action": action,
            "max_frames": wait_frames,
            "frames_spent": 0,
        })

    def _hold_action(self, n_dof: int) -> ArticulationAction:
        return ArticulationAction(joint_positions=[None] * n_dof)

    def _hand_from_block(self, block_pos: np.ndarray, ori_wxyz: np.ndarray) -> np.ndarray:
        rot = quats_to_rot_matrices(np.asarray(ori_wxyz, dtype=np.float64).reshape(1, 4))[0]
        return np.asarray(block_pos, dtype=np.float64) - rot @ np.array(
            [0.0, 0.0, self._tool_offset], dtype=np.float64
        )

    def _current_hand_pose(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        hand_pos, hand_rot = self._art_kinematics.compute_end_effector_pose()
        hand_quat = rot_matrices_to_quats(hand_rot)
        if hand_quat.ndim > 1:
            hand_quat = hand_quat[0]
        return np.asarray(hand_pos, dtype=np.float64), np.asarray(hand_quat, dtype=np.float64)

    def _build_hand_waypoints(self, current_cmd: dict) -> typing.Tuple[np.ndarray, np.ndarray]:
        hand_start_pos, hand_start_quat = self._current_hand_pose()
        hand_goal_pos = self._hand_from_block(current_cmd["pos"], current_cmd["ori"])
        # Shift both endpoints by tool offset so Lula's straight task-space segment
        # stays above the block; segment completion still checks single-offset goal.
        block_points = np.stack([hand_start_pos, hand_goal_pos], axis=0)
        orientations = np.stack([hand_start_quat, current_cmd["ori"]], axis=0)
        hand_positions = np.array(
            [self._hand_from_block(p, orientations[i]) for i, p in enumerate(block_points)],
            dtype=np.float64,
        )
        return hand_positions, orientations

    def _build_trajectory_for_current_command(self, current_cmd: dict) -> None:
        hand_positions, orientations = self._build_hand_waypoints(current_cmd)
        self._segment_goal_hand = self._hand_from_block(current_cmd["pos"], current_cmd["ori"])

        trajectory = self._task_traj_gen.compute_task_space_trajectory_from_points(
            hand_positions, orientations, self._ee_frame
        )

        if trajectory is None:
            carb.log_warn(
                "Lula task-space trajectory failed; falling back to single-point IK."
            )
            ik_action, success = self._art_kinematics.compute_inverse_kinematics(
                target_position=self._segment_goal_hand,
                target_orientation=current_cmd["ori"],
                position_tolerance=current_cmd.get("pos_tolerance") or self._pos_tolerance,
                orientation_tolerance=self._ori_tolerance,
            )
            if not success:
                carb.log_warn("IK fallback did not report convergence.")
            self._action_sequence = [ik_action]
            return

        art_traj = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)
        self._action_sequence = art_traj.get_action_sequence()

    def _init_linear_segment(self, current_cmd: dict) -> None:
        start, _ = self._current_hand_pose()
        goal = self._hand_from_block(current_cmd["pos"], current_cmd["ori"])
        self._segment_goal_hand = goal
        delta = goal - start
        length = float(np.linalg.norm(delta))
        self._linear_start = start
        self._linear_dir = delta / length if length > 1e-9 else np.zeros(3, dtype=np.float64)
        self._linear_length = length
        self._linear_progress = 0.0
        self._linear_ik_warned = False

    def _linear_ik_action(self, current_cmd: dict, n_dof: int) -> ArticulationAction:
        next_progress = min(
            self._linear_progress + current_cmd["linear_step"], self._linear_length
        )
        target = self._linear_start + self._linear_dir * next_progress
        action, success = self._art_kinematics.compute_inverse_kinematics(
            target_position=target,
            target_orientation=current_cmd["ori"],
        )
        if not success:
            # Hold and retry the same target next frame so the crawl never
            # outruns IK; commanded poses stay on the line by construction.
            if not self._linear_ik_warned:
                carb.log_warn(
                    f"Linear segment IK failed at {np.round(target, 4)}; holding."
                )
                self._linear_ik_warned = True
            return self._hold_action(n_dof)
        self._linear_progress = next_progress
        return action

    def _segment_goal_reached(self, current_cmd: dict) -> bool:
        tolerance = current_cmd.get("pos_tolerance") or self._pos_tolerance
        hand_pos, _ = self._current_hand_pose()
        return float(np.linalg.norm(self._segment_goal_hand - hand_pos)) < tolerance

    def _clear_segment_playback(self) -> None:
        self._action_sequence = []
        self._action_index = 0
        self._segment_ready = False
        self._segment_goal_hand = None
        self._linear_start = None
        self._linear_dir = np.zeros(3, dtype=np.float64)
        self._linear_length = 0.0
        self._linear_progress = 0.0
        self._linear_ik_warned = False

    def forward(self, current_joint_positions: np.ndarray) -> ArticulationAction:
        n_dof = current_joint_positions.shape[0]

        while not self.is_done():
            current_cmd = self._command_queue[self._current_command_index]

            if current_cmd["type"] == "gripper":
                current_cmd["frames_spent"] += 1
                if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                    self._current_command_index += 1
                    self._clear_segment_playback()
                    continue
                return self._gripper.forward(action=current_cmd["action"])

            if current_cmd["type"] == "cartesian":
                if not self._segment_ready:
                    if current_cmd.get("linear"):
                        self._init_linear_segment(current_cmd)
                    else:
                        self._build_trajectory_for_current_command(current_cmd)
                    self._segment_ready = True
                    self._action_index = 0
                    current_cmd["frames_spent"] = 0

                if current_cmd.get("linear"):
                    current_cmd["frames_spent"] += 1
                    if (
                        self._linear_progress >= self._linear_length
                        and self._segment_goal_reached(current_cmd)
                    ) or current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                        self._current_command_index += 1
                        self._clear_segment_playback()
                        continue
                    return self._linear_ik_action(current_cmd, n_dof)

                if self._action_index < len(self._action_sequence):
                    action = self._action_sequence[self._action_index]
                    self._action_index += 1
                    current_cmd["frames_spent"] += 1
                    return action

                current_cmd["frames_spent"] += 1
                if self._segment_goal_reached(current_cmd) or (
                    current_cmd["frames_spent"] >= current_cmd["max_frames"]
                ):
                    self._current_command_index += 1
                    self._clear_segment_playback()
                    continue

                return self._hold_action(n_dof)

            carb.log_warn(f"Unknown command type: {current_cmd.get('type')}")
            return self._hold_action(n_dof)

        return self._hold_action(n_dof)

    def reset(self) -> None:
        super().reset()
        self._current_command_index = 0
        self._clear_segment_playback()
        for cmd in self._command_queue:
            cmd["frames_spent"] = 0

    def is_done(self) -> bool:
        return self._current_command_index >= len(self._command_queue)
