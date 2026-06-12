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
    LulaKinematicsSolver,
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
        lula_kinematics: LulaKinematicsSolver,
        task_traj_gen: LulaTaskSpaceTrajectoryGenerator,
        art_kinematics: ArticulationKinematicsSolver,
        gripper: Gripper,
        tool_offset: float = 0.1,
        physics_dt: float = 1.0 / 60.0,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        cartesian_step: float = 0.001,
        insert_velocity_scale: float = 0.3,
        ee_frame: str = "panda_hand",
    ) -> None:
        BaseController.__init__(self, name=name)
        self._robot = robot_articulation
        self._lula_kinematics = lula_kinematics
        self._task_traj_gen = task_traj_gen
        self._art_kinematics = art_kinematics
        self._gripper = gripper
        self._ee_frame = ee_frame

        self._tool_offset = tool_offset
        self._physics_dt = physics_dt
        self._pos_tolerance = position_tolerance
        self._ori_tolerance = orientation_tolerance
        self._cartesian_step = cartesian_step
        self._insert_velocity_scale = insert_velocity_scale

        self._default_velocity_limits = np.asarray(
            self._task_traj_gen.get_c_space_velocity_limits(), dtype=np.float64
        ).copy()
        self._default_acceleration_limits = np.asarray(
            self._task_traj_gen.get_c_space_acceleration_limits(), dtype=np.float64
        ).copy()

        self._command_queue: typing.List[dict] = []
        self._current_command_index = 0
        self._action_sequence: typing.List[ArticulationAction] = []
        self._action_index = 0
        self._segment_ready = False
        self._traj_debug: typing.Optional[dict] = None

    def add_cartesian_waypoint(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        max_frames: int = 400,
        pos_tolerance: typing.Optional[float] = None,
        linear: bool = False,
        cartesian_step: typing.Optional[float] = None,
        track_block: bool = False,
        hold_yz: bool = False,
        slow_motion: bool = False,
    ) -> None:
        self._command_queue.append({
            "type": "cartesian",
            "pos": np.asarray(position, dtype=np.float64),
            "ori": np.asarray(orientation, dtype=np.float64),
            "max_frames": max_frames,
            "frames_spent": 0,
            "pos_tolerance": pos_tolerance,
            "linear": linear,
            "cartesian_step": cartesian_step,
            "track_block": track_block,
            "hold_yz": hold_yz,
            "slow_motion": slow_motion,
        })

    def add_gripper_command(self, action: str, wait_frames: int = 60) -> None:
        self._command_queue.append({
            "type": "gripper",
            "action": action,
            "max_frames": wait_frames,
            "frames_spent": 0,
        })

    def get_traj_debug_state(self) -> typing.Optional[dict]:
        return self._traj_debug

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

    def _sample_block_positions(
        self,
        start_block: np.ndarray,
        goal_block: np.ndarray,
        linear: bool,
        hold_yz: bool,
        step: float,
    ) -> np.ndarray:
        start_block = np.asarray(start_block, dtype=np.float64)
        goal_block = np.asarray(goal_block, dtype=np.float64)
        if not linear:
            return np.stack([start_block, goal_block], axis=0)

        if hold_yz:
            dist = float(abs(goal_block[0] - start_block[0]))
            n = max(2, int(np.ceil(dist / step)) + 1)
            xs = np.linspace(start_block[0], goal_block[0], n)
            return np.array(
                [[x, start_block[1], start_block[2]] for x in xs], dtype=np.float64
            )

        dist = float(np.linalg.norm(goal_block - start_block))
        n = max(2, int(np.ceil(dist / step)) + 1)
        return np.linspace(start_block, goal_block, n)

    def _build_hand_waypoints(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        goal_block = current_cmd["pos"]
        goal_ori = current_cmd["ori"]
        step = current_cmd.get("cartesian_step") or self._cartesian_step

        if current_cmd["track_block"] and current_tracked_position is not None:
            start_block = np.asarray(current_tracked_position, dtype=np.float64)
        else:
            hand_pos, hand_quat = self._current_hand_pose()
            start_block = (
                hand_pos
                + quats_to_rot_matrices(hand_quat.reshape(1, 4))[0]
                @ np.array([0.0, 0.0, self._tool_offset])
            )

        block_points = self._sample_block_positions(
            start_block,
            goal_block,
            current_cmd["linear"],
            current_cmd.get("hold_yz", False),
            step,
        )

        if current_cmd["linear"]:
            orientations = np.tile(goal_ori.reshape(1, 4), (block_points.shape[0], 1))
        else:
            hand_start_pos, hand_start_quat = self._current_hand_pose()
            hand_goal_pos = self._hand_from_block(goal_block, goal_ori)
            block_points = np.stack([hand_start_pos, hand_goal_pos], axis=0)
            orientations = np.stack([hand_start_quat, goal_ori], axis=0)

        hand_positions = np.array(
            [self._hand_from_block(p, orientations[i]) for i, p in enumerate(block_points)],
            dtype=np.float64,
        )
        return hand_positions, orientations

    def _apply_motion_limits(self, slow: bool) -> None:
        if slow:
            scale = self._insert_velocity_scale
            self._task_traj_gen.set_c_space_velocity_limits(self._default_velocity_limits * scale)
            self._task_traj_gen.set_c_space_acceleration_limits(
                self._default_acceleration_limits * scale
            )
        else:
            self._task_traj_gen.set_c_space_velocity_limits(self._default_velocity_limits)
            self._task_traj_gen.set_c_space_acceleration_limits(self._default_acceleration_limits)

    def _build_trajectory_for_current_command(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> bool:
        hand_positions, orientations = self._build_hand_waypoints(
            current_cmd, current_tracked_position
        )

        self._apply_motion_limits(current_cmd.get("slow_motion", False))
        try:
            trajectory = self._task_traj_gen.compute_task_space_trajectory_from_points(
                hand_positions, orientations, self._ee_frame
            )
        finally:
            self._apply_motion_limits(False)

        if trajectory is None:
            carb.log_warn(
                "Lula task-space trajectory failed; falling back to single-point IK."
            )
            goal_hand = self._hand_from_block(current_cmd["pos"], current_cmd["ori"])
            ik_action, success = self._art_kinematics.compute_inverse_kinematics(
                target_position=goal_hand,
                target_orientation=current_cmd["ori"],
                position_tolerance=current_cmd.get("pos_tolerance") or self._pos_tolerance,
                orientation_tolerance=self._ori_tolerance,
            )
            if not success:
                carb.log_warn("IK fallback did not report convergence.")
            self._action_sequence = [ik_action]
            self._traj_debug = {
                "cmd_index": self._current_command_index,
                "num_actions": 1,
                "duration_s": 0.0,
                "fallback_ik": True,
                "goal_block": current_cmd["pos"].copy(),
            }
            return True

        art_traj = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)
        self._action_sequence = art_traj.get_action_sequence()
        self._traj_debug = {
            "cmd_index": self._current_command_index,
            "num_actions": len(self._action_sequence),
            "duration_s": art_traj.get_trajectory_duration(),
            "fallback_ik": False,
            "goal_block": current_cmd["pos"].copy(),
            "num_waypoints": hand_positions.shape[0],
        }
        return True

    def _segment_goal_reached(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> bool:
        tolerance = current_cmd.get("pos_tolerance") or self._pos_tolerance
        goal = current_cmd["pos"]
        if current_cmd["track_block"] and current_tracked_position is not None:
            tracked = np.asarray(current_tracked_position, dtype=np.float64)
            return float(np.linalg.norm(goal - tracked)) < tolerance

        hand_pos, _ = self._current_hand_pose()
        goal_hand = self._hand_from_block(goal, current_cmd["ori"])
        return float(np.linalg.norm(goal_hand - hand_pos)) < tolerance

    def _clear_segment_playback(self) -> None:
        self._action_sequence = []
        self._action_index = 0
        self._segment_ready = False

    def forward(
        self,
        current_joint_positions: np.ndarray,
        current_tracked_position: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        n_dof = current_joint_positions.shape[0]
        self._traj_debug = None

        if self.is_done():
            return self._hold_action(n_dof)

        current_cmd = self._command_queue[self._current_command_index]

        if current_cmd["type"] == "gripper":
            current_cmd["frames_spent"] += 1
            if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(current_joint_positions, current_tracked_position)
            return self._gripper.forward(action=current_cmd["action"])

        if current_cmd["type"] == "cartesian":
            if not self._segment_ready:
                if not self._build_trajectory_for_current_command(
                    current_cmd, current_tracked_position
                ):
                    return self._hold_action(n_dof)
                self._segment_ready = True
                self._action_index = 0
                current_cmd["frames_spent"] = 0

            if self._action_index < len(self._action_sequence):
                action = self._action_sequence[self._action_index]
                self._action_index += 1
                current_cmd["frames_spent"] += 1
                return action

            current_cmd["frames_spent"] += 1
            if self._segment_goal_reached(current_cmd, current_tracked_position) or (
                current_cmd["frames_spent"] >= current_cmd["max_frames"]
            ):
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(current_joint_positions, current_tracked_position)

            return self._hold_action(n_dof)

        return self._hold_action(n_dof)

    def reset(self) -> None:
        BaseController.reset(self)
        self._current_command_index = 0
        self._clear_segment_playback()
        self._traj_debug = None
        for cmd in self._command_queue:
            cmd["frames_spent"] = 0

    def is_done(self) -> bool:
        return self._current_command_index >= len(self._command_queue)
