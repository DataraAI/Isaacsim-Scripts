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


class FrankaMotionController(BaseController):
    """Small waypoint controller for this insertion demo.

    Supports three motion styles:
    - Lula task-space trajectory for normal Cartesian moves.
    - joint_interp=True for smooth joint-space transit/reorientation.
    - linear=True for per-frame IK straight-line segments.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        task_traj_gen: LulaTaskSpaceTrajectoryGenerator,
        art_kinematics: ArticulationKinematicsSolver,
        gripper: Gripper,
        tool_offset: float = 0.05,
        physics_dt: float = 1.0 / 60.0,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        ee_frame: str = "panda_hand",
        debug: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._robot = robot_articulation
        self._task_traj_gen = task_traj_gen
        self._art_kinematics = art_kinematics
        self._gripper = gripper
        self._tool_offset = float(tool_offset)
        self._physics_dt = float(physics_dt)
        self._pos_tolerance = float(position_tolerance)
        self._ori_tolerance = float(orientation_tolerance)
        self._ee_frame = ee_frame
        self._debug = bool(debug)
        self.clear_queue()

    def clear_queue(self) -> None:
        self._command_queue: typing.List[dict] = []
        self._current_command_index = 0
        self._clear_segment_playback()

    def add_cartesian_waypoint(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        max_frames: int = 400,
        pos_tolerance: typing.Optional[float] = None,
        linear: bool = False,
        linear_step: float = 0.001,
        joint_interp: bool = False,
        joint_steps: int = 120,
        hold_gripper: bool = False,
        target_is_hand: bool = False,
        label: str = "",
    ) -> None:
        self._command_queue.append({
            "type": "cartesian",
            "pos": np.asarray(position, dtype=np.float64),
            "ori": np.asarray(orientation, dtype=np.float64),
            "max_frames": int(max_frames),
            "frames_spent": 0,
            "pos_tolerance": pos_tolerance,
            "linear": bool(linear),
            "linear_step": float(linear_step),
            "joint_interp": bool(joint_interp),
            "joint_steps": int(joint_steps),
            "hold_gripper": bool(hold_gripper),
            "target_is_hand": bool(target_is_hand),
            "label": str(label),
        })

    def add_gripper_command(self, action: str, wait_frames: int = 60) -> None:
        self._command_queue.append({
            "type": "gripper",
            "action": str(action),
            "max_frames": int(wait_frames),
            "frames_spent": 0,
        })

    def forward(self, current_joint_positions: np.ndarray) -> ArticulationAction:
        n_dof = int(current_joint_positions.shape[0])

        while not self.is_done():
            cmd = self._command_queue[self._current_command_index]

            if cmd["type"] == "gripper":
                cmd["frames_spent"] += 1
                if cmd["frames_spent"] >= cmd["max_frames"]:
                    if self._debug:
                        carb.log_info(f"[Controller] Gripper {cmd['action']!r} complete")
                    self._advance_command()
                    continue
                return self._gripper.forward(action=cmd["action"])

            if cmd["type"] != "cartesian":
                carb.log_warn(f"Unknown command type: {cmd.get('type')}")
                return self._hold_action(n_dof)

            if not self._segment_ready:
                if cmd.get("joint_interp"):
                    self._init_joint_interp_segment(cmd, current_joint_positions, n_dof)
                elif cmd.get("linear"):
                    self._init_linear_segment(cmd)
                else:
                    self._build_trajectory_segment(cmd)
                self._segment_ready = True
                self._action_index = 0
                cmd["frames_spent"] = 0

            if cmd.get("joint_interp"):
                cmd["frames_spent"] += 1
                finished = self._joint_interp_step >= self._joint_interp_steps
                timed_out = cmd["frames_spent"] >= cmd["max_frames"]
                if finished or timed_out:
                    self._log_waypoint_complete(cmd, timed_out)
                    self._advance_command()
                    continue
                return self._joint_interp_action(cmd, n_dof)

            if cmd.get("linear"):
                cmd["frames_spent"] += 1
                if self._segment_failed:
                    # A failed linear IK step means the commanded straight line is not currently
                    # executable. Do not silently advance to the next command and close the
                    # gripper in empty space. Freeze so the failure is obvious.
                    return self._hold_action(n_dof)
                goal_reached = self._linear_progress >= self._linear_length and self._segment_goal_reached(cmd)
                timed_out = cmd["frames_spent"] >= cmd["max_frames"]
                if goal_reached or timed_out:
                    self._log_waypoint_complete(cmd, timed_out)
                    self._advance_command()
                    continue
                return self._linear_ik_action(cmd, n_dof)

            if self._action_index < len(self._action_sequence):
                action = self._action_sequence[self._action_index]
                self._action_index += 1
                cmd["frames_spent"] += 1
                return self._with_closed_gripper(action, n_dof) if cmd.get("hold_gripper") else action

            cmd["frames_spent"] += 1
            timed_out = cmd["frames_spent"] >= cmd["max_frames"]
            if self._segment_goal_reached(cmd) or timed_out:
                self._log_waypoint_complete(cmd, timed_out)
                self._advance_command()
                continue
            return self._hold_action(n_dof)

        return self._hold_action(n_dof)

    def current_label(self) -> str:
        if self.is_done():
            return ""
        return str(self._command_queue[self._current_command_index].get("label", ""))

    def reset(self) -> None:
        super().reset()
        self._current_command_index = 0
        self._clear_segment_playback()
        for cmd in self._command_queue:
            cmd["frames_spent"] = 0

    def is_done(self) -> bool:
        return self._current_command_index >= len(self._command_queue)

    def _advance_command(self) -> None:
        self._current_command_index += 1
        self._clear_segment_playback()

    def _clear_segment_playback(self) -> None:
        self._action_sequence: typing.List[ArticulationAction] = []
        self._action_index = 0
        self._segment_ready = False
        self._segment_goal_hand = None
        self._linear_start = None
        self._linear_dir = np.zeros(3, dtype=np.float64)
        self._linear_length = 0.0
        self._linear_progress = 0.0
        self._linear_ik_warned = False
        self._segment_failed = False
        self._joint_interp_start = None
        self._joint_interp_goal = None
        self._joint_interp_step = 0
        self._joint_interp_steps = 0
        self._joint_interp_warned = False

    def _hold_action(self, n_dof: int) -> ArticulationAction:
        return ArticulationAction(joint_positions=[None] * n_dof)

    def _gripper_joint_indices(self, n_dof: int) -> typing.List[int]:
        closed = np.asarray(self._gripper.joint_closed_positions, dtype=np.float64).flatten()
        names = None
        for attr in ("dof_names", "joint_names"):
            value = getattr(self._robot, attr, None)
            if value is not None:
                names = list(value)
                break

        gripper_names = []
        for attr in ("joint_prim_names", "_joint_prim_names"):
            value = getattr(self._gripper, attr, None)
            if value is not None:
                gripper_names = list(value)
                break

        if names and gripper_names:
            indices = [names.index(name) for name in gripper_names if name in names]
            if len(indices) == len(closed):
                return indices
        return list(range(max(0, n_dof - len(closed)), n_dof))

    def _with_closed_gripper(self, action: ArticulationAction, n_dof: int) -> ArticulationAction:
        raw_positions = None if action is None else getattr(action, "joint_positions", None)
        positions = [None] * n_dof if raw_positions is None else list(raw_positions)
        if len(positions) < n_dof:
            positions.extend([None] * (n_dof - len(positions)))
        elif len(positions) > n_dof:
            positions = positions[:n_dof]

        closed = np.asarray(self._gripper.joint_closed_positions, dtype=np.float64).flatten()
        for finger_i, joint_i in enumerate(self._gripper_joint_indices(n_dof)):
            if finger_i < len(closed) and 0 <= joint_i < len(positions):
                positions[joint_i] = float(closed[finger_i])
        return ArticulationAction(joint_positions=positions)

    def _hand_from_block(self, block_pos: np.ndarray, ori_wxyz: np.ndarray) -> np.ndarray:
        rot = quats_to_rot_matrices(np.asarray(ori_wxyz, dtype=np.float64).reshape(1, 4))[0]
        return np.asarray(block_pos, dtype=np.float64) - rot @ np.array([0.0, 0.0, self._tool_offset])

    def _goal_hand_from_command(self, cmd: dict) -> np.ndarray:
        if cmd.get("target_is_hand", False):
            return np.asarray(cmd["pos"], dtype=np.float64)
        return self._hand_from_block(cmd["pos"], cmd["ori"])

    def _current_hand_pose(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        hand_pos, hand_rot = self._art_kinematics.compute_end_effector_pose()
        hand_quat = rot_matrices_to_quats(hand_rot)
        if hand_quat.ndim > 1:
            hand_quat = hand_quat[0]
        return np.asarray(hand_pos, dtype=np.float64), np.asarray(hand_quat, dtype=np.float64)

    def _build_hand_waypoints(self, cmd: dict) -> typing.Tuple[np.ndarray, np.ndarray]:
        hand_start_pos, hand_start_quat = self._current_hand_pose()
        hand_goal_pos = self._goal_hand_from_command(cmd)
        orientations = np.stack([hand_start_quat, cmd["ori"]], axis=0)

        if cmd.get("target_is_hand", False):
            positions = np.stack([hand_start_pos, hand_goal_pos], axis=0).astype(np.float64)
            return positions, orientations

        # Preserve the behavior of the working baseline for block-center waypoints.
        # Those waypoints were tuned with the legacy TOOL_OFFSET conversion.
        block_points = np.stack([hand_start_pos, hand_goal_pos], axis=0)
        positions = np.array(
            [self._hand_from_block(p, orientations[i]) for i, p in enumerate(block_points)],
            dtype=np.float64,
        )
        return positions, orientations

    def _build_trajectory_segment(self, cmd: dict) -> None:
        hand_positions, orientations = self._build_hand_waypoints(cmd)
        self._segment_goal_hand = self._goal_hand_from_command(cmd)
        tag = self._debug_tag(cmd)

        if self._debug:
            carb.log_info(
                f"[Controller] Building trajectory{tag}: target={np.round(cmd['pos'], 4)} "
                f"hand_target={np.round(self._segment_goal_hand, 4)}"
            )

        trajectory = self._task_traj_gen.compute_task_space_trajectory_from_points(
            hand_positions, orientations, self._ee_frame
        )
        if trajectory is None:
            carb.log_warn(f"Lula task-space trajectory failed{tag}; falling back to IK.")
            action, success = self._art_kinematics.compute_inverse_kinematics(
                target_position=self._segment_goal_hand,
                target_orientation=cmd["ori"],
                position_tolerance=cmd.get("pos_tolerance") or self._pos_tolerance,
                orientation_tolerance=self._ori_tolerance,
            )
            if not success:
                carb.log_warn(f"IK fallback did not report convergence{tag}.")
            self._action_sequence = [action]
            return

        art_traj = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)
        self._action_sequence = art_traj.get_action_sequence()

    def _init_linear_segment(self, cmd: dict) -> None:
        start, _ = self._current_hand_pose()
        goal = self._goal_hand_from_command(cmd)
        self._segment_goal_hand = goal
        delta = goal - start
        self._linear_length = float(np.linalg.norm(delta))
        self._linear_start = start
        self._linear_dir = delta / self._linear_length if self._linear_length > 1e-9 else np.zeros(3)
        self._linear_progress = 0.0
        self._linear_ik_warned = False

    def _linear_ik_action(self, cmd: dict, n_dof: int) -> ArticulationAction:
        next_progress = min(self._linear_progress + cmd["linear_step"], self._linear_length)
        target = self._linear_start + self._linear_dir * next_progress
        action, success = self._art_kinematics.compute_inverse_kinematics(
            target_position=target,
            target_orientation=cmd["ori"],
            position_tolerance=cmd.get("pos_tolerance") or self._pos_tolerance,
            orientation_tolerance=self._ori_tolerance,
        )
        if not success:
            if not self._linear_ik_warned:
                carb.log_warn(
                    f"Linear IK failed at {np.round(target, 4)}; halting command queue "
                    f"{self._debug_tag(cmd)}."
                )
                self._linear_ik_warned = True
            self._segment_failed = True
            return self._hold_action(n_dof)

        # Only advance the virtual path after IK succeeds. The previous version
        # advanced progress before solving, so one failed step could make the
        # target run away while the robot stayed frozen.
        self._linear_progress = next_progress
        return self._with_closed_gripper(action, n_dof) if cmd.get("hold_gripper") else action

    def _init_joint_interp_segment(self, cmd: dict, current_joint_positions: np.ndarray, n_dof: int) -> None:
        self._segment_goal_hand = self._goal_hand_from_command(cmd)
        action, success = self._art_kinematics.compute_inverse_kinematics(
            target_position=self._segment_goal_hand,
            target_orientation=cmd["ori"],
            position_tolerance=cmd.get("pos_tolerance") or self._pos_tolerance,
            orientation_tolerance=self._ori_tolerance,
        )
        start = np.asarray(current_joint_positions, dtype=np.float64).copy()
        goal = start.copy()
        if success and getattr(action, "joint_positions", None) is not None:
            for i, value in enumerate(list(action.joint_positions)[:n_dof]):
                if value is not None:
                    goal[i] = float(value)
        else:
            carb.log_warn(f"Joint-interp IK target failed{self._debug_tag(cmd)}; holding.")
            self._joint_interp_warned = True

        self._joint_interp_start = start
        self._joint_interp_goal = goal
        self._joint_interp_steps = max(1, int(cmd.get("joint_steps", 120)))
        self._joint_interp_step = 0

    def _joint_interp_action(self, cmd: dict, n_dof: int) -> ArticulationAction:
        self._joint_interp_step += 1
        t = min(1.0, self._joint_interp_step / float(self._joint_interp_steps))
        s = t * t * (3.0 - 2.0 * t)
        q = self._joint_interp_start + s * (self._joint_interp_goal - self._joint_interp_start)
        action = ArticulationAction(joint_positions=q.tolist())
        return self._with_closed_gripper(action, n_dof) if cmd.get("hold_gripper") else action

    def _segment_goal_reached(self, cmd: dict) -> bool:
        tolerance = cmd.get("pos_tolerance") or self._pos_tolerance
        hand_pos, _ = self._current_hand_pose()
        return float(np.linalg.norm(self._segment_goal_hand - hand_pos)) < tolerance

    def _log_waypoint_complete(self, cmd: dict, timed_out: bool) -> None:
        if not self._debug:
            return
        hand_pos, _ = self._current_hand_pose()
        dist = float(np.linalg.norm(self._segment_goal_hand - hand_pos)) if self._segment_goal_hand is not None else np.nan
        note = " TIMEOUT" if timed_out else ""
        carb.log_info(
            f"[Controller] Waypoint complete{self._debug_tag(cmd)}: "
            f"final_err={dist * 1000.0:.2f}mm frames={cmd['frames_spent']}{note}"
        )

    @staticmethod
    def _debug_tag(cmd: dict) -> str:
        label = str(cmd.get("label", ""))
        return f" [{label}]" if label else ""
