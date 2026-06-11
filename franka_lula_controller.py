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
        debug: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._robot = robot_articulation
        self._task_traj_gen = task_traj_gen
        self._art_kinematics = art_kinematics
        self._gripper = gripper
        self._ee_frame = ee_frame
        self._debug = debug

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

        self._joint_interp_start: typing.Optional[np.ndarray] = None
        self._joint_interp_goal: typing.Optional[np.ndarray] = None
        self._joint_interp_step = 0
        self._joint_interp_steps = 0
        self._joint_interp_warned = False

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
        """Queue a Cartesian goal.

        linear=True guarantees straight-line end-effector motion: the target is
        stepped along the start->goal line by linear_step meters per frame with
        per-frame IK, instead of relying on Lula trajectory generation.

        joint_interp=True bypasses Lula and uses one IK solve for the final
        pose, then smoothly interpolates joints to that IK solution. Use this
        for transport segments where Cartesian straightness is not task-critical.

        hold_gripper=True keeps commanding the gripper's closed target during
        this waypoint. Use it after a successful grasp. Without this, a joint
        interpolation can preserve the current finger width instead of applying
        squeeze force, and the block can slip during transit.

        target_is_hand=True means position is already a desired hand/end-effector
        target. Use this when the task script has already converted a desired
        block-center target into the exact hand target using the measured grasp
        offset. Leave it False for the older block-center waypoint behavior.

        label is an optional string used in debug output to identify waypoints.
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
            "joint_interp": joint_interp,
            "joint_steps": joint_steps,
            "hold_gripper": hold_gripper,
            "target_is_hand": target_is_hand,
            "label": label,
        })

    def add_gripper_command(self, action: str, wait_frames: int = 60) -> None:
        self._command_queue.append({
            "type": "gripper",
            "action": action,
            "max_frames": wait_frames,
            "frames_spent": 0,
        })

    def clear_queue(self) -> None:
        """Remove all queued commands and reset playback state.

        Use this before re-queuing a new phase of commands so previous
        commands don't persist and the index starts clean.
        """
        self._command_queue = []
        self._current_command_index = 0
        self._clear_segment_playback()

    def _hold_action(self, n_dof: int) -> ArticulationAction:
        return ArticulationAction(joint_positions=[None] * n_dof)

    def _gripper_joint_indices(self, n_dof: int) -> typing.List[int]:
        """Best-effort lookup of the finger joint indices in the articulation action."""
        closed = np.asarray(self._gripper.joint_closed_positions, dtype=np.float64).flatten()
        n_fingers = int(closed.shape[0])

        # Try common Isaac Sim name accessors first.
        names = None
        for attr in ("dof_names", "joint_names"):
            candidate = getattr(self._robot, attr, None)
            if candidate is not None:
                names = list(candidate)
                break

        gripper_names = []
        for attr in ("joint_prim_names", "_joint_prim_names"):
            candidate = getattr(self._gripper, attr, None)
            if candidate is not None:
                gripper_names = list(candidate)
                break

        if names and gripper_names:
            indices = []
            for gripper_name in gripper_names:
                if gripper_name in names:
                    indices.append(names.index(gripper_name))
            if len(indices) == n_fingers:
                return indices

        # Franka arm joints are followed by the two finger joints in the usual USD.
        # Fallback to the last N DOFs if name lookup is unavailable.
        return list(range(max(0, n_dof - n_fingers), n_dof))

    def _with_closed_gripper(self, action: ArticulationAction, n_dof: int) -> ArticulationAction:
        """Return an action that also keeps commanding the fingers closed.

        Important: IK/Lula actions often contain only the 7 arm joint values.
        The full Franka articulation has 9 DOFs when the two finger joints are
        included. If we extend joint_positions to 9 but pass through a 7-entry
        joint_velocities or joint_efforts array, Isaac's ArticulationController
        crashes while indexing DOF 7/8. So this function normalizes positions
        to n_dof and intentionally drops velocities/efforts.
        """
        raw_positions = None if action is None else getattr(action, "joint_positions", None)

        if raw_positions is None:
            joint_positions = [None] * n_dof
        else:
            joint_positions = list(raw_positions)
            if len(joint_positions) < n_dof:
                joint_positions.extend([None] * (n_dof - len(joint_positions)))
            elif len(joint_positions) > n_dof:
                joint_positions = joint_positions[:n_dof]

        closed = np.asarray(self._gripper.joint_closed_positions, dtype=np.float64).flatten()
        for finger_i, joint_i in enumerate(self._gripper_joint_indices(n_dof)):
            if finger_i < len(closed) and 0 <= joint_i < len(joint_positions):
                joint_positions[joint_i] = float(closed[finger_i])

        return ArticulationAction(joint_positions=joint_positions)

    def _hand_from_block(self, block_pos: np.ndarray, ori_wxyz: np.ndarray) -> np.ndarray:
        rot = quats_to_rot_matrices(np.asarray(ori_wxyz, dtype=np.float64).reshape(1, 4))[0]
        return np.asarray(block_pos, dtype=np.float64) - rot @ np.array(
            [0.0, 0.0, self._tool_offset], dtype=np.float64
        )

    def _goal_hand_from_command(self, current_cmd: dict) -> np.ndarray:
        """Return the hand target for a command.

        By default, command positions are interpreted as block-center targets and
        converted using the legacy TOOL_OFFSET behavior. When target_is_hand=True,
        the command position is already the desired hand target. This is used for
        high-accuracy insertion after the task script converts the desired block
        center path using the measured grasp offset.
        """
        if current_cmd.get("target_is_hand", False):
            return np.asarray(current_cmd["pos"], dtype=np.float64)
        return self._hand_from_block(current_cmd["pos"], current_cmd["ori"])

    def _current_hand_pose(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        hand_pos, hand_rot = self._art_kinematics.compute_end_effector_pose()
        hand_quat = rot_matrices_to_quats(hand_rot)
        if hand_quat.ndim > 1:
            hand_quat = hand_quat[0]
        return np.asarray(hand_pos, dtype=np.float64), np.asarray(hand_quat, dtype=np.float64)

    def _build_hand_waypoints(self, current_cmd: dict) -> typing.Tuple[np.ndarray, np.ndarray]:
        hand_start_pos, hand_start_quat = self._current_hand_pose()
        hand_goal_pos = self._goal_hand_from_command(current_cmd)
        orientations = np.stack([hand_start_quat, current_cmd["ori"]], axis=0)

        if current_cmd.get("target_is_hand", False):
            hand_positions = np.stack([hand_start_pos, hand_goal_pos], axis=0).astype(np.float64)
            return hand_positions, orientations

        # Preserve legacy behavior for existing grasp/transit waypoints. Those
        # waypoints were tuned with command positions interpreted as block-center
        # targets and TOOL_OFFSET conversion inside the controller.
        block_points = np.stack([hand_start_pos, hand_goal_pos], axis=0)
        hand_positions = np.array(
            [self._hand_from_block(p, orientations[i]) for i, p in enumerate(block_points)],
            dtype=np.float64,
        )
        return hand_positions, orientations

    def _build_trajectory_for_current_command(self, current_cmd: dict) -> None:
        hand_positions, orientations = self._build_hand_waypoints(current_cmd)
        self._segment_goal_hand = self._goal_hand_from_command(current_cmd)

        label = current_cmd.get("label", "")
        tag = f" [{label}]" if label else ""

        if self._debug:
            carb.log_info(
                f"[Controller] Building trajectory{tag} → target {np.round(current_cmd['pos'], 4)}"
                f"  hand_target={np.round(self._segment_goal_hand, 4)}"
            )

        trajectory = self._task_traj_gen.compute_task_space_trajectory_from_points(
            hand_positions, orientations, self._ee_frame
        )

        if trajectory is None:
            carb.log_warn(
                f"Lula task-space trajectory failed{tag}; falling back to single-point IK."
            )
            ik_action, success = self._art_kinematics.compute_inverse_kinematics(
                target_position=self._segment_goal_hand,
                target_orientation=current_cmd["ori"],
                position_tolerance=current_cmd.get("pos_tolerance") or self._pos_tolerance,
                orientation_tolerance=self._ori_tolerance,
            )
            if not success:
                carb.log_warn(f"IK fallback did not report convergence{tag}.")
            self._action_sequence = [ik_action]
            return

        art_traj = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)
        self._action_sequence = art_traj.get_action_sequence()

        if self._debug:
            carb.log_info(f"[Controller] Trajectory{tag} built: {len(self._action_sequence)} steps")

    def _init_linear_segment(self, current_cmd: dict) -> None:
        start, _ = self._current_hand_pose()
        goal = self._goal_hand_from_command(current_cmd)
        self._segment_goal_hand = goal
        delta = goal - start
        length = float(np.linalg.norm(delta))
        self._linear_start = start
        self._linear_dir = delta / length if length > 1e-9 else np.zeros(3, dtype=np.float64)
        self._linear_length = length
        self._linear_progress = 0.0
        self._linear_ik_warned = False
        self._joint_interp_start = None
        self._joint_interp_goal = None
        self._joint_interp_step = 0
        self._joint_interp_steps = 0
        self._joint_interp_warned = False

    def _linear_ik_action(self, current_cmd: dict, n_dof: int) -> ArticulationAction:
        next_progress = min(
            self._linear_progress + current_cmd["linear_step"], self._linear_length
        )
        target = self._linear_start + self._linear_dir * next_progress
        action, success = self._art_kinematics.compute_inverse_kinematics(
            target_position=target,
            target_orientation=current_cmd["ori"],
            position_tolerance=current_cmd.get("pos_tolerance") or self._pos_tolerance,
            orientation_tolerance=self._ori_tolerance,
        )
        if not success:
            if not self._linear_ik_warned:
                carb.log_warn(
                    f"Linear segment IK failed at {np.round(target, 4)}; holding."
                )
                self._linear_ik_warned = True
            return self._hold_action(n_dof)
        self._linear_progress = next_progress
        if current_cmd.get("hold_gripper"):
            return self._with_closed_gripper(action, n_dof)
        return action

    def _init_joint_interp_segment(
        self,
        current_cmd: dict,
        current_joint_positions: np.ndarray,
        n_dof: int,
    ) -> None:
        """Compute one IK target, then move there smoothly in joint space.

        This intentionally does not call Lula. It is for transport moves where
        the object only needs to get safely to the next area. Final insertion
        should still use linear=True.
        """
        self._segment_goal_hand = self._goal_hand_from_command(current_cmd)
        label = current_cmd.get("label", "")
        tag = f" [{label}]" if label else ""

        ik_action, success = self._art_kinematics.compute_inverse_kinematics(
            target_position=self._segment_goal_hand,
            target_orientation=current_cmd["ori"],
            position_tolerance=current_cmd.get("pos_tolerance") or self._pos_tolerance,
            orientation_tolerance=self._ori_tolerance,
        )

        if not success or ik_action.joint_positions is None:
            carb.log_warn(f"Joint-interp IK target failed{tag}; holding.")
            self._joint_interp_start = np.asarray(current_joint_positions, dtype=np.float64).copy()
            self._joint_interp_goal = self._joint_interp_start.copy()
            self._joint_interp_steps = 1
            self._joint_interp_step = 0
            self._joint_interp_warned = True
            return

        start = np.asarray(current_joint_positions, dtype=np.float64).copy()
        goal_raw = list(ik_action.joint_positions)

        # IK actions may leave non-cspace joints as None. Preserve current values,
        # especially the gripper finger joints, so the grasp does not open.
        goal = start.copy()
        for i in range(min(len(goal_raw), n_dof)):
            if goal_raw[i] is not None:
                goal[i] = float(goal_raw[i])

        self._joint_interp_start = start
        self._joint_interp_goal = goal
        self._joint_interp_steps = max(1, int(current_cmd.get("joint_steps", 120)))
        self._joint_interp_step = 0
        self._joint_interp_warned = False

        if self._debug:
            carb.log_info(
                f"[Controller] Joint-interp{tag}: steps={self._joint_interp_steps} "
                f"target={np.round(current_cmd['pos'], 4)}"
            )

    def _joint_interp_action(self, current_cmd: dict, n_dof: int) -> ArticulationAction:
        if self._joint_interp_start is None or self._joint_interp_goal is None:
            if not self._joint_interp_warned:
                carb.log_warn("Joint interpolation requested before initialization; holding.")
                self._joint_interp_warned = True
            return self._hold_action(n_dof)

        self._joint_interp_step += 1
        t = min(1.0, self._joint_interp_step / float(self._joint_interp_steps))
        # smoothstep avoids a hard acceleration impulse at the start/end.
        a = t * t * (3.0 - 2.0 * t)
        q = self._joint_interp_start + a * (self._joint_interp_goal - self._joint_interp_start)
        action = ArticulationAction(joint_positions=q.tolist())
        if current_cmd.get("hold_gripper"):
            return self._with_closed_gripper(action, n_dof)
        return action

    def _segment_goal_reached(self, current_cmd: dict) -> bool:
        tolerance = current_cmd.get("pos_tolerance") or self._pos_tolerance
        hand_pos, _ = self._current_hand_pose()
        dist = float(np.linalg.norm(self._segment_goal_hand - hand_pos))
        return dist < tolerance

    def _log_waypoint_complete(self, current_cmd: dict, timed_out: bool) -> None:
        if not self._debug:
            return
        label = current_cmd.get("label", "")
        tag = f" [{label}]" if label else ""
        hand_pos, _ = self._current_hand_pose()
        dist = float(np.linalg.norm(self._segment_goal_hand - hand_pos))
        timeout_note = " (TIMEOUT — goal not reached)" if timed_out else ""
        carb.log_info(
            f"[Controller] Waypoint complete{tag}: "
            f"final_err={dist*1000:.2f}mm  frames={current_cmd['frames_spent']}{timeout_note}"
        )

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
        self._joint_interp_start = None
        self._joint_interp_goal = None
        self._joint_interp_step = 0
        self._joint_interp_steps = 0
        self._joint_interp_warned = False

    def forward(self, current_joint_positions: np.ndarray) -> ArticulationAction:
        n_dof = current_joint_positions.shape[0]

        while not self.is_done():
            current_cmd = self._command_queue[self._current_command_index]

            if current_cmd["type"] == "gripper":
                current_cmd["frames_spent"] += 1
                if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                    if self._debug:
                        carb.log_info(
                            f"[Controller] Gripper '{current_cmd['action']}' complete "
                            f"(frames={current_cmd['frames_spent']})"
                        )
                    self._current_command_index += 1
                    self._clear_segment_playback()
                    continue
                return self._gripper.forward(action=current_cmd["action"])

            if current_cmd["type"] == "cartesian":
                if not self._segment_ready:
                    if current_cmd.get("joint_interp"):
                        self._init_joint_interp_segment(current_cmd, current_joint_positions, n_dof)
                    elif current_cmd.get("linear"):
                        self._init_linear_segment(current_cmd)
                    else:
                        self._build_trajectory_for_current_command(current_cmd)
                    self._segment_ready = True
                    self._action_index = 0
                    current_cmd["frames_spent"] = 0

                if current_cmd.get("joint_interp"):
                    current_cmd["frames_spent"] += 1
                    timed_out = current_cmd["frames_spent"] >= current_cmd["max_frames"]
                    finished = self._joint_interp_step >= self._joint_interp_steps
                    if finished or timed_out:
                        self._log_waypoint_complete(current_cmd, timed_out)
                        self._current_command_index += 1
                        self._clear_segment_playback()
                        continue
                    return self._joint_interp_action(current_cmd, n_dof)

                if current_cmd.get("linear"):
                    current_cmd["frames_spent"] += 1
                    timed_out = current_cmd["frames_spent"] >= current_cmd["max_frames"]
                    goal_reached = (
                        self._linear_progress >= self._linear_length
                        and self._segment_goal_reached(current_cmd)
                    )
                    if goal_reached or timed_out:
                        self._log_waypoint_complete(current_cmd, timed_out)
                        self._current_command_index += 1
                        self._clear_segment_playback()
                        continue
                    return self._linear_ik_action(current_cmd, n_dof)

                if self._action_index < len(self._action_sequence):
                    action = self._action_sequence[self._action_index]
                    self._action_index += 1
                    current_cmd["frames_spent"] += 1
                    if current_cmd.get("hold_gripper"):
                        return self._with_closed_gripper(action, n_dof)
                    return action

                current_cmd["frames_spent"] += 1
                timed_out = current_cmd["frames_spent"] >= current_cmd["max_frames"]
                if self._segment_goal_reached(current_cmd) or timed_out:
                    self._log_waypoint_complete(current_cmd, timed_out)
                    self._current_command_index += 1
                    self._clear_segment_playback()
                    continue

                return self._hold_action(n_dof)

            carb.log_warn(f"Unknown command type: {current_cmd.get('type')}")
            return self._hold_action(n_dof)

        return self._hold_action(n_dof)

    def current_label(self) -> str:
        """Return the label of the command currently being executed, if any."""
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