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
    """Waypoint controller for block-center goals using IK by default.

    Closed-loop IK: align_yz_only, x_only_insert (locked Y/Z, crawl X),
    z_only_lift (locked X/Y, crawl Z). Same pattern for horizontal vs vertical.
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
        settle_frames: int = 10,
        max_ik_error: float = 0.02,
        insert_max_ik_error: float = 0.05,
        insert_orientation_tolerance: float = 0.02,
        ee_frame: str = "panda_hand",
    ) -> None:
        BaseController.__init__(self, name=name)
        self._robot = robot_articulation
        self._lula_kinematics = lula_kinematics
        self._task_traj_gen = task_traj_gen
        self._art_kinematics = art_kinematics
        self._joints_view = art_kinematics.get_joints_subset()
        self._gripper = gripper
        self._ee_frame = ee_frame

        self._tool_offset = tool_offset
        self._physics_dt = physics_dt
        self._pos_tolerance = position_tolerance
        self._ori_tolerance = orientation_tolerance
        self._cartesian_step = cartesian_step
        self._insert_velocity_scale = insert_velocity_scale
        self._settle_frames = settle_frames
        self._max_ik_error = max_ik_error
        self._insert_max_ik_error = insert_max_ik_error
        self._insert_ori_tolerance = insert_orientation_tolerance

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
        use_trajectory: bool = False,
        linear: bool = False,
        cartesian_step: typing.Optional[float] = None,
        track_block: bool = False,
        align_yz_only: bool = False,
        align_y_only: bool = False,
        align_z_only: bool = False,
        x_only_insert: bool = False,
        z_only_lift: bool = False,
        hold_yz: bool = False,
        slow_motion: bool = False,
    ) -> None:
        if hold_yz:
            x_only_insert = True
        self._command_queue.append({
            "type": "cartesian",
            "pos": np.asarray(position, dtype=np.float64),
            "ori": np.asarray(orientation, dtype=np.float64),
            "max_frames": max_frames,
            "frames_spent": 0,
            "pos_tolerance": pos_tolerance,
            "use_trajectory": use_trajectory,
            "linear": linear,
            "cartesian_step": cartesian_step,
            "track_block": track_block,
            "align_yz_only": align_yz_only,
            "align_y_only": align_y_only,
            "align_z_only": align_z_only,
            "x_only_insert": x_only_insert,
            "z_only_lift": z_only_lift,
            "slow_motion": slow_motion,
            "closed_loop": (
                align_yz_only
                or align_y_only
                or align_z_only
                or x_only_insert
                or z_only_lift
            ),
            "settle_count": 0,
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

    def _block_from_tracked_or_fk(
        self, current_tracked_position: typing.Optional[np.ndarray]
    ) -> np.ndarray:
        if current_tracked_position is not None:
            return np.asarray(current_tracked_position, dtype=np.float64)
        hand_pos, hand_quat = self._current_hand_pose()
        return (
            hand_pos
            + quats_to_rot_matrices(hand_quat.reshape(1, 4))[0]
            @ np.array([0.0, 0.0, self._tool_offset], dtype=np.float64)
        )

    def _ik_tolerance(self, current_cmd: dict) -> float:
        return current_cmd.get("pos_tolerance") or self._pos_tolerance

    def _yz_error(self, tracked: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(tracked[1:3] - goal[1:3]))

    def _x_insert_reached(
        self,
        tracked: np.ndarray,
        goal: np.ndarray,
        tol: float,
        current_cmd: dict,
    ) -> bool:
        sign = current_cmd.get("insert_sign")
        if sign is None or abs(sign) < 1e-6:
            return abs(tracked[0] - goal[0]) < tol
        if sign > 0:
            return tracked[0] >= goal[0] - tol
        return tracked[0] <= goal[0] + tol

    def _z_lift_reached(
        self,
        tracked: np.ndarray,
        goal: np.ndarray,
        tol: float,
        current_cmd: dict,
    ) -> bool:
        sign = current_cmd.get("lift_sign")
        if sign is None or abs(sign) < 1e-6:
            return abs(tracked[2] - goal[2]) < tol
        if sign > 0:
            return tracked[2] >= goal[2] - tol
        return tracked[2] <= goal[2] + tol

    def _xy_error(self, tracked: np.ndarray, locked_xy: np.ndarray) -> float:
        return float(np.linalg.norm(tracked[0:2] - locked_xy))

    def _yz_drift(self, tracked: np.ndarray, locked_yz: np.ndarray) -> float:
        return float(np.linalg.norm(tracked[1:3] - locked_yz))

    def _clip_axis_subgoal(
        self,
        sub_axis: float,
        tracked_axis: float,
        goal_axis: float,
        axis_sign: float,
    ) -> float:
        """Keep commanded axis between block and goal so IK does not run ahead at the end."""
        if axis_sign < 0:
            lo, hi = goal_axis, tracked_axis
        else:
            lo, hi = tracked_axis, goal_axis
        if lo > hi:
            return goal_axis
        return float(np.clip(sub_axis, lo, hi))

    def _solve_ik_to_block(
        self,
        block_pos: np.ndarray,
        ori_wxyz: np.ndarray,
        warm_start: typing.Optional[np.ndarray],
        pos_tolerance: float,
    ) -> typing.Tuple[typing.Optional[ArticulationAction], bool]:
        hand_pos = self._hand_from_block(block_pos, ori_wxyz)
        if warm_start is None:
            warm_start = self._joints_view.get_joint_positions()
        if warm_start is not None and hasattr(warm_start, "cpu"):
            warm_start = warm_start.cpu().numpy()

        ik_joints, success = self._lula_kinematics.compute_inverse_kinematics(
            self._ee_frame,
            hand_pos,
            ori_wxyz,
            warm_start,
            position_tolerance=pos_tolerance,
            orientation_tolerance=self._ori_tolerance,
        )
        if ik_joints is None or not np.all(np.isfinite(np.asarray(ik_joints, dtype=np.float64))):
            return None, False
        return self._joints_view.make_articulation_action(
            np.asarray(ik_joints, dtype=np.float64), None
        ), success

    def _init_closed_loop_segment(
        self,
        current_cmd: dict,
        tracked: typing.Optional[np.ndarray] = None,
    ) -> None:
        current_cmd["settle_count"] = 0
        current_cmd["frames_spent"] = 0
        if current_cmd.get("x_only_insert") and tracked is not None:
            goal = current_cmd["pos"]
            # Lock Y from the tracked block at segment start, but never allow Z to
            # drop below the commanded insert plane when x_insert begins.
            # This avoids a boundary-sample dip where the first x_insert frame
            # captures a stale/lower tracked Z and drags the block downward.
            current_cmd["locked_yz"] = np.array(
                [tracked[1], max(tracked[2], goal[2])], dtype=np.float64
            )
            delta = float(goal[0] - tracked[0])
            current_cmd["insert_sign"] = np.sign(delta) if abs(delta) > 1e-6 else -1.0
        if current_cmd.get("z_only_lift") and tracked is not None:
            goal = current_cmd["pos"]
            current_cmd["locked_xy"] = tracked[0:2].copy()
            delta = float(goal[2] - tracked[2])
            current_cmd["lift_sign"] = np.sign(delta) if abs(delta) > 1e-6 else 1.0

    def _subgoal_block(self, current_cmd: dict, tracked: np.ndarray) -> np.ndarray:
        goal = current_cmd["pos"]
        if current_cmd.get("align_yz_only"):
            return goal.copy()
        if current_cmd.get("align_y_only"):
            return np.array([goal[0], goal[1], tracked[2]], dtype=np.float64)
        if current_cmd.get("align_z_only"):
            return np.array([goal[0], tracked[1], goal[2]], dtype=np.float64)
        tol = self._ik_tolerance(current_cmd)
        if current_cmd.get("z_only_lift"):
            locked_xy = current_cmd.get("locked_xy", goal[0:2])
            if self._z_lift_reached(tracked, goal, tol, current_cmd):
                return np.array([locked_xy[0], locked_xy[1], goal[2]], dtype=np.float64)
            step = current_cmd.get("cartesian_step") or self._cartesian_step
            dz = float(goal[2] - tracked[2])
            sub_z = float(tracked[2] + np.sign(dz) * min(step, abs(dz)))
            return np.array([locked_xy[0], locked_xy[1], sub_z], dtype=np.float64)
        if current_cmd.get("x_only_insert"):
            locked_yz = current_cmd.get("locked_yz", goal[1:3])
            if self._x_insert_reached(tracked, goal, tol, current_cmd):
                return np.array([goal[0], locked_yz[0], locked_yz[1]], dtype=np.float64)
            step = current_cmd.get("cartesian_step") or self._cartesian_step
            insert_sign = float(current_cmd.get("insert_sign", -1.0))
            # Advance from block pose only (never command X past the block toward the port).
            if insert_sign < 0:
                sub_x = float(
                    tracked[0] - min(step, max(0.0, tracked[0] - goal[0]))
                )
            else:
                sub_x = float(
                    tracked[0] + min(step, max(0.0, goal[0] - tracked[0]))
                )
            sub_x = self._clip_axis_subgoal(sub_x, tracked[0], goal[0], insert_sign)
            return np.array([sub_x, locked_yz[0], locked_yz[1]], dtype=np.float64)

    def _update_settle_count(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> None:
        if not current_cmd["track_block"] or current_tracked_position is None:
            current_cmd["settle_count"] = 0
            return

        tracked = np.asarray(current_tracked_position, dtype=np.float64)
        goal = current_cmd["pos"]
        tol = self._ik_tolerance(current_cmd)

        if current_cmd.get("align_yz_only"):
            ok = (
                abs(tracked[1] - goal[1]) < tol
                and abs(tracked[2] - goal[2]) < tol
            )
        elif current_cmd.get("align_y_only"):
            ok = abs(tracked[1] - goal[1]) < tol
        elif current_cmd.get("align_z_only"):
            ok = abs(tracked[2] - goal[2]) < tol
        elif current_cmd.get("x_only_insert"):
            ok = self._x_insert_reached(tracked, goal, tol, current_cmd)
        elif current_cmd.get("z_only_lift"):
            ok = self._z_lift_reached(tracked, goal, tol, current_cmd)
        else:
            ok = float(np.linalg.norm(goal - tracked)) < tol

        if ok:
            current_cmd["settle_count"] += 1
        else:
            current_cmd["settle_count"] = 0

    def _closed_loop_segment_done(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> bool:
        return current_cmd["settle_count"] >= self._settle_frames

    def _forward_closed_loop(
        self,
        current_cmd: dict,
        current_joint_positions: np.ndarray,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> ArticulationAction:
        n_dof = current_joint_positions.shape[0]

        tracked = self._block_from_tracked_or_fk(current_tracked_position)
        goal = current_cmd["pos"]
        tol = self._ik_tolerance(current_cmd)

        if current_cmd.get("align_yz_only"):
            mode = "align_yz"
        elif current_cmd.get("align_y_only"):
            mode = "align_y"
        elif current_cmd.get("align_z_only"):
            mode = "align_z"
        elif current_cmd.get("z_only_lift"):
            mode = "z_lift"
        else:
            mode = "x_insert"

        if current_cmd.get("x_only_insert") and self._x_insert_reached(
            tracked, goal, tol, current_cmd
        ):
            locked_yz = current_cmd.get("locked_yz", goal[1:3])
            current_cmd["frames_spent"] += 1
            self._update_settle_count(current_cmd, current_tracked_position)
            self._traj_debug = {
                "cmd_index": self._current_command_index,
                "mode": "x_insert",
                "subgoal": np.array([goal[0], locked_yz[0], locked_yz[1]]),
                "x_err": float(tracked[0] - goal[0]),
                "yz_drift": self._yz_drift(tracked, locked_yz),
                "settle_count": current_cmd["settle_count"],
            }
            if self._closed_loop_segment_done(current_cmd, current_tracked_position):
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(current_joint_positions, current_tracked_position)
            if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                carb.log_warn(
                    f"Closed-loop waypoint timed out (x_insert): "
                    f"x_err={tracked[0] - goal[0]:.4f} m, "
                    f"yz_drift={self._yz_drift(tracked, locked_yz):.4f} m"
                )
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(current_joint_positions, current_tracked_position)
            return self._hold_action(n_dof)

        if current_cmd.get("z_only_lift") and self._z_lift_reached(
            tracked, goal, tol, current_cmd
        ):
            locked_xy = current_cmd.get("locked_xy", goal[0:2])
            current_cmd["frames_spent"] += 1
            self._update_settle_count(current_cmd, current_tracked_position)
            self._traj_debug = {
                "cmd_index": self._current_command_index,
                "mode": "z_lift",
                "subgoal": np.array([locked_xy[0], locked_xy[1], goal[2]]),
                "z_err": float(tracked[2] - goal[2]),
                "xy_err": self._xy_error(tracked, locked_xy),
                "settle_count": current_cmd["settle_count"],
            }
            if self._closed_loop_segment_done(current_cmd, current_tracked_position):
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(current_joint_positions, current_tracked_position)
            if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                carb.log_warn(
                    f"Closed-loop waypoint timed out (z_lift): "
                    f"z_err={tracked[2] - goal[2]:.4f} m"
                )
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(current_joint_positions, current_tracked_position)
            return self._hold_action(n_dof)

        subgoal = self._subgoal_block(current_cmd, tracked)

        ori_tol = self._insert_ori_tolerance
        warm_start = self._joints_view.get_joint_positions()
        hand_pos = self._hand_from_block(subgoal, current_cmd["ori"])
        if warm_start is not None and hasattr(warm_start, "cpu"):
            warm_start = warm_start.cpu().numpy()
        ik_joints, success = self._lula_kinematics.compute_inverse_kinematics(
            self._ee_frame,
            hand_pos,
            current_cmd["ori"],
            warm_start,
            position_tolerance=tol,
            orientation_tolerance=ori_tol,
        )
        if ik_joints is None or not np.all(np.isfinite(np.asarray(ik_joints, dtype=np.float64))):
            action = None
        else:
            action = self._joints_view.make_articulation_action(
                np.asarray(ik_joints, dtype=np.float64), None
            )

        max_fk_err = (
            self._insert_max_ik_error
            if current_cmd.get("x_only_insert") or current_cmd.get("z_only_lift")
            else self._max_ik_error
        )
        fk_reject = False
        if action is not None and not success and action.joint_positions is not None:
            fk_pos, _ = self._lula_kinematics.compute_forward_kinematics(
                self._ee_frame, np.asarray(action.joint_positions, dtype=np.float64)
            )
            hand_target = self._hand_from_block(subgoal, current_cmd["ori"])
            fk_reject = float(np.linalg.norm(fk_pos - hand_target)) > max_fk_err

        if action is None or fk_reject:
            current_cmd["frames_spent"] += 1
            if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                carb.log_warn(
                    f"Closed-loop waypoint timed out ({mode}, ik_hold): "
                    f"frames={current_cmd['frames_spent']}"
                )
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(current_joint_positions, current_tracked_position)
            return self._hold_action(n_dof)

        current_cmd["frames_spent"] += 1
        self._update_settle_count(current_cmd, current_tracked_position)

        dbg = {
            "cmd_index": self._current_command_index,
            "mode": mode,
            "subgoal": subgoal.copy(),
            "settle_count": current_cmd["settle_count"],
        }
        if mode == "z_lift":
            locked_xy = current_cmd.get("locked_xy", goal[0:2])
            dbg["z_err"] = float(tracked[2] - goal[2])
            dbg["xy_err"] = self._xy_error(tracked, locked_xy)
        elif mode == "x_insert":
            locked_yz = current_cmd.get("locked_yz", goal[1:3])
            dbg["x_err"] = float(tracked[0] - goal[0])
            dbg["yz_drift"] = self._yz_drift(tracked, locked_yz)
        else:
            dbg["x_err"] = float(tracked[0] - goal[0])
            dbg["yz_err"] = tracked[1:3] - goal[1:3]
        self._traj_debug = dbg

        if self._closed_loop_segment_done(current_cmd, current_tracked_position):
            self._current_command_index += 1
            self._clear_segment_playback()
            return self.forward(current_joint_positions, current_tracked_position)

        if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
            if mode == "z_lift":
                locked_xy = current_cmd.get("locked_xy", goal[0:2])
                carb.log_warn(
                    f"Closed-loop waypoint timed out ({mode}): "
                    f"z_err={tracked[2] - goal[2]:.4f} m, "
                    f"xy_err={self._xy_error(tracked, locked_xy):.4f} m"
                )
            elif mode == "x_insert":
                locked_yz = current_cmd.get("locked_yz", goal[1:3])
                carb.log_warn(
                    f"Closed-loop waypoint timed out ({mode}): "
                    f"x_err={tracked[0] - goal[0]:.4f} m, "
                    f"yz_drift={self._yz_drift(tracked, locked_yz):.4f} m"
                )
            else:
                carb.log_warn(
                    f"Closed-loop waypoint timed out ({mode}): "
                    f"x_err={tracked[0] - goal[0]:.4f} m, "
                    f"yz_err={self._yz_error(tracked, goal):.4f} m"
                )
            self._current_command_index += 1
            self._clear_segment_playback()
            return self.forward(current_joint_positions, current_tracked_position)

        return action

    def _build_single_ik_action(self, current_cmd: dict) -> bool:
        warm_start = self._joints_view.get_joint_positions()
        action, success = self._solve_ik_to_block(
            current_cmd["pos"],
            current_cmd["ori"],
            warm_start,
            self._ik_tolerance(current_cmd),
        )
        if action is None:
            carb.log_warn("IK failed for cartesian waypoint.")
            return False
        if not success:
            carb.log_warn("IK did not report convergence for cartesian waypoint.")
        self._action_sequence = [action]
        self._traj_debug = {
            "cmd_index": self._current_command_index,
            "num_actions": 1,
            "mode": "ik_single",
            "goal_block": current_cmd["pos"].copy(),
        }
        return True

    def _sample_block_positions_linear(
        self,
        start_block: np.ndarray,
        goal_block: np.ndarray,
        hold_yz: bool,
        step: float,
    ) -> np.ndarray:
        start_block = np.asarray(start_block, dtype=np.float64)
        goal_block = np.asarray(goal_block, dtype=np.float64)
        if hold_yz:
            dist = float(abs(goal_block[0] - start_block[0]))
            n = max(2, int(np.ceil(dist / step)) + 1)
            xs = np.linspace(start_block[0], goal_block[0], n)
            return np.array(
                [[x, goal_block[1], goal_block[2]] for x in xs], dtype=np.float64
            )
        dist = float(np.linalg.norm(goal_block - start_block))
        n = max(2, int(np.ceil(dist / step)) + 1)
        return np.linspace(start_block, goal_block, n)

    def _build_trajectory_segment(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> bool:
        goal_block = current_cmd["pos"]
        goal_ori = current_cmd["ori"]
        step = current_cmd.get("cartesian_step") or self._cartesian_step
        start_block = self._block_from_tracked_or_fk(current_tracked_position)

        if current_cmd["linear"]:
            block_points = self._sample_block_positions_linear(
                start_block, goal_block, current_cmd.get("x_only_insert", False), step
            )
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

        if current_cmd.get("slow_motion"):
            scale = self._insert_velocity_scale
            self._task_traj_gen.set_c_space_velocity_limits(self._default_velocity_limits * scale)
            self._task_traj_gen.set_c_space_acceleration_limits(
                self._default_acceleration_limits * scale
            )
        try:
            trajectory = self._task_traj_gen.compute_task_space_trajectory_from_points(
                hand_positions, orientations, self._ee_frame
            )
        finally:
            self._task_traj_gen.set_c_space_velocity_limits(self._default_velocity_limits)
            self._task_traj_gen.set_c_space_acceleration_limits(self._default_acceleration_limits)

        if trajectory is None:
            carb.log_warn("Lula trajectory failed; falling back to single IK.")
            return self._build_single_ik_action(current_cmd)

        art_traj = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)
        self._action_sequence = art_traj.get_action_sequence()
        self._traj_debug = {
            "cmd_index": self._current_command_index,
            "num_actions": len(self._action_sequence),
            "mode": "trajectory",
            "duration_s": art_traj.get_trajectory_duration(),
            "goal_block": goal_block.copy(),
        }
        return True

    def _build_segment_actions(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> bool:
        if current_cmd.get("closed_loop"):
            start_block = self._block_from_tracked_or_fk(current_tracked_position)
            self._init_closed_loop_segment(current_cmd, start_block)
            return True
        if current_cmd.get("use_trajectory"):
            return self._build_trajectory_segment(current_cmd, current_tracked_position)
        return self._build_single_ik_action(current_cmd)

    def _segment_goal_reached(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> bool:
        tolerance = self._ik_tolerance(current_cmd)
        goal = current_cmd["pos"]

        if current_cmd["track_block"] and current_tracked_position is not None:
            tracked = np.asarray(current_tracked_position, dtype=np.float64)
            return float(np.linalg.norm(goal - tracked)) < tolerance

        hand_pos, _ = self._current_hand_pose()
        goal_hand = self._hand_from_block(goal, current_cmd["ori"])
        return float(np.linalg.norm(goal_hand - hand_pos)) < tolerance

    def _current_segment_action(self, n_dof: int) -> ArticulationAction:
        if self._action_sequence:
            return self._action_sequence[min(self._action_index, len(self._action_sequence) - 1)]
        return self._hold_action(n_dof)

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
            if current_cmd.get("closed_loop"):
                if not self._segment_ready:
                    self._build_segment_actions(current_cmd, current_tracked_position)
                    self._segment_ready = True
                return self._forward_closed_loop(
                    current_cmd, current_joint_positions, current_tracked_position
                )

            if not self._segment_ready:
                if not self._build_segment_actions(current_cmd, current_tracked_position):
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

            return self._current_segment_action(n_dof)

        return self._hold_action(n_dof)

    def reset(self) -> None:
        BaseController.reset(self)
        self._current_command_index = 0
        self._clear_segment_playback()
        self._traj_debug = None
        for cmd in self._command_queue:
            cmd["frames_spent"] = 0
            if cmd["type"] == "cartesian":
                cmd["settle_count"] = 0
                cmd.pop("insert_sign", None)
                cmd.pop("lift_sign", None)
                cmd.pop("locked_xy", None)
                cmd.pop("locked_yz", None)

    def is_done(self) -> bool:
        return self._current_command_index >= len(self._command_queue)
