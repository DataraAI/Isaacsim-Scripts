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
        # Hand-local vector from the panda_hand frame to the grasped block center.
        # Defaults to the nominal tool offset along local Z; recalibrated from the
        # real grasp once the module is held so lateral grasp bias is compensated.
        self._grasp_offset_local = np.array([0.0, 0.0, tool_offset], dtype=np.float64)
        self._grasp_calibrated = False
        # Integral correction (world frame, perpendicular to the insert axis) that
        # drives the measured block lateral onto the goal during align, cancelling
        # steady-state IK tracking error. Frozen and reused during insertion.
        self._align_lat_correction = np.zeros(3, dtype=np.float64)
        self._lateral_feedback_gain = 0.10
        self._max_lateral_correction = 0.008
        self._lateral_feedback_deadband = 0.0010
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
        self._last_arm_action: typing.Optional[ArticulationAction] = None
        self._insert_contact_monitor: typing.Optional[object] = None
        self._post_contact_arm_joints: typing.Optional[np.ndarray] = None

    def set_insert_contact_monitor(self, monitor) -> None:
        self._insert_contact_monitor = monitor

    def _insert_contact_detected(self) -> bool:
        if self._insert_contact_monitor is None:
            return False
        return bool(self._insert_contact_monitor.in_contact())

    def _hold_arm_joints_action(
        self, current_joint_positions: np.ndarray
    ) -> ArticulationAction:
        return self._joints_view.make_articulation_action(
            np.asarray(current_joint_positions[:7], dtype=np.float64), None
        )

    def _advance_command(self) -> None:
        self._current_command_index += 1
        self._clear_segment_playback()

    def _stop_on_insert_contact(
        self,
        current_cmd: dict,
        current_joint_positions: np.ndarray,
    ) -> typing.Optional[ArticulationAction]:
        if not current_cmd.get("stop_on_module_contact"):
            return None
        if not self._insert_contact_detected():
            return None
        self._post_contact_arm_joints = np.asarray(
            current_joint_positions[:7], dtype=np.float64
        ).copy()
        hold = self._hold_arm_joints_action(current_joint_positions)
        self._cache_arm_action(hold)
        self._advance_command()
        carb.log_info("Insert stopped: QSFP module contact with port.")
        self._traj_debug = {
            "cmd_index": self._current_command_index - 1,
            "mode": "module_port_contact_stop",
        }
        return hold

    def _hand_pose_from_arm_joints(
        self, arm_joints: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(arm_joints, dtype=np.float64).reshape(-1)
        fk_pos, fk_rot = self._lula_kinematics.compute_forward_kinematics(
            self._ee_frame, joints
        )
        fk_quat = rot_matrices_to_quats(fk_rot)
        if fk_quat.ndim > 1:
            fk_quat = fk_quat[0]
        return np.asarray(fk_pos, dtype=np.float64), np.asarray(fk_quat, dtype=np.float64)

    def _block_from_arm_joints(self, arm_joints: np.ndarray) -> np.ndarray:
        hand_pos, hand_quat = self._hand_pose_from_arm_joints(arm_joints)
        rot = quats_to_rot_matrices(hand_quat.reshape(1, 4))[0]
        return hand_pos + rot @ self._grasp_offset_local

    def _block_for_command(
        self,
        current_cmd: dict,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> np.ndarray:
        # During post-release retreat track the hand directly (not a virtual block
        # center with grasp offset) so IK does not keep the fingers wrapped around
        # the seated module and drag it out of the port.
        if current_cmd.get("post_contact_retreat"):
            hand_pos, _ = self._current_hand_pose()
            return hand_pos
        return self._block_from_tracked_or_fk(current_tracked_position)

    def _warm_start_for_command(self, current_cmd: dict) -> typing.Optional[np.ndarray]:
        warm_start = self._joints_view.get_joint_positions()
        if warm_start is not None and hasattr(warm_start, "cpu"):
            warm_start = warm_start.cpu().numpy()
        return warm_start

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
        insert_axis: typing.Optional[np.ndarray] = None,
        insert_origin: typing.Optional[np.ndarray] = None,
        search_wiggle: bool = False,
        wiggle_amplitude: float = 0.0002,
        wiggle_period_frames: int = 30,
        compliant_insert: bool = False,
        contact_force_threshold: float = 15.0,
        hold_current_orientation: bool = False,
        stop_on_module_contact: bool = False,
        post_contact_retreat: bool = False,
        keep_gripper_open: bool = False,
        settle_frames: typing.Optional[int] = None,
        orientation_tolerance: typing.Optional[float] = None,
        target_hand_at_position: bool = False,
        verify_pick_lift: bool = False,
        verify_pick_min_z: typing.Optional[float] = None,
    ) -> None:
        if hold_yz:
            x_only_insert = True
        self._command_queue.append({
            "type": "cartesian",
            "pos": np.asarray(position, dtype=np.float64),
            "goal_pos": np.asarray(position, dtype=np.float64),
            "ori": np.asarray(orientation, dtype=np.float64),
            "goal_ori": np.asarray(orientation, dtype=np.float64),
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
            "insert_axis": (
                np.asarray(insert_axis, dtype=np.float64)
                if insert_axis is not None
                else None
            ),
            "insert_origin": (
                np.asarray(insert_origin, dtype=np.float64)
                if insert_origin is not None
                else None
            ),
            "search_wiggle": search_wiggle,
            "wiggle_amplitude": wiggle_amplitude,
            "wiggle_period_frames": max(wiggle_period_frames, 1),
            "compliant_insert": compliant_insert,
            "contact_force_threshold": contact_force_threshold,
            "hold_current_orientation": hold_current_orientation,
            "stop_on_module_contact": stop_on_module_contact,
            "post_contact_retreat": post_contact_retreat,
            "keep_gripper_open": keep_gripper_open,
            "settle_frames": settle_frames,
            "orientation_tolerance": orientation_tolerance,
            "target_hand_at_position": target_hand_at_position,
            "verify_pick_lift": verify_pick_lift,
            "verify_pick_min_z": verify_pick_min_z,
            "closed_loop": (
                align_yz_only
                or align_y_only
                or align_z_only
                or x_only_insert
                or z_only_lift
            ),
            "settle_count": 0,
        })

    def add_gripper_command(
        self,
        action: str,
        wait_frames: int = 60,
        *,
        full_open: bool = False,
        freeze_arm: bool = False,
    ) -> None:
        self._command_queue.append({
            "type": "gripper",
            "action": action,
            "max_frames": wait_frames,
            "frames_spent": 0,
            "full_open": full_open,
            "freeze_arm": freeze_arm,
        })

    def get_traj_debug_state(self) -> typing.Optional[dict]:
        return self._traj_debug

    def _axial_coord(
        self, position: np.ndarray, origin: np.ndarray, axis: np.ndarray
    ) -> float:
        return float(np.dot(position - origin, axis))

    def _lateral_offset_vec(
        self, position: np.ndarray, origin: np.ndarray, axis: np.ndarray
    ) -> np.ndarray:
        delta = np.asarray(position, dtype=np.float64) - origin
        return delta - axis * np.dot(delta, axis)

    def _pos_from_axial_lateral(
        self,
        origin: np.ndarray,
        axis: np.ndarray,
        axial: float,
        lateral: np.ndarray,
    ) -> np.ndarray:
        return origin + axis * axial + lateral

    def _uses_axis_insert(self, current_cmd: dict) -> bool:
        axis = current_cmd.get("insert_axis")
        return axis is not None and np.linalg.norm(axis) > 1e-9

    def _segment_origin(self, current_cmd: dict, goal: np.ndarray) -> np.ndarray:
        origin = current_cmd.get("insert_origin")
        if origin is not None:
            return np.asarray(origin, dtype=np.float64)
        return np.asarray(goal, dtype=np.float64)

    def _estimate_contact_force(self) -> float:
        try:
            efforts = self._robot.get_measured_joint_efforts()
            if efforts is None:
                return 0.0
            arr = np.asarray(efforts, dtype=np.float64).reshape(-1)
            return float(np.sum(np.abs(arr[:7])))
        except Exception:
            return 0.0

    def _effective_insert_step(self, current_cmd: dict, base_step: float) -> float:
        if not current_cmd.get("compliant_insert"):
            return base_step
        force = self._estimate_contact_force()
        threshold = float(current_cmd.get("contact_force_threshold", 15.0))
        if force <= threshold:
            return base_step
        return base_step * max(0.2, threshold / max(force, 1e-6))

    def _wiggle_lateral(
        self, current_cmd: dict, lateral: np.ndarray
    ) -> np.ndarray:
        if not current_cmd.get("search_wiggle"):
            return lateral
        amp = float(current_cmd.get("wiggle_amplitude", 0.0002))
        period = int(current_cmd.get("wiggle_period_frames", 30))
        phase = (current_cmd["frames_spent"] % period) / period * 2.0 * np.pi
        axis = current_cmd.get("insert_axis")
        if axis is None:
            return lateral + np.array([0.0, amp * np.sin(phase), 0.0], dtype=np.float64)
        # Wiggle in a plane perpendicular to insert axis.
        rot = quats_to_rot_matrices(
            np.asarray(current_cmd["ori"], dtype=np.float64).reshape(1, 4)
        )[0]
        lat_dir = rot[:, 1]
        lat_dir = lat_dir - axis * np.dot(lat_dir, axis)
        n = np.linalg.norm(lat_dir)
        if n < 1e-9:
            lat_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            lat_dir = lat_dir / n
        return lateral + lat_dir * (amp * np.sin(phase))

    def _axis_insert_reached(
        self,
        tracked: np.ndarray,
        goal: np.ndarray,
        tol: float,
        current_cmd: dict,
    ) -> bool:
        origin = self._segment_origin(current_cmd, goal)
        axis = current_cmd["insert_axis"]
        sign = float(current_cmd.get("insert_sign", 1.0))
        t_tracked = self._axial_coord(tracked, origin, axis)
        t_goal = self._axial_coord(goal, origin, axis)
        if sign > 0:
            return t_tracked >= t_goal - tol
        return t_tracked <= t_goal + tol

    def _lateral_error(
        self, tracked: np.ndarray, goal: np.ndarray, current_cmd: dict
    ) -> float:
        if not self._uses_axis_insert(current_cmd):
            return self._yz_error(tracked, goal)
        origin = self._segment_origin(current_cmd, goal)
        axis = current_cmd["insert_axis"]
        goal_lat = self._lateral_offset_vec(goal, origin, axis)
        tracked_lat = self._lateral_offset_vec(tracked, origin, axis)
        return float(np.linalg.norm(goal_lat - tracked_lat))

    def _hold_action(self, n_dof: int) -> ArticulationAction:
        return ArticulationAction(joint_positions=[None] * n_dof)

    def _cache_arm_action(self, action: typing.Optional[ArticulationAction]) -> None:
        if action is not None and action.joint_positions is not None:
            self._last_arm_action = action

    def _command_orientation(self, current_cmd: dict) -> np.ndarray:
        if current_cmd.get("hold_current_orientation"):
            _, hand_quat = self._current_hand_pose()
            return np.asarray(hand_quat, dtype=np.float64)
        if (
            current_cmd.get("post_contact_retreat")
            and self._post_contact_arm_joints is not None
        ):
            _, hand_quat = self._hand_pose_from_arm_joints(self._post_contact_arm_joints)
            return np.asarray(hand_quat, dtype=np.float64)
        return np.asarray(current_cmd["ori"], dtype=np.float64)

    def _arm_joints_for_gripper(
        self,
        current_cmd: dict,
        current_joint_positions: np.ndarray,
    ) -> np.ndarray:
        if current_cmd.get("freeze_arm") and self._post_contact_arm_joints is not None:
            return self._post_contact_arm_joints
        if "arm_hold_joints" not in current_cmd:
            current_cmd["arm_hold_joints"] = np.asarray(
                current_joint_positions[:7], dtype=np.float64
            ).copy()
        return current_cmd["arm_hold_joints"]

    def _gripper_with_frozen_arm(
        self,
        current_cmd: dict,
        gripper_action: ArticulationAction,
        current_joint_positions: np.ndarray,
    ) -> ArticulationAction:
        arm_joints = self._arm_joints_for_gripper(current_cmd, current_joint_positions)

        n_dof = int(current_joint_positions.shape[0])
        merged: typing.List[typing.Optional[float]] = [None] * n_dof
        for i in range(7):
            merged[i] = float(arm_joints[i])

        if current_cmd.get("full_open") and current_cmd["action"] == "open":
            opened = np.asarray(self._gripper.joint_opened_positions, dtype=np.float64)
            for i, idx in enumerate(self._gripper.active_joint_indices):
                merged[int(idx)] = float(opened[i])
        elif gripper_action.joint_positions is not None:
            for i, pos in enumerate(gripper_action.joint_positions):
                if pos is not None:
                    merged[i] = float(pos)

        return ArticulationAction(joint_positions=merged)

    def _merge_gripper_open(
        self, action: ArticulationAction, n_dof: int
    ) -> ArticulationAction:
        """Keep fingers fully open so retreat motion cannot drag the released module."""
        merged: typing.List[typing.Optional[float]] = [None] * n_dof
        if action.joint_positions is not None:
            for i, pos in enumerate(action.joint_positions):
                if pos is not None:
                    merged[i] = float(pos)
        opened = np.asarray(self._gripper.joint_opened_positions, dtype=np.float64)
        for i, idx in enumerate(self._gripper.active_joint_indices):
            merged[int(idx)] = float(opened[i])
        return ArticulationAction(joint_positions=merged)

    def _hand_from_block(self, block_pos: np.ndarray, ori_wxyz: np.ndarray) -> np.ndarray:
        rot = quats_to_rot_matrices(np.asarray(ori_wxyz, dtype=np.float64).reshape(1, 4))[0]
        return np.asarray(block_pos, dtype=np.float64) - rot @ self._grasp_offset_local

    def _hand_goal_from_command(
        self, current_cmd: dict, position: np.ndarray, ori_wxyz: np.ndarray
    ) -> np.ndarray:
        if current_cmd.get("target_hand_at_position"):
            return np.asarray(position, dtype=np.float64)
        return self._hand_from_block(position, ori_wxyz)

    def reset_grasp_calibration(self) -> None:
        self._grasp_calibrated = False
        self._grasp_offset_local = np.array(
            [0.0, 0.0, self._tool_offset], dtype=np.float64
        )
        self._align_lat_correction = np.zeros(3, dtype=np.float64)

    def _current_hand_pose(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        pose = self._art_kinematics.compute_end_effector_pose()
        if pose is None or pose[0] is None or pose[1] is None:
            warm_start = self._joints_view.get_joint_positions()
            if warm_start is None:
                raise RuntimeError("End-effector pose unavailable before articulation initialization.")
            if hasattr(warm_start, "cpu"):
                warm_start = warm_start.cpu().numpy()
            hand_pos, hand_rot = self._lula_kinematics.compute_forward_kinematics(
                self._ee_frame, np.asarray(warm_start, dtype=np.float64)
            )
        else:
            hand_pos, hand_rot = pose
        hand_quat = rot_matrices_to_quats(hand_rot)
        if hand_quat.ndim > 1:
            hand_quat = hand_quat[0]
        return np.asarray(hand_pos, dtype=np.float64), np.asarray(hand_quat, dtype=np.float64)

    def _calibrate_grasp_offset(self, tracked: np.ndarray) -> None:
        """Measure the real hand->block offset in the hand frame to remove grasp bias."""
        hand_pos, hand_quat = self._current_hand_pose()
        rot = quats_to_rot_matrices(np.asarray(hand_quat, dtype=np.float64).reshape(1, 4))[0]
        offset_world = np.asarray(tracked, dtype=np.float64) - hand_pos
        self._grasp_offset_local = rot.T @ offset_world
        self._grasp_calibrated = True
        carb.log_info(
            f"Calibrated grasp offset (hand-local)={self._grasp_offset_local} "
            f"nominal=[0,0,{self._tool_offset}]"
        )

    def _block_from_tracked_or_fk(
        self, current_tracked_position: typing.Optional[np.ndarray]
    ) -> np.ndarray:
        if current_tracked_position is not None:
            return np.asarray(current_tracked_position, dtype=np.float64)
        hand_pos, hand_quat = self._current_hand_pose()
        return (
            hand_pos
            + quats_to_rot_matrices(hand_quat.reshape(1, 4))[0]
            @ self._grasp_offset_local
        )

    def _ik_tolerance(self, current_cmd: dict) -> float:
        return current_cmd.get("pos_tolerance") or self._pos_tolerance

    def _yz_error(self, tracked: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(tracked[1:3] - goal[1:3]))

    def _yz_error_components(
        self, tracked: np.ndarray, goal: np.ndarray
    ) -> typing.Tuple[float, float, float]:
        delta = np.asarray(tracked[1:3], dtype=np.float64) - np.asarray(
            goal[1:3], dtype=np.float64
        )
        return float(delta[0]), float(delta[1]), float(np.linalg.norm(delta))

    def _locked_yz_for_debug(self, current_cmd: dict, goal: np.ndarray) -> np.ndarray:
        if self._uses_axis_insert(current_cmd):
            origin = self._segment_origin(current_cmd, goal)
            axis = current_cmd["insert_axis"]
            if current_cmd.get("post_contact_retreat") and "locked_lateral" in current_cmd:
                target_lat = current_cmd["locked_lateral"]
            else:
                target_lat = (
                    self._lateral_offset_vec(goal, origin, axis)
                    + self._align_lat_correction
                )
            return (origin + target_lat)[1:3]
        return current_cmd.get("locked_yz", goal[1:3])

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
        current_cmd: typing.Optional[dict] = None,
    ) -> typing.Tuple[typing.Optional[ArticulationAction], bool]:
        if current_cmd is not None:
            hand_pos = self._hand_goal_from_command(current_cmd, block_pos, ori_wxyz)
        else:
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
        if (
            current_cmd.get("track_block")
            and tracked is not None
            and not self._grasp_calibrated
        ):
            self._calibrate_grasp_offset(tracked)
        if current_cmd.get("align_yz_only") and self._uses_axis_insert(current_cmd):
            self._align_lat_correction = np.zeros(3, dtype=np.float64)
        if current_cmd.get("post_contact_retreat") and self._uses_axis_insert(current_cmd):
            hand_pos, hand_quat = self._current_hand_pose()
            if tracked is None:
                tracked = hand_pos
            block_goal = np.asarray(current_cmd["pos"], dtype=np.float64).copy()
            hand_goal = self._hand_from_block(
                block_goal, np.asarray(hand_quat, dtype=np.float64)
            )
            current_cmd["pos"] = hand_goal
            origin = self._segment_origin(current_cmd, block_goal)
            axis = current_cmd["insert_axis"]
            current_cmd["locked_lateral"] = self._lateral_offset_vec(
                np.asarray(tracked, dtype=np.float64), origin, axis
            )
            t_tracked = self._axial_coord(tracked, origin, axis)
            t_goal = self._axial_coord(hand_goal, origin, axis)
            delta_t = t_goal - t_tracked
            if abs(delta_t) > 1e-6:
                hand_goal = hand_goal + axis * (np.sign(delta_t) * 0.01)
                current_cmd["pos"] = hand_goal
                t_goal = self._axial_coord(hand_goal, origin, axis)
                delta_t = t_goal - t_tracked
            current_cmd["insert_sign"] = np.sign(delta_t) if abs(delta_t) > 1e-6 else -1.0
        elif current_cmd.get("x_only_insert") and tracked is not None:
            goal = current_cmd["pos"]
            if self._uses_axis_insert(current_cmd):
                origin = self._segment_origin(current_cmd, goal)
                axis = current_cmd["insert_axis"]
                current_cmd["locked_lateral"] = self._lateral_offset_vec(
                    tracked, origin, axis
                )
                t_tracked = self._axial_coord(tracked, origin, axis)
                t_goal = self._axial_coord(goal, origin, axis)
                delta_t = t_goal - t_tracked
                if abs(delta_t) > 1e-6:
                    current_cmd["insert_sign"] = np.sign(delta_t)
                elif current_cmd.get("post_contact_retreat"):
                    current_cmd["insert_sign"] = -1.0
                else:
                    current_cmd["insert_sign"] = 1.0
            else:
                current_cmd["locked_yz"] = tracked[1:3].copy()
                delta = float(goal[0] - tracked[0])
                current_cmd["insert_sign"] = np.sign(delta) if abs(delta) > 1e-6 else -1.0
        if current_cmd.get("z_only_lift") and tracked is not None:
            goal = current_cmd["pos"]
            current_cmd["locked_xy"] = tracked[0:2].copy()
            delta = float(goal[2] - tracked[2])
            current_cmd["lift_sign"] = np.sign(delta) if abs(delta) > 1e-6 else 1.0

    def _update_lateral_correction(
        self, tracked: np.ndarray, origin: np.ndarray, axis: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        """Integrate measured lateral error to cancel steady-state IK tracking bias."""
        goal_lat = self._lateral_offset_vec(goal, origin, axis)
        measured_lat = self._lateral_offset_vec(tracked, origin, axis)
        err = goal_lat - measured_lat
        # Deadband: once the block is essentially on the line, stop integrating so
        # plant lag can't wind the correction past the target (overshoot).
        if float(np.linalg.norm(err)) <= self._lateral_feedback_deadband:
            return goal_lat + self._align_lat_correction
        self._align_lat_correction = (
            self._align_lat_correction + self._lateral_feedback_gain * err
        )
        norm = float(np.linalg.norm(self._align_lat_correction))
        if norm > self._max_lateral_correction:
            self._align_lat_correction *= self._max_lateral_correction / norm
        return goal_lat + self._align_lat_correction

    def _subgoal_block(self, current_cmd: dict, tracked: np.ndarray) -> np.ndarray:
        goal = current_cmd["pos"]
        if current_cmd.get("align_yz_only"):
            if self._uses_axis_insert(current_cmd):
                origin = self._segment_origin(current_cmd, goal)
                axis = current_cmd["insert_axis"]
                axial = self._axial_coord(tracked, origin, axis)
                cmd_lat = self._update_lateral_correction(tracked, origin, axis, goal)
                return self._pos_from_axial_lateral(origin, axis, axial, cmd_lat)
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
            if self._uses_axis_insert(current_cmd):
                origin = self._segment_origin(current_cmd, goal)
                axis = current_cmd["insert_axis"]
                if current_cmd.get("post_contact_retreat"):
                    # Pull straight back from the release pose; keep the lateral
                    # line fixed at the point where the hand was when retreat began.
                    locked_lat = current_cmd.get(
                        "locked_lateral",
                        self._lateral_offset_vec(tracked, origin, axis),
                    )
                else:
                    locked_lat = (
                        self._lateral_offset_vec(goal, origin, axis)
                        + self._align_lat_correction
                    )
                if self._axis_insert_reached(tracked, goal, tol, current_cmd):
                    t_goal = self._axial_coord(goal, origin, axis)
                    return self._pos_from_axial_lateral(origin, axis, t_goal, locked_lat)
                step = self._effective_insert_step(
                    current_cmd, current_cmd.get("cartesian_step") or self._cartesian_step
                )
                sign = float(current_cmd.get("insert_sign", 1.0))
                t = self._axial_coord(tracked, origin, axis)
                t_goal = self._axial_coord(goal, origin, axis)
                t_new = t + sign * min(step, abs(t_goal - t))
                if sign < 0:
                    t_new = max(t_new, t_goal)
                else:
                    t_new = min(t_new, t_goal)
                wiggle_lat = self._wiggle_lateral(current_cmd, locked_lat)
                return self._pos_from_axial_lateral(origin, axis, t_new, wiggle_lat)
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
        tracked = self._block_for_command(current_cmd, current_tracked_position)
        goal = current_cmd["pos"]
        tol = self._ik_tolerance(current_cmd)

        if current_cmd.get("align_yz_only"):
            if self._uses_axis_insert(current_cmd):
                ok = self._lateral_error(tracked, goal, current_cmd) < tol
            else:
                ok = (
                    abs(tracked[1] - goal[1]) < tol
                    and abs(tracked[2] - goal[2]) < tol
                )
        elif current_cmd.get("align_y_only"):
            ok = abs(tracked[1] - goal[1]) < tol
        elif current_cmd.get("align_z_only"):
            ok = abs(tracked[2] - goal[2]) < tol
        elif current_cmd.get("x_only_insert"):
            if self._uses_axis_insert(current_cmd):
                ok = self._axis_insert_reached(tracked, goal, tol, current_cmd)
            else:
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
        settle_frames = current_cmd.get("settle_frames") or self._settle_frames
        return current_cmd["settle_count"] >= settle_frames

    def _forward_closed_loop(
        self,
        current_cmd: dict,
        current_joint_positions: np.ndarray,
        current_tracked_position: typing.Optional[np.ndarray],
    ) -> ArticulationAction:
        n_dof = current_joint_positions.shape[0]

        contact_stop = self._stop_on_insert_contact(current_cmd, current_joint_positions)
        if contact_stop is not None:
            return contact_stop

        tracked = self._block_for_command(current_cmd, current_tracked_position)
        goal = current_cmd["pos"]
        tol = self._ik_tolerance(current_cmd)

        subgoal = self._subgoal_block(current_cmd, tracked)

        ori_cmd = self._command_orientation(current_cmd)
        ori_tol = current_cmd.get("orientation_tolerance") or self._insert_ori_tolerance
        warm_start = self._warm_start_for_command(current_cmd)
        if current_cmd.get("post_contact_retreat"):
            hand_pos = np.asarray(subgoal, dtype=np.float64)
        else:
            hand_pos = self._hand_goal_from_command(current_cmd, subgoal, ori_cmd)
        ik_joints, success = self._lula_kinematics.compute_inverse_kinematics(
            self._ee_frame,
            hand_pos,
            ori_cmd,
            warm_start,
            position_tolerance=tol,
            orientation_tolerance=ori_tol,
        )
        if ik_joints is None or not np.all(np.isfinite(np.asarray(ik_joints, dtype=np.float64))):
            action = None
        else:
            ik_arr = np.asarray(ik_joints, dtype=np.float64)
            action = self._joints_view.make_articulation_action(ik_arr, None)

        if action is None:
            if current_cmd.get("post_contact_retreat"):
                if self._last_arm_action is not None:
                    return self._merge_gripper_open(self._last_arm_action, n_dof)
                return self._merge_gripper_open(
                    self._hold_arm_joints_action(current_joint_positions), n_dof
                )
            return self._hold_action(n_dof)

        max_fk_err = (
            self._insert_max_ik_error
            if current_cmd.get("x_only_insert") or current_cmd.get("z_only_lift")
            else self._max_ik_error
        )
        if (
            not current_cmd.get("post_contact_retreat")
            and not success
            and action.joint_positions is not None
        ):
            fk_pos, _ = self._lula_kinematics.compute_forward_kinematics(
                self._ee_frame, np.asarray(action.joint_positions, dtype=np.float64)
            )
            hand_target = self._hand_goal_from_command(current_cmd, subgoal, ori_cmd)
            if float(np.linalg.norm(fk_pos - hand_target)) > max_fk_err:
                return self._hold_action(n_dof)

        current_cmd["frames_spent"] += 1
        self._update_settle_count(current_cmd, current_tracked_position)

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
            locked_yz = self._locked_yz_for_debug(current_cmd, goal)
            dbg["x_err"] = float(tracked[0] - goal[0])
            y_drift, z_drift, yz_drift = self._yz_error_components(
                tracked, np.array([goal[0], locked_yz[0], locked_yz[1]])
            )
            dbg["y_drift"] = y_drift
            dbg["z_drift"] = z_drift
            dbg["yz_drift"] = yz_drift
        else:
            y_err, z_err, yz_err = self._yz_error_components(tracked, goal)
            dbg["x_err"] = float(tracked[0] - goal[0])
            dbg["y_err"] = y_err
            dbg["z_err"] = z_err
            dbg["yz_err"] = yz_err
        self._traj_debug = dbg

        if self._closed_loop_segment_done(current_cmd, current_tracked_position):
            if current_cmd.get("x_only_insert") and current_cmd.get("track_block"):
                self._save_post_contact_arm_joints(current_joint_positions)
            elif current_cmd.get("post_contact_retreat"):
                self._post_contact_arm_joints = None
            self._cache_arm_action(action)
            self._current_command_index += 1
            self._clear_segment_playback()
            if current_cmd.get("post_contact_retreat"):
                return self._merge_gripper_open(action, n_dof)
            return action

        if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
            if mode == "z_lift":
                locked_xy = current_cmd.get("locked_xy", goal[0:2])
                carb.log_warn(
                    f"Closed-loop waypoint timed out ({mode}): "
                    f"z_err={tracked[2] - goal[2]:.4f} m, "
                    f"xy_err={self._xy_error(tracked, locked_xy):.4f} m"
                )
            elif mode == "x_insert":
                locked_yz = self._locked_yz_for_debug(current_cmd, goal)
                y_drift, z_drift, yz_drift = self._yz_error_components(
                    tracked, np.array([goal[0], locked_yz[0], locked_yz[1]])
                )
                carb.log_warn(
                    f"Closed-loop waypoint timed out ({mode}): "
                    f"x_err={tracked[0] - goal[0]:.4f} m, "
                    f"y_drift={y_drift:.4f} m, "
                    f"z_drift={z_drift:.4f} m, "
                    f"yz_drift={yz_drift:.4f} m"
                )
                if current_cmd.get("track_block"):
                    self._save_post_contact_arm_joints(current_joint_positions)
            else:
                y_err, z_err, yz_err = self._yz_error_components(tracked, goal)
                carb.log_warn(
                    f"Closed-loop waypoint timed out ({mode}): "
                    f"x_err={tracked[0] - goal[0]:.4f} m, "
                    f"y_err={y_err:.4f} m, "
                    f"z_err={z_err:.4f} m, "
                    f"yz_err={yz_err:.4f} m"
                )
            self._current_command_index += 1
            self._clear_segment_playback()
            return self.forward(current_joint_positions, current_tracked_position)

        self._cache_arm_action(action)
        if current_cmd.get("post_contact_retreat"):
            return self._merge_gripper_open(action, n_dof)
        return action

    def _save_post_contact_arm_joints(self, current_joint_positions: np.ndarray) -> None:
        self._post_contact_arm_joints = np.asarray(
            current_joint_positions[:7], dtype=np.float64
        ).copy()

    def _build_single_ik_action(self, current_cmd: dict) -> bool:
        warm_start = self._joints_view.get_joint_positions()
        ori_cmd = self._command_orientation(current_cmd)
        action, success = self._solve_ik_to_block(
            current_cmd["pos"],
            ori_cmd,
            warm_start,
            self._ik_tolerance(current_cmd),
            current_cmd=current_cmd,
        )
        if action is None:
            carb.log_warn("IK failed for cartesian waypoint.")
            return False
        if not success:
            carb.log_info("IK returned an approximate solution for cartesian waypoint.")
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
        goal_ori = self._command_orientation(current_cmd)
        step = current_cmd.get("cartesian_step") or self._cartesian_step
        if current_cmd.get("target_hand_at_position"):
            hand_start_pos, hand_start_quat = self._current_hand_pose()
            hand_goal_pos = np.asarray(goal_block, dtype=np.float64)
            hand_positions = np.stack([hand_start_pos, hand_goal_pos], axis=0)
            orientations = np.stack([hand_start_quat, goal_ori], axis=0)
        else:
            start_block = self._block_from_tracked_or_fk(current_tracked_position)

            if current_cmd["linear"]:
                block_points = self._sample_block_positions_linear(
                    start_block, goal_block, current_cmd.get("x_only_insert", False), step
                )
                orientations = np.tile(goal_ori.reshape(1, 4), (block_points.shape[0], 1))
                hand_positions = np.array(
                    [
                        self._hand_goal_from_command(current_cmd, p, orientations[i])
                        for i, p in enumerate(block_points)
                    ],
                    dtype=np.float64,
                )
            else:
                hand_start_pos, hand_start_quat = self._current_hand_pose()
                hand_goal_pos = self._hand_goal_from_command(
                    current_cmd, goal_block, goal_ori
                )
                hand_positions = np.stack([hand_start_pos, hand_goal_pos], axis=0)
                orientations = np.stack([hand_start_quat, goal_ori], axis=0)

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
            start_block = self._block_for_command(current_cmd, current_tracked_position)
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
        goal_hand = self._hand_goal_from_command(current_cmd, goal, current_cmd["ori"])
        return float(np.linalg.norm(goal_hand - hand_pos)) < tolerance

    def _finalize_cartesian_action(
        self, action: ArticulationAction, current_cmd: dict, n_dof: int
    ) -> ArticulationAction:
        if current_cmd.get("keep_gripper_open") or current_cmd.get(
            "post_contact_retreat"
        ):
            return self._merge_gripper_open(action, n_dof)
        return action

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
        if current_joint_positions is None:
            warm = self._joints_view.get_joint_positions()
            if warm is not None:
                if hasattr(warm, "cpu"):
                    warm = warm.cpu().numpy()
                n_dof = int(np.asarray(warm).reshape(-1).shape[0])
            else:
                n_dof = 9
            return self._hold_action(n_dof)

        joints = np.asarray(current_joint_positions, dtype=np.float64)
        if joints.ndim > 1:
            joints = joints.reshape(-1)
        n_dof = joints.shape[0]

        if self.is_done():
            return self._hold_action(n_dof)

        current_cmd = self._command_queue[self._current_command_index]

        if current_cmd["type"] == "gripper":
            if current_cmd["frames_spent"] == 0:
                self._clear_segment_playback()
            if current_cmd.get("full_open") and current_cmd["action"] == "open":
                gripper_action = ArticulationAction(
                    joint_positions=[None] * n_dof
                )
            else:
                gripper_action = self._gripper.forward(action=current_cmd["action"])
            action = self._gripper_with_frozen_arm(
                current_cmd, gripper_action, joints
            )
            current_cmd["frames_spent"] += 1
            if current_cmd["frames_spent"] >= current_cmd["max_frames"]:
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(joints, current_tracked_position)
            return action

        if current_cmd["type"] == "cartesian":
            if current_cmd.get("closed_loop"):
                if not self._segment_ready:
                    self._build_segment_actions(current_cmd, current_tracked_position)
                    self._segment_ready = True
                return self._forward_closed_loop(
                    current_cmd, joints, current_tracked_position
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
                self._cache_arm_action(action)
                return self._finalize_cartesian_action(action, current_cmd, n_dof)

            current_cmd["frames_spent"] += 1
            if self._segment_goal_reached(current_cmd, current_tracked_position) or (
                current_cmd["frames_spent"] >= current_cmd["max_frames"]
            ):
                self._current_command_index += 1
                self._clear_segment_playback()
                return self.forward(joints, current_tracked_position)

            return self._finalize_cartesian_action(
                self._current_segment_action(n_dof), current_cmd, n_dof
            )

        return self._hold_action(n_dof)

    def reset(self) -> None:
        BaseController.reset(self)
        self._current_command_index = 0
        self._clear_segment_playback()
        self._segment_ready = False
        self._traj_debug = None
        self._last_arm_action = None
        self._post_contact_arm_joints = None
        self.reset_grasp_calibration()
        for cmd in self._command_queue:
            cmd["frames_spent"] = 0
            if cmd["type"] == "gripper":
                cmd.pop("arm_hold_joints", None)
            if cmd["type"] == "cartesian":
                cmd["settle_count"] = 0
                if "goal_pos" in cmd:
                    cmd["pos"] = np.asarray(cmd["goal_pos"], dtype=np.float64).copy()
                if "goal_ori" in cmd:
                    cmd["ori"] = np.asarray(cmd["goal_ori"], dtype=np.float64).copy()
                cmd.pop("insert_sign", None)
                cmd.pop("lift_sign", None)
                cmd.pop("locked_xy", None)
                cmd.pop("locked_yz", None)
                cmd.pop("locked_lateral", None)
                cmd.pop("arm_hold_joints", None)

    def is_done(self) -> bool:
        return self._current_command_index >= len(self._command_queue)