"""Unit-aware strict controller used only by the six-arm demo."""

from __future__ import annotations

import numpy as np

from franka_motion_controller import FrankaMotionController
from ur5e_6x_cable_insertions import config as cfg
from ur5e_6x_cable_insertions.runtime_support import (
    meters_to_stage,
    stage_to_meters,
    validate_meters_per_unit,
)


class Ur5eSixArmMotionController(FrankaMotionController):
    def __init__(self, *args, meters_per_unit: float, **kwargs):
        self._meters_per_unit = validate_meters_per_unit(meters_per_unit)
        self._six_arm_failure_reason = ""
        super().__init__(*args, **kwargs)

    def clear_queue(self) -> None:
        super().clear_queue()
        self._six_arm_failure_reason = ""
        # Preserve last commanded joints from the arm lock so the next segment
        # does not rebuild targets from a gravity-sagged measurement.
        lock = getattr(self, "_lock_joint_positions", None)
        self._last_commanded_joint_positions = list(lock) if lock is not None else None

    def add_cartesian_waypoint(self, position, orientation, **kwargs):
        kwargs = dict(kwargs)
        # Positions are converted to stage units; tolerances must match.
        if "pos_tolerance" in kwargs and kwargs["pos_tolerance"] is not None:
            kwargs["pos_tolerance"] = (
                float(kwargs["pos_tolerance"]) / self._meters_per_unit
            )
        if kwargs.get("linear") and "linear_step" in kwargs:
            kwargs["linear_step"] = float(kwargs["linear_step"]) / self._meters_per_unit
        return super().add_cartesian_waypoint(
            meters_to_stage(position, self._meters_per_unit),
            orientation,
            **kwargs,
        )

    def add_pose_settle(
        self,
        position,
        *,
        tolerance_m: float,
        stable_frames: int,
        max_frames: int,
        label: str,
    ) -> None:
        """Hold the prior drive target until measured FK is stably aligned."""

        self._command_queue.append({
            "type": "pose_settle",
            "pos": meters_to_stage(position, self._meters_per_unit),
            "tolerance_m": float(tolerance_m),
            "stable_frames": max(1, int(stable_frames)),
            "max_frames": max(1, int(max_frames)),
            "frames_spent": 0,
            "stable_count": 0,
            "label": str(label),
        })

    def add_joint_waypoint(
        self,
        arm_positions,
        *,
        joint_steps: int = 240,
        open_gripper: bool = False,
        hold_gripper: bool = False,
        label: str = "",
    ) -> None:
        """Queue a direct named-arm interpolation.

        ``hold_gripper`` keeps fingers at the closed command (carry cable).
        ``open_gripper`` forces the open command (mutually exclusive in practice).
        """

        target = np.asarray(arm_positions, dtype=np.float64).reshape(-1)
        if target.shape != (len(cfg.UR5E_ARM_JOINT_NAMES),):
            raise ValueError("arm_positions must match UR5E_ARM_JOINT_NAMES")
        self._command_queue.append({
            "type": "joint_waypoint",
            "arm_positions": target,
            "joint_steps": max(1, int(joint_steps)),
            "step": 0,
            "start": None,
            "open_gripper": bool(open_gripper),
            "hold_gripper": bool(hold_gripper),
            "label": str(label),
        })

    def current_hand_pose_meters(self):
        position, orientation = super()._current_hand_pose()
        return stage_to_meters(position, self._meters_per_unit), orientation

    def _named_gripper_indices(self, count: int, n_dof: int) -> list[int]:
        robot_names = list(getattr(self._robot, "dof_names", []) or [])
        gripper = getattr(self, "_gripper", None)
        gripper_names = list(
            getattr(
                gripper,
                "joint_prim_names",
                getattr(gripper, "_joint_prim_names", []),
            )
            or []
        )
        indices = [robot_names.index(name) for name in gripper_names if name in robot_names]
        if len(indices) >= count:
            return indices[:count]
        return list(range(max(0, n_dof - count), n_dof))

    def _configured_gripper_indices(self) -> list[int]:
        robot_names = list(getattr(self._robot, "dof_names", []) or [])
        gripper = getattr(self, "_gripper", None)
        gripper_names = list(
            getattr(
                gripper,
                "joint_prim_names",
                getattr(gripper, "_joint_prim_names", []),
            )
            or []
        )
        return [robot_names.index(name) for name in gripper_names if name in robot_names]

    def _configured_arm_indices(self) -> list[int]:
        robot_names = list(getattr(self._robot, "dof_names", []) or [])
        return [
            robot_names.index(name)
            for name in cfg.UR5E_ARM_JOINT_NAMES
            if name in robot_names
        ]

    def forward(self, current_joint_positions):
        n_dof = int(np.asarray(current_joint_positions).size)
        while not self.is_done():
            cmd = self._command_queue[self._current_command_index]
            extended = self._forward_extended_command(
                cmd, current_joint_positions, n_dof
            )
            if extended is not None:
                return extended
            break

        # Base forward may advance into pose_settle/joint_waypoint in the same tick;
        # FrankaMotionController delegates those via _forward_extended_command.
        action = super().forward(current_joint_positions)
        commanded = None if action is None else getattr(action, "joint_positions", None)
        if commanded is not None:
            current = np.asarray(current_joint_positions, dtype=np.float64)
            previous = getattr(self, "_last_commanded_joint_positions", None)
            arm_indices = self._configured_arm_indices()
            gripper_indices = self._configured_gripper_indices()
            controlled_indices = set(arm_indices + gripper_indices)
            if not controlled_indices:
                controlled_indices = set(range(current.size))
            if previous is None or len(previous) != current.size:
                targets = [
                    float(current[i]) if i in controlled_indices else None
                    for i in range(current.size)
                ]
            else:
                targets = list(previous)
            command_type = ""
            hold_arm = False
            if self._current_command_index < len(self._command_queue):
                cur = self._command_queue[self._current_command_index]
                command_type = str(cur.get("type", ""))
                hold_arm = bool(cur.get("hold_arm"))
            joint_indices = getattr(action, "joint_indices", None)
            if joint_indices is not None:
                indices = [int(index) for index in list(joint_indices)]
            else:
                count = min(len(commanded), current.size)
                # hold_arm gripper cmds return a full articulation vector — do not
                # remap through finger-only indices (that can drop arm targets).
                if command_type == "gripper" and not hold_arm and count < current.size:
                    indices = self._named_gripper_indices(count, current.size)
                else:
                    indices = list(range(count))
            for index, value in zip(indices, list(commanded)):
                if value is not None and index in controlled_indices:
                    targets[index] = float(value)
            action.joint_positions = targets
            if hasattr(action, "joint_indices"):
                action.joint_indices = None
            self._last_commanded_joint_positions = list(targets)
            # Keep idle/grasp freeze in sync with the last applied arm command.
            remember = getattr(self, "_remember_lock", None)
            if callable(remember) and command_type in (
                "gripper",
                "hold",
                "cartesian",
                "joint_waypoint",
            ):
                remember(targets)
        return action

    def _forward_extended_command(self, cmd, current_joint_positions, n_dof):
        """Handle pose_settle / joint_waypoint (also called from parent after advance)."""

        cmd_type = cmd.get("type")
        if cmd_type == "joint_waypoint":
            current = np.asarray(current_joint_positions, dtype=np.float64).reshape(-1)
            arm_indices = self._configured_arm_indices()
            if len(arm_indices) != len(cfg.UR5E_ARM_JOINT_NAMES):
                self._segment_failed = True
                self._six_arm_failure_reason = "UR5e arm joints unavailable for home waypoint"
                return self._hold_action(n_dof)
            if cmd["start"] is None:
                cmd["start"] = current[arm_indices].copy()
            cmd["step"] += 1
            t = min(1.0, float(cmd["step"]) / float(cmd["joint_steps"]))
            blend = t * t * (3.0 - 2.0 * t)
            arm_target = (
                (1.0 - blend) * np.asarray(cmd["start"], dtype=np.float64)
                + blend * np.asarray(cmd["arm_positions"], dtype=np.float64)
            )
            action = self._hold_action(n_dof)
            positions = list(action.joint_positions)
            for index, value in zip(arm_indices, arm_target):
                positions[index] = float(value)
            if cmd.get("open_gripper"):
                opened = np.asarray(
                    self._gripper.joint_opened_positions, dtype=np.float64
                ).reshape(-1)
                for finger_i, index in enumerate(self._configured_gripper_indices()):
                    if finger_i < opened.size:
                        positions[index] = float(opened[finger_i])
            elif cmd.get("hold_gripper"):
                closed = np.asarray(
                    self._gripper.joint_closed_positions, dtype=np.float64
                ).reshape(-1)
                for finger_i, index in enumerate(self._configured_gripper_indices()):
                    if finger_i < closed.size:
                        positions[index] = float(closed[finger_i])
            action.joint_positions = positions
            if t >= 1.0:
                self._advance_command()
            return action

        if cmd_type != "pose_settle":
            return None

        cmd["frames_spent"] += 1
        hand_stage, _orientation = super()._current_hand_pose()
        error_m = float(
            np.linalg.norm(
                stage_to_meters(hand_stage - cmd["pos"], self._meters_per_unit)
            )
        )
        cmd["best_error_m"] = min(
            float(cmd.get("best_error_m", float("inf"))),
            error_m,
        )
        cmd["stable_count"] = (
            int(cmd["stable_count"]) + 1
            if error_m <= float(cmd["tolerance_m"])
            else 0
        )
        if int(cmd["stable_count"]) >= int(cmd["stable_frames"]):
            print(
                f"[Ur5eSixArm settle] REACHED [{cmd['label']}] "
                f"error={error_m:.4f}m frames={cmd['frames_spent']}"
            )
            self._advance_command()
            # Fall through so the next command can run this tick.
            if not self.is_done():
                nxt = self._command_queue[self._current_command_index]
                if nxt.get("type") in ("pose_settle", "joint_waypoint"):
                    return self._forward_extended_command(
                        nxt, current_joint_positions, n_dof
                    )
            return self._hold_action(n_dof)
        if int(cmd["frames_spent"]) >= int(cmd["max_frames"]):
            self._segment_failed = True
            self._six_arm_failure_reason = (
                f"Timed out aligning before grasp close: error={error_m:.4f}m "
                f"best={float(cmd['best_error_m']):.4f}m"
            )
            print(f"[Ur5eSixArm settle] FAILURE {self._six_arm_failure_reason}")
        return self._hold_action(n_dof)

    def _init_joint_interp_segment(self, cmd, current_joint_positions, n_dof):
        super()._init_joint_interp_segment(cmd, current_joint_positions, n_dof)
        if self._joint_interp_warned:
            self._segment_failed = True
            label = str(cmd.get("label", "unlabelled waypoint"))
            self._six_arm_failure_reason = f"IK failed for {label}"
            self._log_joint_limit_diagnostics(
                current_joint_positions,
                label=label,
                target_hand=getattr(self, "_segment_goal_hand", None),
                target_ori=cmd.get("ori"),
            )

    def _log_joint_limit_diagnostics(
        self,
        current_joint_positions,
        *,
        label: str,
        target_hand=None,
        target_ori=None,
    ) -> None:
        """Print arm DOF positions vs limits when Lula IK rejects a waypoint."""

        names = list(getattr(self._robot, "dof_names", []) or [])
        current = np.asarray(current_joint_positions, dtype=np.float64).reshape(-1)
        arm_indices = self._configured_arm_indices()
        limits = None
        for getter in ("get_dof_limits", "get_joint_limits", "dof_limits"):
            attr = getattr(self._robot, getter, None)
            try:
                limits = attr() if callable(attr) else attr
            except Exception:
                limits = None
            if limits is not None:
                break
        print(f"[IK FAIL] Lula IK found no solution for [{label}]")
        print(
            "  Cause: target hand pose+ori unreachable for this seed "
            "(not a confirmed joint-limit trip; limits API unavailable). "
            "q below is the last successful arm config."
        )
        if target_hand is not None:
            print(f"  target_hand_stage={np.round(np.asarray(target_hand), 4)}")
        if target_ori is not None:
            print(f"  target_ori={np.round(np.asarray(target_ori, dtype=np.float64), 4)}")
        rows = []
        for index in arm_indices:
            name = names[index] if index < len(names) else f"dof_{index}"
            q = float(current[index]) if index < current.size else float("nan")
            lo = hi = None
            if limits is not None:
                try:
                    pair = np.asarray(limits, dtype=np.float64).reshape(-1, 2)[index]
                    lo, hi = float(pair[0]), float(pair[1])
                except Exception:
                    lo = hi = None
            if lo is not None and hi is not None and np.isfinite([lo, hi, q]).all():
                margin = min(q - lo, hi - q)
                flag = " NEAR_LIMIT" if margin < np.deg2rad(5.0) else ""
                rows.append(
                    f"  {name}: q={np.rad2deg(q):+.1f}° "
                    f"lim=[{np.rad2deg(lo):+.1f},{np.rad2deg(hi):+.1f}]° "
                    f"margin={np.rad2deg(margin):.1f}°{flag}"
                )
            else:
                rows.append(f"  {name}: q={np.rad2deg(q):+.1f}° (limits unavailable)")
        print("\n".join(rows) if rows else "  (no arm DOFs resolved)")

    def has_failed(self) -> bool:
        return bool(self._segment_failed)

    def failure_reason(self) -> str:
        return self._six_arm_failure_reason
