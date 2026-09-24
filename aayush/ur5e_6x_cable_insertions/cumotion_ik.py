"""Minimal cuMotion IK for TipOffset→seat (Lula remains default elsewhere)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ur5e_6x_cable_insertions import config as cfg

_ROBOT = None
_LOAD_ERROR: Optional[str] = None


def cumotion_config_dir() -> Path:
    override = getattr(cfg, "CUMOTION_UR5E_CONFIG_DIR", None)
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "robot_configurations" / "ur5e"


def _ensure_robot():
    global _ROBOT, _LOAD_ERROR
    if _ROBOT is not None:
        return _ROBOT
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    try:
        from isaacsim.robot_motion.cumotion import load_cumotion_robot
    except Exception as exc:  # pragma: no cover - only fails outside Isaac
        _LOAD_ERROR = f"cuMotion extension unavailable: {exc}"
        raise RuntimeError(_LOAD_ERROR) from exc

    directory = cumotion_config_dir()
    try:
        _ROBOT = load_cumotion_robot(directory)
    except Exception as exc:
        _LOAD_ERROR = f"Failed to load cuMotion UR5e from {directory}: {exc}"
        raise RuntimeError(_LOAD_ERROR) from exc
    return _ROBOT


def _world_pose_to_base_pose(
    target_pos_m: np.ndarray,
    target_quat_wxyz: np.ndarray,
    base_pos_m: np.ndarray,
    base_quat_wxyz: np.ndarray,
):
    import cumotion

    def _pose(pos, quat):
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(q))
        if n > 1e-12:
            q = q / n
        return cumotion.Pose3(
            translation=p,
            rotation=cumotion.Rotation3(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        )

    return _pose(base_pos_m, base_quat_wxyz).inverse() * _pose(
        target_pos_m, target_quat_wxyz
    )


def solve_arm_ik(
    *,
    target_pos_m: np.ndarray,
    target_quat_wxyz: np.ndarray,
    base_pos_m: np.ndarray,
    base_quat_wxyz: np.ndarray,
    seed_arm_rad: np.ndarray,
    ee_frame: str,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> tuple[bool, Optional[np.ndarray], str]:
    """Solve 6-DOF arm IK in meters / radians.

    Returns ``(success, q_arm[6] or None, detail)``.
    """

    import cumotion

    robot = _ensure_robot()
    kinematics = robot.kinematics
    frame = str(ee_frame or cfg.UR5E_EE_FRAME)
    seed = np.asarray(seed_arm_rad, dtype=np.float64).reshape(-1)
    if seed.size != len(cfg.UR5E_ARM_JOINT_NAMES):
        return False, None, f"seed size {seed.size} != 6"

    target_pose = _world_pose_to_base_pose(
        target_pos_m, target_quat_wxyz, base_pos_m, base_quat_wxyz
    )
    ik_cfg = cumotion.IkConfig()
    ik_cfg.position_tolerance = float(max(1e-6, position_tolerance_m))
    ik_cfg.orientation_tolerance = float(max(1e-6, orientation_tolerance_rad))
    ik_cfg.cspace_seeds = [seed]
    ik_cfg.max_num_descents = 50

    try:
        result = cumotion.solve_ik(kinematics, target_pose, frame, ik_cfg)
    except Exception as exc:
        return False, None, f"solve_ik raised: {exc}"

    if not bool(result.success):
        return (
            False,
            None,
            (
                f"no solution (pos_err={float(result.position_error):.6f}m "
                f"frame={frame})"
            ),
        )

    q = np.asarray(result.cspace_position, dtype=np.float64).reshape(-1)
    # Map by controlled joint names in case XRDF order differs.
    names = list(robot.controlled_joint_names)
    if len(names) == q.size and names == list(cfg.UR5E_ARM_JOINT_NAMES):
        arm = q.copy()
    else:
        arm = np.zeros(len(cfg.UR5E_ARM_JOINT_NAMES), dtype=np.float64)
        name_to_q = {n: float(q[i]) for i, n in enumerate(names) if i < q.size}
        for i, name in enumerate(cfg.UR5E_ARM_JOINT_NAMES):
            if name not in name_to_q:
                return False, None, f"missing joint {name} in cuMotion solution"
            arm[i] = name_to_q[name]
    return True, arm, f"ok pos_err={float(result.position_error):.6e}m"
