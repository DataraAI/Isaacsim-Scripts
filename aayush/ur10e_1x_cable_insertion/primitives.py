"""BT primitives: detect E_part006_44, top-down grasp, and lift."""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom

from ur10e_1x_cable_insertion import config as cfg


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def prim_bbox(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = np.array(box.GetMin(), dtype=np.float64)
    maximum = np.array(box.GetMax(), dtype=np.float64)
    return minimum, maximum, 0.5 * (minimum + maximum)


def grasp_part_center(context) -> np.ndarray:
    stage = context.services["stage"]
    path = context.services.get("grasp_part_path", cfg.GRASP_PART_PATH)
    _mn, _mx, center = prim_bbox(stage, path)
    return center.copy()


def detect_grasp_part(context) -> bool:
    """Perceive E_part006_44 and cache its world center."""

    stage = context.services["stage"]
    path = context.services.get("grasp_part_path", cfg.GRASP_PART_PATH)
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print(f"[BT PERCEPTION] grasp part missing: {path}")
        return False
    center = grasp_part_center(context)
    block_top = float(context.services.get("block_top_z", 1.125))
    if center[2] < block_top - 0.25 or center[2] > block_top + 0.35:
        print(f"[BT PERCEPTION] REJECT bad detection center={np.round(center, 4)}")
        return False
    context.services["detected_grasp_point"] = center.copy()
    context.services["live_grasp_point"] = center.copy()
    context.blackboard.add("target_visible")
    print(f"[BT PERCEPTION] E_part006_44 @ {np.round(center, 4)}")
    return True


def _hand_above_tip(tip: np.ndarray) -> np.ndarray:
    """Hand pose for fingers-down tool: tip is TOOL_OFFSET below the Lula ee."""

    hand = np.asarray(tip, dtype=np.float64).copy()
    hand[2] += float(cfg.TOOL_OFFSET_M)
    return hand


def _add_waypoint(controller, position, orientation, **kwargs) -> None:
    kwargs.setdefault("joint_interp", True)
    kwargs.setdefault("target_is_hand", True)
    kwargs.setdefault("max_frames", 900)
    controller.add_cartesian_waypoint(
        np.asarray(position, dtype=np.float64),
        _normalize_quat(orientation),
        **kwargs,
    )


def queue_move(context) -> None:
    controller = context.services["motion_controller"]
    raw = context.step.inputs.get("position", cfg.OBSERVE_HAND)
    position = np.asarray(raw, dtype=np.float64).reshape(3)
    orientation = _normalize_quat(
        np.asarray(context.step.inputs.get("orientation_wxyz", cfg.GRASP_ORIENTATION), dtype=np.float64)
    )
    _add_waypoint(
        controller,
        position,
        orientation,
        label=f"{context.step.name}: observe",
        joint_steps=200,
        pos_tolerance=0.03,
    )


def queue_grasp(context) -> None:
    """Hover above E_part006_44 → open → descend → close → lift."""

    controller = context.services["motion_controller"]
    orientation = _normalize_quat(cfg.GRASP_ORIENTATION)
    point = np.asarray(
        context.services.get("live_grasp_point", grasp_part_center(context)),
        dtype=np.float64,
    ).reshape(3)
    context.services["live_grasp_point"] = point.copy()

    tip_grasp = point.copy()
    tip_grasp[0] += float(cfg.GRASP_X_OFFSET_M)
    tip_grasp[2] += float(cfg.GRASP_DESCEND_CLEARANCE_M)
    tip_hover = tip_grasp.copy()
    tip_hover[2] += float(cfg.GRASP_HOVER_CLEARANCE_M)

    hand_hover = _hand_above_tip(tip_hover)
    hand_grasp = _hand_above_tip(tip_grasp)
    hand_lift = hand_grasp.copy()
    hand_lift[2] += float(cfg.GRASP_LIFT_CLEARANCE_M) + abs(float(cfg.GRASP_DESCEND_CLEARANCE_M))

    print(
        f"[BT GRASP] top-down part={np.round(point, 4)} "
        f"x_offset={cfg.GRASP_X_OFFSET_M:+.4f}\n"
        f"  tip_hover={np.round(tip_hover, 4)} tip_grasp={np.round(tip_grasp, 4)}\n"
        f"  hand_hover={np.round(hand_hover, 4)} hand_grasp={np.round(hand_grasp, 4)} "
        f"hand_lift={np.round(hand_lift, 4)}"
    )

    _add_waypoint(
        controller, hand_hover, orientation,
        label=f"{context.step.name}: hover-above", joint_steps=240, pos_tolerance=0.025,
    )
    controller.add_gripper_command(action="open", wait_frames=50)
    _add_waypoint(
        controller, hand_grasp, orientation,
        label=f"{context.step.name}: descend", joint_steps=200, pos_tolerance=0.008,
    )
    # Close firmly, dwell while clamped, then lift with hold_gripper so fingers
    # cannot ease open during the joint-space ascent.
    controller.add_gripper_command(action="close", wait_frames=int(cfg.GRASP_CLOSE_WAIT_FRAMES))
    _add_waypoint(
        controller, hand_grasp, orientation,
        label=f"{context.step.name}: squeeze-hold",
        hold_gripper=True,
        joint_steps=max(20, int(cfg.GRASP_SQUEEZE_HOLD_FRAMES)),
        pos_tolerance=0.01,
    )
    _add_waypoint(
        controller, hand_lift, orientation,
        label=f"{context.step.name}: lift",
        hold_gripper=True,
        joint_steps=200,
        pos_tolerance=0.02,
    )


def check_physical_grasp(context) -> bool:
    stage = context.services["stage"]
    path = context.services.get("grasp_part_path", cfg.GRASP_PART_PATH)
    try:
        _mn, _mx, center = prim_bbox(stage, path)
    except Exception as exc:
        print(f"[BT GRASP] validate bbox failed: {exc}")
        return False

    block_top = float(context.services.get("block_top_z", 1.125))
    lifted = float(center[2]) >= block_top + 0.04

    fingers = None
    try:
        fingers = np.asarray(
            context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
        ).reshape(-1)
    except Exception:
        pass
    closed_enough = True
    if fingers is not None and fingers.size:
        closed_enough = float(np.max(np.abs(fingers))) >= cfg.ROBOTIQ_CONTACT_RAD

    passed = bool(lifted and closed_enough)
    if passed:
        context.blackboard.add("cable_held")
    print(
        f"[BT GRASP] validate center={np.round(center, 4)} lifted={lifted} "
        f"closed={closed_enough} fingers={None if fingers is None else np.round(fingers, 3)} "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def inspect_workspace(context) -> bool:
    robot = context.services["robot"]
    jp = robot.get_joint_positions()
    ok = jp is not None
    print(f"[BT INSPECT] robot_state_valid={ok}")
    return bool(ok)
