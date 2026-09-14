"""Behaviour-tree motion primitives for the six UR5e cable-insertion cells."""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom

from insertion_features.cache import world_features_for_head, world_features_for_jack
from ur5e_6x_cable_insertions import config as cfg
from ur5e_6x_cable_insertions.alignment import (
    clamp_nudge,
    evaluate_alignment,
    insert_target_tip,
    mating_gap_along_axis,
)
from ur5e_6x_cable_insertions.runtime_support import (
    grasp_tip_from_part,
    physical_grasp_is_valid,
)


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def _normalize_vec(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-9 else np.zeros(3, dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = _normalize_quat(a)
    bw, bx, by, bz = _normalize_quat(b)
    return _normalize_quat(
        np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )
    )


def _yaw_about_world_z(quat_wxyz: np.ndarray, yaw_deg: float) -> np.ndarray:
    half = 0.5 * np.deg2rad(float(yaw_deg))
    return _quat_multiply(
        np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64),
        quat_wxyz,
    )


def _apply_world_rotvec(quat_wxyz: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return _normalize_quat(quat_wxyz)
    axis = vector / angle
    delta = np.concatenate(([np.cos(0.5 * angle)], axis * np.sin(0.5 * angle)))
    return _quat_multiply(delta, quat_wxyz)


def _quat_to_rot_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat(quat_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _hand_from_tip(tip: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray(tip, dtype=np.float64).reshape(3) - _quat_to_rot_matrix(
        orientation_wxyz
    ) @ np.array([0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64)


def _current_tip(context) -> tuple[np.ndarray, np.ndarray]:
    hand, orientation = context.services["motion_controller"].current_hand_pose_meters()
    quat = _normalize_quat(orientation)
    tip = np.asarray(hand, dtype=np.float64).reshape(3) + _quat_to_rot_matrix(quat) @ np.array(
        [0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64
    )
    return tip, quat


def _add_tip_waypoint(context, tip, orientation, *, label: str, **kwargs) -> None:
    tip_value = np.asarray(tip, dtype=np.float64).reshape(3)
    context.services.setdefault("ik_tip_by_label", {})[label] = tip_value.copy()
    context.services["motion_controller"].add_cartesian_waypoint(
        _hand_from_tip(tip_value, orientation),
        _normalize_quat(orientation),
        target_is_hand=True,
        label=label,
        **kwargs,
    )


def _station_tag(context) -> str:
    station_id = context.services.get("station_id")
    return f" {station_id}" if station_id else ""


def meters_per_unit(stage) -> float:
    """Return the USD stage scale in meters per authored unit."""

    try:
        return float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        return 1.0


def prim_bbox(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a prim's world AABB in meters."""

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    scale = meters_per_unit(stage)
    minimum = np.asarray(box.GetMin(), dtype=np.float64) * scale
    maximum = np.asarray(box.GetMax(), dtype=np.float64) * scale
    return minimum, maximum, 0.5 * (minimum + maximum)


def _world_from_prim(stage, prim_path: str, *, translation_in_meters: bool) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing transform prim: {prim_path}")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    transform = np.asarray(matrix, dtype=np.float64).T
    if translation_in_meters:
        transform[:3, 3] *= meters_per_unit(stage)
    return transform


def _live_features(context):
    stage = context.services["stage"]
    world_from_head = _world_from_prim(
        stage, str(context.services["path45"]), translation_in_meters=True
    )
    world_from_pack = _world_from_prim(
        stage, str(context.services["port_pack_path"]), translation_in_meters=False
    )
    crystal = world_features_for_head(cfg.HEAD45_NAME, world_from_head)
    port = world_features_for_jack(
        str(context.services["jack_id"]),
        world_from_pack,
        meters_per_unit=meters_per_unit(stage),
    )
    return crystal, port


def _grasp_part_path(context) -> str:
    path = context.services.get("grasp_part_path")
    if not path:
        raise RuntimeError("services['grasp_part_path'] is required")
    return str(path)


def grasp_part_center(context) -> np.ndarray:
    _minimum, _maximum, center = prim_bbox(
        context.services["stage"], _grasp_part_path(context)
    )
    return center.copy()


def queue_move(context) -> None:
    raw = context.step.inputs.get("position")
    if raw is None:
        raw = context.services.get("observe_hand")
    if raw is None:
        raise RuntimeError("observation hand position is missing")
    context.services["motion_controller"].add_cartesian_waypoint(
        np.asarray(raw, dtype=np.float64).reshape(3),
        _normalize_quat(context.step.inputs.get("orientation_wxyz", cfg.OBSERVE_ORIENTATION)),
        target_is_hand=True,
        joint_interp=True,
        joint_steps=200,
        max_frames=1200,
        pos_tolerance=0.03,
        label=f"{context.step.name}: observe",
    )


def detect_grasp_part(context) -> bool:
    path = _grasp_part_path(context)
    prim = context.services["stage"].GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print(f"[BT PERCEPTION{_station_tag(context)}] grasp part missing: {path}")
        return False
    center = grasp_part_center(context)
    block_top = float(context.services.get("block_top_z", center[2]))
    if not block_top - 0.25 <= float(center[2]) <= block_top + 0.35:
        print(f"[BT PERCEPTION{_station_tag(context)}] rejected center={center}")
        return False
    context.services["detected_grasp_point"] = center.copy()
    context.services["live_grasp_point"] = center.copy()
    context.services["initial_grasp_point"] = center.copy()
    context.blackboard.add("target_visible")
    return True


def queue_grasp(context) -> None:
    """Use the configured 30-degree tilt to hover, close physically, and lift."""

    controller = context.services["motion_controller"]
    approach = _normalize_vec(cfg.GRASP_APPROACH_DIR)
    orientation = _normalize_quat(cfg.GRASP_ORIENTATION)
    point = np.asarray(
        context.services.get("live_grasp_point", grasp_part_center(context)),
        dtype=np.float64,
    ).reshape(3)
    context.services["initial_grasp_point"] = point.copy()
    context.services["grasp_orientation"] = orientation.copy()

    grasp_tip = point + np.array(
        [
            float(cfg.GRASP_X_OFFSET_M),
            float(cfg.GRASP_Y_ALIGNMENT_OFFSET_M),
            float(cfg.GRASP_DESCEND_CLEARANCE_M),
        ],
        dtype=np.float64,
    )
    hover_tip = grasp_tip - approach * float(cfg.GRASP_HOVER_CLEARANCE_M)
    lift_tip = grasp_tip + np.array(
        [
            0.0,
            0.0,
            float(cfg.GRASP_LIFT_CLEARANCE_M)
            + abs(float(cfg.GRASP_DESCEND_CLEARANCE_M)),
        ],
        dtype=np.float64,
    )

    controller.add_gripper_command(action="open", wait_frames=50)
    _add_tip_waypoint(
        context,
        hover_tip,
        orientation,
        label=f"{context.step.name}: hover-{cfg.GRASP_TILT_FROM_DOWN_DEG:.0f}deg",
        joint_interp=True,
        joint_steps=240,
        max_frames=1200,
        pos_tolerance=0.025,
    )
    _add_tip_waypoint(
        context,
        grasp_tip,
        orientation,
        label=f"{context.step.name}: descend",
        joint_interp=True,
        joint_steps=200,
        max_frames=1200,
        pos_tolerance=0.008,
    )
    controller.add_pose_settle(
        _hand_from_tip(grasp_tip, orientation),
        tolerance_m=0.04,
        stable_frames=15,
        max_frames=300,
        label=f"{context.step.name}: settle",
    )
    controller.add_gripper_command(
        action="close", wait_frames=int(cfg.GRASP_CLOSE_WAIT_FRAMES)
    )
    context.services["monitor_cable_hold"] = True
    _add_tip_waypoint(
        context,
        grasp_tip,
        orientation,
        label=f"{context.step.name}: squeeze",
        hold_gripper=True,
        joint_interp=True,
        joint_steps=max(20, int(cfg.GRASP_SQUEEZE_HOLD_FRAMES)),
        max_frames=600,
        pos_tolerance=0.01,
    )
    _add_tip_waypoint(
        context,
        lift_tip,
        orientation,
        label=f"{context.step.name}: lift",
        hold_gripper=True,
        joint_interp=True,
        joint_steps=200,
        max_frames=1200,
        pos_tolerance=0.02,
    )


def check_physical_grasp(context) -> bool:
    try:
        center = grasp_part_center(context)
        initial = np.asarray(
            context.services["initial_grasp_point"], dtype=np.float64
        ).reshape(3)
        expected_tip, _orientation = _current_tip(context)
        fingers = np.asarray(
            context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
        ).reshape(-1)
    except Exception as exc:
        print(f"[BT GRASP{_station_tag(context)}] validation failed: {exc}")
        return False
    actual_tip = grasp_tip_from_part(center, cfg.GRASP_X_OFFSET_M)
    actual_tip[1] += float(cfg.GRASP_Y_ALIGNMENT_OFFSET_M)
    passed = physical_grasp_is_valid(
        initial_part=initial,
        current_part=actual_tip,
        expected_tip=expected_tip,
        fingers=fingers,
        min_lift_m=cfg.GRASP_MIN_LIFT_M,
        max_tip_error_m=cfg.CABLE_IN_GRIPPER_MAX_ERR_M,
        contact_rad=cfg.ROBOTIQ_CONTACT_RAD,
    )
    if passed:
        context.blackboard.add("cable_held")
    return bool(passed)


def _find_named_descendant(stage, root_path: str, name: str) -> str | None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    wanted = name.lower()
    for prim in Usd.PrimRange(root):
        if prim.GetName().lower() == wanted:
            return str(prim.GetPath())
    return None


def compute_port_targets(
    stage, contacts_path: str | None = None
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Resolve temporary pin-based insert and +X approach targets."""

    if not contacts_path or not stage.GetPrimAtPath(contacts_path).IsValid():
        raise RuntimeError(f"Port contacts missing: {contacts_path}")
    path_a = _find_named_descendant(stage, contacts_path, cfg.PORT_PIN_A_NAME)
    path_b = _find_named_descendant(stage, contacts_path, cfg.PORT_PIN_B_NAME)
    if not path_a or not path_b:
        raise RuntimeError(f"Missing target pins under {contacts_path}")
    _amin, _amax, center_a = prim_bbox(stage, path_a)
    _bmin, _bmax, center_b = prim_bbox(stage, path_b)
    insert = 0.5 * (center_a + center_b)
    approach = insert.copy()
    approach[0] += float(cfg.PORT_APPROACH_X_OFFSET_M)
    return insert, approach, path_a, path_b


def _port_tip_via(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    frac = float(np.clip(fraction, 0.0, 1.0))
    tip = (1.0 - frac) * start + frac * target
    if frac < 1.0:
        tip[2] = max(float(start[2]), float(target[2])) + float(
            cfg.PORT_APPROACH_VIA_Z_CLEARANCE_M
        )
    return tip


def queue_port_offset(context) -> None:
    """Reorient, then carry the held cable to the feature-based +X offset."""

    _crystal, port = _live_features(context)
    target_tip = np.asarray(port.mating_center, dtype=np.float64) + np.array(
        [1.0, 0.0, 0.0], dtype=np.float64
    ) * float(cfg.PORT_APPROACH_X_OFFSET_M)
    start_tip, start_orientation = _current_tip(context)
    end_orientation = _yaw_about_world_z(
        context.services.get("grasp_orientation", start_orientation),
        cfg.PORT_APPROACH_YAW_DEG,
    )
    context.services["port_insert_point"] = np.asarray(
        port.mating_center, dtype=np.float64
    ).copy()
    context.services["port_offset_point"] = target_tip.copy()
    context.services["port_approach_orientation"] = end_orientation.copy()
    context.services["monitor_cable_hold"] = True

    yaw_tip = start_tip.copy()
    yaw_tip[2] = max(float(start_tip[2]), float(target_tip[2])) + float(
        cfg.PORT_APPROACH_VIA_Z_CLEARANCE_M
    )
    yaw_steps = max(2, int(cfg.PORT_APPROACH_YAW_STEPS))
    for index in range(1, yaw_steps + 1):
        fraction = float(index) / float(yaw_steps)
        orientation = _yaw_about_world_z(
            context.services.get("grasp_orientation", start_orientation),
            fraction * float(cfg.PORT_APPROACH_YAW_DEG),
        )
        _add_tip_waypoint(
            context,
            yaw_tip,
            orientation,
            label=f"{context.step.name}: yaw-{index}",
            hold_gripper=True,
            joint_interp=True,
            joint_steps=max(60, int(360 / yaw_steps)),
            max_frames=900,
            pos_tolerance=0.035,
        )
    fractions = [
        float(value)
        for value in cfg.PORT_APPROACH_VIA_FRACTIONS
        if 0.0 < float(value) < 1.0
    ] + [1.0]
    for index, fraction in enumerate(fractions):
        _add_tip_waypoint(
            context,
            _port_tip_via(start_tip, target_tip, fraction),
            end_orientation,
            label=f"{context.step.name}: offset-{fraction:.2f}",
            hold_gripper=True,
            joint_interp=True,
            joint_steps=160 if index == len(fractions) - 1 else 120,
            max_frames=1200,
            pos_tolerance=(
                float(cfg.PORT_APPROACH_TOLERANCE_M)
                if index == len(fractions) - 1
                else 0.03
            ),
        )


def check_at_port_offset(context) -> bool:
    target = context.services.get("port_offset_point")
    if target is None:
        try:
            _crystal, port = _live_features(context)
            target = np.asarray(port.mating_center, dtype=np.float64) + np.array(
                [1.0, 0.0, 0.0], dtype=np.float64
            ) * float(cfg.PORT_APPROACH_X_OFFSET_M)
        except Exception:
            return False
    try:
        tip, _orientation = _current_tip(context)
    except Exception:
        return False
    at_offset = float(np.linalg.norm(tip - np.asarray(target))) <= float(
        cfg.PORT_APPROACH_TOLERANCE_M
    )
    held, _info = cable_still_in_gripper(context)
    if at_offset and held:
        context.blackboard.add("at_port_offset")
        context.blackboard.add("cable_held")
        return True
    return False


def queue_align_and_insert(context) -> None:
    """Seed the incremental loop; subsequent commands are queued by its tick hook."""

    context.services["_align_insert_frames"] = 0
    context.services["monitor_cable_hold"] = True
    context.services["motion_controller"].add_hold_command(
        wait_frames=1,
        hold_gripper=True,
        label=f"{context.step.name}: align-start",
    )


def _abort_align_insert(context, reason: str, *, clear_queue: bool = True) -> None:
    context.services["abort_simulation"] = True
    context.services["abort_reason"] = reason
    context.blackboard.discard("at_port_insert")
    if clear_queue:
        context.services["motion_controller"].clear_queue()
    print(f"[BT ALIGN{_station_tag(context)}] FAILURE: {reason}")


def tick_align_and_insert(context):
    """while_running hook: interleaved align nudges and insert micro-steps."""

    from behaviour_tree_insertion.runtime import Status

    hold_status = monitor_cable_hold(context)
    if hold_status is Status.FAILURE or context.services.get("abort_simulation"):
        return Status.FAILURE

    controller = context.services["motion_controller"]
    frames = int(context.services.get("_align_insert_frames", 0)) + 1
    context.services["_align_insert_frames"] = frames
    if frames > int(cfg.ALIGN_INSERT_MAX_FRAMES):
        _abort_align_insert(context, "align/insert frame budget exceeded")
        return Status.FAILURE
    if controller.has_failed():
        _abort_align_insert(
            context,
            controller.failure_reason() or "IK failure during align/insert",
            clear_queue=False,
        )
        return Status.FAILURE
    if not controller.is_done():
        return Status.RUNNING

    try:
        crystal, port = _live_features(context)
        residual = evaluate_alignment(
            crystal,
            port,
            latch_z_margin_m=cfg.LATCH_Z_MARGIN_M,
            mating_side_margin_m=cfg.MATING_SIDE_MARGIN_M,
            axis_dot_min=cfg.AXIS_DOT_MIN,
        )
        current_tip, orientation = _current_tip(context)
        if not residual.passed:
            pos_nudge, rot_nudge = clamp_nudge(
                residual.pos_error_m,
                residual.rot_error_rad,
                max_pos_m=cfg.ALIGN_NUDGE_POS_MAX_M,
                max_rot_rad=cfg.ALIGN_NUDGE_ROT_MAX_RAD,
            )
            _add_tip_waypoint(
                context,
                current_tip + pos_nudge,
                _apply_world_rotvec(orientation, rot_nudge),
                label=f"{context.step.name}: align-{frames}",
                hold_gripper=True,
                joint_interp=True,
                joint_steps=30,
                max_frames=180,
                pos_tolerance=0.003,
            )
            return Status.RUNNING

        live_gap = mating_gap_along_axis(crystal, port)
        if abs(float(live_gap)) <= float(cfg.MATING_TOUCH_GAP_M):
            context.blackboard.add("at_port_insert")
            return Status.SUCCESS

        target_tip = insert_target_tip(
            current_tip,
            port.insertion_axis,
            cfg.INSERT_STEP_M,
        )
        _add_tip_waypoint(
            context,
            target_tip,
            orientation,
            label=f"{context.step.name}: insert-{frames}",
            hold_gripper=True,
            joint_interp=False,
            linear=True,
            linear_step=float(cfg.INSERT_STEP_M),
            max_frames=180,
            pos_tolerance=0.0015,
        )
        return Status.RUNNING
    except Exception as exc:
        _abort_align_insert(context, f"feature alignment failed: {exc}")
        return Status.FAILURE


def check_at_port_insert(context) -> bool:
    if "at_port_insert" not in context.blackboard:
        return False
    held, _info = cable_still_in_gripper(context)
    if not held:
        context.blackboard.discard("at_port_insert")
        context.blackboard.discard("cable_held")
        return False
    try:
        crystal, port = _live_features(context)
        gap = mating_gap_along_axis(crystal, port)
    except Exception:
        return False
    return abs(float(gap)) <= float(cfg.MATING_TOUCH_GAP_M)


def cable_still_in_gripper(context) -> tuple[bool, dict]:
    info: dict = {}
    try:
        expected_tip, _orientation = _current_tip(context)
        part = grasp_part_center(context)
        actual_tip = grasp_tip_from_part(part, cfg.GRASP_X_OFFSET_M)
        actual_tip[1] += float(cfg.GRASP_Y_ALIGNMENT_OFFSET_M)
        error = float(np.linalg.norm(actual_tip - expected_tip))
        fingers = np.asarray(
            context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
        ).reshape(-1)
        closed = not fingers.size or float(np.max(np.abs(fingers))) >= float(
            cfg.ROBOTIQ_CONTACT_RAD
        )
        info.update(
            tip_error_m=error,
            fingers=fingers.copy(),
            closed=closed,
            expected_tip=expected_tip,
            actual_tip=actual_tip,
        )
        held = error <= float(cfg.CABLE_IN_GRIPPER_MAX_ERR_M) and closed
        info["held"] = bool(held)
        return bool(held), info
    except Exception as exc:
        info["error"] = str(exc)
        return False, info


def _abort_cable_lost(context, info: dict) -> None:
    reason = f"cable left gripper: {info}"
    context.services["abort_simulation"] = True
    context.services["abort_reason"] = reason
    context.blackboard.discard("cable_held")
    context.services["motion_controller"].clear_queue()
    print(f"[BT CABLE LOST{_station_tag(context)}] {reason}")


def monitor_cable_hold(context):
    """Monitor physical hold during every closed-gripper carry command."""

    from behaviour_tree_insertion.runtime import Status

    if context.services.get("abort_simulation"):
        return Status.FAILURE
    if not context.services.get("monitor_cable_hold"):
        return None
    controller = context.services["motion_controller"]
    if controller.is_done():
        return None
    command = controller._command_queue[controller._current_command_index]
    carrying = command.get("hold_gripper", False)
    closing = command.get("type") == "gripper" and command.get("action") == "close"
    if not carrying and not closing:
        return None
    frame = int(context.services.get("_cable_hold_frame", 0)) + 1
    context.services["_cable_hold_frame"] = frame
    if closing or frame % max(1, int(cfg.CABLE_HOLD_CHECK_EVERY_N_FRAMES)):
        return None
    held, info = cable_still_in_gripper(context)
    if held:
        return None
    _abort_cable_lost(context, info)
    return Status.FAILURE


def queue_release_gripper(context) -> None:
    """Release by opening the gripper only; no synthetic attachment is used."""

    context.services["monitor_cable_hold"] = False
    context.services["motion_controller"].add_gripper_command(
        action="open",
        wait_frames=int(getattr(cfg, "GRASP_RELEASE_WAIT_FRAMES", 90)),
    )


def check_gripper_released(context) -> bool:
    try:
        fingers = np.asarray(
            context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
        ).reshape(-1)
        opened = np.asarray(
            context.services["robot"].gripper.joint_opened_positions, dtype=np.float64
        ).reshape(-1)
        released = fingers.size == opened.size and bool(
            np.allclose(fingers, opened, atol=0.08)
        )
    except Exception:
        released = False
    if released:
        context.blackboard.add("cable_released")
        context.blackboard.discard("cable_held")
    return released


def queue_return_home(context) -> None:
    """Joint-interpolate the named UR5e arm home while keeping fingers open."""

    controller = context.services["motion_controller"]
    controller.add_gripper_command(action="open", wait_frames=30)
    controller.add_joint_waypoint(
        cfg.UR5E_HOME_ARM,
        joint_steps=300,
        open_gripper=True,
        label=f"{context.step.name}: UR5e home",
    )


def check_at_home(context) -> bool:
    joints = context.services["robot"].get_joint_positions()
    if joints is None:
        return False
    names = list(getattr(context.services["robot"], "dof_names", []) or [])
    try:
        arm = np.array(
            [joints[names.index(name)] for name in cfg.UR5E_ARM_JOINT_NAMES],
            dtype=np.float64,
        )
    except (ValueError, IndexError):
        return False
    at_home = bool(np.allclose(arm, cfg.UR5E_HOME_ARM, atol=0.08))
    if at_home:
        context.blackboard.add("at_home")
    return at_home
