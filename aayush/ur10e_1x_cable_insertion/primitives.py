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


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


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
    """Hand pose so the tool tip lands on ``tip`` (tip = hand + R @ [0,0,TOOL_OFFSET])."""

    rot = _quat_to_rot_matrix(orientation_wxyz)
    return np.asarray(tip, dtype=np.float64) - rot @ np.array(
        [0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64
    )


def _add_waypoint(controller, position, orientation, *, debug_tip=None, tip_registry=None, **kwargs) -> None:
    kwargs.setdefault("joint_interp", True)
    kwargs.setdefault("target_is_hand", True)
    kwargs.setdefault("max_frames", 900)
    # Keep tip metadata in aayush only — do not pass unknown kwargs into shared controller.
    label = str(kwargs.get("label", ""))
    if tip_registry is not None and debug_tip is not None and label:
        tip_registry[label] = np.asarray(debug_tip, dtype=np.float64).reshape(3).copy()
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
        np.asarray(
            context.step.inputs.get("orientation_wxyz", cfg.OBSERVE_ORIENTATION),
            dtype=np.float64,
        )
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
    """Tilt 60° toward +X, hover along that approach → open → descend → close → lift."""

    controller = context.services["motion_controller"]
    tip_registry = context.services.setdefault("ik_tip_by_label", {})
    tip_registry.clear()
    context.services["_ik_cmd_idx"] = None
    approach = _normalize_vec(cfg.GRASP_APPROACH_DIR)
    orientation = _normalize_quat(cfg.GRASP_ORIENTATION)
    point = np.asarray(
        context.services.get("live_grasp_point", grasp_part_center(context)),
        dtype=np.float64,
    ).reshape(3)
    context.services["live_grasp_point"] = point.copy()
    context.services["grasp_orientation"] = orientation.copy()

    tip_grasp = point.copy()
    tip_grasp[0] += float(cfg.GRASP_X_OFFSET_M)
    tip_grasp[2] += float(cfg.GRASP_DESCEND_CLEARANCE_M)
    # Hover back along the tilted approach (away from the tip / toward −approach).
    tip_hover = tip_grasp - approach * float(cfg.GRASP_HOVER_CLEARANCE_M)

    hand_hover = _hand_from_tip(tip_hover, orientation)
    hand_grasp = _hand_from_tip(tip_grasp, orientation)
    hand_lift = hand_grasp.copy()
    hand_lift[2] += float(cfg.GRASP_LIFT_CLEARANCE_M) + abs(float(cfg.GRASP_DESCEND_CLEARANCE_M))

    print(
        f"[BT GRASP] tilt={cfg.GRASP_TILT_FROM_DOWN_DEG:.0f}° toward +X "
        f"part={np.round(point, 4)} x_offset={cfg.GRASP_X_OFFSET_M:+.4f}\n"
        f"  approach={np.round(approach, 4)}\n"
        f"  tip_hover={np.round(tip_hover, 4)} tip_grasp={np.round(tip_grasp, 4)}\n"
        f"  hand_hover={np.round(hand_hover, 4)} hand_grasp={np.round(hand_grasp, 4)} "
        f"hand_lift={np.round(hand_lift, 4)}"
    )

    _add_waypoint(
        controller, hand_hover, orientation,
        label=f"{context.step.name}: hover-tilted", joint_steps=240, pos_tolerance=0.025,
        debug_tip=tip_hover, tip_registry=tip_registry,
    )
    controller.add_gripper_command(action="open", wait_frames=50)
    _add_waypoint(
        controller, hand_grasp, orientation,
        label=f"{context.step.name}: descend-tilted", joint_steps=200, pos_tolerance=0.008,
        debug_tip=tip_grasp, tip_registry=tip_registry,
    )
    controller.add_gripper_command(action="close", wait_frames=int(cfg.GRASP_CLOSE_WAIT_FRAMES))
    # Hold checks start only on hold_gripper waypoints (squeeze/lift), not hover/open/descend.
    context.services["monitor_cable_hold"] = True
    _add_waypoint(
        controller, hand_grasp, orientation,
        label=f"{context.step.name}: squeeze-hold",
        hold_gripper=True,
        joint_steps=max(20, int(cfg.GRASP_SQUEEZE_HOLD_FRAMES)),
        pos_tolerance=0.01,
        debug_tip=tip_grasp, tip_registry=tip_registry,
    )
    tip_lift = tip_grasp + np.array(
        [0.0, 0.0, float(cfg.GRASP_LIFT_CLEARANCE_M) + abs(float(cfg.GRASP_DESCEND_CLEARANCE_M))],
        dtype=np.float64,
    )
    _add_waypoint(
        controller, hand_lift, orientation,
        label=f"{context.step.name}: lift",
        hold_gripper=True,
        joint_steps=200,
        pos_tolerance=0.02,
        debug_tip=tip_lift, tip_registry=tip_registry,
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


def _find_named_descendant(stage, root_path: str, name: str) -> str | None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    wanted = name.lower()
    for prim in Usd.PrimRange(root):
        if prim.GetName().lower() == wanted:
            return str(prim.GetPath())
    return None


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
    """Apply a world-Z yaw to a tool orientation (tool −Z stays world −Z)."""

    half = 0.5 * np.deg2rad(float(yaw_deg))
    yaw_q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
    return _quat_multiply(yaw_q, quat_wxyz)


def compute_port_targets(stage) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Insert = mid of pins 1907/1910; approach = insert with +X standoff."""

    contacts = cfg.PORT_CONTACTS_PATH
    if not stage.GetPrimAtPath(contacts).IsValid():
        raise RuntimeError(f"Port contacts missing: {contacts}")
    path_a = _find_named_descendant(stage, contacts, cfg.PORT_PIN_A_NAME)
    path_b = _find_named_descendant(stage, contacts, cfg.PORT_PIN_B_NAME)
    if not path_a or not path_b:
        raise RuntimeError(
            f"Missing pins under {contacts}: "
            f"{cfg.PORT_PIN_A_NAME}={path_a} {cfg.PORT_PIN_B_NAME}={path_b}"
        )
    _amin, _amax, ca = prim_bbox(stage, path_a)
    _bmin, _bmax, cb = prim_bbox(stage, path_b)
    insert = 0.5 * (ca + cb)
    approach = insert.copy()
    approach[0] += float(cfg.PORT_APPROACH_X_OFFSET_M)
    return insert, approach, path_a, path_b


def _lift_tip_from_context(context) -> np.ndarray:
    """World tip pose at the end of the grasp lift (start of port maneuver)."""

    point = np.asarray(
        context.services.get("live_grasp_point", grasp_part_center(context)),
        dtype=np.float64,
    ).reshape(3)
    tip = point.copy()
    tip[0] += float(cfg.GRASP_X_OFFSET_M)
    tip[2] += (
        float(cfg.GRASP_DESCEND_CLEARANCE_M)
        + float(cfg.GRASP_LIFT_CLEARANCE_M)
        + abs(float(cfg.GRASP_DESCEND_CLEARANCE_M))
    )
    return tip


def _port_tip_via(tip_start: np.ndarray, tip_end: np.ndarray, frac: float) -> np.ndarray:
    """Lerp tip toward the offset; keep vias at least as high as the port tip."""

    t = float(np.clip(frac, 0.0, 1.0))
    tip = (1.0 - t) * tip_start + t * tip_end
    if t < 1.0 - 1e-9:
        tip[2] = max(float(tip[2]), float(tip_end[2])) + float(
            cfg.PORT_APPROACH_VIA_Z_CLEARANCE_M
        )
    return tip


def queue_port_approach(context) -> None:
    """Carry the grasped cable: yaw → offset vias → insert vias, gripper locked closed.

    Simultaneous tip lerp + −180° yaw was producing unreachable mid-poses (IK
    failed around −45°). Split: finish yaw while still near tip_start, translate
    to the +0.02 X offset, then continue tip into the insert point. Every arm
    segment uses hold_gripper so fingers never command open after the grasp.
    """

    controller = context.services["motion_controller"]
    stage = context.services["stage"]
    tip_registry = context.services.setdefault("ik_tip_by_label", {})
    tip_registry.clear()
    context.services["_ik_cmd_idx"] = None
    ori_start = _normalize_quat(
        context.services.get("grasp_orientation", cfg.GRASP_ORIENTATION)
    )
    yaw_total = float(cfg.PORT_APPROACH_YAW_DEG)
    ori_end = _yaw_about_world_z(ori_start, yaw_total)

    insert, approach, path_a, path_b = compute_port_targets(stage)
    context.services["port_insert_point"] = insert.copy()
    context.services["port_approach_point"] = approach.copy()
    context.services["port_approach_orientation"] = ori_end.copy()

    tip_start = _lift_tip_from_context(context)
    tip_end = approach.copy()
    # Slight raise at the lift tip so yaw clears the support / table.
    tip_yaw = tip_start.copy()
    tip_yaw[2] = max(float(tip_start[2]), float(tip_end[2])) + float(
        cfg.PORT_APPROACH_VIA_Z_CLEARANCE_M
    )

    yaw_steps = max(2, int(cfg.PORT_APPROACH_YAW_STEPS))
    via_fracs = [float(f) for f in cfg.PORT_APPROACH_VIA_FRACTIONS if 0.0 < float(f) < 1.0]
    via_fracs.append(1.0)
    insert_fracs = [
        float(f) for f in cfg.PORT_INSERT_VIA_FRACTIONS if 0.0 < float(f) < 1.0
    ]
    insert_fracs.append(1.0)
    context.services["monitor_cable_hold"] = True

    print(
        f"[BT PORT] staged transit {path_a} / {path_b}\n"
        f"  insert={np.round(insert, 4)} approach={np.round(approach, 4)} "
        f"(+{cfg.PORT_APPROACH_X_OFFSET_M:.3f} X)\n"
        f"  tip_start={np.round(tip_start, 4)} tip_yaw={np.round(tip_yaw, 4)} "
        f"tip_end={np.round(tip_end, 4)}\n"
        f"  phase1: yaw 0 → {yaw_total:.0f} deg over {yaw_steps} steps at tip_yaw\n"
        f"  phase2: tip vias fracs={via_fracs} → offset (fingers locked closed)\n"
        f"  phase3: tip vias fracs={insert_fracs} → insert (fingers locked closed)\n"
        f"  ori_start={np.round(ori_start, 4)} ori_end={np.round(ori_end, 4)}"
    )

    # Phase 1 — reorient near the lift tip (known-good workspace).
    for i in range(1, yaw_steps + 1):
        t = float(i) / float(yaw_steps)
        yaw_deg = t * yaw_total
        orientation = _yaw_about_world_z(ori_start, yaw_deg)
        hand = _hand_from_tip(tip_yaw, orientation)
        _add_waypoint(
            controller,
            hand,
            orientation,
            label=f"{context.step.name}: port-yaw-{yaw_deg:.0f}",
            hold_gripper=True,
            joint_steps=max(100, int(360 / yaw_steps)),
            pos_tolerance=0.035,
            max_frames=900,
            debug_tip=tip_yaw, tip_registry=tip_registry,
        )

    # Phase 2 — step the tip closer to the offset with orientation fixed.
    for i, frac in enumerate(via_fracs):
        tip = _port_tip_via(tip_start, tip_end, frac)
        hand = _hand_from_tip(tip, ori_end)
        is_final = i == len(via_fracs) - 1
        _add_waypoint(
            controller,
            hand,
            ori_end,
            label=f"{context.step.name}: port-via-{frac:.2f}",
            hold_gripper=True,
            joint_steps=160 if is_final else 120,
            pos_tolerance=(
                float(cfg.PORT_APPROACH_TOLERANCE_M) if is_final else 0.03
            ),
            max_frames=900,
            debug_tip=tip, tip_registry=tip_registry,
        )

    # Phase 3 — offset → insert tip, fingers stay locked closed (never open).
    for i, frac in enumerate(insert_fracs):
        t = float(frac)
        tip = (1.0 - t) * tip_end + t * insert
        hand = _hand_from_tip(tip, ori_end)
        is_final = i == len(insert_fracs) - 1
        _add_waypoint(
            controller,
            hand,
            ori_end,
            label=f"{context.step.name}: insert-via-{frac:.2f}",
            hold_gripper=True,
            joint_steps=140 if is_final else 100,
            pos_tolerance=(
                float(cfg.PORT_INSERT_TOLERANCE_M) if is_final else 0.03
            ),
            max_frames=900,
            debug_tip=tip, tip_registry=tip_registry,
        )


def check_at_port_approach(context) -> bool:
    """True when the grasped tip is near the pre-insert offset."""

    stage = context.services["stage"]
    approach = context.services.get("port_approach_point")
    if approach is None:
        try:
            _insert, approach, _a, _b = compute_port_targets(stage)
            context.services["port_approach_point"] = approach.copy()
        except Exception as exc:
            print(f"[BT PORT] approach validate: cannot resolve port: {exc}")
            return False

    try:
        tip = grasp_part_center(context)
        tip[0] += float(cfg.GRASP_X_OFFSET_M)
    except Exception as exc:
        print(f"[BT PORT] approach validate: tip bbox failed: {exc}")
        return False

    err = float(np.linalg.norm(tip - np.asarray(approach, dtype=np.float64)))
    ok = err <= float(cfg.PORT_APPROACH_TOLERANCE_M) + 0.02
    if ok:
        context.blackboard.add("at_port_approach")
    print(
        f"[BT PORT] approach validate tip={np.round(tip, 4)} "
        f"target={np.round(np.asarray(approach), 4)} err={err:.4f} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def check_at_port_insert(context) -> bool:
    """True when the grasped tip is near the RJ45 insert point (after offset)."""

    stage = context.services["stage"]
    insert = context.services.get("port_insert_point")
    approach = context.services.get("port_approach_point")
    if insert is None or approach is None:
        try:
            insert, approach, _a, _b = compute_port_targets(stage)
            context.services["port_insert_point"] = insert.copy()
            context.services["port_approach_point"] = approach.copy()
        except Exception as exc:
            print(f"[BT PORT] insert validate: cannot resolve port: {exc}")
            return False

    try:
        tip = grasp_part_center(context)
        tip[0] += float(cfg.GRASP_X_OFFSET_M)
    except Exception as exc:
        print(f"[BT PORT] insert validate: tip bbox failed: {exc}")
        return False

    err = float(np.linalg.norm(tip - np.asarray(insert, dtype=np.float64)))
    ok = err <= float(cfg.PORT_INSERT_TOLERANCE_M) + 0.02
    if ok:
        context.blackboard.add("at_port_approach")
        context.blackboard.add("at_port_insert")
    print(
        f"[BT PORT] insert validate tip={np.round(tip, 4)} "
        f"target={np.round(np.asarray(insert), 4)} err={err:.4f} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def cable_still_in_gripper(context) -> tuple[bool, dict]:
    """True when the grasped tip still tracks the tool tip (cable has not slipped out).

    Finger open/closed is logged, but abort uses tip error only. Otherwise a failed
    close pulse (fingers≈0) looks like "cable lost" even when the part never moved.
    """

    info: dict = {}
    try:
        controller = context.services["motion_controller"]
        hand, quat = controller._current_hand_pose()
        tip_expected = np.asarray(hand, dtype=np.float64) + _quat_to_rot_matrix(quat) @ np.array(
            [0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64
        )
        part = grasp_part_center(context)
        grasp_tip = part.copy()
        grasp_tip[0] += float(cfg.GRASP_X_OFFSET_M)
        tip_err = float(np.linalg.norm(grasp_tip - tip_expected))
        info["part"] = np.round(part, 4)
        info["grasp_tip"] = np.round(grasp_tip, 4)
        info["tip_expected"] = np.round(tip_expected, 4)
        info["tip_err_m"] = tip_err
    except Exception as exc:
        info["error"] = f"pose/bbox failed: {exc}"
        return False, info

    fingers = None
    closed_enough = True
    try:
        fingers = np.asarray(
            context.services["robot"].gripper.get_joint_positions(), dtype=np.float64
        ).reshape(-1)
        if fingers.size:
            closed_enough = float(np.max(np.abs(fingers))) >= float(cfg.ROBOTIQ_CONTACT_RAD)
        info["fingers"] = np.round(fingers, 3)
    except Exception:
        info["fingers"] = None

    info["closed_enough"] = bool(closed_enough)
    # Geometric slip only — part left the tool tip.
    in_grip = tip_err <= float(cfg.CABLE_IN_GRIPPER_MAX_ERR_M)
    info["in_gripper"] = bool(in_grip)
    return in_grip, info


def _format_cable_status(info: dict) -> str:
    return (
        f"in_gripper={info.get('in_gripper')} "
        f"tip_err={info.get('tip_err_m', float('nan')):.4f}m "
        f"(max={cfg.CABLE_IN_GRIPPER_MAX_ERR_M:.3f}) "
        f"closed={info.get('closed_enough')} fingers={info.get('fingers')} "
        f"part={info.get('part')} tip_expected={info.get('tip_expected')}"
    )


def _cmd_tip_str(context, cmd: dict | None) -> str:
    if not cmd:
        return ""
    tip = context.services.get("ik_tip_by_label", {}).get(str(cmd.get("label", "")))
    return f" tip={np.round(tip, 4)}" if tip is not None else ""


def _log_ik_progress(context) -> None:
    """Print REACHED / NEXT when the shared controller advances (aayush-only)."""

    controller = context.services["motion_controller"]
    idx = int(controller._current_command_index)
    prev = context.services.get("_ik_cmd_idx")
    queue = controller._command_queue

    def _next_cartesian(start: int):
        for j in range(start, len(queue)):
            if queue[j].get("type") == "cartesian":
                return queue[j]
        return None

    if prev is None:
        nxt = _next_cartesian(idx) if not controller.is_done() else None
        if nxt is not None:
            print(
                f"[BT IK] NEXT [{nxt.get('label', '')}] "
                f"hand_target={np.round(nxt['pos'], 4)}{_cmd_tip_str(context, nxt)}"
            )
        context.services["_ik_cmd_idx"] = idx
        return

    if idx <= prev:
        return

    for i in range(prev, min(idx, len(queue))):
        cmd = queue[i]
        if cmd.get("type") != "cartesian":
            continue
        print(
            f"[BT IK] REACHED [{cmd.get('label', '')}] "
            f"hand_target={np.round(cmd['pos'], 4)}{_cmd_tip_str(context, cmd)} "
            f"frames={cmd.get('frames_spent', '?')}"
        )
        if context.services.get("monitor_cable_hold") and cmd.get("hold_gripper"):
            held, info = cable_still_in_gripper(context)
            print(f"[BT GRIP] after REACHED [{cmd.get('label', '')}]: {_format_cable_status(info)}")
            if not held:
                _abort_cable_lost(
                    context, info, where=f"after reaching [{cmd.get('label', '')}]"
                )
                context.services["_ik_cmd_idx"] = idx
                return

    if controller.is_done():
        print("[BT IK] NEXT (none — queue end)")
    else:
        nxt = _next_cartesian(idx)
        if nxt is not None:
            print(
                f"[BT IK] NEXT [{nxt.get('label', '')}] "
                f"hand_target={np.round(nxt['pos'], 4)}{_cmd_tip_str(context, nxt)}"
            )
        else:
            cur = queue[idx] if idx < len(queue) else None
            kind = cur.get("type") if cur else "?"
            print(f"[BT IK] NEXT (non-cartesian: {kind})")
    context.services["_ik_cmd_idx"] = idx


def monitor_cable_hold(context):
    """Per-frame IK progress + hold monitor after grasp close through insert."""

    from behaviour_tree_insertion.runtime import Status

    if context.services.get("abort_simulation"):
        return Status.FAILURE

    _log_ik_progress(context)
    if context.services.get("abort_simulation"):
        return Status.FAILURE
    if not context.services.get("monitor_cable_hold"):
        return None

    controller = context.services["motion_controller"]
    if controller.is_done():
        return None

    cmd = controller._command_queue[controller._current_command_index]
    # Only while fingers are commanded locked on a carry waypoint.
    if cmd.get("type") != "cartesian" or not cmd.get("hold_gripper"):
        return None

    every = max(1, int(cfg.CABLE_HOLD_CHECK_EVERY_N_FRAMES))
    frame = int(context.services.get("_cable_hold_frame", 0)) + 1
    context.services["_cable_hold_frame"] = frame
    if frame % every != 0:
        return None

    held, info = cable_still_in_gripper(context)
    label = cmd.get("label", "?")
    if frame % max(every * 10, 30) == 0:
        print(f"[BT GRIP] during [{label}]: {_format_cable_status(info)}")
    if held:
        return None
    _abort_cable_lost(context, info, where=f"during [{label}]")
    return Status.FAILURE


def _abort_cable_lost(context, info: dict, *, where: str) -> None:
    msg = (
        f"[BT CABLE LOST] Cable left the gripper {where}. "
        f"{_format_cable_status(info)}. Closing simulation."
    )
    print(msg)
    context.services["abort_simulation"] = True
    context.services["abort_reason"] = msg
    try:
        context.services["motion_controller"].clear_queue()
    except Exception:
        pass
    app = context.services.get("simulation_app")
    if app is not None:
        try:
            app.close()
        except Exception as exc:
            print(f"[BT CABLE LOST] simulation_app.close() failed: {exc}")
