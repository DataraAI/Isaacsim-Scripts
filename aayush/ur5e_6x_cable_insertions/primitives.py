"""Behaviour-tree primitives: hover → orient+tilt → descend to neck → close grasp."""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom

from ur5e_6x_cable_insertions import config as cfg


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-9 else arr


def meters_per_unit(stage) -> float:
    """Return the USD stage scale in meters per authored unit."""

    try:
        value = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        value = 0.01
    if value <= 0.0:
        raise RuntimeError(f"Invalid metersPerUnit={value}")
    return value


def _safe_round_list(arr, ndigits: int = 5) -> list | None:
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        return [round(float(x), ndigits) for x in a.tolist()]
    except Exception:
        return None


def _refresh_insert_diag_geometry(context, *, label: str | None = None) -> None:
    """Stash tip/crystal/port geometry for the insert diagnostics recorder."""

    if context.services.get("insert_diagnostics") is None:
        return
    geom: dict = {}
    if label:
        context.services["_insert_diag_label"] = label
        geom["label"] = label
    try:
        controller = context.services["motion_controller"]
        tool = _align_insert_tool_ori(context)
        tip = measured_fingertip_meters(controller, tool)
        geom["tip_m"] = _safe_round_list(tip)
        geom["tool_ori_wxyz"] = _safe_round_list(tool)
    except Exception as exc:
        geom["tip_error"] = str(exc)
    try:
        crystal = live_crystal_features(context)
        c_mc = np.asarray(crystal.mating_center, dtype=np.float64).reshape(3)
        geom["crystal_mating_m"] = _safe_round_list(c_mc)
        axis = getattr(crystal, "insertion_axis", None)
        if axis is not None:
            geom["crystal_axis"] = _safe_round_list(axis)
        try:
            from ur5e_6x_cable_insertions.alignment import insert_direction_crystal_neg_x

            geom["insert_dir"] = _safe_round_list(insert_direction_crystal_neg_x(crystal))
        except Exception:
            pass
    except Exception as exc:
        geom["crystal_error"] = str(exc)
    try:
        port = live_port_features(context)
        p_mc = np.asarray(port.mating_center, dtype=np.float64).reshape(3)
        geom["port_mating_m"] = _safe_round_list(p_mc)
        if "crystal_mating_m" in geom and geom["crystal_mating_m"] is not None:
            delta = p_mc - np.asarray(geom["crystal_mating_m"], dtype=np.float64)
            geom["mating_delta_m"] = _safe_round_list(delta)
            geom["mating_gap_along_x_m"] = round(float(delta[0]), 5)
            geom["mating_gap_norm_m"] = round(float(np.linalg.norm(delta)), 5)
    except Exception as exc:
        geom["port_error"] = str(exc)
    context.services["_insert_diag_geometry"] = geom


def _station_tag(context) -> str:
    station_id = context.services.get("station_id")
    return f" {station_id}" if station_id else ""


def _station_side(context) -> str:
    spec = context.services.get("station_spec")
    if spec is not None:
        return str(getattr(spec, "side", "positive"))
    return "positive"


def _grasp_tool_and_lula(context) -> tuple[np.ndarray, np.ndarray]:
    """Side-keyed grasp tool + Lula oris (cached on services after orient)."""

    tool = context.services.get("grasp_tool_orientation")
    lula = context.services.get("grasp_orientation")
    if tool is not None and lula is not None:
        return _normalize_quat(tool), _normalize_quat(lula)
    side = _station_side(context)
    tool = _normalize_quat(cfg.grasp_tool_orientation(side))
    lula = _normalize_quat(cfg.grasp_orientation(side))
    context.services["grasp_tool_orientation"] = tool.copy()
    context.services["grasp_orientation"] = lula.copy()
    return tool, lula


def prim_bbox_meters(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World AABB of a prim in meters (min, max, center)."""

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mpu = meters_per_unit(stage)
    minimum = np.array(box.GetMin(), dtype=np.float64) * mpu
    maximum = np.array(box.GetMax(), dtype=np.float64) * mpu
    return minimum, maximum, 0.5 * (minimum + maximum)


def cable_world_translate_meters(stage, cable_root_path: str) -> np.ndarray:
    """World translation of the Network cable Xform, in meters."""

    prim = stage.GetPrimAtPath(cable_root_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing cable prim {cable_root_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    translate = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    mpu = meters_per_unit(stage)
    return np.array(
        [float(translate[0]), float(translate[1]), float(translate[2])],
        dtype=np.float64,
    ) * mpu


def _station_height(context) -> str:
    spec = context.services.get("station_spec")
    if spec is not None:
        return str(spec.height)
    return "top"


def _hover_z_stage(context) -> float:
    return float(cfg.hover_z_stage_for(_station_height(context)))


def _grasp_tip_z_stage(context) -> float:
    return float(cfg.grasp_tip_z_stage_for(_station_height(context)))


def hover_tip_above_cable(stage, cable_root_path: str, z_stage: float) -> np.ndarray:
    """Fingertip target: cable XY, Z from stage units → meters."""

    cable = cable_world_translate_meters(stage, cable_root_path)
    mpu = meters_per_unit(stage)
    tip = cable.copy()
    tip[2] = float(z_stage) * mpu
    return tip


def home_tip_above_work_table(stage, spec) -> np.ndarray:
    """Fingertip home: work-table center X, station cable/head39 Y, hover Z.

    Clears the DataHall rack. X comes from the shared WorkTable for this height;
    Y stays on the station cable (fallback head39) so NegY/PosY homes differ.
    """

    mpu = meters_per_unit(stage)
    z_stage = float(cfg.hover_z_stage_for(getattr(spec, "height", "top")))
    table_path = cfg.work_table_path_for(getattr(spec, "height", "top"))
    try:
        t_mn, t_mx, t_center = prim_bbox_meters(stage, table_path)
        table_x = float(t_center[0])
    except Exception:
        # Authored translate if bbox fails.
        try:
            table_x = float(
                cable_world_translate_meters(stage, table_path)[0]
            )
        except Exception:
            table_x = 0.45  # ~WorkTable center at mpu=0.01

    try:
        cable = cable_world_translate_meters(stage, spec.cable_root_path)
        tip_y = float(cable[1])
    except Exception:
        try:
            _mn, _mx, h39 = prim_bbox_meters(stage, spec.path39)
            tip_y = float(h39[1])
        except Exception:
            tip_y = 0.0

    tip = np.array([table_x, tip_y, z_stage * mpu], dtype=np.float64)
    return tip


def hand_from_tip(tip: np.ndarray, tool_quat_wxyz: np.ndarray) -> np.ndarray:
    """Hand/TCP so fingertips land on ``tip`` (tip = hand + R_tool @ [0,0,offset])."""

    rot = cfg._quat_to_rot_matrix(np.asarray(tool_quat_wxyz, dtype=np.float64))
    return np.asarray(tip, dtype=np.float64).reshape(3) - rot @ np.array(
        [0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64
    )


def tip_from_hand(hand: np.ndarray, tool_quat_wxyz: np.ndarray) -> np.ndarray:
    """Fingertip from Lula hand pose (inverse of :func:`hand_from_tip`)."""

    rot = cfg._quat_to_rot_matrix(np.asarray(tool_quat_wxyz, dtype=np.float64))
    return np.asarray(hand, dtype=np.float64).reshape(3) + rot @ np.array(
        [0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64
    )


def measured_fingertip_meters(controller, tool_quat_wxyz: np.ndarray) -> np.ndarray:
    """World fingertip from the live hand FK (meters)."""

    hand_m, _ori = controller.current_hand_pose_meters()
    return tip_from_hand(hand_m, tool_quat_wxyz)


def _grasp_part_path(context) -> str:
    path = context.services.get("grasp_part_path")
    if path:
        return str(path)
    spec = context.services.get("station_spec")
    if spec is not None:
        return str(spec.grasp_part_path)
    raise RuntimeError("grasp_part_path missing from services")


def grasp_x_offset_m(context) -> float:
    """World-X offset from neck center toward the −X end (ur10e_1x recipe)."""

    cached = context.services.get("grasp_x_offset_m")
    if cached is not None:
        return float(cached)
    stage = context.services["stage"]
    path = _grasp_part_path(context)
    part_min, _part_max, center = prim_bbox_meters(stage, path)
    toward_end = float(cfg.GRASP_TOWARD_NEG_X_FRAC) * (
        float(part_min[0]) - float(center[0])
    )
    offset = toward_end + float(cfg.GRASP_X_OFFSET_M)
    context.services["grasp_x_offset_m"] = float(offset)
    return float(offset)


def neck_grasp_tip_meters(context) -> np.ndarray:
    """Pinch tip on ``E_part006_44``.

    When ``GRASP_TIP_X_STAGE`` is set, TipGrasp X is that absolute stage-unit
    value (same convention as ``GRASP_TIP_Z_STAGE``). Otherwise relative X knobs
    nudge from the neck bbox center toward ±X.
    """

    stage = context.services["stage"]
    path = _grasp_part_path(context)
    part_min, part_max, center = prim_bbox_meters(stage, path)
    tip = center.copy()
    tip[1] += float(cfg.GRASP_Y_ALIGNMENT_OFFSET_M)
    mpu = meters_per_unit(stage)
    tip[1] += float(getattr(cfg, "GRASP_TIP_Y_STAGE_DELTA", 0.0)) * mpu
    tip[2] = _grasp_tip_z_stage(context) * mpu + float(
        getattr(cfg, "GRASP_TIP_Z_BIAS_M", 0.0)
    )
    abs_x = getattr(cfg, "GRASP_TIP_X_STAGE", None)
    if abs_x is not None:
        tip[0] = float(abs_x) * mpu
        context.services["grasp_x_offset_m"] = float(tip[0] - center[0])
    else:
        x_offset = grasp_x_offset_m(context)
        tip[0] += float(x_offset)
        tip[0] += float(cfg.GRASP_TIP_X_STAGE_DELTA) * mpu

    cable_path = context.services.get("cable_root_path")
    if cable_path:
        try:
            cable = cable_world_translate_meters(stage, cable_path)
            context.services["cable_x_m"] = float(cable[0])
        except Exception:
            pass
    context.services["neck_x_m"] = float(center[0])
    context.services["neck_span_x_m"] = float(part_max[0] - part_min[0])

    # Optional override: force cable-root X (usually over Left block — off by default).
    if abs_x is None and bool(cfg.GRASP_USE_CABLE_ROOT_X) and cable_path:
        try:
            tip[0] = float(context.services["cable_x_m"]) + float(cfg.GRASP_X_OFFSET_M)
        except Exception:
            pass

    spec = context.services.get("station_spec")
    if spec is not None:
        margin = float(cfg.GRASP_BLOCK_CLEARANCE_M)
        try:
            left_mn, left_mx, _ = prim_bbox_meters(stage, spec.left_block_path)
            right_mn, right_mx, _ = prim_bbox_meters(stage, spec.right_block_path)
            # Left/Right sit on opposite ±Y sides of the cable after the layout rotate.
            if float(left_mx[1]) < float(right_mn[1]):
                lo = float(left_mx[1]) + margin
                hi = float(right_mn[1]) - margin
            elif float(right_mx[1]) < float(left_mn[1]):
                lo = float(right_mx[1]) + margin
                hi = float(left_mn[1]) - margin
            else:
                lo, hi = float("nan"), float("nan")
            if lo < hi and not (lo <= float(tip[1]) <= hi):
                clamp = bool(getattr(cfg, "GRASP_CLAMP_TIP_Y_TO_BLOCK_GAP", True))
                if clamp:
                    tip_y0 = float(tip[1])
                    tip[1] = float(np.clip(tip_y0, lo, hi))
                    print(
                        f"[BT DESCEND{_station_tag(context)}] clamp tip Y "
                        f"{tip_y0:.4f} → {tip[1]:.4f} into Left/Right gap "
                        f"[{lo:.4f}, {hi:.4f}]"
                    )
                else:
                    print(
                        f"[BT DESCEND{_station_tag(context)}] warn: tip Y={tip[1]:.4f} "
                        f"outside Left/Right gap [{lo:.4f}, {hi:.4f}] — keeping "
                        f"commanded TipGrasp Y"
                    )
        except Exception as exc:
            print(
                f"[BT DESCEND{_station_tag(context)}] block gap check skipped: {exc}"
            )
    return tip


def _pad_bbox_centers_m(context) -> tuple[np.ndarray, np.ndarray, str, str] | None:
    """Live left/right fingertip pad bbox centers (meters), or None."""

    stage = context.services["stage"]
    spec = context.services.get("station_spec")
    robot_path = getattr(spec, "robot_prim_path", None) if spec is not None else None
    if not robot_path:
        return None
    left_path, right_path = _finger_pad_prim_paths(stage, str(robot_path))
    if not left_path or not right_path:
        return None
    _lmin, _lmax, l_c = prim_bbox_meters(stage, left_path)
    _rmin, _rmax, r_c = prim_bbox_meters(stage, right_path)
    return (
        np.asarray(l_c, dtype=np.float64).reshape(3),
        np.asarray(r_c, dtype=np.float64).reshape(3),
        left_path,
        right_path,
    )


def _tip_nudge_center_pads_on_neck(context, tip: np.ndarray) -> np.ndarray:
    """Shift tip in XY so the open-pad midpoint lands on the neck center.

    Keeps Z (descend depth). Caps the nudge so a bad pad read cannot yank far.
    """

    pads = _pad_bbox_centers_m(context)
    if pads is None:
        print(
            f"[BT GRASP{_station_tag(context)}] pad-center skip — "
            "left/right pads not found"
        )
        return np.asarray(tip, dtype=np.float64).reshape(3)
    left_c, right_c, left_path, right_path = pads
    neck_path = _grasp_part_path(context)
    _nmin, _nmax, neck_c = prim_bbox_meters(context.services["stage"], neck_path)
    neck_c = np.asarray(neck_c, dtype=np.float64).reshape(3)
    mid = 0.5 * (left_c + right_c)
    nudge = np.zeros(3, dtype=np.float64)
    nudge[0] = float(neck_c[0] - mid[0])
    nudge[1] = float(neck_c[1] - mid[1])
    mag = float(np.linalg.norm(nudge))
    max_n = float(getattr(cfg, "GRASP_PAD_CENTER_MAX_NUDGE_M", 0.012))
    if mag > max_n and mag > 1e-12:
        nudge = nudge * (max_n / mag)
    tip_out = np.asarray(tip, dtype=np.float64).reshape(3).copy() + nudge
    print(
        f"[BT GRASP{_station_tag(context)}] center pads on neck "
        f"nudge_xy={np.round(nudge[:2], 5)} (|n|={mag:.4f}m≤{max_n:.4f})\n"
        f"  left={np.round(left_c, 4)} right={np.round(right_c, 4)} "
        f"mid={np.round(mid, 4)} neck={np.round(neck_c, 4)}\n"
        f"  tip {np.round(tip, 4)} → {np.round(tip_out, 4)}\n"
        f"  pads L={left_path} R={right_path}"
    )
    return tip_out


def queue_move(context) -> None:
    """Hover above the Network cable: fingertips at hover XYZ, gripper straight down.

    Targets the **fingertips** (not the Lula wrist frame) so the hand sits
    ``TOOL_OFFSET_M`` above the tip when fingers point down — clearing the
    cable and CableBlocks. A high approach via was removed: tip Z≈380 put the
    hand outside UR5e reach and IK aborted with no motion.
    """

    from ur5e_6x_cable_insertions.scene import spawn_station_debug_markers

    stage = context.services["stage"]
    cable_path = context.services["cable_root_path"]
    spec = context.services.get("station_spec")
    tool_ori = _normalize_quat(cfg.OBSERVE_TOOL_ORIENTATION)
    lula_ori = _normalize_quat(
        context.step.inputs.get("orientation_wxyz", cfg.OBSERVE_ORIENTATION)
    )
    cable = cable_world_translate_meters(stage, cable_path)
    tip_hover = hover_tip_above_cable(stage, cable_path, _hover_z_stage(context))
    hand_hover = hand_from_tip(tip_hover, tool_ori)
    context.services["hover_tip"] = tip_hover.copy()
    context.services["hover_hand"] = hand_hover.copy()
    if spec is not None:
        spawn_station_debug_markers(
            stage,
            spec,
            cable_m=cable,
            tip_hover_m=tip_hover,
            hand_hover_m=hand_hover,
        )
    print(
        f"[BT HOVER{_station_tag(context)}] cable={cable_path}\n"
        f"  cable={np.round(cable, 4)} tip_hover={np.round(tip_hover, 4)} "
        f"(z_stage={_hover_z_stage(context)}) hand={np.round(hand_hover, 4)}\n"
        f"  markers under {getattr(spec, 'debug_marker_root', cfg.DEBUG_MARKER_ROOT)} "
        f"(default hidden; toggle visibility in Isaac)"
    )
    controller = context.services["motion_controller"]
    controller.clear_queue()
    controller.add_gripper_command("open", wait_frames=30)
    controller.add_cartesian_waypoint(
        hand_hover,
        lula_ori,
        target_is_hand=True,
        joint_interp=True,
        joint_steps=int(cfg.HOVER_JOINT_STEPS),
        max_frames=max(2400, int(cfg.HOVER_JOINT_STEPS) * 4),
        pos_tolerance=float(cfg.HOVER_POS_TOLERANCE_M),
        label=f"{context.step.name}: hover",
    )
    controller.add_cartesian_waypoint(
        hand_hover,
        lula_ori,
        target_is_hand=True,
        joint_interp=True,
        joint_steps=int(cfg.HOVER_SETTLE_FRAMES),
        max_frames=max(300, int(cfg.HOVER_SETTLE_FRAMES) * 3),
        pos_tolerance=float(cfg.HOVER_POS_TOLERANCE_M),
        label=f"{context.step.name}: settle",
    )


def queue_orient_tilt(context) -> None:
    """Keep tip at hover XYZ; yaw+tilt so fingers straddle the cable on ±X.

    Side-keyed ``grasp_*_orientation``: tool +Z tilted ``GRASP_TILT_FROM_DOWN_DEG``
    from −Z toward −Y (NegativeY) or +Y (PositiveY); open axis ≈ world ±X.
    Hand is recomputed from the tip so the fingertips stay put while the wrist
    rotates.
    """

    stage = context.services["stage"]
    cable_path = context.services["cable_root_path"]
    tip = context.services.get("hover_tip")
    if tip is None:
        tip = hover_tip_above_cable(stage, cable_path, _hover_z_stage(context))
    else:
        tip = np.asarray(tip, dtype=np.float64).reshape(3)
    side = _station_side(context)
    tool_ori = _normalize_quat(cfg.grasp_tool_orientation(side))
    lula_ori = _normalize_quat(cfg.grasp_orientation(side))
    hand = hand_from_tip(tip, tool_ori)
    context.services["orient_tip"] = tip.copy()
    context.services["orient_hand"] = hand.copy()
    context.services["grasp_tool_orientation"] = tool_ori.copy()
    context.services["grasp_orientation"] = lula_ori.copy()
    lean = "−Y" if cfg._normalize_side_key(side) == "negative" else "+Y"
    print(
        f"[BT ORIENT{_station_tag(context)}] tilt={cfg.GRASP_TILT_FROM_DOWN_DEG:.0f}° "
        f"−Z→{lean}; finger yaw={cfg.FINGER_OPEN_YAW_ABOUT_TOOL_Z_DEG:.0f}° "
        f"(pads straddle cable ±X)\n"
        f"  tip={np.round(tip, 4)} hand={np.round(hand, 4)}"
    )
    controller = context.services["motion_controller"]
    controller.clear_queue()
    controller.add_gripper_command("open", wait_frames=10)
    controller.add_cartesian_waypoint(
        hand,
        lula_ori,
        target_is_hand=True,
        joint_interp=True,
        joint_steps=int(cfg.ORIENT_JOINT_STEPS),
        max_frames=max(2400, int(cfg.ORIENT_JOINT_STEPS) * 4),
        pos_tolerance=float(cfg.ORIENT_POS_TOLERANCE_M),
        label=f"{context.step.name}: orient-tilt",
    )
    controller.add_cartesian_waypoint(
        hand,
        lula_ori,
        target_is_hand=True,
        joint_interp=True,
        joint_steps=int(cfg.ORIENT_SETTLE_FRAMES),
        max_frames=max(300, int(cfg.ORIENT_SETTLE_FRAMES) * 3),
        pos_tolerance=float(cfg.ORIENT_POS_TOLERANCE_M),
        label=f"{context.step.name}: settle",
    )


def queue_descend_to_neck(context) -> None:
    """Descend once into the Left/Right block gap at ``E_part006_44``."""

    from ur5e_6x_cable_insertions.scene import (
        resolve_grasp_part_path,
        spawn_station_debug_markers,
    )

    stage = context.services["stage"]
    spec = context.services.get("station_spec")
    if spec is not None and not context.services.get("grasp_part_path"):
        context.services["grasp_part_path"] = resolve_grasp_part_path(stage, spec)

    tool_ori, lula_ori = _grasp_tool_and_lula(context)
    tip_grasp = neck_grasp_tip_meters(context)
    hand_grasp = hand_from_tip(tip_grasp, tool_ori)
    context.services["live_grasp_point"] = tip_grasp.copy()
    context.services["grasp_tip"] = tip_grasp.copy()
    context.services["grasp_hand"] = hand_grasp.copy()
    context.services["grasp_tool_orientation"] = tool_ori.copy()
    context.services["grasp_orientation"] = lula_ori.copy()

    if spec is not None:
        spawn_station_debug_markers(
            stage,
            spec,
            tip_grasp_m=tip_grasp,
            hand_grasp_m=hand_grasp,
        )

    cable_x = context.services.get("cable_x_m")
    print(
        f"[BT DESCEND{_station_tag(context)}] neck={_grasp_part_path(context)}\n"
        f"  tip_grasp={np.round(tip_grasp, 4)} hand_grasp={np.round(hand_grasp, 4)}\n"
        f"  tip_X={tip_grasp[0]:.4f} neck_center_X="
        f"{context.services.get('neck_x_m', float('nan')):.4f} "
        f"x_offset={context.services.get('grasp_x_offset_m', 0.0):+.4f} "
        f"(frac={cfg.GRASP_TOWARD_NEG_X_FRAC:.2f} toward −X) "
        f"z_bias={float(getattr(cfg, 'GRASP_TIP_Z_BIAS_M', 0.0)):+.4f}m "
        f"cable_X="
        f"{'n/a' if cable_x is None else f'{float(cable_x):.4f}'}"
    )
    controller = context.services["motion_controller"]
    controller.clear_queue()
    controller.add_gripper_command("open", wait_frames=10)
    controller.add_cartesian_waypoint(
        hand_grasp,
        lula_ori,
        target_is_hand=True,
        joint_interp=True,
        joint_steps=int(cfg.DESCEND_JOINT_STEPS),
        max_frames=max(2400, int(cfg.DESCEND_JOINT_STEPS) * 4),
        pos_tolerance=float(cfg.DESCEND_POS_TOLERANCE_M),
        label=f"{context.step.name}: descend-neck",
    )
    controller.add_cartesian_waypoint(
        hand_grasp,
        lula_ori,
        target_is_hand=True,
        joint_interp=True,
        joint_steps=int(cfg.DESCEND_SETTLE_FRAMES),
        max_frames=max(300, int(cfg.DESCEND_SETTLE_FRAMES) * 3),
        pos_tolerance=float(cfg.DESCEND_POS_TOLERANCE_M),
        label=f"{context.step.name}: settle",
    )
    # Capture arm joints at grasp height so close cannot reuse a stale lock
    # (or float under gravity) and dip the crystal neck.
    controller.add_hold_command(
        max(20, int(cfg.DESCEND_SETTLE_FRAMES) // 2),
        hold_gripper=False,
        label=f"{context.step.name}: pre-grasp-freeze",
    )


def queue_close_grasp(context) -> None:
    """Close the Robotiq fingers on the neck; keep the arm frozen.

    Before closing: optionally shift tip in XY so open left/right pads straddle
    the neck equally (fixes one-finger misses). Then freeze arm joints and close.
    """

    tool_ori, lula_ori = _grasp_tool_and_lula(context)
    tip = context.services.get("grasp_tip")
    if tip is None:
        tip = neck_grasp_tip_meters(context)
    else:
        tip = np.asarray(tip, dtype=np.float64).reshape(3)

    controller = context.services["motion_controller"]
    tip_cmd = tip.copy()
    if bool(getattr(cfg, "GRASP_CENTER_PADS_ON_NECK", True)):
        tip_cmd = _tip_nudge_center_pads_on_neck(context, tip)
    tip_cmd = np.asarray(tip_cmd, dtype=np.float64).reshape(3)
    hand = hand_from_tip(tip_cmd, tool_ori)
    context.services["grasp_tip"] = tip_cmd.copy()
    context.services["grasp_hand"] = hand.copy()
    context.services["live_grasp_point"] = tip_cmd.copy()

    nudge = tip_cmd - tip
    need_center_move = float(np.linalg.norm(nudge[:2])) > 0.001
    controller.clear_queue()
    if need_center_move:
        print(
            f"[BT GRASP{_station_tag(context)}] pre-close pad-center move "
            f"Δxy={np.round(nudge[:2], 4)} tip→{np.round(tip_cmd, 4)}"
        )
        controller.add_cartesian_waypoint(
            hand,
            lula_ori,
            target_is_hand=True,
            joint_interp=False,
            linear=True,
            linear_step=0.002,
            hold_gripper=False,  # keep fingers open while centering
            joint_steps=int(getattr(cfg, "GRASP_PAD_CENTER_JOINT_STEPS", 60)),
            max_frames=240,
            pos_tolerance=0.003,
            label=f"{context.step.name}: center-pads",
        )

    robot = context.services.get("robot") or getattr(controller, "_robot", None)
    # Snapshot live joints after any center move is queued; hold_arm close uses
    # the lock refreshed once the center segment finishes via remember below.
    if robot is not None and not need_center_move:
        try:
            q = np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)
            if hasattr(q, "cpu"):
                q = q.cpu().numpy()
            remember = getattr(controller, "_remember_lock", None)
            if callable(remember):
                remember(q)
            controller._last_commanded_joint_positions = list(q)
        except Exception:
            pass

    measured = measured_fingertip_meters(controller, tool_ori)
    dz = float(measured[2] - tip_cmd[2])
    print(
        f"[BT GRASP{_station_tag(context)}] close on neck (arm frozen after center) "
        f"cmd_tip={np.round(tip_cmd, 4)} measured_tip={np.round(measured, 4)} "
        f"dZ={dz:+.4f}m hand={np.round(hand, 4)}"
    )
    context.services["monitor_cable_hold"] = True
    # Freeze at the (possibly centered) pose: brief hold then close.
    if need_center_move:
        controller.add_hold_command(
            15,
            hold_gripper=False,
            label=f"{context.step.name}: pre-close-freeze",
        )
    else:
        lock = getattr(controller, "_lock_joint_positions", None)
        if lock is not None:
            controller._last_commanded_joint_positions = list(lock)
    controller.add_gripper_command(
        "close",
        wait_frames=int(cfg.GRASP_CLOSE_WAIT_FRAMES),
        hold_arm=True,
    )
    controller.add_hold_command(
        int(cfg.GRASP_SQUEEZE_HOLD_FRAMES),
        hold_gripper=True,
        label=f"{context.step.name}: squeeze-hold",
    )


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


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Unit-quaternion slerp; picks the short arc."""

    a = _normalize_quat(q0)
    b = _normalize_quat(q1)
    t = float(np.clip(t, 0.0, 1.0))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat(a + t * (b - a))
    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_0 = float(np.sin(theta_0))
    theta = theta_0 * t
    s0 = float(np.sin(theta_0 - theta) / sin_0)
    s1 = float(np.sin(theta) / sin_0)
    return _normalize_quat(s0 * a + s1 * b)


def _yaw_about_world_z(quat_wxyz: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Apply a world-Z yaw (tool −Z stays roughly world −Z for vertical grasp)."""

    half = 0.5 * np.deg2rad(float(yaw_deg))
    yaw_q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
    return _quat_multiply(yaw_q, quat_wxyz)


def _pitch_about_world_y(quat_wxyz: np.ndarray, pitch_deg: float) -> np.ndarray:
    """Apply a world-Y pitch (nose-up clears Mesh4679 during angled approach)."""

    half = 0.5 * np.deg2rad(float(pitch_deg))
    pitch_q = np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)
    return _quat_multiply(pitch_q, quat_wxyz)


def _rotate_vec_by_quat(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by unit quaternion ``quat_wxyz`` (world-frame)."""

    q = _normalize_quat(quat_wxyz)
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    # q * (0, v) * q_conj
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    qv = np.array(
        [
            -x * v[0] - y * v[1] - z * v[2],
            w * v[0] + y * v[2] - z * v[1],
            w * v[1] + z * v[0] - x * v[2],
            w * v[2] + x * v[1] - y * v[0],
        ],
        dtype=np.float64,
    )
    # q_conj = (w, -x, -y, -z)
    return np.array(
        [
            -qv[0] * x + qv[1] * w - qv[2] * z + qv[3] * y,
            -qv[0] * y + qv[1] * z + qv[2] * w - qv[3] * x,
            -qv[0] * z - qv[1] * y + qv[2] * x + qv[3] * w,
        ],
        dtype=np.float64,
    )


def _quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion rotating unit vector ``a`` onto ``b``."""

    v0 = _normalize_vec(a)
    v1 = _normalize_vec(b)
    dot = float(np.dot(v0, v1))
    if dot < -0.999999:
        # 180°: pick a stable orthogonal axis.
        axis = np.cross(v0, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(v0, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = _normalize_vec(axis)
        return _normalize_quat(np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64))
    cross = np.cross(v0, v1)
    w = 1.0 + dot
    return _normalize_quat(
        np.array([w, float(cross[0]), float(cross[1]), float(cross[2])], dtype=np.float64)
    )


def _pitch_tool_z_to_tilt(tool_ori: np.ndarray, tilt_from_down_deg: float) -> np.ndarray:
    """Pitch tool +Z to ``tilt_from_down_deg`` from world −Z, keeping XY heading.

    θ=0 → straight down (−Z); θ=90° → horizontal in the current XY heading
    (gripper / wrist roughly parallel to the XY plane).
    """

    tool_z = _rotate_vec_by_quat(tool_ori, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    xy = np.array([float(tool_z[0]), float(tool_z[1]), 0.0], dtype=np.float64)
    nxy = float(np.linalg.norm(xy))
    if nxy < 1e-9:
        heading = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    else:
        heading = xy / nxy
    tilt = np.deg2rad(float(tilt_from_down_deg))
    target = heading * float(np.sin(tilt)) + np.array(
        [0.0, 0.0, -float(np.cos(tilt))], dtype=np.float64
    )
    target = _normalize_vec(target)
    if float(np.linalg.norm(tool_z - target)) < 1e-9:
        return _normalize_quat(tool_ori)
    q_pitch = _quat_from_two_vectors(tool_z, target)
    return _normalize_quat(_quat_multiply(q_pitch, tool_ori))


def _roll_about_axis_flatten_tool_z(
    tool_ori: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """Rotate about ``axis`` so tool +Z is as parallel to the XY plane as possible."""

    a = _normalize_vec(axis)
    tool_z = _rotate_vec_by_quat(tool_ori, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # (R_a(φ) v) · ez = C cosφ + S sinφ + K, with K=(a·v)(a·ez).
    k = float(np.dot(a, tool_z) * a[2])
    c = float(np.dot(tool_z, ez) - k)
    s = float(np.dot(np.cross(a, tool_z), ez))
    amp = float(np.hypot(c, s))
    if amp < 1e-9:
        return _normalize_quat(tool_ori)
    # R cos(φ − α) = −K  →  cos(φ − α) = −K/R
    target = float(np.clip(-k / amp, -1.0, 1.0))
    alpha = float(np.arctan2(s, c))
    delta = float(np.arccos(target))
    # Prefer the smaller |φ|; break ties by keeping tool +Z toward −X.
    best_q = _normalize_quat(tool_ori)
    best_score = None
    for phi in (alpha + delta, alpha - delta):
        half = 0.5 * float(phi)
        q_roll = _normalize_quat(
            np.array(
                [
                    np.cos(half),
                    a[0] * np.sin(half),
                    a[1] * np.sin(half),
                    a[2] * np.sin(half),
                ],
                dtype=np.float64,
            )
        )
        cand = _normalize_quat(_quat_multiply(q_roll, tool_ori))
        z = _rotate_vec_by_quat(cand, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        # Primary: |z_z| small; secondary: prefer −X heading; tertiary: smaller |φ|.
        score = (abs(float(z[2])), -float(z[0]), abs(float(phi)))
        if best_score is None or score < best_score:
            best_score = score
            best_q = cand
    return best_q


def _tool_ori_align_axis_to_world_x(
    tool_ori: np.ndarray,
    crystal_axis: np.ndarray,
    *,
    prefer: str = "-x",
    flatten_xy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrist-tilt ``tool_ori`` so ``crystal_axis`` becomes colinear with world ±X.

    ``prefer`` is ``\"-x\"`` (into the jack) or ``\"+x\"``. When ``flatten_xy``
    is set, also roll about the aligned axis so tool +Z stays near the XY plane
    — that twist rolls the cable about its insertion axis and is off by default.
    Returns (new_tool_ori, aligned_axis).
    """

    axis = _normalize_vec(crystal_axis)
    plus_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    minus_x = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    prefer_key = str(prefer).strip().lower()
    if prefer_key in ("-x", "neg", "negative", "minus"):
        target = minus_x
    elif prefer_key in ("+x", "pos", "positive", "plus"):
        target = plus_x
    else:
        target = (
            plus_x
            if abs(float(np.dot(axis, plus_x))) >= abs(float(np.dot(axis, minus_x)))
            else minus_x
        )
    q_delta = _quat_from_two_vectors(axis, target)
    tool_aligned = _quat_multiply(q_delta, tool_ori)
    axis_aligned = _normalize_vec(_rotate_vec_by_quat(q_delta, axis))
    if flatten_xy:
        tool_aligned = _roll_about_axis_flatten_tool_z(tool_aligned, axis_aligned)
        # Axis is unchanged by a pure roll about itself.
    return tool_aligned, axis_aligned


def compute_tip_offset_meters(stage, spec, port, standoff_m: float) -> np.ndarray:
    """Fingertip at copper-pin jack center YZ + +X standoff (ur10e recipe).

    Prefer ``tip_offset_for_crystal_mating`` for TipOffset — fingertip ≠ crystal
    mating center, so pin YZ leaves the crystal below/above the jack.
    """

    from ur5e_6x_cable_insertions.scene import find_descendant

    contacts = getattr(spec, "port_contacts_path", None)
    path_a = path_b = None
    if contacts:
        path_a = find_descendant(stage, contacts, cfg.PORT_PIN_A_NAME)
        path_b = find_descendant(stage, contacts, cfg.PORT_PIN_B_NAME)
    if path_a and path_b:
        _amin, _amax, ca = prim_bbox_meters(stage, path_a)
        _bmin, _bmax, cb = prim_bbox_meters(stage, path_b)
        center = 0.5 * (
            np.asarray(ca, dtype=np.float64).reshape(3)
            + np.asarray(cb, dtype=np.float64).reshape(3)
        )
        tip = center.copy()
        tip[0] = float(center[0]) + abs(float(standoff_m))
        return tip

    from ur5e_6x_cable_insertions.alignment import port_standoff_target

    print(
        f"[BT MANEUVER] copper pins missing under {contacts}; "
        "TipOffset falls back to mating_center"
    )
    return port_standoff_target(port, float(standoff_m))


def tip_offset_for_crystal_mating(
    *,
    tip_now: np.ndarray,
    tool_ori_now: np.ndarray,
    crystal_mc_now: np.ndarray,
    tool_ori_tgt: np.ndarray,
    port_mc: np.ndarray,
    standoff_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tip at ``tool_ori_tgt`` that places crystal mating at port YZ + X standoff.

    Rigid grasp: ``crystal = tip + R(tool) @ p_local``. Solves for tip so the
    crystal mating center lands on ``[port.x+standoff, port.y, port.z]``.
    Returns ``(tip_end, desired_crystal_mc)``.
    """

    tip_now = np.asarray(tip_now, dtype=np.float64).reshape(3)
    crystal_mc_now = np.asarray(crystal_mc_now, dtype=np.float64).reshape(3)
    port_mc = np.asarray(port_mc, dtype=np.float64).reshape(3)
    r_now = cfg._quat_to_rot_matrix(_normalize_quat(tool_ori_now))
    r_tgt = cfg._quat_to_rot_matrix(_normalize_quat(tool_ori_tgt))
    p_local = r_now.T @ (crystal_mc_now - tip_now)
    desired_crystal = port_mc.copy()
    desired_crystal[0] = float(port_mc[0]) + abs(float(standoff_m))
    tip_end = desired_crystal - (r_tgt @ p_local)
    return tip_end, desired_crystal


def _measure_post_lift_tip(context) -> np.ndarray:
    """Live fingertip after lift: prefer FK tip, else planned lift_tip."""

    tool_ori, _ = _grasp_tool_and_lula(context)
    controller = context.services.get("motion_controller")
    if controller is not None and hasattr(controller, "current_hand_pose_meters"):
        try:
            tip = measured_fingertip_meters(controller, tool_ori)
            if np.all(np.isfinite(tip)):
                return np.asarray(tip, dtype=np.float64).reshape(3)
        except Exception as exc:
            print(f"[BT MANEUVER{_station_tag(context)}] live tip FK failed: {exc}")
    return _lift_tip_from_services(context)


def _lift_tip_from_services(context) -> np.ndarray:
    tip = context.services.get("lift_tip")
    if tip is not None:
        return np.asarray(tip, dtype=np.float64).reshape(3)
    grasp = context.services.get("grasp_tip")
    if grasp is None:
        grasp = neck_grasp_tip_meters(context)
    tip = np.asarray(grasp, dtype=np.float64).reshape(3).copy()
    tip[2] += float(cfg.GRASP_LIFT_CLEARANCE_M) + abs(float(cfg.GRASP_DESCEND_CLEARANCE_M))
    return tip


def live_port_features(context):
    """Cached jack features posed with the live RJ45 group transform."""

    from insertion_features.port_features import world_features_from_rj45_group

    stage = context.services["stage"]
    spec = context.services.get("station_spec")
    if spec is None:
        raise RuntimeError("station_spec required for port features")
    jack = world_features_from_rj45_group(
        stage, spec.jack_id, spec.port_pack_path
    )
    return jack.features


def live_crystal_features(context):
    """Cached crystal-head45 features posed with the live head transform."""

    from insertion_features.cable_features import world_features_from_crystal_head

    stage = context.services["stage"]
    spec = context.services.get("station_spec")
    if spec is None:
        raise RuntimeError("station_spec required for crystal features")
    head = world_features_from_crystal_head(stage, spec.path45)
    return head.features


def _rotate_about_world_z(vec: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a world vector about +Z by ``yaw_deg`` (CCW positive)."""

    v = np.asarray(vec, dtype=np.float64).reshape(3)
    rad = np.deg2rad(float(yaw_deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array(
        [c * v[0] - s * v[1], s * v[0] + c * v[1], float(v[2])],
        dtype=np.float64,
    )


def _signed_xy_yaw_deg(axis0: np.ndarray, axis1: np.ndarray) -> float:
    """Signed world-Z yaw (deg) taking XY projections from axis0 → axis1."""

    a0 = _normalize_vec(axis0)
    a1 = _normalize_vec(axis1)
    x0, y0 = float(a0[0]), float(a0[1])
    x1, y1 = float(a1[0]), float(a1[1])
    n0 = float(np.hypot(x0, y0))
    n1 = float(np.hypot(x1, y1))
    if n0 < 1e-9 or n1 < 1e-9:
        return 0.0
    x0, y0 = x0 / n0, y0 / n0
    x1, y1 = x1 / n1, y1 / n1
    return float(np.rad2deg(np.arctan2(x0 * y1 - y0 * x1, x0 * x1 + y0 * y1)))


def _maneuver_insertion_axes(context, port, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Start/end crystal insertion axes for lift→offset (world metres directions).

    Start: live head45 axis (fallback side-keyed ±Y). End: live port axis,
    flipped so travel into the jack is toward −X when needed.
    """

    side_key = cfg._normalize_side_key(side)
    expected0 = (
        np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if side_key == "negative"
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    try:
        crystal = live_crystal_features(context)
        axis0 = _normalize_vec(crystal.insertion_axis)
        context.services["crystal_features"] = crystal
    except Exception as exc:
        print(
            f"[BT MANEUVER{_station_tag(context)}] live crystal axis failed "
            f"({exc}); using side fallback {expected0}"
        )
        axis0 = expected0.copy()
    if float(np.dot(axis0, expected0)) < 0.0:
        axis0 = -axis0

    axis1 = _normalize_vec(port.insertion_axis)
    # Jack openings face the cables (+X); insertion travel into the pack is −X.
    if float(axis1[0]) > 0.1:
        axis1 = -axis1
    return axis0, axis1


def _maneuver_yaw_total_deg(axis0: np.ndarray, axis1: np.ndarray, side: str) -> float:
    """World-Z yaw for axis0→axis1 (short way).

    NegY −Y→−X is ~−90° (CW). PosY +Y→−X is ~+90° (CCW).
    """

    short = _signed_xy_yaw_deg(axis0, axis1)
    if abs(short) < 1e-6:
        # Axes already aligned in XY — use the conventional side yaw.
        return float(cfg.port_approach_yaw_deg(side))
    return short


def queue_lift_cable(context) -> None:
    """Lift the grasped neck clear of the CableBlocks (fingers stay closed)."""

    from ur5e_6x_cable_insertions.scene import spawn_station_debug_markers

    tool_ori, lula_ori = _grasp_tool_and_lula(context)
    tip_grasp = context.services.get("grasp_tip")
    if tip_grasp is None:
        tip_grasp = neck_grasp_tip_meters(context)
    else:
        tip_grasp = np.asarray(tip_grasp, dtype=np.float64).reshape(3)
    tip_lift = tip_grasp.copy()
    tip_lift[2] += float(cfg.GRASP_LIFT_CLEARANCE_M) + abs(
        float(cfg.GRASP_DESCEND_CLEARANCE_M)
    )
    hand_lift = hand_from_tip(tip_lift, tool_ori)
    context.services["lift_tip"] = tip_lift.copy()
    context.services["lift_hand"] = hand_lift.copy()

    spec = context.services.get("station_spec")
    if spec is not None:
        spawn_station_debug_markers(
            context.services["stage"],
            spec,
            tip_lift_m=tip_lift,
        )

    print(
        f"[BT LIFT{_station_tag(context)}] "
        f"tip_grasp={np.round(tip_grasp, 4)} tip_lift={np.round(tip_lift, 4)} "
        f"hand={np.round(hand_lift, 4)} "
        f"(hold_gripper + friction μ={cfg.GRASP_FRICTION_STATIC})"
    )
    context.services["monitor_cable_hold"] = True
    context.services["collision_abort_active"] = False
    monitor = context.services.get("collision_monitor")
    if monitor is not None and hasattr(monitor, "set_logging"):
        monitor.set_logging(False)
        monitor.clear_hit_flag()
    controller = context.services["motion_controller"]
    controller.clear_queue()
    # Brief re-assert close; arm lock is reused from post-grasp freeze so this
    # cannot re-capture a gravity-sagged pose.
    controller.add_gripper_command(
        "close",
        wait_frames=30,
        hold_arm=True,
    )
    # Cartesian linear rise — joint-space interp dips the tip mid-path and
    # bends the crystal before the lift clears the blocks.
    controller.add_cartesian_waypoint(
        hand_lift,
        lula_ori,
        target_is_hand=True,
        joint_interp=False,
        linear=True,
        linear_step=float(cfg.MANEUVER_LINEAR_STEP_M),
        hold_gripper=True,
        joint_steps=int(cfg.LIFT_JOINT_STEPS),
        max_frames=max(2400, int(cfg.LIFT_JOINT_STEPS) * 4),
        pos_tolerance=float(cfg.LIFT_POS_TOLERANCE_M),
        label=f"{context.step.name}: lift",
    )


def _arm_joints_from_robot(robot) -> np.ndarray | None:
    """Named UR5e arm joints from a live articulation, or None."""

    try:
        names = list(robot.dof_names)
        q = np.asarray(robot.get_joint_positions(), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    out = np.zeros(len(cfg.UR5E_ARM_JOINT_NAMES), dtype=np.float64)
    for i, name in enumerate(cfg.UR5E_ARM_JOINT_NAMES):
        if name not in names:
            return None
        out[i] = float(q[names.index(name)])
    return out


def _arm_joints_from_ik_action(robot, action) -> np.ndarray | None:
    if action is None or getattr(action, "joint_positions", None) is None:
        return None
    try:
        names = list(robot.dof_names)
    except Exception:
        return None
    jp = list(action.joint_positions)
    out = np.zeros(len(cfg.UR5E_ARM_JOINT_NAMES), dtype=np.float64)
    for i, name in enumerate(cfg.UR5E_ARM_JOINT_NAMES):
        if name not in names:
            return None
        idx = names.index(name)
        if idx >= len(jp) or jp[idx] is None:
            return None
        out[i] = float(jp[idx])
    return out


def _long_way_pan_delta(pan0: float, pan1: float) -> float:
    """Signed Δpan taking the long way around (≈270° when short is ≈90°)."""

    d_short = float(np.arctan2(np.sin(pan1 - pan0), np.cos(pan1 - pan0)))
    if abs(d_short) < 1e-9:
        # Already aligned — still spin nearly 270° CCW for clearance.
        return float(np.deg2rad(cfg.PORT_NEGY_BASE_SWING_DEG))
    return d_short - float(np.sign(d_short)) * 2.0 * np.pi


def _neg_y_base_sweep_arm_waypoints(
    q_start: np.ndarray, q_end: np.ndarray
) -> list[np.ndarray]:
    """Joint waypoints: long-way shoulder_pan + mid clearance on lift/elbow."""

    q0 = np.asarray(q_start, dtype=np.float64).reshape(6)
    q1 = np.asarray(q_end, dtype=np.float64).reshape(6)
    d_pan = _long_way_pan_delta(float(q0[0]), float(q1[0]))
    n = max(3, int(cfg.PORT_NEGY_BASE_WAYPOINTS))
    lift_bump = float(cfg.PORT_NEGY_CLEARANCE_SHOULDER_LIFT_DELTA)
    elbow_bump = float(
        getattr(cfg, "PORT_NEGY_CLEARANCE_ELBOW_DELTA", 0.35)
    )
    waypoints: list[np.ndarray] = []
    for i in range(1, n + 1):
        t = float(i) / float(n)
        s = t * t * (3.0 - 2.0 * t)
        clear = float(np.sin(np.pi * t))
        q = (1.0 - s) * q0 + s * q1
        q[0] = float(q0[0]) + s * d_pan
        q[1] = float(q[1]) + lift_bump * clear
        q[2] = float(q[2]) + elbow_bump * clear
        waypoints.append(q.copy())
    # Force exact IK end joints on the last sample; keep pan continuous (unwrapped).
    waypoints[-1] = q1.copy()
    waypoints[-1][0] = float(q0[0]) + d_pan
    return waypoints


def _ik_arm_for_hand(
    controller,
    hand_m: np.ndarray,
    lula_ori: np.ndarray,
    stage,
    warm_start_arm: np.ndarray | None = None,
) -> np.ndarray | None:
    """Lula IK for hand pose. Optional arm warm-start via temporary joint write."""

    from ur5e_6x_cable_insertions.runtime_support import meters_to_stage

    mpu = meters_per_unit(stage)
    hand_stage = meters_to_stage(hand_m, mpu)
    robot = getattr(controller, "_robot", None)
    saved = None
    if warm_start_arm is not None and robot is not None:
        try:
            names = list(robot.dof_names)
            saved = np.asarray(robot.get_joint_positions(), dtype=np.float64).copy()
            q = saved.copy()
            arm = np.asarray(warm_start_arm, dtype=np.float64).reshape(-1)
            for i, name in enumerate(cfg.UR5E_ARM_JOINT_NAMES):
                if i >= arm.size or name not in names:
                    continue
                q[names.index(name)] = float(arm[i])
            robot.set_joint_positions(q)
        except Exception:
            saved = None
    try:
        try:
            action, success = controller._art_kinematics.compute_inverse_kinematics(
                target_position=hand_stage,
                target_orientation=np.asarray(lula_ori, dtype=np.float64),
                position_tolerance=float(
                    getattr(controller, "_pos_tolerance", cfg.HOVER_IK_POS_TOLERANCE_M)
                ),
                orientation_tolerance=float(
                    getattr(controller, "_ori_tolerance", cfg.HOVER_IK_ORI_TOLERANCE_RAD)
                ),
            )
        except Exception:
            return None
        if not success:
            return None
        return _arm_joints_from_ik_action(controller._robot, action)
    finally:
        if saved is not None and robot is not None:
            try:
                robot.set_joint_positions(saved)
            except Exception:
                pass


def _plan_pan_reconfig_arm_waypoints(
    controller,
    hand_m: np.ndarray,
    lula_ori: np.ndarray,
    stage,
    *,
    delta_rad: float,
    samples: int,
    pan_tol_rad: float,
) -> list[np.ndarray]:
    """Hold tip/ori; swing shoulder_pan by ``delta_rad`` over ``samples`` IK seeds."""

    q0 = _ik_arm_for_hand(controller, hand_m, lula_ori, stage)
    if q0 is None:
        return []
    n = max(1, int(samples))
    out: list[np.ndarray] = []
    for s in range(1, n + 1):
        frac = float(s) / float(n)
        pan_target = float(q0[0]) + frac * float(delta_rad)
        seed = q0.copy()
        seed[0] = pan_target
        q = _ik_arm_for_hand(
            controller, hand_m, lula_ori, stage, warm_start_arm=seed
        )
        if q is None:
            continue
        pan_err = abs(
            float(np.arctan2(np.sin(q[0] - pan_target), np.cos(q[0] - pan_target)))
        )
        if pan_err > float(pan_tol_rad):
            # Lula snapped back — force pan on seed and retry once.
            forced = q.copy()
            forced[0] = pan_target
            q2 = _ik_arm_for_hand(
                controller, hand_m, lula_ori, stage, warm_start_arm=forced
            )
            if q2 is None:
                continue
            pan_err2 = abs(
                float(
                    np.arctan2(np.sin(q2[0] - pan_target), np.cos(q2[0] - pan_target))
                )
            )
            if pan_err2 > float(pan_tol_rad):
                continue
            q = q2
        out.append(q.copy())
    return out


def _queue_pan_reconfig_at_waypoint(
    context, controller, stage, wp: dict, *, tag: str
) -> int:
    """Queue joint waypoints that swing shoulder_pan while holding ``wp`` tip+ori.

    Returns number of joint waypoints queued (0 = skipped / IK failed).
    """

    tip = np.asarray(wp["tip"], dtype=np.float64).reshape(3)
    tool_ori = _normalize_quat(wp["orientation_wxyz"])
    lula_ori = cfg.lula_orientation_from_tool(tool_ori)
    hand = hand_from_tip(tip, tool_ori)
    delta = float(np.deg2rad(cfg.MANEUVER_PAN_RECONFIG_DELTA_DEG))
    samples = int(cfg.MANEUVER_PAN_RECONFIG_SAMPLES)
    joints = _plan_pan_reconfig_arm_waypoints(
        controller,
        hand,
        lula_ori,
        stage,
        delta_rad=delta,
        samples=samples,
        pan_tol_rad=float(cfg.MANEUVER_PAN_RECONFIG_PAN_TOL_RAD),
    )
    if not joints:
        print(
            f"[BT MANEUVER{tag}] pan-reconfig IK failed — continuing cartesian"
        )
        return 0
    steps = int(cfg.MANEUVER_PAN_RECONFIG_JOINT_STEPS)
    for i, q in enumerate(joints, start=1):
        controller.add_joint_waypoint(
            q,
            joint_steps=max(40, steps),
            hold_gripper=True,
            label=f"{context.step.name}: port-pan-reconfig-{i}/{len(joints)}",
        )
    print(
        f"[BT MANEUVER{tag}] pan-reconfig "
        f"delta={cfg.MANEUVER_PAN_RECONFIG_DELTA_DEG:.1f}° "
        f"samples={len(joints)}/{samples} tip={np.round(tip, 4)}"
    )
    return len(joints)


def queue_maneuver_to_port_offset(context) -> None:
    """After TipLift: few TipLift→TipOffset poses with crystal axis −Y→−X.

    TipOffset is copper-pin YZ + +X standoff. Orientation is a pure world-Z yaw
    (grasp tilt / Z ori held), then a final wrist tilt so insertion_axis is −X.
    """

    from ur5e_6x_cable_insertions.clearance import (
        collect_maneuver_obstacle_aabbs,
        densify_polyline_with_t,
        discover_clear_tip_polyline,
        tip_hits_obstacles,
    )
    from ur5e_6x_cable_insertions.scene import (
        spawn_maneuver_orientation_markers,
        spawn_station_debug_markers,
    )
    from ur5e_6x_cable_insertions import maneuver_cache

    controller = context.services["motion_controller"]
    stage = context.services["stage"]
    spec = context.services.get("station_spec")
    side = getattr(spec, "side", "positive") if spec is not None else "positive"
    side_key = cfg._normalize_side_key(side)
    station_id = getattr(spec, "station_id", None) or context.services.get(
        "station_id", "unknown"
    )

    tool_start, _lula_start = _grasp_tool_and_lula(context)
    tip_lift = _measure_post_lift_tip(context)
    context.services["lift_tip"] = tip_lift.copy()
    port = live_port_features(context)
    crystal = live_crystal_features(context)
    context.services["crystal_features"] = crystal
    cached_station = None
    if bool(cfg.MANEUVER_CACHE_LOAD):
        cached_station = maneuver_cache.load_station(
            str(station_id), path=cfg.MANEUVER_CACHE_PATH
        )
    axis0, axis1 = _maneuver_insertion_axes(context, port, side)
    yaw_total = _maneuver_yaw_total_deg(axis0, axis1, side)
    # Keep grasp tilt: lift→offset is a pure world-Z yaw (−Y→−X). Pitching the
    # wrist flat was wrecking Z orientation and the TipOffset poses.
    grasp_tilt = float(cfg.GRASP_TILT_FROM_DOWN_DEG)
    maneuver_tilt = float(
        getattr(cfg, "MANEUVER_TOOL_TILT_FROM_DOWN_DEG", grasp_tilt)
    )
    tool_yawed = _yaw_about_world_z(tool_start, yaw_total)
    if abs(float(maneuver_tilt) - float(grasp_tilt)) < 1e-3:
        tool_end = tool_yawed.copy()
        axis_yawed = _normalize_vec(_rotate_about_world_z(axis0, yaw_total))
    else:
        tool_end = _pitch_tool_z_to_tilt(tool_yawed, maneuver_tilt)
        axis_yawed = _rotate_about_world_z(axis0, yaw_total)
        tool_z0 = _rotate_vec_by_quat(
            tool_yawed, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        tool_z1 = _rotate_vec_by_quat(
            tool_end, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        q_pitch_end = _quat_from_two_vectors(tool_z0, tool_z1)
        axis_yawed = _normalize_vec(_rotate_vec_by_quat(q_pitch_end, axis_yawed))
    if float(np.dot(axis_yawed, axis1)) < 0.0:
        axis_yawed = -axis_yawed
    # TipOffset: crystal axis → world −X only (no flatten roll about the cable).
    tool_align = tool_end.copy()
    axis_align = _normalize_vec(axis_yawed)
    if bool(getattr(cfg, "PORT_OFFSET_ALIGN_AXIS_TO_X", True)):
        tool_align, axis_align = _tool_ori_align_axis_to_world_x(
            tool_end,
            axis_yawed,
            prefer="-x",
            flatten_xy=bool(getattr(cfg, "PORT_OFFSET_FLATTEN_TOOL_Z", False)),
        )
    lula_align = cfg.lula_orientation_from_tool(tool_align)

    # TipOffset tip: place *crystal mating center* at port mating YZ with +X
    # standoff (rigid grasp). Copper-pin fingertip YZ left the crystal ~6 mm low.
    pin_tip = compute_tip_offset_meters(
        stage, spec, port, float(cfg.PORT_APPROACH_X_OFFSET_M)
    )
    tip_end, desired_crystal = tip_offset_for_crystal_mating(
        tip_now=tip_lift,
        tool_ori_now=tool_start,
        crystal_mc_now=crystal.mating_center,
        tool_ori_tgt=tool_align,
        port_mc=port.mating_center,
        standoff_m=float(cfg.PORT_APPROACH_X_OFFSET_M),
    )
    tip_end = tip_end.copy()
    tip_end[2] += float(getattr(cfg, "PORT_OFFSET_Z_BIAS_M", 0.0))
    print(
        f"[BT MANEUVER{_station_tag(context)}] TipOffset from crystal mating "
        f"(not fingertip/pin)\n"
        f"  port_mc={np.round(port.mating_center, 4)} "
        f"desired_crystal={np.round(desired_crystal, 4)}\n"
        f"  crystal_now={np.round(crystal.mating_center, 4)} "
        f"tip_lift={np.round(tip_lift, 4)}\n"
        f"  tip_end={np.round(tip_end, 4)} "
        f"(pin_ref={np.round(pin_tip, 4)} Δ={np.round(tip_end - pin_tip, 4)})"
    )

    context.services["port_features"] = port
    context.services["port_offset_tip"] = tip_end.copy()
    context.services["port_offset_crystal"] = desired_crystal.copy()
    context.services["port_approach_orientation"] = lula_align.copy()
    context.services["port_approach_tool_orientation"] = tool_align.copy()
    context.services["port_offset_yaw_tool_orientation"] = tool_end.copy()
    context.services["maneuver_axis_start"] = axis0.copy()
    context.services["maneuver_axis_end"] = axis_align.copy()
    context.services["maneuver_yaw_total_deg"] = float(yaw_total)
    context.services["maneuver_reached_offset"] = False

    use_base_sweep = (
        side_key == "negative" and bool(getattr(cfg, "PORT_NEGY_USE_BASE_SWEEP", True))
    )
    swing = (
        f"axis yaw {yaw_total:+.1f}° (joint pan)"
        if use_base_sweep
        else f"axis yaw {yaw_total:+.1f}°"
    )

    obstacles = collect_maneuver_obstacle_aabbs(
        stage, spec, tip_start=tip_lift, tip_end=tip_end
    )

    if spec is not None:
        spawn_station_debug_markers(
            stage,
            spec,
            tip_lift_m=tip_lift,
            tip_offset_m=tip_end,
        )

    def _ori_axis_at(t: float) -> tuple[np.ndarray, np.ndarray]:
        # World-Z yaw only when maneuver tilt == grasp tilt (Z ori stays put).
        yaw = float(t) * yaw_total
        tool_yawed_t = _yaw_about_world_z(tool_start, yaw)
        if abs(float(maneuver_tilt) - float(grasp_tilt)) < 1e-3:
            tool_ori = tool_yawed_t
            axis = _rotate_about_world_z(axis0, yaw)
            return tool_ori, _normalize_vec(axis)
        tilt = (1.0 - float(t)) * grasp_tilt + float(t) * maneuver_tilt
        tool_ori = _pitch_tool_z_to_tilt(tool_yawed_t, tilt)
        axis = _rotate_about_world_z(axis0, yaw)
        z0 = _rotate_vec_by_quat(
            tool_yawed_t, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        z1 = _rotate_vec_by_quat(
            tool_ori, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        if float(np.linalg.norm(z0 - z1)) > 1e-9:
            axis = _rotate_vec_by_quat(_quat_from_two_vectors(z0, z1), axis)
        return tool_ori, _normalize_vec(axis)

    def _wp_dict(
        t: float,
        tip: np.ndarray,
        tool_ori: np.ndarray,
        axis: np.ndarray,
        *,
        from_cache: bool,
        label: str | None = None,
    ) -> dict:
        is_final = float(t) >= 1.0 - 1e-9
        return {
            "t": float(t),
            "tip": np.asarray(tip, dtype=np.float64).reshape(3),
            "orientation_wxyz": _normalize_quat(tool_ori),
            "insertion_axis": _normalize_vec(axis),
            "label": (
                label
                if label is not None
                else ("port-offset" if is_final else f"port-maneuver-{t:.2f}")
            ),
            "from_cache": bool(from_cache),
        }

    strategy = "simple-lift-to-offset"
    poly: list[np.ndarray] = []
    plan: list[dict] = []
    known = 0.0
    attempt_to = 1.0
    use_simple = bool(getattr(cfg, "MANEUVER_SIMPLE_LIFT_TO_OFFSET", True))

    if use_simple:
        n_wp = max(2, int(getattr(cfg, "MANEUVER_SIMPLE_WAYPOINTS", 4)))
        for i in range(n_wp):
            t = float(i) / float(n_wp - 1)
            tip = (1.0 - t) * tip_lift + t * tip_end
            tool_ori, axis = _ori_axis_at(t)
            if float(t) >= 1.0 - 1e-9:
                tool_ori = tool_end.copy()
                axis = _normalize_vec(axis_yawed)
            plan.append(_wp_dict(t, tip, tool_ori, axis, from_cache=False))
            poly.append(tip.copy())
        strategy = f"simple-lift-to-offset n={n_wp} yaw={yaw_total:+.1f}°"
        context.services["maneuver_clearance_strategy"] = strategy
        context.services["maneuver_clearance_vias"] = [p.copy() for p in poly]
        print(
            f"[BT MANEUVER{_station_tag(context)}] simple TipLift→TipOffset "
            f"({n_wp} poses, crystal-mating tip_offset, yaw-only −Y→−X)"
        )
    else:
        poly, strategy = discover_clear_tip_polyline(tip_lift, tip_end, obstacles)
        tip_samples = densify_polyline_with_t(poly)
        tool0, axis_t0 = _ori_axis_at(0.0)
        plan.append(_wp_dict(0.0, tip_lift, tool0, axis_t0, from_cache=False))
        for t, tip in tip_samples:
            if float(t) <= 1e-9:
                continue
            tool_ori, axis = _ori_axis_at(t)
            hit = tip_hits_obstacles(
                tip, obstacles, tip_end=tip_end, tool_ori=tool_ori
            )
            if hit is not None and float(t) < 1.0 - 1e-9:
                print(
                    f"[BT MANEUVER{_station_tag(context)}] WARN sample t={t:.3f} "
                    f"still overlaps {hit} — keeping path anyway"
                )
            plan.append(_wp_dict(t, tip, tool_ori, axis, from_cache=False))
        if not plan or float(plan[-1]["t"]) < 1.0 - 1e-9:
            plan.append(
                _wp_dict(1.0, tip_end, tool_end, axis_yawed, from_cache=False)
            )
        context.services["maneuver_clearance_strategy"] = strategy
        context.services["maneuver_clearance_vias"] = [p.copy() for p in poly]

    by_t: dict[float, dict] = {}
    for wp in plan:
        by_t[round(float(wp["t"]), 5)] = wp
    plan = [by_t[k] for k in sorted(by_t)]
    if plan:
        plan[-1]["label"] = "port-offset-arrive"
        plan[-1]["tip"] = tip_end.copy()
        plan[-1]["orientation_wxyz"] = _normalize_quat(tool_end)
        plan[-1]["insertion_axis"] = _normalize_vec(axis_yawed)
    if bool(getattr(cfg, "PORT_OFFSET_ALIGN_AXIS_TO_X", True)) and plan:
        plan.append(
            _wp_dict(
                1.0,
                tip_end,
                tool_align,
                axis_align,
                from_cache=False,
                label="port-offset",
            )
        )
    elif plan:
        plan[-1]["label"] = "port-offset"
        plan[-1]["orientation_wxyz"] = _normalize_quat(tool_align)
        plan[-1]["insertion_axis"] = _normalize_vec(axis_align)

    if spec is not None:
        spawn_maneuver_orientation_markers(stage, spec, plan)

    print(
        f"[BT MANEUVER{_station_tag(context)}] port offset "
        f"jack={getattr(spec, 'jack_id', '?')} side={side} swing={swing}\n"
        f"  axes: start={np.round(axis0, 4)} → yawed={np.round(axis_yawed, 4)} "
        f"→ align={np.round(axis_align, 4)} yaw_total={yaw_total:+.1f}°\n"
        f"  clearance: strategy={strategy} obstacles={len(obstacles)} "
        f"vias={len(poly)}\n"
        f"  path: TipLift→TipOffset (−X axis) ({len(plan)} poses) "
        f"cache known={known:.2f} attempt_to={attempt_to:.2f}\n"
        f"  mode={'joint pan' if use_base_sweep else 'cartesian'}\n"
        f"  mating={np.round(port.mating_center, 4)} "
        f"standoff=+{cfg.PORT_APPROACH_X_OFFSET_M:.3f}m "
        f"tip_lift={np.round(tip_lift, 4)} tip_offset={np.round(tip_end, 4)} "
        f"axis_align={np.round(axis_align, 4)}"
    )
    n_show = min(12, len(obstacles))
    for path in (o[2] for o in obstacles[:n_show]):
        print(f"  [obstacle] {path}")
    if len(obstacles) > n_show:
        print(f"  [obstacle] … +{len(obstacles) - n_show} more")
    for i, p in enumerate(poly):
        print(f"  [via-{i}] tip={np.round(p, 4)}")
    for wp in plan:
        print(
            f"  [{wp['label']}] t={wp['t']:.3f} "
            f"tip={np.round(wp['tip'], 4)} "
            f"axis={np.round(wp['insertion_axis'], 4)} "
            f"ori={np.round(wp['orientation_wxyz'], 4)}"
        )

    # Motion queue skips t=0 (already at lift).
    motion_plan = [wp for wp in plan if float(wp["t"]) > 1e-9]
    context.services["monitor_cable_hold"] = True
    context.services["collision_abort_active"] = False
    monitor = context.services.get("collision_monitor")
    if monitor is not None and hasattr(monitor, "set_logging"):
        # Cable-mesh logging is insert-only.
        monitor.set_logging(False)
        monitor.clear_hit_flag()
    context.services["maneuver_cache_station_id"] = str(station_id)
    context.services["maneuver_tip_offset"] = tip_end.copy()
    context.services["maneuver_angled_plan"] = motion_plan
    controller.clear_queue()

    if use_base_sweep and motion_plan:
        robot = context.services.get("robot") or getattr(controller, "_robot", None)
        q_start = _arm_joints_from_robot(robot) if robot is not None else None
        lift_bump = float(cfg.PORT_NEGY_CLEARANCE_SHOULDER_LIFT_DELTA)
        elbow_bump = float(getattr(cfg, "PORT_NEGY_CLEARANCE_ELBOW_DELTA", 0.35))
        queued = 0
        if q_start is None:
            print(
                f"[BT MANEUVER{_station_tag(context)}] NegY joint seed missing — "
                "cartesian path"
            )
            use_base_sweep = False
        else:
            pan0 = float(q_start[0])
            for wp in motion_plan:
                tip = np.asarray(wp["tip"], dtype=np.float64).reshape(3)
                tool_ori = _normalize_quat(wp["orientation_wxyz"])
                lula_ori = cfg.lula_orientation_from_tool(tool_ori)
                hand = hand_from_tip(tip, tool_ori)
                q = _ik_arm_for_hand(controller, hand, lula_ori, stage)
                if q is None:
                    print(
                        f"[BT MANEUVER{_station_tag(context)}] NegY IK miss at "
                        f"t={wp['t']:.2f} — cartesian fallback"
                    )
                    use_base_sweep = False
                    controller.clear_queue()
                    queued = 0
                    break
                t = float(wp["t"])
                q = q.copy()
                # Do not override IK joints — that is what made the crystal
                # ignore tip waypoints and swing through Rack_Core.
                short = str(wp["label"])
                is_tip_offset = short == "port-offset"
                is_offset = short in ("port-offset", "port-offset-arrive")
                controller.add_joint_waypoint(
                    q,
                    joint_steps=max(
                        80,
                        int(
                            cfg.PORT_OFFSET_ALIGN_JOINT_STEPS
                            if is_tip_offset
                            else (
                                cfg.MANEUVER_OFFSET_JOINT_STEPS
                                if is_offset
                                else cfg.PORT_NEGY_BASE_JOINT_STEPS
                            )
                        ),
                    ),
                    hold_gripper=True,
                    label=f"{context.step.name}: {short}",
                )
                queued += 1
            if use_base_sweep and queued == len(motion_plan):
                context.services["port_offset_hand"] = hand_from_tip(
                    tip_end, tool_align
                ).copy()
                return
            if queued:
                controller.clear_queue()

    print(
        f"[BT MANEUVER{_station_tag(context)}] executing cartesian linear tip path "
        f"({len(motion_plan)} segments, step={cfg.MANEUVER_LINEAR_STEP_M:.4f}m) "
        f"via_tol={cfg.MANEUVER_VIA_POS_TOLERANCE_M:.3f}m/"
        f"{np.rad2deg(cfg.MANEUVER_VIA_ORI_TOLERANCE_RAD):.0f}° "
        f"late(≥wp{int(getattr(cfg, 'MANEUVER_LATE_WP_INDEX', 40))})="
        f"{float(getattr(cfg, 'MANEUVER_LATE_POS_TOLERANCE_M', 0.008)):.3f}m/"
        f"{np.rad2deg(float(getattr(cfg, 'MANEUVER_LATE_ORI_TOLERANCE_RAD', 0.08))):.0f}° "
        f"offset_tol={cfg.MANEUVER_OFFSET_POS_TOLERANCE_M:.3f}m/"
        f"{np.rad2deg(cfg.MANEUVER_OFFSET_ORI_TOLERANCE_RAD):.0f}°"
    )

    def _queue_cartesian_segment(wps: list[dict], *, index_offset: int = 0) -> None:
        late_idx = int(getattr(cfg, "MANEUVER_LATE_WP_INDEX", 40))
        late_pos = float(getattr(cfg, "MANEUVER_LATE_POS_TOLERANCE_M", 0.008))
        late_ori = float(getattr(cfg, "MANEUVER_LATE_ORI_TOLERANCE_RAD", 0.08))
        late_frames = int(getattr(cfg, "MANEUVER_LATE_MAX_FRAMES", 1200))
        for i, wp in enumerate(wps):
            plan_index = index_offset + i + 1  # 1-based among motion_plan
            tip = np.asarray(wp["tip"], dtype=np.float64).reshape(3)
            tool_ori = _normalize_quat(wp["orientation_wxyz"])
            lula_ori = cfg.lula_orientation_from_tool(tool_ori)
            short = str(wp["label"])
            is_tip_offset = short == "port-offset"
            is_arrive = short == "port-offset-arrive"
            hand = hand_from_tip(tip, tool_ori)
            # Task-space linear segments so the crystal tracks the clearance tips.
            # Mid vias use loose pos/ori so we keep moving; TipOffset / late WPs
            # are tighter so the arm actually reaches the planned pose.
            if is_tip_offset:
                # One joint-interp from arrive→align ori swings the tip through a
                # large free-space arc (log: Y −1.03→−0.97 into the chassis).
                # Blend ori at fixed TipOffset XYZ with many small joint steps.
                n_blend = max(
                    1, int(getattr(cfg, "PORT_OFFSET_ORI_BLEND_STEPS", 12))
                )
                blend_steps = max(
                    20,
                    int(getattr(cfg, "PORT_OFFSET_ORI_BLEND_JOINT_STEPS", 40)),
                )
                blend_frames = max(
                    blend_steps + 30,
                    int(cfg.PORT_OFFSET_ALIGN_MAX_FRAMES) // max(1, n_blend),
                )
                # Previous waypoint ori is arrive (tool_end); recover from plan.
                ori_from = None
                if i > 0:
                    ori_from = _normalize_quat(wps[i - 1]["orientation_wxyz"])
                if ori_from is None:
                    ori_from = _normalize_quat(
                        context.services.get(
                            "port_offset_yaw_tool_orientation", tool_ori
                        )
                    )
                for k in range(1, n_blend + 1):
                    tk = float(k) / float(n_blend)
                    ori_k = _quat_slerp(ori_from, tool_ori, tk)
                    lula_k = cfg.lula_orientation_from_tool(ori_k)
                    hand_k = hand_from_tip(tip, ori_k)
                    controller.add_cartesian_waypoint(
                        hand_k,
                        lula_k,
                        target_is_hand=True,
                        joint_interp=True,
                        linear=False,
                        linear_step=float(cfg.MANEUVER_LINEAR_STEP_M),
                        hold_gripper=True,
                        joint_steps=blend_steps,
                        max_frames=blend_frames,
                        pos_tolerance=float(cfg.PORT_OFFSET_ALIGN_POS_TOLERANCE_M),
                        ori_tolerance=float(cfg.PORT_OFFSET_ALIGN_ORI_TOLERANCE_RAD),
                        label=f"{context.step.name}: port-offset-{k}/{n_blend}",
                    )
                continue
            if is_arrive:
                joint_steps = int(cfg.MANEUVER_OFFSET_JOINT_STEPS)
                max_frames = max(int(cfg.MANEUVER_OFFSET_MAX_FRAMES), late_frames)
                pos_tol = min(float(cfg.MANEUVER_OFFSET_POS_TOLERANCE_M), late_pos)
                ori_tol = min(float(cfg.MANEUVER_OFFSET_ORI_TOLERANCE_RAD), late_ori)
                use_joint = False
            elif plan_index >= late_idx:
                joint_steps = max(80, int(cfg.MANEUVER_YAW_JOINT_STEPS))
                max_frames = late_frames
                pos_tol = late_pos
                ori_tol = late_ori
                use_joint = False
            else:
                joint_steps = max(80, int(cfg.MANEUVER_YAW_JOINT_STEPS))
                max_frames = int(cfg.MANEUVER_VIA_MAX_FRAMES)
                pos_tol = float(cfg.MANEUVER_VIA_POS_TOLERANCE_M)
                ori_tol = float(cfg.MANEUVER_VIA_ORI_TOLERANCE_RAD)
                use_joint = False
            controller.add_cartesian_waypoint(
                hand,
                lula_ori,
                target_is_hand=True,
                joint_interp=bool(use_joint),
                linear=not bool(use_joint),
                linear_step=float(cfg.MANEUVER_LINEAR_STEP_M),
                hold_gripper=True,
                joint_steps=joint_steps,
                max_frames=max_frames,
                pos_tolerance=pos_tol,
                ori_tolerance=ori_tol,
                label=f"{context.step.name}: {short}",
            )

    pan_reconfig = (
        side_key == "negative"
        and bool(getattr(cfg, "MANEUVER_PAN_RECONFIG_ENABLE", True))
        and not use_base_sweep
    )
    n_after = int(getattr(cfg, "MANEUVER_PAN_RECONFIG_AFTER_WP", 8))
    if pan_reconfig and n_after > 0 and len(motion_plan) >= n_after:
        before = motion_plan[:n_after]
        after = motion_plan[n_after:]
        _queue_cartesian_segment(before, index_offset=0)
        _queue_pan_reconfig_at_waypoint(
            context,
            controller,
            stage,
            before[-1],
            tag=_station_tag(context),
        )
        _queue_cartesian_segment(after, index_offset=n_after)
    else:
        if pan_reconfig and n_after > 0:
            print(
                f"[BT MANEUVER{_station_tag(context)}] pan-reconfig skipped "
                f"(plan_len={len(motion_plan)} < after_wp={n_after})"
            )
        _queue_cartesian_segment(motion_plan)
    context.services["port_offset_hand"] = hand_from_tip(tip_end, tool_align).copy()


def _quat_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Unit quaternion from world-frame axis-angle vector (rad)."""

    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = aa / theta
    half = 0.5 * theta
    s = float(np.sin(half))
    return _normalize_quat(
        np.array(
            [np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s],
            dtype=np.float64,
        )
    )


def _align_insert_tool_ori(context) -> np.ndarray:
    tool = context.services.get("port_approach_tool_orientation")
    if tool is None:
        tool = context.services.get("grasp_tool_orientation")
    if tool is None:
        tool, _ = _grasp_tool_and_lula(context)
    return _normalize_quat(tool)


def _offset_tip_for_reach(context, tool_ori: np.ndarray) -> np.ndarray:
    """Tip to hold during reachability reconfig (prefer live FK, else TipOffset)."""

    controller = context.services.get("motion_controller")
    if controller is not None:
        try:
            tip = measured_fingertip_meters(controller, tool_ori)
            if np.all(np.isfinite(tip)):
                return np.asarray(tip, dtype=np.float64).reshape(3)
        except Exception:
            pass
    for key in ("port_offset_tip", "maneuver_tip_offset"):
        tip = context.services.get(key)
        if tip is not None:
            return np.asarray(tip, dtype=np.float64).reshape(3)
    raise RuntimeError("no TipOffset tip available for insert reachability")


def _seat_tip_for_crystal_at_port(
    *,
    tip_now: np.ndarray,
    tool_ori: np.ndarray,
    crystal_mc: np.ndarray,
    port_mc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Tip that places crystal mating center on the port mating center (rigid grasp)."""

    tip_seat, desired = tip_offset_for_crystal_mating(
        tip_now=tip_now,
        tool_ori_now=tool_ori,
        crystal_mc_now=crystal_mc,
        tool_ori_tgt=tool_ori,
        port_mc=port_mc,
        standoff_m=0.0,
    )
    return tip_seat, desired


def _try_ik_seat_tip(
    context,
    tip_seat: np.ndarray,
    tool_ori: np.ndarray,
    *,
    warm_start_arm: np.ndarray | None = None,
) -> np.ndarray | None:
    """Lula IK for the seated tip (crystal mating ≡ port mating)."""

    controller = context.services["motion_controller"]
    stage = context.services["stage"]
    lula_ori = cfg.lula_orientation_from_tool(tool_ori)
    hand = hand_from_tip(tip_seat, tool_ori)
    return _ik_arm_for_hand(
        controller, hand, lula_ori, stage, warm_start_arm=warm_start_arm
    )


def _ensure_insert_reachability(context) -> str:
    """Precondition at TipOffset: IK must reach crystal-mating ≡ port-mating.

    Returns ``\"ok\"``, ``\"queued_reconfig\"``, or ``\"fail\"``.
    On fail-to-reach, queues shoulder_pan reconfig while holding the offset tip,
    then re-checks seat IK with the new arm seeds.
    """

    if context.services.get("insert_reach_ok"):
        return "ok"
    if not bool(getattr(cfg, "INSERT_REACH_CHECK", True)):
        context.services["insert_reach_ok"] = True
        return "ok"

    tag = _station_tag(context)
    controller = context.services["motion_controller"]
    stage = context.services["stage"]
    tool_ori = _align_insert_tool_ori(context)

    try:
        crystal = live_crystal_features(context)
        port = live_port_features(context)
    except Exception as exc:
        context.services["abort_simulation"] = True
        context.services["abort_reason"] = f"insert reach feature read failed: {exc}"
        print(f"[BT INSERT{tag}] FAIL reach features: {exc}")
        return "fail"

    tip_off = _offset_tip_for_reach(context, tool_ori)
    c_mc = np.asarray(crystal.mating_center, dtype=np.float64).reshape(3)
    p_mc = np.asarray(port.mating_center, dtype=np.float64).reshape(3)
    tip_seat, desired_c = _seat_tip_for_crystal_at_port(
        tip_now=tip_off,
        tool_ori=tool_ori,
        crystal_mc=c_mc,
        port_mc=p_mc,
    )
    context.services["insert_seat_tip"] = tip_seat.copy()
    context.services["insert_seat_crystal"] = desired_c.copy()

    q_seat = _try_ik_seat_tip(context, tip_seat, tool_ori)
    if q_seat is not None:
        context.services["insert_reach_ok"] = True
        print(
            f"[BT INSERT{tag}] reach OK — seat IK found "
            f"tip_seat={np.round(tip_seat, 4)} "
            f"crystal→port={np.round(desired_c, 4)} "
            f"(from tip_off={np.round(tip_off, 4)})"
        )
        return "ok"

    print(
        f"[BT INSERT{tag}] seat IK miss — crystal mating cannot reach port mating "
        f"from current arm config\n"
        f"  tip_off={np.round(tip_off, 4)} tip_seat={np.round(tip_seat, 4)}\n"
        f"  crystal_mc={np.round(c_mc, 4)} port_mc={np.round(p_mc, 4)} "
        f"desired_crystal={np.round(desired_c, 4)}"
    )

    deltas = context.services.get("insert_reach_deltas_left")
    if deltas is None:
        deltas = list(getattr(cfg, "INSERT_REACH_RECONFIG_DELTAS_DEG", (-90.0, 90.0)))
    else:
        deltas = list(deltas)

    hand_off = hand_from_tip(tip_off, tool_ori)
    lula_off = cfg.lula_orientation_from_tool(tool_ori)
    samples = int(getattr(cfg, "INSERT_REACH_RECONFIG_SAMPLES", 6))
    pan_tol = float(getattr(cfg, "INSERT_REACH_RECONFIG_PAN_TOL_RAD", 0.35))
    steps = int(getattr(cfg, "INSERT_REACH_RECONFIG_JOINT_STEPS", 120))

    while deltas:
        delta_deg = float(deltas.pop(0))
        joints = _plan_pan_reconfig_arm_waypoints(
            controller,
            hand_off,
            lula_off,
            stage,
            delta_rad=float(np.deg2rad(delta_deg)),
            samples=samples,
            pan_tol_rad=pan_tol,
        )
        if not joints:
            print(
                f"[BT INSERT{tag}] reach reconfig δpan={delta_deg:+.1f}° "
                "IK failed — try next"
            )
            continue
        unlocked = False
        for seed in joints:
            if _try_ik_seat_tip(context, tip_seat, tool_ori, warm_start_arm=seed) is not None:
                unlocked = True
                break
        if not unlocked:
            print(
                f"[BT INSERT{tag}] reach reconfig δpan={delta_deg:+.1f}° "
                f"({len(joints)} seeds) still no seat IK — try next"
            )
            continue

        for i, q in enumerate(joints, start=1):
            controller.add_joint_waypoint(
                q,
                joint_steps=max(40, steps),
                hold_gripper=True,
                label=f"{context.step.name}: insert-reach-reconfig-{i}/{len(joints)}",
            )
        context.services["insert_reach_pending"] = True
        context.services["insert_reach_deltas_left"] = deltas
        print(
            f"[BT INSERT{tag}] reach reconfig queued δpan={delta_deg:+.1f}° "
            f"samples={len(joints)}/{samples} (hold tip_off, unlock seat IK)"
        )
        return "queued_reconfig"

    context.services["abort_simulation"] = True
    context.services["abort_reason"] = (
        "insert reachability: no IK for crystal mating → port mating "
        "after offset pan reconfigs"
    )
    context.services["align_insert_active"] = False
    print(
        f"[BT INSERT{tag}] FAIL reach — seat tip unreachable after pan reconfigs "
        f"tip_seat={np.round(tip_seat, 4)}"
    )
    return "fail"


def _begin_align_translate_after_reach(context) -> None:
    """Load insert cache (optional) and queue the first align/translate step."""

    from ur5e_6x_cable_insertions import insert_cache

    controller = context.services["motion_controller"]
    spec = context.services.get("station_spec")
    station_id = getattr(spec, "station_id", None) or context.services.get(
        "station_id", "unknown"
    )
    if bool(getattr(cfg, "INSERT_CACHE_LOAD", True)):
        try:
            cached = insert_cache.load_station(
                str(station_id), path=getattr(cfg, "INSERT_CACHE_PATH", None)
            )
        except Exception:
            cached = None
        if (
            cached is not None
            and bool(cached.get("verified"))
            and cached.get("waypoints")
        ):
            context.services["insert_cache_queue"] = list(cached["waypoints"])
            print(
                f"[BT INSERT{_station_tag(context)}] replaying "
                f"{len(cached['waypoints'])} cached translate poses"
            )

    mode = (
        "axis-only translate"
        if not bool(getattr(cfg, "ALIGN_INSERT_ENABLE_YZ", False))
        else "align→translate"
    )
    print(
        f"[BT INSERT{_station_tag(context)}] start {mode} "
        f"(step={cfg.INSERT_STEP_M:.4f}m along crystal axis; "
        f"touch_gap≤{cfg.MATING_TOUCH_GAP_M:.4f}m; "
        f"yz_align={bool(getattr(cfg, 'ALIGN_INSERT_ENABLE_YZ', False))})"
    )
    _queue_next_align_insert(context)
    if controller.is_done() and not context.services.get("port_inserted"):
        controller.add_hold_command(
            30, hold_gripper=True, label=f"{context.step.name}: insert-seed-hold"
        )


def _queue_align_insert_cartesian(
    context, tip: np.ndarray, tool_ori: np.ndarray, *, label: str
) -> None:
    controller = context.services["motion_controller"]
    lula_ori = cfg.lula_orientation_from_tool(tool_ori)
    hand = hand_from_tip(tip, tool_ori)
    # Pure tip translation (align-yz / insert-step): linear Cartesian keeps the
    # wrist ori fixed so joint-interp cannot swing the tip and spit the cable.
    # Only use joint-interp when the label implies a rotation nudge.
    want_rot = "rot" in str(label).lower() or "ori" in str(label).lower()
    controller.add_cartesian_waypoint(
        hand,
        lula_ori,
        target_is_hand=True,
        joint_interp=bool(want_rot),
        linear=not bool(want_rot),
        linear_step=float(cfg.ALIGN_STEP_LINEAR_STEP_M),
        hold_gripper=True,
        joint_steps=int(cfg.ALIGN_STEP_JOINT_STEPS),
        max_frames=int(cfg.ALIGN_STEP_MAX_FRAMES),
        pos_tolerance=float(cfg.ALIGN_STEP_POS_TOLERANCE_M),
        ori_tolerance=float(cfg.ALIGN_STEP_ORI_TOLERANCE_RAD),
        label=f"{context.step.name}: {label}",
    )


def _spawn_align_insert_debug(
    context,
    *,
    crystal,
    port,
    residual,
    tip_measured_m,
    tip_cmd_m=None,
    tool_ori=None,
    insert_dir=None,
) -> None:
    """Show one TipCmd marker at the next align/insert target (replaces prior)."""

    from ur5e_6x_cable_insertions.scene import spawn_insert_step_marker

    del port, residual, tip_measured_m, tool_ori
    spec = context.services.get("station_spec")
    stage = context.services.get("stage")
    if spec is None or stage is None or tip_cmd_m is None:
        return
    try:
        spawn_insert_step_marker(
            stage,
            spec,
            tip_cmd_m=tip_cmd_m,
            crystal=crystal,
            insert_dir=insert_dir,
        )
    except Exception as exc:
        print(f"[BT INSERT{_station_tag(context)}] debug markers skip: {exc}")


def _record_insert_pose(
    context,
    *,
    tip: np.ndarray,
    tool_ori: np.ndarray,
    axis: np.ndarray | None,
    kind: str,
    label: str,
) -> None:
    plan = context.services.setdefault("insert_path_plan", [])
    n = len(plan)
    plan.append(
        {
            "t": float(n + 1),
            "tip": np.asarray(tip, dtype=np.float64).reshape(3).copy(),
            "orientation_wxyz": _normalize_quat(tool_ori),
            "insertion_axis": (
                None
                if axis is None
                else np.asarray(axis, dtype=np.float64).reshape(3).copy()
            ),
            "kind": str(kind),
            "label": str(label),
        }
    )


def _save_insert_cache(context, *, verified: bool) -> None:
    if not bool(getattr(cfg, "INSERT_CACHE_SAVE", True)):
        return
    plan = context.services.get("insert_path_plan") or []
    if not plan:
        return
    spec = context.services.get("station_spec")
    station_id = getattr(spec, "station_id", None) or context.services.get(
        "station_id", "unknown"
    )
    try:
        from ur5e_6x_cable_insertions import insert_cache

        insert_cache.save_station(
            str(station_id),
            waypoints=plan,
            verified=bool(verified),
            path=getattr(cfg, "INSERT_CACHE_PATH", None),
        )
    except Exception as exc:
        print(f"[INSERT CACHE] save failed: {exc}")


def _queue_next_align_insert(context) -> None:
    """Translate along crystal axis in small steps (YZ align optional / off).

    TipOffset is the YZ align. Live loop records each tip pose into insert_cache.
    """

    from ur5e_6x_cable_insertions.alignment import (
        evaluate_alignment,
        insert_direction_crystal_neg_x,
        insert_target_tip,
    )

    controller = context.services["motion_controller"]

    # Replay cached align/translate poses first (skip live feature loop).
    cache_q = context.services.get("insert_cache_queue")
    if cache_q:
        wp = cache_q.pop(0)
        tip_tgt = np.asarray(wp["tip"], dtype=np.float64).reshape(3)
        tool_tgt = _normalize_quat(wp["orientation_wxyz"])
        axis = wp.get("insertion_axis")
        kind = str(wp.get("kind") or wp.get("label") or "cache")
        label = str(wp.get("label") or f"cache-{kind}")
        context.services["port_approach_tool_orientation"] = tool_tgt.copy()
        context.services["port_approach_orientation"] = cfg.lula_orientation_from_tool(
            tool_tgt
        ).copy()
        try:
            crystal = live_crystal_features(context)
        except Exception:
            crystal = None
        if crystal is not None:
            _spawn_align_insert_debug(
                context,
                crystal=crystal,
                port=None,
                residual=None,
                tip_measured_m=measured_fingertip_meters(controller, tool_tgt),
                tip_cmd_m=tip_tgt,
                tool_ori=tool_tgt,
                insert_dir=axis,
            )
        _queue_align_insert_cartesian(context, tip_tgt, tool_tgt, label=label)
        print(
            f"[BT INSERT{_station_tag(context)}] cache-replay "
            f"{label} tip={np.round(tip_tgt, 4)} remaining={len(cache_q)}"
        )
        if not cache_q:
            context.services["insert_cache_queue"] = None
            print(
                f"[BT INSERT{_station_tag(context)}] cache replay done; "
                "live translate-axis"
            )
        return

    try:
        crystal = live_crystal_features(context)
        port = live_port_features(context)
    except Exception as exc:
        context.services["abort_simulation"] = True
        context.services["abort_reason"] = f"align/translate feature read failed: {exc}"
        print(f"[BT INSERT{_station_tag(context)}] FAIL features: {exc}")
        context.services["align_insert_active"] = False
        from ur5e_6x_cable_insertions.insert_diagnostics import finish_insert_diagnostics

        finish_insert_diagnostics(context, reason="feature_read_failed")
        return

    residual = evaluate_alignment(
        crystal,
        port,
        latch_z_margin_m=float(cfg.LATCH_Z_MARGIN_M),
        mating_side_margin_m=float(cfg.MATING_SIDE_MARGIN_M),
        axis_dot_min=float(cfg.AXIS_DOT_MIN),
        mating_center_yz_tol_m=float(
            getattr(cfg, "MATING_CENTER_YZ_TOL_M", 0.0015)
        ),
        latch_y_margin_m=float(getattr(cfg, "LATCH_Y_MARGIN_M", cfg.MATING_SIDE_MARGIN_M)),
    )
    gap = float(residual.mating_gap_m)
    touch = float(cfg.MATING_TOUCH_GAP_M)
    frames = int(context.services.get("align_insert_frames", 0))
    log_every = max(1, int(cfg.ALIGN_LOG_EVERY_N))
    tool_ori = _align_insert_tool_ori(context)
    tip = measured_fingertip_meters(controller, tool_ori)
    c_mc = np.asarray(crystal.mating_center, dtype=np.float64).reshape(3)
    p_mc = np.asarray(port.mating_center, dtype=np.float64).reshape(3)
    yz_enabled = bool(getattr(cfg, "ALIGN_INSERT_ENABLE_YZ", False))

    # Seated: gap closed along insert axis. When YZ align is enabled, also require
    # mating-center YZ match; axis-only mode trusts TipOffset YZ.
    seated = abs(gap) <= touch
    if yz_enabled:
        seated = seated and bool(residual.mating_centers_ok)
    if seated:
        context.services["port_inserted"] = True
        context.services["align_insert_active"] = False
        _spawn_align_insert_debug(
            context,
            crystal=crystal,
            port=port,
            residual=residual,
            tip_measured_m=tip,
            tip_cmd_m=tip,
            tool_ori=tool_ori,
        )
        controller.add_hold_command(
            20,
            hold_gripper=True,
            label=f"{context.step.name}: inserted-hold",
        )
        _save_insert_cache(context, verified=True)
        print(
            f"[BT INSERT{_station_tag(context)}] DONE gap={gap:+.4f}m "
            f"(≤{touch:.4f}) frames={frames} "
            f"centers={residual.mating_centers_ok} "
            f"axis={residual.axis_ok} sides={residual.mating_sides_ok} "
            f"latch={residual.latch_z_ok} yz_align={yz_enabled}"
        )
        from ur5e_6x_cable_insertions.insert_diagnostics import finish_insert_diagnostics

        _refresh_insert_diag_geometry(context, label="seated")
        finish_insert_diagnostics(context, reason="seated")
        return

    # --- Align: match crystal mating YZ to port mating YZ (features only) ---
    # Temporarily disabled: TipOffset YZ is the align; insert is axis translate only.
    # Re-enable with ALIGN_INSERT_ENABLE_YZ = True (and uncomment below).
    # if yz_enabled and not residual.mating_centers_ok:
    #     # Full mating-center ΔYZ — do NOT use latch-polluted residual.pos_error.
    #     dyz = np.array(
    #         [0.0, float(p_mc[1] - c_mc[1]), float(p_mc[2] - c_mc[2])],
    #         dtype=np.float64,
    #     )
    #     mag = float(np.linalg.norm(dyz))
    #     max_yz = float(getattr(cfg, "ALIGN_YZ_MAX_M", 0.025))
    #     if mag > max_yz and mag > 1e-12:
    #         dyz = dyz * (max_yz / mag)
    #     tip_tgt = tip + dyz
    #     tool_tgt = tool_ori
    #     label = "align-yz"
    #     _spawn_align_insert_debug(
    #         context,
    #         crystal=crystal,
    #         port=port,
    #         residual=residual,
    #         tip_measured_m=tip,
    #         tip_cmd_m=tip_tgt,
    #         tool_ori=tool_tgt,
    #         insert_dir=dyz if mag > 1e-9 else None,
    #     )
    #     _queue_align_insert_cartesian(context, tip_tgt, tool_tgt, label=label)
    #     _record_insert_pose(
    #         context,
    #         tip=tip_tgt,
    #         tool_ori=tool_tgt,
    #         axis=None,
    #         kind="align",
    #         label=label,
    #     )
    #     if frames % log_every == 0:
    #         print(
    #             f"[BT INSERT{_station_tag(context)}] align-yz "
    #             f"centers={residual.mating_centers_ok} "
    #             f"gap={gap:+.4f}m dYZ={np.round(dyz, 4)} (|Δ|={mag:.4f}m)\n"
    #             f"  crystal_mating={np.round(c_mc, 4)} "
    #             f"port_mating={np.round(p_mc, 4)}\n"
    #             f"  tip_meas={np.round(tip, 4)} tip_cmd={np.round(tip_tgt, 4)}"
    #         )
    #     return

    # --- Translate: small step along crystal insertion axis; record tip poses ---
    axis = insert_direction_crystal_neg_x(crystal)
    step = float(cfg.INSERT_STEP_M)
    # Do not overshoot past the port mating plane along the axis.
    if gap < 0.0:
        step = min(step, abs(gap))
    if step <= 1e-9:
        # Gap already within touch but seated check failed (YZ mode); hold.
        context.services["port_inserted"] = True
        context.services["align_insert_active"] = False
        _save_insert_cache(context, verified=True)
        print(
            f"[BT INSERT{_station_tag(context)}] DONE (step=0) gap={gap:+.4f}m"
        )
        from ur5e_6x_cable_insertions.insert_diagnostics import finish_insert_diagnostics

        _refresh_insert_diag_geometry(context, label="seated-step0")
        finish_insert_diagnostics(context, reason="seated_step0")
        return
    tip_tgt = insert_target_tip(tip, axis, step)
    label = "translate-axis"
    _spawn_align_insert_debug(
        context,
        crystal=crystal,
        port=port,
        residual=residual,
        tip_measured_m=tip,
        tip_cmd_m=tip_tgt,
        tool_ori=tool_ori,
        insert_dir=axis,
    )
    _queue_align_insert_cartesian(context, tip_tgt, tool_ori, label=label)
    _record_insert_pose(
        context,
        tip=tip_tgt,
        tool_ori=tool_ori,
        axis=axis,
        kind="translate",
        label=label,
    )
    if frames % log_every == 0:
        print(
            f"[BT INSERT{_station_tag(context)}] translate "
            f"gap={gap:+.4f}m step={step:.4f}m "
            f"dir={np.round(axis, 3)} tip→{np.round(tip_tgt, 4)}\n"
            f"  crystal_mating={np.round(c_mc, 4)} "
            f"port_mating={np.round(p_mc, 4)}"
        )
    # Diag samples on IK arrival (tick_align_and_insert), not when queuing.
    context.services["_insert_diag_label"] = label
    context.services["_insert_diag_pending_extra"] = {
        "gap_m": round(float(gap), 5),
        "step_m": round(float(step), 5),
        "insert_dir": _safe_round_list(axis),
        "tip_cmd_m": _safe_round_list(tip_tgt),
    }
    _refresh_insert_diag_geometry(context, label=label)


def queue_align_and_insert(context) -> None:
    """After TipOffset: reach check → align YZ → translate along crystal axis."""

    # Refuse to start insert unless TipOffset was actually reached.
    if not check_at_port_offset(context):
        context.services["abort_simulation"] = True
        context.services["abort_reason"] = (
            "align_and_insert refused: not at TipOffset "
            "(maneuver did not reach port_offset_tip)"
        )
        context.services["align_insert_active"] = False
        context.services["maneuver_reached_offset"] = False
        raise RuntimeError(context.services["abort_reason"])

    # Maneuver delivered TipOffset — protect that cache from insert slips.
    context.services["maneuver_reached_offset"] = True
    context.services["align_insert_frames"] = 0
    context.services["port_inserted"] = False
    context.services["align_insert_active"] = True
    context.services["monitor_cable_hold"] = True
    context.services["insert_path_plan"] = []
    context.services["insert_cache_queue"] = None
    context.services["insert_reach_ok"] = False
    context.services["insert_reach_pending"] = False
    context.services["insert_reach_deltas_left"] = None
    # Insert: log every cable↔mesh contact (incl. Ethernet). Never abort on them.
    context.services["collision_abort_active"] = False
    monitor = context.services.get("collision_monitor")
    spec = context.services.get("station_spec")
    if monitor is not None:
        if hasattr(monitor, "clear_ignore_obstacles"):
            monitor.clear_ignore_obstacles()
        if hasattr(monitor, "set_skip_prefixes") and spec is not None:
            if bool(getattr(cfg, "INSERT_CONTACT_SKIP_ROBOT", True)):
                monitor.set_skip_prefixes((str(spec.robot_prim_path),))
            else:
                monitor.set_skip_prefixes(())
        if hasattr(monitor, "set_logging"):
            monitor.set_logging(bool(getattr(cfg, "INSERT_DIAG_CONTACT_LOG", True)))
        if hasattr(monitor, "clear_recent_contacts"):
            monitor.clear_recent_contacts()
        monitor.clear_hit_flag()

    from ur5e_6x_cable_insertions.insert_diagnostics import start_insert_diagnostics

    # Expose crystal paths for material binding snapshot.
    if spec is not None:
        context.services["path45"] = str(getattr(spec, "path45", "") or "")
        context.services["path39"] = str(getattr(spec, "path39", "") or "")
        context.services["robot_prim_path"] = str(
            getattr(spec, "robot_prim_path", "") or ""
        )
        context.services["cable_path"] = str(
            getattr(spec, "cable_root_path", "") or ""
        )
    start_insert_diagnostics(context)

    controller = context.services["motion_controller"]
    controller.clear_queue()

    print(
        f"[BT INSERT{_station_tag(context)}] TipOffset reach precondition — "
        "IK must place crystal mating on port mating before align→translate"
    )
    status = _ensure_insert_reachability(context)
    if status == "fail":
        return
    if status == "queued_reconfig":
        return
    _begin_align_translate_after_reach(context)


def advance_align_insert(context) -> None:
    """When the prior align/translate (or reach reconfig) finishes, queue next."""

    if not context.services.get("align_insert_active"):
        return
    if context.services.get("port_inserted"):
        return
    if context.services.get("abort_simulation"):
        return

    frames = int(context.services.get("align_insert_frames", 0)) + 1
    context.services["align_insert_frames"] = frames
    if frames >= int(cfg.ALIGN_INSERT_MAX_FRAMES):
        context.services["abort_simulation"] = True
        context.services["abort_reason"] = (
            f"align/translate timeout after {frames} frames"
        )
        context.services["align_insert_active"] = False
        _save_insert_cache(context, verified=False)
        print(
            f"[BT INSERT{_station_tag(context)}] FAIL timeout frames={frames}"
        )
        from ur5e_6x_cable_insertions.insert_diagnostics import finish_insert_diagnostics

        _refresh_insert_diag_geometry(context, label="timeout")
        finish_insert_diagnostics(context, reason="timeout")
        return

    controller = context.services["motion_controller"]
    if not controller.is_done():
        # Still tracking a translate/align segment — no per-timestep diag spam.
        return

    # Offset pan-reconfig finished — re-check seat IK, then start align→translate.
    if context.services.pop("insert_reach_pending", False):
        status = _ensure_insert_reachability(context)
        if status == "fail":
            return
        if status == "queued_reconfig":
            return
        _begin_align_translate_after_reach(context)
        return

    # Translate IK target reached: one insert-diag sample at this pose.
    from ur5e_6x_cable_insertions.insert_diagnostics import sample_insert_diagnostics

    arrived_label = str(
        context.services.get("_insert_diag_label") or "translate-arrived"
    )
    pending_extra = context.services.pop("_insert_diag_pending_extra", None)
    _refresh_insert_diag_geometry(context, label=arrived_label)
    sample_insert_diagnostics(
        context,
        label=arrived_label,
        extra=pending_extra if isinstance(pending_extra, dict) else None,
    )

    _queue_next_align_insert(context)


def while_align_and_insert(context) -> None:
    """Cable-hold monitor + align/translate planner each BT tick."""

    monitor_cable_hold(context)
    advance_align_insert(context)


def check_port_inserted(context) -> bool:
    """Postcondition: mating gap closed under feature alignment."""

    ok = bool(context.services.get("port_inserted"))
    if ok:
        print(
            f"[BT INSERT{_station_tag(context)}] validate port_inserted=True "
            f"frames={context.services.get('align_insert_frames', 0)}"
        )
    return ok


def check_at_port_offset(context) -> bool:
    """Postcondition: fingertip (and crystal mating) actually at TipOffset.

    BT ``success_conditions`` only stamp blackboard strings — they do not measure
    geometry. Without this validate, a collision abort that clears the queue
    looks like success and align→translate starts mid-path.
    """

    if context.services.get("abort_simulation"):
        print(
            f"[BT MANEUVER{_station_tag(context)}] at_port_offset FAIL "
            f"(abort: {context.services.get('abort_reason', '?')})"
        )
        return False

    tip_tgt = context.services.get("port_offset_tip")
    if tip_tgt is None:
        tip_tgt = context.services.get("maneuver_tip_offset")
    if tip_tgt is None:
        print(
            f"[BT MANEUVER{_station_tag(context)}] at_port_offset FAIL "
            "(no port_offset_tip)"
        )
        return False
    tip_tgt = np.asarray(tip_tgt, dtype=np.float64).reshape(3)

    tool_ori = context.services.get("port_approach_tool_orientation")
    if tool_ori is None:
        tool_ori = context.services.get("port_offset_yaw_tool_orientation")
    if tool_ori is None:
        tool_ori, _ = _grasp_tool_and_lula(context)
    tool_ori = _normalize_quat(tool_ori)

    controller = context.services["motion_controller"]
    tip_now = measured_fingertip_meters(controller, tool_ori)
    tip_err = float(np.linalg.norm(tip_now - tip_tgt))
    tip_tol = float(
        getattr(
            cfg,
            "MANEUVER_AT_OFFSET_TIP_TOL_M",
            getattr(cfg, "MANEUVER_OFFSET_POS_TOLERANCE_M", 0.04),
        )
    )

    crystal_err = None
    crystal_tol = float(getattr(cfg, "MANEUVER_AT_OFFSET_CRYSTAL_TOL_M", 0.05))
    desired_c = context.services.get("port_offset_crystal")
    if desired_c is not None:
        try:
            crystal = live_crystal_features(context)
            c_mc = np.asarray(crystal.mating_center, dtype=np.float64).reshape(3)
            desired_c = np.asarray(desired_c, dtype=np.float64).reshape(3)
            crystal_err = float(np.linalg.norm(c_mc - desired_c))
        except Exception as exc:
            print(
                f"[BT MANEUVER{_station_tag(context)}] at_port_offset "
                f"crystal check skipped: {exc}"
            )

    tip_ok = tip_err <= tip_tol
    crystal_ok = crystal_err is None or crystal_err <= crystal_tol
    ok = tip_ok and crystal_ok
    context.services["maneuver_reached_offset"] = bool(ok)
    if ok:
        print(
            f"[BT MANEUVER{_station_tag(context)}] at_port_offset OK "
            f"tip_err={tip_err:.4f}m (≤{tip_tol:.4f})"
            + (
                f" crystal_err={crystal_err:.4f}m (≤{crystal_tol:.4f})"
                if crystal_err is not None
                else ""
            )
            + f" tip={np.round(tip_now, 4)} tgt={np.round(tip_tgt, 4)}"
        )
    else:
        print(
            f"[BT MANEUVER{_station_tag(context)}] at_port_offset FAIL "
            f"tip_err={tip_err:.4f}m (≤{tip_tol:.4f}) "
            f"tip={np.round(tip_now, 4)} tgt={np.round(tip_tgt, 4)}"
            + (
                f" crystal_err={crystal_err:.4f}m (≤{crystal_tol:.4f})"
                if crystal_err is not None
                else ""
            )
            + " — not starting align→translate"
        )
    return ok


def _expected_fingertip_meters(context) -> np.ndarray:
    """FK fingertip in meters from Lula hand + tool mount offset."""

    from ur5e_6x_cable_insertions.runtime_support import stage_to_meters

    controller = context.services["motion_controller"]
    hand_stage, lula_quat = controller._current_hand_pose()
    mpu = float(getattr(controller, "_meters_per_unit", meters_per_unit(context.services["stage"])))
    hand_m = stage_to_meters(hand_stage, mpu)
    tool_quat = cfg.tool_orientation_from_lula(lula_quat)
    rot = cfg._quat_to_rot_matrix(tool_quat)
    return hand_m + rot @ np.array(
        [0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64
    )


def _aabb_separation_m(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> float:
    """Euclidean gap between two AABBs (0 if they overlap / touch)."""

    amin = np.asarray(a_min, dtype=np.float64).reshape(3)
    amax = np.asarray(a_max, dtype=np.float64).reshape(3)
    bmin = np.asarray(b_min, dtype=np.float64).reshape(3)
    bmax = np.asarray(b_max, dtype=np.float64).reshape(3)
    gap = np.maximum(0.0, np.maximum(amin - bmax, bmin - amax))
    return float(np.linalg.norm(gap))


def _finger_pad_prim_paths(stage, robot_prim_path: str) -> tuple[str | None, str | None]:
    """Best left/right fingertip (pad) prim paths under the robot."""

    left_cands: list[tuple[int, str]] = []
    right_cands: list[tuple[int, str]] = []
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return None, None
    tokens = tuple(str(t).lower() for t in cfg.FINGERTIP_NAME_TOKENS)
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        path = str(prim.GetPath())
        path_l = path.lower()
        if not any(tok in name or tok in path_l for tok in tokens):
            continue
        # Prefer *pad* meshes over outer finger links.
        score = 0
        if "pad" in name or "pad" in path_l:
            score += 2
        if "inner" in name or "inner" in path_l:
            score += 1
        if "left" in name or "left" in path_l:
            left_cands.append((score, path))
        elif "right" in name or "right" in path_l:
            right_cands.append((score, path))
    left = max(left_cands, key=lambda x: x[0])[1] if left_cands else None
    right = max(right_cands, key=lambda x: x[0])[1] if right_cands else None
    return left, right


def both_fingers_contact_neck(context) -> tuple[bool, dict]:
    """True when left AND right fingertip pads are within gap of the neck AABB."""

    info: dict = {
        "left_pad_path": None,
        "right_pad_path": None,
        "left_pad_gap_m": None,
        "right_pad_gap_m": None,
        "finger_contact_max_gap_m": float(
            getattr(cfg, "GRASP_FINGER_CONTACT_MAX_GAP_M", 0.004)
        ),
    }
    stage = context.services["stage"]
    spec = context.services.get("station_spec")
    robot_path = getattr(spec, "robot_prim_path", None) if spec is not None else None
    if not robot_path:
        robot = context.services.get("robot")
        if robot is not None:
            robot_path = getattr(robot, "prim_path", None) or getattr(
                robot, "_prim_path", None
            )
    if not robot_path:
        info["error"] = "robot prim path missing for finger contact"
        return False, info

    left_path, right_path = _finger_pad_prim_paths(stage, str(robot_path))
    info["left_pad_path"] = left_path
    info["right_pad_path"] = right_path
    if not left_path or not right_path:
        info["error"] = (
            f"need left+right pads under {robot_path}; "
            f"found left={left_path} right={right_path}"
        )
        return False, info

    neck_path = _grasp_part_path(context)
    try:
        nmin, nmax, _ = prim_bbox_meters(stage, neck_path)
        lmin, lmax, _ = prim_bbox_meters(stage, left_path)
        rmin, rmax, _ = prim_bbox_meters(stage, right_path)
    except Exception as exc:
        info["error"] = f"pad/neck bbox failed: {exc}"
        return False, info

    left_gap = _aabb_separation_m(lmin, lmax, nmin, nmax)
    right_gap = _aabb_separation_m(rmin, rmax, nmin, nmax)
    info["left_pad_gap_m"] = round(left_gap, 5)
    info["right_pad_gap_m"] = round(right_gap, 5)
    max_gap = float(info["finger_contact_max_gap_m"])
    ok = left_gap <= max_gap and right_gap <= max_gap

    # Straddle: neck center X must lie between the two pad centers.
    if bool(getattr(cfg, "GRASP_REQUIRE_NECK_BETWEEN_PADS", True)):
        l_c = 0.5 * (np.asarray(lmin, dtype=np.float64) + np.asarray(lmax, dtype=np.float64))
        r_c = 0.5 * (np.asarray(rmin, dtype=np.float64) + np.asarray(rmax, dtype=np.float64))
        n_c = 0.5 * (np.asarray(nmin, dtype=np.float64) + np.asarray(nmax, dtype=np.float64))
        lo_x = float(min(l_c[0], r_c[0]))
        hi_x = float(max(l_c[0], r_c[0]))
        # Small slack so near-equal pad X (failed open) does not false-fail.
        slack = max(0.001, 0.25 * (hi_x - lo_x))
        between = (lo_x - slack) <= float(n_c[0]) <= (hi_x + slack)
        info["neck_between_pads"] = bool(between)
        info["pad_x_span"] = [round(lo_x, 5), round(hi_x, 5)]
        info["neck_x"] = round(float(n_c[0]), 5)
        ok = bool(ok and between)
    return ok, info


def cable_still_in_gripper(context) -> tuple[bool, dict]:
    """True when the neck still tracks the fingertips and fingers stay closed."""

    info: dict = {}
    try:
        tip_expected = _expected_fingertip_meters(context)
        path = _grasp_part_path(context)
        part = prim_bbox_meters(context.services["stage"], path)[2]
        grasp_tip = part.copy()
        grasp_tip[0] += float(grasp_x_offset_m(context))
        if bool(cfg.GRASP_USE_CABLE_ROOT_X):
            cable_path = context.services.get("cable_root_path")
            if cable_path:
                grasp_tip[0] = float(
                    cable_world_translate_meters(context.services["stage"], cable_path)[0]
                ) + float(cfg.GRASP_X_OFFSET_M)
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
            # Require every reported finger DOF past contact (not just max).
            closed_enough = bool(
                np.all(np.abs(fingers) >= float(cfg.ROBOTIQ_CONTACT_RAD))
            )
        info["fingers"] = np.round(fingers, 3)
    except Exception:
        info["fingers"] = None

    info["closed_enough"] = bool(closed_enough)

    # Continuous hold monitor: tip tracking + closed fingers only.
    # Both-pad contact is enforced at grasp validate (check_cable_in_gripper).
    in_grip = bool(
        tip_err <= float(cfg.CABLE_IN_GRIPPER_MAX_ERR_M) and closed_enough
    )
    info["in_gripper"] = in_grip
    return in_grip, info


def check_cable_in_gripper(context) -> bool:
    """Postcondition after grasp close: both pads on neck, cable held."""

    held, info = cable_still_in_gripper(context)
    both_ok = True
    if held and bool(getattr(cfg, "GRASP_BOTH_FINGERS_CONTACT", True)):
        both_ok, contact_info = both_fingers_contact_neck(context)
        info.update(contact_info)
        info["both_fingers_contact"] = bool(both_ok)
        held = bool(held and both_ok)
        info["in_gripper"] = held
    elif bool(getattr(cfg, "GRASP_BOTH_FINGERS_CONTACT", True)):
        # Still report pad gaps when tip/closed already failed.
        _both_ok, contact_info = both_fingers_contact_neck(context)
        info.update(contact_info)
        info["both_fingers_contact"] = bool(_both_ok)
    else:
        info["both_fingers_contact"] = None

    print(
        f"[BT GRIP{_station_tag(context)}] grasp validate: {_format_cable_status(info)}"
    )
    if not held:
        reason = "after grasp close validate"
        if info.get("both_fingers_contact") is False:
            reason = (
                "after grasp close validate (both-fingers contact failed: "
                f"left_gap={info.get('left_pad_gap_m')} "
                f"right_gap={info.get('right_pad_gap_m')})"
            )
        _abort_cable_lost(context, info, where=reason)
    return held


def _format_cable_status(info: dict) -> str:
    both = info.get("both_fingers_contact")
    both_s = ""
    if both is not None:
        both_s = (
            f" both_pads={both}"
            f"(L={info.get('left_pad_gap_m')}, R={info.get('right_pad_gap_m')}, "
            f"max={info.get('finger_contact_max_gap_m')}"
        )
        if info.get("neck_between_pads") is not None:
            both_s += f", between={info.get('neck_between_pads')}"
        both_s += ")"
    return (
        f"in_gripper={info.get('in_gripper')} "
        f"tip_err={info.get('tip_err_m', float('nan')):.4f}m "
        f"(max={cfg.CABLE_IN_GRIPPER_MAX_ERR_M:.3f}) "
        f"closed={info.get('closed_enough')} fingers={info.get('fingers')} "
        f"part={info.get('part')} tip_expected={info.get('tip_expected')}"
        + both_s
        + (f" error={info.get('error')}" if info.get("error") else "")
    )


def monitor_cable_hold(context):
    """Per-frame hold check after grasp: abort on slip or DataHall collision."""

    from behaviour_tree_insertion.runtime import Status

    if context.services.get("abort_simulation"):
        return Status.FAILURE
    if not context.services.get("monitor_cable_hold"):
        return None

    controller = context.services["motion_controller"]
    if controller.is_done():
        return None
    try:
        cmd = controller._command_queue[controller._current_command_index]
    except Exception:
        return None
    if cmd.get("type") not in ("cartesian", "hold", "joint_waypoint") or not cmd.get(
        "hold_gripper"
    ):
        return None

    every = max(1, int(cfg.CABLE_HOLD_CHECK_EVERY_N_FRAMES))
    frame = int(context.services.get("_cable_hold_frame", 0)) + 1
    context.services["_cable_hold_frame"] = frame

    if bool(cfg.MANEUVER_ABORT_ON_DATAHALL_CONTACT) and context.services.get(
        "collision_abort_active", False
    ):
        monitor = context.services.get("collision_monitor")
        if monitor is not None:
            hit = monitor.consume_hit()
            if hit is not None:
                watched, obstacle = hit
                reason = (
                    f"DataHall/switch collision while [{cmd.get('label', '?')}]: "
                    f"{watched} ↔ {obstacle}"
                )
                print(f"[BT COLLISION{_station_tag(context)}] ABORT {reason}")
                context.services["abort_simulation"] = True
                context.services["abort_reason"] = reason
                context.services["monitor_cable_hold"] = False
                context.services["maneuver_reached_offset"] = False
                try:
                    controller.clear_queue()
                    # Empty queue alone looks like success to IsaacControllerPrimitive
                    # — mark failure like cable-lost does.
                    if hasattr(controller, "_segment_failed"):
                        controller._segment_failed = True
                    if hasattr(controller, "_six_arm_failure_reason"):
                        controller._six_arm_failure_reason = reason
                except Exception:
                    pass
                return Status.FAILURE

    if frame % every != 0:
        return None

    held, info = cable_still_in_gripper(context)
    context.services["_last_cable_hold_info"] = dict(info)
    label = cmd.get("label", "?")
    if frame % max(every * 10, 30) == 0:
        print(
            f"[BT GRIP{_station_tag(context)}] during [{label}]: "
            f"{_format_cable_status(info)}"
        )
    if held:
        return None
    _abort_cable_lost(context, info, where=f"during [{label}]")
    return Status.FAILURE


def _abort_cable_lost(context, info: dict, *, where: str) -> None:
    """Stop arm/gripper motion and log the failed postcondition; keep sim alive."""

    msg = (
        f"[BT CABLE LOST{_station_tag(context)}] Cable left the gripper {where}. "
        f"Failed postcondition: cable_between_fingers. {_format_cable_status(info)}. "
        f"Stopping arm+gripper motion; simulation stays active."
    )
    print(msg)
    context.services["abort_simulation"] = True
    context.services["abort_reason"] = msg
    context.services["monitor_cable_hold"] = False
    context.services["_last_cable_hold_info"] = dict(info)
    if context.services.get("align_insert_active") or context.services.get(
        "insert_diagnostics"
    ):
        from ur5e_6x_cable_insertions.insert_diagnostics import (
            finish_insert_diagnostics,
            sample_insert_diagnostics,
        )

        try:
            _refresh_insert_diag_geometry(context, label="cable_lost")
            sample_insert_diagnostics(
                context,
                label="cable_lost",
                extra={"cable_hold": dict(info), "where": where},
            )
            finish_insert_diagnostics(context, reason="cable_lost")
        except Exception as exc:
            print(f"[INSERT DIAG] cable_lost finalize failed: {exc}")
        context.services["align_insert_active"] = False
    try:
        controller = context.services["motion_controller"]
        controller.clear_queue()
        # So IsaacControllerPrimitive.tick returns FAILURE this frame (not SUCCESS
        # after an empty queue).
        if hasattr(controller, "_segment_failed"):
            controller._segment_failed = True
        if hasattr(controller, "_six_arm_failure_reason"):
            controller._six_arm_failure_reason = msg
    except Exception:
        pass
