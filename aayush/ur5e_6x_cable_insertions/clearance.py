"""USD AABB clearance + multi-via rediscovery for TipLift→TipOffset.

Obstacles include station switch/table solids **and** DataHall / switch meshes
that intersect the station workspace. Tip + hand proxies are tested. Discovery
prefers lateral-first paths (hold safe +X, move Y/Z, then approach) so the arm
does not lerp through the rack.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ur5e_6x_cable_insertions import config as cfg


def aabb_overlap(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
    *,
    margin: float = 0.0,
) -> bool:
    m = float(margin)
    return bool(
        float(a_min[0]) - m <= float(b_max[0])
        and float(a_max[0]) + m >= float(b_min[0])
        and float(a_min[1]) - m <= float(b_max[1])
        and float(a_max[1]) + m >= float(b_min[1])
        and float(a_min[2]) - m <= float(b_max[2])
        and float(a_max[2]) + m >= float(b_min[2])
    )


def _stage_mpu(stage) -> float:
    from pxr import UsdGeom

    try:
        mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        mpu = 0.01
    return mpu if mpu > 0.0 else 0.01


def prim_world_aabb_meters(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mpu = _stage_mpu(stage)
    minimum = np.array(box.GetMin(), dtype=np.float64) * mpu
    maximum = np.array(box.GetMax(), dtype=np.float64) * mpu
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        return None
    if float(np.min(maximum - minimum)) < 1e-6:
        return None
    return minimum, maximum


def _switch_chassis_path(port_pack_path: str) -> str | None:
    marker = "/AS4610_01"
    if marker not in port_pack_path:
        return None
    return port_pack_path.split(marker, 1)[0] + marker


def _workspace_aabb(
    tip_start: np.ndarray, tip_end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(tip_start, dtype=np.float64).reshape(3)
    b = np.asarray(tip_end, dtype=np.float64).reshape(3)
    pad = np.asarray(cfg.MANEUVER_WORKSPACE_PAD_M, dtype=np.float64).reshape(3)
    return np.minimum(a, b) - pad, np.maximum(a, b) + pad


def _collect_mesh_obstacles_under(
    stage,
    root_path: str,
    ws_min: np.ndarray,
    ws_max: np.ndarray,
    *,
    max_extent: float,
    max_count: int,
    label_prefix: str,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    mpu = _stage_mpu(stage)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    for prim in Usd.PrimRange(root):
        if len(out) >= max_count:
            break
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        except Exception:
            continue
        amin = np.array(box.GetMin(), dtype=np.float64) * mpu
        amax = np.array(box.GetMax(), dtype=np.float64) * mpu
        if not np.all(np.isfinite(amin)) or not np.all(np.isfinite(amax)):
            continue
        extent = float(np.max(amax - amin))
        if extent < 1e-4 or extent > max_extent:
            continue
        if not aabb_overlap(amin, amax, ws_min, ws_max, margin=0.0):
            continue
        path = str(prim.GetPath())
        out.append((amin, amax, f"{label_prefix}:{path}"))
    return out


def resolve_datahall_root(stage) -> str | None:
    """Return the first valid DataHall root on the stage."""

    for path in getattr(
        cfg,
        "DATAHALL_PRIM_PATH_FALLBACKS",
        (str(cfg.DATAHALL_PRIM_PATH),),
    ):
        prim = stage.GetPrimAtPath(str(path))
        if prim and prim.IsValid():
            return str(path)
    return None


def collect_maneuver_obstacle_aabbs(
    stage,
    spec=None,
    *,
    tip_start: np.ndarray | None = None,
    tip_end: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Switch/table solids + DataHall/switch meshes in the station workspace."""

    paths: list[str] = []
    datahall_root = resolve_datahall_root(stage)
    if datahall_root:
        paths.append(datahall_root)
    if spec is not None:
        pack = str(getattr(spec, "port_pack_path", "") or "")
        if pack:
            paths.append(pack)
            chassis = _switch_chassis_path(pack)
            if chassis and chassis not in paths:
                paths.append(chassis)
            if "/Switch/" in pack:
                switch = pack.split("/Switch/", 1)[0] + "/Switch"
                if switch not in paths:
                    paths.append(switch)
        height = getattr(spec, "height", None)
        if height is not None:
            paths.append(cfg.work_table_path_for(height))
        for attr in ("left_block_path", "right_block_path", "support_floor_path"):
            p = getattr(spec, attr, None)
            if p:
                paths.append(str(p))
    for root in getattr(cfg, "MANEUVER_EXTRA_OBSTACLE_PATHS", ()):
        paths.append(str(root))

    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    seen: set[str] = set()
    max_extent = float(cfg.MANEUVER_OBSTACLE_MAX_EXTENT_M)
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        aabb = prim_world_aabb_meters(stage, path)
        if aabb is None:
            continue
        amin, amax = aabb
        extent = float(np.max(amax - amin))
        # Always keep explicitly listed extras even if large (Rack_Core subtree).
        is_extra = path in getattr(cfg, "MANEUVER_EXTRA_OBSTACLE_PATHS", ())
        if extent > max_extent and not is_extra and path == datahall_root:
            print(
                f"[CLEARANCE] skip oversized obstacle {path} "
                f"extent={extent:.2f}m > {max_extent:.2f}m (mesh harvest still runs)"
            )
            continue
        if extent > max_extent and not is_extra:
            print(
                f"[CLEARANCE] skip oversized obstacle {path} "
                f"extent={extent:.2f}m > {max_extent:.2f}m"
            )
            continue
        out.append((amin, amax, path))

    if tip_start is not None and tip_end is not None:
        ws_min, ws_max = _workspace_aabb(tip_start, tip_end)
        mesh_max = float(cfg.MANEUVER_DATAHALL_MESH_MAX_EXTENT_M)
        mesh_cap = int(cfg.MANEUVER_DATAHALL_MESH_MAX_COUNT)
        mesh_roots = []
        if datahall_root:
            mesh_roots.append((datahall_root, "DataHall"))
        mesh_roots.append((str(cfg.NETWORK_SWITCHES_SCOPE), "SwitchMesh"))
        for root, prefix in mesh_roots:
            meshes = _collect_mesh_obstacles_under(
                stage,
                root,
                ws_min,
                ws_max,
                max_extent=mesh_max,
                max_count=mesh_cap,
                label_prefix=prefix,
            )
            for amin, amax, path in meshes:
                if path in seen:
                    continue
                seen.add(path)
                out.append((amin, amax, path))
        print(
            f"[CLEARANCE] workspace obstacles={len(out)} "
            f"datahall_root={datahall_root!r} "
            f"(pad={np.round(cfg.MANEUVER_WORKSPACE_PAD_M, 3)})"
        )
    return out


collect_obstacle_aabbs = collect_maneuver_obstacle_aabbs


def tip_proxy_aabb(
    tip: np.ndarray,
    half_extents: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    he = np.asarray(
        half_extents
        if half_extents is not None
        else cfg.MANEUVER_CRYSTAL_PROXY_HALF_EXTENTS_M,
        dtype=np.float64,
    ).reshape(3)
    return tip - he, tip + he


def hand_from_tip_tool(tip: np.ndarray, tool_quat_wxyz: np.ndarray) -> np.ndarray:
    """Hand/TCP so fingertips land on tip (same convention as primitives)."""

    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    rot = cfg._quat_to_rot_matrix(np.asarray(tool_quat_wxyz, dtype=np.float64))
    return tip - rot @ np.array([0.0, 0.0, float(cfg.TOOL_OFFSET_M)], dtype=np.float64)


def _near_goal_corridor(tip: np.ndarray, tip_end: np.ndarray | None) -> bool:
    if tip_end is None:
        return False
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    end = np.asarray(tip_end, dtype=np.float64).reshape(3)
    radius = float(cfg.MANEUVER_GOAL_CORRIDOR_RADIUS_M)
    if float(np.linalg.norm(tip - end)) <= radius:
        return True
    if float(tip[0]) + 1e-6 >= float(end[0]) - float(cfg.MANEUVER_GOAL_CORRIDOR_X_SLACK_M):
        if abs(float(tip[1]) - float(end[1])) <= radius and abs(
            float(tip[2]) - float(end[2])
        ) <= radius:
            return True
    return False


def _proxy_hits(
    center: np.ndarray,
    obstacles: Sequence[tuple[np.ndarray, np.ndarray, str]],
    half_extents: np.ndarray,
    margin: float,
) -> str | None:
    pmin, pmax = tip_proxy_aabb(center, half_extents)
    for omin, omax, path in obstacles:
        if aabb_overlap(pmin, pmax, omin, omax, margin=margin):
            return path
    return None


def tip_hits_obstacles(
    tip: np.ndarray,
    obstacles: Sequence[tuple[np.ndarray, np.ndarray, str]],
    *,
    half_extents: np.ndarray | None = None,
    margin: float | None = None,
    tip_end: np.ndarray | None = None,
    tool_ori: np.ndarray | None = None,
) -> str | None:
    """Crystal tip (+ optional hand) proxy vs obstacles."""

    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    if _near_goal_corridor(tip, tip_end):
        return None
    m = float(cfg.MANEUVER_CLEARANCE_MARGIN_M if margin is None else margin)
    tip_he = np.asarray(
        half_extents
        if half_extents is not None
        else cfg.MANEUVER_CRYSTAL_PROXY_HALF_EXTENTS_M,
        dtype=np.float64,
    ).reshape(3)
    hit = _proxy_hits(tip, obstacles, tip_he, m)
    if hit is not None:
        return hit
    if tool_ori is not None and bool(cfg.MANEUVER_CHECK_HAND_PROXY):
        hand = hand_from_tip_tool(tip, tool_ori)
        if not _near_goal_corridor(hand, tip_end):
            hand_he = np.asarray(
                cfg.MANEUVER_HAND_PROXY_HALF_EXTENTS_M, dtype=np.float64
            ).reshape(3)
            hit = _proxy_hits(hand, obstacles, hand_he, m)
            if hit is not None:
                return f"hand@{hit}"
    return None


def segment_hits_obstacles(
    p0: np.ndarray,
    p1: np.ndarray,
    obstacles: Sequence[tuple[np.ndarray, np.ndarray, str]],
    *,
    samples: int | None = None,
    half_extents: np.ndarray | None = None,
    margin: float | None = None,
    tip_end: np.ndarray | None = None,
    tool_ori0: np.ndarray | None = None,
    tool_ori1: np.ndarray | None = None,
) -> str | None:
    n = max(
        2, int(samples if samples is not None else cfg.MANEUVER_CLEARANCE_SEGMENT_SAMPLES)
    )
    a = np.asarray(p0, dtype=np.float64).reshape(3)
    b = np.asarray(p1, dtype=np.float64).reshape(3)
    for i in range(n + 1):
        t = float(i) / float(n)
        tool = None
        if tool_ori0 is not None and tool_ori1 is not None:
            # Orientation checked at endpoints only when lerping tips; mid uses ori0→ori1 slerp-ish blend via tip_hits without ori mid
            tool = tool_ori0 if t < 0.5 else tool_ori1
        hit = tip_hits_obstacles(
            (1.0 - t) * a + t * b,
            obstacles,
            half_extents=half_extents,
            margin=margin,
            tip_end=tip_end,
            tool_ori=tool,
        )
        if hit is not None:
            return hit
    return None


def polyline_hits_obstacles(
    points: Sequence[np.ndarray],
    obstacles: Sequence[tuple[np.ndarray, np.ndarray, str]],
    **kwargs,
) -> str | None:
    pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in points]
    if not pts:
        return None
    tool_oris = kwargs.pop("tool_oris", None)
    tip_end = kwargs.get("tip_end")
    hit = tip_hits_obstacles(
        pts[0],
        obstacles,
        tool_ori=None if not tool_oris else tool_oris[0],
        **kwargs,
    )
    if hit is not None:
        return hit
    for i in range(len(pts) - 1):
        o0 = tool_oris[i] if tool_oris is not None else None
        o1 = tool_oris[i + 1] if tool_oris is not None else None
        hit = segment_hits_obstacles(
            pts[i],
            pts[i + 1],
            obstacles,
            tip_end=tip_end,
            tool_ori0=o0,
            tool_ori1=o1,
            **{k: v for k, v in kwargs.items() if k != "tip_end"},
        )
        if hit is not None:
            return hit
    return None


def _sample_polyline(points: Sequence[np.ndarray], n_per_seg: int) -> list[np.ndarray]:
    pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in points]
    if len(pts) <= 1:
        return pts
    out: list[np.ndarray] = [pts[0].copy()]
    n = max(1, int(n_per_seg))
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        for k in range(1, n + 1):
            t = float(k) / float(n)
            out.append((1.0 - t) * a + t * b)
    return out


def dip_then_rise_to_offset(
    from_tip: np.ndarray,
    tip_end: np.ndarray,
) -> list[np.ndarray]:
    """Duck under rack lip (Mesh4679), slide −X, then rise to tip_offset.

    Returns waypoints **after** ``from_tip`` (does not include ``from_tip``).
    """

    start = np.asarray(from_tip, dtype=np.float64).reshape(3).copy()
    end = np.asarray(tip_end, dtype=np.float64).reshape(3).copy()
    dip = float(getattr(cfg, "MANEUVER_APPROACH_Z_DIP_M", 0.05))
    low_z = float(end[2]) - dip
    drop = start.copy()
    drop[2] = low_z
    under = end.copy()
    under[2] = low_z
    return [drop, under, end.copy()]


def _stretch_first_poly(
    start: np.ndarray,
    end: np.ndarray,
    *,
    z_clear: float,
    x_ret: float,
    y_blend: float,
    safe_x0: float,
) -> list[np.ndarray]:
    """TipLift → +X stretch → port-Y align → dip −X → rise to TipOffset."""

    lift = start.copy()
    lift[2] = max(float(start[2]), float(end[2])) + float(z_clear)
    safe_x = float(safe_x0) + float(x_ret)
    yb = float(np.clip(y_blend, 0.0, 1.0))
    stretch_y = (1.0 - yb) * float(start[1]) + yb * float(end[1])
    stretch = np.array([safe_x, stretch_y, float(lift[2])], dtype=np.float64)
    align = np.array([safe_x, float(end[1]), float(lift[2])], dtype=np.float64)
    ahead = float(getattr(cfg, "MANEUVER_DIP_START_X_AHEAD_M", 0.08))
    gate = end.copy()
    gate[0] = float(end[0]) + max(
        ahead, float(x_ret) * 0.5, float(cfg.MANEUVER_OFFSET_ABOVE_X_RETRACT_M)
    )
    gate[2] = float(lift[2])
    # Descend at the gate, then dip-slide to tip_offset.
    return (
        [start.copy(), lift, stretch, align, gate]
        + dip_then_rise_to_offset(gate, end)
    )


def discover_clear_tip_polyline(
    tip_start: np.ndarray,
    tip_end: np.ndarray,
    obstacles: Sequence[tuple[np.ndarray, np.ndarray, str]],
) -> tuple[list[np.ndarray], str]:
    """Search TipLift→TipOffset; prefer stretch-first (+X out, then approach)."""

    start = np.asarray(tip_start, dtype=np.float64).reshape(3).copy()
    end = np.asarray(tip_end, dtype=np.float64).reshape(3).copy()
    z_clears = [float(z) for z in cfg.MANEUVER_SEARCH_Z_CLEAR_M]
    x_retracts = [float(x) for x in cfg.MANEUVER_SEARCH_X_RETRACT_M]
    stretch_min = float(getattr(cfg, "MANEUVER_STRETCH_X_MIN_M", 0.25))
    stretch_x = [x for x in x_retracts if x + 1e-9 >= stretch_min]
    if not stretch_x:
        stretch_x = [max(stretch_min, x_retracts[-1] if x_retracts else stretch_min)]
    y_blends = [float(y) for y in getattr(cfg, "MANEUVER_STRETCH_Y_BLEND", (0.35,))]
    check_kw = {"tip_end": end}

    def _ok(poly: list[np.ndarray]) -> bool:
        return polyline_hits_obstacles(poly, obstacles, **check_kw) is None

    safe_x0 = max(float(start[0]), float(end[0]))

    # 1) Stretch-first (primary): lift → +X stretch (partial Y) → align port Y → above → end.
    for z_clear in z_clears:
        for x_ret in stretch_x:
            for yb in y_blends:
                poly = _stretch_first_poly(
                    start,
                    end,
                    z_clear=z_clear,
                    x_ret=x_ret,
                    y_blend=yb,
                    safe_x0=safe_x0,
                )
                if _ok(poly):
                    return (
                        poly,
                        f"stretch-first z={z_clear:.3f} x+={x_ret:.3f} yb={yb:.2f}",
                    )

    # 2) Lateral-first fallback: lift → hold safe X at port Y → above → end.
    for z_clear in z_clears:
        for x_ret in stretch_x:
            lift = start.copy()
            lift[2] = max(float(start[2]), float(end[2])) + z_clear
            safe_x = safe_x0 + x_ret
            lateral = np.array(
                [safe_x, float(end[1]), float(lift[2])], dtype=np.float64
            )
            above = end.copy()
            above[0] = float(end[0]) + max(
                x_ret * 0.5, float(cfg.MANEUVER_OFFSET_ABOVE_X_RETRACT_M)
            )
            above[2] = float(lift[2])
            poly = [start, lift, lateral, above, end]
            if _ok(poly):
                return poly, f"lateral-first z={z_clear:.3f} x+={x_ret:.3f}"

    # 3) Two-via: safe-X mid then offset-above.
    for z_clear in z_clears:
        for x_ret in stretch_x:
            for yb in cfg.MANEUVER_SEARCH_Y_BLEND:
                via1 = start.copy()
                via1[1] = float((1.0 - float(yb)) * start[1] + float(yb) * end[1])
                via1[0] = safe_x0 + x_ret
                via1[2] = max(float(start[2]), float(end[2])) + z_clear
                via2 = end.copy()
                via2[0] = float(end[0]) + max(
                    x_ret, float(cfg.MANEUVER_OFFSET_ABOVE_X_RETRACT_M)
                )
                via2[2] = max(float(start[2]), float(end[2])) + z_clear
                poly = [start, via1, via2, end]
                if _ok(poly):
                    return (
                        poly,
                        f"two-via z={z_clear:.3f} x+={x_ret:.3f} yb={float(yb):.2f}",
                    )

    # 4) Single via at safe X.
    for z_clear in z_clears:
        for x_ret in stretch_x:
            for yb in cfg.MANEUVER_SEARCH_Y_BLEND:
                via = (1.0 - float(yb)) * start + float(yb) * end
                via = via.copy()
                via[0] = safe_x0 + x_ret
                via[2] = max(float(start[2]), float(end[2])) + z_clear
                poly = [start, via, end]
                if _ok(poly):
                    return (
                        poly,
                        f"single-via z={z_clear:.3f} x+={x_ret:.3f} yb={float(yb):.2f}",
                    )

    # 5) Fallback: strongest stretch-first (may still hit — caller logs).
    z_clear = z_clears[-1] if z_clears else float(cfg.PORT_APPROACH_VIA_Z_CLEARANCE_M)
    x_ret = stretch_x[-1]
    yb = y_blends[len(y_blends) // 2]
    poly = _stretch_first_poly(
        start, end, z_clear=z_clear, x_ret=x_ret, y_blend=yb, safe_x0=safe_x0
    )
    hit = polyline_hits_obstacles(poly, obstacles, **check_kw)
    label = f"fallback-stretch z={z_clear:.3f} x+={x_ret:.3f} yb={yb:.2f}"
    if hit is not None:
        label += f" STILL_HITS={hit}"
    return poly, label


def densify_polyline_with_t(
    points: Sequence[np.ndarray],
    *,
    samples_per_seg: int | None = None,
) -> list[tuple[float, np.ndarray]]:
    pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in points]
    if not pts:
        return []
    if len(pts) == 1:
        return [(0.0, pts[0].copy())]

    n = max(
        1,
        int(
            samples_per_seg
            if samples_per_seg is not None
            else cfg.MANEUVER_POLYLINE_SAMPLES_PER_SEG
        ),
    )
    dense = _sample_polyline(pts, n)
    cleaned: list[np.ndarray] = [dense[0]]
    for p in dense[1:]:
        if float(np.linalg.norm(p - cleaned[-1])) > 1e-5:
            cleaned.append(p)
    if float(np.linalg.norm(cleaned[-1] - pts[-1])) > 1e-6:
        cleaned.append(pts[-1].copy())
    else:
        cleaned[-1] = pts[-1].copy()

    lengths = [0.0]
    for i in range(1, len(cleaned)):
        lengths.append(lengths[-1] + float(np.linalg.norm(cleaned[i] - cleaned[i - 1])))
    total = lengths[-1]
    if total < 1e-9:
        return [(0.0, cleaned[0].copy()), (1.0, cleaned[-1].copy())]
    return [(float(L / total), p.copy()) for L, p in zip(lengths, cleaned)]
