"""Progressive angled-approach pose cache (tip meters + tool wxyz).

When Lula IK fails mid-path, the last successful angled ``t`` is saved. The next
run replays cached poses and only advances a small step toward TipOffset so the
demo can creep forward run-to-run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent / "maneuver_cache.json"


def _as_vec3(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3].copy()


def _as_quat(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
        return None
    q = arr[:4].copy()
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return None
    return q / n


def make_pose(position, orientation_wxyz) -> dict[str, np.ndarray]:
    tip = _as_vec3(position)
    ori = _as_quat(orientation_wxyz)
    if tip is None or ori is None:
        raise ValueError("make_pose requires finite position[3] and orientation_wxyz[4]")
    return {"position": tip, "orientation_wxyz": ori}


def _pose_to_json(pose: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        "position": [float(x) for x in np.asarray(pose["position"]).reshape(3)],
        "orientation_wxyz": [
            float(x) for x in np.asarray(pose["orientation_wxyz"]).reshape(4)
        ],
    }


def _waypoint_to_json(wp: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "t": float(wp["t"]),
        "label": str(wp.get("label", "")),
        "tip": [float(x) for x in np.asarray(wp["tip"]).reshape(3)],
        "orientation_wxyz": [
            float(x) for x in np.asarray(wp["orientation_wxyz"]).reshape(4)
        ],
    }
    axis = _as_vec3(wp.get("insertion_axis"))
    if axis is not None:
        out["insertion_axis"] = [float(x) for x in axis]
    return out


def _load_root(path: Path) -> dict:
    if not path.is_file():
        return {"stations": {}}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {"stations": {}}
    if not isinstance(raw, dict):
        return {"stations": {}}
    stations = raw.get("stations")
    if not isinstance(stations, dict):
        stations = {}
    return {"stations": stations}


def load_station(
    station_id: str, *, path: Path | None = None
) -> dict[str, Any] | None:
    root = _load_root(path or CACHE_PATH)
    raw = root["stations"].get(str(station_id))
    if not isinstance(raw, dict):
        return None
    waypoints = []
    for item in raw.get("waypoints") or []:
        if not isinstance(item, dict):
            continue
        tip = _as_vec3(item.get("tip") or item.get("position"))
        ori = _as_quat(item.get("orientation_wxyz"))
        t = float(item.get("t", -1.0))
        if tip is None or ori is None or not (0.0 < t <= 1.0 + 1e-9):
            continue
        waypoints.append(
            {
                "t": t,
                "label": str(item.get("label", "")),
                "tip": tip,
                "orientation_wxyz": ori,
            }
        )
    waypoints.sort(key=lambda w: float(w["t"]))
    tip_offset = _as_vec3(raw.get("tip_offset"))
    return {
        "station_id": str(station_id),
        "angled_progress": float(raw.get("angled_progress", 0.0)),
        "verified": bool(raw.get("verified", False)),
        "waypoints": waypoints,
        "tip_offset": tip_offset,
    }


def save_station(
    station_id: str,
    *,
    angled_progress: float,
    waypoints: list[dict[str, Any]],
    tip_offset,
    verified: bool = False,
    path: Path | None = None,
) -> Path:
    out = path or CACHE_PATH
    root = _load_root(out)
    tip_off = _as_vec3(tip_offset)
    payload = {
        "angled_progress": float(np.clip(angled_progress, 0.0, 1.0)),
        "verified": bool(verified),
        "tip_offset": (
            [float(x) for x in tip_off.reshape(3)] if tip_off is not None else None
        ),
        "waypoints": [_waypoint_to_json(w) for w in waypoints],
    }
    root["stations"][str(station_id)] = payload
    out.write_text(json.dumps(root, indent=2) + "\n")
    print(
        f"[MANEUVER CACHE] wrote {out.name} station={station_id} "
        f"progress={payload['angled_progress']:.2f} "
        f"waypoints={len(payload['waypoints'])} verified={verified}"
    )
    return out


def parse_angled_t_from_label(label: str) -> float | None:
    """Extract ``t`` from maneuver labels (``port-maneuver-0.80``, ``port-offset``, …)."""

    text = str(label or "")
    if "port-offset" in text:
        return 1.0
    for key in (
        "port-maneuver-",
        "port-translate-",
        "port-angled-",
        "port-via-",
    ):
        if key not in text:
            continue
        tail = text.split(key, 1)[1]
        token = tail.split()[0] if tail else ""
        token = token.strip(":")
        try:
            return float(token)
        except ValueError:
            return None
    return None


def progress_before_failure(failed_t: float, planned_ts: list[float]) -> float:
    """Largest planned t strictly less than the failed sample."""

    prior = [float(t) for t in planned_ts if float(t) < float(failed_t) - 1e-9]
    if not prior:
        return 0.0
    return float(max(prior))


def last_safe_progress_before_rack(
    waypoints: list[dict[str, Any]],
    *,
    min_tip_x: float,
) -> float:
    """Largest ``t`` whose tip X stays on the safe (+X) side of the rack face."""

    safe_ts = [
        float(w["t"])
        for w in waypoints
        if float(np.asarray(w["tip"], dtype=np.float64).reshape(3)[0]) >= float(min_tip_x)
    ]
    if not safe_ts:
        return 0.0
    return float(max(safe_ts))


def retract_tip_offset_x(tip_offset, retract_m: float) -> np.ndarray | None:
    """Push tip_offset further in +X (away from DataHall rack)."""

    tip = _as_vec3(tip_offset)
    if tip is None:
        return None
    out = tip.copy()
    out[0] = float(out[0]) + float(retract_m)
    return out
