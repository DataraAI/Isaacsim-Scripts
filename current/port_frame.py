"""Port frame utilities for QSFP / connector insertion from USD prim poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import carb
import numpy as np
from isaacsim.core.utils.xforms import get_world_pose


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("Cannot normalize zero-length vector")
    return v / n


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


@dataclass
class PortFrame:
    """World-frame insert geometry derived from a connector USD prim."""

    prim_path: str
    insert_origin: np.ndarray
    insert_axis: np.ndarray
    insert_rot: np.ndarray
    pick_down_rot: np.ndarray
    lateral_offset: np.ndarray

    @classmethod
    def from_prim_path(
        cls,
        prim_path: str,
        local_insert_axis: np.ndarray | None = None,
        lateral_offset: np.ndarray | None = None,
        robot_position: np.ndarray | None = None,
    ) -> Optional["PortFrame"]:
        position, orientation = get_world_pose(prim_path)
        if position is None:
            return None

        if lateral_offset is None:
            lateral_offset = np.zeros(3, dtype=np.float64)

        rot = _quat_wxyz_to_rot(np.asarray(orientation, dtype=np.float64))
        if local_insert_axis is None:
            local_insert_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        insert_axis = _normalize(rot @ _normalize(local_insert_axis))
        lat = np.asarray(lateral_offset, dtype=np.float64)
        origin = np.asarray(position, dtype=np.float64) + lat

        if robot_position is not None:
            port_to_robot = np.asarray(robot_position, dtype=np.float64) - origin
            if float(np.dot(insert_axis, port_to_robot)) > 0.0:
                insert_axis = -insert_axis

        z_axis = insert_axis
        # Keep the same wrist roll as the known-good -X insertion pose when
        # possible: module local +Y stays near world +Y while local +Z points
        # along the insert axis.
        roll_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_axis = roll_hint - z_axis * np.dot(roll_hint, z_axis)
        if np.linalg.norm(y_axis) < 1e-9:
            y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            y_axis = y_axis - z_axis * np.dot(y_axis, z_axis)
        y_axis = _normalize(y_axis)
        x_axis = _normalize(np.cross(y_axis, z_axis))
        insert_rot = _rot_to_quat_wxyz(np.column_stack([x_axis, y_axis, z_axis]))
        pick_down_rot = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

        return cls(
            prim_path=prim_path,
            insert_origin=origin,
            insert_axis=insert_axis,
            insert_rot=insert_rot,
            pick_down_rot=pick_down_rot,
            lateral_offset=lat,
        )

    def point_along_axis(self, axial_distance: float) -> np.ndarray:
        return self.insert_origin + self.insert_axis * axial_distance

    def center_goal_for_tip_depth(
        self,
        tip_axial_m: float,
        module_half_length: float,
        module_orientation_wxyz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return module-center goal so the leading tip sits tip_axial_m past the port origin."""
        ori = (
            self.insert_rot
            if module_orientation_wxyz is None
            else np.asarray(module_orientation_wxyz, dtype=np.float64)
        )
        tip_goal = self.point_along_axis(tip_axial_m)
        half = float(module_half_length)
        for sign in (1.0, -1.0):
            center = tip_goal - sign * self.insert_axis * half
            lead = self._leading_tip_position(center, ori, half)
            if abs(self.axial_coordinate(lead) - tip_axial_m) < 1e-5:
                return center
        return self.point_along_axis(tip_axial_m - half)

    def insert_frame_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return orthonormal port-frame axes; local +Z is the insert axis."""
        rot = _quat_wxyz_to_rot(np.asarray(self.insert_rot, dtype=np.float64))
        x_axis = _normalize(rot[:, 0])
        y_axis = _normalize(rot[:, 1])
        z_axis = _normalize(self.insert_axis)
        return x_axis, y_axis, z_axis

    def approach_position(self, standoff: float) -> np.ndarray:
        return self.point_along_axis(-abs(standoff))

    def seat_goal(self, seat_depth: float) -> np.ndarray:
        return self.point_along_axis(seat_depth)

    def hover_position(self, height: float) -> np.ndarray:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return self.insert_origin + world_up * height

    def axial_coordinate(self, position: np.ndarray) -> float:
        return float(np.dot(position - self.insert_origin, self.insert_axis))

    def lateral_error(self, position: np.ndarray) -> float:
        delta = np.asarray(position, dtype=np.float64) - self.insert_origin
        axial = self.axial_coordinate(position)
        perp = delta - self.insert_axis * axial
        return float(np.linalg.norm(perp))

    def _module_tip_positions(
        self,
        center: np.ndarray,
        orientation_wxyz: np.ndarray,
        half_length: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rot = _quat_wxyz_to_rot(np.asarray(orientation_wxyz, dtype=np.float64))
        length_axis = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        return center + length_axis * half_length, center - length_axis * half_length

    def _leading_tip_position(
        self,
        center: np.ndarray,
        orientation_wxyz: np.ndarray,
        half_length: float,
    ) -> np.ndarray:
        tip_a, tip_b = self._module_tip_positions(center, orientation_wxyz, half_length)
        if self.axial_coordinate(tip_a) >= self.axial_coordinate(tip_b):
            return tip_a
        return tip_b

    def evaluate_seat(
        self,
        module_position: np.ndarray,
        seat_depth: float,
        module_orientation: np.ndarray | None = None,
        module_half_length: float = 0.0,
        lateral_tol: float = 0.005,
        depth_fraction: float = 0.9,
    ) -> Tuple[bool, dict]:
        center = np.asarray(module_position, dtype=np.float64)
        center_axial = self.axial_coordinate(center)

        if module_orientation is not None and module_half_length > 0.0:
            tip_pos = self._leading_tip_position(
                center, module_orientation, module_half_length
            )
            tip_axial = self.axial_coordinate(tip_pos)
            lat_err = self.lateral_error(tip_pos)
        else:
            tip_pos = center
            tip_axial = center_axial + module_half_length
            lat_err = self.lateral_error(center)

        min_tip_axial = seat_depth * depth_fraction
        depth_ok = tip_axial >= min_tip_axial
        lat_ok = lat_err <= lateral_tol
        passed = depth_ok and lat_ok
        return passed, {
            "lateral_error_m": lat_err,
            "axial_depth_m": center_axial,
            "tip_axial_m": tip_axial,
            "min_tip_axial_m": min_tip_axial,
            "target_depth_m": seat_depth,
            "lateral_ok": lat_ok,
            "depth_ok": depth_ok,
            "passed": passed,
        }


def list_qsfp_port_paths(stage, switches_root: str = "/World/DataHall/Network_Switches") -> list[str]:
    from pxr import Usd

    root = stage.GetPrimAtPath(switches_root)
    if not root.IsValid():
        return []
    paths = []
    # Switches are USD instances; connector prims live under instance proxies.
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not path.startswith(switches_root):
            continue
        name = prim.GetName()
        if "QSFP_DD_Connector" in name and name.endswith("_A_01"):
            paths.append(path)
    return sorted(set(paths))


def select_spaced_port_paths(
    stage,
    count: int,
    min_separation_m: float,
    switches_root: str = "/World/DataHall/Network_Switches",
) -> list[str]:
    """Pick ports whose insert origins are at least min_separation_m apart."""
    all_paths = list_qsfp_port_paths(stage, switches_root=switches_root)
    frames: list[PortFrame] = []
    valid_paths: list[str] = []
    for path in all_paths:
        frame = PortFrame.from_prim_path(path)
        if frame is None:
            continue
        frames.append(frame)
        valid_paths.append(path)
    if not frames or count <= 0:
        return []

    selected = [0]
    while len(selected) < min(count, len(frames)):
        best_idx = None
        best_min_dist = -1.0
        for i, frame in enumerate(frames):
            if i in selected:
                continue
            origin = frame.insert_origin
            min_dist = min(
                float(np.linalg.norm(origin - frames[j].insert_origin))
                for j in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx is None:
            break
        if best_min_dist < min_separation_m:
            carb.log_warn(
                f"Port spacing fallback: best remaining separation "
                f"{best_min_dist:.4f} m < requested {min_separation_m:.4f} m"
            )
        selected.append(best_idx)

    return [valid_paths[i] for i in selected]