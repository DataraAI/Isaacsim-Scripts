"""Feature alignment residuals and insert micro-step helpers (Isaac-free)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from insertion_features.geometry import ConnectorFeatures


@dataclass(frozen=True)
class AlignmentResidual:
    latch_z_ok: bool
    mating_sides_ok: bool
    axis_ok: bool
    mating_centers_ok: bool
    mating_gap_m: float
    pos_error_m: np.ndarray  # tip translation suggestion in world metres
    rot_error_rad: np.ndarray  # small-angle axis-angle suggestion
    passed: bool


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def assert_unit_linear_scale(
    transform: np.ndarray, *, label: str, atol: float = 1.0e-3
) -> None:
    """Reject feature transforms whose linear columns contain authored scale."""

    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    norms = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.all(np.isfinite(norms)) or not np.allclose(
        norms, np.ones(3), atol=float(atol), rtol=0.0
    ):
        raise ValueError(
            f"{label} feature transform must have unit scale; column norms={norms}"
        )


def port_standoff_target(port: ConnectorFeatures, standoff_m: float) -> np.ndarray:
    """Tip target just before −X insertion: higher world-X than the Ethernet face.

    DataHall RJ45 openings face the cables (+X). The cable approaches from +X and
    only then moves −X into the jack. Do **not** use ``mating − insertion_axis``
    here: with axis ≈ +X that lands *inside* the port (−X of the face).
    """

    mating = np.asarray(port.mating_center, dtype=np.float64).reshape(3).copy()
    mating[0] = float(mating[0]) + abs(float(standoff_m))
    return mating


def insert_direction_crystal_neg_x(crystal: ConnectorFeatures) -> np.ndarray:
    """Unit insert step direction: crystal axis with negative world-X sense."""

    axis = _unit(crystal.insertion_axis)
    if float(np.dot(axis, np.array([-1.0, 0.0, 0.0], dtype=np.float64))) < 0.0:
        axis = -axis
    return axis


def mating_gap_along_axis(crystal: ConnectorFeatures, port: ConnectorFeatures) -> float:
    """Signed mating-center separation along the crystal −X insert direction."""

    axis = insert_direction_crystal_neg_x(crystal)
    return float(np.dot(crystal.mating_center - port.mating_center, axis))


def evaluate_alignment(
    crystal: ConnectorFeatures,
    port: ConnectorFeatures,
    *,
    latch_z_margin_m: float,
    mating_side_margin_m: float,
    axis_dot_min: float,
    mating_center_yz_tol_m: float = 0.0015,
    latch_y_margin_m: float | None = None,
) -> AlignmentResidual:
    """World-YZ alignment gates + tip/ori nudge residuals."""

    y_margin = (
        float(mating_side_margin_m)
        if latch_y_margin_m is None
        else float(latch_y_margin_m)
    )
    port_axis = _unit(port.insertion_axis)
    crystal_axis = _unit(crystal.insertion_axis)
    # Either sense is fine for collinearity; flip for rotation residual only.
    if float(np.dot(crystal_axis, port_axis)) < 0.0:
        crystal_axis_for_rot = -crystal_axis
    else:
        crystal_axis_for_rot = crystal_axis

    c_mc = np.asarray(crystal.mating_center, dtype=np.float64).reshape(3)
    p_mc = np.asarray(port.mating_center, dtype=np.float64).reshape(3)
    tol = float(mating_center_yz_tol_m)
    mating_centers_ok = bool(
        abs(float(c_mc[1] - p_mc[1])) <= tol and abs(float(c_mc[2] - p_mc[2])) <= tol
    )

    # Mating-corner rectangles in world YZ: crystal ⊆ port (with margin).
    p_corners = np.asarray(port.mating_corners, dtype=np.float64).reshape(-1, 3)
    c_corners = np.asarray(crystal.mating_corners, dtype=np.float64).reshape(-1, 3)
    side_m = float(mating_side_margin_m)
    py0 = float(np.min(p_corners[:, 1])) + side_m
    py1 = float(np.max(p_corners[:, 1])) - side_m
    pz0 = float(np.min(p_corners[:, 2])) + side_m
    pz1 = float(np.max(p_corners[:, 2])) - side_m
    if py1 < py0 or pz1 < pz0:
        mating_sides_ok = False
    else:
        inside = (
            (c_corners[:, 1] >= py0)
            & (c_corners[:, 1] <= py1)
            & (c_corners[:, 2] >= pz0)
            & (c_corners[:, 2] <= pz1)
        )
        mating_sides_ok = bool(np.all(inside))

    axis_dot = abs(float(np.dot(_unit(crystal.insertion_axis), port_axis)))
    axis_ok = axis_dot >= float(axis_dot_min)

    c_latch = np.asarray(crystal.latch_keypoints, dtype=np.float64).reshape(-1, 3)
    p_latch = np.asarray(port.latch_keypoints, dtype=np.float64).reshape(-1, 3)
    crystal_latch_z = float(np.max(c_latch[:, 2]))
    port_latch_z = float(np.min(p_latch[:, 2]))
    latch_z_ok = crystal_latch_z < port_latch_z - float(latch_z_margin_m)
    port_latch_y0 = float(np.min(p_latch[:, 1])) + y_margin
    port_latch_y1 = float(np.max(p_latch[:, 1])) - y_margin
    if port_latch_y1 < port_latch_y0:
        latch_y_ok = False
    else:
        latch_y_ok = bool(
            np.all(
                (c_latch[:, 1] >= port_latch_y0) & (c_latch[:, 1] <= port_latch_y1)
            )
        )
    latch_ok = bool(latch_z_ok and latch_y_ok)

    # Tip nudge: cancel mating-center world YZ; deepen Z if latch still high.
    pos_error = np.array(
        [0.0, float(p_mc[1] - c_mc[1]), float(p_mc[2] - c_mc[2])],
        dtype=np.float64,
    )
    if not latch_z_ok:
        latch_dz = (port_latch_z - float(latch_z_margin_m)) - crystal_latch_z
        if latch_dz < pos_error[2]:
            pos_error[2] = latch_dz
    if not latch_y_ok:
        c_latch_y = float(np.mean(c_latch[:, 1]))
        if c_latch_y < port_latch_y0:
            pos_error[1] += port_latch_y0 - c_latch_y
        elif c_latch_y > port_latch_y1:
            pos_error[1] += port_latch_y1 - c_latch_y

    # Rotation: align crystal_axis → port_axis (small-angle approx).
    rot_error = np.cross(crystal_axis_for_rot, port_axis)

    gap = mating_gap_along_axis(crystal, port)
    passed = bool(
        latch_ok and mating_sides_ok and axis_ok and mating_centers_ok
    )
    return AlignmentResidual(
        latch_z_ok=latch_ok,
        mating_sides_ok=mating_sides_ok,
        axis_ok=axis_ok,
        mating_centers_ok=mating_centers_ok,
        mating_gap_m=gap,
        pos_error_m=pos_error,
        rot_error_rad=rot_error,
        passed=passed,
    )


def clamp_nudge(
    pos_error_m: np.ndarray,
    rot_error_rad: np.ndarray,
    *,
    max_pos_m: float,
    max_rot_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos_error_m, dtype=np.float64).reshape(3)
    rot = np.asarray(rot_error_rad, dtype=np.float64).reshape(3)
    pn = float(np.linalg.norm(pos))
    rn = float(np.linalg.norm(rot))
    if pn > float(max_pos_m) and pn > 1e-12:
        pos = pos * (float(max_pos_m) / pn)
    if rn > float(max_rot_rad) and rn > 1e-12:
        rot = rot * (float(max_rot_rad) / rn)
    return pos, rot


def insert_target_tip(
    current_tip_m: np.ndarray,
    port_axis_unit: np.ndarray,
    step_m: float,
) -> np.ndarray:
    tip = np.asarray(current_tip_m, dtype=np.float64).reshape(3)
    axis = _unit(port_axis_unit)
    return tip + float(step_m) * axis
