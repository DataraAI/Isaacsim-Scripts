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
    """Return a point outside the port, opposite its insertion direction."""

    return np.asarray(port.mating_center, dtype=np.float64) - float(
        standoff_m
    ) * _unit(port.insertion_axis)


def mating_gap_along_axis(crystal: ConnectorFeatures, port: ConnectorFeatures) -> float:
    """Signed distance from port mating plane to crystal mating center along port axis."""

    axis = _unit(port.insertion_axis)
    return float(np.dot(crystal.mating_center - port.mating_center, axis))


def evaluate_alignment(
    crystal: ConnectorFeatures,
    port: ConnectorFeatures,
    *,
    latch_z_margin_m: float,
    mating_side_margin_m: float,
    axis_dot_min: float,
) -> AlignmentResidual:
    port_axis = _unit(port.insertion_axis)
    crystal_axis = _unit(crystal.insertion_axis)
    # Prefer same sense as port (flip crystal if anti-parallel).
    if float(np.dot(crystal_axis, port_axis)) < 0.0:
        crystal_axis = -crystal_axis

    crystal_latch_z = float(np.max(crystal.latch_keypoints[:, 2]))
    port_latch_z = float(np.min(port.latch_keypoints[:, 2]))
    latch_z_ok = crystal_latch_z < port_latch_z - float(latch_z_margin_m)

    width = _unit(port.width_axis)
    up = _unit(port.up_axis)
    # Port mating rectangle half-extents from corners.
    rel = port.mating_corners - port.mating_center
    half_w = float(np.max(np.abs(rel @ width)))
    half_u = float(np.max(np.abs(rel @ up)))
    c_rel = crystal.mating_corners - port.mating_center
    inside = (
        (np.abs(c_rel @ width) <= half_w - float(mating_side_margin_m))
        & (np.abs(c_rel @ up) <= half_u - float(mating_side_margin_m))
    )
    mating_sides_ok = bool(np.all(inside))

    axis_dot = abs(float(np.dot(crystal_axis, port_axis)))
    axis_ok = axis_dot >= float(axis_dot_min)

    # Lateral error: crystal mating center projected into port width/up plane.
    delta = crystal.mating_center - port.mating_center
    lateral = (float(np.dot(delta, width)) * width) + (float(np.dot(delta, up)) * up)
    # Latch Z error: raise/lower tip so crystal latch max Z clears below port min Z.
    z_err = np.zeros(3, dtype=np.float64)
    if not latch_z_ok:
        z_err[2] = (port_latch_z - float(latch_z_margin_m)) - crystal_latch_z
    pos_error = -lateral + z_err  # move tip to cancel crystal offset

    # Rotation: align crystal_axis → port_axis (small-angle approx).
    cross = np.cross(crystal_axis, port_axis)
    rot_error = cross  # magnitude ~ sin(theta) ≈ theta for small errors

    gap = mating_gap_along_axis(crystal, port)
    passed = bool(latch_z_ok and mating_sides_ok and axis_ok)
    return AlignmentResidual(
        latch_z_ok=latch_z_ok,
        mating_sides_ok=mating_sides_ok,
        axis_ok=axis_ok,
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
