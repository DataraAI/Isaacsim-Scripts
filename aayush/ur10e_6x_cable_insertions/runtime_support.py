"""Isaac-free runtime helpers for the six-arm cable demo."""

from __future__ import annotations

import math
import numpy as np


def validate_meters_per_unit(value: float) -> float:
    mpu = float(value)
    if not math.isfinite(mpu) or mpu <= 0.0:
        raise ValueError(f"metersPerUnit must be finite and positive, got {value!r}")
    return mpu


def meters_to_stage(position, meters_per_unit: float) -> np.ndarray:
    mpu = validate_meters_per_unit(meters_per_unit)
    return np.asarray(position, dtype=np.float64) / mpu


def stage_to_meters(position, meters_per_unit: float) -> np.ndarray:
    mpu = validate_meters_per_unit(meters_per_unit)
    return np.asarray(position, dtype=np.float64) * mpu


def physical_grasp_is_valid(
    *,
    initial_part,
    current_part,
    expected_tip,
    fingers,
    min_lift_m: float,
    max_tip_error_m: float,
    contact_rad: float,
) -> bool:
    initial = np.asarray(initial_part, dtype=np.float64).reshape(3)
    current = np.asarray(current_part, dtype=np.float64).reshape(3)
    tip = np.asarray(expected_tip, dtype=np.float64).reshape(3)
    finger_values = np.asarray(fingers, dtype=np.float64).reshape(-1)
    lifted = float(current[2] - initial[2]) >= float(min_lift_m)
    near_tip = float(np.linalg.norm(current - tip)) <= float(max_tip_error_m)
    closed = bool(finger_values.size) and (
        float(np.max(np.abs(finger_values))) >= float(contact_rad)
    )
    return bool(lifted and near_tip and closed)
