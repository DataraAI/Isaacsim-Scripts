#!/usr/bin/env python3
"""Pure domain model and qualification logic for alignment stress testing."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
from statistics import median
from typing import Mapping, Sequence

import numpy as np

STRESS_SCHEMA_VERSION = 1
STRESS_SEED = 20260722
Y_OFFSETS_MM = (-10, 0, 10)
Z_OFFSETS_MM = (-10, 0, 10)
REPEATS_PER_POSE = 3
REQUIRED_RUN_COUNT = 27
CHILD_TIMEOUT_S = 240.0
PARENT_TIMEOUT_S = 270.0
STEP_LIMIT_MM = 1.000001
ORIENTATION_LIMIT_DEG = 0.572958
GROUND_TRUTH_TARGET_LIMIT_MM = 1.0
RUN_ID_PATTERN = re.compile(r"^y([+-]\d{2})_z([+-]\d{2})_r([123])$")

CHILD_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "start_y_offset_mm",
    "start_z_offset_mm",
    "repeat",
    "started_at",
    "ended_at",
    "runtime_duration_s",
    "completed",
    "internal_timed_out",
    "track_acquired",
    "visual_alignment_locked",
    "final_center_error_px",
    "final_range_error_mm",
    "final_tool_target_world_m",
    "final_actual_tool_world_m",
    "final_physical_tracking_error_mm",
    "maximum_target_step_mm",
    "maximum_orientation_deviation_deg",
    "perception_rejection_count",
    "track_reacquisition_count",
    "fatal_error",
    "insertion_command_count",
)


@dataclass(frozen=True, order=True)
class StressCase:
    y_offset_mm: int
    z_offset_mm: int
    repeat: int

    def __post_init__(self) -> None:
        if self.y_offset_mm not in Y_OFFSETS_MM:
            raise ValueError("y offset must be -10, 0, or +10 mm")
        if self.z_offset_mm not in Z_OFFSETS_MM:
            raise ValueError("z offset must be -10, 0, or +10 mm")
        if not 1 <= self.repeat <= REPEATS_PER_POSE:
            raise ValueError("repeat must be 1, 2, or 3")

    @property
    def run_id(self) -> str:
        return f"y{self.y_offset_mm:+03d}_z{self.z_offset_mm:+03d}_r{self.repeat}"

    @property
    def directory_name(self) -> str:
        return (
            f"y{self.y_offset_mm:+03d}_z{self.z_offset_mm:+03d}_"
            f"repeat-{self.repeat}"
        )

    @classmethod
    def from_run_id(cls, run_id: str) -> "StressCase":
        match = RUN_ID_PATTERN.fullmatch(str(run_id))
        if match is None:
            raise ValueError(f"invalid stress run ID: {run_id!r}")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))


@dataclass(frozen=True)
class StressRunArgs:
    case: StressCase
    result_json: Path
    timeout_s: float
    exit_after_complete: bool


def build_stress_cases(seed: int = STRESS_SEED) -> list[StressCase]:
    cases = [
        StressCase(y, z, repeat)
        for y in Y_OFFSETS_MM
        for z in Z_OFFSETS_MM
        for repeat in range(1, REPEATS_PER_POSE + 1)
    ]
    random.Random(seed).shuffle(cases)
    return cases


def parse_stress_run_args(argv: Sequence[str]) -> StressRunArgs | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start-y-offset-mm", type=int)
    parser.add_argument("--start-z-offset-mm", type=int)
    parser.add_argument("--stress-run-id")
    parser.add_argument("--stress-result-json", type=Path)
    parser.add_argument("--stress-timeout-s", type=float)
    parser.add_argument("--exit-after-complete", action="store_true")
    namespace, _unknown = parser.parse_known_args(list(argv))

    supplied = (
        namespace.start_y_offset_mm,
        namespace.start_z_offset_mm,
        namespace.stress_run_id,
        namespace.stress_result_json,
        namespace.stress_timeout_s,
    )
    stress_requested = namespace.exit_after_complete or any(
        value is not None for value in supplied
    )
    if not stress_requested:
        return None
    if any(value is None for value in supplied) or not namespace.exit_after_complete:
        raise ValueError(
            "all stress-run arguments, including --exit-after-complete, "
            "must be supplied together"
        )

    case = StressCase.from_run_id(namespace.stress_run_id)
    if (case.y_offset_mm, case.z_offset_mm) != (
        namespace.start_y_offset_mm,
        namespace.start_z_offset_mm,
    ):
        raise ValueError("run ID and start offsets disagree")

    timeout_s = float(namespace.stress_timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("stress timeout must be finite and positive")

    return StressRunArgs(
        case=case,
        result_json=Path(namespace.stress_result_json),
        timeout_s=timeout_s,
        exit_after_complete=True,
    )


def derive_stress_config(base_config, args: StressRunArgs):
    x, y, z = base_config.ik.initial_position
    return replace(
        base_config,
        ik=replace(
            base_config.ik,
            initial_position=(
                float(x),
                float(y) + args.case.y_offset_mm / 1000.0,
                float(z) + args.case.z_offset_mm / 1000.0,
            ),
        ),
    )


def quaternion_angular_distance_deg(first, second) -> float:
    """Return the shortest angular distance between scalar-first quaternions."""
    a = np.asarray(first, dtype=np.float64).reshape(4)
    b = np.asarray(second, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("quaternions must be finite")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1.0e-12 or b_norm <= 1.0e-12:
        raise ValueError("quaternions must be nonzero")
    cosine = abs(float(np.dot(a / a_norm, b / b_norm)))
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(2.0 * math.acos(cosine))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def new_child_result(args: StressRunArgs, started_at: str) -> dict[str, object]:
    return {
        "schema_version": STRESS_SCHEMA_VERSION,
        "run_id": args.case.run_id,
        "start_y_offset_mm": args.case.y_offset_mm,
        "start_z_offset_mm": args.case.z_offset_mm,
        "repeat": args.case.repeat,
        "started_at": str(started_at),
        "ended_at": "",
        "runtime_duration_s": None,
        "completed": False,
        "internal_timed_out": False,
        "track_acquired": False,
        "visual_alignment_locked": False,
        "final_center_error_px": None,
        "final_range_error_mm": None,
        "final_tool_target_world_m": None,
        "final_actual_tool_world_m": None,
        "final_physical_tracking_error_mm": None,
        "maximum_target_step_mm": 0.0,
        "maximum_orientation_deviation_deg": 0.0,
        "perception_rejection_count": 0,
        "track_reacquisition_count": 0,
        "fatal_error": "",
        "insertion_command_count": 0,
    }


def expected_preinsert_target_world_m(center, normal, standoff_m) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float64).reshape(3)
    normal_array = np.asarray(normal, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center_array)) or not np.all(np.isfinite(normal_array)):
        raise ValueError("ground-truth center and normal must be finite")
    normal_norm = float(np.linalg.norm(normal_array))
    if normal_norm <= 1.0e-12:
        raise ValueError("ground-truth normal must be nonzero")
    distance = float(standoff_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("standoff must be finite and positive")
    return center_array + normal_array / normal_norm * distance


def _finite_number(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_vector3(payload: Mapping[str, object], key: str) -> np.ndarray | None:
    value = payload.get(key)
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    return vector if np.all(np.isfinite(vector)) else None


def finalize_parent_result(
    *,
    child_payload: Mapping[str, object],
    subprocess_exit_status: int,
    parent_hard_timed_out: bool,
    console_log_path: str,
    child_result_parse_status: str,
    truth_center_world_m,
    truth_normal_world,
    preinsert_standoff_m: float,
) -> dict[str, object]:
    child = dict(child_payload)
    expected_target = expected_preinsert_target_world_m(
        truth_center_world_m,
        truth_normal_world,
        preinsert_standoff_m,
    )
    final_target = _finite_vector3(child, "final_tool_target_world_m")
    target_error_mm = (
        1000.0 * float(np.linalg.norm(final_target - expected_target))
        if final_target is not None
        else None
    )

    failed: list[str] = []
    missing = [field for field in CHILD_REQUIRED_FIELDS if field not in child]
    if missing:
        failed.append("required_fields")

    if int(subprocess_exit_status) != 0:
        failed.append("subprocess_exit_status")
    if bool(parent_hard_timed_out):
        failed.append("parent_hard_timed_out")
    if child_result_parse_status != "valid":
        failed.append("child_result_parse_status")
    if child.get("schema_version") != STRESS_SCHEMA_VERSION:
        failed.append("schema_version")

    boolean_gates = (
        ("completed", True),
        ("internal_timed_out", False),
        ("track_acquired", True),
        ("visual_alignment_locked", True),
    )
    for key, expected in boolean_gates:
        if child.get(key) is not expected:
            failed.append(key)

    fatal_error = child.get("fatal_error")
    if not isinstance(fatal_error, str) or fatal_error:
        failed.append("fatal_error")

    insertion_count = _finite_number(child, "insertion_command_count")
    if insertion_count is None or insertion_count != 0.0:
        failed.append("insertion_command_count")

    numeric_limits = (
        ("runtime_duration_s", 240.0, False),
        ("final_center_error_px", 2.0, False),
        ("final_range_error_mm", 3.0, True),
        ("final_physical_tracking_error_mm", 0.3, False),
        ("maximum_target_step_mm", STEP_LIMIT_MM, False),
        ("maximum_orientation_deviation_deg", ORIENTATION_LIMIT_DEG, False),
    )
    for key, limit, absolute in numeric_limits:
        value = _finite_number(child, key)
        gate_name = "absolute_final_range_error_mm" if absolute else key
        if value is None or (abs(value) if absolute else value) > limit:
            failed.append(gate_name)

    if target_error_mm is None or target_error_mm > GROUND_TRUTH_TARGET_LIMIT_MM:
        failed.append("ground_truth_target_error_mm")

    # Required counters and both final poses must also be finite, even when they
    # are not separate threshold gates.
    for key in ("perception_rejection_count", "track_reacquisition_count"):
        value = _finite_number(child, key)
        if value is None or value < 0.0:
            failed.append("required_fields")
    if final_target is None or _finite_vector3(child, "final_actual_tool_world_m") is None:
        failed.append("required_fields")

    result = dict(child)
    result.update(
        {
            "subprocess_exit_status": int(subprocess_exit_status),
            "parent_hard_timed_out": bool(parent_hard_timed_out),
            "console_log_path": str(console_log_path),
            "child_result_parse_status": str(child_result_parse_status),
            "expected_preinsert_target_world_m": expected_target.tolist(),
            "ground_truth_target_error_mm": target_error_mm,
            "failed_gates": sorted(set(failed)),
            "qualified": not failed,
        }
    )
    return result


def _finite_values(results: Sequence[Mapping[str, object]], key: str, *, absolute=False) -> list[float]:
    values: list[float] = []
    for result in results:
        value = _finite_number(result, key)
        if value is not None:
            values.append(abs(value) if absolute else value)
    return values


def _duration_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(median(values)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def aggregate_suite(
    results: Sequence[Mapping[str, object]],
    execution_order: Sequence[str],
) -> dict[str, object]:
    result_list = [dict(item) for item in results]
    passed = sum(bool(item.get("qualified")) for item in result_list)
    failed_count = len(result_list) - passed

    failures_by_pose: Counter[str] = Counter()
    failures_by_gate: Counter[str] = Counter()
    for item in result_list:
        if bool(item.get("qualified")):
            continue
        try:
            case = StressCase(
                int(item["start_y_offset_mm"]),
                int(item["start_z_offset_mm"]),
                int(item["repeat"]),
            )
            pose = f"y{case.y_offset_mm:+03d}_z{case.z_offset_mm:+03d}"
        except (KeyError, TypeError, ValueError):
            pose = "unknown"
        failures_by_pose[pose] += 1
        gates = item.get("failed_gates", [])
        if isinstance(gates, (list, tuple)):
            for gate in gates:
                failures_by_gate[str(gate)] += 1

    run_ids = [str(item.get("run_id", "")) for item in result_list]
    expected_order = list(execution_order)
    complete_matrix = (
        len(result_list) == REQUIRED_RUN_COUNT
        and len(expected_order) == REQUIRED_RUN_COUNT
        and len(set(run_ids)) == REQUIRED_RUN_COUNT
        and set(run_ids) == set(expected_order)
    )

    durations = _finite_values(result_list, "runtime_duration_s")
    worst_fields = {
        "worst_center_error_px": ("final_center_error_px", False),
        "worst_absolute_range_error_mm": ("final_range_error_mm", True),
        "worst_ground_truth_target_error_mm": ("ground_truth_target_error_mm", False),
        "worst_physical_tracking_error_mm": ("final_physical_tracking_error_mm", False),
        "maximum_target_step_mm": ("maximum_target_step_mm", False),
        "maximum_orientation_deviation_deg": ("maximum_orientation_deviation_deg", False),
    }
    worst: dict[str, float | None] = {}
    for output_key, (source_key, absolute) in worst_fields.items():
        values = _finite_values(result_list, source_key, absolute=absolute)
        worst[output_key] = max(values) if values else None

    rejection_values = _finite_values(result_list, "perception_rejection_count")
    reacquisition_values = _finite_values(result_list, "track_reacquisition_count")

    summary: dict[str, object] = {
        "schema_version": STRESS_SCHEMA_VERSION,
        "seed": STRESS_SEED,
        "required_run_count": REQUIRED_RUN_COUNT,
        "completed_run_count": len(result_list),
        "passed_run_count": passed,
        "failed_run_count": failed_count,
        "unique_run_count": len(set(run_ids)),
        "execution_order": expected_order,
        "completed_run_ids": run_ids,
        "failures_by_pose": dict(sorted(failures_by_pose.items())),
        "failures_by_gate": dict(sorted(failures_by_gate.items())),
        "duration_s": _duration_summary(durations),
        "perception_rejection_total": int(sum(rejection_values)),
        "track_reacquisition_total": int(sum(reacquisition_values)),
        "QUALIFIED": complete_matrix and passed == REQUIRED_RUN_COUNT,
    }
    summary.update(worst)
    return summary
