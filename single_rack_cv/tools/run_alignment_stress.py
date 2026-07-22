#!/usr/bin/env python3
"""Run and score the deterministic 27-run alignment stress qualification."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from stress_alignment import (
    CHILD_REQUIRED_FIELDS,
    CHILD_TIMEOUT_S,
    PARENT_TIMEOUT_S,
    STRESS_SCHEMA_VERSION,
    STRESS_SEED,
    StressCase,
    StressRunArgs,
    aggregate_suite,
    build_stress_cases,
    finalize_parent_result,
    new_child_result,
    write_json_atomic,
)

OUTPUT_ROOT = CONFIG.camera.output_dir / "alignment_stress"
GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmarks" / "front_plane_ground_truth.json"
ISAAC_PYTHON = Path.home() / "isaacsim" / "python.sh"
PREINSERT_STANDOFF_M = 0.050
EXPECTED_RESOLUTION = [960, 1280]
EXPECTED_TRUTH_SOURCE = "automatic_rtx_mesh_raycast_front_bezel_plane"

CSV_FIELDS = [
    "run_id",
    "start_y_offset_mm",
    "start_z_offset_mm",
    "repeat",
    "runtime_duration_s",
    "subprocess_exit_status",
    "internal_timed_out",
    "parent_hard_timed_out",
    "completed",
    "track_acquired",
    "visual_alignment_locked",
    "final_center_error_px",
    "final_range_error_mm",
    "final_physical_tracking_error_mm",
    "ground_truth_target_error_mm",
    "maximum_target_step_mm",
    "maximum_orientation_deviation_deg",
    "perception_rejection_count",
    "track_reacquisition_count",
    "insertion_command_count",
    "child_result_parse_status",
    "qualified",
    "failed_gates",
    "fatal_error",
]


def build_child_command(
    isaac_python: Path,
    project_root: Path,
    case: StressCase,
    child_result_json: Path,
) -> list[str]:
    return [
        str(isaac_python),
        str(project_root / "main.py"),
        "--start-y-offset-mm",
        str(case.y_offset_mm),
        "--start-z-offset-mm",
        str(case.z_offset_mm),
        "--stress-run-id",
        case.run_id,
        "--stress-result-json",
        str(child_result_json),
        "--stress-timeout-s",
        f"{CHILD_TIMEOUT_S:.1f}",
        "--exit-after-complete",
    ]


def _finite_vector3(payload: Mapping[str, object], key: str) -> list[float]:
    try:
        vector = np.asarray(payload[key], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ground truth {key} must contain three numbers") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"ground truth {key} must be finite")
    return [float(value) for value in vector]


def load_valid_ground_truth(path: Path) -> dict[str, object]:
    truth_path = Path(path)
    try:
        payload = json.loads(truth_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"ground truth is missing: {truth_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"ground truth is unreadable: {truth_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("ground truth root must be an object")
    if int(payload.get("schema_version", -1)) < 4:
        raise ValueError("ground truth schema must be at least 4")
    if payload.get("camera_resolution_height_width") != EXPECTED_RESOLUTION:
        raise ValueError("ground truth resolution must be 1280x960")
    if payload.get("source") != EXPECTED_TRUTH_SOURCE:
        raise ValueError("ground truth source is not the automatic RTX extractor")
    if not str(payload.get("control_usage", "")).lower().startswith("forbidden"):
        raise ValueError("ground truth must be marked forbidden for control")
    center = _finite_vector3(payload, "center_world_m")
    normal = _finite_vector3(payload, "normal_world")
    if float(np.linalg.norm(normal)) <= 1.0e-12:
        raise ValueError("ground truth normal must be nonzero")
    used_paths = payload.get("used_prim_paths")
    if not isinstance(used_paths, list) or not used_paths:
        raise ValueError("ground truth must list used rack prim paths")
    if not all(
        isinstance(item, str) and item.startswith(CONFIG.scene.rack_path)
        for item in used_paths
    ):
        raise ValueError("ground truth contains hits outside the configured rack")
    validated = dict(payload)
    validated["center_world_m"] = center
    validated["normal_world"] = normal
    return validated


def _terminate_process_group(process: subprocess.Popen, grace_s: float = 10.0) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return int(process.wait(timeout=grace_s))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait(timeout=grace_s))


def run_one_case(
    isaac_python: Path,
    project_root: Path,
    case: StressCase,
    run_directory: Path,
) -> tuple[int, bool, float]:
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=False)
    child_result = run_directory / "child_result.json"
    console_log = run_directory / "console.log"
    command = build_child_command(
        isaac_python,
        project_root,
        case,
        child_result,
    )
    started = time.monotonic()
    hard_timeout = False
    with console_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_status = int(process.wait(timeout=PARENT_TIMEOUT_S))
        except subprocess.TimeoutExpired:
            hard_timeout = True
            exit_status = _terminate_process_group(process)
        except BaseException:
            _terminate_process_group(process)
            raise
    return exit_status, hard_timeout, time.monotonic() - started


def _synthesized_child(
    case: StressCase,
    path: Path,
    reason: str,
) -> dict[str, object]:
    args = StressRunArgs(
        case=case,
        result_json=path,
        timeout_s=CHILD_TIMEOUT_S,
        exit_after_complete=True,
    )
    payload = new_child_result(args, "")
    payload.update(
        {
            "ended_at": "",
            "runtime_duration_s": 0.0,
            "fatal_error": reason,
        }
    )
    return payload


def load_child_result(
    case: StressCase,
    path: Path,
) -> tuple[dict[str, object], str]:
    child_path = Path(path)
    if not child_path.is_file():
        return _synthesized_child(case, child_path, "child result is missing"), "missing"
    try:
        payload = json.loads(child_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            _synthesized_child(case, child_path, f"child result is malformed: {exc}"),
            "malformed",
        )
    if not isinstance(payload, dict):
        return _synthesized_child(case, child_path, "child result root is not an object"), "malformed"
    identity = (
        payload.get("run_id"),
        payload.get("start_y_offset_mm"),
        payload.get("start_z_offset_mm"),
        payload.get("repeat"),
    )
    expected = (
        case.run_id,
        case.y_offset_mm,
        case.z_offset_mm,
        case.repeat,
    )
    if identity != expected:
        return _synthesized_child(case, child_path, "child result identity mismatch"), "mismatch"
    if any(field not in payload for field in CHILD_REQUIRED_FIELDS):
        return _synthesized_child(case, child_path, "child result schema is incomplete"), "malformed"
    return dict(payload), "valid"


def create_suite_directory(output_root: Path = OUTPUT_ROOT) -> Path:
    root = Path(output_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(100):
        name = stamp if suffix == 0 else f"{stamp}-{suffix:02d}"
        candidate = root / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a unique suite directory")


def _csv_value(value):
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if value is None:
        return ""
    return value


def write_suite_outputs(
    suite_directory: Path,
    results: Sequence[Mapping[str, object]],
    execution_order: Sequence[str],
) -> dict[str, object]:
    suite_directory = Path(suite_directory)
    summary = aggregate_suite(results, execution_order)
    write_json_atomic(suite_directory / "summary.json", summary)

    csv_path = suite_directory / "summary.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = {field: _csv_value(result.get(field)) for field in CSV_FIELDS}
            writer.writerow(row)
    temporary_csv.replace(csv_path)

    duration = summary["duration_s"]
    report_lines = [
        "ALIGNMENT STRESS QUALIFICATION",
        f"passed_run_count={summary['passed_run_count']}",
        f"failed_run_count={summary['failed_run_count']}",
        f"QUALIFIED={summary['QUALIFIED']}",
        f"completed_run_count={summary['completed_run_count']}",
        f"required_run_count={summary['required_run_count']}",
        f"seed={summary['seed']}",
        f"worst_center_error_px={summary['worst_center_error_px']}",
        f"worst_absolute_range_error_mm={summary['worst_absolute_range_error_mm']}",
        f"worst_ground_truth_target_error_mm={summary['worst_ground_truth_target_error_mm']}",
        f"worst_physical_tracking_error_mm={summary['worst_physical_tracking_error_mm']}",
        f"maximum_target_step_mm={summary['maximum_target_step_mm']}",
        f"maximum_orientation_deviation_deg={summary['maximum_orientation_deviation_deg']}",
        f"duration_s_min={duration['minimum']}",
        f"duration_s_median={duration['median']}",
        f"duration_s_p95={duration['p95']}",
        f"duration_s_max={duration['maximum']}",
        f"perception_rejection_total={summary['perception_rejection_total']}",
        f"track_reacquisition_total={summary['track_reacquisition_total']}",
        f"failures_by_pose={json.dumps(summary['failures_by_pose'], sort_keys=True)}",
        f"failures_by_gate={json.dumps(summary['failures_by_gate'], sort_keys=True)}",
        "execution_order=" + ",".join(summary["execution_order"]),
    ]
    report_path = suite_directory / "report.txt"
    temporary_report = report_path.with_suffix(".txt.tmp")
    temporary_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    return summary


def _validate_infrastructure(isaac_python: Path, truth_path: Path) -> dict[str, object]:
    if not isaac_python.is_file():
        raise RuntimeError(f"Isaac Python launcher is missing: {isaac_python}")
    if not os.access(isaac_python, os.X_OK):
        raise RuntimeError(f"Isaac Python launcher is not executable: {isaac_python}")
    return load_valid_ground_truth(truth_path)


def main() -> int:
    suite_directory: Path | None = None
    results: list[dict[str, object]] = []
    cases = build_stress_cases(STRESS_SEED)
    execution_order = [case.run_id for case in cases]
    try:
        truth = _validate_infrastructure(ISAAC_PYTHON, GROUND_TRUTH_PATH)
        suite_directory = create_suite_directory()
        print(f"[ALIGNMENT STRESS] output={suite_directory}", flush=True)
        for index, case in enumerate(cases, start=1):
            run_directory = suite_directory / "runs" / case.directory_name
            print(f"[{index:02d}/27] START {case.run_id}", flush=True)
            exit_status, hard_timeout, _parent_duration = run_one_case(
                ISAAC_PYTHON,
                PROJECT_ROOT,
                case,
                run_directory,
            )
            child_path = run_directory / "child_result.json"
            child, parse_status = load_child_result(case, child_path)
            result = finalize_parent_result(
                child_payload=child,
                subprocess_exit_status=exit_status,
                parent_hard_timed_out=hard_timeout,
                console_log_path=str(
                    (run_directory / "console.log").relative_to(suite_directory)
                ),
                child_result_parse_status=parse_status,
                truth_center_world_m=truth["center_world_m"],
                truth_normal_world=truth["normal_world"],
                preinsert_standoff_m=PREINSERT_STANDOFF_M,
            )
            write_json_atomic(run_directory / "result.json", result)
            results.append(result)
            summary = write_suite_outputs(
                suite_directory,
                results,
                execution_order,
            )
            status = "PASS" if result["qualified"] else "FAIL"
            print(
                f"[{index:02d}/27] {status} {case.run_id} "
                f"gates={result['failed_gates']}",
                flush=True,
            )
        return 0 if summary["QUALIFIED"] else 2
    except KeyboardInterrupt:
        print("[ALIGNMENT STRESS] interrupted", file=sys.stderr, flush=True)
        if suite_directory is not None:
            write_suite_outputs(suite_directory, results, execution_order)
        return 1
    except Exception as exc:
        print(f"[ALIGNMENT STRESS] infrastructure failure: {exc}", file=sys.stderr, flush=True)
        if suite_directory is not None:
            write_suite_outputs(suite_directory, results, execution_order)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
