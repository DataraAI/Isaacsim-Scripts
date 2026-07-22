# Alignment Stress-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 27-run qualification harness that launches a fresh Isaac Sim process for every world-frame Y/Z start offset and fails unless all 27 runs satisfy the approved safety and accuracy gates.

**Architecture:** `stress_alignment.py` is pure Python and owns the matrix, run-ID parsing, config derivation, child/final result schemas, benchmark-only target scoring, gates, and suite aggregation. `sim.py` only observes existing commands and exposes a stress snapshot. `main.py` gains an opt-in child mode that never reads ground truth. `tools/run_alignment_stress.py` validates canonical ground truth before launching children, starts 27 sequential Isaac subprocesses, finalizes each result after the child exits, and writes suite reports.

**Tech Stack:** Python 3.12, standard library, NumPy, Isaac Sim 6.0.0, Lula IK, existing YOLOE/SGBM controller, `unittest`, Bash.

## Global Constraints

- Platform remains Ubuntu 24.04 with Isaac Sim 6.0.0.
- Camera resolution remains `(960, 1280)` height-width.
- Runtime control remains image-only.
- The child process must never read RTX/USD ground truth.
- The parent may read benchmark ground truth only outside the live control loop.
- Wrist orientation remains fixed and vision never commands orientation.
- Existing ToolCenter target updates remain bounded to 1 mm.
- Qualification comparison allows exactly `1e-9 m`, which is `0.000001 mm`, so the recorded maximum must be `<=1.000001 mm`.
- Failed observations hold the target and trigger existing reacquisition behavior.
- No insertion command is added.
- Normal `main.py` behavior remains unchanged without stress arguments.
- Matrix is world Y/Z `(-10, 0, +10)` mm, X unchanged, three repeats per pose.
- Execution order uses seed `20260722`.
- Child internal timeout is 240 seconds.
- Parent hard timeout is 270 seconds.
- Ground-truth gate compares the final ToolCenter target against the expected 50 mm pre-insert target derived from ground-truth opening center and normalized front-plane normal.
- Orientation deviation is recorded in degrees and must be `<=0.572958°`, equivalent to 0.01 rad.
- Suite qualification requires 27/27 runs.

---

## File Map

- Create `single_rack_cv/stress_alignment.py`: pure domain model, schemas, scoring, gates, and aggregation.
- Create `single_rack_cv/tests/test_alignment_stress.py`: pure tests for matrix, parsing, config offsets, scoring, and gates.
- Modify `single_rack_cv/sim.py`: passive measurement fields and `stress_snapshot()`.
- Modify `single_rack_cv/main.py`: optional child stress lifecycle and `child_result.json` emission.
- Create `single_rack_cv/tools/run_alignment_stress.py`: parent orchestration, finalized `result.json`, and suite reports.
- Create `single_rack_cv/tests/test_alignment_stress_runner.py`: parent-runner tests without Isaac startup.
- Create `single_rack_cv/tools/run_alignment_stress.sh`: sanitized one-command launcher.
- Modify `single_rack_cv/tests/test_runtime_wiring.py`: structural safety guards.
- Modify `single_rack_cv/README.md`: exact command, output, exit codes, and kill switch.

---

### Task 1: Pure stress domain model, schemas, and gates

**Files:**
- Create: `single_rack_cv/stress_alignment.py`
- Create: `single_rack_cv/tests/test_alignment_stress.py`

**Interfaces:**
- Produces `StressCase`, `StressRunArgs`, `build_stress_cases`, `parse_stress_run_args`, `derive_stress_config`, `new_child_result`, `expected_preinsert_target_world_m`, `finalize_parent_result`, `aggregate_suite`, and `write_json_atomic`.
- `StressCase.from_run_id()` extracts repeat from `y+00_z-10_r2`; no extra `--stress-repeat` argument is introduced.

- [ ] **Step 1: Write failing matrix, parsing, and config tests**

```python
from __future__ import annotations

import math
from pathlib import Path
import unittest

from config import CONFIG
from stress_alignment import (
    STRESS_SEED,
    StressCase,
    StressRunArgs,
    build_stress_cases,
    derive_stress_config,
    parse_stress_run_args,
)


class AlignmentStressTests(unittest.TestCase):
    def test_matrix_is_nine_poses_three_repeats(self):
        cases = build_stress_cases()
        self.assertEqual(len(cases), 27)
        counts = {}
        for case in cases:
            key = (case.y_offset_mm, case.z_offset_mm)
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(
            counts,
            {(y, z): 3 for y in (-10, 0, 10) for z in (-10, 0, 10)},
        )
        self.assertEqual(len({case.run_id for case in cases}), 27)

    def test_shuffle_is_reproducible(self):
        first = [case.run_id for case in build_stress_cases(STRESS_SEED)]
        second = [case.run_id for case in build_stress_cases(STRESS_SEED)]
        other = [case.run_id for case in build_stress_cases(STRESS_SEED + 1)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_run_id_round_trip(self):
        case = StressCase(y_offset_mm=-10, z_offset_mm=10, repeat=2)
        self.assertEqual(case.run_id, "y-10_z+10_r2")
        self.assertEqual(StressCase.from_run_id(case.run_id), case)

    def test_cli_uses_only_approved_arguments(self):
        args = parse_stress_run_args(
            [
                "--start-y-offset-mm", "-10",
                "--start-z-offset-mm", "10",
                "--stress-run-id", "y-10_z+10_r2",
                "--stress-result-json", "/tmp/child_result.json",
                "--stress-timeout-s", "240",
                "--exit-after-complete",
            ]
        )
        self.assertEqual(args.case, StressCase(-10, 10, 2))
        self.assertEqual(args.result_json, Path("/tmp/child_result.json"))

    def test_config_offsets_change_only_world_y_and_z(self):
        args = StressRunArgs(
            case=StressCase(10, -10, 1),
            result_json=Path("/tmp/child_result.json"),
            timeout_s=240.0,
            exit_after_complete=True,
        )
        derived = derive_stress_config(CONFIG, args)
        x, y, z = CONFIG.ik.initial_position
        self.assertEqual(derived.ik.initial_position[0], x)
        self.assertAlmostEqual(derived.ik.initial_position[1], y + 0.010)
        self.assertAlmostEqual(derived.ik.initial_position[2], z - 0.010)
        self.assertEqual(
            derived.ik.initial_orientation_wxyz,
            CONFIG.ik.initial_orientation_wxyz,
        )
        self.assertEqual(CONFIG.ik.initial_position, (x, y, z))
```

- [ ] **Step 2: Run and confirm failure**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
/usr/bin/python3 -m unittest -v tests.test_alignment_stress
```

Expected: import failure because `stress_alignment.py` does not exist.

- [ ] **Step 3: Implement matrix, run-ID parsing, CLI parsing, and config derivation**

```python
from __future__ import annotations

import argparse
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
CHILD_TIMEOUT_S = 240.0
PARENT_TIMEOUT_S = 270.0
STEP_LIMIT_MM = 1.000001
ORIENTATION_LIMIT_DEG = 0.572958
RUN_ID_PATTERN = re.compile(r"^y([+-]\d{2})_z([+-]\d{2})_r([123])$")


@dataclass(frozen=True, order=True)
class StressCase:
    y_offset_mm: int
    z_offset_mm: int
    repeat: int

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
        match = RUN_ID_PATTERN.fullmatch(run_id)
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
    namespace, _ = parser.parse_known_args(list(argv))
    supplied = (
        namespace.start_y_offset_mm,
        namespace.start_z_offset_mm,
        namespace.stress_run_id,
        namespace.stress_result_json,
        namespace.stress_timeout_s,
    )
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise ValueError("all stress-run arguments must be supplied together")
    case = StressCase.from_run_id(namespace.stress_run_id)
    if case.y_offset_mm != namespace.start_y_offset_mm:
        raise ValueError("run ID and Y offset disagree")
    if case.z_offset_mm != namespace.start_z_offset_mm:
        raise ValueError("run ID and Z offset disagree")
    if case.y_offset_mm not in Y_OFFSETS_MM or case.z_offset_mm not in Z_OFFSETS_MM:
        raise ValueError("stress offsets must be -10, 0, or +10 mm")
    timeout_s = float(namespace.stress_timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("stress timeout must be finite and positive")
    return StressRunArgs(
        case=case,
        result_json=namespace.stress_result_json,
        timeout_s=timeout_s,
        exit_after_complete=bool(namespace.exit_after_complete),
    )


def derive_stress_config(base_config, args: StressRunArgs):
    x, y, z = base_config.ik.initial_position
    stressed_ik = replace(
        base_config.ik,
        initial_position=(
            float(x),
            float(y) + args.case.y_offset_mm / 1000.0,
            float(z) + args.case.z_offset_mm / 1000.0,
        ),
    )
    return replace(base_config, ik=stressed_ik)
```

The existing `ik.initial_position` is a panda-hand pose. With fixed orientation and fixed hand-to-tool transform, adding world Y/Z to that pose applies the same world Y/Z delta to ToolCenter without changing X or orientation.

- [ ] **Step 4: Write failing child/final schema and scoring tests**

```python
from stress_alignment import (
    expected_preinsert_target_world_m,
    finalize_parent_result,
    new_child_result,
)

    def _passing_child(self):
        return {
            "schema_version": 1,
            "run_id": "y+00_z+00_r1",
            "start_y_offset_mm": 0,
            "start_z_offset_mm": 0,
            "repeat": 1,
            "started_at": "2026-07-22T00:00:00+00:00",
            "ended_at": "2026-07-22T00:01:00+00:00",
            "runtime_duration_s": 60.0,
            "completed": True,
            "internal_timed_out": False,
            "track_acquired": True,
            "visual_alignment_locked": True,
            "final_center_error_px": 2.0,
            "final_range_error_mm": -3.0,
            "final_tool_target_world_m": [1.05, 2.0, 3.0],
            "final_actual_tool_world_m": [1.0503, 2.0, 3.0],
            "final_physical_tracking_error_mm": 0.3,
            "maximum_target_step_mm": 1.000001,
            "maximum_orientation_deviation_deg": 0.572958,
            "perception_rejection_count": 0,
            "track_reacquisition_count": 0,
            "fatal_error": "",
            "insertion_command_count": 0,
        }

    def test_expected_target_uses_center_plus_normalized_outward_normal(self):
        target = expected_preinsert_target_world_m(
            center_world_m=[1.0, 2.0, 3.0],
            normal_world=[2.0, 0.0, 0.0],
            standoff_m=0.05,
        )
        self.assertTrue(np.allclose(target, [1.05, 2.0, 3.0]))

    def test_exact_gate_boundaries_pass(self):
        result = finalize_parent_result(
            child_payload=self._passing_child(),
            subprocess_exit_status=0,
            parent_hard_timed_out=False,
            console_log_path="runs/y+00_z+00_repeat-1/console.log",
            child_result_parse_status="valid",
            truth_center_world_m=[1.0, 2.0, 3.0],
            truth_normal_world=[1.0, 0.0, 0.0],
            preinsert_standoff_m=0.05,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(result["failed_gates"], [])
        self.assertAlmostEqual(result["ground_truth_target_error_mm"], 0.0)

    def test_one_over_each_limit_fails(self):
        changes = {
            "final_center_error_px": 2.000001,
            "final_range_error_mm": 3.000001,
            "final_physical_tracking_error_mm": 0.300001,
            "maximum_target_step_mm": 1.000002,
            "maximum_orientation_deviation_deg": 0.572959,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                child = self._passing_child()
                child[key] = value
                result = finalize_parent_result(
                    child_payload=child,
                    subprocess_exit_status=0,
                    parent_hard_timed_out=False,
                    console_log_path="console.log",
                    child_result_parse_status="valid",
                    truth_center_world_m=[1.0, 2.0, 3.0],
                    truth_normal_world=[1.0, 0.0, 0.0],
                    preinsert_standoff_m=0.05,
                )
                self.assertFalse(result["qualified"])
```

- [ ] **Step 5: Implement child schema, target derivation, finalization, and gates**

```python
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_child_result(args: StressRunArgs, started_at: str) -> dict[str, object]:
    return {
        "schema_version": STRESS_SCHEMA_VERSION,
        "run_id": args.case.run_id,
        "start_y_offset_mm": args.case.y_offset_mm,
        "start_z_offset_mm": args.case.z_offset_mm,
        "repeat": args.case.repeat,
        "started_at": started_at,
        "ended_at": "",
        "runtime_duration_s": math.nan,
        "completed": False,
        "internal_timed_out": False,
        "track_acquired": False,
        "visual_alignment_locked": False,
        "final_center_error_px": math.nan,
        "final_range_error_mm": math.nan,
        "final_tool_target_world_m": None,
        "final_actual_tool_world_m": None,
        "final_physical_tracking_error_mm": math.nan,
        "maximum_target_step_mm": 0.0,
        "maximum_orientation_deviation_deg": 0.0,
        "perception_rejection_count": 0,
        "track_reacquisition_count": 0,
        "fatal_error": "",
        "insertion_command_count": 0,
    }


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expected_preinsert_target_world_m(
    center_world_m: Sequence[float],
    normal_world: Sequence[float],
    standoff_m: float,
) -> np.ndarray:
    center = np.asarray(center_world_m, dtype=np.float64).reshape(3)
    normal = np.asarray(normal_world, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(normal)):
        raise ValueError("ground-truth center and normal must be finite")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-12:
        raise ValueError("ground-truth normal must be nonzero")
    if not math.isfinite(standoff_m) or standoff_m <= 0.0:
        raise ValueError("standoff must be finite and positive")
    return center + (normal / normal_norm) * float(standoff_m)


def _finite(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def finalize_parent_result(
    *,
    child_payload: Mapping[str, object],
    subprocess_exit_status: int,
    parent_hard_timed_out: bool,
    console_log_path: str,
    child_result_parse_status: str,
    truth_center_world_m: Sequence[float],
    truth_normal_world: Sequence[float],
    preinsert_standoff_m: float,
) -> dict[str, object]:
    result = dict(child_payload)
    expected_target = expected_preinsert_target_world_m(
        truth_center_world_m,
        truth_normal_world,
        preinsert_standoff_m,
    )
    try:
        final_target = np.asarray(
            result["final_tool_target_world_m"],
            dtype=np.float64,
        ).reshape(3)
        if not np.all(np.isfinite(final_target)):
            raise ValueError("non-finite final target")
        target_error_mm = 1000.0 * float(np.linalg.norm(final_target - expected_target))
    except Exception:
        target_error_mm = math.nan

    result.update(
        {
            "subprocess_exit_status": int(subprocess_exit_status),
            "parent_hard_timed_out": bool(parent_hard_timed_out),
            "console_log_path": str(console_log_path),
            "child_result_parse_status": str(child_result_parse_status),
            "expected_preinsert_target_world_m": [float(v) for v in expected_target],
            "ground_truth_target_error_mm": target_error_mm,
        }
    )

    failed: list[str] = []
    boolean_gates = {
        "subprocess_exit_status": subprocess_exit_status == 0,
        "internal_timeout": result.get("internal_timed_out") is False,
        "parent_hard_timeout": parent_hard_timed_out is False,
        "completed": result.get("completed") is True,
        "track_acquired": result.get("track_acquired") is True,
        "visual_alignment_locked": result.get("visual_alignment_locked") is True,
        "child_result_parse": child_result_parse_status == "valid",
        "fatal_error": not str(result.get("fatal_error", "")).strip(),
        "no_insertion": result.get("insertion_command_count") == 0,
    }
    failed.extend(name for name, passed in boolean_gates.items() if not passed)

    measured = {
        "runtime_duration_s": _finite(result, "runtime_duration_s"),
        "final_center_error_px": _finite(result, "final_center_error_px"),
        "absolute_final_range_error_mm": (
            None
            if _finite(result, "final_range_error_mm") is None
            else abs(float(result["final_range_error_mm"]))
        ),
        "final_physical_tracking_error_mm": _finite(
            result, "final_physical_tracking_error_mm"
        ),
        "ground_truth_target_error_mm": (
            target_error_mm if math.isfinite(target_error_mm) else None
        ),
        "maximum_target_step_mm": _finite(result, "maximum_target_step_mm"),
        "maximum_orientation_deviation_deg": _finite(
            result, "maximum_orientation_deviation_deg"
        ),
    }
    limits = {
        "runtime_duration_s": 240.0,
        "final_center_error_px": 2.0,
        "absolute_final_range_error_mm": 3.0,
        "final_physical_tracking_error_mm": 0.3,
        "ground_truth_target_error_mm": 1.0,
        "maximum_target_step_mm": STEP_LIMIT_MM,
        "maximum_orientation_deviation_deg": ORIENTATION_LIMIT_DEG,
    }
    for name, limit in limits.items():
        value = measured[name]
        if value is None or value > limit:
            failed.append(name)

    result["failed_gates"] = sorted(set(failed))
    result["qualified"] = not result["failed_gates"]
    return result
```

- [ ] **Step 6: Implement 27/27 aggregation**

`aggregate_suite(results, execution_order)` must return:

```python
{
    "schema_version": 1,
    "seed": 20260722,
    "required_run_count": 27,
    "completed_run_count": len(results),
    "passed_run_count": passed,
    "failed_run_count": len(results) - passed,
    "QUALIFIED": len(results) == 27 and passed == 27,
    "execution_order": list(execution_order),
    "failure_counts_by_pose": pose_failures,
    "failure_counts_by_gate": gate_failures,
    "duration_s": {
        "minimum": min(durations),
        "median": median(durations),
        "p95": percentile_95,
        "maximum": max(durations),
    },
}
```

Add tests proving 26/27 fails, missing/non-finite values fail, wrong-port target error fails, and failure grouping is exact.

- [ ] **Step 7: Run pure tests and commit**

```bash
/usr/bin/python3 -m unittest -v tests.test_alignment_stress
git add single_rack_cv/stress_alignment.py \
        single_rack_cv/tests/test_alignment_stress.py
git commit -m "Add alignment stress domain model"
```

---

### Task 2: Passive runtime instrumentation

**Files:**
- Modify: `single_rack_cv/sim.py`
- Modify: `single_rack_cv/tests/test_alignment_stress.py`

**Interfaces:**
- Produces `SimulationRuntime.stress_snapshot() -> dict[str, object]` matching the child schema fields.
- Instrumentation observes the existing target and actual ToolCenter; it does not create a new control path.

- [ ] **Step 1: Add failing quaternion-distance tests**

```python
from stress_alignment import quaternion_angular_distance_deg

    def test_quaternion_distance_treats_sign_flip_as_zero(self):
        self.assertAlmostEqual(
            quaternion_angular_distance_deg(
                (1.0, 0.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0),
            ),
            0.0,
            places=12,
        )

    def test_quaternion_distance_reports_ten_degrees(self):
        half = math.radians(5.0)
        q = (math.cos(half), 0.0, math.sin(half), 0.0)
        self.assertAlmostEqual(
            quaternion_angular_distance_deg((1.0, 0.0, 0.0, 0.0), q),
            10.0,
            places=12,
        )
```

- [ ] **Step 2: Implement pure quaternion distance**

```python
def quaternion_angular_distance_deg(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
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
```

- [ ] **Step 3: Add passive fields to `VisualServoState`**

```python
    track_acquired_ever: bool = False
    visual_alignment_locked_ever: bool = False
    track_acquisition_count: int = 0
    perception_rejection_count: int = 0
    final_center_error_px: float = math.nan
    final_range_error_m: float = math.nan
    maximum_target_step_m: float = 0.0
    maximum_orientation_deviation_deg: float = 0.0
```

In `SimulationRuntime.__init__`:

```python
        self._initial_tool_orientation_wxyz: np.ndarray | None = None
```

- [ ] **Step 4: Instrument existing events**

In `note_perception_failure()`:

```python
        state.perception_rejection_count += 1
```

At the start of `observe_visual_servo()` after the IK guard:

```python
        state.final_center_error_px = float(np.linalg.norm(observation.center_error_px))
        state.final_range_error_m = float(observation.range_error_m)
```

After `compute_bounded_step()`:

```python
        state.maximum_target_step_m = max(
            state.maximum_target_step_m,
            float(np.linalg.norm(step_world_m)),
        )
```

When acquisition succeeds:

```python
        state.track_acquired_ever = True
        state.track_acquisition_count += 1
```

When alignment locks:

```python
                state.visual_alignment_locked_ever = True
```

In `_create_ik()` after `tool_orientation` is calculated:

```python
        self._initial_tool_orientation_wxyz = np.asarray(
            tool_orientation,
            dtype=np.float64,
        ).copy()
```

In `update_ik()` immediately after reading the target pose:

```python
        if self._initial_tool_orientation_wxyz is not None:
            deviation_deg = quaternion_angular_distance_deg(
                self._initial_tool_orientation_wxyz,
                desired_tool_orientation,
            )
            self.visual_servo.maximum_orientation_deviation_deg = max(
                self.visual_servo.maximum_orientation_deviation_deg,
                deviation_deg,
            )
```

- [ ] **Step 5: Add `stress_snapshot()`**

```python
    def stress_snapshot(self) -> dict[str, object]:
        if self.ik is None:
            raise RuntimeError("stress snapshot requires initialized IK")
        self._update_actual_tool_frame(self.ik)
        target_position, _ = self.ik.target.get_world_pose()
        actual_position, _ = self.ik.actual_tool.get_world_pose()
        physical_error_m = self._tool_target_position_error_m()
        state = self.visual_servo
        return {
            "completed": bool(state.complete),
            "track_acquired": bool(state.track_acquired_ever),
            "visual_alignment_locked": bool(state.visual_alignment_locked_ever),
            "final_center_error_px": float(state.final_center_error_px),
            "final_range_error_mm": 1000.0 * float(state.final_range_error_m),
            "final_tool_target_world_m": [float(v) for v in target_position],
            "final_actual_tool_world_m": [float(v) for v in actual_position],
            "final_physical_tracking_error_mm": 1000.0 * physical_error_m,
            "maximum_target_step_mm": 1000.0 * state.maximum_target_step_m,
            "maximum_orientation_deviation_deg": (
                state.maximum_orientation_deviation_deg
            ),
            "perception_rejection_count": state.perception_rejection_count,
            "track_reacquisition_count": max(0, state.track_acquisition_count - 1),
            "insertion_command_count": 0,
        }
```

- [ ] **Step 6: Run tests, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v tests.test_alignment_stress
"$HOME/isaacsim/python.sh" -m py_compile stress_alignment.py sim.py
git add single_rack_cv/stress_alignment.py \
        single_rack_cv/sim.py \
        single_rack_cv/tests/test_alignment_stress.py
git commit -m "Instrument alignment stress metrics"
```

---

### Task 3: Opt-in child lifecycle in `main.py`

**Files:**
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Consumes pure helpers from Task 1 and `runtime.stress_snapshot()` from Task 2.
- Writes only `child_result.json`; it does not know subprocess exit status, parent timeout, expected target, ground-truth error, failed gates, or final qualification.

- [ ] **Step 1: Add failing structural tests**

```python
    def test_stress_mode_is_optional(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("parse_stress_run_args", source)
        self.assertIn("stress_args is None", source)
        self.assertIn("derive_stress_config", source)
        self.assertIn("runtime.stress_snapshot()", source)
        self.assertIn("write_json_atomic", source)

    def test_child_runtime_has_no_ground_truth_access(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("front_plane_ground_truth", source)
        self.assertNotIn("expected_preinsert_target", source)
        self.assertNotIn("ground_truth_target_error", source)
```

- [ ] **Step 2: Parse stress mode before Isaac startup**

```python
import time
from stress_alignment import (
    derive_stress_config,
    new_child_result,
    parse_stress_run_args,
    utc_now_iso,
    write_json_atomic,
)

stress_args = parse_stress_run_args(sys.argv[1:])
RUNTIME_CONFIG = CONFIG if stress_args is None else derive_stress_config(CONFIG, stress_args)
```

Replace runtime uses of `CONFIG` with `RUNTIME_CONFIG`. Preserve the canonical import and default behavior.

Use a run-local tee path in stress mode:

```python
run_output_path = (
    RUNTIME_CONFIG.camera.output_dir / "run_output_latest.txt"
    if stress_args is None
    else stress_args.result_json.parent / "runtime_output.txt"
)
```

- [ ] **Step 3: Add internal timeout and auto-exit**

Before the outer `try`:

```python
started_at = utc_now_iso()
started_monotonic = time.monotonic()
internal_timed_out = False
fatal_error = ""
child_exit_status = 0
```

After `runtime.update_visual_servo_completion()` in the loop:

```python
        if stress_args is not None:
            elapsed_s = time.monotonic() - started_monotonic
            if elapsed_s >= stress_args.timeout_s:
                internal_timed_out = True
                child_exit_status = 2
                warn("Alignment stress run reached its internal 240 second timeout.")
                break
            if stress_args.exit_after_complete and runtime.visual_servo.complete:
                break
```

In the outer exception block:

```python
except Exception:
    fatal_error = traceback.format_exc()
    print(
        "\n[SINGLE RACK RGB STEREO SERVO] FATAL ERROR\n" + fatal_error,
        flush=True,
    )
    child_exit_status = 1
    if stress_args is None:
        raise
```

- [ ] **Step 4: Always write `child_result.json` in stress mode**

Before runtime shutdown in `finally`:

```python
        if stress_args is not None:
            child_payload = new_child_result(stress_args, started_at)
            if runtime is not None and runtime.ik is not None:
                child_payload.update(runtime.stress_snapshot())
            child_payload.update(
                {
                    "ended_at": utc_now_iso(),
                    "runtime_duration_s": time.monotonic() - started_monotonic,
                    "internal_timed_out": internal_timed_out,
                    "fatal_error": fatal_error,
                }
            )
            write_json_atomic(stress_args.result_json, child_payload)
```

After shutdown:

```python
if stress_args is not None:
    raise SystemExit(child_exit_status)
```

Do not add process-exit status to the child payload. The parent observes it after termination.

- [ ] **Step 5: Run structural tests, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v tests.test_runtime_wiring
"$HOME/isaacsim/python.sh" -m py_compile main.py sim.py stress_alignment.py
git add single_rack_cv/main.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Add isolated alignment stress child mode"
```

---

### Task 4: Parent runner, finalized results, and reports

**Files:**
- Create: `single_rack_cv/tools/run_alignment_stress.py`
- Create: `single_rack_cv/tests/test_alignment_stress_runner.py`

**Interfaces:**
- Validates existing `benchmarks/front_plane_ground_truth.json` before any child starts.
- Passes `runs/<case>/child_result.json` to the child.
- Writes finalized `runs/<case>/result.json` after the child exits.
- Does not regenerate missing or invalid ground truth; that is infrastructure failure exit 1 per the approved spec.

- [ ] **Step 1: Write failing command, truth-validation, and synthesis tests**

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from stress_alignment import StressCase

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "tools" / "run_alignment_stress.py"
    spec = importlib.util.spec_from_file_location("stress_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AlignmentStressRunnerTests(unittest.TestCase):
    def test_child_command_uses_only_approved_arguments(self):
        runner = load_runner()
        case = StressCase(-10, 10, 2)
        command = runner.build_child_command(
            Path("/home/aayush/isaacsim/python.sh"),
            ROOT,
            case,
            Path("/tmp/child_result.json"),
        )
        self.assertIn("--start-y-offset-mm", command)
        self.assertIn("--start-z-offset-mm", command)
        self.assertIn("--stress-run-id", command)
        self.assertIn(case.run_id, command)
        self.assertIn("--stress-result-json", command)
        self.assertIn("--stress-timeout-s", command)
        self.assertIn("--exit-after-complete", command)
        self.assertNotIn("--stress-repeat", command)

    def test_invalid_truth_aborts_instead_of_regenerating(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.load_valid_ground_truth(path)
```

- [ ] **Step 2: Implement imports, constants, and exact child command**

```python
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stress_alignment import (
    CHILD_TIMEOUT_S,
    PARENT_TIMEOUT_S,
    STRESS_SEED,
    StressCase,
    aggregate_suite,
    build_stress_cases,
    finalize_parent_result,
    write_json_atomic,
)

GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmarks" / "front_plane_ground_truth.json"
OUTPUT_ROOT = PROJECT_ROOT / "camera_output" / "alignment_stress"


def build_child_command(
    isaac_python: Path,
    project_root: Path,
    case: StressCase,
    child_result_json: Path,
) -> list[str]:
    return [
        str(isaac_python),
        str(project_root / "main.py"),
        "--start-y-offset-mm", str(case.y_offset_mm),
        "--start-z-offset-mm", str(case.z_offset_mm),
        "--stress-run-id", case.run_id,
        "--stress-result-json", str(child_result_json),
        "--stress-timeout-s", str(CHILD_TIMEOUT_S),
        "--exit-after-complete",
    ]
```

- [ ] **Step 3: Validate canonical ground truth without generating it**

```python
def load_valid_ground_truth(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"ground truth not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("camera_resolution_height_width") != [960, 1280]:
        raise ValueError("ground truth resolution is not 1280x960")
    if not str(payload.get("control_usage", "")).lower().startswith("forbidden"):
        raise ValueError("ground truth is not marked benchmark-only")
    center = payload.get("center_world_m")
    normal = payload.get("normal_world")
    if not isinstance(center, list) or len(center) != 3:
        raise ValueError("ground truth center_world_m is invalid")
    if not isinstance(normal, list) or len(normal) != 3:
        raise ValueError("ground truth normal_world is invalid")
    if not all(math.isfinite(float(value)) for value in center + normal):
        raise ValueError("ground truth center/normal contains non-finite values")
    if math.sqrt(sum(float(value) ** 2 for value in normal)) <= 1.0e-12:
        raise ValueError("ground truth normal is zero")
    return payload
```

- [ ] **Step 4: Implement one fresh child process**

```python
def run_one_case(
    isaac_python: Path,
    case: StressCase,
    run_directory: Path,
) -> tuple[int, bool, float]:
    run_directory.mkdir(parents=True, exist_ok=False)
    child_result_json = run_directory / "child_result.json"
    console_log = run_directory / "console.log"
    command = build_child_command(
        isaac_python,
        PROJECT_ROOT,
        case,
        child_result_json,
    )
    started = time.monotonic()
    hard_timed_out = False
    with console_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_status = process.wait(timeout=PARENT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            hard_timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_status = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_status = process.wait(timeout=10.0)
    return exit_status, hard_timed_out, time.monotonic() - started
```

- [ ] **Step 5: Load or synthesize the child payload**

`load_child_result(case, path)` must return `(payload, parse_status)`:

- valid JSON matching case: `parse_status="valid"`;
- missing file: a schema-shaped failed child payload and `parse_status="missing"`;
- malformed JSON: a schema-shaped failed child payload and `parse_status="malformed"`;
- run-ID/offset mismatch: a schema-shaped failed child payload and `parse_status="mismatch"`.

The synthetic payload must include `fatal_error` describing the exact problem and finite `runtime_duration_s=0.0`; finalization will still fail parse, completion, and process gates.

- [ ] **Step 6: Finalize each run only after the child exits**

```python
child_payload, parse_status = load_child_result(case, child_result_json)
result = finalize_parent_result(
    child_payload=child_payload,
    subprocess_exit_status=exit_status,
    parent_hard_timed_out=hard_timed_out,
    console_log_path=str(console_log.relative_to(suite_dir)),
    child_result_parse_status=parse_status,
    truth_center_world_m=truth["center_world_m"],
    truth_normal_world=truth["normal_world"],
    preinsert_standoff_m=0.050,
)
write_json_atomic(run_directory / "result.json", result)
```

The runner must never pass truth data or truth paths to `main.py`.

- [ ] **Step 7: Implement unique output directory and suite reports**

Use UTC timestamp with collision suffix and `exist_ok=False`.

`write_suite_outputs(suite_dir, results, execution_order)` must write:

- `summary.json` from `aggregate_suite`, plus worst center, absolute range, target error, physical error, target step, orientation deviation, rejection total, and reacquisition total;
- `summary.csv` with one row per finalized run;
- `report.txt` starting with:

```text
ALIGNMENT STRESS QUALIFICATION
passed_run_count=<n>
failed_run_count=<n>
QUALIFIED=<True|False>
```

CSV fields:

```python
CSV_FIELDS = [
    "run_id", "start_y_offset_mm", "start_z_offset_mm", "repeat",
    "runtime_duration_s", "subprocess_exit_status", "internal_timed_out",
    "parent_hard_timed_out", "completed", "track_acquired",
    "visual_alignment_locked", "final_center_error_px",
    "final_range_error_mm", "final_physical_tracking_error_mm",
    "ground_truth_target_error_mm", "maximum_target_step_mm",
    "maximum_orientation_deviation_deg", "perception_rejection_count",
    "track_reacquisition_count", "insertion_command_count",
    "child_result_parse_status", "qualified", "failed_gates", "fatal_error",
]
```

- [ ] **Step 8: Implement suite lifecycle and exit codes**

```python
def main() -> int:
    isaac_python = Path.home() / "isaacsim" / "python.sh"
    if not isaac_python.is_file():
        print(f"ERROR: Isaac launcher not found: {isaac_python}", file=sys.stderr)
        return 1
    try:
        truth = load_valid_ground_truth(GROUND_TRUTH_PATH)
        suite_dir = create_suite_directory()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    cases = build_stress_cases(STRESS_SEED)
    execution_order = [case.run_id for case in cases]
    results = []
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index:02d}/27] START {case.run_id}", flush=True)
            run_directory = suite_dir / "runs" / case.directory_name
            exit_status, hard_timeout, _ = run_one_case(
                isaac_python,
                case,
                run_directory,
            )
            child, parse_status = load_child_result(
                case,
                run_directory / "child_result.json",
            )
            result = finalize_parent_result(
                child_payload=child,
                subprocess_exit_status=exit_status,
                parent_hard_timed_out=hard_timeout,
                console_log_path=str(
                    (run_directory / "console.log").relative_to(suite_dir)
                ),
                child_result_parse_status=parse_status,
                truth_center_world_m=truth["center_world_m"],
                truth_normal_world=truth["normal_world"],
                preinsert_standoff_m=0.050,
            )
            write_json_atomic(run_directory / "result.json", result)
            results.append(result)
            write_suite_outputs(suite_dir, results, execution_order)
            print(
                f"[{index:02d}/27] "
                f"{'PASS' if result['qualified'] else 'FAIL'} {case.run_id}",
                flush=True,
            )
    except KeyboardInterrupt:
        write_suite_outputs(suite_dir, results, execution_order)
        return 1

    summary = write_suite_outputs(suite_dir, results, execution_order)
    print((suite_dir / "report.txt").read_text(encoding="utf-8"))
    return 0 if summary["QUALIFIED"] else 2
```

- [ ] **Step 9: Finish pure runner tests**

Cover exact command, missing/malformed/mismatched child result, 270-second timeout constant, invalid truth, normalized normal target derivation, unique directory allocation, 26/27 report failure, and CSV serialization.

- [ ] **Step 10: Run tests and commit**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_alignment_stress \
  tests.test_alignment_stress_runner
git add single_rack_cv/tools/run_alignment_stress.py \
        single_rack_cv/tests/test_alignment_stress_runner.py
git commit -m "Add alignment stress parent runner"
```

---

### Task 5: Shell launcher, structural guards, and README

**Files:**
- Create: `single_rack_cv/tools/run_alignment_stress.sh`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`
- Modify: `single_rack_cv/README.md`

- [ ] **Step 1: Add structural tests**

```python
    def test_parent_scores_truth_after_child(self):
        runner = (ROOT / "tools" / "run_alignment_stress.py").read_text(
            encoding="utf-8"
        )
        child_index = runner.index("run_one_case(")
        finalize_index = runner.index("finalize_parent_result(", child_index)
        self.assertLess(child_index, finalize_index)
        self.assertIn("front_plane_ground_truth.json", runner)

    def test_child_has_no_truth_and_no_insertion(self):
        main_source = (ROOT / "main.py").read_text(encoding="utf-8").lower()
        sim_source = (ROOT / "sim.py").read_text(encoding="utf-8")
        self.assertNotIn("front_plane_ground_truth", main_source)
        self.assertIn('"insertion_command_count": 0', sim_source)
        self.assertNotIn("insert_along", sim_source)

    def test_existing_step_limit_is_unchanged(self):
        config_source = (ROOT / "config.py").read_text(encoding="utf-8")
        self.assertIn("max_target_step_m: float = 0.001", config_source)
```

- [ ] **Step 2: Create sanitized launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

unset LD_LIBRARY_PATH
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset GZ_CONFIG_PATH
unset IGN_CONFIG_PATH
unset CONDA_PREFIX
unset VIRTUAL_ENV

printf '[ALIGNMENT STRESS] 3x3 world Y/Z grid, 3 repeats, 27 runs\n'
printf '[ALIGNMENT STRESS] child timeout=240s parent timeout=270s\n'
printf '[ALIGNMENT STRESS] qualification requires 27/27\n'

exec /usr/bin/python3 tools/run_alignment_stress.py
```

```bash
chmod +x tools/run_alignment_stress.sh
```

- [ ] **Step 3: Update README**

Add:

```markdown
## Alignment start-pose stress qualification

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
set -o pipefail
bash tools/run_alignment_stress.sh \
  2>&1 | tee camera_output/alignment_stress_console.txt
status=${PIPESTATUS[0]}
echo "alignment stress exit status: $status"
```

- `0`: all 27 runs qualified.
- `2`: suite completed with at least one failed run.
- `1`: invalid/missing ground truth, missing Isaac launcher, unwritable output, or interruption.

Each run contains `console.log`, `child_result.json`, and finalized `result.json`. Do not proceed to insertion unless `report.txt` shows `passed_run_count=27`, `failed_run_count=0`, and `QUALIFIED=True`.
```

Remove the obsolete README recovery-branch section because those temporary branches were deleted after the cleanup merge.

- [ ] **Step 4: Test and commit**

```bash
bash -n tools/run_alignment_stress.sh
/usr/bin/python3 -m unittest -v tests.test_runtime_wiring
git add single_rack_cv/tools/run_alignment_stress.sh \
        single_rack_cv/tests/test_runtime_wiring.py \
        single_rack_cv/README.md
git commit -m "Document alignment stress qualification"
```

---

### Task 6: Verification and workstation qualification

**Files:**
- No source files unless a test exposes a defect.
- Generated evidence remains ignored under `single_rack_cv/camera_output/`.

- [ ] **Step 1: Run complete tests**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" -m unittest -v \
  tests.test_front_plane \
  tests.test_live_control \
  tests.test_runtime_wiring \
  tests.test_benchmark \
  tests.test_ground_truth \
  tests.test_repo_cleanliness \
  tests.test_automatic_port_ground_truth \
  tests.test_alignment_stress \
  tests.test_alignment_stress_runner
```

Expected: zero failures and errors.

- [ ] **Step 2: Compile and check shell**

```bash
"$HOME/isaacsim/python.sh" -m py_compile \
  stress_alignment.py sim.py main.py tools/run_alignment_stress.py
bash -n tools/run_alignment_stress.sh
```

Expected: exit 0.

- [ ] **Step 3: Re-run frozen benchmark**

```bash
set -o pipefail
bash tools/run_benchmark.sh \
  2>&1 | tee camera_output/pre_stress_benchmark_console.txt
benchmark_status=${PIPESTATUS[0]}
echo "benchmark exit status: $benchmark_status"
cat camera_output/front_plane_benchmark/report.txt
```

Expected: status 0 and `QUALIFIED=True`. Do not tune estimator thresholds to recover regressions.

- [ ] **Step 4: Run one nominal child smoke test**

```bash
rm -rf camera_output/alignment_stress_smoke
mkdir -p camera_output/alignment_stress_smoke
set -o pipefail
"$HOME/isaacsim/python.sh" main.py \
  --start-y-offset-mm 0 \
  --start-z-offset-mm 0 \
  --stress-run-id y+00_z+00_r1 \
  --stress-result-json camera_output/alignment_stress_smoke/child_result.json \
  --stress-timeout-s 240 \
  --exit-after-complete \
  2>&1 | tee camera_output/alignment_stress_smoke/console.log
status=${PIPESTATUS[0]}
echo "smoke exit status: $status"
cat camera_output/alignment_stress_smoke/child_result.json
```

Expected:

- exits automatically with status 0;
- `completed=true`;
- `track_acquired=true`;
- `visual_alignment_locked=true`;
- target and actual ToolCenter world positions are present;
- no parent fields or ground-truth fields are present;
- max step `<=1.000001 mm`;
- orientation deviation `<=0.572958°`;
- insertion count 0.

Kill switch: do not launch 27 runs if this child lifecycle fails.

- [ ] **Step 5: Run full suite**

```bash
set -o pipefail
bash tools/run_alignment_stress.sh \
  2>&1 | tee camera_output/alignment_stress_console.txt
stress_status=${PIPESTATUS[0]}
echo "alignment stress exit status: $stress_status"
latest_dir=$(find camera_output/alignment_stress -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
echo "latest suite: $latest_dir"
cat "$latest_dir/report.txt"
```

Required result:

```text
passed_run_count=27
failed_run_count=0
QUALIFIED=True
```

Also verify seed `20260722`, 27 unique run IDs, every pose exactly three times, no insertion, max step `<=1.000001 mm`, orientation `<=0.572958°`, target error `<=1 mm`, physical error `<=0.3 mm`, center error `<=2 px`, and absolute range error `<=3 mm`.

- [ ] **Step 6: Inspect failures without weakening gates**

```bash
/usr/bin/python3 - "$latest_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for path in sorted((root / "runs").glob("*/result.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("qualified"):
        print(path.parent.name, payload.get("failed_gates"), payload.get("fatal_error"))
PY
```

Inspect each sibling `console.log`. Fix perception, reachability, process lifecycle, or instrumentation. Do not remove poses or enlarge tolerances.

- [ ] **Step 7: Verify repository cleanliness**

```bash
git status --short
git diff --check
git ls-files single_rack_cv/camera_output
```

Expected: no generated output tracked and no whitespace errors.

- [ ] **Step 8: Final evidence checkpoint**

Before merge, record test pass count, benchmark metrics, smoke result, full suite directory, exact 27/27 status, and confirmation that no insertion path was added.
