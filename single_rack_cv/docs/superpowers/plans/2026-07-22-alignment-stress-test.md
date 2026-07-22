# Alignment Stress-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 27-run qualification harness that starts a fresh Isaac Sim process for every world-frame Y/Z start offset and fails unless all 27 runs satisfy the approved safety and accuracy gates.

**Architecture:** `stress_alignment.py` is pure Python and owns the matrix, run-ID parsing, config derivation, strict JSON schemas, expected-target calculation, gates, and aggregation. `sim.py` passively records existing controller behavior. `main.py` gains an opt-in child mode that never reads ground truth. `tools/run_alignment_stress.py` validates canonical benchmark truth before launching children, finalizes each result only after the child exits, and writes suite reports.

**Tech Stack:** Python 3.12, standard library, NumPy, Isaac Sim 6.0.0, Lula IK, existing YOLOE/SGBM controller, `unittest`, Bash.

## Global Constraints

- Ubuntu 24.04, Isaac Sim 6.0.0, camera resolution `(960, 1280)`.
- Runtime remains image-only; the child must never read RTX/USD ground truth.
- Parent ground truth is scoring-only and is never passed to the child.
- X and wrist orientation remain unchanged; vision never commands orientation.
- Existing target-step limit remains 1 mm. Qualification permits `1e-9 m = 0.000001 mm`, so `maximum_target_step_mm <= 1.000001`.
- Orientation deviation is quaternion angular distance in degrees and must be `<=0.572958°`.
- Failed observations hold the target and use existing reacquisition behavior.
- No insertion command is added.
- Default `main.py` behavior is unchanged without stress arguments.
- Matrix: world Y/Z `(-10, 0, +10)` mm, 3 repeats, seed `20260722`, 27 runs.
- Child timeout: 240 seconds. Parent hard timeout: 270 seconds.
- Expected target: ground-truth opening center plus normalized outward front-plane normal times the existing 0.050 m standoff.
- Suite passes only at 27/27.

---

## File Map

- Create `single_rack_cv/stress_alignment.py`.
- Create `single_rack_cv/tests/test_alignment_stress.py`.
- Modify `single_rack_cv/sim.py`.
- Modify `single_rack_cv/main.py`.
- Create `single_rack_cv/tools/run_alignment_stress.py`.
- Create `single_rack_cv/tests/test_alignment_stress_runner.py`.
- Create `single_rack_cv/tools/run_alignment_stress.sh`.
- Modify `single_rack_cv/tests/test_runtime_wiring.py`.
- Modify `single_rack_cv/README.md`.

---

### Task 1: Pure matrix, parsing, schemas, scoring, and aggregation

**Files:**
- Create: `single_rack_cv/stress_alignment.py`
- Create: `single_rack_cv/tests/test_alignment_stress.py`

**Interfaces:**

```python
StressCase(y_offset_mm: int, z_offset_mm: int, repeat: int)
StressCase.from_run_id(run_id: str) -> StressCase
build_stress_cases(seed: int = 20260722) -> list[StressCase]
parse_stress_run_args(argv: Sequence[str]) -> StressRunArgs | None
derive_stress_config(base_config, args: StressRunArgs)
new_child_result(args: StressRunArgs, started_at: str) -> dict[str, object]
expected_preinsert_target_world_m(center, normal, standoff_m) -> np.ndarray
finalize_parent_result(...) -> dict[str, object]
aggregate_suite(results, execution_order) -> dict[str, object]
write_json_atomic(path, payload) -> None
```

- [ ] **Step 1: Write failing matrix, parsing, and config tests**

```python
from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

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
        self.assertEqual(len({case.directory_name for case in cases}), 27)

    def test_seed_is_reproducible(self):
        first = [case.run_id for case in build_stress_cases(STRESS_SEED)]
        second = [case.run_id for case in build_stress_cases(STRESS_SEED)]
        other = [case.run_id for case in build_stress_cases(STRESS_SEED + 1)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_run_id_round_trip(self):
        case = StressCase(-10, 10, 2)
        self.assertEqual(case.run_id, "y-10_z+10_r2")
        self.assertEqual(StressCase.from_run_id(case.run_id), case)

    def test_cli_uses_approved_arguments_only(self):
        args = parse_stress_run_args([
            "--start-y-offset-mm", "-10",
            "--start-z-offset-mm", "10",
            "--stress-run-id", "y-10_z+10_r2",
            "--stress-result-json", "/tmp/child_result.json",
            "--stress-timeout-s", "240",
            "--exit-after-complete",
        ])
        self.assertEqual(args.case, StressCase(-10, 10, 2))
        self.assertEqual(args.result_json, Path("/tmp/child_result.json"))

    def test_lone_stress_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_stress_run_args(["--exit-after-complete"])

    def test_offsets_change_only_world_y_and_z(self):
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

- [ ] **Step 2: Run and verify failure**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
/usr/bin/python3 -m unittest -v tests.test_alignment_stress
```

Expected: import failure because `stress_alignment.py` does not exist.

- [ ] **Step 3: Implement the matrix and approved CLI**

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
    stress_requested = namespace.exit_after_complete or any(
        value is not None for value in supplied
    )
    if not stress_requested:
        return None
    if any(value is None for value in supplied):
        raise ValueError("all stress-run arguments must be supplied together")
    case = StressCase.from_run_id(namespace.stress_run_id)
    if (case.y_offset_mm, case.z_offset_mm) != (
        namespace.start_y_offset_mm,
        namespace.start_z_offset_mm,
    ):
        raise ValueError("run ID and start offsets disagree")
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
```

The configured fixed startup pose is the panda-hand pose. Because wrist orientation and the hand-to-tool transform remain fixed, a world Y/Z shift of that pose produces the identical world Y/Z shift at ToolCenter.

- [ ] **Step 4: Write failing schema, strict-JSON, target, and gate tests**

```python
from stress_alignment import (
    expected_preinsert_target_world_m,
    finalize_parent_result,
    write_json_atomic,
)

    def passing_child(self):
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

    def test_expected_target_normalizes_normal(self):
        target = expected_preinsert_target_world_m(
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 0.0],
            0.05,
        )
        self.assertTrue(np.allclose(target, [1.05, 2.0, 3.0]))

    def test_exact_boundaries_pass(self):
        result = finalize_parent_result(
            child_payload=self.passing_child(),
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

    def test_nonfinite_is_written_as_null_and_fails_gate(self):
        child = self.passing_child()
        child["final_center_error_px"] = math.nan
        with self.subTest("strict JSON"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "payload.json"
                write_json_atomic(path, child)
                self.assertIn('"final_center_error_px": null', path.read_text())
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

Add one-over-limit tests for 2 px, 3 mm, 0.3 mm, 1 mm target error, 1.000001 mm step, and 0.572958° orientation.

- [ ] **Step 5: Implement strict JSON and child schema**

```python
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def new_child_result(args: StressRunArgs, started_at: str) -> dict[str, object]:
    return {
        "schema_version": STRESS_SCHEMA_VERSION,
        "run_id": args.case.run_id,
        "start_y_offset_mm": args.case.y_offset_mm,
        "start_z_offset_mm": args.case.z_offset_mm,
        "repeat": args.case.repeat,
        "started_at": started_at,
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
```

- [ ] **Step 6: Implement expected target and parent finalization**

```python
def expected_preinsert_target_world_m(center, normal, standoff_m) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float64).reshape(3)
    normal_array = np.asarray(normal, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center_array)) or not np.all(np.isfinite(normal_array)):
        raise ValueError("ground-truth center and normal must be finite")
    norm = float(np.linalg.norm(normal_array))
    if norm <= 1.0e-12:
        raise ValueError("ground-truth normal must be nonzero")
    if not math.isfinite(standoff_m) or standoff_m <= 0.0:
        raise ValueError("standoff must be finite and positive")
    return center_array + normal_array / norm * float(standoff_m)


def _finite(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None
```

`finalize_parent_result()` must preserve child fields and add:

```python
{
    "subprocess_exit_status": int(subprocess_exit_status),
    "parent_hard_timed_out": bool(parent_hard_timed_out),
    "console_log_path": console_log_path,
    "child_result_parse_status": child_result_parse_status,
    "expected_preinsert_target_world_m": expected_target.tolist(),
    "ground_truth_target_error_mm": target_error_mm,
    "failed_gates": sorted(set(failed)),
    "qualified": not failed,
}
```

Boolean failures: nonzero process status, internal timeout, parent timeout, incomplete run, no track, no alignment lock, parse status not `valid`, fatal error text, or insertion count not zero.

Numeric failures:

```python
{
    "runtime_duration_s": 240.0,
    "final_center_error_px": 2.0,
    "absolute_final_range_error_mm": 3.0,
    "final_physical_tracking_error_mm": 0.3,
    "ground_truth_target_error_mm": 1.0,
    "maximum_target_step_mm": 1.000001,
    "maximum_orientation_deviation_deg": 0.572958,
}
```

Missing or non-finite values fail.

- [ ] **Step 7: Implement aggregation and test 27/27**

`aggregate_suite()` must include seed, required/completed/passed/failed counts, exact execution order, failures by pose/gate, duration min/median/p95/max, and:

```python
"QUALIFIED": len(results) == 27 and passed_run_count == 27
```

Tests must prove 26/27 fails and failed gates cannot be hidden by averages.

- [ ] **Step 8: Run and commit**

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

- [ ] **Step 1: Add failing quaternion tests**

```python
from stress_alignment import quaternion_angular_distance_deg

    def test_quaternion_sign_flip_is_zero(self):
        self.assertAlmostEqual(
            quaternion_angular_distance_deg(
                (1.0, 0.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0),
            ),
            0.0,
            places=12,
        )

    def test_known_ten_degree_rotation(self):
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
def quaternion_angular_distance_deg(first, second) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(4)
    b = np.asarray(second, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("quaternions must be finite")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1.0e-12 or b_norm <= 1.0e-12:
        raise ValueError("quaternions must be nonzero")
    cosine = abs(float(np.dot(a / a_norm, b / b_norm)))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, cosine))))
```

- [ ] **Step 3: Add passive `VisualServoState` fields**

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

- [ ] **Step 4: Instrument existing events without adding commands**

- Increment `perception_rejection_count` at the start of `note_perception_failure()`.
- Record center/range errors at the start of `observe_visual_servo()` after the IK guard.
- Update `maximum_target_step_m` immediately after `compute_bounded_step()`.
- Set `track_acquired_ever=True` and increment `track_acquisition_count` when acquisition succeeds.
- Set `visual_alignment_locked_ever=True` when alignment locks.
- Capture initial ToolCenter orientation in `_create_ik()`.
- Measure target-orientation deviation in `update_ik()` with `quaternion_angular_distance_deg()`.

Exact measurement statements:

```python
state.final_center_error_px = float(np.linalg.norm(observation.center_error_px))
state.final_range_error_m = float(observation.range_error_m)
state.maximum_target_step_m = max(
    state.maximum_target_step_m,
    float(np.linalg.norm(step_world_m)),
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
        state = self.visual_servo
        return {
            "completed": bool(state.complete),
            "track_acquired": bool(state.track_acquired_ever),
            "visual_alignment_locked": bool(state.visual_alignment_locked_ever),
            "final_center_error_px": _json_number_or_none(state.final_center_error_px),
            "final_range_error_mm": _json_number_or_none(
                1000.0 * state.final_range_error_m
            ),
            "final_tool_target_world_m": [float(v) for v in target_position],
            "final_actual_tool_world_m": [float(v) for v in actual_position],
            "final_physical_tracking_error_mm": _json_number_or_none(
                1000.0 * self._tool_target_position_error_m()
            ),
            "maximum_target_step_mm": 1000.0 * state.maximum_target_step_m,
            "maximum_orientation_deviation_deg": (
                state.maximum_orientation_deviation_deg
            ),
            "perception_rejection_count": state.perception_rejection_count,
            "track_reacquisition_count": max(0, state.track_acquisition_count - 1),
            "insertion_command_count": 0,
        }
```

Define `_json_number_or_none(value)` in `sim.py` as `float(value)` when finite, otherwise `None`.

- [ ] **Step 6: Test, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v tests.test_alignment_stress
"$HOME/isaacsim/python.sh" -m py_compile stress_alignment.py sim.py
git add single_rack_cv/stress_alignment.py \
        single_rack_cv/sim.py \
        single_rack_cv/tests/test_alignment_stress.py
git commit -m "Instrument alignment stress metrics"
```

---

### Task 3: Opt-in child stress mode

**Files:**
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

- [ ] **Step 1: Add failing structural tests**

```python
    def test_stress_mode_is_optional(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("parse_stress_run_args", source)
        self.assertIn("stress_args is None", source)
        self.assertIn("derive_stress_config", source)
        self.assertIn("runtime.stress_snapshot()", source)
        self.assertIn("write_json_atomic", source)

    def test_child_has_no_ground_truth(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("front_plane_ground_truth", source)
        self.assertNotIn("expected_preinsert_target", source)
        self.assertNotIn("ground_truth_target_error", source)
```

- [ ] **Step 2: Parse and derive config before Isaac startup**

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
run_output_path = (
    RUNTIME_CONFIG.camera.output_dir / "run_output_latest.txt"
    if stress_args is None
    else stress_args.result_json.parent / "runtime_output.txt"
)
```

Replace runtime uses of `CONFIG` with `RUNTIME_CONFIG` while preserving the canonical `CONFIG` import.

- [ ] **Step 3: Add child timeout and auto-exit**

```python
started_at = utc_now_iso()
started_monotonic = time.monotonic()
internal_timed_out = False
fatal_error = ""
child_exit_status = 0
```

After `runtime.update_visual_servo_completion()`:

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

Stress exceptions record traceback and set status 1; default mode still re-raises.

- [ ] **Step 4: Always write `child_result.json` before shutdown**

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

After the existing shutdown block:

```python
if stress_args is not None:
    raise SystemExit(child_exit_status)
```

Do not write subprocess exit status, parent timeout, ground truth, failed gates, or qualification in the child.

- [ ] **Step 5: Test, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v tests.test_runtime_wiring
"$HOME/isaacsim/python.sh" -m py_compile main.py sim.py stress_alignment.py
git add single_rack_cv/main.py single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Add isolated alignment stress child mode"
```

---

### Task 4: Parent runner and finalized suite reports

**Files:**
- Create: `single_rack_cv/tools/run_alignment_stress.py`
- Create: `single_rack_cv/tests/test_alignment_stress_runner.py`

- [ ] **Step 1: Write failing command and ground-truth tests**

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
        self.assertIn("--stress-run-id", command)
        self.assertIn(case.run_id, command)
        self.assertIn("--stress-result-json", command)
        self.assertIn("--exit-after-complete", command)
        self.assertNotIn("--stress-repeat", command)

    def test_invalid_truth_is_infrastructure_failure(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.load_valid_ground_truth(path)
```

- [ ] **Step 2: Implement exact command and strict scene-specific truth validation**

```python
def build_child_command(isaac_python, project_root, case, child_result_json):
    return [
        str(isaac_python), str(project_root / "main.py"),
        "--start-y-offset-mm", str(case.y_offset_mm),
        "--start-z-offset-mm", str(case.z_offset_mm),
        "--stress-run-id", case.run_id,
        "--stress-result-json", str(child_result_json),
        "--stress-timeout-s", "240.0",
        "--exit-after-complete",
    ]
```

`load_valid_ground_truth(path)` must require:

```python
payload["camera_resolution_height_width"] == [960, 1280]
payload["source"] == "automatic_rtx_mesh_raycast_front_bezel_plane"
str(payload["control_usage"]).lower().startswith("forbidden")
len(payload["center_world_m"]) == 3
len(payload["normal_world"]) == 3
payload["used_prim_paths"] is a nonempty list
all(path.startswith(CONFIG.scene.rack_path) for path in payload["used_prim_paths"])
```

Center and normal must be finite, and normal magnitude must be nonzero. Missing or invalid truth returns suite exit 1. The runner does not regenerate it.

- [ ] **Step 3: Implement one fresh process with timeout and Ctrl-C cleanup**

```python
def run_one_case(isaac_python, case, run_directory):
    run_directory.mkdir(parents=True, exist_ok=False)
    child_result = run_directory / "child_result.json"
    console_log = run_directory / "console.log"
    command = build_child_command(isaac_python, PROJECT_ROOT, case, child_result)
    started = time.monotonic()
    hard_timeout = False
    with console_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_status = process.wait(timeout=270.0)
        except subprocess.TimeoutExpired:
            hard_timeout = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_status = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_status = process.wait(timeout=10.0)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10.0)
            raise
    return exit_status, hard_timeout, time.monotonic() - started
```

- [ ] **Step 4: Implement child parsing and finalized result**

`load_child_result(case, path)` returns `(payload, parse_status)` where status is `valid`, `missing`, `malformed`, or `mismatch`. Missing/malformed/mismatched data is synthesized into the full child schema with `fatal_error` and `runtime_duration_s=0.0`.

After the process exits:

```python
child, parse_status = load_child_result(case, child_result_path)
result = finalize_parent_result(
    child_payload=child,
    subprocess_exit_status=exit_status,
    parent_hard_timed_out=hard_timeout,
    console_log_path=str(console_log.relative_to(suite_dir)),
    child_result_parse_status=parse_status,
    truth_center_world_m=truth["center_world_m"],
    truth_normal_world=truth["normal_world"],
    preinsert_standoff_m=0.050,
)
write_json_atomic(run_directory / "result.json", result)
```

Truth data is never passed to `main.py`.

- [ ] **Step 5: Implement unique suite directory**

```python
def create_suite_directory() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(100):
        name = stamp if suffix == 0 else f"{stamp}-{suffix:02d}"
        candidate = OUTPUT_ROOT / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a unique suite directory")
```

- [ ] **Step 6: Implement reports and progress**

`write_suite_outputs()` writes after every completed run so interruption leaves usable partial evidence:

- `summary.json`: aggregate plus worst center, absolute range, target error, physical error, step, orientation, rejection total, reacquisition total.
- `summary.csv`: one finalized run per row.
- `report.txt` starts exactly:

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

- [ ] **Step 7: Implement suite `main()` and exit codes**

Before launching children, validate Isaac launcher, truth file, and output directory. Iterate `build_stress_cases(20260722)` sequentially. Print `[01/27] START ...` and PASS/FAIL. Return 0 only for 27/27, 2 for completed qualification failure, and 1 for infrastructure failure or interruption.

- [ ] **Step 8: Complete runner tests**

Tests must cover command construction, 270-second constant, missing/malformed/mismatched child result, scene-specific truth validation, normalized normal target, unique directory allocation, CSV serialization, 26/27 failure, and process-group termination by mocking `Popen`, `os.killpg`, and timeout exceptions.

- [ ] **Step 9: Run and commit**

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

    def test_child_has_no_truth_or_insertion(self):
        main_source = (ROOT / "main.py").read_text(encoding="utf-8").lower()
        sim_source = (ROOT / "sim.py").read_text(encoding="utf-8")
        self.assertNotIn("front_plane_ground_truth", main_source)
        self.assertIn('"insertion_command_count": 0', sim_source)
        self.assertNotIn("insert_along", sim_source)

    def test_existing_step_limit_is_unchanged(self):
        config_source = (ROOT / "config.py").read_text(encoding="utf-8")
        self.assertIn("max_target_step_m: float = 0.001", config_source)
```

- [ ] **Step 2: Create launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
unset LD_LIBRARY_PATH PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
unset GZ_CONFIG_PATH IGN_CONFIG_PATH CONDA_PREFIX VIRTUAL_ENV
printf '[ALIGNMENT STRESS] 3x3 world Y/Z grid, 3 repeats, 27 runs\n'
printf '[ALIGNMENT STRESS] child timeout=240s parent timeout=270s\n'
printf '[ALIGNMENT STRESS] qualification requires 27/27\n'
exec /usr/bin/python3 tools/run_alignment_stress.py
```

```bash
chmod +x tools/run_alignment_stress.sh
```

- [ ] **Step 3: Update README**

Add the exact command:

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
set -o pipefail
bash tools/run_alignment_stress.sh \
  2>&1 | tee camera_output/alignment_stress_console.txt
status=${PIPESTATUS[0]}
echo "alignment stress exit status: $status"
```

Document exit codes 0/2/1, output layout containing `console.log`, `child_result.json`, and `result.json`, and the kill switch: do not start insertion unless `passed_run_count=27`, `failed_run_count=0`, and `QUALIFIED=True`.

Remove the obsolete README recovery-branch section because those branches were intentionally deleted.

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

- [ ] **Step 1: Run all tests**

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

Expected: status 0 and `QUALIFIED=True`. Do not tune thresholds to hide regression.

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

Expected: automatic status 0, completed/track/alignment true, target and actual poses present, no parent or truth fields, max step `<=1.000001`, orientation `<=0.572958`, insertion count 0.

Kill switch: do not launch 27 runs if smoke lifecycle fails.

- [ ] **Step 5: Run full suite**

```bash
set -o pipefail
bash tools/run_alignment_stress.sh \
  2>&1 | tee camera_output/alignment_stress_console.txt
stress_status=${PIPESTATUS[0]}
echo "alignment stress exit status: $stress_status"
latest_dir=$(find camera_output/alignment_stress -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
cat "$latest_dir/report.txt"
```

Required:

```text
passed_run_count=27
failed_run_count=0
QUALIFIED=True
```

Verify seed, 27 unique IDs, every pose three times, no insertion, step/orientation/target/physical/center/range limits.

- [ ] **Step 6: Inspect failures without changing gates**

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

Inspect each sibling `console.log`. Fix the actual defect. Do not remove poses or widen tolerances.

- [ ] **Step 7: Verify repository cleanliness**

```bash
git status --short
git diff --check
git ls-files single_rack_cv/camera_output
```

Expected: generated output untracked and no whitespace errors.

- [ ] **Step 8: Final evidence checkpoint**

Before merge, record test pass count, frozen benchmark metrics, nominal smoke result, full suite path, exact 27/27 result, and confirmation that no insertion path was added.
