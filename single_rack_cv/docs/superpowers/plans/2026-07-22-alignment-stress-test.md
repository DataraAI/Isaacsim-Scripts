# Alignment Stress-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 27-run qualification harness that launches a fresh Isaac Sim process for each world-frame Y/Z start offset, preserves the existing image-only controller, and fails unless all 27 runs meet the locked safety and accuracy gates.

**Architecture:** A pure-Python `stress_alignment.py` module owns the matrix, schemas, config derivation, gate evaluation, and aggregation. `sim.py` exposes passive runtime instrumentation, while `main.py` gains an opt-in stress mode that writes one child result and exits after completion or 240 seconds. A parent runner launches 27 sequential Isaac subprocesses, scores each child result against benchmark-only world-space ground truth after the child exits, and writes suite reports.

**Tech Stack:** Python 3.12 standard library, NumPy, Isaac Sim 6.0.0, Lula IK, existing YOLOE/SGBM pipeline, `unittest`, Bash.

## Global Constraints

- Platform remains Ubuntu 24.04 with Isaac Sim 6.0.0.
- Canonical camera resolution remains `(960, 1280)` height-width.
- Runtime control remains image-only; RTX/USD ground truth is forbidden inside the live control loop.
- Ground truth may be read only by the parent process after a child run exits.
- Wrist orientation remains fixed; vision never commands orientation.
- Every ToolCenter target update remains bounded to 1.0 mm plus `1e-9` mm numerical epsilon.
- Failed observations hold the current target and trigger reacquisition.
- No insertion command is added.
- Normal `main.py` behavior remains unchanged when stress arguments are absent.
- Stress matrix is world Y/Z offsets `(-10, 0, +10)` mm with X unchanged and three repeats per pose.
- Execution order uses seed `20260722` and is recorded.
- Child runtime timeout is 240 seconds; parent hard timeout is 270 seconds.
- Qualification requires 27/27 passing runs.
- Do not weaken gates, remove failed poses, or average away failures.

---

## File Structure

- Create `single_rack_cv/stress_alignment.py`: pure matrix, CLI option model, frozen-config derivation, result schema, gate evaluation, aggregation, and atomic JSON writing.
- Create `single_rack_cv/tests/test_alignment_stress.py`: pure tests for matrix, config offsets, serialization, gate boundaries, aggregation, and numerical validation.
- Modify `single_rack_cv/sim.py`: passive instrumentation and one public `stress_snapshot()` method; no second control path.
- Modify `single_rack_cv/main.py`: optional stress arguments, child timeout, auto-exit, and child result emission.
- Create `single_rack_cv/tools/run_alignment_stress.py`: sequential parent orchestrator, ground-truth validation/generation, subprocess handling, scoring, and reports.
- Create `single_rack_cv/tests/test_alignment_stress_runner.py`: pure tests for child command construction, timeout synthesis, malformed/missing JSON handling, and output formatting.
- Create `single_rack_cv/tools/run_alignment_stress.sh`: sanitized one-command entry point.
- Modify `single_rack_cv/tests/test_runtime_wiring.py`: structural guardrails proving default runtime preservation, post-run-only truth, fixed orientation, 1 mm step, and no insertion path.
- Modify `single_rack_cv/README.md`: exact stress-test command, exit codes, output layout, and qualification rule.

---

### Task 1: Pure stress-test domain model and gates

**Files:**
- Create: `single_rack_cv/stress_alignment.py`
- Create: `single_rack_cv/tests/test_alignment_stress.py`

**Interfaces:**
- Consumes: `config.CONFIG`, whose `ik.initial_position` is the existing fixed panda-hand startup pose. Because orientation is fixed and the hand-to-tool offset is along the fixed tool transform, applying world Y/Z deltas to the hand startup pose produces the same world Y/Z deltas at ToolCenter.
- Produces:
  - `StressCase(y_offset_mm: int, z_offset_mm: int, repeat: int)`
  - `StressRunArgs(case: StressCase, result_json: Path, timeout_s: float, exit_after_complete: bool)`
  - `build_stress_cases(seed: int = 20260722) -> list[StressCase]`
  - `parse_stress_run_args(argv: Sequence[str]) -> StressRunArgs | None`
  - `derive_stress_config(base_config: Config, args: StressRunArgs) -> Config`
  - `new_child_result(args: StressRunArgs, started_at: str) -> dict[str, object]`
  - `evaluate_parent_result(payload: dict[str, object], truth_center_world_m: Sequence[float]) -> dict[str, object]`
  - `aggregate_suite(results: Sequence[dict[str, object]], execution_order: Sequence[str]) -> dict[str, object]`
  - `write_json_atomic(path: Path, payload: Mapping[str, object]) -> None`

- [ ] **Step 1: Write failing matrix and config-derivation tests**

Add tests that require the exact 27-case set, reproducible shuffle, unique IDs, and unchanged X/orientation:

```python
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

from config import CONFIG
from stress_alignment import (
    STRESS_SEED,
    StressCase,
    StressRunArgs,
    aggregate_suite,
    build_stress_cases,
    derive_stress_config,
    evaluate_parent_result,
    new_child_result,
    write_json_atomic,
)


class AlignmentStressTests(unittest.TestCase):
    def test_matrix_is_exact_three_by_three_with_three_repeats(self):
        cases = build_stress_cases()
        self.assertEqual(len(cases), 27)
        counts = {}
        for case in cases:
            counts[(case.y_offset_mm, case.z_offset_mm)] = (
                counts.get((case.y_offset_mm, case.z_offset_mm), 0) + 1
            )
        self.assertEqual(
            counts,
            {(y, z): 3 for y in (-10, 0, 10) for z in (-10, 0, 10)},
        )
        self.assertEqual(len({case.run_id for case in cases}), 27)
        self.assertEqual(len({case.directory_name for case in cases}), 27)

    def test_shuffle_is_reproducible_and_seeded(self):
        first = [case.run_id for case in build_stress_cases(STRESS_SEED)]
        second = [case.run_id for case in build_stress_cases(STRESS_SEED)]
        other = [case.run_id for case in build_stress_cases(STRESS_SEED + 1)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_offsets_change_only_world_y_and_z(self):
        case = StressCase(y_offset_mm=10, z_offset_mm=-10, repeat=1)
        args = StressRunArgs(
            case=case,
            result_json=Path("/tmp/result.json"),
            timeout_s=240.0,
            exit_after_complete=True,
        )
        derived = derive_stress_config(CONFIG, args)
        base_position = CONFIG.ik.initial_position
        self.assertAlmostEqual(derived.ik.initial_position[0], base_position[0])
        self.assertAlmostEqual(derived.ik.initial_position[1], base_position[1] + 0.010)
        self.assertAlmostEqual(derived.ik.initial_position[2], base_position[2] - 0.010)
        self.assertEqual(
            derived.ik.initial_orientation_wxyz,
            CONFIG.ik.initial_orientation_wxyz,
        )
        self.assertEqual(CONFIG.ik.initial_position, base_position)
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
/usr/bin/python3 -m unittest -v tests.test_alignment_stress
```

Expected: import failure because `stress_alignment.py` does not exist.

- [ ] **Step 3: Implement the exact matrix, argument model, and config derivation**

Create `stress_alignment.py` with these definitions:

```python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
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
STEP_EPSILON_MM = 1.0e-9


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


@dataclass(frozen=True)
class StressRunArgs:
    case: StressCase
    result_json: Path
    timeout_s: float = CHILD_TIMEOUT_S
    exit_after_complete: bool = True


def build_stress_cases(seed: int = STRESS_SEED) -> list[StressCase]:
    cases = [
        StressCase(y_offset_mm=y, z_offset_mm=z, repeat=repeat)
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
    parser.add_argument("--stress-repeat", type=int)
    parser.add_argument("--stress-run-id")
    parser.add_argument("--stress-result-json", type=Path)
    parser.add_argument("--stress-timeout-s", type=float, default=CHILD_TIMEOUT_S)
    parser.add_argument("--exit-after-complete", action="store_true")
    namespace, _ = parser.parse_known_args(list(argv))
    supplied = [
        namespace.start_y_offset_mm,
        namespace.start_z_offset_mm,
        namespace.stress_repeat,
        namespace.stress_run_id,
        namespace.stress_result_json,
    ]
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise ValueError("All stress-run arguments must be supplied together.")
    if namespace.start_y_offset_mm not in Y_OFFSETS_MM:
        raise ValueError("start Y offset must be one of -10, 0, +10 mm.")
    if namespace.start_z_offset_mm not in Z_OFFSETS_MM:
        raise ValueError("start Z offset must be one of -10, 0, +10 mm.")
    if namespace.stress_repeat not in range(1, REPEATS_PER_POSE + 1):
        raise ValueError("stress repeat must be 1, 2, or 3.")
    if not math.isfinite(namespace.stress_timeout_s) or namespace.stress_timeout_s <= 0:
        raise ValueError("stress timeout must be finite and positive.")
    case = StressCase(
        y_offset_mm=namespace.start_y_offset_mm,
        z_offset_mm=namespace.start_z_offset_mm,
        repeat=namespace.stress_repeat,
    )
    if namespace.stress_run_id != case.run_id:
        raise ValueError(
            f"stress run ID {namespace.stress_run_id!r} does not match {case.run_id!r}."
        )
    return StressRunArgs(
        case=case,
        result_json=namespace.stress_result_json,
        timeout_s=float(namespace.stress_timeout_s),
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

- [ ] **Step 4: Write failing result-gate and aggregation tests**

Add one valid fixture and boundary tests:

```python
    def _passing_payload(self):
        return {
            "schema_version": 1,
            "run_id": "y+00_z+00_r1",
            "start_y_offset_mm": 0,
            "start_z_offset_mm": 0,
            "repeat": 1,
            "started_at": "2026-07-22T00:00:00+00:00",
            "ended_at": "2026-07-22T00:01:00+00:00",
            "duration_s": 60.0,
            "process_exit_status": 0,
            "completed": True,
            "timed_out": False,
            "track_acquired": True,
            "visual_alignment_locked": True,
            "final_center_error_px": 2.0,
            "final_range_error_mm": -3.0,
            "final_physical_tracking_error_mm": 0.3,
            "final_estimated_opening_world_m": [1.0, 2.0, 3.0],
            "ground_truth_position_error_mm": None,
            "maximum_target_step_mm": 1.0,
            "maximum_orientation_deviation_rad": 0.01,
            "perception_rejection_count": 0,
            "track_reacquisition_count": 0,
            "fatal_error": "",
            "insertion_command_count": 0,
            "failed_gates": [],
            "qualified": False,
        }

    def test_all_gate_boundaries_pass(self):
        evaluated = evaluate_parent_result(
            self._passing_payload(),
            truth_center_world_m=[1.0005, 2.0, 3.0],
        )
        self.assertTrue(evaluated["qualified"])
        self.assertEqual(evaluated["failed_gates"], [])
        self.assertAlmostEqual(evaluated["ground_truth_position_error_mm"], 0.5)

    def test_nonfinite_or_one_over_limit_fails(self):
        for key, value in (
            ("final_center_error_px", 2.000001),
            ("final_range_error_mm", 3.000001),
            ("final_physical_tracking_error_mm", 0.300001),
            ("maximum_target_step_mm", 1.000001),
            ("maximum_orientation_deviation_rad", 0.010001),
            ("final_center_error_px", math.nan),
        ):
            with self.subTest(key=key, value=value):
                payload = self._passing_payload()
                payload[key] = value
                evaluated = evaluate_parent_result(
                    payload,
                    truth_center_world_m=[1.0, 2.0, 3.0],
                )
                self.assertFalse(evaluated["qualified"])
                self.assertTrue(evaluated["failed_gates"])

    def test_suite_requires_twenty_seven_of_twenty_seven(self):
        passing = evaluate_parent_result(
            self._passing_payload(),
            truth_center_world_m=[1.0, 2.0, 3.0],
        )
        results = [dict(passing, run_id=f"run-{index}") for index in range(27)]
        summary = aggregate_suite(results, [item["run_id"] for item in results])
        self.assertTrue(summary["QUALIFIED"])
        results[-1]["qualified"] = False
        results[-1]["failed_gates"] = ["center_error"]
        summary = aggregate_suite(results, [item["run_id"] for item in results])
        self.assertFalse(summary["QUALIFIED"])
```

- [ ] **Step 5: Implement result creation, atomic writing, evaluation, and aggregation**

Add these behaviors to `stress_alignment.py`:

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
        "duration_s": math.nan,
        "process_exit_status": None,
        "completed": False,
        "timed_out": False,
        "track_acquired": False,
        "visual_alignment_locked": False,
        "final_center_error_px": math.nan,
        "final_range_error_mm": math.nan,
        "final_physical_tracking_error_mm": math.nan,
        "final_estimated_opening_world_m": None,
        "ground_truth_position_error_mm": None,
        "maximum_target_step_mm": 0.0,
        "maximum_orientation_deviation_rad": 0.0,
        "perception_rejection_count": 0,
        "track_reacquisition_count": 0,
        "fatal_error": "",
        "insertion_command_count": 0,
        "failed_gates": [],
        "qualified": False,
    }


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finite_number(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def evaluate_parent_result(
    payload: dict[str, object],
    truth_center_world_m: Sequence[float],
) -> dict[str, object]:
    result = dict(payload)
    failed: list[str] = []
    estimated = result.get("final_estimated_opening_world_m")
    try:
        estimated_array = np.asarray(estimated, dtype=np.float64).reshape(3)
        truth_array = np.asarray(truth_center_world_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(estimated_array)) or not np.all(np.isfinite(truth_array)):
            raise ValueError("non-finite point")
        truth_error_mm = 1000.0 * float(np.linalg.norm(estimated_array - truth_array))
    except Exception:
        truth_error_mm = math.nan
    result["ground_truth_position_error_mm"] = truth_error_mm

    boolean_gates = {
        "process_exit_status": result.get("process_exit_status") == 0,
        "completed": result.get("completed") is True,
        "timed_out": result.get("timed_out") is False,
        "track_acquired": result.get("track_acquired") is True,
        "visual_alignment_locked": result.get("visual_alignment_locked") is True,
        "fatal_error": not str(result.get("fatal_error", "")).strip(),
        "no_insertion": result.get("insertion_command_count") == 0,
    }
    failed.extend(name for name, passed in boolean_gates.items() if not passed)

    limits = {
        "duration_s": 240.0,
        "final_center_error_px": 2.0,
        "absolute_final_range_error_mm": 3.0,
        "final_physical_tracking_error_mm": 0.3,
        "ground_truth_position_error_mm": 1.0,
        "maximum_target_step_mm": 1.0 + STEP_EPSILON_MM,
        "maximum_orientation_deviation_rad": 0.01,
    }
    duration = _finite_number(result, "duration_s")
    center = _finite_number(result, "final_center_error_px")
    range_error = _finite_number(result, "final_range_error_mm")
    tracking = _finite_number(result, "final_physical_tracking_error_mm")
    step = _finite_number(result, "maximum_target_step_mm")
    orientation = _finite_number(result, "maximum_orientation_deviation_rad")
    measured = {
        "duration_s": duration,
        "final_center_error_px": center,
        "absolute_final_range_error_mm": None if range_error is None else abs(range_error),
        "final_physical_tracking_error_mm": tracking,
        "ground_truth_position_error_mm": truth_error_mm if math.isfinite(truth_error_mm) else None,
        "maximum_target_step_mm": step,
        "maximum_orientation_deviation_rad": orientation,
    }
    for name, limit in limits.items():
        value = measured[name]
        if value is None or value > limit:
            failed.append(name)

    result["failed_gates"] = sorted(set(failed))
    result["qualified"] = not result["failed_gates"]
    return result


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_suite(
    results: Sequence[dict[str, object]],
    execution_order: Sequence[str],
) -> dict[str, object]:
    passed = sum(1 for result in results if result.get("qualified") is True)
    durations = [
        float(result["duration_s"])
        for result in results
        if isinstance(result.get("duration_s"), (int, float))
        and math.isfinite(float(result["duration_s"]))
    ]
    gate_failures: dict[str, int] = {}
    pose_failures: dict[str, int] = {}
    for result in results:
        if result.get("qualified") is True:
            continue
        pose = f"y{int(result['start_y_offset_mm']):+03d}_z{int(result['start_z_offset_mm']):+03d}"
        pose_failures[pose] = pose_failures.get(pose, 0) + 1
        for gate in result.get("failed_gates", []):
            gate_failures[str(gate)] = gate_failures.get(str(gate), 0) + 1
    return {
        "schema_version": STRESS_SCHEMA_VERSION,
        "seed": STRESS_SEED,
        "required_run_count": 27,
        "completed_run_count": len(results),
        "passed_run_count": passed,
        "failed_run_count": len(results) - passed,
        "QUALIFIED": len(results) == 27 and passed == 27,
        "execution_order": list(execution_order),
        "failure_counts_by_pose": pose_failures,
        "failure_counts_by_gate": gate_failures,
        "duration_s": {
            "minimum": min(durations) if durations else math.nan,
            "median": median(durations) if durations else math.nan,
            "p95": _percentile(durations, 95.0),
            "maximum": max(durations) if durations else math.nan,
        },
    }
```

- [ ] **Step 6: Run pure tests**

Run:

```bash
/usr/bin/python3 -m unittest -v tests.test_alignment_stress
```

Expected: all matrix, config, gate, non-finite, and 27/27 tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add single_rack_cv/stress_alignment.py \
        single_rack_cv/tests/test_alignment_stress.py
git commit -m "Add alignment stress-test domain model"
```

---

### Task 2: Passive runtime instrumentation

**Files:**
- Modify: `single_rack_cv/sim.py` around `VisualServoState`, `SimulationRuntime.__init__`, `note_perception_failure`, `observe_visual_servo`, `_update_visual_acquisition`, `update_ik`, and completion methods
- Modify: `single_rack_cv/tests/test_alignment_stress.py`

**Interfaces:**
- Consumes: `quaternion_angular_distance_rad()` from `stress_alignment.py`.
- Produces: `SimulationRuntime.stress_snapshot() -> dict[str, object]` containing child-side measurements only.

- [ ] **Step 1: Add failing pure quaternion-distance tests**

```python
from stress_alignment import quaternion_angular_distance_rad

    def test_quaternion_distance_treats_q_and_negative_q_as_same_rotation(self):
        self.assertAlmostEqual(
            quaternion_angular_distance_rad((1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)),
            0.0,
            places=12,
        )

    def test_quaternion_distance_reports_known_angle(self):
        half = math.radians(5.0)
        q = (math.cos(half), 0.0, math.sin(half), 0.0)
        self.assertAlmostEqual(
            quaternion_angular_distance_rad((1.0, 0.0, 0.0, 0.0), q),
            math.radians(10.0),
            places=12,
        )
```

- [ ] **Step 2: Run and verify failure**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_alignment_stress.AlignmentStressTests.test_quaternion_distance_treats_q_and_negative_q_as_same_rotation \
  tests.test_alignment_stress.AlignmentStressTests.test_quaternion_distance_reports_known_angle
```

Expected: import failure for `quaternion_angular_distance_rad`.

- [ ] **Step 3: Implement quaternion distance in the pure module**

```python
def quaternion_angular_distance_rad(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(4)
    b = np.asarray(second, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("quaternions must be finite")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1.0e-12 or b_norm <= 1.0e-12:
        raise ValueError("quaternions must have nonzero length")
    cosine = abs(float(np.dot(a / a_norm, b / b_norm)))
    cosine = min(1.0, max(-1.0, cosine))
    return 2.0 * math.acos(cosine)
```

- [ ] **Step 4: Add passive fields to `VisualServoState`**

Add only measurement state:

```python
    track_acquired_ever: bool = False
    visual_alignment_locked_ever: bool = False
    track_acquisition_count: int = 0
    perception_rejection_count: int = 0
    final_center_error_px: float = math.nan
    final_range_error_m: float = math.nan
    final_estimated_opening_world_m: np.ndarray | None = None
    maximum_target_step_m: float = 0.0
    maximum_orientation_deviation_rad: float = 0.0
```

In `SimulationRuntime.__init__`, initialize:

```python
        self._initial_tool_orientation_wxyz: np.ndarray | None = None
```

- [ ] **Step 5: Instrument existing events without adding commands**

In `note_perception_failure()` increment before existing logic:

```python
        state.perception_rejection_count += 1
```

At the start of `observe_visual_servo()` after validating `self.ik`, record the refined opening observation:

```python
        state.final_center_error_px = float(np.linalg.norm(observation.center_error_px))
        state.final_range_error_m = float(observation.range_error_m)
        state.final_estimated_opening_world_m = np.asarray(
            observation.center_world_xyz_m,
            dtype=np.float64,
        ).copy()
```

Immediately after `compute_bounded_step()`:

```python
        commanded_step_m = float(np.linalg.norm(step_world_m))
        state.maximum_target_step_m = max(
            state.maximum_target_step_m,
            commanded_step_m,
        )
```

When acquisition becomes stable:

```python
        state.track_acquired_ever = True
        state.track_acquisition_count += 1
```

When visual alignment locks:

```python
                state.visual_alignment_locked_ever = True
```

In `_create_ik()`, after `tool_orientation` is computed:

```python
        self._initial_tool_orientation_wxyz = np.asarray(
            tool_orientation,
            dtype=np.float64,
        ).copy()
```

In `update_ik()`, immediately after reading target pose, update the passive orientation measurement:

```python
        if self._initial_tool_orientation_wxyz is not None:
            orientation_deviation = quaternion_angular_distance_rad(
                self._initial_tool_orientation_wxyz,
                desired_tool_orientation,
            )
            self.visual_servo.maximum_orientation_deviation_rad = max(
                self.visual_servo.maximum_orientation_deviation_rad,
                orientation_deviation,
            )
```

- [ ] **Step 6: Add the public snapshot**

Add this method to `SimulationRuntime`:

```python
    def stress_snapshot(self) -> dict[str, object]:
        state = self.visual_servo
        physical_error_m = self._tool_target_position_error_m()
        center_world = state.final_estimated_opening_world_m
        return {
            "completed": bool(state.complete),
            "track_acquired": bool(state.track_acquired_ever),
            "visual_alignment_locked": bool(state.visual_alignment_locked_ever),
            "final_center_error_px": float(state.final_center_error_px),
            "final_range_error_mm": 1000.0 * float(state.final_range_error_m),
            "final_physical_tracking_error_mm": 1000.0 * float(physical_error_m),
            "final_estimated_opening_world_m": (
                None
                if center_world is None
                else [float(value) for value in center_world]
            ),
            "maximum_target_step_mm": 1000.0 * float(state.maximum_target_step_m),
            "maximum_orientation_deviation_rad": float(
                state.maximum_orientation_deviation_rad
            ),
            "perception_rejection_count": int(state.perception_rejection_count),
            "track_reacquisition_count": max(0, state.track_acquisition_count - 1),
            "insertion_command_count": 0,
        }
```

Import `quaternion_angular_distance_rad` from `stress_alignment` near the existing pure helper imports. Do not add any method that moves along the port axis.

- [ ] **Step 7: Run pure tests and compile `sim.py`**

```bash
/usr/bin/python3 -m unittest -v tests.test_alignment_stress
"$HOME/isaacsim/python.sh" -m py_compile sim.py stress_alignment.py
```

Expected: tests pass and compilation exits 0.

- [ ] **Step 8: Commit Task 2**

```bash
git add single_rack_cv/stress_alignment.py \
        single_rack_cv/sim.py \
        single_rack_cv/tests/test_alignment_stress.py
git commit -m "Instrument visual servo stress metrics"
```

---

### Task 3: Opt-in child stress mode in `main.py`

**Files:**
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Consumes: `parse_stress_run_args`, `derive_stress_config`, `new_child_result`, `utc_now_iso`, and `write_json_atomic`.
- Produces: one child `result.json` with runtime measurements but no ground-truth score; exits automatically only in stress mode.

- [ ] **Step 1: Add failing structural tests for opt-in behavior**

```python
    def test_stress_mode_is_optional_and_default_runtime_remains_interactive(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("parse_stress_run_args", source)
        self.assertIn("stress_args is None", source)
        self.assertIn("derive_stress_config", source)
        self.assertIn("exit_after_complete", source)
        self.assertIn("runtime.stress_snapshot()", source)
        self.assertIn("write_json_atomic", source)

    def test_child_runtime_does_not_read_ground_truth(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("front_plane_ground_truth.json", source)
        self.assertNotIn("center_world_m\"]", source)
```

- [ ] **Step 2: Run and verify failure**

```bash
/usr/bin/python3 -m unittest -v tests.test_runtime_wiring
```

Expected: new stress-mode assertions fail.

- [ ] **Step 3: Parse stress mode before starting Isaac**

At the top of `main.py`, add `time` and imports from `stress_alignment`. Resolve configuration like this before constructing output paths:

```python
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

Replace runtime uses of `CONFIG` with `RUNTIME_CONFIG`. Keep the module-level canonical `CONFIG` import unchanged so normal execution is identical.

Select the internal tee path without colliding with the suite log:

```python
if stress_args is None:
    run_output_path = RUNTIME_CONFIG.camera.output_dir / "run_output_latest.txt"
else:
    run_output_path = stress_args.result_json.parent / "runtime_output.txt"
```

- [ ] **Step 4: Add child timing and controlled termination**

Before the outer `try`:

```python
started_at = utc_now_iso()
started_monotonic = time.monotonic()
fatal_error = ""
timed_out = False
child_exit_status = 0
```

At the top of each loop iteration, after `runtime.update_visual_servo_completion()`:

```python
        if stress_args is not None:
            elapsed_s = time.monotonic() - started_monotonic
            if elapsed_s >= stress_args.timeout_s:
                timed_out = True
                child_exit_status = 2
                warn(
                    "Alignment stress run reached its 240 second runtime timeout; "
                    "holding and shutting down."
                )
                break
            if (
                stress_args.exit_after_complete
                and runtime.visual_servo.complete
            ):
                break
```

In the outer exception block, preserve default behavior while converting stress failures to a result:

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

- [ ] **Step 5: Write the child result in `finally`**

Before stopping the runtime, build from `new_child_result()` and merge `runtime.stress_snapshot()` when available:

```python
        if stress_args is not None:
            payload = new_child_result(stress_args, started_at)
            if runtime is not None:
                payload.update(runtime.stress_snapshot())
            payload.update(
                {
                    "ended_at": utc_now_iso(),
                    "duration_s": time.monotonic() - started_monotonic,
                    "timed_out": timed_out,
                    "fatal_error": fatal_error,
                }
            )
            write_json_atomic(stress_args.result_json, payload)
```

After the existing `finally` finishes:

```python
if stress_args is not None:
    raise SystemExit(child_exit_status)
```

Do not change normal-mode shutdown, exception propagation, controller gains, camera settings, or completion behavior.

- [ ] **Step 6: Run structural tests and compile**

```bash
/usr/bin/python3 -m unittest -v tests.test_runtime_wiring
"$HOME/isaacsim/python.sh" -m py_compile main.py sim.py stress_alignment.py
```

Expected: all structural tests pass and compilation exits 0.

- [ ] **Step 7: Commit Task 3**

```bash
git add single_rack_cv/main.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Add isolated child stress-run mode"
```

---

### Task 4: Parent subprocess orchestrator and post-run scoring

**Files:**
- Create: `single_rack_cv/tools/run_alignment_stress.py`
- Create: `single_rack_cv/tests/test_alignment_stress_runner.py`

**Interfaces:**
- Consumes: `build_stress_cases`, `evaluate_parent_result`, `aggregate_suite`, `write_json_atomic`, existing `tools/generate_ground_truth.py`, and `benchmarks/front_plane_ground_truth.json`.
- Produces: timestamped suite directory with per-run logs/results plus `summary.json`, `summary.csv`, and `report.txt`.

- [ ] **Step 1: Write failing command and synthesis tests**

Dynamically import the runner without launching Isaac:

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
    spec = importlib.util.spec_from_file_location("alignment_stress_runner_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AlignmentStressRunnerTests(unittest.TestCase):
    def test_child_command_contains_exact_case_and_timeout(self):
        runner = load_runner()
        case = StressCase(-10, 10, 2)
        command = runner.build_child_command(
            isaac_python=Path("/home/aayush/isaacsim/python.sh"),
            project_root=ROOT,
            case=case,
            result_json=Path("/tmp/result.json"),
        )
        self.assertEqual(command[0], "/home/aayush/isaacsim/python.sh")
        self.assertIn("--start-y-offset-mm", command)
        self.assertIn("-10", command)
        self.assertIn("--start-z-offset-mm", command)
        self.assertIn("10", command)
        self.assertIn("--stress-repeat", command)
        self.assertIn("2", command)
        self.assertIn("--stress-run-id", command)
        self.assertIn(case.run_id, command)
        self.assertIn("--exit-after-complete", command)

    def test_missing_result_is_synthesized_as_failure(self):
        runner = load_runner()
        case = StressCase(0, 0, 1)
        payload = runner.load_or_synthesize_child_result(
            case=case,
            result_json=Path("/definitely/missing/result.json"),
            process_exit_status=1,
            timed_out=False,
            duration_s=12.0,
            error_text="child result missing",
        )
        self.assertFalse(payload["completed"])
        self.assertEqual(payload["process_exit_status"], 1)
        self.assertIn("missing", payload["fatal_error"])
```

- [ ] **Step 2: Run and verify failure**

```bash
/usr/bin/python3 -m unittest -v tests.test_alignment_stress_runner
```

Expected: runner module does not exist.

- [ ] **Step 3: Implement runner constants, imports, and command construction**

Create `tools/run_alignment_stress.py` using only standard-library imports plus the pure project module:

```python
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
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
    evaluate_parent_result,
    new_child_result,
    StressRunArgs,
    utc_now_iso,
    write_json_atomic,
)

GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmarks" / "front_plane_ground_truth.json"
OUTPUT_ROOT = PROJECT_ROOT / "camera_output" / "alignment_stress"


def build_child_command(
    isaac_python: Path,
    project_root: Path,
    case: StressCase,
    result_json: Path,
) -> list[str]:
    return [
        str(isaac_python),
        str(project_root / "main.py"),
        "--start-y-offset-mm", str(case.y_offset_mm),
        "--start-z-offset-mm", str(case.z_offset_mm),
        "--stress-repeat", str(case.repeat),
        "--stress-run-id", case.run_id,
        "--stress-result-json", str(result_json),
        "--stress-timeout-s", str(CHILD_TIMEOUT_S),
        "--exit-after-complete",
    ]
```

- [ ] **Step 4: Implement ground-truth validation and generation**

```python
def load_valid_ground_truth(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("camera_resolution_height_width") != [960, 1280]:
        raise ValueError("ground truth resolution is not 1280x960")
    if not str(payload.get("control_usage", "")).lower().startswith("forbidden"):
        raise ValueError("ground truth is not marked benchmark-only")
    center = payload.get("center_world_m")
    if not isinstance(center, list) or len(center) != 3:
        raise ValueError("ground truth center_world_m is invalid")
    return payload


def ensure_ground_truth(isaac_python: Path) -> dict[str, object]:
    try:
        return load_valid_ground_truth(GROUND_TRUTH_PATH)
    except Exception:
        GROUND_TRUTH_PATH.unlink(missing_ok=True)
        command = [str(isaac_python), str(PROJECT_ROOT / "tools" / "generate_ground_truth.py")]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            timeout=PARENT_TIMEOUT_S,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ground-truth generation failed with status {completed.returncode}"
            )
        return load_valid_ground_truth(GROUND_TRUTH_PATH)
```

This is the only place the stress suite reads benchmark ground truth. Children never receive the truth path or center.

- [ ] **Step 5: Implement child-result loading and failure synthesis**

```python
def load_or_synthesize_child_result(
    case: StressCase,
    result_json: Path,
    process_exit_status: int,
    timed_out: bool,
    duration_s: float,
    error_text: str,
) -> dict[str, object]:
    if result_json.is_file():
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        except Exception as exc:
            error_text = f"malformed child result: {exc}"
        else:
            payload["process_exit_status"] = int(process_exit_status)
            payload["timed_out"] = bool(timed_out or payload.get("timed_out"))
            payload["duration_s"] = float(duration_s)
            if error_text and not str(payload.get("fatal_error", "")).strip():
                payload["fatal_error"] = error_text
            return payload
    args = StressRunArgs(
        case=case,
        result_json=result_json,
        timeout_s=CHILD_TIMEOUT_S,
        exit_after_complete=True,
    )
    payload = new_child_result(args, utc_now_iso())
    payload.update(
        {
            "ended_at": utc_now_iso(),
            "duration_s": float(duration_s),
            "process_exit_status": int(process_exit_status),
            "timed_out": bool(timed_out),
            "fatal_error": error_text or "child result missing",
        }
    )
    return payload
```

- [ ] **Step 6: Implement one isolated subprocess run**

Use a new process group so timeout and Ctrl-C terminate the whole Isaac process tree:

```python
def run_one_case(
    isaac_python: Path,
    case: StressCase,
    run_directory: Path,
) -> dict[str, object]:
    run_directory.mkdir(parents=True, exist_ok=False)
    result_json = run_directory / "result.json"
    console_log = run_directory / "console.log"
    command = build_child_command(isaac_python, PROJECT_ROOT, case, result_json)
    started = time.monotonic()
    timed_out = False
    error_text = ""
    with console_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=PARENT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            error_text = "parent hard timeout after 270 seconds"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=10.0)
    duration_s = time.monotonic() - started
    return load_or_synthesize_child_result(
        case=case,
        result_json=result_json,
        process_exit_status=returncode,
        timed_out=timed_out,
        duration_s=duration_s,
        error_text=error_text,
    )
```

- [ ] **Step 7: Implement unique output directories and reports**

Create a directory using UTC time and refuse overwrite:

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
    raise RuntimeError("could not allocate a unique alignment-stress directory")
```

Implement `write_suite_outputs(suite_dir, results, execution_order)` to:

1. Call `aggregate_suite()`.
2. Add worst finite values for center, absolute range, ground truth, physical tracking, target step, and orientation.
3. Add total perception rejections and reacquisitions.
4. Write `summary.json` atomically.
5. Write one CSV row per run with all schema fields.
6. Write a text report beginning with `ALIGNMENT STRESS QUALIFICATION`, `passed_run_count`, `failed_run_count`, and `QUALIFIED`.

Use these exact CSV fields:

```python
CSV_FIELDS = [
    "run_id", "start_y_offset_mm", "start_z_offset_mm", "repeat",
    "duration_s", "process_exit_status", "completed", "timed_out",
    "track_acquired", "visual_alignment_locked", "final_center_error_px",
    "final_range_error_mm", "final_physical_tracking_error_mm",
    "ground_truth_position_error_mm", "maximum_target_step_mm",
    "maximum_orientation_deviation_rad", "perception_rejection_count",
    "track_reacquisition_count", "insertion_command_count", "qualified",
    "failed_gates", "fatal_error",
]
```

Serialize `failed_gates` as `;`.join(...) in CSV.

- [ ] **Step 8: Implement suite `main()` and exit mapping**

```python
def main() -> int:
    isaac_python = Path.home() / "isaacsim" / "python.sh"
    if not isaac_python.is_file():
        print(f"ERROR: Isaac launcher not found: {isaac_python}", file=sys.stderr)
        return 1
    suite_dir = create_suite_directory()
    truth = ensure_ground_truth(isaac_python)
    truth_center = truth["center_world_m"]
    cases = build_stress_cases(STRESS_SEED)
    execution_order = [case.run_id for case in cases]
    results: list[dict[str, object]] = []
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index:02d}/27] START {case.run_id}", flush=True)
            child = run_one_case(
                isaac_python=isaac_python,
                case=case,
                run_directory=suite_dir / "runs" / case.directory_name,
            )
            evaluated = evaluate_parent_result(child, truth_center)
            write_json_atomic(
                suite_dir / "runs" / case.directory_name / "result.json",
                evaluated,
            )
            results.append(evaluated)
            write_suite_outputs(suite_dir, results, execution_order)
            status = "PASS" if evaluated["qualified"] else "FAIL"
            print(f"[{index:02d}/27] {status} {case.run_id}", flush=True)
    except KeyboardInterrupt:
        write_suite_outputs(suite_dir, results, execution_order)
        print("Interrupted; partial report written.", file=sys.stderr)
        return 1
    summary = write_suite_outputs(suite_dir, results, execution_order)
    print((suite_dir / "report.txt").read_text(encoding="utf-8"), flush=True)
    return 0 if summary["QUALIFIED"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 9: Complete runner tests**

Add tests for:

- malformed JSON synthesis;
- valid result preserving child metrics while parent overwrites process status and duration;
- unique output allocation without overwrite;
- report contains `QUALIFIED=False` for 26/27;
- command uses a fresh result path and exact 240-second child timeout;
- parent timeout constant is 270 seconds;
- ground-truth validator rejects missing `control_usage`, wrong resolution, and malformed center.

Use temporary directories and patch module constants; do not start subprocesses.

- [ ] **Step 10: Run pure runner tests**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_alignment_stress \
  tests.test_alignment_stress_runner
```

Expected: all tests pass without importing Isaac.

- [ ] **Step 11: Commit Task 4**

```bash
git add single_rack_cv/tools/run_alignment_stress.py \
        single_rack_cv/tests/test_alignment_stress_runner.py
git commit -m "Add isolated alignment stress orchestrator"
```

---

### Task 5: Shell entry point, documentation, and structural safety guards

**Files:**
- Create: `single_rack_cv/tools/run_alignment_stress.sh`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`
- Modify: `single_rack_cv/README.md`

**Interfaces:**
- Consumes: parent runner from Task 4.
- Produces: one supported workstation command and source-level proof that runtime safety constraints remain intact.

- [ ] **Step 1: Add failing structural assertions**

Extend `test_runtime_wiring.py`:

```python
    def test_stress_runner_scores_truth_only_after_child_process(self):
        source = (ROOT / "tools" / "run_alignment_stress.py").read_text(
            encoding="utf-8"
        )
        run_index = source.index("run_one_case(")
        evaluate_index = source.index("evaluate_parent_result(", run_index)
        self.assertLess(run_index, evaluate_index)
        self.assertIn("front_plane_ground_truth.json", source)
        self.assertNotIn("ground_truth", (ROOT / "main.py").read_text(encoding="utf-8").lower())

    def test_stress_path_preserves_one_millimeter_step_and_no_insertion(self):
        config_source = (ROOT / "config.py").read_text(encoding="utf-8")
        sim_source = (ROOT / "sim.py").read_text(encoding="utf-8")
        stress_source = (ROOT / "stress_alignment.py").read_text(encoding="utf-8")
        self.assertIn("max_target_step_m: float = 0.001", config_source)
        self.assertIn("maximum_target_step_mm", sim_source)
        self.assertIn("insertion_command_count\": 0", sim_source)
        self.assertNotIn("insert_along", sim_source)
        self.assertIn("maximum_orientation_deviation_rad", stress_source)
```

- [ ] **Step 2: Run and verify failure**

```bash
/usr/bin/python3 -m unittest -v tests.test_runtime_wiring
```

Expected: shell/docs-related source assertions fail until files are added and final strings match.

- [ ] **Step 3: Create the sanitized shell launcher**

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

printf '[ALIGNMENT STRESS] matrix=3x3 world Y/Z, repeats=3, runs=27\n'
printf '[ALIGNMENT STRESS] child timeout=240s, parent timeout=270s\n'
printf '[ALIGNMENT STRESS] qualification requires 27/27\n'

exec /usr/bin/python3 tools/run_alignment_stress.py
```

Set executable permission:

```bash
chmod +x tools/run_alignment_stress.sh
```

- [ ] **Step 4: Document exact operation**

Add a README section with:

```markdown
## Alignment start-pose stress qualification

This launches 27 fresh Isaac processes: a 3×3 world-frame Y/Z grid at -10, 0, and +10 mm, repeated three times in deterministic shuffled order. X and wrist orientation remain unchanged.

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
set -o pipefail
bash tools/run_alignment_stress.sh \
  2>&1 | tee camera_output/alignment_stress_console.txt
status=${PIPESTATUS[0]}
echo "alignment stress exit status: $status"
```

Exit codes:

- `0`: all 27 runs qualified;
- `2`: suite completed but at least one run failed;
- `1`: infrastructure failure or interruption.

The suite writes `summary.json`, `summary.csv`, `report.txt`, and one log/result pair per run under `camera_output/alignment_stress/<timestamp>/`.

Do not proceed to insertion unless the report says `QUALIFIED=True` and `passed_run_count=27`.
```

Also remove the obsolete README recovery-branch section because those temporary branches were intentionally deleted after the cleanup merge.

- [ ] **Step 5: Run shell syntax and structural tests**

```bash
bash -n tools/run_alignment_stress.sh
/usr/bin/python3 -m unittest -v tests.test_runtime_wiring
```

Expected: shell syntax exits 0 and structural tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add single_rack_cv/tools/run_alignment_stress.sh \
        single_rack_cv/tests/test_runtime_wiring.py \
        single_rack_cv/README.md
git commit -m "Document alignment stress qualification"
```

---

### Task 6: Full verification and workstation qualification

**Files:**
- No new source files unless verification exposes a defect.
- Generated outputs remain under ignored `single_rack_cv/camera_output/`.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: evidence that existing benchmark/live behavior is preserved and the new 27-run harness is trustworthy.

- [ ] **Step 1: Run the complete pure and structural suite**

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

Expected: zero failures and zero errors.

- [ ] **Step 2: Compile all modified Python entry points and check shell syntax**

```bash
"$HOME/isaacsim/python.sh" -m py_compile \
  stress_alignment.py \
  sim.py \
  main.py \
  tools/run_alignment_stress.py
bash -n tools/run_alignment_stress.sh
```

Expected: all commands exit 0.

- [ ] **Step 3: Re-run the existing front-plane benchmark**

```bash
set -o pipefail
bash tools/run_benchmark.sh \
  2>&1 | tee camera_output/pre_stress_benchmark_console.txt
benchmark_status=${PIPESTATUS[0]}
echo "benchmark exit status: $benchmark_status"
cat camera_output/front_plane_benchmark/report.txt
```

Expected:

- exit status `0`;
- `QUALIFIED=True`;
- pair success at least 95%;
- zero track switches;
- all frozen gates unchanged.

Do not modify estimator thresholds to recover a regression.

- [ ] **Step 4: Run one nominal child stress process before spending hours on 27 runs**

```bash
mkdir -p camera_output/alignment_stress_smoke
"$HOME/isaacsim/python.sh" main.py \
  --start-y-offset-mm 0 \
  --start-z-offset-mm 0 \
  --stress-repeat 1 \
  --stress-run-id y+00_z+00_r1 \
  --stress-result-json camera_output/alignment_stress_smoke/result.json \
  --stress-timeout-s 240 \
  --exit-after-complete \
  2>&1 | tee camera_output/alignment_stress_smoke/console.log
status=${PIPESTATUS[0]}
echo "smoke exit status: $status"
cat camera_output/alignment_stress_smoke/result.json
```

Expected:

- process exits on its own with status `0`;
- `completed=true`;
- `track_acquired=true`;
- `visual_alignment_locked=true`;
- no ground-truth score exists yet in the child file;
- `maximum_target_step_mm<=1.000000001`;
- `maximum_orientation_deviation_rad<=0.01`;
- `insertion_command_count=0`.

Kill switch: stop here if the nominal smoke run fails. Do not start the 27-run suite until the child lifecycle is correct.

- [ ] **Step 5: Run the full 27-run suite**

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

Expected qualification:

```text
passed_run_count=27
failed_run_count=0
QUALIFIED=True
```

Also verify from `summary.json`:

- every one of the nine Y/Z poses appears exactly three times;
- seed is `20260722`;
- all 27 run IDs are unique;
- worst center error is at most 2 px;
- worst absolute range error is at most 3 mm;
- worst benchmark-ground-truth error is at most 1 mm;
- worst physical tracking error is at most 0.3 mm;
- maximum target step is at most `1.000000001` mm;
- maximum orientation deviation is at most 0.01 rad;
- insertion command total is zero.

- [ ] **Step 6: Inspect every failed run if the suite is not 27/27**

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

For each failure, inspect its sibling `console.log`. Fix the underlying perception, reachability, lifecycle, or instrumentation defect. Do not remove the pose, increase tolerances, or convert 27/27 into a percentage gate.

- [ ] **Step 7: Verify repository cleanliness**

```bash
git status --short
git diff --check
git ls-files single_rack_cv/camera_output
```

Expected:

- no generated camera output is tracked;
- `git diff --check` prints nothing;
- only intended source, tests, docs, and scripts are modified.

- [ ] **Step 8: Commit any verification-only fixes, then run the affected tests again**

Use a narrowly named commit for each discovered defect. Example for a timeout-process-tree correction:

```bash
git add single_rack_cv/tools/run_alignment_stress.py \
        single_rack_cv/tests/test_alignment_stress_runner.py
git commit -m "Fix stress child timeout cleanup"
/usr/bin/python3 -m unittest -v tests.test_alignment_stress_runner
```

Do not create a catch-all “fixes” commit.

- [ ] **Step 9: Final evidence checkpoint**

Record in the pull request description:

- complete unit-test command and pass count;
- benchmark `QUALIFIED=True` metrics;
- nominal smoke result;
- full stress suite directory;
- `27/27` result or exact failing poses;
- confirmation that no insertion command exists.

Do not merge unless both the frozen benchmark and full stress suite pass.
