# single_rack_cv Layout Reorganization Design

Date: 2026-08-07

## Goal

Reorganize only `single_rack_cv/` so the top-level directory is easy to understand without changing runtime behavior, safety limits, launch commands, calibration values, or the tested insertion pipeline.

The exact launch contract must remain:

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh main.py
```

`main.py`, `sim.py`, `config.py`, `debug.py`, and `README.md` remain at the top level.

## Non-goals

This is not a behavioral refactor. It must not change perception algorithms, thresholds, camera geometry, cable mounting, ToolCenter geometry, handoff logic, insertion targets, safety gates, logging semantics, or benchmark acceptance criteria.

No files outside `single_rack_cv/` may be modified.

## Proposed Structure

```text
single_rack_cv/
├── main.py
├── sim.py
├── config.py
├── debug.py
├── README.md
├── vision/
├── cable/
├── control/
├── robot/
├── runtime/
├── benchmarks/
├── tools/
├── tests/
├── docs/
└── assets/
```

### `vision/`

Owns camera/stereo perception and front-lip localization:

- `perception.py`
- `stereo_geometry.py`
- `front_plane.py`
- `live_control.py`
- `live_control_projective.py`
- `outer_bezel_center.py`
- `outer_bezel_projective_center.py`
- `front_lip_calibration.py`
- `plane_rectified_types.py`
- `plane_rectified_geometry.py`
- `plane_rectified_fit_utils.py`
- `plane_rectified_fitting.py`
- `plane_rectified_width_hypotheses.py`
- `plane_rectified_front_lip.py`

### `cable/`

Owns connector geometry, USD-derived TCP calibration, fixed-joint mounting, scale handling, and deformable-tail setup:

- `cable_geometry.py`
- `cable_mount.py`
- `scale_aware_cable_mount.py`
- `connector_tcp.py`
- `connector_tcp_usd.py`
- `tail_preshape.py`
- `affine_root_geometry.py`

### `control/`

Owns pure or mostly pure controller/state-machine logic:

- `insertion.py`
- `settled_insertion.py`
- `insertion_target_trim.py`
- `plug_axis_insertion.py`
- `orientation_hold.py`
- `handoff_position_hold.py`
- `stereo_handoff.py`
- `precontact_alignment.py`
- `validation_window.py`
- `tool_goal_trim.py`

### `robot/`

Owns Franka/hand geometry and host-array compatibility helpers:

- `angled_hand_config.py`
- `angled_grasp_centering.py`
- `hand_plug_geometry.py`
- `host_array_bridge.py`
- `articulation_host_bridge.py`

### `runtime/`

Owns the layered Isaac Sim runtime wrappers used by `main.py`:

- `cable_runtime.py`
- `angled_hand_runtime.py`
- `stereo_handoff_runtime.py`
- `settled_stereo_handoff_runtime.py`
- `handoff_position_hold_runtime.py`
- `full_insertion_base_runtime.py`
- `full_insertion_runtime.py`
- `precontact_runtime.py`

Existing `benchmarks/`, `tools/`, `tests/`, `docs/`, and `assets/` remain conceptually unchanged.

## Import Migration

Moved modules become real packages using `__init__.py` files. Imports are rewritten explicitly, for example:

```python
from plane_rectified_types import PlaneFrame
```

becomes:

```python
from vision.plane_rectified_types import PlaneFrame
```

The same rule applies across vision, cable, control, robot, and runtime modules.

The migration must avoid runtime `sys.path` hacks. Imports should be normal package imports from the `single_rack_cv` working directory.

## Compatibility Policy

The primary compatibility contract is the user-facing launch command, not preservation of every historical internal import path.

Thin top-level compatibility shims may remain only where they materially reduce breakage risk for tests, tools, or the existing runtime bootstrap. They should only re-export the relocated implementation and must contain no duplicated logic.

A shim must have a concrete consumer. Do not keep wrappers merely because an old path once existed.

The existing `cable_runtime.py` / `cable_runtime/` name collision must be removed during the migration. The production implementation should live unambiguously under `runtime/`.

## Runtime Ownership

`main.py` remains the canonical executable entry point and imports the final runtime from `runtime/`.

`sim.py` remains top-level because it is the central Isaac Sim abstraction used broadly by the runtime stack and keeping it stable reduces migration risk.

`config.py` remains top-level as the single canonical configuration source.

`debug.py` remains top-level for operator convenience.

## Tests

Tests stay in `tests/`, but their imports and any hardcoded path assertions must be updated for the new package structure.

`tests/test_repo_cleanliness.py` must be changed so it validates the new structure instead of expecting old root-level paths. It should also reject accidental reintroduction of the retired flat layout for modules intentionally moved into packages.

No production test may be deleted merely because its import path changes.

## Verification Gates

The refactor is acceptable only if all of the following pass on the reorganization branch:

1. Repository cleanliness/import wiring tests pass.
2. The same focused qualification suite used before the restructure passes.
3. The canonical command still launches unchanged:

   ```bash
   ~/isaacsim/python.sh main.py
   ```

4. A complete Isaac Sim qualification run reaches the same successful end state as the pre-refactor baseline, including 48/48 insertion commands with the existing safety gates intact.

The pre-refactor qualified `main` remains the rollback baseline until all four gates pass.

## Safety Invariants

The restructure must not modify values or behavior related to:

- 12.9 mm visible front-lip width prior
- five bounded front-lip search-width hypotheses
- 0.5 mm independent-eye center disagreement gate
- 0.05 mm/pixel plane rectification
- 50 mm vision-derived handoff goal
- 0.300 mm physical handoff completion tolerance
- insertion-only world calibration `[0.0, -0.00030, -0.00045]` m
- 48-command two-stage insertion sequence
- 0.500 mm lateral insertion safety limit
- 1.000 degree orientation safety limit
- mount integrity, attachment integrity, IK preflight, and timeout gates

If a diff changes any of these values or the logic enforcing them, the change is outside this refactor and must be reverted.

## Failure / Rollback Rule

If imports, tests, Isaac startup, perception qualification, handoff, or insertion regress after moving files, do not compensate by relaxing gates or changing algorithms. Fix the packaging/import issue. If the packaging issue cannot be isolated quickly, abandon the branch and keep the current qualified `main`.

## Merge Strategy

All reorganization work happens on `refactor/single-rack-layout`.

Do not modify `main` directly. Open a PR only after static/unit verification. Merge only after the workstation test suite and one complete Isaac Sim qualification run pass.
