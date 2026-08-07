# single_rack_cv Layout Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `single_rack_cv/` into responsibility-based Python packages while preserving the exact `~/isaacsim/python.sh main.py` launch contract and every validated perception, cable, handoff, and insertion behavior.

**Architecture:** Keep `main.py`, `sim.py`, `config.py`, `debug.py`, and `README.md` at the root. Move implementation modules into `vision/`, `robot/`, `cable/`, `control/`, and `runtime/` using explicit package imports and no `sys.path` hacks. Remove the current `cable_runtime.py` versus `cable_runtime/` ambiguity by storing the base runtime in `runtime/cable_runtime_base.py` and the live PhysX facade in `runtime/cable_runtime.py`.

**Tech Stack:** Python through Isaac Sim 6.0.0 `python.sh`, NVIDIA Isaac Sim, OpenUSD/PXR, NumPy, OpenCV, PyTorch, `unittest`, Git/GitHub.

## Global Constraints

- Modify only paths below `single_rack_cv/`.
- Preserve this exact launch contract:

  ```bash
  cd ~/Isaacsim-Scripts/single_rack_cv
  ~/isaacsim/python.sh main.py
  ```

- Keep exactly these root Python files: `main.py`, `sim.py`, `config.py`, `debug.py`.
- Keep `README.md` at the root.
- Do not change perception algorithms, thresholds, camera geometry, cable mounting, ToolCenter geometry, handoff logic, insertion targets, safety gates, logging semantics, or benchmark acceptance criteria.
- Do not add runtime `sys.path` manipulation.
- Do not delete tests because import paths change.
- Do not modify the repository-root `.gitignore`.
- Do not mass-rewrite historical documents under `single_rack_cv/docs/`.
- Preserve these validated invariants exactly:
  - visible front-lip width prior: 12.9 mm;
  - visible front-lip height: 7.0 mm;
  - calibrated search-width ceiling: 11.4 mm;
  - five bounded front-lip search-width hypotheses;
  - independent-eye center disagreement gate: 0.5 mm;
  - plane rectification: 0.05 mm/pixel;
  - camera-derived handoff goal: 50 mm;
  - physical handoff completion tolerance: 0.300 mm;
  - insertion-only world calibration: `[0.0, -0.00030, -0.00045]` m;
  - insertion schedule: 48 commands;
  - lateral insertion safety limit: 0.500 mm;
  - orientation safety limit: 1.000 degree;
  - mount integrity, attachment integrity, IK preflight, and timeout gates.
- The qualified rollback baseline is `main` commit `636d4f8a79f021b8e3c73f4dfc726c9148654534`.

## Locked Final Layout

```text
single_rack_cv/
├── main.py
├── sim.py
├── config.py
├── debug.py
├── README.md
├── vision/
├── robot/
├── cable/
├── control/
├── runtime/
├── benchmarks/
├── tools/
├── tests/
├── docs/
└── assets/
```

The package ownership is fixed as follows.

**`vision/`**

```text
__init__.py
perception.py
stereo_geometry.py
front_plane.py
live_control.py
live_control_projective.py
aperture_center.py
outer_bezel_center.py
outer_bezel_projective_center.py
front_lip_calibration.py
plane_rectified_types.py
plane_rectified_geometry.py
plane_rectified_fit_utils.py
plane_rectified_fitting.py
plane_rectified_width_hypotheses.py
plane_rectified_front_lip.py
```

**`robot/`**

```text
__init__.py
angled_hand_config.py
angled_grasp_centering.py
hand_plug_geometry.py
host_array_bridge.py
articulation_host_bridge.py
```

**`cable/`**

```text
__init__.py
cable_geometry.py
cable_mount.py
scale_aware_cable_mount.py
connector_tcp.py
connector_tcp_usd.py
tail_preshape.py
affine_root_geometry.py
```

**`control/`**

```text
__init__.py
insertion.py
settled_insertion.py
insertion_target_trim.py
plug_axis_insertion.py
orientation_hold.py
handoff_position_hold.py
stereo_handoff.py
precontact_alignment.py
validation_window.py
tool_goal_trim.py
```

**`runtime/`**

```text
__init__.py
cable_runtime_base.py
cable_runtime.py
angled_hand_runtime.py
stereo_handoff_runtime.py
settled_stereo_handoff_runtime.py
handoff_position_hold_runtime.py
full_insertion_base_runtime.py
full_insertion_runtime.py
precontact_runtime.py
```

`aperture_center.py` is active vision support and therefore belongs in `vision/`. `automatic_port_ground_truth.py` is offline benchmark/ground-truth support and therefore moves to `benchmarks/automatic_port_ground_truth.py`. These are location changes only.

---

### Task 1: Create Package Boundaries and Encode the New Layout Contract

**Files**
- Create: `vision/__init__.py`
- Create: `robot/__init__.py`
- Create: `cable/__init__.py`
- Create: `control/__init__.py`
- Create: `runtime/__init__.py`
- Modify: `tests/test_repo_cleanliness.py`

- [ ] **Step 1: Add a failing package-existence test**

Add to `RepositoryCleanlinessTests` before creating the package files:

```python
    def test_responsibility_packages_exist(self):
        for package in ("vision", "robot", "cable", "control", "runtime"):
            init_path = ROOT / package / "__init__.py"
            self.assertTrue(init_path.is_file(), init_path)
```

- [ ] **Step 2: Prove the new test fails**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_repo_cleanliness.RepositoryCleanlinessTests.test_responsibility_packages_exist
```

Expected: `FAIL` because the package initializers do not yet exist.

- [ ] **Step 3: Create minimal package initializers**

Each initializer contains only a docstring, for example:

```python
"""Vision components for the single-rack RGB stereo pipeline."""
```

Use similarly specific docstrings for `robot`, `cable`, `control`, and `runtime`. Do not eagerly import sibling modules.

- [ ] **Step 4: Prove the package test passes**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_repo_cleanliness.RepositoryCleanlinessTests.test_responsibility_packages_exist
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add vision robot cable control runtime tests/test_repo_cleanliness.py
git commit -m "refactor: add single rack package boundaries"
```

---

### Task 2: Move the Vision Stack

**Files moved to `vision/`**

```text
perception.py
stereo_geometry.py
front_plane.py
live_control.py
live_control_projective.py
aperture_center.py
outer_bezel_center.py
outer_bezel_projective_center.py
front_lip_calibration.py
plane_rectified_types.py
plane_rectified_geometry.py
plane_rectified_fit_utils.py
plane_rectified_fitting.py
plane_rectified_width_hypotheses.py
plane_rectified_front_lip.py
```

**Files modified**
- `main.py`
- `sim.py`
- all moved vision modules whose sibling imports are currently flat
- current `benchmarks/` and `tools/` files that import a moved vision module
- vision tests and vision source-wiring tests

- [ ] **Step 1: Make a representative test require the package path**

Change the imports in `tests/test_plane_rectified_front_lip.py` to package-qualified imports. The key imports must include:

```python
from vision.plane_rectified_fitting import fit_rectified_front_lip
from vision.plane_rectified_geometry import build_plane_frame
from vision.plane_rectified_types import PlaneFrame, RectifiedEye
```

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_plane_rectified_front_lip
```

Expected before the move: import failure.

- [ ] **Step 2: Move the complete vision file set**

```bash
git mv perception.py vision/perception.py
git mv stereo_geometry.py vision/stereo_geometry.py
git mv front_plane.py vision/front_plane.py
git mv live_control.py vision/live_control.py
git mv live_control_projective.py vision/live_control_projective.py
git mv aperture_center.py vision/aperture_center.py
git mv outer_bezel_center.py vision/outer_bezel_center.py
git mv outer_bezel_projective_center.py vision/outer_bezel_projective_center.py
git mv front_lip_calibration.py vision/front_lip_calibration.py
git mv plane_rectified_types.py vision/plane_rectified_types.py
git mv plane_rectified_geometry.py vision/plane_rectified_geometry.py
git mv plane_rectified_fit_utils.py vision/plane_rectified_fit_utils.py
git mv plane_rectified_fitting.py vision/plane_rectified_fitting.py
git mv plane_rectified_width_hypotheses.py vision/plane_rectified_width_hypotheses.py
git mv plane_rectified_front_lip.py vision/plane_rectified_front_lip.py
```

- [ ] **Step 3: Convert every vision sibling import by prefix only**

For every import whose module is in the vision ownership list, preserve the imported symbol list exactly and change only the module name from `MODULE` to `vision.MODULE`.

Examples:

```python
from plane_rectified_types import PlaneFrame
```

becomes:

```python
from vision.plane_rectified_types import PlaneFrame
```

and:

```python
from stereo_geometry import triangulate_pixel_pair, unit_vector
```

becomes:

```python
from vision.stereo_geometry import triangulate_pixel_pair, unit_vector
```

Do not alter any function body, constant, threshold, or dataclass field in this step.

- [ ] **Step 4: Update root consumers**

`main.py` must import vision components with these module paths:

```python
from vision.front_lip_calibration import (
    VISIBLE_FRONT_LIP_HEIGHT_M,
    VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
    VISIBLE_FRONT_LIP_WIDTH_M,
)
from vision.live_control_projective import refine_live_observation
from vision.perception import YOLOEPortDetector, process_stereo_port
```

`sim.py` must keep the same imported perception symbols but source them from `vision.perception`:

```python
from vision.perception import (
    CameraFrame,
    CameraModel,
    PortDetection,
    StereoFrame,
    StereoPortObservation,
    build_virtual_camera_model,
    compute_bounded_step,
    compute_desired_port_camera_usd,
    normalize_rgb,
)
```

Do not alter the current locations of `config`, `debug`, or `sim` imports.

- [ ] **Step 5: Update current benchmark/tool consumers**

Search only `benchmarks/` and `tools/` for imports of the 15 moved vision module names. Change each matching module qualifier to `vision.<module>` while preserving imported symbols and executable behavior.

The benchmark source assertion in `tests/test_benchmark.py` must expect:

```python
self.assertIn("from vision.front_plane import", source)
```

- [ ] **Step 6: Update vision tests and source paths**

Migrate imports/path assertions in these tests:

```text
test_aperture_center.py
test_aperture_center_latch_asymmetry.py
test_front_plane.py
test_geometry_math.py
test_live_control.py
test_outer_bezel_center.py
test_plane_rectified_front_lip.py
test_plane_rectified_front_lip_workstation_fixture.py
test_front_lip_search_calibration.py
test_front_lip_left_bezel_rejection.py
test_visible_front_lip_calibration.py
test_visible_front_lip_geometry.py
test_plane_rectified_runtime_wiring.py
test_front_rim_plane_runtime_wiring.py
```

For source-reading tests use:

```python
VISION_ROOT = ROOT / "vision"
```

and read moved sources from `VISION_ROOT`. String assertions that inspect imports must expect the `vision.` prefix.

- [ ] **Step 7: Run the vision regression checkpoint**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_aperture_center \
  tests.test_aperture_center_latch_asymmetry \
  tests.test_front_plane \
  tests.test_geometry_math \
  tests.test_live_control \
  tests.test_outer_bezel_center \
  tests.test_plane_rectified_front_lip \
  tests.test_front_lip_search_calibration \
  tests.test_front_lip_left_bezel_rejection \
  tests.test_visible_front_lip_calibration \
  tests.test_visible_front_lip_geometry \
  tests.test_plane_rectified_runtime_wiring \
  tests.test_front_rim_plane_runtime_wiring
```

Expected: `OK`.

- [ ] **Step 8: Commit**

```bash
git add main.py sim.py vision benchmarks tools tests
git commit -m "refactor: group single rack vision modules"
```

---

### Task 3: Move Robot and Hand Geometry Helpers

**Files moved to `robot/`**

```text
angled_hand_config.py
angled_grasp_centering.py
hand_plug_geometry.py
host_array_bridge.py
articulation_host_bridge.py
```

- [ ] **Step 1: Make `test_hand_plug_geometry` require the new package**

Change its module source to `robot.hand_plug_geometry` while preserving the existing imported symbols. Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_hand_plug_geometry
```

Expected before the move: import failure.

- [ ] **Step 2: Move all five files**

```bash
git mv angled_hand_config.py robot/angled_hand_config.py
git mv angled_grasp_centering.py robot/angled_grasp_centering.py
git mv hand_plug_geometry.py robot/hand_plug_geometry.py
git mv host_array_bridge.py robot/host_array_bridge.py
git mv articulation_host_bridge.py robot/articulation_host_bridge.py
```

- [ ] **Step 3: Convert robot-related imports by prefix only**

For all production consumers, preserve imported symbols and change the relevant module names to:

```text
robot.angled_hand_config
robot.angled_grasp_centering
robot.hand_plug_geometry
robot.host_array_bridge
robot.articulation_host_bridge
```

In `cable/scale_aware_cable_mount.py` after Task 4, the articulation wrapper import must be:

```python
from robot.articulation_host_bridge import HostSafeDofPropertiesArticulation
```

- [ ] **Step 4: Update robot tests and path assertions**

Migrate:

```text
test_angled_grasp_centering.py
test_hand_plug_geometry.py
test_host_array_bridge.py
test_articulation_host_bridge.py
```

Runtime source-wiring tests that inspect `angled_hand_config.py` must read `ROOT / "robot" / "angled_hand_config.py"`.

- [ ] **Step 5: Run the robot-helper checkpoint**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_angled_grasp_centering \
  tests.test_hand_plug_geometry \
  tests.test_host_array_bridge \
  tests.test_articulation_host_bridge
```

Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add robot tests
git add -u .
git commit -m "refactor: group single rack robot helpers"
```

---

### Task 4: Move Cable Geometry, Mounting, TCP, and Tail Support

**Files moved to `cable/`**

```text
cable_geometry.py
cable_mount.py
scale_aware_cable_mount.py
connector_tcp.py
connector_tcp_usd.py
tail_preshape.py
affine_root_geometry.py
```

- [ ] **Step 1: Make `test_cable_geometry` require `cable.cable_geometry`**

Change only its module qualifier and run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_cable_geometry
```

Expected before the move: import failure.

- [ ] **Step 2: Move all seven cable modules**

```bash
git mv cable_geometry.py cable/cable_geometry.py
git mv cable_mount.py cable/cable_mount.py
git mv scale_aware_cable_mount.py cable/scale_aware_cable_mount.py
git mv connector_tcp.py cable/connector_tcp.py
git mv connector_tcp_usd.py cable/connector_tcp_usd.py
git mv tail_preshape.py cable/tail_preshape.py
git mv affine_root_geometry.py cable/affine_root_geometry.py
```

- [ ] **Step 3: Convert cable sibling imports by prefix only**

Preserve imported symbols. Critical module-object behavior in `cable/scale_aware_cable_mount.py` must use:

```python
from cable import cable_mount as cable_mount_module
```

It must continue patching and restoring `cable_mount_module.compute_world_from_root_for_tip`, `cable_mount_module._numpy_to_gf_matrix`, and `cable_mount_module.detect_plug_frame` inside the existing `try/finally` structure.

`cable/connector_tcp_usd.py` must source `PlugFrame` and `validate_transform` from `cable.cable_geometry`, `_world_transform` from `cable.cable_mount`, and TCP derivation functions/types from `cable.connector_tcp`.

`cable/scale_aware_cable_mount.py` must source `HostSafeDofPropertiesArticulation` from `robot.articulation_host_bridge`.

- [ ] **Step 4: Update cable tests and cable source-wiring paths**

Migrate imports/path assertions in:

```text
test_affine_root_geometry.py
test_cable_geometry.py
test_connector_tcp.py
test_scale_aware_cable_mount.py
test_tail_preshape.py
test_cable_mount_contract.py
test_connector_tcp_runtime_wiring.py
```

Source-reading tests use:

```python
CABLE_ROOT = ROOT / "cable"
```

- [ ] **Step 5: Update still-root runtime consumers to `cable.` module paths**

This is a temporary location only; runtime files themselves move in Task 6. Preserve the current import ordering and imported symbol lists.

In `full_insertion_base_runtime.py`, preserve this exact module-order relationship:

```python
from cable import connector_tcp_usd as _connector_tcp_usd

_connector_tcp_usd.PRECONTACT_ALIGNMENT_ONLY = False

from cable import scale_aware_cable_mount as _scale_aware_cable_mount

_scale_aware_cable_mount.PRECONTACT_ALIGNMENT_ONLY = False
```

The settled runtime import remains after those assignments.

- [ ] **Step 6: Run the cable checkpoint**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_affine_root_geometry \
  tests.test_cable_geometry \
  tests.test_connector_tcp \
  tests.test_scale_aware_cable_mount \
  tests.test_tail_preshape \
  tests.test_cable_mount_contract \
  tests.test_connector_tcp_runtime_wiring
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add cable tests *_runtime.py
git add -u .
git commit -m "refactor: group single rack cable modules"
```

---

### Task 5: Move Pure Control and Insertion State Machines

**Files moved to `control/`**

```text
insertion.py
settled_insertion.py
insertion_target_trim.py
plug_axis_insertion.py
orientation_hold.py
handoff_position_hold.py
stereo_handoff.py
precontact_alignment.py
validation_window.py
tool_goal_trim.py
```

- [ ] **Step 1: Make `test_two_stage_insertion` require control package paths**

Its imports must source insertion classes from `control.insertion` and `TrimmedConsecutivePoseInsertionController` from `control.insertion_target_trim` while preserving the existing symbol list. Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_two_stage_insertion
```

Expected before the move: import failure.

- [ ] **Step 2: Move all ten control modules**

```bash
git mv insertion.py control/insertion.py
git mv settled_insertion.py control/settled_insertion.py
git mv insertion_target_trim.py control/insertion_target_trim.py
git mv plug_axis_insertion.py control/plug_axis_insertion.py
git mv orientation_hold.py control/orientation_hold.py
git mv handoff_position_hold.py control/handoff_position_hold.py
git mv stereo_handoff.py control/stereo_handoff.py
git mv precontact_alignment.py control/precontact_alignment.py
git mv validation_window.py control/validation_window.py
git mv tool_goal_trim.py control/tool_goal_trim.py
```

- [ ] **Step 3: Convert control sibling imports by prefix only**

Examples of required module paths:

```text
control.insertion
control.settled_insertion
control.insertion_target_trim
control.plug_axis_insertion
control.orientation_hold
control.handoff_position_hold
control.stereo_handoff
control.precontact_alignment
control.validation_window
control.tool_goal_trim
```

Preserve every imported symbol and every controller body unchanged.

- [ ] **Step 4: Update still-root runtime consumers**

Change their module qualifiers to `control.` while preserving imported symbols. In particular:

```python
from control.settled_insertion import ConsecutivePoseInsertionController
from control.insertion_target_trim import TrimmedConsecutivePoseInsertionController
from control.plug_axis_insertion import ExplicitInsertionAxisAdapter
from control.handoff_position_hold import update_handoff_position_command
```

- [ ] **Step 5: Update the pure-control tests**

Migrate imports in:

```text
test_partial_insertion.py
test_consecutive_pose_insertion.py
test_fine_insertion_settling.py
test_two_stage_insertion.py
test_insertion_orientation_guard.py
test_handoff_position_hold.py
test_orientation_hold.py
test_plug_axis_insertion.py
test_validation_window.py
test_stereo_handoff.py
test_precontact_alignment.py
```

Do not include `test_tool_goal_trim.py` in this checkpoint because it inspects runtime file locations; it is migrated and run in Task 6.

- [ ] **Step 6: Run the pure-controller checkpoint**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_partial_insertion \
  tests.test_consecutive_pose_insertion \
  tests.test_fine_insertion_settling \
  tests.test_two_stage_insertion \
  tests.test_insertion_orientation_guard \
  tests.test_handoff_position_hold \
  tests.test_orientation_hold \
  tests.test_plug_axis_insertion \
  tests.test_validation_window \
  tests.test_stereo_handoff \
  tests.test_precontact_alignment
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add control tests *_runtime.py
git add -u .
git commit -m "refactor: group single rack control modules"
```

---

### Task 6: Move Runtime Wrappers and Remove the `cable_runtime` Collision

**Moves**

```text
cable_runtime.py                -> runtime/cable_runtime_base.py
cable_runtime/__init__.py       -> runtime/cable_runtime.py
angled_hand_runtime.py          -> runtime/angled_hand_runtime.py
stereo_handoff_runtime.py       -> runtime/stereo_handoff_runtime.py
settled_stereo_handoff_runtime.py -> runtime/settled_stereo_handoff_runtime.py
handoff_position_hold_runtime.py  -> runtime/handoff_position_hold_runtime.py
full_insertion_base_runtime.py    -> runtime/full_insertion_base_runtime.py
full_insertion_runtime.py         -> runtime/full_insertion_runtime.py
precontact_runtime.py             -> runtime/precontact_runtime.py
```

- [ ] **Step 1: Make a runtime wiring test require the final paths**

In `tests/test_precontact_runtime_wiring.py`, define:

```python
RUNTIME_ROOT = ROOT / "runtime"
```

and read `full_insertion_runtime.py`, `handoff_position_hold_runtime.py`, and `full_insertion_base_runtime.py` from `RUNTIME_ROOT`. Change source-string expectations to `runtime.` package imports. Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_precontact_runtime_wiring
```

Expected before the move: failure because final runtime paths do not exist.

- [ ] **Step 2: Move the base and live cable runtime implementations**

```bash
git mv cable_runtime.py runtime/cable_runtime_base.py
git mv cable_runtime/__init__.py runtime/cable_runtime.py
```

- [ ] **Step 3: Remove the dynamic import loader from the live facade**

In `runtime/cable_runtime.py`, delete the `importlib.util`/`Path` loader that points back to root `cable_runtime.py`. Replace it with:

```python
from runtime.cable_runtime_base import (
    CableMountedSimulationRuntime as _BaseCableMountedSimulationRuntime,
)
```

Keep this inheritance unchanged:

```python
class CableMountedSimulationRuntime(_BaseCableMountedSimulationRuntime):
```

Do not alter the class body.

- [ ] **Step 4: Move the seven remaining runtime wrappers**

```bash
git mv angled_hand_runtime.py runtime/angled_hand_runtime.py
git mv stereo_handoff_runtime.py runtime/stereo_handoff_runtime.py
git mv settled_stereo_handoff_runtime.py runtime/settled_stereo_handoff_runtime.py
git mv handoff_position_hold_runtime.py runtime/handoff_position_hold_runtime.py
git mv full_insertion_base_runtime.py runtime/full_insertion_base_runtime.py
git mv full_insertion_runtime.py runtime/full_insertion_runtime.py
git mv precontact_runtime.py runtime/precontact_runtime.py
```

- [ ] **Step 5: Convert runtime-to-runtime imports by prefix only**

The inheritance chain must resolve through these module paths:

```text
runtime.cable_runtime
runtime.angled_hand_runtime
runtime.stereo_handoff_runtime
runtime.settled_stereo_handoff_runtime
runtime.full_insertion_base_runtime
runtime.handoff_position_hold_runtime
runtime.full_insertion_runtime
```

Known exact class imports that must be preserved with only module prefixes changed include:

```python
from runtime.angled_hand_runtime import AngledHandCableRuntime
```

```python
from runtime.stereo_handoff_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
```

```python
from runtime.settled_stereo_handoff_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
```

```python
from runtime.full_insertion_base_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
```

```python
from runtime.handoff_position_hold_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
```

`runtime/settled_stereo_handoff_runtime.py` must import the base live facade as:

```python
from runtime.cable_runtime import (
    CableMountedSimulationRuntime as _CableMountedSimulationRuntime,
)
```

- [ ] **Step 6: Preserve the precontact patch ordering**

In `runtime/full_insertion_base_runtime.py`, this sequence remains before importing the settled runtime:

```python
from cable import connector_tcp_usd as _connector_tcp_usd

_connector_tcp_usd.PRECONTACT_ALIGNMENT_ONLY = False

from cable import scale_aware_cable_mount as _scale_aware_cable_mount

_scale_aware_cable_mount.PRECONTACT_ALIGNMENT_ONLY = False
```

Then import the settled runtime from `runtime.settled_stereo_handoff_runtime`. Do not move that import above the assignments.

- [ ] **Step 7: Finalize root `main.py` imports**

After `SimulationApp` starts, `main.py` must contain:

```python
from runtime.full_insertion_runtime import (
    AngledHandStereoHandoffRuntime as CableMountedSimulationRuntime,
)
from debug import DebugOutputs
from vision.live_control_projective import refine_live_observation
from vision.perception import YOLOEPortDetector, process_stereo_port
from sim import warn
```

- [ ] **Step 8: Update all runtime/path wiring tests**

Use:

```python
RUNTIME_ROOT = ROOT / "runtime"
```

Migrate path and import-string assertions in:

```text
test_runtime_wiring.py
test_two_stage_runtime_wiring.py
test_angled_hand_runtime_wiring.py
test_precontact_runtime_wiring.py
test_handoff_position_hold_runtime_wiring.py
test_stereo_handoff_runtime_wiring.py
test_orientation_hold_runtime_wiring.py
test_startup_geometry_settle.py
test_connector_tcp_runtime_wiring.py
test_tool_goal_trim.py
```

`test_runtime_wiring.py` must inspect base GPU/mount behavior in `runtime/cable_runtime_base.py` and live PhysX/insertion behavior in `runtime/cable_runtime.py`.

`test_two_stage_runtime_wiring.py` must read `runtime/cable_runtime.py`, not the removed `cable_runtime/__init__.py`.

Do not weaken any behavioral assertion; change only file paths and expected module qualifiers.

- [ ] **Step 9: Run the runtime checkpoint**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_runtime_wiring \
  tests.test_two_stage_runtime_wiring \
  tests.test_angled_hand_runtime_wiring \
  tests.test_precontact_runtime_wiring \
  tests.test_handoff_position_hold_runtime_wiring \
  tests.test_stereo_handoff_runtime_wiring \
  tests.test_orientation_hold_runtime_wiring \
  tests.test_startup_geometry_settle \
  tests.test_connector_tcp_runtime_wiring \
  tests.test_tool_goal_trim
```

Expected: `OK`.

- [ ] **Step 10: Prove the old name collision is gone**

```bash
test ! -f cable_runtime.py
test ! -e cable_runtime
python3 - <<'PY'
from pathlib import Path
assert Path("runtime/cable_runtime_base.py").is_file()
assert Path("runtime/cable_runtime.py").is_file()
print("cable runtime collision removed")
PY
```

Expected:

```text
cable runtime collision removed
```

- [ ] **Step 11: Commit**

```bash
git add main.py runtime tests
git add -u .
git commit -m "refactor: group single rack runtime stack"
```

---

### Task 7: Move Offline Ground Truth, Enforce Root Cleanliness, and Update Current README Paths

**Files**
- Move: `automatic_port_ground_truth.py` to `benchmarks/automatic_port_ground_truth.py`
- Modify: `tests/test_automatic_port_ground_truth.py`
- Modify: `tests/test_benchmark.py`
- Modify: `tests/test_ground_truth.py` if it references a moved source path
- Modify: `tests/test_repo_cleanliness.py`
- Modify: `README.md`

- [ ] **Step 1: Move automatic ground-truth support**

```bash
git mv automatic_port_ground_truth.py benchmarks/automatic_port_ground_truth.py
```

Change `tests/test_automatic_port_ground_truth.py` to import its existing symbol list from `benchmarks.automatic_port_ground_truth`.

- [ ] **Step 2: Encode the final root/package layout in `test_repo_cleanliness.py`**

Update `PRODUCTION_FILES` to package-aware paths. It must include root `main.py`, `config.py`, `sim.py`, representative current vision/runtime production files, the existing benchmark files, and existing benchmark tools.

Add this exact top-level test:

```python
    def test_top_level_python_files_are_intentionally_small(self):
        actual = {path.name for path in ROOT.glob("*.py")}
        self.assertEqual(actual, {"main.py", "sim.py", "config.py", "debug.py"})
```

Add `FORBIDDEN_FLAT_PATHS` containing every file moved from the root in Tasks 2 through 7 and test it with:

```python
    def test_flat_implementation_layout_does_not_return(self):
        for relative in FORBIDDEN_FLAT_PATHS:
            self.assertFalse((ROOT / relative).exists(), relative)
```

Retain the pre-existing legacy/dead path checks and the read-only `.gitignore` assertions.

- [ ] **Step 3: Update current README paths only**

Update current file/architecture references to `vision/`, `robot/`, `cable/`, `control/`, and `runtime/`. Preserve this command exactly:

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" main.py
```

Do not change behavioral numbers or rewrite historical planning documents.

- [ ] **Step 4: Run the support/cleanliness checkpoint**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_repo_cleanliness \
  tests.test_benchmark \
  tests.test_ground_truth \
  tests.test_automatic_port_ground_truth
```

Expected: `OK`.

- [ ] **Step 5: Prove the root is visibly clean**

```bash
find . -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sort
```

Expected exactly:

```text
config.py
debug.py
main.py
sim.py
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks tests README.md
git add -u .
git commit -m "refactor: finalize single rack layout"
```

---

### Task 8: Static and Python Regression Qualification

- [ ] **Step 1: Compile the final Python layout**

```bash
~/isaacsim/python.sh -m compileall -q \
  main.py sim.py config.py debug.py \
  vision robot cable control runtime benchmarks tools tests
```

Expected: exit status 0.

- [ ] **Step 2: Run the same focused qualification group used before this reorganization**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_repo_cleanliness \
  tests.test_live_control \
  tests.test_front_rim_plane_runtime_wiring \
  tests.test_plane_rectified_front_lip \
  tests.test_plane_rectified_runtime_wiring \
  tests.test_front_lip_left_bezel_rejection \
  tests.test_front_lip_search_calibration \
  tests.test_visible_front_lip_calibration \
  tests.test_visible_front_lip_geometry \
  tests.test_handoff_position_hold \
  tests.test_handoff_position_hold_runtime_wiring \
  tests.test_precontact_runtime_wiring \
  tests.test_startup_geometry_settle \
  tests.test_two_stage_insertion
```

Expected: `OK`. The previous focused set contained 55 passing tests. The final count may increase because this refactor deliberately adds layout-cleanliness test methods; success is `OK`, not preservation of the old count.

- [ ] **Step 3: Run the additional packaging-sensitive tests**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_runtime_wiring \
  tests.test_two_stage_runtime_wiring \
  tests.test_angled_hand_runtime_wiring \
  tests.test_stereo_handoff_runtime_wiring \
  tests.test_orientation_hold_runtime_wiring \
  tests.test_connector_tcp_runtime_wiring \
  tests.test_affine_root_geometry \
  tests.test_cable_geometry \
  tests.test_connector_tcp \
  tests.test_scale_aware_cable_mount \
  tests.test_tail_preshape \
  tests.test_angled_grasp_centering \
  tests.test_hand_plug_geometry \
  tests.test_host_array_bridge \
  tests.test_articulation_host_bridge \
  tests.test_partial_insertion \
  tests.test_consecutive_pose_insertion \
  tests.test_fine_insertion_settling \
  tests.test_insertion_orientation_guard \
  tests.test_orientation_hold \
  tests.test_plug_axis_insertion \
  tests.test_validation_window \
  tests.test_stereo_handoff \
  tests.test_tool_goal_trim \
  tests.test_precontact_alignment \
  tests.test_benchmark \
  tests.test_ground_truth \
  tests.test_automatic_port_ground_truth
```

Expected: `OK`.

- [ ] **Step 4: Check for accidental flat imports in production code**

Use a Python AST scan rather than a text grep. The scan must inspect `main.py`, `sim.py`, and every `.py` file directly under `vision/`, `robot/`, `cable/`, `control/`, and `runtime/`. Fail if an `Import` or `ImportFrom` references a moved root module name without its package prefix.

The moved root names are the basenames listed in Tasks 2 through 7. The only allowed unqualified local imports are the intentionally root-resident modules `config`, `debug`, and `sim`.

Expected terminal result from the scan:

```text
no accidental flat production imports
```

- [ ] **Step 5: Inspect the diff for behavioral drift**

```bash
git diff --stat 636d4f8a79f021b8e3c73f4dfc726c9148654534...HEAD -- single_rack_cv
git diff 636d4f8a79f021b8e3c73f4dfc726c9148654534...HEAD -- config.py
```

`config.py` must have no behavioral change. For moved production files, review the diff as rename/import-path changes; reject any unexpected threshold, constant, controller equation, or safety-logic edit.

- [ ] **Step 6: Commit packaging-only corrections if verification exposes any**

If corrections are required:

```bash
git add .
git commit -m "test: fix single rack package wiring"
```

If no correction is required, do not create an empty commit.

---

### Task 9: Open the PR and Run the Real Isaac Workstation Qualification

- [ ] **Step 1: Confirm scope and clean branch state**

```bash
git status --short
git diff --name-only 636d4f8a79f021b8e3c73f4dfc726c9148654534...HEAD
```

Every changed path must begin with `single_rack_cv/` when run from the repository root. Working-tree status must be empty before qualification.

- [ ] **Step 2: Open PR**

Use title:

```text
refactor: organize single_rack_cv by responsibility
```

The PR body must state that this is behavior-preserving, the exact launch command is unchanged, the root now has only four Python files, the `cable_runtime` collision is removed, and merge is blocked on one complete 48/48 Isaac run.

- [ ] **Step 3: Pull the branch on the Isaac workstation**

```bash
cd ~/Isaacsim-Scripts
git fetch origin
git switch refactor/single-rack-layout
git pull --ff-only
cd single_rack_cv
git status --short
```

Expected: no status output.

- [ ] **Step 4: Re-run the focused qualification tests on the workstation**

Run the exact Task 8 Step 2 command. Expected: `OK`.

- [ ] **Step 5: Launch the application with the unchanged command**

```bash
~/isaacsim/python.sh main.py 2>&1 | tee camera_output/layout_reorg_qualification.txt
```

Do not add `PYTHONPATH`, a module-mode launch, or a different working directory.

- [ ] **Step 6: Require complete end-to-end success**

Acceptance requires:

```text
RGB QUALIFIED PORT-POSE ALIGNMENT COMPLETE
FROZEN HANDOFF POSITION HOLD ACTIVE
TWO-STAGE PORT ENTRY STARTED
settled command: 48/48
PARTIAL INSERTION COMPLETE
```

Also require:
- physical handoff error at or below 0.300 mm;
- final calibrated-line lateral deviation at or below 0.500 mm;
- final orientation error at or below 1.000 degree;
- approximately +10 mm terminal depth inside the opening;
- mount/fixed-joint and built-in attachment valid;
- no `Traceback`, `FATAL ERROR`, unexpected abort, or timeout.

The pre-refactor post-merge reference run reached +9.892 mm actual depth, 0.190 mm lateral deviation, and 0.049606 degree orientation error. Those exact values are a reference, not a new tighter gate.

- [ ] **Step 7: Save a compact log summary**

```bash
grep -E \
"RGB FRONT LIP WIDTH PRIOR|QUALIFIED|HANDOFF POSITION HOLD|ALIGNMENT COMPLETE|TWO-STAGE PORT ENTRY|48/48|PARTIAL INSERTION COMPLETE|lateral|orientation|FATAL ERROR|Traceback|ABORT" \
camera_output/layout_reorg_qualification.txt | tail -n 300
```

- [ ] **Step 8: Apply the rollback rule on any failure**

If the branch fails while baseline `636d4f8...` is known-good, fix only the packaging/import/path defect. Do not compensate by relaxing a gate, changing calibration, or changing controller behavior. If the packaging defect cannot be isolated, abandon the branch and keep qualified `main`.

---

### Task 10: Merge Only the Exact Qualified Revision

- [ ] **Step 1: Freeze the branch after the successful Isaac run**

Any new commit makes the workstation qualification stale and requires re-running Task 9 Steps 4 through 7.

- [ ] **Step 2: Record the exact tested revision**

```bash
git rev-parse HEAD
git status --short
```

The SHA must match the PR head and status must be empty.

- [ ] **Step 3: Merge the PR to `main`**

Merge only the exact tested head.

- [ ] **Step 4: Pull `main` and verify the final root**

```bash
cd ~/Isaacsim-Scripts
git switch main
git pull --ff-only
cd single_rack_cv
find . -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sort
```

Expected exactly:

```text
config.py
debug.py
main.py
sim.py
```

The canonical command remains:

```bash
~/isaacsim/python.sh main.py
```

No further refactor or cleanup belongs in this change.
