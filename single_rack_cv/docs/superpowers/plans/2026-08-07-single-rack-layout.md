# single_rack_cv Layout Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `single_rack_cv/` into responsibility-based Python packages while preserving the exact `~/isaacsim/python.sh main.py` launch contract and every validated perception, handoff, cable, and insertion behavior.

**Architecture:** Keep `main.py`, `sim.py`, `config.py`, `debug.py`, and `README.md` at the root. Move implementation modules into `vision/`, `cable/`, `control/`, `robot/`, and `runtime/`, using explicit package imports and no `sys.path` hacks. Remove the current `cable_runtime.py` / `cable_runtime/` ambiguity by splitting the base runtime into `runtime/cable_runtime_base.py` and the live facade into `runtime/cable_runtime.py`.

**Tech Stack:** Python 3 via Isaac Sim 6.0.0 `python.sh`, NVIDIA Isaac Sim, OpenUSD/PXR, NumPy, OpenCV, PyTorch, `unittest`, Git/GitHub.

## Global Constraints

- Modify **only** paths below `single_rack_cv/`.
- Preserve the exact launch command:

  ```bash
  cd ~/Isaacsim-Scripts/single_rack_cv
  ~/isaacsim/python.sh main.py
  ```

- Keep `main.py`, `sim.py`, `config.py`, `debug.py`, and `README.md` at the root.
- Do not change perception algorithms, thresholds, camera geometry, cable mounting, ToolCenter geometry, handoff logic, insertion targets, safety gates, logging semantics, or benchmark acceptance criteria.
- Do not add runtime `sys.path` manipulation.
- Preserve the 12.9 mm visible-front-lip width prior, five bounded width hypotheses, 0.5 mm independent-eye disagreement gate, and 0.05 mm/pixel rectification.
- Preserve the camera-derived 50 mm handoff goal and 0.300 mm physical handoff completion tolerance.
- Preserve insertion-only calibration `[0.0, -0.00030, -0.00045]` m.
- Preserve the 48-command insertion schedule, 0.500 mm lateral limit, 1.000 degree orientation limit, mount integrity, attachment integrity, IK preflight, and timeout gates.
- Do not delete tests because import paths change.
- Do not modify root `.gitignore`; `tests/test_repo_cleanliness.py` may continue reading it.
- Historical planning/spec documents under `single_rack_cv/docs/` are records and should not be mass-rewritten for package paths.
- The qualified `main` commit `636d4f8a79f021b8e3c73f4dfc726c9148654534` is the rollback baseline until the reorganization passes both tests and a complete Isaac Sim run.

---

## Locked Final File Structure

```text
single_rack_cv/
├── main.py
├── sim.py
├── config.py
├── debug.py
├── README.md
├── vision/
│   ├── __init__.py
│   ├── perception.py
│   ├── stereo_geometry.py
│   ├── front_plane.py
│   ├── live_control.py
│   ├── live_control_projective.py
│   ├── aperture_center.py
│   ├── outer_bezel_center.py
│   ├── outer_bezel_projective_center.py
│   ├── front_lip_calibration.py
│   ├── plane_rectified_types.py
│   ├── plane_rectified_geometry.py
│   ├── plane_rectified_fit_utils.py
│   ├── plane_rectified_fitting.py
│   ├── plane_rectified_width_hypotheses.py
│   └── plane_rectified_front_lip.py
├── cable/
│   ├── __init__.py
│   ├── cable_geometry.py
│   ├── cable_mount.py
│   ├── scale_aware_cable_mount.py
│   ├── connector_tcp.py
│   ├── connector_tcp_usd.py
│   ├── tail_preshape.py
│   └── affine_root_geometry.py
├── control/
│   ├── __init__.py
│   ├── insertion.py
│   ├── settled_insertion.py
│   ├── insertion_target_trim.py
│   ├── plug_axis_insertion.py
│   ├── orientation_hold.py
│   ├── handoff_position_hold.py
│   ├── stereo_handoff.py
│   ├── precontact_alignment.py
│   ├── validation_window.py
│   └── tool_goal_trim.py
├── robot/
│   ├── __init__.py
│   ├── angled_hand_config.py
│   ├── angled_grasp_centering.py
│   ├── hand_plug_geometry.py
│   ├── host_array_bridge.py
│   └── articulation_host_bridge.py
├── runtime/
│   ├── __init__.py
│   ├── cable_runtime_base.py
│   ├── cable_runtime.py
│   ├── angled_hand_runtime.py
│   ├── stereo_handoff_runtime.py
│   ├── settled_stereo_handoff_runtime.py
│   ├── handoff_position_hold_runtime.py
│   ├── full_insertion_base_runtime.py
│   ├── full_insertion_runtime.py
│   └── precontact_runtime.py
├── benchmarks/
│   ├── automatic_port_ground_truth.py
│   └── ... existing benchmark files ...
├── tools/
├── tests/
├── docs/
└── assets/
```

`aperture_center.py` was not named in the first sketch but is active vision support and belongs in `vision/`. `automatic_port_ground_truth.py` is offline benchmark/ground-truth support and belongs in `benchmarks/`. These are location corrections only; their behavior is unchanged.

---

### Task 1: Establish Package Boundaries and Make the Layout Contract Testable

**Files:**
- Create: `single_rack_cv/vision/__init__.py`
- Create: `single_rack_cv/cable/__init__.py`
- Create: `single_rack_cv/control/__init__.py`
- Create: `single_rack_cv/robot/__init__.py`
- Create: `single_rack_cv/runtime/__init__.py`
- Modify: `single_rack_cv/tests/test_repo_cleanliness.py`

**Interfaces:**
- Consumes: current root layout and the existing repository cleanliness test.
- Produces: five importable package namespaces with intentionally minimal `__init__.py` files and a test that encodes the approved root/package contract.

- [ ] **Step 1: Add a failing package-boundary test before creating the packages**

Add this test to `RepositoryCleanlinessTests`:

```python
    def test_responsibility_packages_exist(self):
        for package in ("vision", "cable", "control", "robot", "runtime"):
            init_path = ROOT / package / "__init__.py"
            self.assertTrue(init_path.is_file(), init_path)
```

Do not yet add the package files.

- [ ] **Step 2: Run the new test and verify the expected failure**

Run from `single_rack_cv/`:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_repo_cleanliness.RepositoryCleanlinessTests.test_responsibility_packages_exist
```

Expected result: `FAIL`, with the first missing package `__init__.py` reported.

- [ ] **Step 3: Create minimal package initializers**

Each of these files must contain only a short package docstring and no eager imports:

```python
"""single_rack_cv vision package."""
```

Use the same pattern with `cable`, `control`, `robot`, and `runtime` substituted appropriately. Do not import Isaac Sim, OpenCV, PyTorch, or sibling modules from these `__init__.py` files.

- [ ] **Step 4: Run the package-boundary test again**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_repo_cleanliness.RepositoryCleanlinessTests.test_responsibility_packages_exist
```

Expected result: `OK`.

- [ ] **Step 5: Commit the package skeleton**

```bash
git add single_rack_cv/vision/__init__.py \
        single_rack_cv/cable/__init__.py \
        single_rack_cv/control/__init__.py \
        single_rack_cv/robot/__init__.py \
        single_rack_cv/runtime/__init__.py \
        single_rack_cv/tests/test_repo_cleanliness.py
git commit -m "refactor: add single rack package boundaries"
```

---

### Task 2: Move the Vision Stack as One Cohesive Unit

**Files:**
- Move to `single_rack_cv/vision/`: `perception.py`, `stereo_geometry.py`, `front_plane.py`, `live_control.py`, `live_control_projective.py`, `aperture_center.py`, `outer_bezel_center.py`, `outer_bezel_projective_center.py`, `front_lip_calibration.py`, `plane_rectified_types.py`, `plane_rectified_geometry.py`, `plane_rectified_fit_utils.py`, `plane_rectified_fitting.py`, `plane_rectified_width_hypotheses.py`, `plane_rectified_front_lip.py`
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/sim.py`
- Modify vision consumers under `single_rack_cv/benchmarks/` and `single_rack_cv/tools/`
- Modify affected vision tests and vision wiring tests under `single_rack_cv/tests/`

**Interfaces:**
- Consumes: `config.py`, `debug.py`, camera/runtime objects from `sim.py`, NumPy/OpenCV/PyTorch.
- Produces: `vision.perception`, `vision.front_plane`, `vision.live_control_projective`, and the plane-rectified front-lip API used by `main.py` and tests.

- [ ] **Step 1: Update one test first so the package path is required**

Change the imports at the top of `tests/test_plane_rectified_front_lip.py` to:

```python
from vision.plane_rectified_fitting import fit_rectified_front_lip
from vision.plane_rectified_geometry import build_plane_frame
from vision.plane_rectified_types import PlaneFrame, RectifiedEye
```

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_plane_rectified_front_lip
```

Expected result before the move: import failure because the implementations are not yet under `vision/`.

- [ ] **Step 2: Move the complete vision file set using `git mv`**

From `single_rack_cv/`:

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

- [ ] **Step 3: Convert intra-vision imports to explicit package imports**

For every moved vision file, change imports such as:

```python
from plane_rectified_types import PlaneFrame
from stereo_geometry import triangulate_pixel_pair
from front_plane import estimate_front_plane
```

to:

```python
from vision.plane_rectified_types import PlaneFrame
from vision.stereo_geometry import triangulate_pixel_pair
from vision.front_plane import estimate_front_plane
```

Specific dependencies that must remain intact:

```python
# vision/plane_rectified_geometry.py
from vision.plane_rectified_types import (
    DEFAULT_RECTIFIED_RESOLUTION_M,
    PlaneFrame,
    RectifiedEye,
    _unit,
)

# vision/plane_rectified_fitting.py
from vision.plane_rectified_types import (
    FrontLipFit,
    MAX_EDGE_REPROJECTION_PX,
    MAX_OPPOSITE_EDGE_ANGLE_DEG,
    RectifiedEye,
)
from vision.plane_rectified_fit_utils import (...)

# vision/plane_rectified_width_hypotheses.py
from vision.plane_rectified_fitting import fit_rectified_front_lip

# vision/plane_rectified_front_lip.py
from vision.plane_rectified_geometry import (...)
from vision.plane_rectified_fitting import _fit_joint_front_lip
from vision.plane_rectified_types import (...)
from vision.plane_rectified_width_hypotheses import fit_rectified_front_lip_width_prior

# vision/live_control_projective.py
from vision.outer_bezel_projective_center import (...)
```

Do not modify numerical constants or algorithms while changing these imports.

- [ ] **Step 4: Update the two root production consumers**

In `main.py`, the production imports must become:

```python
from config import CONFIG
from vision.front_lip_calibration import (
    VISIBLE_FRONT_LIP_HEIGHT_M,
    VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
    VISIBLE_FRONT_LIP_WIDTH_M,
)
```

and after `SimulationApp` starts:

```python
from runtime.full_insertion_runtime import (
    AngledHandStereoHandoffRuntime as CableMountedSimulationRuntime,
)
from debug import DebugOutputs
from vision.live_control_projective import refine_live_observation
from vision.perception import YOLOEPortDetector, process_stereo_port
from sim import warn
```

The `runtime.full_insertion_runtime` import will not resolve until Task 6; keep the branch in-progress and use task-specific tests rather than `main.py` execution during intermediate migration.

In `sim.py`, replace its flat `perception` imports with:

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

- [ ] **Step 5: Update benchmark/tool imports that consume vision**

At minimum, update `benchmarks/front_plane_benchmark.py` from:

```python
from front_plane import ...
```

to:

```python
from vision.front_plane import ...
```

Update any other `benchmarks/` or `tools/` source that imports `perception`, `front_plane`, `stereo_geometry`, or other moved vision modules. Preserve all benchmark gates and dataset paths.

- [ ] **Step 6: Update vision tests and literal path assertions**

Convert direct imports to `vision.*` in all affected tests, including:

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
```

For tests that read source files, change root paths, for example:

```python
VISION_ROOT = ROOT / "vision"
source = (VISION_ROOT / "plane_rectified_front_lip.py").read_text(encoding="utf-8")
```

Update string-wiring assertions to expect package imports, for example:

```python
self.assertIn(
    "from vision.outer_bezel_projective_center import",
    source,
)
```

and:

```python
self.assertIn(
    "from vision.live_control_projective import refine_live_observation",
    main_source,
)
```

For `test_visible_front_lip_calibration.py`, explicitly expect:

```python
self.assertIn("from vision.front_lip_calibration import (", source)
```

- [ ] **Step 7: Run the vision-focused regression group**

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

Expected result: `OK`. The workstation-fixture test may be run separately only when its fixture environment variable/data is available.

- [ ] **Step 8: Commit the vision migration**

```bash
git add single_rack_cv/main.py single_rack_cv/sim.py \
        single_rack_cv/vision single_rack_cv/benchmarks \
        single_rack_cv/tools single_rack_cv/tests
git commit -m "refactor: group single rack vision modules"
```

---

### Task 3: Move Cable Geometry, Mounting, TCP, and Tail Support

**Files:**
- Move to `single_rack_cv/cable/`: `cable_geometry.py`, `cable_mount.py`, `scale_aware_cable_mount.py`, `connector_tcp.py`, `connector_tcp_usd.py`, `tail_preshape.py`, `affine_root_geometry.py`
- Modify cable consumers in runtime modules temporarily at their current paths
- Modify cable/geometry tests and wiring tests

**Interfaces:**
- Consumes: `config.py`, PXR/OpenUSD, `sim.py`, robot host bridge after Task 4.
- Produces: `cable.cable_mount.CableMount`, `cable.scale_aware_cable_mount.ScaleAwareCableMount`, connector TCP derivation and cable geometry utilities.

- [ ] **Step 1: Make the cable test import the target package path**

Change `tests/test_cable_geometry.py` to import from:

```python
from cable.cable_geometry import (
    angular_error_deg,
    compute_attachment_bounds,
    compute_world_from_root_for_tip,
    detect_plug_frame,
    matrix_to_quaternion_wxyz,
    rigid_pose_from_affine,
    validate_affine_transform,
    validate_mount_window,
    validate_transform,
)
```

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_cable_geometry
```

Expected result before move: import failure.

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

- [ ] **Step 3: Convert cable-internal imports without changing monkeypatch semantics**

Use explicit package imports. Critical examples:

```python
# cable/connector_tcp_usd.py
from cable.cable_geometry import PlugFrame, validate_transform
from cable.cable_mount import _world_transform
from cable.connector_tcp import (
    InsertionTcpDerivation,
    MeshComponentBounds,
    connected_component_bounds,
    derive_insertion_tcp,
)
```

`cable/scale_aware_cable_mount.py` must import the module object that it patches:

```python
from cable import cable_mount as cable_mount_module
from cable.affine_root_geometry import (
    compute_world_from_root_for_tip_preserving_affine,
)
from cable.cable_geometry import (
    matrix_to_quaternion_wxyz,
    rigid_pose_from_affine,
    validate_affine_transform,
)
from cable.cable_mount import CableMount, _world_transform
from cable.connector_tcp_usd import (
    PRECONTACT_ALIGNMENT_ONLY,
    PRECONTACT_HOLD_OFFSET_M,
    TCP_PROBE_ONLY,
    author_tcp_probe_markers,
    derive_plug_frame_from_mesh,
    log_tcp_derivation,
)
from cable.tail_preshape import preshape_free_hanging_tail
```

Do not replace the `try/finally` monkeypatch/restore pattern.

- [ ] **Step 4: Update cable tests and source-path assertions**

Convert imports in:

```text
test_affine_root_geometry.py
test_cable_geometry.py
test_connector_tcp.py
test_scale_aware_cable_mount.py
test_tail_preshape.py
```

Examples:

```python
from cable.affine_root_geometry import compute_world_from_root_for_tip_preserving_affine
from cable.cable_geometry import rigid_pose_from_affine
from cable.connector_tcp import MeshComponentBounds, connected_component_bounds, derive_insertion_tcp
from cable.scale_aware_cable_mount import _matrix_to_gf_quatf_compatible
from cable.tail_preshape import preshape_free_hanging_tail
```

For source-reading tests use:

```python
CABLE_ROOT = ROOT / "cable"
source = (CABLE_ROOT / "scale_aware_cable_mount.py").read_text(encoding="utf-8")
```

Update `test_connector_tcp_runtime_wiring.py` similarly so all connector/TCP source paths point under `cable/` while `main.py` remains at root.

- [ ] **Step 5: Update current runtime consumers to import `cable.*`**

Before the runtime files are moved in Task 6, update their imports in place. In particular:

```python
from cable.cable_geometry import angular_error_deg
from cable.scale_aware_cable_mount import ScaleAwareCableMount
```

and in `full_insertion_base_runtime.py` preserve this ordering:

```python
from cable import connector_tcp_usd as _connector_tcp_usd

_connector_tcp_usd.PRECONTACT_ALIGNMENT_ONLY = False

from cable import scale_aware_cable_mount as _scale_aware_cable_mount

_scale_aware_cable_mount.PRECONTACT_ALIGNMENT_ONLY = False
```

The later import of the settled runtime must still occur after those two assignments.

- [ ] **Step 6: Run cable-specific tests**

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

Expected result: `OK`.

- [ ] **Step 7: Commit the cable migration**

```bash
git add single_rack_cv/cable single_rack_cv/tests \
        single_rack_cv/*runtime.py
git commit -m "refactor: group single rack cable modules"
```

---

### Task 4: Move Robot and Hand Geometry Helpers

**Files:**
- Move to `single_rack_cv/robot/`: `angled_hand_config.py`, `angled_grasp_centering.py`, `hand_plug_geometry.py`, `host_array_bridge.py`, `articulation_host_bridge.py`
- Modify affected runtime and cable imports
- Modify robot-helper tests

**Interfaces:**
- Consumes: NumPy/PyTorch and `sim.py` geometry conversion helpers where applicable.
- Produces: hand pitch geometry, camera/plug recentering, host-safe array bridges, and articulation wrappers.

- [ ] **Step 1: Make one robot-helper test require the new package**

Change `tests/test_hand_plug_geometry.py` to:

```python
from robot.hand_plug_geometry import (
    compute_angled_hand_pose_preserving_tool,
    expected_camera_baseline_axis_world,
    horizontal_axis_error_deg,
    measure_hand_plug_geometry,
    validate_downward_hand_pitch_deg,
)
```

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_hand_plug_geometry
```

Expected result before move: import failure.

- [ ] **Step 2: Move the five robot/helper files**

```bash
git mv angled_hand_config.py robot/angled_hand_config.py
git mv angled_grasp_centering.py robot/angled_grasp_centering.py
git mv hand_plug_geometry.py robot/hand_plug_geometry.py
git mv host_array_bridge.py robot/host_array_bridge.py
git mv articulation_host_bridge.py robot/articulation_host_bridge.py
```

- [ ] **Step 3: Update robot-internal and cross-package imports**

Examples:

```python
# robot/angled_grasp_centering.py
from robot.hand_plug_geometry import ...
```

Update cable code:

```python
# cable/scale_aware_cable_mount.py
from robot.articulation_host_bridge import HostSafeDofPropertiesArticulation
```

Update runtime code at its current path:

```python
from robot.angled_grasp_centering import (...)
from robot.angled_hand_config import ANGLED_HAND_CONFIG, AngledHandConfig
from robot.hand_plug_geometry import (...)
from robot.host_array_bridge import to_numpy_cpu
```

- [ ] **Step 4: Update robot/helper tests**

Convert imports in:

```text
test_angled_grasp_centering.py
test_hand_plug_geometry.py
test_host_array_bridge.py
test_articulation_host_bridge.py
```

and any source-path assertions in runtime wiring tests so `angled_hand_config.py` is read from `ROOT / "robot" / "angled_hand_config.py"`.

- [ ] **Step 5: Run robot/helper tests**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_angled_grasp_centering \
  tests.test_hand_plug_geometry \
  tests.test_host_array_bridge \
  tests.test_articulation_host_bridge
```

Expected result: `OK`.

- [ ] **Step 6: Commit the robot/helper migration**

```bash
git add single_rack_cv/robot single_rack_cv/cable \
        single_rack_cv/tests single_rack_cv/*runtime.py
git commit -m "refactor: group single rack robot helpers"
```

---

### Task 5: Move Pure Control and Insertion State Machines

**Files:**
- Move to `single_rack_cv/control/`: `insertion.py`, `settled_insertion.py`, `insertion_target_trim.py`, `plug_axis_insertion.py`, `orientation_hold.py`, `handoff_position_hold.py`, `stereo_handoff.py`, `precontact_alignment.py`, `validation_window.py`, `tool_goal_trim.py`
- Modify affected runtime imports
- Modify control tests

**Interfaces:**
- Consumes: NumPy and dataclasses; no packaging-time Isaac Sim side effects.
- Produces: insertion controllers/events, handoff qualification/position hold, orientation hold, validation windows, and explicit insertion-axis adapters.

- [ ] **Step 1: Make the core insertion test require `control.insertion`**

Change `tests/test_two_stage_insertion.py` imports to:

```python
from control.insertion import (
    InsertionLimits,
    InsertionPhase,
    InsertionSample,
    InsertionStage,
    PartialInsertionController,
)
from control.insertion_target_trim import TrimmedConsecutivePoseInsertionController
```

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_two_stage_insertion
```

Expected result before move: import failure.

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

- [ ] **Step 3: Convert control-internal imports**

Required examples:

```python
# control/settled_insertion.py
from control.insertion import (...)

# control/insertion_target_trim.py
from control.settled_insertion import ConsecutivePoseInsertionController

# control/plug_axis_insertion.py
from control.insertion import (...)

# control/precontact_alignment.py
from control.insertion import (...)
```

Do not change controller constants, command sequencing, or metrics calculations.

- [ ] **Step 4: Update runtime imports to `control.*`**

Examples:

```python
from control.insertion import (
    InsertionEvent,
    InsertionLimits,
    InsertionPhase,
    InsertionSample,
    PartialInsertionController,
)
from control.settled_insertion import ConsecutivePoseInsertionController
from control.insertion_target_trim import TrimmedConsecutivePoseInsertionController
from control.plug_axis_insertion import ExplicitInsertionAxisAdapter
from control.handoff_position_hold import update_handoff_position_command
from control.stereo_handoff import (...)
from control.orientation_hold import (...)
```

- [ ] **Step 5: Update all pure-control tests**

At minimum migrate imports/path assertions in:

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
test_tool_goal_trim.py
test_precontact_alignment.py
```

`test_tool_goal_trim.py` source paths for production runtime modules should not be finalized until Task 6; after Task 6 they must point under `runtime/`.

- [ ] **Step 6: Run the pure-controller suite**

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
  tests.test_tool_goal_trim \
  tests.test_precontact_alignment
```

Expected result: `OK` except any source-path-only test intentionally waiting for the Task 6 runtime move. If one of those path-only assertions fails, update it in Task 6 rather than creating a compatibility copy at root.

- [ ] **Step 7: Commit the control migration**

```bash
git add single_rack_cv/control single_rack_cv/tests \
        single_rack_cv/*runtime.py
git commit -m "refactor: group single rack control modules"
```

---

### Task 6: Move the Runtime Stack and Eliminate the `cable_runtime` Name Collision

**Files:**
- Move root base `single_rack_cv/cable_runtime.py` to `single_rack_cv/runtime/cable_runtime_base.py`
- Move old package implementation `single_rack_cv/cable_runtime/__init__.py` to `single_rack_cv/runtime/cable_runtime.py`
- Move to `single_rack_cv/runtime/`: `angled_hand_runtime.py`, `stereo_handoff_runtime.py`, `settled_stereo_handoff_runtime.py`, `handoff_position_hold_runtime.py`, `full_insertion_base_runtime.py`, `full_insertion_runtime.py`, `precontact_runtime.py`
- Delete now-empty old `single_rack_cv/cable_runtime/` directory from Git tracking
- Modify `main.py` and all runtime wiring/path tests

**Interfaces:**
- Consumes: top-level `sim.py` and `config.py`; `vision.*`, `cable.*`, `control.*`, `robot.*`.
- Produces: `runtime.full_insertion_runtime.AngledHandStereoHandoffRuntime`, imported by root `main.py` as `CableMountedSimulationRuntime`.

- [ ] **Step 1: Change the main wiring test to require the target runtime package**

Update `tests/test_precontact_runtime_wiring.py` so it reads:

```python
RUNTIME_ROOT = ROOT / "runtime"
source = (ROOT / "main.py").read_text()
export_source = (RUNTIME_ROOT / "full_insertion_runtime.py").read_text()
hold_source = (RUNTIME_ROOT / "handoff_position_hold_runtime.py").read_text()
```

and expects:

```python
self.assertIn("from runtime.full_insertion_runtime import (", source)
self.assertIn("from runtime.handoff_position_hold_runtime import (", export_source)
self.assertIn("from runtime.full_insertion_base_runtime import (", hold_source)
```

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_precontact_runtime_wiring
```

Expected result before the move: path/import assertion failure.

- [ ] **Step 2: Move the base and live cable runtimes to unambiguous filenames**

```bash
git mv cable_runtime.py runtime/cable_runtime_base.py
git mv cable_runtime/__init__.py runtime/cable_runtime.py
```

After the tracked `__init__.py` is moved, the old `cable_runtime/` directory should contain no tracked files.

- [ ] **Step 3: Replace the dynamic base-loader in `runtime/cable_runtime.py`**

Delete the old `importlib.util`, `sys`, `Path`, `_BASE_PATH`, `_SPEC`, and `exec_module` loader block. Replace it with the normal package import:

```python
from runtime.cable_runtime_base import (
    CableMountedSimulationRuntime as _BaseCableMountedSimulationRuntime,
)
```

Retain:

```python
class CableMountedSimulationRuntime(_BaseCableMountedSimulationRuntime):
```

This is the only intended structural change to this facade. All live PhysX/FK behavior, visual-servo overrides, insertion setup, logging, and validation remain unchanged.

- [ ] **Step 4: Move the remaining runtime wrappers**

```bash
git mv angled_hand_runtime.py runtime/angled_hand_runtime.py
git mv stereo_handoff_runtime.py runtime/stereo_handoff_runtime.py
git mv settled_stereo_handoff_runtime.py runtime/settled_stereo_handoff_runtime.py
git mv handoff_position_hold_runtime.py runtime/handoff_position_hold_runtime.py
git mv full_insertion_base_runtime.py runtime/full_insertion_base_runtime.py
git mv full_insertion_runtime.py runtime/full_insertion_runtime.py
git mv precontact_runtime.py runtime/precontact_runtime.py
```

- [ ] **Step 5: Convert runtime-to-runtime imports to explicit package imports**

Required chain:

```python
# runtime/angled_hand_runtime.py
from runtime.cable_runtime import CableMountedSimulationRuntime

# runtime/stereo_handoff_runtime.py
from runtime.angled_hand_runtime import AngledHandCableRuntime

# runtime/settled_stereo_handoff_runtime.py
from runtime.stereo_handoff_runtime import (...)

# runtime/full_insertion_base_runtime.py
from runtime.settled_stereo_handoff_runtime import (...)

# runtime/handoff_position_hold_runtime.py
from runtime.full_insertion_base_runtime import (...)

# runtime/full_insertion_runtime.py
from runtime.handoff_position_hold_runtime import (...)
```

Use `control.*`, `cable.*`, and `robot.*` imports for supporting modules. Keep `from sim import ...` and `from config import ...` top-level because those files intentionally remain at the root.

- [ ] **Step 6: Preserve full-insertion precontact-patch ordering exactly**

In `runtime/full_insertion_base_runtime.py`, this order must remain:

```python
from cable import connector_tcp_usd as _connector_tcp_usd

_connector_tcp_usd.PRECONTACT_ALIGNMENT_ONLY = False

from cable import scale_aware_cable_mount as _scale_aware_cable_mount

_scale_aware_cable_mount.PRECONTACT_ALIGNMENT_ONLY = False

from runtime.settled_stereo_handoff_runtime import (...)
```

Do not move the settled runtime import above the two module-variable assignments.

- [ ] **Step 7: Finalize `main.py` runtime import**

Ensure the post-`SimulationApp` import is exactly package-qualified:

```python
from runtime.full_insertion_runtime import (
    AngledHandStereoHandoffRuntime as CableMountedSimulationRuntime,
)
```

Do not rename the local `CableMountedSimulationRuntime` alias because the rest of `main.py` and logs are already qualified around that interface.

- [ ] **Step 8: Update every runtime wiring test to the new paths/import strings**

Create a common local convention in each test:

```python
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "runtime"
```

Update affected tests including:

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

Key path changes:

```python
RUNTIME_ROOT / "cable_runtime_base.py"
RUNTIME_ROOT / "cable_runtime.py"
RUNTIME_ROOT / "angled_hand_runtime.py"
RUNTIME_ROOT / "settled_stereo_handoff_runtime.py"
RUNTIME_ROOT / "full_insertion_base_runtime.py"
RUNTIME_ROOT / "handoff_position_hold_runtime.py"
RUNTIME_ROOT / "full_insertion_runtime.py"
```

Key import-string changes must expect `runtime.`/`control.`/`cable.` package prefixes. For example:

```python
self.assertIn("from runtime.full_insertion_runtime import", main_source)
self.assertIn("from runtime.handoff_position_hold_runtime import (", export_source)
self.assertIn("from runtime.full_insertion_base_runtime import (", hold_source)
self.assertIn(
    "from control.settled_insertion import ConsecutivePoseInsertionController",
    settled_source,
)
```

`test_two_stage_runtime_wiring.py` must now read `runtime/cable_runtime.py`, not the deleted `cable_runtime/__init__.py`.

`test_runtime_wiring.py` must distinguish:
- base GPU/mount behavior in `runtime/cable_runtime_base.py`
- live PhysX/insertion facade behavior in `runtime/cable_runtime.py`

Do not weaken assertions merely to make them pass.

- [ ] **Step 9: Run all runtime wiring/structure tests**

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

Expected result: `OK`.

- [ ] **Step 10: Verify the old collision is actually gone**

From `single_rack_cv/`:

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

Expected output:

```text
cable runtime collision removed
```

- [ ] **Step 11: Commit the runtime migration**

```bash
git add single_rack_cv/main.py single_rack_cv/runtime \
        single_rack_cv/tests
git add -u single_rack_cv
git commit -m "refactor: group single rack runtime stack"
```

---

### Task 7: Move Offline Ground-Truth Support, Finalize Root Cleanliness, and Update Current Documentation

**Files:**
- Move: `single_rack_cv/automatic_port_ground_truth.py` → `single_rack_cv/benchmarks/automatic_port_ground_truth.py`
- Modify: `single_rack_cv/tests/test_automatic_port_ground_truth.py`
- Modify: `single_rack_cv/tests/test_benchmark.py`
- Modify: `single_rack_cv/tests/test_ground_truth.py` only if a moved module path is referenced
- Modify: `single_rack_cv/tests/test_repo_cleanliness.py`
- Modify: `single_rack_cv/README.md`

**Interfaces:**
- Consumes: the completed package layout from Tasks 1–6.
- Produces: a root with only the four approved Python files, package-aware cleanliness tests, and current README paths matching the final layout.

- [ ] **Step 1: Move offline automatic ground-truth support**

```bash
git mv automatic_port_ground_truth.py benchmarks/automatic_port_ground_truth.py
```

Update `tests/test_automatic_port_ground_truth.py`:

```python
from benchmarks.automatic_port_ground_truth import (
    RaycastGroundTruthConfig,
    RaycastHit,
    build_automatic_ground_truth,
    intersect_ray_with_plane,
    offset_rim_samples_outward,
)
```

Do not move `tools/extract_front_rim_ground_truth.py`; it is already correctly categorized as a tool.

- [ ] **Step 2: Make `test_repo_cleanliness.py` encode the final package layout**

Update `PRODUCTION_FILES` so current production/support paths are package-aware, for example:

```python
PRODUCTION_FILES = (
    ROOT / "main.py",
    ROOT / "config.py",
    ROOT / "sim.py",
    ROOT / "vision" / "front_plane.py",
    ROOT / "vision" / "stereo_geometry.py",
    ROOT / "vision" / "live_control.py",
    ROOT / "vision" / "live_control_projective.py",
    ROOT / "vision" / "plane_rectified_front_lip.py",
    ROOT / "runtime" / "full_insertion_runtime.py",
    ROOT / "runtime" / "handoff_position_hold_runtime.py",
    ROOT / "runtime" / "full_insertion_base_runtime.py",
    ROOT / "runtime" / "settled_stereo_handoff_runtime.py",
    ROOT / "benchmarks" / "front_plane_benchmark.py",
    ROOT / "benchmarks" / "capture_dataset.py",
    ROOT / "tools" / "run_benchmark_isaac.py",
    ROOT / "tools" / "run_benchmark.sh",
    ROOT / "tools" / "generate_ground_truth.py",
)
```

Add an exact root Python allowlist test:

```python
    def test_top_level_python_files_are_intentionally_small(self):
        actual = {path.name for path in ROOT.glob("*.py")}
        self.assertEqual(actual, {"main.py", "sim.py", "config.py", "debug.py"})
```

Add the moved flat implementation paths to a separate `FORBIDDEN_FLAT_PATHS` tuple or to `FORBIDDEN_PATHS`. It must include every implementation intentionally relocated from the root, including:

```python
FORBIDDEN_FLAT_PATHS = (
    "perception.py",
    "stereo_geometry.py",
    "front_plane.py",
    "live_control.py",
    "live_control_projective.py",
    "aperture_center.py",
    "outer_bezel_center.py",
    "outer_bezel_projective_center.py",
    "front_lip_calibration.py",
    "plane_rectified_types.py",
    "plane_rectified_geometry.py",
    "plane_rectified_fit_utils.py",
    "plane_rectified_fitting.py",
    "plane_rectified_width_hypotheses.py",
    "plane_rectified_front_lip.py",
    "cable_geometry.py",
    "cable_mount.py",
    "scale_aware_cable_mount.py",
    "connector_tcp.py",
    "connector_tcp_usd.py",
    "tail_preshape.py",
    "affine_root_geometry.py",
    "insertion.py",
    "settled_insertion.py",
    "insertion_target_trim.py",
    "plug_axis_insertion.py",
    "orientation_hold.py",
    "handoff_position_hold.py",
    "stereo_handoff.py",
    "precontact_alignment.py",
    "validation_window.py",
    "tool_goal_trim.py",
    "angled_hand_config.py",
    "angled_grasp_centering.py",
    "hand_plug_geometry.py",
    "host_array_bridge.py",
    "articulation_host_bridge.py",
    "cable_runtime.py",
    "angled_hand_runtime.py",
    "stereo_handoff_runtime.py",
    "settled_stereo_handoff_runtime.py",
    "handoff_position_hold_runtime.py",
    "full_insertion_base_runtime.py",
    "full_insertion_runtime.py",
    "precontact_runtime.py",
    "automatic_port_ground_truth.py",
)
```

Test it with:

```python
    def test_flat_implementation_layout_does_not_return(self):
        for relative in FORBIDDEN_FLAT_PATHS:
            self.assertFalse((ROOT / relative).exists(), relative)
```

Retain the pre-existing legacy/dead path checks; do not replace them with this new list.

- [ ] **Step 3: Update current README file references only**

Update the current architecture/file references in `README.md` to point to `vision/`, `cable/`, `control/`, `robot/`, and `runtime/`. Preserve:

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" main.py
```

Preserve all current behavioral/safety statements and numerical values. Do not rewrite historical planning documents under `docs/superpowers/`.

- [ ] **Step 4: Update benchmark structure tests for package imports**

In `tests/test_benchmark.py`, change the expected import string from flat `front_plane` to package-qualified vision import:

```python
self.assertIn("from vision.front_plane import", source)
```

Keep every benchmark acceptance-gate assertion unchanged.

- [ ] **Step 5: Run benchmark, ground-truth, and cleanliness structure tests**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_repo_cleanliness \
  tests.test_benchmark \
  tests.test_ground_truth \
  tests.test_automatic_port_ground_truth
```

Expected result: `OK`.

- [ ] **Step 6: Verify the root is visibly clean**

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

Then verify package directories:

```bash
find vision cable control robot runtime -maxdepth 1 -type f -name '*.py' -printf '%h/%f\n' | sort
```

Inspect the output against the locked file structure at the top of this plan.

- [ ] **Step 7: Commit final layout/docs cleanup**

```bash
git add single_rack_cv/benchmarks single_rack_cv/tests \
        single_rack_cv/README.md
git add -u single_rack_cv
git commit -m "refactor: finalize single rack layout"
```

---

### Task 8: Static Verification and Full Python Regression Before Opening a PR

**Files:**
- No new production behavior.
- Fix only packaging/import/path mistakes discovered by these commands.

**Interfaces:**
- Consumes: completed organized branch.
- Produces: evidence that package imports and the focused qualification suite pass before Isaac Sim runtime qualification.

- [ ] **Step 1: Compile every current Python subtree**

From `single_rack_cv/`:

```bash
~/isaacsim/python.sh -m compileall -q \
  main.py sim.py config.py debug.py \
  vision cable control robot runtime benchmarks tools tests
```

Expected: exit status `0` and no syntax errors.

- [ ] **Step 2: Run the same focused qualification suite that passed before the reorganization**

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

Expected: all tests pass. The historical baseline for this focused set is 55 passing tests; if the count changes only because new cleanliness/layout test methods were intentionally added, require `OK` rather than forcing the count back to 55.

- [ ] **Step 3: Run the additional modules directly affected by packaging**

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

- [ ] **Step 4: Search for accidental flat production imports**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path('.')
moved = {
    'perception', 'front_plane', 'stereo_geometry', 'live_control',
    'live_control_projective', 'front_lip_calibration', 'plane_rectified_types',
    'plane_rectified_geometry', 'plane_rectified_fitting',
    'plane_rectified_front_lip', 'cable_geometry', 'cable_mount',
    'scale_aware_cable_mount', 'connector_tcp', 'connector_tcp_usd',
    'tail_preshape', 'insertion', 'settled_insertion', 'insertion_target_trim',
    'plug_axis_insertion', 'orientation_hold', 'handoff_position_hold',
    'stereo_handoff', 'angled_hand_config', 'angled_grasp_centering',
    'hand_plug_geometry', 'host_array_bridge', 'articulation_host_bridge',
    'full_insertion_runtime', 'full_insertion_base_runtime',
    'handoff_position_hold_runtime', 'settled_stereo_handoff_runtime',
    'stereo_handoff_runtime', 'angled_hand_runtime', 'precontact_runtime',
}
violations = []
for path in [root / 'main.py', root / 'sim.py'] + list((root / 'vision').glob('*.py')) + list((root / 'cable').glob('*.py')) + list((root / 'control').glob('*.py')) + list((root / 'robot').glob('*.py')) + list((root / 'runtime').glob('*.py')):
    text = path.read_text(encoding='utf-8')
    for name in moved:
        if f'from {name} import' in text or f'import {name}\n' in text:
            violations.append(f'{path}: flat import {name}')
if violations:
    raise SystemExit('\n'.join(violations))
print('no accidental flat production imports')
PY
```

Expected:

```text
no accidental flat production imports
```

- [ ] **Step 5: Confirm no behavioral constants changed relative to `main`**

Use Git diff to inspect only intentional path/import/documentation changes:

```bash
git diff --stat 636d4f8a79f021b8e3c73f4dfc726c9148654534...HEAD -- single_rack_cv
git diff 636d4f8a79f021b8e3c73f4dfc726c9148654534...HEAD -- \
  single_rack_cv/config.py \
  single_rack_cv/vision/front_lip_calibration.py \
  single_rack_cv/vision/plane_rectified_types.py \
  single_rack_cv/runtime/full_insertion_runtime.py \
  single_rack_cv/runtime/handoff_position_hold_runtime.py
```

Expected: `config.py` has no behavioral changes; the moved files differ from their baseline versions only in imports/path-related text, not constants/controller equations/safety logic.

- [ ] **Step 6: Commit any packaging-only corrections found by verification**

If the previous commands expose import/path mistakes, fix only those mistakes and commit them:

```bash
git add single_rack_cv
git commit -m "test: fix single rack package wiring"
```

If no corrections were needed, do not create an empty commit.

---

### Task 9: Open the Review PR and Perform the Workstation Isaac Qualification

**Files:**
- No new code unless the workstation exposes a packaging defect.

**Interfaces:**
- Consumes: statically verified `refactor/single-rack-layout` branch.
- Produces: PR diff plus one complete real Isaac Sim run using the unchanged launch command.

- [ ] **Step 1: Confirm the branch is clean and only `single_rack_cv/` changed**

```bash
git status --short
git diff --name-only 636d4f8a79f021b8e3c73f4dfc726c9148654534...HEAD \
  | grep -v '^single_rack_cv/' \
  && { echo 'ERROR: change outside single_rack_cv'; exit 1; } \
  || true
```

Expected: clean working tree and no actual changed filename outside `single_rack_cv/`.

- [ ] **Step 2: Open a PR from `refactor/single-rack-layout` to `main`**

PR title:

```text
refactor: organize single_rack_cv by responsibility
```

PR body must state:

```text
- behavior-preserving package/layout refactor only
- exact main.py launch command preserved
- root reduced to main.py, sim.py, config.py, debug.py
- vision/cable/control/robot/runtime responsibilities separated
- cable_runtime file/package collision removed
- no perception, calibration, motion, or safety values changed
- Python/static qualification passed before workstation run
- merge blocked until one complete 48/48 Isaac Sim qualification passes
```

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

- [ ] **Step 4: Re-run the focused qualification suite on the workstation**

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

Expected: `OK`.

- [ ] **Step 5: Run the canonical application command exactly as before**

```bash
~/isaacsim/python.sh main.py 2>&1 | tee camera_output/layout_reorg_qualification.txt
```

Do not prepend a package module invocation, `PYTHONPATH`, `sys.path`, or a different working directory. The unchanged command is itself part of the acceptance test.

- [ ] **Step 6: Require a complete successful physical qualification**

The run is acceptable only if it reaches the same functional end state as the pre-refactor baseline:

```text
RGB QUALIFIED PORT-POSE ALIGNMENT COMPLETE
FROZEN HANDOFF POSITION HOLD ACTIVE
TWO-STAGE PORT ENTRY STARTED
...
settled command: 48/48
PARTIAL INSERTION COMPLETE
```

Acceptance conditions:
- perception qualifies normally;
- camera-derived handoff remains 50 mm;
- physical handoff completion remains at or below 0.300 mm;
- insertion reaches 48/48 commands;
- final depth remains approximately +10 mm inside the opening;
- final calibrated-line lateral deviation is at or below 0.500 mm;
- final orientation error is at or below 1.000 degree;
- fixed-joint/mount and built-in attachment remain valid;
- no `Traceback`, `FATAL ERROR`, unexpected abort, or timeout.

The pre-refactor post-merge reference run achieved +9.892 mm actual depth, 0.190 mm lateral deviation, and 0.049606 degree orientation error. Those exact numbers are not required to repeat; the existing safety/terminal criteria are the requirement.

- [ ] **Step 7: Extract a compact qualification summary**

```bash
grep -E \
"RGB FRONT LIP WIDTH PRIOR|QUALIFIED|HANDOFF POSITION HOLD|ALIGNMENT COMPLETE|TWO-STAGE PORT ENTRY|48/48|PARTIAL INSERTION COMPLETE|lateral|orientation|FATAL ERROR|Traceback|ABORT" \
  camera_output/layout_reorg_qualification.txt | tail -n 300
```

Attach/preserve the full log for review.

- [ ] **Step 8: Use the rollback rule if the application fails**

If the refactor branch fails while `main` at `636d4f8...` remains known-good:
- do **not** relax perception or motion gates;
- do **not** change calibration values;
- identify the missing/incorrect package import, source path, or module-object reference;
- fix that packaging defect on `refactor/single-rack-layout`;
- rerun the relevant tests and complete Isaac qualification from the beginning.

If the packaging regression cannot be isolated, abandon the branch and retain the qualified `main`.

---

### Task 10: Merge Only the Exact Qualified Tree

**Files:**
- No new behavior.

**Interfaces:**
- Consumes: a PR whose exact head tree passed static tests and one complete Isaac run.
- Produces: organized `main` with no behavior change.

- [ ] **Step 1: Freeze changes after the successful workstation run**

After qualification succeeds, do not make cleanup/refactor edits on the PR branch. If any commit is added, the Isaac qualification is stale and must be repeated.

- [ ] **Step 2: Verify the PR head SHA/tree is the same revision tested on the workstation**

Record:

```bash
git rev-parse HEAD
git status --short
```

Expected: the SHA matches the PR head and status is empty.

- [ ] **Step 3: Merge the PR to `main`**

Merge only after the exact tested head is confirmed.

- [ ] **Step 4: Pull `main` locally and verify the launch surface**

```bash
cd ~/Isaacsim-Scripts
git switch main
git pull --ff-only
cd single_rack_cv
find . -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sort
```

Expected:

```text
config.py
debug.py
main.py
sim.py
```

The user-facing command remains:

```bash
~/isaacsim/python.sh main.py
```

No further behavior cleanup is part of this reorganization.
