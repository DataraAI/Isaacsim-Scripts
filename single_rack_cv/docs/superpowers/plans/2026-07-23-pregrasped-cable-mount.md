# Pregrasped Deformable Cable Mount Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start the canonical single-rack visual-servo runtime with the supplied network cable permanently mounted to the Franka hand, with the RJ45 insertion tip coincident with the existing ToolCenter and the cable tail remaining deformable.

**Architecture:** A pure NumPy module computes the RJ45 tip frame, deterministic roll, attachment mask, and one-time cable-root transform. An Isaac-specific `CableMount` component loads the asset, verifies the current Omni Physics deformable schema, enables GPU PhysX, creates a hand-fixed rigid proxy, authors a masked auto deformable attachment, and validates the mount before YOLOE starts. The existing visual-servo control loop remains translation-only and does not gain insertion behavior.

**Tech Stack:** Ubuntu 24.04, Isaac Sim 6.0.0 / Kit 110, Python 3.12, NumPy, OpenUSD (`pxr`), Omni PhysX deformables, Lula IK, `unittest`, Bash.

## Global Constraints

- Work on branch `feature/pregrasped-cable-mount`, based on `main` commit `6ded651ef8db386aaa21ae2445f49e103b921da9`.
- Cable USD: `/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd`.
- Cable root: `/World/NetworkCable`.
- Tracked plug: `/World/NetworkCable/E_crystal_head1_45`.
- The connector remains mounted for the complete process lifetime; no release path is added.
- The cable tail remains deformable; do not rigidify the whole asset.
- `/World/ToolCenter` keeps its existing numerical hand transform `(0.0, 0.0, 0.1034)` with identity local orientation.
- Mount the RJ45 nose-face center onto ToolCenter; do not add a second connector offset in perception, control, or future insertion code.
- Detect the longitudinal axis from plug-local bounds. Require `longest / second_longest >= 1.5`.
- Determine the cable-side sign from the cable-root center projected into the plug-local longitudinal axis. Ambiguity is fatal.
- Align plug nose axis to ToolCenter local `+Z`; align the widest transverse plug axis to ToolCenter local `+Y`.
- Attachment mask covers the complete plug bounds, adds `0.5 mm` on both transverse directions and at the nose, and adds no extension past the cable-side face.
- Use current Omni Physics deformable schemas and `PhysxAutoDeformableAttachmentAPI`. Do not silently fall back to removed legacy attachment schemas.
- Scene device is `cuda:0`; GPU dynamics enabled; broadphase `GPU`; solver `TGS`; physics timestep remains `1/60 s` initially.
- Cosmetic finger total clearance is `1.0 mm`; the proxy, not finger contact, carries the connector.
- Final mount validation uses 30 consecutive frames. Every frame must have tip error `<= 0.5 mm` and axis error `<= 1.0 degree`.
- Existing physical ToolCenter tracking tolerance remains `<= 0.3 mm`; visual target steps remain `<= 1.0 mm`.
- No insertion command, per-frame cable transform, hard-coded world insertion direction, or manual tip-depth offset.
- Any unsupported schema, invalid attachment, unstable mount, or failed nominal visual alignment is a kill switch.

---

## File Map

- Create `single_rack_cv/cable_geometry.py` — pure connector-frame and transform mathematics.
- Create `single_rack_cv/cable_mount.py` — Isaac/USD/PhysX asset loading, proxy, joint, attachment, finger presentation, and validation.
- Create `single_rack_cv/tools/inspect_cable_asset.py` — one-shot local schema and hierarchy inspector.
- Create `single_rack_cv/tests/test_cable_geometry.py` — pure geometry tests.
- Create `single_rack_cv/tests/test_cable_mount_contract.py` — pure configuration/schema-contract tests.
- Modify `single_rack_cv/config.py` — add `CableMountConfig`; switch the scene device to `cuda:0` only when the mount is enabled.
- Modify `single_rack_cv/sim.py` — author cable startup infrastructure and expose `prepare_for_perception()`.
- Modify `single_rack_cv/main.py` — validate mount before debug/YOLOE initialization.
- Modify `single_rack_cv/tests/test_runtime_wiring.py` — structural safety and startup-order checks.
- Modify `single_rack_cv/README.md` — document the pregrasped-cable runtime and smoke-test gates.

---

### Task 1: Pure RJ45 geometry and mount transform

**Files:**
- Create: `single_rack_cv/cable_geometry.py`
- Create: `single_rack_cv/tests/test_cable_geometry.py`

**Interfaces:**
- Produces:
  - `PlugFrame`
  - `AttachmentBounds`
  - `validate_transform(matrix: np.ndarray, label: str) -> np.ndarray`
  - `detect_plug_frame(local_min_m, local_max_m, world_from_plug, cable_center_world_m, *, axis_ratio_min, cable_projection_min_m) -> PlugFrame`
  - `compute_attachment_bounds(local_min_m, local_max_m, frame, padding_m) -> AttachmentBounds`
  - `compute_world_from_root_for_tip(world_from_root, world_from_plug, frame, desired_world_from_tip) -> np.ndarray`
  - `angular_error_deg(axis_a, axis_b) -> float`
- Consumed later by `cable_mount.py` only. It must have no Isaac, USD, PhysX, or `omni` imports.

- [ ] **Step 1: Write failing tests for axis selection, nose detection, roll, and ambiguity**

Create `tests/test_cable_geometry.py` with these concrete cases:

```python
from __future__ import annotations

import unittest
import numpy as np

from cable_geometry import (
    angular_error_deg,
    compute_attachment_bounds,
    compute_world_from_root_for_tip,
    detect_plug_frame,
)


def transform(rotation=np.eye(3), translation=(0.0, 0.0, 0.0)):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


class CableGeometryTests(unittest.TestCase):
    def test_long_axis_x_and_cable_on_negative_side_select_positive_nose(self):
        frame = detect_plug_frame(
            local_min_m=np.array([-0.018, -0.005, -0.006]),
            local_max_m=np.array([+0.018, +0.005, +0.006]),
            world_from_plug=transform(),
            cable_center_world_m=np.array([-0.20, 0.0, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.tip_local_m, [0.018, 0.0, 0.0])
        np.testing.assert_allclose(frame.nose_axis_local, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(frame.wide_axis_local, [0.0, 0.0, 1.0])

    def test_long_axis_y_and_cable_on_positive_side_select_negative_nose(self):
        frame = detect_plug_frame(
            local_min_m=np.array([-0.005, -0.018, -0.006]),
            local_max_m=np.array([+0.005, +0.018, +0.006]),
            world_from_plug=transform(),
            cable_center_world_m=np.array([0.0, +0.20, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.tip_local_m, [0.0, -0.018, 0.0])
        np.testing.assert_allclose(frame.nose_axis_local, [0.0, -1.0, 0.0])

    def test_rotated_world_transform_is_used_for_cable_side_projection(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        frame = detect_plug_frame(
            local_min_m=np.array([-0.018, -0.005, -0.006]),
            local_max_m=np.array([+0.018, +0.005, +0.006]),
            world_from_plug=transform(rotation=rotation, translation=(1.0, 2.0, 3.0)),
            cable_center_world_m=np.array([1.0, 1.8, 3.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.nose_axis_local, [1.0, 0.0, 0.0])

    def test_ambiguous_aspect_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous longitudinal axis"):
            detect_plug_frame(
                np.array([-0.010, -0.009, -0.006]),
                np.array([+0.010, +0.009, +0.006]),
                transform(),
                np.array([-0.1, 0.0, 0.0]),
                axis_ratio_min=1.5,
                cable_projection_min_m=0.002,
            )

    def test_ambiguous_cable_side_projection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous cable-side projection"):
            detect_plug_frame(
                np.array([-0.018, -0.005, -0.006]),
                np.array([+0.018, +0.005, +0.006]),
                transform(),
                np.array([0.0, 0.1, 0.0]),
                axis_ratio_min=1.5,
                cable_projection_min_m=0.002,
            )
```

- [ ] **Step 2: Run the tests and verify the import failure**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
/usr/bin/python3 -m unittest -v tests.test_cable_geometry
```

Expected: failure because `cable_geometry.py` does not exist.

- [ ] **Step 3: Implement the immutable geometry outputs and strict validation**

Create `cable_geometry.py` with these public types and invariants:

```python
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PlugFrame:
    local_min_m: np.ndarray
    local_max_m: np.ndarray
    dimensions_m: np.ndarray
    longitudinal_axis_index: int
    wide_transverse_axis_index: int
    cable_side_sign: int
    tip_local_m: np.ndarray
    nose_axis_local: np.ndarray
    wide_axis_local: np.ndarray
    narrow_axis_local: np.ndarray
    plug_from_tip: np.ndarray


@dataclass(frozen=True)
class AttachmentBounds:
    local_min_m: np.ndarray
    local_max_m: np.ndarray
    center_local_m: np.ndarray
    size_m: np.ndarray


def _finite_vector(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array


def validate_transform(matrix: np.ndarray, label: str) -> np.ndarray:
    matrix = _finite_vector(matrix, (4, 4), label)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{label} must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7):
        raise ValueError(f"{label} rotation must be right handed")
    return matrix
```

Implement `detect_plug_frame` using this exact algorithm:

```python
def detect_plug_frame(
    local_min_m,
    local_max_m,
    world_from_plug,
    cable_center_world_m,
    *,
    axis_ratio_min: float,
    cable_projection_min_m: float,
) -> PlugFrame:
    local_min = _finite_vector(local_min_m, (3,), "local_min_m")
    local_max = _finite_vector(local_max_m, (3,), "local_max_m")
    if np.any(local_max <= local_min):
        raise ValueError("plug bounds must have positive dimensions")
    world_from_plug = validate_transform(world_from_plug, "world_from_plug")
    cable_center_world = _finite_vector(cable_center_world_m, (3,), "cable_center_world_m")
    dimensions = local_max - local_min
    order = np.argsort(dimensions)
    longitudinal = int(order[-1])
    second = int(order[-2])
    if dimensions[longitudinal] / dimensions[second] < axis_ratio_min:
        raise ValueError("ambiguous longitudinal axis")
    plug_from_world = np.linalg.inv(world_from_plug)
    cable_local = (plug_from_world @ np.r_[cable_center_world, 1.0])[:3]
    plug_center = 0.5 * (local_min + local_max)
    projection = float(cable_local[longitudinal] - plug_center[longitudinal])
    if abs(projection) < cable_projection_min_m:
        raise ValueError("ambiguous cable-side projection")
    cable_side_sign = 1 if projection > 0.0 else -1
    nose_sign = -cable_side_sign
    transverse = [index for index in range(3) if index != longitudinal]
    wide = max(transverse, key=lambda index: dimensions[index])
    narrow = next(index for index in transverse if index != wide)
    nose_axis = np.zeros(3)
    nose_axis[longitudinal] = nose_sign
    wide_axis = np.zeros(3)
    wide_axis[wide] = 1.0
    narrow_axis = np.cross(wide_axis, nose_axis)
    narrow_axis /= np.linalg.norm(narrow_axis)
    tip = plug_center.copy()
    tip[longitudinal] = local_max[longitudinal] if nose_sign > 0 else local_min[longitudinal]
    plug_from_tip = np.eye(4)
    plug_from_tip[:3, 0] = narrow_axis
    plug_from_tip[:3, 1] = wide_axis
    plug_from_tip[:3, 2] = nose_axis
    plug_from_tip[:3, 3] = tip
    return PlugFrame(
        local_min_m=local_min,
        local_max_m=local_max,
        dimensions_m=dimensions,
        longitudinal_axis_index=longitudinal,
        wide_transverse_axis_index=wide,
        cable_side_sign=cable_side_sign,
        tip_local_m=tip,
        nose_axis_local=nose_axis,
        wide_axis_local=wide_axis,
        narrow_axis_local=narrow_axis,
        plug_from_tip=plug_from_tip,
    )
```

- [ ] **Step 4: Add failing tests for attachment trimming and root transform mapping**

Append tests that require:

```python
    def test_attachment_bounds_do_not_extend_past_cable_side_face(self):
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            transform(),
            np.array([-0.20, 0.0, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        bounds = compute_attachment_bounds(
            frame.local_min_m,
            frame.local_max_m,
            frame,
            padding_m=0.0005,
        )
        self.assertAlmostEqual(bounds.local_min_m[0], -0.018)
        self.assertAlmostEqual(bounds.local_max_m[0], +0.0185)
        self.assertAlmostEqual(bounds.local_min_m[1], -0.0055)
        self.assertAlmostEqual(bounds.local_max_m[1], +0.0055)

    def test_root_transform_maps_tip_frame_exactly_onto_toolcenter(self):
        world_from_root = transform(translation=(0.2, -0.1, 0.4))
        world_from_plug = transform(translation=(0.5, 0.0, 0.2))
        desired_world_from_tip = transform(translation=(0.7, -0.2, 1.3))
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            world_from_plug,
            np.array([0.1, 0.0, 0.2]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        mounted = compute_world_from_root_for_tip(
            world_from_root,
            world_from_plug,
            frame,
            desired_world_from_tip,
        )
        root_from_plug = np.linalg.inv(world_from_root) @ world_from_plug
        actual_world_from_tip = mounted @ root_from_plug @ frame.plug_from_tip
        np.testing.assert_allclose(actual_world_from_tip, desired_world_from_tip, atol=1e-9)
```

- [ ] **Step 5: Implement attachment bounds, root mapping, and angular error**

```python
def compute_attachment_bounds(local_min_m, local_max_m, frame, padding_m):
    local_min = np.asarray(local_min_m, dtype=np.float64).copy()
    local_max = np.asarray(local_max_m, dtype=np.float64).copy()
    if not math.isfinite(padding_m) or padding_m < 0.0:
        raise ValueError("padding_m must be finite and nonnegative")
    longitudinal = frame.longitudinal_axis_index
    for axis in range(3):
        if axis != longitudinal:
            local_min[axis] -= padding_m
            local_max[axis] += padding_m
    if frame.nose_axis_local[longitudinal] > 0.0:
        local_max[longitudinal] += padding_m
    else:
        local_min[longitudinal] -= padding_m
    center = 0.5 * (local_min + local_max)
    return AttachmentBounds(local_min, local_max, center, local_max - local_min)


def compute_world_from_root_for_tip(
    world_from_root,
    world_from_plug,
    frame,
    desired_world_from_tip,
):
    world_from_root = validate_transform(world_from_root, "world_from_root")
    world_from_plug = validate_transform(world_from_plug, "world_from_plug")
    desired_world_from_tip = validate_transform(desired_world_from_tip, "desired_world_from_tip")
    root_from_plug = np.linalg.inv(world_from_root) @ world_from_plug
    root_from_tip = root_from_plug @ frame.plug_from_tip
    return desired_world_from_tip @ np.linalg.inv(root_from_tip)


def angular_error_deg(axis_a, axis_b):
    a = _finite_vector(axis_a, (3,), "axis_a")
    b = _finite_vector(axis_b, (3,), "axis_b")
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))
```

- [ ] **Step 6: Run pure tests and commit**

```bash
/usr/bin/python3 -m unittest -v tests.test_cable_geometry
/usr/bin/python3 -m py_compile cable_geometry.py tests/test_cable_geometry.py
git add single_rack_cv/cable_geometry.py single_rack_cv/tests/test_cable_geometry.py
git commit -m "Add automatic RJ45 mount geometry"
```

Expected: all Task 1 tests pass.

---

### Task 2: Cable configuration and local asset-schema inspection gate

**Files:**
- Modify: `single_rack_cv/config.py:19-40, 273-304`
- Create: `single_rack_cv/tools/inspect_cable_asset.py`
- Create: `single_rack_cv/tests/test_cable_mount_contract.py`

**Interfaces:**
- Produces `CableMountConfig` and `CONFIG.cable_mount`.
- The inspector emits one JSON object with `asset_exists`, `tracked_plug_valid`, `deformable_candidates`, `schema_family`, and `supported`.
- Task 3 may proceed only when the inspector reports `schema_family="omniphysics"` and `supported=true`.

- [ ] **Step 1: Add failing configuration-contract tests**

```python
from __future__ import annotations

from pathlib import Path
import unittest

from config import CONFIG


class CableMountContractTests(unittest.TestCase):
    def test_canonical_cable_paths_and_limits(self):
        cfg = CONFIG.cable_mount
        self.assertTrue(cfg.enabled)
        self.assertEqual(
            cfg.usd_path,
            "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd",
        )
        self.assertEqual(cfg.root_path, "/World/NetworkCable")
        self.assertEqual(
            cfg.tracked_plug_path,
            "/World/NetworkCable/E_crystal_head1_45",
        )
        self.assertEqual(cfg.proxy_path, "/World/CableMountProxy")
        self.assertEqual(cfg.fixed_joint_path, "/World/CableMountFixedJoint")
        self.assertEqual(cfg.attachment_path, "/World/CableMountAttachment")
        self.assertEqual(cfg.mask_path, "/World/CableMountAttachment/MaskShape")
        self.assertEqual(cfg.validation_frames, 30)
        self.assertAlmostEqual(cfg.max_tip_error_m, 0.0005)
        self.assertAlmostEqual(cfg.max_axis_error_deg, 1.0)
        self.assertAlmostEqual(cfg.attachment_padding_m, 0.0005)
        self.assertAlmostEqual(cfg.finger_total_clearance_m, 0.001)

    def test_gpu_device_is_required_when_mount_is_enabled(self):
        self.assertEqual(CONFIG.scene.device, "cuda:0")
```

- [ ] **Step 2: Run and verify failure because `CableMountConfig` is absent**

```bash
/usr/bin/python3 -m unittest -v tests.test_cable_mount_contract
```

- [ ] **Step 3: Add the frozen configuration**

Insert before `IKConfig` in `config.py`:

```python
@dataclass(frozen=True)
class CableMountConfig:
    enabled: bool = True
    usd_path: str = (
        "/home/aayush/isaacsim_assets/Network cable 001/"
        "model_Networkcable1_69323.usd"
    )
    root_path: str = "/World/NetworkCable"
    tracked_plug_path: str = "/World/NetworkCable/E_crystal_head1_45"
    proxy_path: str = "/World/CableMountProxy"
    fixed_joint_path: str = "/World/CableMountFixedJoint"
    attachment_path: str = "/World/CableMountAttachment"
    mask_path: str = "/World/CableMountAttachment/MaskShape"
    hand_link_name: str = "panda_hand"
    axis_ratio_min: float = 1.5
    cable_projection_min_m: float = 0.002
    attachment_padding_m: float = 0.0005
    finger_total_clearance_m: float = 0.001
    initial_settle_frames: int = 60
    validation_frames: int = 30
    max_tip_error_m: float = 0.0005
    max_axis_error_deg: float = 1.0
```

Change `SceneConfig.device` to `"cuda:0"`, then add to `Config`:

```python
cable_mount: CableMountConfig = field(default_factory=CableMountConfig)
```

- [ ] **Step 4: Create the Isaac asset inspector**

Create `tools/inspect_cable_asset.py`. It must start `SimulationApp`, create a stage, add the cable reference at the configured root, update the app for 30 frames, then print and save a strict report:

```python
report = {
    "asset_exists": Path(cfg.usd_path).is_file(),
    "root_valid": bool(root_prim.IsValid()),
    "tracked_plug_valid": bool(plug_prim.IsValid()),
    "tracked_plug_applied_schemas": list(plug_prim.GetAppliedSchemas()),
    "deformable_candidates": candidates,
    "schema_family": "omniphysics" if omniphysics_candidates else "unsupported",
    "supported": bool(
        Path(cfg.usd_path).is_file()
        and plug_prim.IsValid()
        and len(omniphysics_candidates) == 1
    ),
}
```

Candidate discovery must inspect `GetAppliedSchemas()` and `HasAPI("OmniPhysicsDeformableBodyAPI")` while walking upward from the tracked plug, then search root descendants only if the ancestor walk finds none. Do not classify `PhysxDeformableBodyAPI` alone as supported.

Write atomically to:

```text
camera_output/cable_asset_schema.json
```

Exit codes:

```text
0 = one supported OmniPhysics deformable body found
2 = asset composed but schema unsupported or ambiguous
1 = file/stage/runtime failure
```

- [ ] **Step 5: Add structural tests for the inspector contract**

Tests must read the inspector source and assert all of these literal requirements are present:

```python
self.assertIn('HasAPI("OmniPhysicsDeformableBodyAPI")', source)
self.assertIn('camera_output/cable_asset_schema.json', source)
self.assertIn('"supported"', source)
self.assertNotIn("PhysxPhysicsAttachment.Define", source)
self.assertNotIn("PhysxAutoAttachmentAPI", source)
```

- [ ] **Step 6: Run pure tests and the mandatory workstation schema probe**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract

"$HOME/isaacsim/python.sh" tools/inspect_cable_asset.py
status=$?
cat camera_output/cable_asset_schema.json
exit "$status"
```

Required before Task 3:

```json
{"schema_family": "omniphysics", "supported": true}
```

Kill switch: if status is `2`, stop implementation. Convert or rebuild the cable asset with Isaac Sim 6 Omni Physics deformable schemas before authoring an attachment. Do not add a legacy compatibility branch.

- [ ] **Step 7: Commit configuration and inspector**

```bash
git add single_rack_cv/config.py \
        single_rack_cv/tools/inspect_cable_asset.py \
        single_rack_cv/tests/test_cable_mount_contract.py
git commit -m "Add cable mount configuration and schema gate"
```

---

### Task 3: Isaac cable loading, GPU scene configuration, and one-time placement

**Files:**
- Create: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/sim.py:14-31, 319-339, 734-865`
- Test: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Produces `CableMount.author_before_play(stage, hand_path, world_from_toolcenter) -> None`.
- Stores `plug_frame`, `attachment_bounds`, `deformable_body_path`, and `mounted_world_from_root`.
- Does not start physics, initialize an articulation, or create YOLOE.

- [ ] **Step 1: Add structural tests for one-time authoring and GPU requirements**

Require source-level evidence that:

```python
self.assertIn("class CableMount", cable_mount_source)
self.assertIn("create_auto_deformable_attachment", cable_mount_source)
self.assertIn("PhysxAutoDeformableAttachmentAPI", cable_mount_source)
self.assertIn("UsdGeom.Cube.Define", cable_mount_source)
self.assertIn("set_enabled_gpu_dynamics(True)", sim_source)
self.assertIn("set_broadphase_type(\"GPU\")", sim_source)
self.assertIn("set_solver_type(\"TGS\")", sim_source)
self.assertNotIn("set_world_pose", cable_mount_source)
self.assertNotIn("while ", cable_mount_source)
```

The `while` prohibition applies to `CableMount.author_before_play`; bounded ancestor/descendant iteration may use `for` loops.

- [ ] **Step 2: Implement `CableMount` construction and geometry queries**

Create these public data structures:

```python
@dataclass(frozen=True)
class CableMountDiagnostics:
    deformable_body_path: str
    plug_dimensions_m: tuple[float, float, float]
    longitudinal_axis_index: int
    cable_side_sign: int
    insertion_tip_local_m: tuple[float, float, float]
    attachment_min_local_m: tuple[float, float, float]
    attachment_max_local_m: tuple[float, float, float]
    finger_total_gap_m: float


class CableMount:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mount_cfg = cfg.cable_mount
        self.hand_path = ""
        self.deformable_body_path = ""
        self.plug_frame = None
        self.attachment_bounds = None
        self.diagnostics = None
```

Implement exact helpers:

```python
def _find_unique_descendant(stage, root_path: str, name: str) -> str
def _local_bounds(stage, path: str) -> tuple[np.ndarray, np.ndarray]
def _world_bounds_center(stage, path: str) -> np.ndarray
def _world_transform(stage, path: str) -> np.ndarray
def _set_single_world_transform(stage, path: str, world_from_prim: np.ndarray) -> None
def _discover_omniphysics_deformable(stage, root_path: str, plug_path: str) -> str
```

`_set_single_world_transform` must author one `UsdGeom.XformOp.TypeTransform` on the cable root before play, with no later call site.

- [ ] **Step 3: Configure the physics scene through the Isaac Sim 6 wrapper**

In `sim.py`, after `SimulationManager.setup_simulation(...)` obtains the physics scene:

```python
physics_scene = physics_scenes[0]
physics_scene.set_enabled_gpu_dynamics(True)
physics_scene.set_broadphase_type("GPU")
physics_scene.set_solver_type("TGS")

if not physics_scene.get_enabled_gpu_dynamics():
    raise RuntimeError("Cable mount requires GPU dynamics")
if physics_scene.get_broadphase_type() != "GPU":
    raise RuntimeError("Cable mount requires GPU broadphase")
if physics_scene.get_solver_type() != "TGS":
    raise RuntimeError("Cable mount requires TGS")
```

Remove the current `set_enabled_gpu_dynamics(False)` call.

- [ ] **Step 4: Load and place the cable before play**

`CableMount.author_before_play` must:

1. Validate `os.path.isfile(cfg.usd_path)`.
2. Add the cable reference at `cfg.root_path`.
3. Update the app enough to compose the reference.
4. Find the hand dynamically below `cfg.scene.franka_asset_path`.
5. Query plug-local bounds, plug transform, root transform, and cable-root world center.
6. Call `detect_plug_frame` and `compute_attachment_bounds`.
7. Build `desired_world_from_tip` from the startup ToolCenter pose.
8. Call `compute_world_from_root_for_tip`.
9. Author the cable-root transform exactly once.
10. Re-query geometry and assert tip placement error is below `1e-6 m` before play.

The startup ToolCenter pose is computed with the existing `hand_pose_to_tool_pose` using `IKConfig.initial_position`, `initial_orientation_wxyz`, and the existing ToolCenter local transform. Do not instantiate a second target offset.

- [ ] **Step 5: Run structural and compile checks**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract \
  tests.test_runtime_wiring

"$HOME/isaacsim/python.sh" -m py_compile cable_mount.py sim.py
```

- [ ] **Step 6: Commit GPU setup and one-time placement**

```bash
git add single_rack_cv/cable_mount.py \
        single_rack_cv/sim.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Load and place pregrasped deformable cable"
```

---

### Task 4: Hand-fixed proxy, masked deformable attachment, and collision filtering

**Files:**
- Modify: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Extends `CableMount.author_before_play` to create:
  - `/World/CableMountProxy`
  - `/World/CableMountFixedJoint`
  - `/World/CableMountAttachment`
  - `/World/CableMountAttachment/MaskShape`
- Produces `attachment_is_valid() -> bool` and `fixed_joint_is_valid() -> bool`.

- [ ] **Step 1: Add failing structural tests for exact attachment primitives**

Require:

```python
self.assertIn("UsdPhysics.RigidBodyAPI.Apply", source)
self.assertIn("UsdPhysics.CollisionAPI.Apply", source)
self.assertIn("UsdPhysics.FixedJoint.Define", source)
self.assertIn("deformableUtils.create_auto_deformable_attachment", source)
self.assertIn('physxAutoDeformableAttachment:maskShapes', source)
self.assertIn("UsdPhysics.FilteredPairsAPI", source)
self.assertNotIn("PhysxPhysicsAttachment", source)
self.assertNotIn("PhysxAutoAttachmentAPI", source)
```

- [ ] **Step 2: Create the hidden rigid proxy at the attachment volume**

Use `UsdGeom.Cube.Define(stage, cfg.proxy_path)`, set `size=1.0`, and author a single transform so the cube matches `attachment_bounds` in world space. Apply:

```python
UsdPhysics.RigidBodyAPI.Apply(proxy_prim).CreateRigidBodyEnabledAttr(True)
UsdPhysics.CollisionAPI.Apply(proxy_prim)
UsdPhysics.MassAPI.Apply(proxy_prim).CreateMassAttr(0.001)
UsdGeom.Imageable(proxy_prim).MakeInvisible()
```

The proxy is world-level, not nested under `panda_hand`.

- [ ] **Step 3: Create a fixed joint whose local frames preserve the proxy pose**

Compute:

```python
world_from_hand = _world_transform(stage, hand_path)
world_from_proxy = _world_transform(stage, cfg.proxy_path)
hand_from_proxy = np.linalg.inv(world_from_hand) @ world_from_proxy
```

Author:

```python
joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(cfg.fixed_joint_path))
joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
joint.CreateBody1Rel().SetTargets([Sdf.Path(cfg.proxy_path)])
joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hand_from_proxy[:3, 3]))
joint.CreateLocalRot0Attr().Set(_matrix_to_gf_quat(hand_from_proxy[:3, :3]))
joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
```

Verify both body relationships resolve to valid rigid bodies.

- [ ] **Step 4: Create the mask cube and current auto deformable attachment**

Create the mask as a child of the attachment scope and transform it to the same world volume as the proxy. Then call the official current helper:

```python
from omni.physx.scripts import deformableUtils

deformableUtils.create_auto_deformable_attachment(
    stage,
    cfg.attachment_path,
    self.deformable_body_path,
    cfg.proxy_path,
)
attachment_prim = stage.GetPrimAtPath(cfg.attachment_path)
if not attachment_prim.HasAPI("PhysxAutoDeformableAttachmentAPI"):
    raise RuntimeError("Auto deformable attachment API was not authored")
attachment_prim.GetRelationship(
    "physxAutoDeformableAttachment:maskShapes"
).SetTargets([Sdf.Path(cfg.mask_path)])
attachment_prim.GetAttribute(
    "physxAutoDeformableAttachment:deformableVertexOverlapOffset"
).Set(cfg.attachment_padding_m)
attachment_prim.GetAttribute(
    "physxAutoDeformableAttachment:enableCollisionFiltering"
).Set(True)
```

Do not enable unsupported rigid-surface sampling.

- [ ] **Step 5: Filter proxy collisions against Franka rigid links**

Apply `UsdPhysics.FilteredPairsAPI` to the proxy and add every Franka rigid-body descendant path as filtered pairs. This prevents the invisible load-bearing proxy from fighting the hand or fingers. Do not filter the cable against the rack or environment.

- [ ] **Step 6: Add validity methods and diagnostics**

```python
def fixed_joint_is_valid(self) -> bool:
    prim = self.stage.GetPrimAtPath(self.mount_cfg.fixed_joint_path)
    return bool(prim.IsValid() and prim.IsA(UsdPhysics.FixedJoint))


def attachment_is_valid(self) -> bool:
    prim = self.stage.GetPrimAtPath(self.mount_cfg.attachment_path)
    return bool(
        prim.IsValid()
        and prim.HasAPI("PhysxAutoDeformableAttachmentAPI")
        and len(prim.GetRelationship(
            "physxAutoDeformableAttachment:maskShapes"
        ).GetTargets()) == 1
    )
```

- [ ] **Step 7: Run tests and commit**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract \
  tests.test_runtime_wiring

"$HOME/isaacsim/python.sh" -m py_compile cable_mount.py sim.py
git add single_rack_cv/cable_mount.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Attach deformable connector to Franka hand"
```

---

### Task 5: Cosmetic finger gap and 30-frame mount validation

**Files:**
- Modify: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/sim.py:285-339, 351-384, 1250-1370`
- Create/modify: `single_rack_cv/tests/test_cable_mount_contract.py`

**Interfaces:**
- Produces:
  - `CableMount.configure_fingers(articulation) -> None`
  - `CableMount.sample_validation() -> tuple[float, float]`
  - `CableMount.finalize_validation() -> CableMountValidation`
  - `SimulationRuntime.prepare_for_perception() -> None`

- [ ] **Step 1: Add pure tests for validation-window behavior**

Add a pure helper in `cable_mount.py` only if it can be imported without Isaac; otherwise put it in `cable_geometry.py`:

```python
@dataclass(frozen=True)
class CableMountValidation:
    frame_count: int
    maximum_tip_error_m: float
    maximum_axis_error_deg: float


def validate_mount_window(samples, required_frames, max_tip_error_m, max_axis_error_deg):
    if len(samples) != required_frames:
        raise ValueError("mount validation requires the complete frame window")
    max_tip = max(sample[0] for sample in samples)
    max_axis = max(sample[1] for sample in samples)
    if max_tip > max_tip_error_m:
        raise RuntimeError("RJ45 tip mount error exceeds limit")
    if max_axis > max_axis_error_deg:
        raise RuntimeError("RJ45 axis error exceeds limit")
    return CableMountValidation(required_frames, max_tip, max_axis)
```

Tests must prove one bad frame fails even if the other 29 are perfect.

- [ ] **Step 2: Configure the cosmetic finger gap after articulation initialization**

`configure_fingers` must:

1. Compute plug width along ToolCenter local `+Y` from `plug_frame.dimensions_m[wide_transverse_axis_index]`.
2. Add `finger_total_clearance_m`.
3. Divide by two for symmetric finger-joint positions.
4. Resolve `panda_finger_joint1` and `panda_finger_joint2` by name.
5. Query articulation joint limits.
6. Clamp each position within limits.
7. Set both current positions and position targets.
8. Store the final total gap in diagnostics.

Do not add gripper close/open actions to the runtime loop.

- [ ] **Step 3: Implement frame-by-frame mount measurement**

Each sample must re-query current plug and ToolCenter transforms after physics updates:

```python
tip_world = (
    world_from_plug @ np.r_[self.plug_frame.tip_local_m, 1.0]
)[:3]
nose_world = world_from_plug[:3, :3] @ self.plug_frame.nose_axis_local
tool_position, tool_orientation = self.runtime.ik.actual_tool.get_world_pose()
tool_axis_world = quaternion_wxyz_to_matrix(tool_orientation)[:, 2]
tip_error_m = float(np.linalg.norm(tip_world - tool_position))
axis_error_deg = angular_error_deg(nose_world, tool_axis_world)
```

Every sample also checks fixed-joint validity, attachment validity, deformable-body validity, and GPU dynamics state. Any invalid state raises immediately.

- [ ] **Step 4: Implement `SimulationRuntime.prepare_for_perception()`**

Add a public method with this exact lifecycle:

```python
def prepare_for_perception(self) -> None:
    if self.cable_mount is None:
        return
    cfg = self.cfg.cable_mount
    for _ in range(cfg.initial_settle_frames):
        self.step()
        self.update_ik()
        self._update_startup_settle()
    samples = []
    while len(samples) < cfg.validation_frames:
        self.step()
        self.update_ik()
        self._update_startup_settle()
        if not self.visual_servo.startup_ready:
            continue
        samples.append(self.cable_mount.sample_validation(self))
    validation = validate_mount_window(
        samples,
        cfg.validation_frames,
        cfg.max_tip_error_m,
        cfg.max_axis_error_deg,
    )
    self.cable_mount.log_success(validation)
```

The loop is bounded by a computed hard frame cap:

```python
max_prepare_frames = cfg.initial_settle_frames + cfg.validation_frames + 600
```

If startup never settles within that cap, raise `RuntimeError`. Do not allow an unbounded wait.

- [ ] **Step 5: Wire the mount into scene construction**

In `SimulationRuntime.__init__`, initialize `self.cable_mount = None`.

In `_build_scene`, after Franka/cameras and GPU scene creation but before play:

```python
if self.cfg.cable_mount.enabled:
    self.cable_mount = CableMount(self.cfg)
    self.cable_mount.author_before_play(
        stage=stage,
        hand_path=self._find_unique_descendant(
            self.cfg.scene.franka_asset_path,
            self.cfg.cable_mount.hand_link_name,
        ),
        world_from_toolcenter=startup_world_from_toolcenter,
    )
```

After `_create_ik`, call `self.cable_mount.configure_fingers(self.ik.articulation)`.

- [ ] **Step 6: Add diagnostics tests and commit**

Structural tests require log labels:

```text
[CABLE MOUNT]
validation frames
maximum tip error mm
maximum axis error deg
fixed joint: valid
attachment: valid
cable tail: deformable
GPU dynamics: enabled
```

Run:

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract \
  tests.test_runtime_wiring

"$HOME/isaacsim/python.sh" -m py_compile \
  cable_geometry.py cable_mount.py sim.py
git add single_rack_cv/cable_geometry.py \
        single_rack_cv/cable_mount.py \
        single_rack_cv/sim.py \
        single_rack_cv/tests/test_cable_mount_contract.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Validate pregrasped cable mount before perception"
```

---

### Task 6: Gate YOLOE startup, preserve canonical control, and document the runtime

**Files:**
- Modify: `single_rack_cv/main.py:117-160`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`
- Modify: `single_rack_cv/README.md`

**Interfaces:**
- `main.py` calls `runtime.prepare_for_perception()` exactly once before `DebugOutputs` and `YOLOEPortDetector.initialize()`.
- The canonical loop from frame capture through `runtime.observe_visual_servo(observation)` remains functionally unchanged.

- [ ] **Step 1: Add startup-order and safety tests**

Parse `main.py` source positions and assert:

```python
prepare_index = source.index("runtime.prepare_for_perception()")
debug_index = source.index("debug = DebugOutputs")
yolo_index = source.index("detector.initialize()")
loop_index = source.index("while runtime.is_running()")
self.assertLess(prepare_index, debug_index)
self.assertLess(prepare_index, yolo_index)
self.assertLess(yolo_index, loop_index)
```

Also assert across `main.py`, `sim.py`, and `cable_mount.py`:

```python
for forbidden in (
    "insert_command",
    "command_insertion",
    "release_cable",
    "set_world_pose(position=",  # cable path must never be driven each frame
):
    self.assertNotIn(forbidden, combined_source)
```

Use targeted checks so existing IK target `set_world_pose` calls are not falsely rejected; the prohibition is specifically for `NETWORK_CABLE_ROOT_PATH`, tracked plug, proxy, and mask after play.

- [ ] **Step 2: Move debug and YOLOE construction after preparation**

Required initialization order:

```python
runtime = SimulationRuntime(simulation_app=simulation_app, cfg=CONFIG)
runtime.prepare_for_perception()
debug = DebugOutputs(CONFIG)
detector = YOLOEPortDetector(CONFIG.yoloe)
detector.initialize()
```

A mount-preparation exception must follow the existing fatal-error cleanup path and close Isaac Sim.

- [ ] **Step 3: Update the README**

Document:

- the simulation starts with the connector permanently mounted,
- the tail remains deformable,
- GPU dynamics is required,
- ToolCenter now physically represents the RJ45 tip while retaining the same calibrated transform,
- the runtime still stops at pre-insert alignment,
- the schema-inspection command,
- the normal run command,
- the mount validation gates,
- the kill switch.

Exact commands:

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" tools/inspect_cable_asset.py
"$HOME/isaacsim/python.sh" main.py
```

- [ ] **Step 4: Run regression tests and commit**

```bash
"$HOME/isaacsim/python.sh" -m unittest -v \
  tests.test_front_plane \
  tests.test_live_control \
  tests.test_runtime_wiring \
  tests.test_benchmark \
  tests.test_ground_truth \
  tests.test_repo_cleanliness \
  tests.test_automatic_port_ground_truth \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract

"$HOME/isaacsim/python.sh" -m py_compile \
  cable_geometry.py cable_mount.py config.py sim.py main.py \
  tools/inspect_cable_asset.py

git diff --check
git add single_rack_cv/main.py \
        single_rack_cv/tests/test_runtime_wiring.py \
        single_rack_cv/README.md
git commit -m "Gate visual servo on cable mount validation"
```

---

### Task 7: Isaac workstation smoke test and final evidence

**Files:**
- No source changes unless a real defect is found.
- Generated evidence remains under ignored `single_rack_cv/camera_output/`.

**Interfaces:**
- Produces the only acceptable proof that the actual local asset, attachment schema, GPU physics, and camera controller work together.

- [ ] **Step 1: Confirm the exact branch and clean worktree**

```bash
cd "$HOME/Isaacsim-Scripts" || exit 1
git switch feature/pregrasped-cable-mount
git pull --ff-only origin feature/pregrasped-cable-mount
git status --short
cd single_rack_cv || exit 1
```

Expected: no tracked modifications.

- [ ] **Step 2: Run schema inspection and stop on any unsupported result**

```bash
set -o pipefail
"$HOME/isaacsim/python.sh" tools/inspect_cable_asset.py \
  2>&1 | tee camera_output/cable_asset_schema_console.txt
schema_status=${PIPESTATUS[0]}
cat camera_output/cable_asset_schema.json
printf 'schema status: %s\n' "$schema_status"
test "$schema_status" -eq 0
```

Required: exactly one OmniPhysics deformable candidate and `supported=true`.

- [ ] **Step 3: Run the complete test and compile suite**

```bash
"$HOME/isaacsim/python.sh" -m unittest -v \
  tests.test_front_plane \
  tests.test_live_control \
  tests.test_runtime_wiring \
  tests.test_benchmark \
  tests.test_ground_truth \
  tests.test_repo_cleanliness \
  tests.test_automatic_port_ground_truth \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract

"$HOME/isaacsim/python.sh" -m py_compile \
  cable_geometry.py cable_mount.py config.py sim.py main.py \
  tools/inspect_cable_asset.py
```

Expected: zero failures and compile exit `0`.

- [ ] **Step 4: Run one cable-mounted nominal simulation**

```bash
rm -f camera_output/cable_mount_nominal_console.txt
set -o pipefail
"$HOME/isaacsim/python.sh" main.py \
  2>&1 | tee camera_output/cable_mount_nominal_console.txt
status=${PIPESTATUS[0]}
printf 'runtime status: %s\n' "$status"
```

Allow the run to reach `RGB STEREO VISUAL SERVO COMPLETE`, then stop with Ctrl-C.

- [ ] **Step 5: Require exact startup evidence**

```bash
grep -F "[CABLE MOUNT]" camera_output/cable_mount_nominal_console.txt
grep -F "validation frames: 30/30" camera_output/cable_mount_nominal_console.txt
grep -F "fixed joint: valid" camera_output/cable_mount_nominal_console.txt
grep -F "attachment: valid" camera_output/cable_mount_nominal_console.txt
grep -F "cable tail: deformable" camera_output/cable_mount_nominal_console.txt
grep -F "GPU dynamics: enabled" camera_output/cable_mount_nominal_console.txt
grep -F "RGB STEREO VISUAL SERVO COMPLETE" camera_output/cable_mount_nominal_console.txt
grep -F "next action: hold; no insertion is commanded." camera_output/cable_mount_nominal_console.txt
```

Manually verify logged values:

```text
maximum tip error <= 0.500 mm
maximum axis error <= 1.000 degree
physical tracking error <= 0.300 mm
visual target step never exceeds 1.000 mm
```

- [ ] **Step 6: Inspect the viewport**

Require all of the following:

- connector points toward the rack,
- RJ45 body sits between the fingers without obvious interpenetration,
- cable tail bends and settles rather than moving as a rigid stick,
- connector does not drift relative to the hand during alignment,
- cameras are not blocked by the cable or fingers,
- no invalid attachment actor, rest-shape, or GPU-dynamics errors appear.

- [ ] **Step 7: Apply the kill switch honestly**

Do not proceed to insertion or merge when any of these occur:

- schema inspector reports unsupported or ambiguous deformable structure,
- attachment creation emits invalid actor warnings,
- tip error exceeds `0.5 mm` in any validation frame,
- axis error exceeds `1 degree` in any validation frame,
- cable tail destabilizes the arm or camera,
- nominal visual alignment fails under GPU dynamics,
- physical tracking error exceeds `0.3 mm`.

Fix the actual mount/asset/physics defect. Do not widen limits, attach the complete cable rigidly, or reintroduce per-frame transforms.

- [ ] **Step 8: Final repository evidence**

```bash
git status --short
git diff --check
git ls-files single_rack_cv/camera_output
```

Expected: clean tracked worktree, no whitespace errors, and no generated camera output committed.

Record before review:

```text
schema report path and supported result
unit-test pass count
mount validation maximum tip error
mount validation maximum axis error
physical ToolCenter tracking error
visual-servo completion evidence
confirmation of deformable tail
confirmation of no insertion path
```
