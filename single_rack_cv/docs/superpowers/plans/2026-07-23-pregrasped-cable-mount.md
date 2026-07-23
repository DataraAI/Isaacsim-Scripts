# Pregrasped Deformable Cable Mount Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start the canonical `single_rack_cv` runtime with the supplied network cable permanently mounted to the Franka hand, with the RJ45 insertion tip coincident with the existing ToolCenter and the cable tail remaining deformable.

**Architecture:** `cable_geometry.py` owns all pure connector-frame and mount mathematics. `cable_mount.py` owns Isaac/USD/PhysX integration: asset loading, schema validation, one-time placement, rigid proxy, fixed joint, masked auto deformable attachment, cosmetic finger gap, and 30-frame mount validation. `SimulationRuntime.prepare_for_perception()` blocks YOLOE and camera acquisition until the mount passes every gate.

**Tech Stack:** Ubuntu 24.04, Isaac Sim 6.0.0 / Kit 110, Python 3.12, NumPy, OpenUSD (`pxr`), Omni PhysX deformables, Lula IK, `unittest`, Bash.

## Global Constraints

- Branch: `feature/pregrasped-cable-mount`, based on `main` commit `6ded651ef8db386aaa21ae2445f49e103b921da9`.
- Cable USD: `/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd`.
- Cable root: `/World/NetworkCable`.
- Tracked plug: `/World/NetworkCable/E_crystal_head1_45`.
- Permanent mount for the complete process lifetime; no release or regrasp.
- Cable tail remains deformable; never rigidify the whole asset.
- Existing ToolCenter transform remains numerically unchanged: translation `(0.0, 0.0, 0.1034)` and identity local orientation relative to `panda_hand`.
- RJ45 nose-face center is mounted exactly onto ToolCenter. No second connector offset is allowed elsewhere.
- Detect the plug longitudinal axis from local bounds and require `longest / second_longest >= 1.5`.
- Determine cable-side sign from the cable-root center projected into the plug-local longitudinal axis. Ambiguity is fatal.
- Align plug nose axis to ToolCenter local `+Z`; align widest transverse plug axis to ToolCenter local `+Y`.
- Attachment mask covers complete plug bounds, adds `0.5 mm` on both transverse directions and at the nose, and adds zero extension past the cable-side face.
- Use current `OmniPhysicsDeformableBodyAPI` and `PhysxAutoDeformableAttachmentAPI`. Removed legacy attachment schemas are unsupported.
- This feature branch uses `cuda:0` globally, even when the cable mount is disabled for diagnostic comparison. GPU dynamics enabled, broadphase `GPU`, solver `TGS`, timestep unchanged at `1/60 s` initially.
- Cosmetic finger total clearance: `1.0 mm`. The proxy, not finger contact, carries the cable.
- Mount validation: exactly 30 consecutive frames; every frame must satisfy tip error `<= 0.5 mm` and axis error `<= 1.0 degree`.
- Existing physical ToolCenter tracking tolerance remains `<= 0.3 mm`; target steps remain `<= 1.0 mm`.
- No insertion command, per-frame cable transform, hard-coded world insertion direction, or manual tip-depth offset.
- Unsupported schema, invalid attachment, unstable mount, or failed nominal alignment is a kill switch.

---

## File Map

- Create `single_rack_cv/cable_geometry.py` — pure connector frame, attachment bounds, root transform, angular error, and validation-window logic.
- Create `single_rack_cv/cable_mount.py` — Isaac/USD/PhysX integration.
- Create `single_rack_cv/tools/inspect_cable_asset.py` — local asset-schema probe.
- Create `single_rack_cv/tests/test_cable_geometry.py`.
- Create `single_rack_cv/tests/test_cable_mount_contract.py`.
- Modify `single_rack_cv/config.py`.
- Modify `single_rack_cv/sim.py`.
- Modify `single_rack_cv/main.py`.
- Modify `single_rack_cv/tests/test_runtime_wiring.py`.
- Modify `single_rack_cv/README.md`.

---

### Task 1: Pure connector geometry and validation-window logic

**Files:**
- Create: `single_rack_cv/cable_geometry.py`
- Create: `single_rack_cv/tests/test_cable_geometry.py`

**Interfaces:**

```python
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

@dataclass(frozen=True)
class CableMountValidation:
    frame_count: int
    maximum_tip_error_m: float
    maximum_axis_error_deg: float

def detect_plug_frame(...) -> PlugFrame
def compute_attachment_bounds(...) -> AttachmentBounds
def compute_world_from_root_for_tip(...) -> np.ndarray
def angular_error_deg(axis_a, axis_b) -> float
def validate_mount_window(samples, required_frames, max_tip_error_m, max_axis_error_deg) -> CableMountValidation
```

- [ ] **Step 1: Write failing tests for X/Y/Z longitudinal axes and both nose signs**

Create `tests/test_cable_geometry.py`:

```python
from __future__ import annotations

import unittest
import numpy as np

from cable_geometry import (
    angular_error_deg,
    compute_attachment_bounds,
    compute_world_from_root_for_tip,
    detect_plug_frame,
    validate_mount_window,
)


def matrix(rotation=None, translation=(0.0, 0.0, 0.0)):
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


class CableGeometryTests(unittest.TestCase):
    def test_x_axis_with_cable_on_negative_side_selects_positive_nose(self):
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            matrix(),
            np.array([-0.20, 0.0, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.tip_local_m, [0.018, 0.0, 0.0])
        np.testing.assert_allclose(frame.nose_axis_local, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(frame.wide_axis_local, [0.0, 0.0, 1.0])

    def test_y_axis_with_cable_on_positive_side_selects_negative_nose(self):
        frame = detect_plug_frame(
            np.array([-0.005, -0.018, -0.006]),
            np.array([+0.005, +0.018, +0.006]),
            matrix(),
            np.array([0.0, +0.20, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.tip_local_m, [0.0, -0.018, 0.0])
        np.testing.assert_allclose(frame.nose_axis_local, [0.0, -1.0, 0.0])

    def test_z_axis_is_supported(self):
        frame = detect_plug_frame(
            np.array([-0.005, -0.006, -0.018]),
            np.array([+0.005, +0.006, +0.018]),
            matrix(),
            np.array([0.0, 0.0, -0.20]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.nose_axis_local, [0.0, 0.0, 1.0])

    def test_world_rotation_is_used_when_classifying_cable_side(self):
        rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            matrix(rotation, (1.0, 2.0, 3.0)),
            np.array([1.0, 1.8, 3.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.nose_axis_local, [1.0, 0.0, 0.0])
```

- [ ] **Step 2: Run and confirm import failure**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
/usr/bin/python3 -m unittest -v tests.test_cable_geometry
```

Expected: import failure because `cable_geometry.py` does not exist.

- [ ] **Step 3: Implement strict transform and frame detection**

Create `cable_geometry.py` with finite-shape checks and this exact frame algorithm:

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
    local_min = _finite(local_min_m, (3,), "local_min_m")
    local_max = _finite(local_max_m, (3,), "local_max_m")
    world_from_plug = validate_transform(world_from_plug, "world_from_plug")
    cable_center_world = _finite(cable_center_world_m, (3,), "cable_center_world_m")
    if np.any(local_max <= local_min):
        raise ValueError("plug bounds must have positive dimensions")
    dimensions = local_max - local_min
    order = np.argsort(dimensions)
    longitudinal = int(order[-1])
    second = int(order[-2])
    if dimensions[longitudinal] / dimensions[second] < axis_ratio_min:
        raise ValueError("ambiguous longitudinal axis")
    plug_center = 0.5 * (local_min + local_max)
    plug_from_world = np.linalg.inv(world_from_plug)
    cable_local = (plug_from_world @ np.r_[cable_center_world, 1.0])[:3]
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
    tip[longitudinal] = (
        local_max[longitudinal]
        if nose_sign > 0
        else local_min[longitudinal]
    )
    plug_from_tip = np.eye(4)
    plug_from_tip[:3, 0] = narrow_axis
    plug_from_tip[:3, 1] = wide_axis
    plug_from_tip[:3, 2] = nose_axis
    plug_from_tip[:3, 3] = tip
    return PlugFrame(
        local_min,
        local_max,
        dimensions,
        longitudinal,
        wide,
        cable_side_sign,
        tip,
        nose_axis,
        wide_axis,
        narrow_axis,
        plug_from_tip,
    )
```

`validate_transform` must require a finite 4×4 homogeneous transform, orthonormal rotation, and determinant `+1`.

- [ ] **Step 4: Add failing tests for ambiguity, attachment trimming, root mapping, and one-bad-frame failure**

```python
    def test_ambiguous_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous longitudinal axis"):
            detect_plug_frame(
                np.array([-0.010, -0.009, -0.006]),
                np.array([+0.010, +0.009, +0.006]),
                matrix(),
                np.array([-0.1, 0.0, 0.0]),
                axis_ratio_min=1.5,
                cable_projection_min_m=0.002,
            )

    def test_ambiguous_cable_projection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous cable-side projection"):
            detect_plug_frame(
                np.array([-0.018, -0.005, -0.006]),
                np.array([+0.018, +0.005, +0.006]),
                matrix(),
                np.array([0.0, 0.1, 0.0]),
                axis_ratio_min=1.5,
                cable_projection_min_m=0.002,
            )

    def test_attachment_does_not_extend_past_cable_side(self):
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            matrix(),
            np.array([-0.2, 0.0, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        bounds = compute_attachment_bounds(frame, padding_m=0.0005)
        self.assertAlmostEqual(bounds.local_min_m[0], -0.018)
        self.assertAlmostEqual(bounds.local_max_m[0], +0.0185)
        self.assertAlmostEqual(bounds.local_min_m[1], -0.0055)
        self.assertAlmostEqual(bounds.local_max_m[1], +0.0055)

    def test_root_mapping_puts_tip_frame_exactly_on_toolcenter(self):
        world_from_root = matrix(translation=(0.2, -0.1, 0.4))
        world_from_plug = matrix(translation=(0.5, 0.0, 0.2))
        desired = matrix(translation=(0.7, -0.2, 1.3))
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
            desired,
        )
        root_from_plug = np.linalg.inv(world_from_root) @ world_from_plug
        actual = mounted @ root_from_plug @ frame.plug_from_tip
        np.testing.assert_allclose(actual, desired, atol=1e-9)

    def test_one_bad_validation_frame_fails_complete_window(self):
        samples = [(0.0001, 0.1)] * 29 + [(0.0006, 0.1)]
        with self.assertRaisesRegex(RuntimeError, "tip mount error"):
            validate_mount_window(samples, 30, 0.0005, 1.0)
```

- [ ] **Step 5: Implement attachment bounds, root mapping, angular error, and validation window**

```python
def compute_attachment_bounds(frame: PlugFrame, padding_m: float) -> AttachmentBounds:
    if not math.isfinite(padding_m) or padding_m < 0.0:
        raise ValueError("padding_m must be finite and nonnegative")
    local_min = frame.local_min_m.copy()
    local_max = frame.local_max_m.copy()
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
    frame: PlugFrame,
    desired_world_from_tip,
):
    world_from_root = validate_transform(world_from_root, "world_from_root")
    world_from_plug = validate_transform(world_from_plug, "world_from_plug")
    desired = validate_transform(desired_world_from_tip, "desired_world_from_tip")
    root_from_plug = np.linalg.inv(world_from_root) @ world_from_plug
    root_from_tip = root_from_plug @ frame.plug_from_tip
    return desired @ np.linalg.inv(root_from_tip)


def validate_mount_window(samples, required_frames, max_tip_error_m, max_axis_error_deg):
    samples = list(samples)
    if len(samples) != required_frames:
        raise ValueError("mount validation requires the complete frame window")
    if not samples:
        raise ValueError("mount validation window cannot be empty")
    max_tip = max(float(sample[0]) for sample in samples)
    max_axis = max(float(sample[1]) for sample in samples)
    if not math.isfinite(max_tip) or max_tip < 0.0:
        raise ValueError("tip errors must be finite and nonnegative")
    if not math.isfinite(max_axis) or max_axis < 0.0:
        raise ValueError("axis errors must be finite and nonnegative")
    if max_tip > max_tip_error_m:
        raise RuntimeError("RJ45 tip mount error exceeds limit")
    if max_axis > max_axis_error_deg:
        raise RuntimeError("RJ45 axis error exceeds limit")
    return CableMountValidation(required_frames, max_tip, max_axis)
```

- [ ] **Step 6: Run, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v tests.test_cable_geometry
/usr/bin/python3 -m py_compile cable_geometry.py tests/test_cable_geometry.py
git add single_rack_cv/cable_geometry.py single_rack_cv/tests/test_cable_geometry.py
git commit -m "Add automatic RJ45 mount geometry"
```

---

### Task 2: Configuration and mandatory local schema probe

**Files:**
- Modify: `single_rack_cv/config.py:19-40, 177-224, 287-304`
- Create: `single_rack_cv/tools/inspect_cable_asset.py`
- Create: `single_rack_cv/tests/test_cable_mount_contract.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CableMountConfig:
    enabled: bool
    usd_path: str
    root_path: str
    tracked_plug_path: str
    proxy_path: str
    fixed_joint_path: str
    attachment_path: str
    mask_path: str
    hand_link_name: str
    axis_ratio_min: float
    cable_projection_min_m: float
    attachment_padding_m: float
    finger_total_clearance_m: float
    initial_settle_frames: int
    validation_frames: int
    max_tip_error_m: float
    max_axis_error_deg: float
```

- [ ] **Step 1: Write failing configuration tests**

```python
from __future__ import annotations
import unittest
from config import CONFIG


class CableMountContractTests(unittest.TestCase):
    def test_canonical_paths_and_limits(self):
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
        self.assertAlmostEqual(cfg.attachment_padding_m, 0.0005)
        self.assertAlmostEqual(cfg.finger_total_clearance_m, 0.001)
        self.assertAlmostEqual(cfg.max_tip_error_m, 0.0005)
        self.assertAlmostEqual(cfg.max_axis_error_deg, 1.0)

    def test_feature_branch_uses_cuda_even_for_diagnostic_mount_disable(self):
        self.assertEqual(CONFIG.scene.device, "cuda:0")
```

- [ ] **Step 2: Run and confirm failure**

```bash
/usr/bin/python3 -m unittest -v tests.test_cable_mount_contract
```

- [ ] **Step 3: Add `CableMountConfig`, `CONFIG.cable_mount`, and CUDA scene device**

Add before `IKConfig`:

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

Change `SceneConfig.device` from `"cpu"` to `"cuda:0"`, and add:

```python
cable_mount: CableMountConfig = field(default_factory=CableMountConfig)
```

- [ ] **Step 4: Create the schema inspector**

`tools/inspect_cable_asset.py` must:

1. Start `SimulationApp`.
2. Create a new stage.
3. Add the cable reference at `CONFIG.cable_mount.root_path`.
4. Update 30 frames for composition.
5. Validate the tracked plug.
6. Walk upward from the plug, then search root descendants, for prims with `HasAPI("OmniPhysicsDeformableBodyAPI")`.
7. Require exactly one candidate.
8. Write atomically to `camera_output/cable_asset_schema.json`.

Required report keys:

```python
report = {
    "asset_exists": bool,
    "root_valid": bool,
    "tracked_plug_valid": bool,
    "tracked_plug_applied_schemas": list[str],
    "deformable_candidates": list[str],
    "schema_family": "omniphysics" | "unsupported",
    "supported": bool,
}
```

Exit codes:

```text
0 = exactly one supported OmniPhysics deformable body
2 = composed asset is unsupported or ambiguous
1 = file, stage, or runtime failure
```

- [ ] **Step 5: Add structural tests that prohibit legacy fallback**

```python
source = Path("tools/inspect_cable_asset.py").read_text()
self.assertIn('HasAPI("OmniPhysicsDeformableBodyAPI")', source)
self.assertIn("cable_asset_schema.json", source)
self.assertNotIn("PhysxPhysicsAttachment", source)
self.assertNotIn("PhysxAutoAttachmentAPI", source)
```

- [ ] **Step 6: Run tests and the mandatory workstation probe**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract

set -o pipefail
"$HOME/isaacsim/python.sh" tools/inspect_cable_asset.py \
  2>&1 | tee camera_output/cable_asset_schema_console.txt
status=${PIPESTATUS[0]}
cat camera_output/cable_asset_schema.json
test "$status" -eq 0
```

Required before Task 3:

```json
{"schema_family": "omniphysics", "supported": true}
```

Kill switch: status `2` stops implementation. Convert or rebuild the cable asset with Isaac Sim 6 Omni Physics deformable schemas. Do not add a legacy fallback.

- [ ] **Step 7: Commit**

```bash
git add single_rack_cv/config.py \
        single_rack_cv/tools/inspect_cable_asset.py \
        single_rack_cv/tests/test_cable_mount_contract.py
git commit -m "Add cable mount configuration and schema gate"
```

---

### Task 3: GPU scene, cable loading, and one-time cable placement

**Files:**
- Create: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/sim.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**

```python
class CableMount:
    def __init__(self, cfg: Config): ...
    def author_before_play(
        self,
        stage: Usd.Stage,
        hand_path: str,
        world_from_toolcenter: np.ndarray,
    ) -> None: ...
```

- [ ] **Step 1: Add structural tests for GPU setup and no runtime teleport**

```python
self.assertIn('set_enabled_gpu_dynamics(True)', sim_source)
self.assertIn('set_broadphase_type("GPU")', sim_source)
self.assertIn('set_solver_type("TGS")', sim_source)
self.assertNotIn('set_enabled_gpu_dynamics(False)', sim_source)
self.assertIn('class CableMount', cable_mount_source)
self.assertNotIn('set_world_pose', cable_mount_source)
```

- [ ] **Step 2: Implement `CableMount` state and USD query helpers**

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
    def __init__(self, cfg):
        self.cfg = cfg
        self.mount_cfg = cfg.cable_mount
        self.stage = None
        self.hand_path = ""
        self.deformable_body_path = ""
        self.plug_frame = None
        self.attachment_bounds = None
        self.diagnostics = None
```

Implement helpers:

```python
def _local_bounds(stage, path) -> tuple[np.ndarray, np.ndarray]
def _world_bounds_center(stage, path) -> np.ndarray
def _world_transform(stage, path) -> np.ndarray
def _set_single_transform_op(stage, path, world_from_prim) -> None
def _discover_omniphysics_deformable(stage, root_path, plug_path) -> str
```

`_set_single_transform_op` may be called only before play and only for the cable root.

- [ ] **Step 3: Switch the existing physics scene to GPU PhysX and verify it**

After `SimulationManager.setup_simulation(...)`:

```python
physics_scene = SimulationManager.get_physics_scenes()[0]
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

- [ ] **Step 4: Implement one-time asset loading and placement**

`author_before_play` must:

1. Validate the USD file exists.
2. Add the reference at `/World/NetworkCable`.
3. Validate tracked plug and one OmniPhysics deformable body.
4. Query plug-local bounds, plug world transform, root world transform, and cable-root world center.
5. Call `detect_plug_frame` and `compute_attachment_bounds`.
6. Compute startup ToolCenter from the existing fixed hand pose through `hand_pose_to_tool_pose`.
7. Call `compute_world_from_root_for_tip`.
8. Author exactly one root transform.
9. Re-query and require pre-play tip-placement error `< 1e-6 m`.

Do not move the tracked plug child independently.

- [ ] **Step 5: Integrate cable authoring into `_build_scene` before play**

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

Initialize `self.cable_mount = None` in `SimulationRuntime.__init__`.

- [ ] **Step 6: Test, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract \
  tests.test_runtime_wiring
"$HOME/isaacsim/python.sh" -m py_compile cable_mount.py sim.py
git add single_rack_cv/cable_mount.py \
        single_rack_cv/sim.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Load and place pregrasped deformable cable"
```

---

### Task 4: Rigid proxy, fixed joint, mask, and auto deformable attachment

**Files:**
- Modify: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**

```python
def fixed_joint_is_valid(self) -> bool
def attachment_is_valid(self) -> bool
```

- [ ] **Step 1: Add failing structural tests for current attachment APIs**

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

- [ ] **Step 2: Create a world-level hidden rigid proxy matching the attachment volume**

Use `UsdGeom.Cube.Define(stage, cfg.proxy_path)`, size `1.0`, and a transform derived from mounted `world_from_plug` plus `attachment_bounds.center_local_m` and `attachment_bounds.size_m`. Apply:

```python
UsdPhysics.RigidBodyAPI.Apply(proxy_prim).CreateRigidBodyEnabledAttr(True)
UsdPhysics.CollisionAPI.Apply(proxy_prim)
UsdPhysics.MassAPI.Apply(proxy_prim).CreateMassAttr(0.001)
UsdGeom.Imageable(proxy_prim).MakeInvisible()
```

Do not parent the proxy below `panda_hand`.

- [ ] **Step 3: Create a fixed joint preserving the proxy pose**

```python
world_from_hand = _world_transform(stage, self.hand_path)
world_from_proxy = _world_transform(stage, cfg.proxy_path)
hand_from_proxy = np.linalg.inv(world_from_hand) @ world_from_proxy
joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(cfg.fixed_joint_path))
joint.CreateBody0Rel().SetTargets([Sdf.Path(self.hand_path)])
joint.CreateBody1Rel().SetTargets([Sdf.Path(cfg.proxy_path)])
joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hand_from_proxy[:3, 3]))
joint.CreateLocalRot0Attr().Set(_matrix_to_gf_quat(hand_from_proxy[:3, :3]))
joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
```

- [ ] **Step 4: Create the current auto attachment, then its mask cube**

Call the helper first so it creates the attachment scope:

```python
from omni.physx.scripts import deformableUtils

deformableUtils.create_auto_deformable_attachment(
    stage,
    cfg.attachment_path,
    self.deformable_body_path,
    cfg.proxy_path,
)
```

Then define `UsdGeom.Cube` at `cfg.mask_path`, transform it to the attachment volume, and set:

```python
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

Do not set unsupported rigid-surface sampling fields.

- [ ] **Step 5: Filter proxy collisions against Franka rigid bodies only**

Apply `UsdPhysics.FilteredPairsAPI` to the proxy and set every Franka rigid-body descendant path as a filtered target. Do not filter cable collisions against the rack or ground.

- [ ] **Step 6: Add strict validity methods**

`fixed_joint_is_valid()` verifies prim type plus both body relationships. `attachment_is_valid()` verifies the current API, both attachable relationships, and exactly one mask target.

- [ ] **Step 7: Test and commit**

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

### Task 5: Cosmetic fingers and bounded 30-frame mount validation

**Files:**
- Modify: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/sim.py`
- Modify: `single_rack_cv/tests/test_cable_mount_contract.py`

**Interfaces:**

```python
def CableMount.configure_fingers(self, articulation) -> None
def CableMount.sample_validation(self, runtime) -> tuple[float, float]
def CableMount.log_success(self, validation: CableMountValidation) -> None
def SimulationRuntime.prepare_for_perception(self) -> None
```

- [ ] **Step 1: Implement cosmetic finger positioning**

`configure_fingers` must:

1. Read plug width on `wide_transverse_axis_index`.
2. Add `finger_total_clearance_m`.
3. Divide by two.
4. Resolve `panda_finger_joint1` and `panda_finger_joint2` by name.
5. Query and normalize articulation joint limits.
6. Clamp within limits.
7. Set both current positions and position targets.
8. Store final total gap in diagnostics.

No gripper action is added to the main loop.

- [ ] **Step 2: Implement current-frame mount measurement**

```python
world_from_plug = _world_transform(self.stage, self.mount_cfg.tracked_plug_path)
tip_world = (
    world_from_plug @ np.r_[self.plug_frame.tip_local_m, 1.0]
)[:3]
nose_world = world_from_plug[:3, :3] @ self.plug_frame.nose_axis_local
runtime._update_actual_tool_frame(runtime.ik)
tool_position, tool_orientation = runtime.ik.actual_tool.get_world_pose()
tool_axis = quaternion_wxyz_to_matrix(tool_orientation)[:, 2]
tip_error_m = float(np.linalg.norm(tip_world - tool_position))
axis_error_deg = angular_error_deg(nose_world, tool_axis)
```

Before returning, require valid fixed joint, attachment, deformable body, and enabled GPU dynamics.

- [ ] **Step 3: Implement bounded `prepare_for_perception()` with no `while` loop**

```python
def prepare_for_perception(self) -> None:
    if self.cable_mount is None:
        return
    cfg = self.cfg.cable_mount
    samples = []
    max_prepare_frames = (
        cfg.initial_settle_frames
        + cfg.validation_frames
        + 600
    )
    for frame_count in range(max_prepare_frames):
        self.step()
        self.update_ik()
        self._update_startup_settle()
        if frame_count < cfg.initial_settle_frames:
            continue
        if not self.visual_servo.startup_ready:
            continue
        samples.append(self.cable_mount.sample_validation(self))
        if len(samples) == cfg.validation_frames:
            break
    else:
        raise RuntimeError(
            "Cable mount did not settle and validate within the startup frame cap"
        )
    validation = validate_mount_window(
        samples,
        cfg.validation_frames,
        cfg.max_tip_error_m,
        cfg.max_axis_error_deg,
    )
    self.cable_mount.log_success(validation)
```

- [ ] **Step 4: Wire finger configuration after IK creation**

After `_create_ik` returns:

```python
if self.cable_mount is not None:
    self.cable_mount.configure_fingers(self.ik.articulation)
```

- [ ] **Step 5: Require complete diagnostics**

Successful log must include:

```text
[CABLE MOUNT]
tracked plug: /World/NetworkCable/E_crystal_head1_45
deformable body: ...
plug dimensions mm: [...]
insertion-tip local position m: [...]
attachment bounds local m: min=[...] max=[...]
finger total gap mm: ...
validation frames: 30/30
maximum tip error mm: ...
maximum axis error deg: ...
fixed joint: valid
attachment: valid
cable tail: deformable
GPU dynamics: enabled
```

- [ ] **Step 6: Test, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract \
  tests.test_runtime_wiring
"$HOME/isaacsim/python.sh" -m py_compile cable_geometry.py cable_mount.py sim.py
git add single_rack_cv/cable_geometry.py \
        single_rack_cv/cable_mount.py \
        single_rack_cv/sim.py \
        single_rack_cv/tests/test_cable_mount_contract.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Validate cable mount before perception"
```

---

### Task 6: Gate YOLOE startup and preserve the canonical controller

**Files:**
- Modify: `single_rack_cv/main.py:117-160`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`
- Modify: `single_rack_cv/README.md`

**Interfaces:**
- `runtime.prepare_for_perception()` is called once before `DebugOutputs`, detector construction, and `detector.initialize()`.
- Existing frame capture, perception, front-plane refinement, bounded translation, and hold-on-failure code remains unchanged.

- [ ] **Step 1: Add exact startup-order tests**

```python
prepare = source.index("runtime.prepare_for_perception()")
debug = source.index("debug = DebugOutputs")
yolo = source.index("detector.initialize()")
loop = source.index("while runtime.is_running()")
self.assertLess(prepare, debug)
self.assertLess(prepare, yolo)
self.assertLess(yolo, loop)
```

Add targeted source checks proving no post-play transform is applied to the cable root, tracked plug, proxy, or mask, and no insertion/release function exists.

- [ ] **Step 2: Move debug and YOLOE initialization behind mount validation**

```python
runtime = SimulationRuntime(
    simulation_app=simulation_app,
    cfg=CONFIG,
)
runtime.prepare_for_perception()
debug = DebugOutputs(CONFIG)
detector = YOLOEPortDetector(CONFIG.yoloe)
detector.initialize()
```

Any mount exception follows the existing fatal cleanup path and closes Isaac Sim.

- [ ] **Step 3: Update README commands and truth claims**

Document:

- pregrasped permanent connector mount,
- deformable tail,
- GPU physics requirement,
- ToolCenter physically represents the RJ45 tip without changing its calibrated numerical transform,
- runtime still ends at pre-insert hold,
- schema probe command,
- normal run command,
- exact mount gates,
- kill switch.

Commands:

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" tools/inspect_cable_asset.py
"$HOME/isaacsim/python.sh" main.py
```

- [ ] **Step 4: Run full tests and commit**

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

### Task 7: Isaac workstation qualification and kill switch

**Files:**
- No source changes unless a real defect is found.
- Generated evidence remains under ignored `single_rack_cv/camera_output/`.

- [ ] **Step 1: Confirm branch and clean worktree**

```bash
cd "$HOME/Isaacsim-Scripts" || exit 1
git switch feature/pregrasped-cable-mount
git pull --ff-only origin feature/pregrasped-cable-mount
git status --short
cd single_rack_cv || exit 1
```

- [ ] **Step 2: Run schema probe**

```bash
set -o pipefail
"$HOME/isaacsim/python.sh" tools/inspect_cable_asset.py \
  2>&1 | tee camera_output/cable_asset_schema_console.txt
schema_status=${PIPESTATUS[0]}
cat camera_output/cable_asset_schema.json
test "$schema_status" -eq 0
```

Required: exactly one deformable candidate, `schema_family="omniphysics"`, `supported=true`.

- [ ] **Step 3: Run complete tests and compile checks**

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

- [ ] **Step 4: Run one nominal cable-mounted simulation**

```bash
rm -f camera_output/cable_mount_nominal_console.txt
set -o pipefail
"$HOME/isaacsim/python.sh" main.py \
  2>&1 | tee camera_output/cable_mount_nominal_console.txt
status=${PIPESTATUS[0]}
printf 'runtime status: %s\n' "$status"
```

Allow the run to reach `RGB STEREO VISUAL SERVO COMPLETE`, then stop with Ctrl-C.

- [ ] **Step 5: Require logged evidence**

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

Require numeric evidence:

```text
maximum tip error <= 0.500 mm
maximum axis error <= 1.000 degree
physical ToolCenter tracking error <= 0.300 mm
maximum target step <= 1.000 mm
```

- [ ] **Step 6: Inspect viewport behavior**

Require:

- connector points toward rack,
- connector sits between fingers without obvious interpenetration,
- tail bends and settles rather than moving as a rigid stick,
- connector does not drift relative to hand,
- cable/fingers do not block cameras,
- no invalid attachment actor, rest-shape, or GPU-dynamics errors.

- [ ] **Step 7: Apply kill switch**

Do not merge or begin insertion when any of these occur:

- unsupported/ambiguous schema,
- invalid attachment actor warning,
- any validation frame exceeds `0.5 mm` tip error,
- any validation frame exceeds `1 degree` axis error,
- cable destabilizes arm/cameras,
- nominal alignment fails under GPU dynamics,
- physical tracking error exceeds `0.3 mm`.

Fix the actual asset, attachment, or physics defect. Do not widen limits, rigidify the complete cable, or add per-frame transforms.

- [ ] **Step 8: Record final evidence and repository cleanliness**

```bash
git status --short
git diff --check
git ls-files single_rack_cv/camera_output
```

Record:

```text
schema report path and supported result
unit-test pass count
maximum tip error
maximum axis error
physical ToolCenter tracking error
visual-servo completion evidence
confirmation that tail remained deformable
confirmation that no insertion path was added
```
