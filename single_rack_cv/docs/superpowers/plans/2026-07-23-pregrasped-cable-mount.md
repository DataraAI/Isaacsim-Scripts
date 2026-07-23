# Pregrasped Deformable Cable Mount Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start the canonical `single_rack_cv` runtime with the supplied cable permanently mounted by directly fixed-jointing the existing rigid RJ45 plug to `panda_hand`, while preserving the asset-authored plug-to-deformable-tail attachment.

**Architecture:** `cable_geometry.py` owns pure connector-frame and one-time placement math. `cable_mount.py` owns asset loading, topology verification, direct hand-to-plug fixed joint, narrow collision filtering, finger presentation, and mount validation. The runtime verifies but never reauthors the existing `PhysxAutoDeformableAttachmentAPI` linking `/World/NetworkCable/E_line_35` to `/World/NetworkCable/E_crystal_head1_45`.

**Tech Stack:** Ubuntu 24.04, Isaac Sim 6.0.0 / Kit 110, Python 3.12, NumPy, OpenUSD (`pxr`), Omni Physics deformables, Lula IK, `unittest`, Bash.

## Global Constraints

- Branch: `feature/pregrasped-cable-mount`.
- Cable USD: `/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd`.
- Cable root: `/World/NetworkCable`.
- Tracked rigid plug: `/World/NetworkCable/E_crystal_head1_45`.
- Discovered deformable tail: `/World/NetworkCable/E_line_35`.
- Preserved built-in attachment: `/World/NetworkCable/E_line_35/attachment`.
- Built-in attachment targets must remain exactly tail ↔ tracked plug.
- Permanent mount for the complete process lifetime; no release or regrasp.
- Cable tail remains deformable; never rigidify the complete asset.
- Existing ToolCenter transform remains numerically unchanged: translation `(0.0, 0.0, 0.1034)` and identity local orientation relative to `panda_hand`.
- Mount the RJ45 nose-face center exactly onto ToolCenter. No second connector offset is allowed.
- Detect longitudinal axis from local bounds and require `longest / second_longest >= 1.5`.
- Align plug nose axis to ToolCenter local `+Z`; align widest transverse plug axis to ToolCenter local `+Y`.
- Use a direct `UsdPhysics.FixedJoint` between dynamically discovered `panda_hand` and the tracked rigid plug.
- Do not create `/World/CableMountProxy`, a new auto deformable attachment, or an attachment mask.
- Preserve tracked-plug collisions against rack, port, tail, and ground.
- Filter collisions only against Franka hand/finger rigid bodies.
- Scene device `cuda:0`; GPU dynamics enabled; broadphase `GPU`; solver `TGS`; timestep `1/60 s` initially.
- Cosmetic finger total clearance: `1.0 mm`.
- Mount validation: exactly 30 consecutive frames; every frame tip error `<= 0.5 mm`, axis error `<= 1.0 degree`.
- Existing physical ToolCenter tracking tolerance remains `<= 0.3 mm`; target steps remain `<= 1.0 mm`.
- No insertion command, post-play cable transform, hard-coded world insertion direction, or manual tip-depth offset.

---

## File Map

- Existing `single_rack_cv/cable_geometry.py` — pure geometry and validation logic.
- Create `single_rack_cv/cable_mount.py` — topology verification and direct plug mount.
- Existing `single_rack_cv/tools/inspect_cable_asset.py` — schema and topology probe.
- Modify `single_rack_cv/config.py` — remove obsolete proxy/new-attachment fields and add hand/finger names.
- Modify `single_rack_cv/sim.py` — GPU scene, cable startup, direct joint, bounded validation.
- Modify `single_rack_cv/main.py` — block debug/YOLOE until mount validation passes.
- Modify `single_rack_cv/tests/test_cable_mount_contract.py`.
- Modify `single_rack_cv/tests/test_runtime_wiring.py`.
- Modify `single_rack_cv/README.md`.

---

### Task 1: Pure connector geometry and validation-window logic

**Status:** Completed and locally verified.

**Files:**
- `single_rack_cv/cable_geometry.py`
- `single_rack_cv/tests/test_cable_geometry.py`

**Evidence:** 15 geometry tests pass; Python compilation passes.

---

### Task 2: Configuration and asset topology probe

**Status:** Completed through topology discovery; configuration cleanup is folded into Task 3.

**Files:**
- `single_rack_cv/tools/inspect_cable_asset.py`
- `single_rack_cv/tests/test_cable_mount_contract.py`
- `single_rack_cv/config.py`

**Workstation evidence:**

```text
tracked plug rigid body: /World/NetworkCable/E_crystal_head1_45
deformable tail: /World/NetworkCable/E_line_35
existing attachment: /World/NetworkCable/E_line_35/attachment
attachment attachable0: /World/NetworkCable/E_line_35
attachment attachable1: /World/NetworkCable/E_crystal_head1_45
schema family: omniphysics
supported: true
```

---

### Task 3: Correct configuration and topology contract

**Files:**
- Modify: `single_rack_cv/config.py`
- Modify: `single_rack_cv/tests/test_cable_mount_contract.py`

**Interfaces produced:**

```python
@dataclass(frozen=True)
class CableMountConfig:
    enabled: bool
    usd_path: str
    root_path: str
    tracked_plug_path: str
    fixed_joint_path: str
    hand_link_name: str
    finger_link_names: tuple[str, str]
    finger_joint_names: tuple[str, str]
    axis_ratio_min: float
    cable_projection_min_m: float
    finger_total_clearance_m: float
    initial_settle_frames: int
    validation_frames: int
    max_tip_error_m: float
    max_axis_error_deg: float
```

- [ ] **Step 1: Write failing tests for corrected configuration**

Add to `tests/test_cable_mount_contract.py`:

```python
def test_config_uses_direct_rigid_plug_mount(self):
    cfg = CONFIG.cable_mount
    self.assertEqual(cfg.fixed_joint_path, "/World/CableMountFixedJoint")
    self.assertEqual(cfg.hand_link_name, "panda_hand")
    self.assertEqual(
        cfg.finger_link_names,
        ("panda_leftfinger", "panda_rightfinger"),
    )
    self.assertEqual(
        cfg.finger_joint_names,
        ("panda_finger_joint1", "panda_finger_joint2"),
    )
    self.assertFalse(hasattr(cfg, "proxy_path"))
    self.assertFalse(hasattr(cfg, "attachment_path"))
    self.assertFalse(hasattr(cfg, "mask_path"))
```

- [ ] **Step 2: Run and verify failure**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
/usr/bin/python3 -m unittest -v tests.test_cable_mount_contract
```

Expected: obsolete fields still exist and hand/finger fields are missing.

- [ ] **Step 3: Replace obsolete fields in `CableMountConfig`**

Use:

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
    fixed_joint_path: str = "/World/CableMountFixedJoint"
    hand_link_name: str = "panda_hand"
    finger_link_names: tuple[str, str] = (
        "panda_leftfinger",
        "panda_rightfinger",
    )
    finger_joint_names: tuple[str, str] = (
        "panda_finger_joint1",
        "panda_finger_joint2",
    )
    axis_ratio_min: float = 1.5
    cable_projection_min_m: float = 0.002
    finger_total_clearance_m: float = 0.001
    initial_settle_frames: int = 60
    validation_frames: int = 30
    max_tip_error_m: float = 0.0005
    max_axis_error_deg: float = 1.0
```

- [ ] **Step 4: Run, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract
/usr/bin/python3 -m py_compile config.py tests/test_cable_mount_contract.py
git add single_rack_cv/config.py single_rack_cv/tests/test_cable_mount_contract.py
git commit -m "Correct cable mount configuration for rigid plug"
```

---

### Task 4: Cable topology verification and one-time placement

**Files:**
- Create: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces produced:**

```python
@dataclass(frozen=True)
class CableTopology:
    deformable_body_path: str
    existing_attachment_path: str
    attachment_target0: str
    attachment_target1: str

@dataclass
class CableMountDiagnostics:
    plug_dimensions_m: tuple[float, float, float]
    tip_local_m: tuple[float, float, float]
    deformable_body_path: str
    existing_attachment_path: str
    finger_total_gap_m: float

class CableMount:
    def __init__(self, cfg: Config) -> None
    def author_before_play(
        self,
        stage: Usd.Stage,
        hand_path: str,
        world_from_toolcenter: np.ndarray,
    ) -> None
    def fixed_joint_is_valid(self) -> bool
    def built_in_attachment_is_preserved(self) -> bool
```

- [ ] **Step 1: Write failing structural tests**

Add to `tests/test_runtime_wiring.py`:

```python
def test_cable_mount_uses_existing_rigid_plug_topology(self):
    source = (ROOT / "cable_mount.py").read_text(encoding="utf-8")
    self.assertIn("PhysicsRigidBodyAPI", source)
    self.assertIn("PhysxAutoDeformableAttachmentAPI", source)
    self.assertIn("built_in_attachment_is_preserved", source)
    self.assertNotIn("CableMountProxy", source)
    self.assertNotIn("create_auto_deformable_attachment", source)
    self.assertNotIn("maskShapes", source)
```

- [ ] **Step 2: Run and verify import/file failure**

```bash
/usr/bin/python3 -m unittest -v tests.test_runtime_wiring
```

Expected: `cable_mount.py` does not exist.

- [ ] **Step 3: Implement strict topology discovery**

`CableMount._discover_topology(stage)` must:

1. Validate `tracked_plug_path` exists.
2. Require `tracked_plug.HasAPI(UsdPhysics.RigidBodyAPI)`.
3. Find exactly one descendant under `root_path` with `HasAPI("OmniPhysicsDeformableBodyAPI")`.
4. Find exactly one prim under the deformable body with `HasAPI("PhysxAutoDeformableAttachmentAPI")` whose two attachable relationships target the deformable body and tracked plug, in either relationship order.
5. Record the exact paths and targets in `CableTopology`.
6. Raise on zero, multiple, or mismatched candidates.

Use helpers:

```python
def _relationship_targets(prim: Usd.Prim, names: tuple[str, ...]) -> list[str]:
    for name in names:
        relationship = prim.GetRelationship(name)
        if relationship.IsValid():
            targets = [str(path) for path in relationship.GetTargets()]
            if targets:
                return targets
    return []
```

Support the relationship names actually reported by the probe and fail closed if neither exists.

- [ ] **Step 4: Implement one-time asset placement**

`author_before_play` must:

1. Require the USD file.
2. Add the cable reference at `root_path` before play.
3. Discover and store topology.
4. Query plug-local bounds, plug world transform, root world transform, and cable-root world-bounds center.
5. Call `detect_plug_frame`.
6. Call `compute_world_from_root_for_tip`.
7. Clear root xform op order and author one matrix transform on `/World/NetworkCable`.
8. Re-query the tip transform and require pre-play position error `< 1e-6 m` and axis error `< 1e-6 degree`.
9. Never move the tracked plug child independently.

- [ ] **Step 5: Run, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract \
  tests.test_runtime_wiring
"$HOME/isaacsim/python.sh" -m py_compile cable_mount.py
git add single_rack_cv/cable_mount.py single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Verify and place connected cable asset"
```

---

### Task 5: GPU scene and direct hand-to-plug fixed joint

**Files:**
- Modify: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/sim.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`

- [ ] **Step 1: Add failing structural tests for direct joint and GPU scene**

```python
def test_direct_plug_joint_and_gpu_scene_are_wired(self):
    mount = (ROOT / "cable_mount.py").read_text(encoding="utf-8")
    sim = (ROOT / "sim.py").read_text(encoding="utf-8")
    self.assertIn("UsdPhysics.FixedJoint.Define", mount)
    self.assertIn("tracked_plug_path", mount)
    self.assertNotIn("CableMountProxy", mount)
    self.assertIn("set_enabled_gpu_dynamics(True)", sim)
    self.assertIn('set_broadphase_type("GPU")', sim)
    self.assertIn('set_solver_type("TGS")', sim)
```

- [ ] **Step 2: Switch the shared scene to GPU PhysX and verify settings**

Replace the CPU/GPU-disable block in `sim.py` with:

```python
physics_scenes = SimulationManager.get_physics_scenes()
if not physics_scenes:
    raise RuntimeError("No physics scene was created")
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

- [ ] **Step 3: Create the direct fixed joint**

After one-time placement and before play:

```python
world_from_hand = _world_transform(stage, self.hand_path)
world_from_plug = _world_transform(stage, self.mount_cfg.tracked_plug_path)
hand_from_plug = np.linalg.inv(world_from_hand) @ world_from_plug
joint = UsdPhysics.FixedJoint.Define(
    stage,
    Sdf.Path(self.mount_cfg.fixed_joint_path),
)
joint.CreateBody0Rel().SetTargets([Sdf.Path(self.hand_path)])
joint.CreateBody1Rel().SetTargets([
    Sdf.Path(self.mount_cfg.tracked_plug_path)
])
joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hand_from_plug[:3, 3]))
joint.CreateLocalRot0Attr().Set(
    _matrix_to_gf_quat(hand_from_plug[:3, :3])
)
joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
```

`fixed_joint_is_valid()` must verify type and exact body targets.

- [ ] **Step 4: Filter only hand/finger collisions**

Resolve `panda_hand`, `panda_leftfinger`, and `panda_rightfinger` dynamically under the Franka asset. Apply `UsdPhysics.FilteredPairsAPI` to the tracked plug and set only those three rigid-body paths as filtered targets. Do not filter rack, port, tail, ground, or the full robot.

- [ ] **Step 5: Verify built-in attachment preservation**

Store the attachment path and targets before joint authoring. `built_in_attachment_is_preserved()` must re-read the prim and require:

```text
same prim path
PhysxAutoDeformableAttachmentAPI still applied
same two attachable targets
no mask relationship added
```

- [ ] **Step 6: Integrate pre-play authoring into `_build_scene`**

Before `app_utils.play()`:

```python
self.cable_mount = CableMount(self.cfg)
self.cable_mount.author_before_play(
    stage=stage,
    hand_path=hand_path,
    world_from_toolcenter=startup_world_from_toolcenter,
)
```

Compute `startup_world_from_toolcenter` from the existing fixed startup hand pose and unchanged ToolCenter offset.

- [ ] **Step 7: Test, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract \
  tests.test_runtime_wiring
"$HOME/isaacsim/python.sh" -m py_compile cable_mount.py sim.py
git add single_rack_cv/cable_mount.py single_rack_cv/sim.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Joint rigid RJ45 plug directly to Franka hand"
```

---

### Task 6: Cosmetic fingers and bounded mount validation

**Files:**
- Modify: `single_rack_cv/cable_mount.py`
- Modify: `single_rack_cv/sim.py`
- Modify: `single_rack_cv/tests/test_cable_mount_contract.py`

**Interfaces produced:**

```python
CableMount.configure_fingers(articulation: Articulation) -> None
CableMount.sample_validation(runtime: SimulationRuntime) -> tuple[float, float]
CableMount.log_success(validation: CableMountValidation) -> None
SimulationRuntime.prepare_for_perception() -> None
```

- [ ] **Step 1: Implement cosmetic finger positioning**

Read the plug width on `wide_transverse_axis_index`, add `finger_total_clearance_m`, divide by two, resolve both configured finger joints, clamp to articulation limits, set current positions and position targets, and store the total gap in diagnostics. No gripper action is added to the main loop.

- [ ] **Step 2: Implement current-frame validation sampling**

Each sample must:

1. Compute actual plug-tip world position and nose axis.
2. Update `/World/ToolCenter` from the real hand pose.
3. Compute tip and axis errors.
4. Require direct fixed-joint validity.
5. Require tracked-plug rigid-body validity.
6. Require built-in attachment preservation.
7. Require deformable-tail validity.
8. Require GPU dynamics enabled.

- [ ] **Step 3: Implement bounded `prepare_for_perception()`**

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

- [ ] **Step 4: Require complete diagnostics**

Print:

```text
[CABLE MOUNT]
tracked plug
deformable body
preserved attachment
plug dimensions mm
insertion-tip local position m
finger total gap mm
validation frames: 30/30
maximum tip error mm
maximum axis error deg
fixed joint: valid
built-in attachment: preserved
cable tail: deformable
GPU dynamics: enabled
```

- [ ] **Step 5: Test, compile, and commit**

```bash
/usr/bin/python3 -m unittest -v \
  tests.test_cable_geometry \
  tests.test_cable_mount_contract \
  tests.test_runtime_wiring
"$HOME/isaacsim/python.sh" -m py_compile cable_geometry.py cable_mount.py sim.py
git add single_rack_cv/cable_mount.py single_rack_cv/sim.py \
        single_rack_cv/tests/test_cable_mount_contract.py \
        single_rack_cv/tests/test_runtime_wiring.py
git commit -m "Validate direct cable mount before perception"
```

---

### Task 7: Gate YOLOE and preserve the canonical controller

**Files:**
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`
- Modify: `single_rack_cv/README.md`

- [ ] **Step 1: Add startup-order and safety tests**

```python
prepare = source.index("runtime.prepare_for_perception()")
debug = source.index("debug = DebugOutputs")
yolo = source.index("detector.initialize()")
loop = source.index("while runtime.is_running()")
self.assertLess(prepare, debug)
self.assertLess(prepare, yolo)
self.assertLess(yolo, loop)
```

Also require no post-play transform writes to cable/plug, no proxy, no new auto attachment, no release, and no insertion function.

- [ ] **Step 2: Move debug and YOLOE initialization behind validation**

```python
runtime = SimulationRuntime(simulation_app=simulation_app, cfg=CONFIG)
runtime.prepare_for_perception()
debug = DebugOutputs(CONFIG)
detector = YOLOEPortDetector(CONFIG.yoloe)
detector.initialize()
```

- [ ] **Step 3: Update README truth claims and commands**

Document the direct rigid-plug joint, preserved built-in deformable attachment, GPU requirement, RJ45-tip ToolCenter meaning, schema/topology probe, normal run command, mount limits, and no-insertion boundary.

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
git add single_rack_cv/main.py single_rack_cv/tests/test_runtime_wiring.py \
        single_rack_cv/README.md
git commit -m "Gate visual servo on direct cable mount validation"
```

---

### Task 8: Workstation qualification and kill switch

- [ ] **Step 1: Run topology probe**

```bash
cd "$HOME/Isaacsim-Scripts/single_rack_cv" || exit 1
"$HOME/isaacsim/python.sh" tools/inspect_cable_asset.py
```

Require the same rigid plug, deformable tail, and built-in attachment targets discovered during planning.

- [ ] **Step 2: Run complete tests and compile checks**

Use the Task 7 commands; every command must exit `0`.

- [ ] **Step 3: Run one nominal cable-mounted simulation**

```bash
set -o pipefail
"$HOME/isaacsim/python.sh" main.py \
  2>&1 | tee camera_output/cable_mount_nominal_console.txt
```

Allow the run to reach `RGB STEREO VISUAL SERVO COMPLETE`, then stop with Ctrl-C.

- [ ] **Step 4: Require logged evidence**

```bash
grep -F "[CABLE MOUNT]" camera_output/cable_mount_nominal_console.txt
grep -F "validation frames: 30/30" camera_output/cable_mount_nominal_console.txt
grep -F "fixed joint: valid" camera_output/cable_mount_nominal_console.txt
grep -F "built-in attachment: preserved" camera_output/cable_mount_nominal_console.txt
grep -F "cable tail: deformable" camera_output/cable_mount_nominal_console.txt
grep -F "GPU dynamics: enabled" camera_output/cable_mount_nominal_console.txt
grep -F "RGB STEREO VISUAL SERVO COMPLETE" camera_output/cable_mount_nominal_console.txt
grep -F "next action: hold; no insertion is commanded." \
  camera_output/cable_mount_nominal_console.txt
```

Require:

```text
maximum tip error <= 0.500 mm
maximum axis error <= 1.000 degree
physical ToolCenter tracking error <= 0.300 mm
maximum target step <= 1.000 mm
```

- [ ] **Step 5: Inspect viewport behavior**

Require the connector to point toward the rack, sit between fingers without obvious interpenetration, remain fixed relative to the hand, preserve the flexible tail, avoid camera occlusion, and produce no invalid-joint or attachment errors.

- [ ] **Step 6: Apply kill switch**

Do not merge or begin insertion if the direct joint is unstable, the built-in attachment changes or fails, the tail destabilizes the arm/cameras, any mount limit fails, nominal alignment fails, or physical tracking error exceeds `0.3 mm`.

Do not widen limits, duplicate attachments, rigidify the complete cable, or add per-frame transforms.