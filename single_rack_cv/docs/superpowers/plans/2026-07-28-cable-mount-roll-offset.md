# Cable Mount Roll and Forward Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Roll the pregrasped RJ45 connector exactly 90 degrees around its insertion axis and place its tip 30 mm farther forward from `panda_hand` without changing insertion direction, ToolCenter calibration, fixed-joint topology, deformable-tail topology, or validation limits.

**Architecture:** Add one pure geometry helper that derives a presentation-adjusted RJ45 tip frame from the existing ToolCenter frame. Keep the global IK ToolCenter untouched; `CableMount.author_before_play()` uses the adjusted frame only for the one-time pre-physics cable-root placement, after which the existing real plug-to-hand fixed joint owns the pose.

**Tech Stack:** Python 3, NumPy, `unittest`, OpenUSD/PhysX integration in Isaac Sim 6.0.0.

## Global Constraints

- Apply a `+90.0` degree local roll around ToolCenter local `+Z`.
- Translate the RJ45 tip exactly `0.030` m along ToolCenter local `+Z`.
- Preserve the connector nose direction along ToolCenter local `+Z`.
- Do not modify `IKConfig.tool_center_local_position_m` or camera calibration.
- Do not add post-play transforms, proxies, deformable attachments, or tolerance widening.
- Keep startup limits at tip error `<= 0.5 mm` and axis error `<= 1.0 degree`.

---

### Task 1: Pure presentation-frame geometry

**Files:**
- Modify: `single_rack_cv/cable_geometry.py`
- Test: `single_rack_cv/tests/test_cable_geometry.py`

**Interfaces:**
- Consumes: `world_from_toolcenter: np.ndarray`, `roll_deg: float`, `forward_offset_m: float`.
- Produces: `compute_presented_tip_frame(world_from_toolcenter, *, roll_deg, forward_offset_m) -> np.ndarray`.

- [ ] **Step 1: Write the failing geometry tests**

Add the import and tests:

```python
from cable_geometry import compute_presented_tip_frame


def test_presented_tip_frame_rolls_90_degrees_and_moves_forward(self):
    tool = matrix(translation=(0.4, -0.2, 1.1))
    presented = compute_presented_tip_frame(
        tool,
        roll_deg=90.0,
        forward_offset_m=0.030,
    )
    np.testing.assert_allclose(presented[:3, 2], tool[:3, 2], atol=1e-12)
    np.testing.assert_allclose(
        presented[:3, 3],
        tool[:3, 3] + 0.030 * tool[:3, 2],
        atol=1e-12,
    )
    np.testing.assert_allclose(presented[:3, 0], tool[:3, 1], atol=1e-12)
    np.testing.assert_allclose(presented[:3, 1], -tool[:3, 0], atol=1e-12)
    validate_transform(presented, "presented")


def test_presented_tip_frame_zero_adjustment_is_identity(self):
    tool = matrix(
        rotation=np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        translation=(0.7, 0.1, 1.3),
    )
    presented = compute_presented_tip_frame(
        tool,
        roll_deg=0.0,
        forward_offset_m=0.0,
    )
    np.testing.assert_allclose(presented, tool, atol=1e-12)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
cd single_rack_cv
python -m unittest \
  tests.test_cable_geometry.CableGeometryTests.test_presented_tip_frame_rolls_90_degrees_and_moves_forward \
  tests.test_cable_geometry.CableGeometryTests.test_presented_tip_frame_zero_adjustment_is_identity -v
```

Expected: import failure because `compute_presented_tip_frame` does not exist.

- [ ] **Step 3: Implement the minimal pure helper**

Add to `cable_geometry.py`:

```python
def compute_presented_tip_frame(
    world_from_toolcenter: np.ndarray,
    *,
    roll_deg: float,
    forward_offset_m: float,
) -> np.ndarray:
    """Return a rigid RJ45 tip frame adjusted in ToolCenter-local coordinates."""

    tool = validate_transform(world_from_toolcenter, "world_from_toolcenter")
    roll = float(roll_deg)
    offset = _finite_nonnegative(forward_offset_m, "forward_offset_m")
    if not math.isfinite(roll):
        raise ValueError("roll_deg must be finite")

    radians = math.radians(roll)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    local_adjustment = np.eye(4, dtype=np.float64)
    local_adjustment[:3, :3] = np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    local_adjustment[2, 3] = offset
    return validate_transform(
        tool @ local_adjustment,
        "presented_world_from_tip",
    )
```

- [ ] **Step 4: Run the complete pure geometry suite**

Run:

```bash
cd single_rack_cv
python -m unittest tests.test_cable_geometry -v
```

Expected: all cable geometry tests pass.

- [ ] **Step 5: Commit the geometry unit**

```bash
git add single_rack_cv/cable_geometry.py single_rack_cv/tests/test_cable_geometry.py
git commit -m "feat(cable): add mount presentation transform"
```

### Task 2: Configuration and pre-play mount wiring

**Files:**
- Modify: `single_rack_cv/config.py`
- Modify: `single_rack_cv/cable_mount.py`
- Test: `single_rack_cv/tests/test_runtime_wiring.py`

**Interfaces:**
- Consumes: `CableMountConfig.presentation_roll_deg`, `CableMountConfig.forward_tip_offset_m`, and `compute_presented_tip_frame(...)` from Task 1.
- Produces: the adjusted `desired_world_from_tip` used by `compute_world_from_root_for_tip(...)` before physics starts.

- [ ] **Step 1: Write the failing structural wiring test**

Add:

```python
def test_cable_mount_presentation_is_configured_and_preplay_only(self):
    config_source = (ROOT / "config.py").read_text(encoding="utf-8")
    mount_source = (ROOT / "cable_mount.py").read_text(encoding="utf-8")
    self.assertIn("presentation_roll_deg: float = 90.0", config_source)
    self.assertIn("forward_tip_offset_m: float = 0.030", config_source)
    self.assertIn("compute_presented_tip_frame", mount_source)
    adjustment = mount_source.index("compute_presented_tip_frame(")
    placement = mount_source.index("compute_world_from_root_for_tip(")
    joint = mount_source.index("self._author_fixed_joint()")
    self.assertLess(adjustment, placement)
    self.assertLess(placement, joint)
    self.assertNotIn("set_world_pose", mount_source)
```

- [ ] **Step 2: Run the focused structural test and confirm failure**

Run:

```bash
cd single_rack_cv
python -m unittest \
  tests.test_runtime_wiring.RuntimeWiringTests.test_cable_mount_presentation_is_configured_and_preplay_only -v
```

Expected: failure because the two config fields and helper call do not exist.

- [ ] **Step 3: Add explicit cable presentation config**

In `CableMountConfig`, add:

```python
presentation_roll_deg: float = 90.0
forward_tip_offset_m: float = 0.030
```

Do not change `IKConfig.tool_center_local_position_m`.

- [ ] **Step 4: Wire the adjusted frame into one-time placement**

Import the helper in `cable_mount.py`:

```python
from cable_geometry import compute_presented_tip_frame
```

Replace the direct ToolCenter assignment with:

```python
world_from_toolcenter = validate_transform(
    world_from_toolcenter,
    "world_from_toolcenter",
)
desired_world_from_tip = compute_presented_tip_frame(
    world_from_toolcenter,
    roll_deg=self.mount_cfg.presentation_roll_deg,
    forward_offset_m=self.mount_cfg.forward_tip_offset_m,
)
```

Keep all subsequent root placement, fixed-joint creation, collision filtering, and attachment checks unchanged.

- [ ] **Step 5: Update mount validation semantics**

In `sample_validation()`, derive the same expected presented frame from the hand-based ToolCenter pose before comparing tip and axis:

```python
world_from_toolcenter = np.eye(4, dtype=np.float64)
world_from_toolcenter[:3, :3] = world_from_hand @ hand_from_tool
world_from_toolcenter[:3, 3] = tool_position
expected_world_from_tip = compute_presented_tip_frame(
    world_from_toolcenter,
    roll_deg=self.mount_cfg.presentation_roll_deg,
    forward_offset_m=self.mount_cfg.forward_tip_offset_m,
)
tip_error_m = float(
    np.linalg.norm(tip_world - expected_world_from_tip[:3, 3])
)
axis_error_deg = angular_error_deg(
    nose_world,
    expected_world_from_tip[:3, 2],
)
```

This preserves the existing `0.5 mm` and `1.0 degree` limits while validating the new intended physical frame rather than the old unoffset ToolCenter origin.

- [ ] **Step 6: Run the structural and pure test suites**

Run:

```bash
cd single_rack_cv
python -m unittest tests.test_runtime_wiring tests.test_cable_geometry -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the runtime wiring**

```bash
git add single_rack_cv/config.py single_rack_cv/cable_mount.py single_rack_cv/tests/test_runtime_wiring.py
git commit -m "feat(cable): roll and extend pregrasped connector"
```

### Task 3: Regression verification and workstation smoke test handoff

**Files:**
- Verify: `single_rack_cv/tests/`
- Verify on workstation: `single_rack_cv/main.py`

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: a branch ready for the user to pull and physically inspect in Isaac Sim.

- [ ] **Step 1: Run all repository-side tests**

Run:

```bash
cd single_rack_cv
python -m unittest discover -s tests -v
```

Expected: all tests pass with no regressions.

- [ ] **Step 2: Confirm the diff does not change forbidden systems**

Run:

```bash
git diff HEAD~2..HEAD -- \
  single_rack_cv/config.py \
  single_rack_cv/cable_geometry.py \
  single_rack_cv/cable_mount.py \
  single_rack_cv/tests/test_cable_geometry.py \
  single_rack_cv/tests/test_runtime_wiring.py
```

Confirm the diff contains no camera changes, no `IKConfig.tool_center_local_position_m` change, no new attachment, no proxy, no post-play cable transform, and no tolerance increase.

- [ ] **Step 3: Pull and run on the Isaac workstation**

```bash
cd ~/Isaacsim-Scripts
git switch feature/pregrasped-cable-mount
git pull --ff-only
cd single_rack_cv
~/.local/share/ov/pkg/isaac-sim-6.0.0/python.sh main.py 2>&1 | tee camera_output/cable_mount_roll_offset_console.txt
```

- [ ] **Step 4: Accept or trigger the kill switch**

Accept only when the viewport and logs show all of the following:

```text
RJ45 rolled 90 degrees around its nose axis
RJ45 tip protrudes 30 mm farther than before
nose still points along the same insertion direction
fixed joint valid
built-in attachment preserved
maximum tip error <= 0.5 mm
maximum axis error <= 1.0 degree
```

Trigger the kill switch if the offset causes camera obstruction, arm instability, deformable-tail overload, fixed-joint failure, attachment failure, or validation failure. Do not widen tolerances to make the run pass.
