# Rear-Body Profile Donor TCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive the RJ45 insertion TCP transverse center from the uniquely dimension-matching rear housing component while preserving the existing connector nose depth.

**Architecture:** Continue extracting descendant USD mesh-component bounds in tracked-plug local coordinates. Select a component by transverse cross-section similarity to the measured 11.4 mm × 7.0 mm port, permit that component to sit behind the nose within a strict setback gate, copy only its two transverse center coordinates into the TCP, and keep the legacy longitudinal nose coordinate unchanged. Keep the marker-only motion lock enabled until the workstation confirms the cyan marker is centered in the insertable housing.

**Tech Stack:** Python 3.12, NumPy, OpenUSD/pxr, unittest, Isaac Sim 6.0.0.

## Global Constraints

- Do not change `EstimatedPortPoint`, `FrozenPortPoint`, camera geometry, perception, or port-center calculations.
- Preserve the legacy connector nose depth exactly.
- Change only the connector TCP transverse coordinates.
- Reject profile donors farther than 20.0 mm behind the nose.
- Reject transverse shifts larger than 3.0 mm.
- Require exactly one best dimensionally qualified donor unless near-equal candidates have centers within 0.35 mm.
- Keep `TCP_PROBE_ONLY = True`; do not enable YOLOE, handoff, or insertion in this implementation.
- Preserve the 0.5 mm lateral and 1.0 degree orientation safety limits.

---

### Task 1: Specify rear-body donor selection with failing tests

**Files:**
- Modify: `single_rack_cv/tests/test_connector_tcp.py`

**Interfaces:**
- Consumes: `derive_insertion_tcp(...) -> InsertionTcpDerivation`
- Produces: regression coverage for a dimensionally valid component 12.67 mm behind the nose, excessive-setback rejection, longitudinal-coordinate preservation, and ambiguity rejection.

- [ ] **Step 1: Replace the obsolete nose-reaching-only test**

Add a test using the measured asset-like components:

```python
def test_selects_dimension_matching_rear_body_as_profile_donor(self):
    result = derive_insertion_tcp(
        legacy_tip_local=np.array([18.076, 0.0, 0.0]),
        longitudinal_axis_index=0,
        nose_axis_local=np.array([1.0, 0.0, 0.0]),
        axis_scale_m_per_local_unit=np.array([0.001, 0.001, 0.001]),
        components=(
            self.component("front_with_latch", [-2.414, -5.246, -5.910], [18.076, 5.246, 5.910]),
            self.component("thin_insert", [5.406, -3.704, -0.450], [16.957, 3.704, 0.450]),
            self.component("rear_body", [-18.076, -5.2475, -4.135], [5.406, 5.2475, 3.000]),
        ),
        aperture_width_m=0.0114,
        aperture_height_m=0.0070,
    )
    self.assertEqual(result.selected_label, "rear_body")
    self.assertEqual(result.tip_local[0], 18.076)
    np.testing.assert_allclose(result.cross_section_m, [0.010495, 0.007135])
```

- [ ] **Step 2: Add excessive-setback rejection**

```python
def test_rejects_dimension_matching_profile_beyond_setback_limit(self):
    with self.assertRaisesRegex(RuntimeError, "setback"):
        derive_insertion_tcp(
            legacy_tip_local=self.legacy_tip,
            longitudinal_axis_index=0,
            nose_axis_local=np.array([1.0, 0.0, 0.0]),
            axis_scale_m_per_local_unit=self.scale,
            components=(self.component("far_body", [-20.0, -5.2, -4.0], [-3.0, 5.2, 3.0]),),
            aperture_width_m=0.0114,
            aperture_height_m=0.0070,
            maximum_profile_setback_m=0.010,
        )
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_connector_tcp
```

Expected: FAIL because `maximum_profile_setback_m` does not exist and the rear body is rejected by the nose-reaching rule.

- [ ] **Step 4: Commit the red tests**

```bash
git add single_rack_cv/tests/test_connector_tcp.py
git commit -m "test: define rear-body profile donor tcp"
```

### Task 2: Implement profile-donor TCP selection

**Files:**
- Modify: `single_rack_cv/connector_tcp.py`

**Interfaces:**
- Consumes: `MeshComponentBounds`, legacy tip, axis scales, port dimensions.
- Produces: `derive_insertion_tcp(..., maximum_profile_setback_m: float = 0.020) -> InsertionTcpDerivation`.

- [ ] **Step 1: Replace nose-reach qualification with setback qualification**

For each component, calculate the signed/absolute setback between the legacy nose and the component’s nearest nose-facing bound. Reject only when the donor lies ahead of the nose or farther than `maximum_profile_setback_m` behind it. Keep the dimension gate unchanged.

- [ ] **Step 2: Rank donors by shape first, setback second**

Use:

```python
score = shape_score + 2.0 * profile_setback_m
```

The dimensional fit must dominate selection; setback is only a deterministic tie-breaker.

- [ ] **Step 3: Preserve nose depth and copy transverse center only**

Keep:

```python
center_local = legacy.copy()
center_local[transverse] = component_center[transverse]
```

Store the profile setback in the existing `nose_gap_m` field so runtime logging remains backward compatible.

- [ ] **Step 4: Improve zero-candidate failure text**

Raise a message that distinguishes dimensional mismatch from setback rejection:

```python
raise RuntimeError(
    "No qualified connector body profile matched the port cross-section "
    "within the maximum setback."
)
```

- [ ] **Step 5: Run focused tests**

```bash
~/isaacsim/python.sh -m unittest -v tests.test_connector_tcp
```

Expected: PASS.

- [ ] **Step 6: Commit implementation**

```bash
git add single_rack_cv/connector_tcp.py single_rack_cv/tests/test_connector_tcp.py
git commit -m "feat: derive tcp from rear housing profile"
```

### Task 3: Update USD diagnostics and wiring contract

**Files:**
- Modify: `single_rack_cv/connector_tcp_usd.py`
- Modify: `single_rack_cv/tests/test_connector_tcp_runtime_wiring.py`

**Interfaces:**
- Consumes: accepted `InsertionTcpDerivation` from Task 2.
- Produces: logs that call the value a profile setback, show the selected donor, and preserve marker-only motion lock.

- [ ] **Step 1: Add a wiring test for the 20 mm setback gate**

Assert the USD adapter calls `derive_insertion_tcp` with:

```python
maximum_profile_setback_m=0.020
```

and retains `TCP_PROBE_ONLY = True`.

- [ ] **Step 2: Update diagnostic terminology**

Change `nose gap mm` to `profile setback mm` in accepted logs. In rejected-component reports, show `profile_setback_mm` and compare it to 20.0 mm.

- [ ] **Step 3: Run focused wiring tests**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_connector_tcp \
  tests.test_connector_tcp_runtime_wiring
```

Expected: PASS.

- [ ] **Step 4: Commit diagnostics**

```bash
git add single_rack_cv/connector_tcp_usd.py single_rack_cv/tests/test_connector_tcp_runtime_wiring.py
git commit -m "fix: report rear-body tcp profile setback"
```

### Task 4: Verify marker-only release candidate

**Files:**
- Verify only; no production changes unless a test exposes a defect.

**Interfaces:**
- Produces: a branch head safe for one workstation marker probe.

- [ ] **Step 1: Compile modified modules**

```bash
~/isaacsim/python.sh -m py_compile connector_tcp.py connector_tcp_usd.py scale_aware_cable_mount.py
```

Expected: no output and exit status 0.

- [ ] **Step 2: Run focused tests**

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_connector_tcp \
  tests.test_connector_tcp_runtime_wiring
```

Expected: all pass.

- [ ] **Step 3: Run full suite**

```bash
~/isaacsim/python.sh -m unittest discover -s tests -t . -p 'test_*.py' -v
```

Expected: zero failures and zero errors; the workstation fixture may remain intentionally skipped when its environment variable is absent.

- [ ] **Step 4: Confirm safety lock statically**

Verify `TCP_PROBE_ONLY = True` and the probe hold loop occurs before `detector.initialize()`.

- [ ] **Step 5: Push and request one marker-only workstation run**

The acceptable runtime result is:

```text
selected component: ...E_part006_44...
body cross-section mm: approximately [10.495, 7.135]
profile setback mm: approximately 12.670
physical TCP shift magnitude: <= 3.0 mm
red and cyan markers visibly separated
cyan marker centered in the rectangular insertable housing
YOLOE, handoff, and insertion remain locked
```
