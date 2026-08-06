# Enable Full Derived-TCP Insertion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the proven 48-command, +10 mm guarded insertion sequence while keeping the mesh-derived RJ45 insertion TCP and all existing safety gates.

**Architecture:** Select `settled_stereo_handoff_runtime.AngledHandStereoHandoffRuntime` in `main.py` instead of the temporary 2 mm precontact wrapper. Keep mesh-derived TCP calibration and probe-lock-on-derivation-failure unchanged. Disable only the precontact-only mode flag; preserve the existing 40 mm coarse stage, 20 mm fine stage, 0.5 mm lateral limit, 1 degree orientation limit, mount validation, IK checks, and step timeouts.

**Tech Stack:** Python 3.12, NumPy, unittest, Isaac Sim 6.0.0, OpenUSD.

## Global Constraints

- Keep `/World/EstimatedPortPoint` and `/World/FrozenPortPoint` unchanged.
- Keep the derived TCP `[nose coordinate, rear-body transverse center]` calibration unchanged.
- Keep `TCP_PROBE_ONLY = False`; rejected derivations must still force the motion-locked probe path.
- Restore exactly 48 commands: eight 5 mm coarse commands plus forty 0.5 mm fine commands.
- Final commanded port depth must be +10.000 mm.
- Do not relax the 0.5 mm lateral or 1 degree orientation limits.

---

### Task 1: Write the full-insertion runtime-selection tests

**Files:**
- Modify: `single_rack_cv/tests/test_precontact_runtime_wiring.py`
- Modify: `single_rack_cv/tests/test_connector_tcp_runtime_wiring.py`

**Interfaces:**
- Consumes: `main.py`, `connector_tcp_usd.py`, `scale_aware_cable_mount.py` source text.
- Produces: regression assertions that full insertion is selected while rejected TCP derivations remain motion-locked.

- [ ] **Step 1: Replace precontact-selection assertions with full-runtime assertions**

Assert that `main.py` imports `settled_stereo_handoff_runtime`, does not import `precontact_runtime`, and still contains the connector-TCP probe lock before detector initialization.

- [ ] **Step 2: Assert the connector mode flags select full insertion**

Assert `TCP_PROBE_ONLY = False` and `PRECONTACT_ALIGNMENT_ONLY = False`, while the rear-profile donor, marker paths, 20 mm profile-setback gate, and no-world-offset requirements remain present.

- [ ] **Step 3: Run the focused tests and confirm they fail before production edits**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_precontact_runtime_wiring \
  tests.test_connector_tcp_runtime_wiring
```

Expected: failures showing `precontact_runtime` is still selected and `PRECONTACT_ALIGNMENT_ONLY` is still `True`.

- [ ] **Step 4: Commit the red tests**

```bash
git add single_rack_cv/tests/test_precontact_runtime_wiring.py \
        single_rack_cv/tests/test_connector_tcp_runtime_wiring.py
git commit -m "test: require full insertion with derived TCP"
```

### Task 2: Restore the full guarded runtime

**Files:**
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/connector_tcp_usd.py`

**Interfaces:**
- Consumes: `settled_stereo_handoff_runtime.AngledHandStereoHandoffRuntime`, `PRECONTACT_ALIGNMENT_ONLY`.
- Produces: the existing 48-command insertion controller with mesh-derived TCP calibration.

- [ ] **Step 1: Select the settled full-insertion runtime**

Replace the import in `main.py` with:

```python
from settled_stereo_handoff_runtime import (
    AngledHandStereoHandoffRuntime as CableMountedSimulationRuntime,
)
```

- [ ] **Step 2: Disable precontact-only mode**

Set:

```python
TCP_PROBE_ONLY = False
PRECONTACT_ALIGNMENT_ONLY = False
PRECONTACT_HOLD_OFFSET_M = 0.002
```

Keep the hold offset constant for the dormant diagnostic module; do not delete the precontact implementation.

- [ ] **Step 3: Run the focused tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_precontact_runtime_wiring \
  tests.test_connector_tcp_runtime_wiring \
  tests.test_two_stage_insertion \
  tests.test_two_stage_runtime_wiring \
  tests.test_startup_geometry_settle
```

Expected: all pass.

- [ ] **Step 4: Run the full pure test suite**

Run:

```bash
~/isaacsim/python.sh -m unittest discover \
  -s tests -t . -p 'test_*.py' -v
```

Expected: zero failures and zero errors; the workstation fixture may remain intentionally skipped unless its environment variable is set.

- [ ] **Step 5: Commit the production switch**

```bash
git add single_rack_cv/main.py single_rack_cv/connector_tcp_usd.py
git commit -m "feat: enable full insertion with derived TCP"
```

### Task 3: Verify the live guarded insertion

**Files:**
- Runtime output: `single_rack_cv/camera_output/run_output_latest.txt`

**Interfaces:**
- Consumes: qualified stereo port pose and mesh-derived TCP.
- Produces: one live run ending at +10 mm without an abort.

- [ ] **Step 1: Confirm startup mode**

The log must contain the 48-command two-stage entry and must not contain `PRECONTACT ALIGNMENT SAFETY MODE ACTIVE`.

- [ ] **Step 2: Run Isaac Sim**

```bash
~/isaacsim/python.sh main.py
```

- [ ] **Step 3: Enforce the completion gate**

Require:

```text
48/48 commands settled
actual port depth approximately +10 mm
lateral drift <= 0.5 mm
orientation error <= 1 degree
ToolCenter tracking error <= 0.3 mm
no mount, IK, attachment, timeout, fatal, or insertion abort
```

- [ ] **Step 4: Keep the PR draft if the live gate fails**

Do not relax thresholds or add a world/vision offset. Diagnose the first failed physical metric instead.
