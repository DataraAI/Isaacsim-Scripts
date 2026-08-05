# Calibrate Visible Front-Lip Width Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept the correctly outlined visible RJ45 front opening by calibrating its production width prior to 15.3 mm without changing the detected center or any motion/safety behavior.

**Architecture:** Keep `PerceptionConfig.port_width_m` as the narrower internal-port model. Add a dedicated visible-front-lip size to `FrontPlaneRuntimeConfig` and pass it only into the plane-rectified front-lip estimator. The production value is 15.3 mm, derived from the live run's 91 measurements (median 15.287 mm). Existing stereo-plane estimation, center reconstruction, ToolCenter handoff, TCP, and insertion gates remain unchanged.

**Tech Stack:** Python 3, dataclasses, NumPy, OpenCV, unittest, Isaac Sim 6.0.

## Global Constraints

- Do not add a manual pixel or world-space center offset.
- Do not change `/World/EstimatedPortPoint` or `/World/FrozenPortPoint` semantics.
- Keep the 0.5 mm stereo center-disagreement gate unchanged.
- Keep the 48-command insertion sequence and +10 mm terminal depth unchanged.
- Keep the 0.5 mm lateral and 1 degree orientation abort limits unchanged.
- Keep the position-hold runtime unchanged.

---

### Task 1: Add a production visible-front-lip calibration

**Files:**
- Modify: `single_rack_cv/config.py`
- Modify: `single_rack_cv/main.py`
- Test: `single_rack_cv/tests/test_visible_front_lip_calibration.py`

**Interfaces:**
- Produces: `FrontPlaneRuntimeConfig.visible_front_lip_width_m: float = 0.0153`
- Produces: `FrontPlaneRuntimeConfig.visible_front_lip_height_m: float = 0.0070`
- Consumes: `refine_live_observation(..., aperture_width_m, aperture_height_m)`

- [ ] **Step 1: Write the failing production-wiring test**

```python
from config import CONFIG


def test_visible_front_lip_uses_live_calibration():
    assert CONFIG.front_plane.visible_front_lip_width_m == 0.0153
    assert CONFIG.front_plane.visible_front_lip_height_m == 0.0070
    source = (ROOT / "main.py").read_text()
    assert "CONFIG.front_plane.visible_front_lip_width_m" in source
    assert "CONFIG.front_plane.visible_front_lip_height_m" in source
    assert "aperture_width_m=CONFIG.perception.port_width_m" not in source
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_visible_front_lip_calibration
```

Expected: FAIL because the dedicated front-lip fields do not exist.

- [ ] **Step 3: Add the dedicated calibration and wire production**

Add to `FrontPlaneRuntimeConfig`:

```python
visible_front_lip_width_m: float = 0.0153
visible_front_lip_height_m: float = 0.0070
```

Change `main.py` so `refine_live_observation` receives those two fields instead of `CONFIG.perception.port_width_m` and `CONFIG.perception.port_height_m`.

- [ ] **Step 4: Run the production-wiring test and verify it passes**

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_visible_front_lip_calibration
```

Expected: PASS.

### Task 2: Prove the calibrated visible box is accepted

**Files:**
- Modify: `single_rack_cv/tests/test_plane_rectified_front_lip.py`

**Interfaces:**
- Consumes: `fit_rectified_front_lip(rectified, aperture_width_m=0.0153, aperture_height_m=0.0070)`
- Produces: regression coverage for a 15.3 mm by 7.0 mm visible opening.

- [ ] **Step 1: Add a 15.3 mm visible-opening regression**

Construct a rectified synthetic image at 0.05 mm/pixel with a 306 px wide by 140 px high dark opening. Call the fitter with `aperture_width_m=0.0153` and `aperture_height_m=0.0070`. Assert the fitted width is within 0.2 mm of 15.3 mm and the center remains within 0.1 mm of the image center.

- [ ] **Step 2: Run the focused perception tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_visible_front_lip_calibration \
  tests.test_plane_rectified_front_lip
```

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run:

```bash
~/isaacsim/python.sh -m unittest discover -s tests -t . -p 'test_*.py' -v
```

Expected: PASS with zero failures.

### Task 3: Workstation live verification

**Files:**
- No code changes.

- [ ] **Step 1: Launch Isaac Sim**

```bash
~/isaacsim/python.sh main.py
```

- [ ] **Step 2: Verify perception qualification**

Within the first 20 valid captures, require at least three accepted `[RGB FRONT LIP]` records with widths near the 15.3 mm cluster and three stable stationary samples. Reject the run if the fitted center moves away from the visible rectangle center.

- [ ] **Step 3: Verify motion**

Require the frozen 50 mm handoff, position-hold completion, all 48 insertion commands, approximately +10 mm final depth, lateral drift at or below 0.5 mm, orientation error at or below 1 degree, and no abort.
