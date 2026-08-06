# Front-lip Width Hypothesis Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the left eye from alternating among the outer bezel, physical mouth, and inner cavity while preserving independent stereo qualification.

**Architecture:** Keep the existing per-search RGB fitter and all of its geometry gates. For each eye, evaluate five bounded side-search widths from 5.0 to 11.4 mm, select the qualified fit closest to the 12.9 mm physical visible-mouth width, and reject the eye if no candidate is within 1.0 mm. The independent-eye 0.5 mm center-disagreement gate remains unchanged.

**Tech Stack:** Python, NumPy, OpenCV, Isaac Sim 6.0 runtime.

## Global Constraints

- Do not use USD, RTX ray hits, rack coordinates, or single-eye fallback.
- Do not move the perception point, 50 mm handoff, or insertion calibration.
- Keep the 0.5 mm stereo disagreement gate unchanged.
- Keep the existing residual, parallelism, support, orientation, and insertion safety gates unchanged.

---

### Task 1: Reproduce the three-edge ambiguity

**Files:**
- Modify: `single_rack_cv/tests/test_front_lip_left_bezel_rejection.py`

- [x] Create a synthetic rectified eye containing a farther outer-bezel edge, a 12.9 mm physical mouth, and a nearer inner-cavity edge.
- [x] Verify the old 5.0 mm search selects a roughly 10 mm pair and the old wide search selects the outer bezel.
- [x] Require production to recover the physical center and width.

### Task 2: Add bounded width hypotheses

**Files:**
- Create: `single_rack_cv/plane_rectified_width_hypotheses.py`
- Modify: `single_rack_cv/plane_rectified_front_lip.py`

- [x] Generate five monotonic search widths from 5.0 to 11.4 mm.
- [x] Run the unchanged independent fitter at each width.
- [x] Rank candidates by physical-width error, then semantic-mask center proximity.
- [x] Fail closed when the best width differs from 12.9 mm by more than 1.0 mm.
- [x] Route both independent eyes through the wrapper before the unchanged stereo gate.

### Task 3: Remove the contaminated calibration

**Files:**
- Modify: `single_rack_cv/front_lip_calibration.py`
- Modify: `single_rack_cv/tests/test_visible_front_lip_calibration.py`
- Modify: `single_rack_cv/tests/test_visible_front_lip_geometry.py`
- Modify: `single_rack_cv/tests/test_front_lip_search_calibration.py`

- [x] Set the physical visible-mouth prior to 12.9 mm from stable 254-260 px overlays at 0.05 mm/px.
- [x] Restore 11.4 mm as the maximum search span.
- [x] Retain the old 15.287 mm cluster only as explicitly rejected outer-bezel evidence.
- [x] Replace stale single-radius and 15.3 mm tests.

### Task 4: Workstation qualification

- [ ] Pull the merged main commit on the Isaac Sim workstation.
- [ ] Run the focused unit tests.
- [ ] Confirm both eyes print width-prior selections near 12.9 mm.
- [ ] Require at least 12 of the first 20 valid captures to pass.
- [ ] Require three stationary samples, handoff completion, and 48/48 insertion commands before freezing again.
