# Automatic Front-Rim Ground-Truth Implementation Plan

**Goal:** Replace interactive calibration with automatic bezel-plane raycast validation.

## Task 1: Pure Geometry

- Add tests for outward rim-ring sampling, dominant front-depth selection, robust plane fitting, forward ray-plane intersection, and insufficient-hit rejection.
- Add `automatic_port_ground_truth.py` and make the tests pass.

## Task 2: Isaac Integration

- Add `tools/extract_front_rim_ground_truth.py`.
- Load the existing scene and qualified YOLOE detector.
- Automatically extract both 2D rims.
- Cast virtual-camera rays through the outward rim ring.
- Fit and save the physical front plane without user input.

## Task 3: Remove Manual Calibration

- Delete `tools/calibrate_front_rim_ground_truth.py`.
- Update benchmark documentation and command names.

## Task 4: Verification Gate

Run:

```bash
$HOME/isaacsim/python.sh -m unittest -v \
  tests.test_automatic_port_ground_truth \
  tests.test_front_rim_2d \
  tests.test_front_rim_stereo

$HOME/isaacsim/python.sh tools/extract_front_rim_ground_truth.py
```

Do not integrate the front-rim controller until the automatic extraction and offline benchmark qualify.
