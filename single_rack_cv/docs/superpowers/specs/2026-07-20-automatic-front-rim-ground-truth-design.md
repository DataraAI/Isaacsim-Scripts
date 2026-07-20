# Automatic Front-Rim Ground-Truth Design

**Date:** 2026-07-20  
**Status:** Approved

## Goal

Eliminate manual port-plane calibration. The robot must estimate the front opening from stereo RGB, while Isaac Sim geometry automatically validates whether the estimate lies on the physical opening plane.

## Control Path

The production controller remains image-only:

1. YOLOE selects the target port in both eyes.
2. The dense front-rim extractor fits top, right, bottom, and left rim lines inside the expanded YOLOE ROI.
3. Corresponding rim samples are triangulated.
4. A robust 3D plane and rectangle are fitted.
5. The fitted rectangle center drives `/World/EstimatedPortPoint`, lateral correction, range correction, and translation-only visual servoing.

No USD prim transform, collision query, raycast result, authored marker, or benchmark JSON may influence robot motion.

## Automatic Validation Path

The benchmark-only validator:

1. Offsets the detected rim samples a few pixels outward so rays land on the solid bezel rather than enter the cavity.
2. Averages corresponding left/right pixels into the virtual-camera image.
3. Casts PhysX rays from the virtual camera through those ring pixels.
4. Keeps rack hits only.
5. Selects the dominant front-depth cluster, rejecting recessed cavity hits.
6. Robustly fits a camera-facing front bezel plane.
7. Intersects the detected opening-center ray with that plane.
8. Writes the resulting reference point and normal to `benchmarks/front_rim_ground_truth.json` for scoring only.

## Failure Behavior

- No manual fallback.
- No cavity-depth offset.
- No robot motion from validation geometry.
- If too few rack hits exist or the fitted plane fails its residual/normal gates, the tool exits with an explicit failure.
- `CONFIG.front_rim.enabled` remains `False` until frozen-frame and live qualification pass.

## Files

- Create `automatic_port_ground_truth.py`.
- Create `tests/test_automatic_port_ground_truth.py`.
- Create `tools/extract_front_rim_ground_truth.py`.
- Delete `tools/calibrate_front_rim_ground_truth.py`.
- Update `docs/front_rim_benchmark.md`.

## Acceptance

- Zero manual transforms or terminal confirmation.
- Pure geometry tests pass.
- Automatic extractor writes schema version 2 with a camera-facing normal, plane residual, hit counts, and used rack prim paths.
- Benchmark data is never imported by production modules.
