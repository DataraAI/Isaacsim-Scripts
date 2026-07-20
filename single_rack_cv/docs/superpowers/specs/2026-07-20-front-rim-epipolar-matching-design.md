# Front-Rim Epipolar Patch Matching Design

## Goal

Replace index-to-index pairing of independently generated left/right bezel samples with verified image correspondences before triangulation.

## Constraints

- Runtime control remains image-only.
- RTX/USD ground truth remains benchmark-only and must never influence matching or control.
- `CONFIG.front_rim.enabled` remains `False` until all qualification gates pass.
- Isaac Sim 6.0.0, Ubuntu 24.04, and the existing NumPy/OpenCV stack remain the target environment.
- The existing Prompt B cavity detector remains unchanged.

## Architecture

`front_rim.py` continues to create left and right cavity-anchored bezel rings. A new pure module, `front_rim_match.py`, uses the refined cavity-center displacement as a coarse stereo prediction, then independently matches every left bezel sample to the right image in a narrow epipolar search window.

Each sample match uses local normalized patch correlation over intensity and image gradients. Matches must pass minimum texture, score, uniqueness, and right-to-left consistency gates. The matcher returns the matched right coordinates plus a validity mask and diagnostics.

`front_rim_stereo.py` consumes this explicit correspondence result. It triangulates only valid matched samples, robustly fits the front-bezel plane, and intersects the left/right opening-center rays with that plane. It no longer assumes equal normalized positions in independently sized boxes are physical correspondences.

## Matching flow

1. Convert both images to grayscale and compute Sobel X/Y channels.
2. Compute the coarse center shift as `right_center_uv - left_center_uv`.
3. For each left bezel sample, predict its right location using that shift.
4. Search ±4 px horizontally and ±2 px vertically at 0.5 px increments.
5. Score candidates using normalized correlation of 7×7 intensity/Sobel descriptors.
6. Require minimum descriptor texture, best score, and best-vs-second-best margin.
7. Search back from the chosen right point into the left image.
8. Require round-trip error ≤0.75 px.
9. Return accepted right coordinates and diagnostics; rejected samples remain invalid.

## Validation

Pure tests cover known subpixel translation recovery, low-texture rejection, ambiguity rejection, and integration with the existing triangulator. The frozen 60-pair benchmark remains the final qualification test.

Qualification gates remain unchanged:

- pair success rate ≥95%
- track switches =0
- radial 3D jitter ≤0.5 mm
- ray-gap p95 ≤0.5 mm
- median plane error ≤0.5 mm
- plane-error p95 ≤1.0 mm

## Kill switch

If verified epipolar matching cannot reach 80% pair success on the frozen dataset without weakening geometric gates, stop tuning it and evaluate local SGBM instead.
