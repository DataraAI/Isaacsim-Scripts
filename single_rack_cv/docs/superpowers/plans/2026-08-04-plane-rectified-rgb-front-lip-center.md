# Plane-Rectified RGB Front-Lip Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mask-only angled-view RJ45 center with an independently validated, plane-rectified RGB front-lip estimator that passes the exact August 4 stereo regression before runtime wiring changes.

**Architecture:** Preserve `outer_bezel_center.py` as the physical front-plane source. Add a focused module that builds a camera-derived metric plane frame, rectifies each RGB eye into that plane, detects the four physical lower-mouth lip boundaries from signed RGB gradients, validates two independent eye fits, and only then performs a joint metric refit. Expose the existing `OuterBezelApertureResult` contract so `live_control_projective.py` and the handoff controller remain unchanged.

**Tech Stack:** Python 3.11, NumPy, OpenCV, existing calibrated camera models, `unittest`, Isaac Sim 6.0.0 runtime.

## Global Constraints

- Keep the 30 degree angled Franka hand and horizontal RJ45 insertion axis unchanged.
- Keep camera mounts and camera calibration unchanged.
- Keep the dense stereo outer-bezel plane estimator and its safety gates unchanged.
- Keep the 0.5 mm independent-eye center-disagreement gate unchanged.
- Keep the 1.5 px maximum RGB edge reprojection residual.
- Keep the three-sample stationary handoff, 50 mm standoff, 5 mm handoff step, 48 insertion commands, 0.5 mm lateral-drift limit, and 1 degree orientation-error limit unchanged.
- No rack transform, port prim, RTX ray, USD ground truth, empirical pixel correction, world-space offset, relaxed threshold, or single-eye fallback.
- Configured 11.4 mm by 7.0 mm aperture dimensions are validation bounds only and must never translate the estimated center.
- Runtime wiring must not change until the exact uploaded-image regression passes.

---

### Task 1: Commit Exact August 4 Regression Fixtures

**Files:**
- Create: `single_rack_cv/tests/fixtures/aug_04_front_lip/rgb_left.png`
- Create: `single_rack_cv/tests/fixtures/aug_04_front_lip/rgb_right.png`
- Create: `single_rack_cv/tests/fixtures/aug_04_front_lip/mask_left.png`
- Create: `single_rack_cv/tests/fixtures/aug_04_front_lip/mask_right.png`
- Create: `single_rack_cv/tests/fixtures/aug_04_front_lip/metadata.json`
- Create: `single_rack_cv/tests/test_plane_rectified_front_lip_regression.py`

**Interfaces:**
- Consumes: Uploaded 1280x960 RGB images, binary masks, calibrated left/right camera metadata from the August 4 run.
- Produces: `_load_aug_04_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CameraModel, CameraModel, PlaneFixture]` for focused offline tests.

- [ ] **Step 1: Add the exact binary fixtures and camera/plane metadata**

Store only the files required to reproduce the failing stationary stereo pair. `metadata.json` must contain image dimensions, intrinsics, `world_from_camera` matrices, and the accepted capture-4 outer-plane origin and normal. Do not store a target center or correction vector.

- [ ] **Step 2: Write the failing regression proving the mask-only estimator violates the 0.5 mm gate**

```python
def test_aug_04_mask_only_centers_disagree_beyond_safety_gate():
    fixture = _load_aug_04_fixture()
    left_uv = lower_mouth_projective_center.aperture_center_pixel(
        fixture.left_rgb, fixture.left_mask, fixture.left_camera
    )
    right_uv = lower_mouth_projective_center.aperture_center_pixel(
        fixture.right_rgb, fixture.right_mask, fixture.right_camera
    )
    left_world = intersect_pixel_with_plane(
        fixture.left_camera, left_uv, fixture.plane_origin, fixture.plane_normal
    )
    right_world = intersect_pixel_with_plane(
        fixture.right_camera, right_uv, fixture.plane_origin, fixture.plane_normal
    )
    assert np.linalg.norm(left_world - right_world) > 0.0005
```

- [ ] **Step 3: Run the focused test and verify the historical failure is reproduced**

Run:

```bash
cd single_rack_cv
python -m unittest -v \
  tests.test_plane_rectified_front_lip_regression
```

Expected: the mask-only regression passes because the recorded pair exceeds 0.5 mm; the new-estimator test is still absent.

- [ ] **Step 4: Commit the fixtures and historical regression**

```bash
git add single_rack_cv/tests/fixtures/aug_04_front_lip \
  single_rack_cv/tests/test_plane_rectified_front_lip_regression.py
git commit -m "test: capture angled RGB front-lip regression"
```

---

### Task 2: Add Camera-Derived Plane Frame and Rectification

**Files:**
- Create: `single_rack_cv/plane_rectified_front_lip.py`
- Create: `single_rack_cv/tests/test_plane_rectified_front_lip.py`

**Interfaces:**
- Consumes: `camera.world_from_camera`, `camera.project_world`, `camera.pixel_to_world_ray`, plane origin and normal, RGB image, semantic mask.
- Produces:
  - `PlaneFrame(origin_world_m, axis_u_world, axis_v_world, normal_world)`
  - `build_plane_frame(left_camera, right_camera, plane_origin_world_m, plane_normal_world) -> PlaneFrame`
  - `rectify_eye_to_plane(rgb, mask, camera, plane_frame, padding_m=0.006, resolution_m=0.00005) -> RectifiedEye`

- [ ] **Step 1: Write failing tests for axis orientation and metric rectification**

```python
def test_plane_frame_is_orthonormal_and_camera_aligned():
    frame = build_plane_frame(left_camera, right_camera, origin, normal)
    self.assertAlmostEqual(np.linalg.norm(frame.axis_u_world), 1.0, places=9)
    self.assertAlmostEqual(np.linalg.norm(frame.axis_v_world), 1.0, places=9)
    self.assertAlmostEqual(float(frame.axis_u_world @ frame.axis_v_world), 0.0, places=9)
    self.assertGreater(frame.axis_u_world @ average_image_right_world, 0.0)
    self.assertGreater(frame.axis_v_world @ average_image_down_world, 0.0)


def test_rectification_maps_one_metric_point_to_same_plane_pixel_in_both_eyes():
    left = rectify_eye_to_plane(...)
    right = rectify_eye_to_plane(...)
    self.assertLess(np.linalg.norm(left.metric_to_pixel(point_uv_m) - right.metric_to_pixel(point_uv_m)), 1e-9)
```

- [ ] **Step 2: Run tests and verify they fail because the module does not exist**

```bash
cd single_rack_cv
python -m unittest -v tests.test_plane_rectified_front_lip
```

Expected: import failure for `plane_rectified_front_lip`.

- [ ] **Step 3: Implement the minimal immutable data classes and plane-frame construction**

The frame must derive its axes only from the average calibrated camera image-right/image-down directions projected onto the measured plane. Reject degenerate projections.

- [ ] **Step 4: Implement mask-envelope projection and bilinear RGB rectification**

The semantic mask may set the rectified bounds only. The output must include RGB, luminance, visibility, metric origin, resolution, and the camera projection map used for diagnostics.

- [ ] **Step 5: Run focused geometry tests**

```bash
python -m unittest -v tests.test_plane_rectified_front_lip
```

Expected: all plane-frame and rectification tests pass.

- [ ] **Step 6: Commit the geometry primitive**

```bash
git add single_rack_cv/plane_rectified_front_lip.py \
  single_rack_cv/tests/test_plane_rectified_front_lip.py
git commit -m "feat: rectify stereo RGB onto measured front plane"
```

---

### Task 3: Fit One Independent RGB Front-Lip Quadrilateral per Eye

**Files:**
- Modify: `single_rack_cv/plane_rectified_front_lip.py`
- Modify: `single_rack_cv/tests/test_plane_rectified_front_lip.py`

**Interfaces:**
- Consumes: `RectifiedEye`, configured aperture width and height.
- Produces:
  - `FrontLipFit(left_line, right_line, top_line, bottom_line, corners_uv_m, center_uv_m, support_counts, residual_px, width_m, height_m)`
  - `fit_rectified_front_lip(rectified, aperture_width_m=0.0114, aperture_height_m=0.0070) -> FrontLipFit`

- [ ] **Step 1: Write failing synthetic tests for physical-edge selection**

Add tests where a recessed cavity edge is stronger than the front lip, latch-notch mask geometry changes while RGB stays fixed, one eye loses a required boundary, and opposite edges are non-parallel.

- [ ] **Step 2: Verify the new tests fail for missing fitting behavior**

```bash
python -m unittest -v tests.test_plane_rectified_front_lip
```

Expected: failures identify missing `fit_rectified_front_lip` behavior.

- [ ] **Step 3: Implement local luminance normalization and signed-gradient candidate extraction**

Use the rectified semantic envelope only to define broad search bands. Select exterior front-lip transitions by polarity and spatial ordering; do not use a global strongest-gradient `argmin` or `argmax`.

- [ ] **Step 4: Implement robust line fitting and support validation**

Require six or more inlier samples per edge, reject fragmented support, enforce maximum 1.5 px edge residual, validate convexity, and ensure the center lies inside the quadrilateral.

- [ ] **Step 5: Implement metric geometry validation**

Require opposite-edge angular disagreement at most 5 degrees, width in `[0.70 * 0.0114, 1.30 * 0.0114]`, and height in `[0.70 * 0.0070, 1.30 * 0.0070]`. Dimensions may only reject; they may not shift any line or center.

- [ ] **Step 6: Run the focused tests**

```bash
python -m unittest -v tests.test_plane_rectified_front_lip
```

Expected: all synthetic physical-edge and failure-mode tests pass.

- [ ] **Step 7: Commit the independent per-eye fitter**

```bash
git add single_rack_cv/plane_rectified_front_lip.py \
  single_rack_cv/tests/test_plane_rectified_front_lip.py
git commit -m "feat: fit physical RGB front lip per rectified eye"
```

---

### Task 4: Add Stereo Validation, Joint Refit, and Exact Uploaded-Pair Gate

**Files:**
- Modify: `single_rack_cv/plane_rectified_front_lip.py`
- Modify: `single_rack_cv/tests/test_plane_rectified_front_lip.py`
- Modify: `single_rack_cv/tests/test_plane_rectified_front_lip_regression.py`

**Interfaces:**
- Consumes: two `FrontLipFit` values and one `PlaneFrame`.
- Produces:
  - `PlaneRectifiedFrontLipResult(center_world_m, left_fit, right_fit, joint_fit, center_disagreement_m, plane_frame)`
  - `estimate_plane_rectified_front_lip_center(...) -> PlaneRectifiedFrontLipResult`

- [ ] **Step 1: Write failing tests for the unchanged 0.5 mm independent-eye gate**

```python
def test_joint_fit_cannot_rescue_disagreeing_eye_centers():
    with self.assertRaisesRegex(RuntimeError, "disagree"):
        fuse_front_lip_fits(left_fit, right_fit, plane_frame, max_center_disagreement_m=0.0005)
```

- [ ] **Step 2: Write the exact uploaded-pair acceptance test**

```python
def test_aug_04_plane_rectified_rgb_pair_passes_safety_gate_without_correction():
    fixture = _load_aug_04_fixture()
    result = estimate_plane_rectified_front_lip_center(...)
    self.assertLessEqual(result.center_disagreement_m, 0.0005)
    self.assertLessEqual(result.left_fit.residual_px, 1.5)
    self.assertLessEqual(result.right_fit.residual_px, 1.5)
    self.assertTrue(point_in_convex_quad(result.joint_fit.center_uv_m, result.left_fit.corners_uv_m))
    self.assertTrue(point_in_convex_quad(result.joint_fit.center_uv_m, result.right_fit.corners_uv_m))
```

- [ ] **Step 3: Run both suites and verify the uploaded-pair test fails before fusion is implemented**

```bash
python -m unittest -v \
  tests.test_plane_rectified_front_lip \
  tests.test_plane_rectified_front_lip_regression
```

Expected: the new estimator test fails for missing stereo fusion, while the historical mask-only failure remains reproduced.

- [ ] **Step 4: Implement independent-eye gate and joint inlier refit**

Compute eye-center distance in metric plane coordinates and reject above 0.5 mm before pooling samples. Jointly refit the four lines from accepted inliers, then calculate the final diagonal-intersection center and world point.

- [ ] **Step 5: Add original-image reprojection validation**

Reproject each eye's fitted boundary samples and final center through calibrated cameras. Reject maximum qualified-edge residual above 1.5 px.

- [ ] **Step 6: Run the exact offline acceptance gate**

```bash
python -m unittest -v \
  tests.test_plane_rectified_front_lip \
  tests.test_plane_rectified_front_lip_regression \
  tests.test_lower_mouth_projective_center \
  tests.test_outer_bezel_center
```

Expected: all tests pass; uploaded pair reports center disagreement at or below 0.5 mm with no correction.

- [ ] **Step 7: Commit the qualified stereo estimator**

```bash
git add single_rack_cv/plane_rectified_front_lip.py \
  single_rack_cv/tests/test_plane_rectified_front_lip.py \
  single_rack_cv/tests/test_plane_rectified_front_lip_regression.py
git commit -m "feat: fuse qualified plane-rectified RGB front lips"
```

---

### Task 5: Wire the Qualified Estimator into the Existing Runtime Contract

**Files:**
- Modify: `single_rack_cv/outer_bezel_projective_center.py`
- Modify: `single_rack_cv/live_control_projective.py`
- Modify: `single_rack_cv/debug.py`
- Modify: `single_rack_cv/tests/test_runtime_wiring.py`
- Modify: `single_rack_cv/tests/test_front_rim_plane_runtime_wiring.py`
- Create: `single_rack_cv/tests/test_plane_rectified_runtime_wiring.py`

**Interfaces:**
- Consumes: existing `estimate_outer_bezel_plane(...)` output and the new `estimate_plane_rectified_front_lip_center(...)` result.
- Produces: unchanged `OuterBezelApertureResult` consumed by `apply_front_plane_result(...)`.

- [ ] **Step 1: Write failing runtime-wiring tests**

Assert runtime control imports the new estimator, does not call `lower_mouth_projective_center.aperture_center_pixel`, preserves `estimate_outer_bezel_plane`, and preserves all handoff/insertion constants.

- [ ] **Step 2: Run wiring tests and verify they fail before production wiring changes**

```bash
python -m unittest -v \
  tests.test_plane_rectified_runtime_wiring \
  tests.test_runtime_wiring \
  tests.test_front_rim_plane_runtime_wiring
```

- [ ] **Step 3: Replace only the center-estimation portion of `outer_bezel_projective_center.py`**

Estimate the outer plane exactly as before, call the plane-rectified RGB estimator, and populate the existing result contract. Do not change plane selection, controller inputs, or safety constants.

- [ ] **Step 4: Add capture diagnostics and debug images**

Save rectified patches, edge maps, fitted overlays, joint fit, and original-image reprojections. Log support counts, residuals, dimensions, and independent-eye center disagreement with distinct failure messages.

- [ ] **Step 5: Run wiring and focused perception suites**

```bash
python -m unittest -v \
  tests.test_plane_rectified_front_lip \
  tests.test_plane_rectified_front_lip_regression \
  tests.test_plane_rectified_runtime_wiring \
  tests.test_runtime_wiring \
  tests.test_front_rim_plane_runtime_wiring \
  tests.test_outer_bezel_center \
  tests.test_live_control
```

Expected: all pass.

- [ ] **Step 6: Run Python compilation for every changed runtime module**

```bash
python -m py_compile \
  single_rack_cv/plane_rectified_front_lip.py \
  single_rack_cv/outer_bezel_projective_center.py \
  single_rack_cv/live_control_projective.py \
  single_rack_cv/debug.py
```

Expected: no output and exit code 0.

- [ ] **Step 7: Commit runtime wiring**

```bash
git add single_rack_cv/plane_rectified_front_lip.py \
  single_rack_cv/outer_bezel_projective_center.py \
  single_rack_cv/live_control_projective.py \
  single_rack_cv/debug.py \
  single_rack_cv/tests/test_plane_rectified_runtime_wiring.py \
  single_rack_cv/tests/test_runtime_wiring.py \
  single_rack_cv/tests/test_front_rim_plane_runtime_wiring.py
git commit -m "feat: use plane-rectified RGB center in live control"
```

---

### Task 6: Final Verification and Workstation Handoff

**Files:**
- Modify: `single_rack_cv/docs/plane_rectified_aperture_center.md`
- Modify: `single_rack_cv/docs/superpowers/specs/2026-08-04-plane-rectified-rgb-front-lip-center-design.md` only if implementation details require factual clarification.

**Interfaces:**
- Consumes: all completed tasks.
- Produces: one tested branch head and exact workstation qualification commands.

- [ ] **Step 1: Run the complete non-Isaac focused suite**

```bash
cd single_rack_cv
python -m unittest -v \
  tests.test_plane_rectified_front_lip \
  tests.test_plane_rectified_front_lip_regression \
  tests.test_plane_rectified_runtime_wiring \
  tests.test_runtime_wiring \
  tests.test_front_rim_plane_runtime_wiring \
  tests.test_outer_bezel_center \
  tests.test_live_control \
  tests.test_stereo_handoff \
  tests.test_consecutive_pose_insertion \
  tests.test_fine_insertion_settling \
  tests.test_orientation_hold
```

Expected: all pass with zero failures and zero errors.

- [ ] **Step 2: Update operator documentation with exact workstation command and acceptance lines**

Document:

```bash
cd ~/Isaacsim-Scripts
git pull --ff-only
cd single_rack_cv
~/isaacsim/python.sh main.py
```

Required workstation evidence:

- at least 12 of the first 20 valid YOLOE captures pass geometry;
- three stationary samples qualify;
- estimated and frozen markers are visibly centered on the physical lower mouth and on the front plane;
- 50 mm handoff completes;
- all 48 insertion commands settle;
- final depth is approximately +10 mm;
- lateral drift is at most 0.5 mm;
- orientation error is at most 1 degree.

- [ ] **Step 3: Re-run focused tests after documentation and final diff review**

```bash
python -m unittest -v \
  tests.test_plane_rectified_front_lip \
  tests.test_plane_rectified_front_lip_regression \
  tests.test_plane_rectified_runtime_wiring
```

Expected: all pass.

- [ ] **Step 4: Commit documentation**

```bash
git add single_rack_cv/docs/plane_rectified_aperture_center.md
git commit -m "docs: qualify plane-rectified RGB front-lip control"
```

- [ ] **Step 5: Keep PR #9 draft until workstation qualification passes**

Do not mark the PR ready or merge based only on offline tests.
