# Plane-Rectified RGB Physical Aperture Center

The live controller computes `/World/EstimatedPortPoint` from the physical front lip of the wide lower RJ45 insertion mouth. The previous mask-only estimator is no longer used for control because the two angled semantic silhouettes do not describe the same physical plane.

## Runtime Geometry

1. YOLOE supplies synchronized left/right RGB images and semantic masks.
2. The existing dense stereo outer-bezel estimator reconstructs the physical rack-face plane.
3. Calibrated camera image-right and image-down directions define a metric coordinate frame on that measured plane.
4. Each RGB eye is independently rectified onto the common plane at 0.05 mm per pixel.
5. The masks define only the rectified search envelope. They do not define the four lip boundaries or the final center.
6. Signed RGB gradients locate the physical left lip, right lip, upper shoulder, and lower lip.
7. Each eye independently fits a convex quadrilateral. Opposite edges must be within 5 degrees before a weak shared-slope parallel fit is applied.
8. Each eye must satisfy the 1.5 px maximum edge-reprojection residual and the configured aperture dimensions may only reject implausible width or height.
9. The two independent metric centers must agree within 0.5 mm before their inlier edge samples are pooled.
10. A joint robust refit produces the final physical center and converts it back to world coordinates.

Configured 11.4 mm by 7.0 mm aperture dimensions are validation bounds only. They never translate a boundary or center.

Runtime does not use a rack transform, port prim, RTX ray hit, USD ground truth, empirical pixel correction, fixed world-space offset, threshold relaxation, or single-eye fallback.

## Frozen Control Behavior

This perception change does not alter:

- the 30 degree angled Franka hand;
- the horizontal RJ45 insertion axis;
- camera mounts or calibration;
- cable FixedJoint or deformable-tail attachment;
- three-sample stationary qualification;
- 50 mm pre-insert standoff;
- 5 mm maximum kinematic handoff step;
- 48-command two-stage insertion sequence;
- 0.5 mm lateral-drift limit;
- 1 degree orientation-error limit;
- orientation hold, arm stiffness, damping, or force limits.

## Diagnostics

Every rectified attempt overwrites the latest available progressive files in `camera_output`:

- `front_lip_rectified_left.png`
- `front_lip_rectified_right.png`
- `front_lip_fit_left.png`
- `front_lip_fit_right.png`
- `front_lip_reprojection_left_eye_fit.png`
- `front_lip_reprojection_right_eye_fit.png`

An accepted pair additionally saves:

- `front_lip_fit_joint.png`
- `front_lip_reprojection_left.png`
- `front_lip_reprojection_right.png`

Accepted captures print one `[RGB FRONT LIP]` line containing independent-eye disagreement, per-eye residuals, per-eye dimensions, joint dimensions, and edge-support counts. Rejected captures retain specific errors such as weak support, nonparallel edges, implausible dimensions, excessive reprojection residual, or center disagreement.

Debug-image write failures are reported but do not override a valid geometric measurement.

## Offline Verification

Run the focused geometry and wiring tests:

```bash
cd ~/Isaacsim-Scripts/single_rack_cv

~/isaacsim/python.sh -m unittest -v \
  tests.test_plane_rectified_front_lip \
  tests.test_plane_rectified_runtime_wiring \
  tests.test_front_rim_plane_runtime_wiring
```

To replay the exact August 4 images already stored in `camera_output`:

```bash
FRONT_LIP_FIXTURE_DIR="$PWD/camera_output" \
  ~/isaacsim/python.sh -m unittest -v \
  tests.test_plane_rectified_front_lip_workstation_fixture
```

The exact uploaded pair qualifies offline with approximately 0.052 mm independent-center disagreement and maximum edge residuals of approximately 0.552 px and 0.725 px. This is an offline regression result, not workstation qualification.

## Workstation Qualification

Run:

```bash
cd ~/Isaacsim-Scripts
git pull --ff-only
cd single_rack_cv
~/isaacsim/python.sh main.py
```

Keep PR #9 draft until one stationary Isaac Sim run proves all of the following:

- at least 12 of the first 20 valid YOLOE captures pass all front-lip geometry gates;
- three stationary center samples qualify;
- the 1 mm `/World/EstimatedPortPoint` marker is visibly centered in the physical lower mouth and lies on the front plane;
- `/World/FrozenPortPoint` remains at that physical center;
- the 50 mm kinematic handoff completes;
- all 48 insertion commands settle;
- final depth is approximately +10 mm;
- lateral drift remains at or below 0.5 mm;
- orientation error remains at or below 1 degree.

A rare single accepted capture is a failed qualification, even if it permits motion.
