# Plane-Rectified RGB Front-Lip Center Design

## Problem

The current angled-view runtime uses the semantic aperture masks to fit a lower-mouth quadrilateral independently in each eye. The implementation discards the RGB image and treats the two dark silhouettes as perspective projections of the same planar boundary.

The August 4 workstation run disproves that assumption:

- only capture 4 qualified;
- the remaining captures were almost all rejected;
- typical reconstructed left/right center disagreement was approximately 1.0-2.0 mm;
- the unchanged safety gate is 0.5 mm;
- several frames also lost lower-mouth horizontal support.

The two masks do not represent the same physical plane. Each silhouette contains a different mixture of the front lip, recessed cavity, latch notch, side wall, and self-occlusion. A projective-center calculation can be mathematically correct for each silhouette while still reconstructing two different physical points.

The robot, mount, hand pose, cable geometry, drive tuning, and insertion controller are not the cause. They passed startup validation. The failure occurs before handoff in the image-to-center stage.

## Decision

Keep the existing dense stereo outer-bezel estimator as the source of the physical rack-face plane. Replace the mask-only lower-mouth center with a plane-rectified RGB front-lip estimator.

The semantic masks will localize the port and define a search region only. The physical center will come from RGB evidence belonging to the front lip after both camera images are mapped into one common metric plane.

Runtime must not use:

- a rack transform;
- a port prim;
- an RTX ray hit;
- USD ground truth;
- an empirical image-space correction;
- a fixed world-space X/Y/Z offset;
- a relaxed 0.5 mm stereo-consistency gate;
- a single-eye fallback.

Configured RJ45 dimensions may be used only as plausibility bounds. They must not translate the estimated center.

## Frozen Subsystems

This change must not alter:

- the 30 degree angled Franka hand;
- the horizontal RJ45 insertion axis;
- hand-to-ToolCenter geometry;
- camera mounts or camera calibration;
- cable FixedJoint or deformable-tail attachment;
- three-sample stationary handoff;
- 50 mm pre-insert standoff;
- 5 mm maximum kinematic handoff step;
- 48-command two-stage insertion sequence;
- orientation hold;
- arm stiffness, damping, or force limits;
- 0.5 mm lateral-drift limit;
- 1 degree orientation-error limit.

## Geometry Pipeline

### 1. Estimate the physical front plane

Use the current dense stereo outer-bezel estimator without changing its depth-selection, spatial-support, plane-residual, or ray-gap gates.

The returned plane provides:

- a world-space origin;
- a camera-facing normal;
- dense support diagnostics;
- the physical depth of the port mouth.

### 2. Build a camera-derived metric plane frame

Construct a right-handed 2D coordinate frame on the measured plane using only calibrated camera geometry:

- project the average stereo-camera image-right direction onto the plane to obtain the horizontal axis;
- orthogonalize and normalize that axis;
- derive the vertical axis from the plane normal and horizontal axis;
- choose axis signs to match the average camera image-right and image-down directions.

No rack or port orientation is read from USD.

### 3. Rectify each RGB eye onto the common plane

For each eye:

1. Project the semantic-mask contour rays onto the measured plane.
2. Form a metric search region from the projected contour envelope plus 6 mm padding on all sides.
3. Sample the RGB image onto that plane at 0.05 mm per rectified pixel using calibrated projection and bilinear interpolation.
4. Produce a visibility mask so invalid or occluded samples cannot become edge evidence.
5. Normalize local luminance and contrast per eye before computing gradients.

The mask is allowed to define the search region. It is not allowed to define the final lip boundaries or center.

### 4. Detect physical front-lip boundaries per eye

In each rectified RGB patch, detect signed bezel-to-opening gradients for four physical boundaries:

- left front lip;
- right front lip;
- upper shoulder of the wide lower mouth;
- lower front lip.

Fit each boundary robustly from RGB gradient samples. The fitter must reject weak, fragmented, or cavity-only evidence rather than selecting the strongest edge blindly.

Because rectification removes perspective, opposite boundaries should be approximately parallel in plane coordinates. Parallelism is a validation condition and a weak fitting regularizer, not a fixed image-space rectangle assumption.

### 5. Preserve independent stereo evidence

Each eye must independently produce a metric front-lip quadrilateral and center on the common plane.

Reject the frame unless:

- both eyes produce four valid front-lip boundaries;
- both quadrilaterals are convex;
- each center lies inside its quadrilateral;
- the two metric centers disagree by at most 0.5 mm;
- each eye's fitted edges reproject onto qualified RGB gradients with at most 1.5 px maximum residual;
- opposite-edge angular disagreement is at most 5 degrees;
- measured width is between 70% and 130% of the configured 11.4 mm aperture width;
- measured height is between 70% and 130% of the configured 7.0 mm aperture height.

This keeps the current stereo-consistency principle. A joint fit is not allowed to hide two disagreeing eye estimates.

### 6. Fuse accepted RGB evidence

After both independent eye fits pass, pool their inlier boundary samples in the common metric plane and perform one robust joint refit of the four physical boundaries.

Compute the final center from the diagonal intersection of the joint metric quadrilateral. Convert that center back to world coordinates using the measured plane frame.

The joint fit improves precision only after stereo agreement has been proven. It must not rescue a rejected eye pair.

### 7. Feed the existing runtime contract

Return the same center and diagnostic contract currently consumed by `live_control_projective.py` and the stationary handoff.

The final world point becomes:

- `/World/EstimatedPortPoint` during live acquisition;
- one sample in the unchanged stationary three-sample qualification;
- `/World/FrozenPortPoint` after qualification.

No controller, handoff, or insertion behavior changes.

## Diagnostics

Save the following for every processed capture:

- left rectified RGB patch;
- right rectified RGB patch;
- left and right signed edge maps;
- per-eye fitted boundary overlays;
- per-eye quadrilateral corners and centers;
- joint fitted boundary overlay and center;
- final center reprojected into both original RGB images;
- per-edge support counts;
- per-eye edge residuals;
- per-eye measured width and height;
- per-eye center disagreement in millimeters;
- outer-plane residual and ray gap.

The runtime log must distinguish failures such as weak edge support, implausible dimensions, reprojection failure, and center disagreement. It must not collapse them into one generic estimator error.

## Tests

### Real uploaded-pair regression

Create cropped test fixtures from the exact August 4 uploaded left/right RGB images and masks.

The regression must first demonstrate that the current mask-only estimator fails the 0.5 mm stereo-consistency requirement on this pair. The new estimator must then satisfy all of the following without a correction vector:

- both rectified eye fits succeed;
- per-eye metric centers disagree by at most 0.5 mm;
- both reprojected centers lie on the physical front-lip opening in the RGB fixtures;
- the joint center remains inside both fitted quadrilaterals;
- configured dimensions are used only by validation code.

Human or simulation ground-truth annotations may be used only to score the offline test. They must never enter runtime calculations.

### Synthetic geometry tests

- Two perspective views of one planar mouth must recover the same metric center.
- A recessed cavity edge stronger than the front lip must not win.
- Changing the latch-notch mask while preserving RGB must move the center by less than 0.1 mm.
- Independent brightness and contrast changes between eyes must preserve the center within 0.1 mm.
- A one-eye cavity-only fit must fail closed.
- A one-eye occlusion that removes a required physical boundary must fail closed.
- Opposite edges that are not approximately parallel after rectification must be rejected.
- Width or height outside the configured plausibility bands must be rejected.
- Eye centers separated by more than 0.5 mm must be rejected before joint fitting.

### Runtime wiring tests

- Runtime must use the plane-rectified RGB estimator.
- Runtime must not use `lower_mouth_projective_center.aperture_center_pixel` for control.
- Semantic masks must be passed only as localization/search-region inputs.
- Outer-bezel plane estimation and all existing plane safety gates must remain unchanged.
- Three-sample qualification, frozen handoff, and insertion limits must remain unchanged.
- No empirical pixel or world correction constant may be introduced.

## Offline Acceptance Gate

Runtime wiring must not change until the exact uploaded-pair regression passes.

Required offline result:

- per-eye center disagreement at most 0.5 mm;
- maximum RGB edge reprojection residual at most 1.5 px per eye;
- no manual correction;
- all existing focused perception tests pass.

If the uploaded pair fails, stop. Do not tune the live controller or loosen safety gates.

## Workstation Acceptance Gate

The implementation remains unqualified until Isaac Sim proves all of the following from a stationary run:

- at least 12 of the first 20 captures with valid YOLOE detections pass the geometry gates;
- three stationary center samples qualify;
- the 1 mm `/World/EstimatedPortPoint` marker is visually centered on the physical lower mouth and lies on the front plane;
- the frozen point remains at that physical center;
- stereo handoff reaches the 50 mm pre-insert pose;
- all 48 insertion commands settle;
- final insertion depth is approximately +10 mm;
- lateral drift remains at or below 0.5 mm;
- orientation error remains at or below 1 degree.

A rare single accepted frame is a failure even if it happens to permit insertion.

## Kill Switch

Reject this architecture if the exact uploaded RGB pair cannot produce two independently valid front-lip fits within 0.5 mm after correct plane rectification.

The next honest alternatives would be:

1. change the camera viewpoint so both physical lips are visible; or
2. fit a calibrated 3D CAD aperture template as a measurement model, with simulation truth used only for scoring.

Do not return to mask-fraction tuning, fixed offsets, threshold relaxation, or single-eye control.
