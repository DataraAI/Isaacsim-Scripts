# Camera-Derived 6-DoF Port Pose Design

## Goal

Restore the accuracy seen from the previous near-horizontal view while keeping the current angled wrist-camera geometry. The controller must derive the complete RJ45 port pose from synchronized stereo RGB alone and remain accurate across view angle changes.

This design forbids empirical correction vectors, hard-coded rack-port coordinates, runtime USD/RTX ground truth, and relaxed safety thresholds.

## Problem Statement

The current runtime estimates a stable 3D opening center, but control remains translation-only and preserves a fixed horizontal plug orientation. That worked when the camera view was close to horizontal because the projected cavity was nearly symmetric and the fixed connector orientation approximately matched the port.

From the current oblique view, a stable point estimate can still be biased. The cavity-mask centroid is not the projective center of the rectangular opening, and a center point alone does not define the port normal or roll. The robot can therefore track the estimated point accurately while entering the opening with a small pose error.

## Scope

Implement a camera-derived 6-DoF port pose consisting of:

- opening center in world coordinates;
- outward-facing port normal;
- horizontal rim axis;
- vertical rim axis;
- a right-handed orthonormal port frame;
- a pre-insert ToolCenter pose at the configured standoff;
- an insertion axis derived from the frozen port normal.

The existing angled hand, stereo hardware model, 50 mm pre-insert standoff, two-stage insertion distances, 0.5 mm lateral-drift limit, 1 degree orientation limit, mount checks, and handoff distance bounds remain unchanged unless a test proves a wiring-only change is required.

## Non-Goals

This work will not:

- add a manual X/Y/Z correction to the estimated port point;
- use the known rack transform, port prim, RTX hit, or USD geometry as runtime input;
- loosen the existing lateral or orientation limits;
- redesign the cable mount or finger presentation;
- solve rack collision topology or physical contact enforcement;
- add an opaque learned pose regressor.

## Architecture

### 1. Per-Eye Inner-Rim Geometry

For each synchronized RGB image:

1. Reuse the existing YOLOE detection and front-plane support mask.
2. Extract edge-support points belonging to the four inner bezel sides.
3. Robustly fit four 2D lines: top, right, bottom, and left.
4. Intersect adjacent lines to recover four ordered image corners.
5. Reject self-intersecting, non-convex, incorrectly ordered, or physically implausible quadrilaterals.

The opening center in each image is the intersection of the two quadrilateral diagonals, not the mask centroid.

### 2. Stereo Corner Triangulation

Match corners by semantic order across both eyes:

- top-left;
- top-right;
- bottom-right;
- bottom-left.

Triangulate each corner independently using the calibrated stereo camera model. Reject the frame when any corner violates epipolar, reprojection, ray-gap, finite-depth, or positive-depth checks.

### 3. Port Frame Construction

Given the four 3D corners:

- center = mean of the four corners;
- raw horizontal axis = mean of right-side corners minus mean of left-side corners;
- raw vertical axis = mean of top-side corners minus mean of bottom-side corners;
- plane normal = cross(horizontal, vertical).

Use a Gram-Schmidt or SVD projection to produce an orthonormal right-handed frame.

Choose the normal sign without world ground truth:

- the outward normal points from the port toward the virtual stereo camera;
- the insertion direction is the negative outward normal.

Reject frames with degenerate axes, reversed corner ordering, excessive non-planarity, or inconsistent width/height.

### 4. Temporal Pose Stability

Maintain a short window of complete port-pose estimates. A handoff candidate is valid only when all of the following pass:

- center spread <= 0.5 mm;
- outward-normal spread <= 0.25 degrees;
- horizontal-axis spread <= 0.25 degrees;
- vertical-axis spread <= 0.25 degrees;
- fitted-plane residual <= 0.5 mm;
- stereo ray gap <= existing configured gate and never above 0.5 mm;
- reprojection error <= existing configured gate;
- all four corners are valid in both eyes;
- measured opening width and height remain physically plausible and temporally consistent;
- the frame remains right-handed.

If any gate fails, hold the current ToolCenter target and reacquire. There is no fallback correction offset.

### 5. 6-DoF Visual Servo

Before handoff, servo both translation and orientation toward the camera-derived pre-insert pose:

- position = port center + outward normal * pre-insert standoff;
- connector insertion axis = negative outward normal;
- connector up/roll axis = camera-derived vertical rim axis.

Translation and orientation steps must remain bounded. Orientation convergence must be checked using an axis-angle or quaternion metric rather than Euler-angle subtraction.

### 6. Frozen Pose Handoff

When the temporal gates pass and the remaining translation enters the bounded handoff region, freeze the complete pose:

- center;
- outward normal;
- horizontal axis;
- vertical axis;
- pre-insert ToolCenter position;
- desired ToolCenter orientation.

The kinematic handoff then converges to this frozen 6-DoF pose. No new camera measurements are blended into the target after freezing.

### 7. Port Entry

The two-stage insertion controller must:

- begin only after the ToolCenter has settled in both position and orientation;
- insert along the frozen negative port normal;
- preserve the frozen port-aligned orientation throughout entry;
- continue enforcing the existing 0.5 mm lateral-drift and 1 degree orientation limits;
- hold immediately on any failed mount, topology, IK, timeout, lateral, or orientation gate.

The controller must not derive the insertion direction from the previous fixed connector axis once a valid frozen port pose exists.

## Visualization

Replace the oversized estimated-port sphere with a diagnostic-only pose marker:

- small center marker with radius no greater than 1 mm;
- red/green/blue axes for the estimated port frame;
- optional corner markers;
- explicit labels for outward normal and insertion direction.

The visualization cannot affect control.

## Data Flow

1. Synchronized left/right RGB frames.
2. YOLOE detection and front-plane support extraction.
3. Per-eye inner-rim line fits.
4. Ordered 2D corner sets.
5. Four stereo-triangulated 3D corners.
6. Orthonormal 6-DoF port pose.
7. Temporal stability window and rejection gates.
8. Bounded translation-and-orientation servo.
9. Frozen 6-DoF handoff target.
10. Two-stage insertion along the frozen camera-derived port normal.

## Error Handling

The runtime must fail closed:

- missing edge support: hold and reacquire;
- ambiguous side labels: reject frame;
- invalid corner intersection: reject frame;
- one-eye-only corner: reject frame;
- bad triangulation: reject frame;
- implausible width/height: reject frame;
- unstable center or axes: hold and reacquire;
- invalid handedness or normal sign: reject frame;
- position or orientation not settled: do not start insertion;
- runtime ground-truth dependency detected: fail tests.

No gate may silently substitute the old centroid or a hard-coded correction.

## Testing Strategy

### Pure Geometry Tests

- perspective-projected rectangle recovers its projective center from diagonal intersection;
- arbitrary camera roll and oblique view preserve corner ordering;
- four triangulated corners reconstruct a known 3D rectangle;
- orthonormal frame construction is right-handed and sign-correct;
- outward normal points toward the virtual camera;
- degenerate or mirrored quadrilaterals are rejected;
- pose is invariant to point ordering after semantic normalization.

### Robustness Tests

- noisy edge points recover center and normal within tolerance;
- missing side support rejects rather than guessing;
- one corrupted corner fails reprojection/ray-gap gates;
- asymmetric cavity texture does not shift the projective center;
- horizontal and angled views of the same synthetic port return the same world pose within tolerance.

### Runtime Wiring Tests

- no production path adds a manual port offset;
- no runtime path reads rack-port ground truth;
- handoff freezes position and orientation together;
- insertion uses the frozen estimated port normal;
- existing safety limits remain unchanged;
- translation-only handoff is no longer selected when a valid pose exists;
- invalid pose holds and reacquires.

### Workstation Qualification

Run both a near-horizontal camera presentation and the current angled presentation. Ground truth may be used only by the benchmark scorer after estimation.

Required for each view:

- center error <= 0.5 mm;
- normal error <= 0.5 degrees;
- roll/up-axis error <= 0.5 degrees;
- temporal center spread <= 0.5 mm;
- temporal axis spread <= 0.25 degrees;
- handoff settles within existing position and orientation gates;
- all 48 insertion commands complete;
- final depth remains approximately +10 mm inside the opening;
- lateral drift remains below 0.5 mm;
- orientation error remains below 1 degree;
- no controller input comes from USD/RTX ground truth or a manual correction vector.

## Acceptance Criteria

The implementation is accepted only when:

1. The camera-derived port pose meets the numerical qualification gates from both horizontal and angled views.
2. The angled-view result is not materially worse than the horizontal-view result.
3. The same code path handles both views without per-view offsets or mode switches.
4. The complete 48-command insertion succeeds from both views.
5. Existing mount, topology, safety, and timeout tests remain green.

## Kill Switch

Do not merge or enable the new runtime path when any of the following occurs:

- center or orientation improves only after adding an empirical correction;
- the angled view fails the same numerical benchmark passed by the horizontal view;
- the estimator is stable but exceeds 0.5 mm center or 0.5 degree axis error;
- insertion requires relaxed safety limits;
- runtime control reads USD/RTX ground truth;
- the 48-command insertion does not complete.

In those cases, retain the qualified translation-only baseline while continuing offline development of the pose estimator.
