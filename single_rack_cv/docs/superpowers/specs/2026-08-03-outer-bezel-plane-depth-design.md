# Outer-Bezel Plane Depth Design

## Problem

The current runtime estimator in `stereo_front_rim_plane.py` derives four RGB boundary lines from a search region anchored to the dark cavity mask. On the stepped RJ45 geometry, the strongest gradients can belong to the recessed shelf and cavity walls. The resulting 3D plane can be internally consistent while still lying several millimeters behind the physical rack face.

This is why `/World/EstimatedPortPoint` and `/World/FrozenPortPoint` can agree with each other yet both remain visibly inside the cavity. Freezing is working; the depth source is wrong.

## Decision

Use the dark opening contour only to determine the insertion-center ray. Estimate depth from a dense stereo reconstruction of the bright outer bezel/front-panel surface surrounding the opening.

Runtime must not use a rack transform, port prim, RTX ray hit, USD ground truth, or fixed world-space depth offset.

## Geometry Pipeline

1. Detect the stepped RJ45 opening in both RGB eyes using the existing detector and masks.
2. Compute the existing image-space physical opening center for each eye.
3. Build a wider sampling region outside the dark opening mask that targets the visible white bezel/front panel.
4. Compute local stereo disparity in that bezel region.
5. Triangulate valid, left-right-consistent bezel pixels.
6. Select the nearest coherent depth cluster to the cameras, because the physical rack face must be in front of the recessed cavity.
7. Require the selected points to span a two-dimensional image region rather than one narrow edge.
8. Fit a robust plane with the existing 0.5 mm residual and ray-gap safety gates.
9. Intersect each eye's opening-center ray with the fitted outer-bezel plane.
10. Fuse the two 3D center estimates only when their disagreement is at most 0.5 mm.
11. Feed the fused physical opening point into the existing stationary three-sample qualification and frozen-goal handoff.

## Visibility Requirements

The old dense-plane implementation required support on all four bezel sides. That is incompatible with the angled eye-in-hand view because the plug, hand, and perspective can hide one or more sides.

The new estimator will accept partial visibility only when all of the following hold:

- enough valid stereo points survive triangulation and clustering;
- the supporting image pixels span at least two separated bezel regions;
- their 2D covariance has two non-degenerate axes, proving area support rather than a single line;
- the 3D points fit one camera-facing plane within the unchanged residual limit;
- the selected plane is the nearest coherent planar cluster in the bezel search region;
- the two eye-center intersections agree within 0.5 mm.

If these conditions fail, the frame is rejected. The system must not fall back to the recessed four-corner estimator.

## Runtime Changes

- Replace the runtime import of `stereo_front_rim_plane.estimate_stereo_aperture_center` with an outer-bezel-plane estimator built from `front_plane.py` and `aperture_center.py` primitives.
- Keep the current image-space center logic because Y/Z centering is already accurate.
- Preserve the one-time stationary qualification, frozen port marker, bounded 5 mm kinematic approach, and two-stage insertion controller.
- Keep `/World/EstimatedPortPoint` as the live fused outer-plane center.
- Keep `/World/FrozenPortPoint` as the median of three qualified live centers.
- Add debug output for bezel support count, support-region count, 2D support spans, plane residual, plane depth, and eye-center disagreement.

## Safety Invariants

The following remain unchanged:

- 0.5 mm maximum lateral insertion drift;
- 1.0 degree maximum insertion orientation error;
- 0.5 mm stereo center-disagreement gate;
- 0.5 mm maximum triangulation ray gap;
- 0.5 mm maximum fitted-plane residual;
- 5 mm maximum kinematic approach step;
- 50 mm pre-insert standoff;
- 48-command insertion sequence and approximately 10 mm final depth.

No empirical depth correction is allowed.

## Tests

### Pure geometry tests

- A synthetic recessed cavity and nearer outer plane must return the outer plane.
- A dense cluster on only the cavity must be rejected when no valid outer-plane support exists.
- Two or three visible bezel regions with two-dimensional spread must be accepted.
- A single narrow edge or collinear support must be rejected.
- A farther dense cavity cluster must lose to a smaller but sufficiently supported nearer front-plane cluster.
- Per-eye center-ray intersections differing by more than 0.5 mm must fail closed.

### Runtime wiring tests

- Runtime must import the outer-bezel estimator.
- Runtime must not import or fall back to `stereo_front_rim_plane`.
- Stationary qualification and frozen-goal handoff must remain unchanged.
- Existing insertion limits must remain unchanged.

### Workstation acceptance

The change is accepted only when Isaac Sim proves:

- both live and frozen markers visibly lie on the physical opening plane, not inside the cavity;
- three stationary estimates qualify with at most 1 mm opening spread;
- the kinematic approach reaches the 50 mm standoff;
- all 48 insertion commands settle;
- final depth is approximately +10 mm;
- settled lateral drift remains at or below 0.5 mm;
- orientation error remains at or below 1 degree.

## Kill Switch

Reject this approach if the angled stereo pair cannot reconstruct a non-degenerate outer-panel patch in repeated stationary runs. At that point, the honest alternatives are changing the camera viewpoint or using USD/prim geometry; adding a fixed depth offset is not acceptable.
