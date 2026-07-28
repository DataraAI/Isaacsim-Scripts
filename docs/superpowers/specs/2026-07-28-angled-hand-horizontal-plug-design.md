# Angled Hand with Horizontal RJ45 Plug Design

## Goal

Change the validated single-rack insertion scene so the Franka hand is pitched downward by 30 degrees while the rigid RJ45 plug remains horizontal and continues to insert along the same horizontal port axis.

From the robot's right-side view, the wrist must sit higher than the fingertips and the fingers must slope downward toward the port. The flexible cable tail remains deformable and free to hang naturally.

## Scope

This milestone changes only the relative orientation between the Franka hand and the rigid plug while preserving the existing automatic stereo alignment and guarded two-stage insertion behavior.

Included:

- 30 degree downward hand pitch
- equal and opposite plug pitch relative to the hand
- horizontal plug nose and body in world space
- wrist-camera poses inherited from the pitched hand
- full requalification of mount geometry, stereo visual servo, and partial insertion

Not included:

- full seating or latch engagement
- gripper release
- robot retreat
- cable-tail path constraints beyond the existing deformable behavior
- relaxed visual, mount, tracking, or insertion thresholds

## Coordinate convention

The angle is defined from the robot's right-side view.

- The physical port insertion axis remains horizontal.
- The RJ45 plug nose axis remains aligned with that horizontal insertion axis.
- The Franka hand is pitched downward by 30 degrees relative to the plug.
- The wrist is higher than the fingertips.
- The fingertips point downward toward the port.

This is a pitch offset, not a roll around the cable axis.

## Geometry architecture

The final world-space plug pose is preserved by splitting the orientation into two transforms:

1. Command the Franka ToolCenter and hand with a 30 degree downward pitch.
2. Apply an equal and opposite 30 degree pitch to the hand-to-plug mount transform.

The resulting transform chain must satisfy:

- world-space plug nose axis equals the existing horizontal insertion axis
- world-space plug body remains level
- hand-to-plug relative pitch equals 30 degrees
- ToolCenter-to-plug-tip positional coincidence remains within the existing mount tolerance

The implementation must derive the counter-pitched mount from transforms, not from an unexplained quaternion constant.

## Camera behavior

Both RGB cameras remain rigidly attached to the Franka hand and therefore inherit the new pitched hand pose.

The implementation must not fake the old camera pose or overwrite the cameras independently. Existing stereo projection, camera-model construction, and target transforms must use the live camera extrinsics produced by the pitched hand.

No manual image offsets, depth offsets, or port-specific pixel corrections may be introduced to compensate for the new viewpoint.

## Visual-servo behavior

The existing automatic visual-servo pipeline remains structurally unchanged:

- synchronized left/right RGB capture
- YOLOE port detection
- stereo correspondence and triangulation
- automatic front-opening plane refinement
- translation-only stop-and-look corrections
- final physical ToolCenter settle gate

The desired port observation may need to be recomputed from the new ToolCenter and camera geometry, but the controller must still converge from live vision without USD or RTX ground-truth assistance.

The visual servo must retain the existing fixed commanded hand orientation throughout alignment. Vision may correct translation only.

## Insertion behavior

After visual alignment completes, the existing insertion controller remains in charge:

- freeze the settled ToolCenter pose
- freeze the horizontal plug insertion axis
- execute 40 mm coarse approach in eight 5 mm commands
- execute 20 mm fine motion in forty 0.5 mm commands
- finish 10 mm inside the physical opening
- hold the final ToolCenter target

The hand remains pitched throughout the complete insertion. The plug remains horizontal throughout the complete insertion.

Insertion targets must continue to be computed from the frozen start pose and frozen horizontal insertion axis, not accumulated from measured motion.

## Configuration

Add one explicit configurable hand-to-plug pitch value with a default of 30 degrees.

The configuration name and documentation must make the direction unambiguous. It must describe the accepted geometry as:

- right-side view
- wrist higher
- fingertips lower toward the port
- plug horizontal

The value must be used by both the commanded hand orientation and the inverse plug mount calculation so the two transforms cannot silently diverge.

## Diagnostics

Startup diagnostics must print:

- configured hand-to-plug pitch in degrees
- world-space hand pitch
- world-space plug pitch
- measured relative hand-to-plug pitch
- plug-axis error from the required horizontal insertion axis
- ToolCenter-to-plug-tip error

The final geometry qualification block must clearly state whether:

- the hand pitch is correct
- the plug is horizontal
- the fixed joint is valid
- the built-in plug-to-tail attachment is preserved

## Abort and failure behavior

Do not start visual servo or insertion if any geometry precondition fails.

Reject the run when any of these occur:

- configured pitch is non-finite or outside a conservative supported range
- hand-to-plug relative pitch differs from 30 degrees beyond the defined geometry tolerance
- plug horizontal-axis error exceeds the existing mount-axis tolerance
- ToolCenter-to-plug-tip error exceeds the existing mount-tip tolerance
- fixed joint is invalid
- built-in deformable attachment is not preserved
- live camera transform construction fails
- Lula IK rejects the pitched hand pose

During visual servo and insertion, retain all existing detector, stereo, tracking, timeout, lateral-drift, orientation, mount, IK, publication, and topology safeguards.

## Testing

### Pure geometry tests

- zero pitch reproduces the existing validated transform
- 30 degree hand pitch plus inverse 30 degree plug pitch preserves the original world-space plug orientation
- measured hand-to-plug relative pitch is 30 degrees
- plug insertion axis remains horizontal
- ToolCenter and plug-tip position remain coincident within numerical tolerance
- the selected sign produces wrist higher and fingertips lower in the robot-right-side view
- opposite-sign pitch fails the directional contract test

### Structural wiring tests

- configuration exposes one shared pitch value
- hand orientation and plug mount both consume the shared value
- no independent camera-pose override is added
- visual servo remains translation-only
- insertion distances, step counts, and safety limits remain unchanged
- README documents the right-side-view convention

### Isaac Sim geometry qualification

Before running the full task, a passing geometry-only startup must show:

- hand pitched downward by 30 degrees from the robot-right-side view
- wrist higher than fingertips
- rigid plug horizontal
- ToolCenter-to-plug-tip error within the existing mount limit
- plug-axis error within the existing mount limit
- cable fixed joint valid
- built-in deformable attachment preserved

### Full-task qualification

A passing workstation run must then show:

- cable mount validation passes 30/30 frames
- stereo track acquisition succeeds from the pitched camera pose
- visual alignment locks using live RGB stereo
- final physical ToolCenter tracking error remains within the existing completion gate
- all 48 approach and insertion commands settle
- final commanded depth relative to the opening is +10 mm
- final measured depth is within the existing insertion tolerance
- lateral drift remains at or below 0.5 mm
- orientation error remains at or below 1 degree
- the plug remains horizontal at completion
- the hand remains pitched at completion
- the runtime holds without seating, release, or retreat

## Kill switch

Stop and inspect geometry rather than loosening thresholds when any of the following appears:

- the plug tilts with the hand instead of remaining horizontal
- the wrong pitch sign places the wrist below the fingertips
- stereo correspondence degrades persistently from the new camera perspective
- visual servo fails to converge cleanly
- mount errors increase materially
- insertion lateral drift grows toward the existing 0.5 mm limit
- port-rim contact appears before the expected opening crossing

The validated `main` behavior remains the rollback point until the complete pitched-hand workstation qualification passes.
