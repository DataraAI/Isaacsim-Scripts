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
- separation of the hand control frame from the plug insertion frame
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

The configurable pitch is supported from 0 through 45 degrees. The accepted milestone value is exactly 30 degrees. Negative values and values above 45 degrees are rejected.

## Separate control and insertion frames

The current validated system treats ToolCenter local +Z and the plug nose axis as the same direction. That assumption becomes false after the hand is pitched and the plug is counter-pitched.

The implementation must therefore maintain two explicit frames:

1. **Hand control frame**
   - Position: remains coincident with the plug insertion tip.
   - Orientation: follows the pitched Franka hand.
   - Purpose: Lula IK target orientation and live ToolCenter tracking.

2. **Plug insertion frame**
   - Origin: the rigid plug insertion tip.
   - +Z axis: the live rigid plug nose axis, horizontal toward the port.
   - Purpose: pre-insert standoff geometry, depth measurement, approach direction, and insertion direction.

No code may infer the insertion direction from ToolCenter local +Z once the pitch offset is nonzero.

At zero pitch, the two frames must reduce to the existing validated geometry.

## Geometry architecture

The final world-space plug pose is preserved by splitting the orientation into two transforms:

1. Command the Franka ToolCenter and hand with a 30 degree downward pitch.
2. Apply an equal and opposite 30 degree pitch to the hand-to-plug mount transform.

The resulting transform chain must satisfy:

- world-space plug nose axis equals the existing horizontal insertion axis
- world-space plug body remains level
- hand-to-plug relative pitch equals 30 degrees within 0.5 degrees
- hand-control-frame origin and plug-tip origin differ by no more than the existing 0.5 mm mount-tip limit
- plug-axis error from the required horizontal insertion axis remains within the existing 1 degree mount-axis limit

The implementation must derive the counter-pitched mount from transforms, not from an unexplained quaternion constant.

## Camera behavior

Both RGB cameras remain rigidly attached to the Franka hand and therefore inherit the new pitched hand pose.

The implementation must not fake the old camera pose or overwrite the cameras independently. Existing stereo projection, camera-model construction, and target transforms must use the live camera extrinsics produced by the pitched hand.

The physical stereo baseline remains unchanged because both cameras retain their existing rigid offsets on the same hand.

No manual image offsets, depth offsets, or port-specific pixel corrections may be introduced to compensate for the new viewpoint.

## Visual-servo behavior

The existing automatic visual-servo pipeline remains structurally unchanged:

- synchronized left/right RGB capture
- YOLOE port detection
- stereo correspondence and triangulation
- automatic front-opening plane refinement
- translation-only stop-and-look corrections
- final physical ToolCenter settle gate

The desired pre-insert geometry must change from “port is 50 mm along ToolCenter local +Z” to “port is 50 mm along the plug insertion frame +Z axis.”

The visual controller may update hand-control-frame position only. It must hold the pitched hand orientation fixed. The plug remains horizontal because its counter-pitched mount is fixed relative to the hand.

The controller must still converge from live vision without USD or RTX ground-truth assistance.

## Insertion behavior

After visual alignment completes, the existing insertion sequence remains:

- freeze the settled hand-control-frame position
- freeze the settled pitched hand orientation
- freeze the validated live plug nose axis as the horizontal insertion axis
- execute 40 mm coarse approach in eight 5 mm commands
- execute 20 mm fine motion in forty 0.5 mm commands
- finish 10 mm inside the physical opening
- hold the final hand-control-frame target

Every new hand-control-frame position equals the frozen start position plus commanded depth along the frozen plug insertion axis. The hand orientation never changes during insertion.

The implementation must not use pitched ToolCenter local +Z for translation and must not accumulate targets from measured motion.

## Configuration

Add one explicit configurable hand-to-plug pitch value with a default of 30 degrees.

The configuration name and documentation must make the direction unambiguous. It must describe the accepted geometry as:

- robot-right-side view
- wrist higher
- fingertips lower toward the port
- plug horizontal

The same value must drive both the commanded hand orientation and inverse plug mount calculation so the two transforms cannot silently diverge.

## Diagnostics

Startup diagnostics must print:

- configured hand-to-plug pitch in degrees
- measured relative hand-to-plug pitch
- hand-control-frame +Z axis
- plug insertion-frame +Z axis
- plug-axis error from horizontal
- hand-control-frame origin to plug-tip error
- fixed-joint validity
- built-in deformable-attachment validity

The geometry qualification block must explicitly state:

- hand pitch: pass or fail
- wrist-higher/fingertips-lower sign: pass or fail
- plug horizontal: pass or fail
- control and insertion frames separated: active or zero-pitch compatibility
- fixed joint: valid or invalid
- built-in attachment: preserved or changed

## Abort and failure behavior

Do not start visual servo or insertion if any geometry precondition fails.

Reject the run when any of these occur:

- configured pitch is non-finite, negative, or above 45 degrees
- measured relative hand-to-plug pitch differs from the configured value by more than 0.5 degrees
- the selected sign does not place the wrist above the fingertips in the robot-right-side view
- plug horizontal-axis error exceeds 1 degree
- hand-control-frame origin to plug-tip error exceeds 0.5 mm
- fixed joint is invalid
- built-in deformable attachment is not preserved
- live camera transform construction fails
- plug insertion-axis construction fails
- Lula IK rejects the pitched hand pose

During visual servo and insertion, retain all existing detector, stereo, tracking, timeout, lateral-drift, orientation, mount, IK, publication, and topology safeguards.

## Testing

### Pure geometry tests

- zero pitch reproduces the existing validated transform and aligned control/insertion axes
- 30 degree hand pitch plus inverse 30 degree plug pitch preserves the original world-space plug orientation
- measured hand-to-plug relative pitch is 30 degrees within 0.5 degrees
- plug insertion axis remains horizontal within 1 degree
- hand-control-frame origin and plug-tip origin remain coincident within numerical tolerance
- the selected sign produces wrist higher and fingertips lower in the robot-right-side view
- opposite-sign pitch fails the directional contract test
- insertion translation follows plug-frame +Z, not hand-control-frame +Z
- invalid pitch values are rejected

### Structural wiring tests

- configuration exposes one shared pitch value
- hand orientation and plug mount both consume the shared value
- visual pre-insert standoff uses the plug insertion axis
- partial insertion uses the plug insertion axis
- Lula target orientation uses the hand control frame
- no independent camera-pose override is added
- visual servo remains translation-only
- insertion distances, step counts, and existing safety limits remain unchanged
- README documents the robot-right-side-view convention

### Isaac Sim geometry qualification

Before running the full task, a passing geometry-only startup must show:

- hand pitched downward by 30 degrees from the robot-right-side view
- wrist higher than fingertips
- rigid plug horizontal within 1 degree
- hand-control-frame origin to plug-tip error at or below 0.5 mm
- measured relative pitch within 0.5 degrees of 30 degrees
- cable fixed joint valid
- built-in deformable attachment preserved

### Full-task qualification

A passing workstation run must then show:

- cable mount validation passes 30/30 frames
- stereo track acquisition succeeds from the pitched camera pose
- visual alignment locks using live RGB stereo
- final physical ToolCenter tracking error remains within the existing 0.3 mm completion gate
- all 48 approach and insertion commands settle
- final commanded depth relative to the opening is +10 mm
- final measured depth is within 0.3 mm of +10 mm
- lateral drift remains at or below 0.5 mm
- hand-orientation error remains at or below 1 degree
- plug horizontal-axis error remains at or below 1 degree
- the runtime holds without seating, release, or retreat

## Kill switch

Stop and inspect geometry rather than loosening thresholds when any of the following appears:

- the plug tilts with the hand instead of remaining horizontal
- the wrong pitch sign places the wrist below the fingertips
- translation follows the pitched hand axis instead of the horizontal plug axis
- stereo correspondence degrades persistently from the new camera perspective
- visual servo fails to converge cleanly
- mount errors increase materially
- insertion lateral drift grows toward the existing 0.5 mm limit
- port-rim contact appears before the expected opening crossing

The validated `main` behavior remains the rollback point until the complete pitched-hand workstation qualification passes.
