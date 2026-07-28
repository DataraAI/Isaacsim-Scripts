# Partial Cable Insertion Design

## Goal
After the existing RGB stereo visual servo completes at its qualified 50 mm pre-insert standoff, move the pregrasped RJ45 connector to the opening and then 10 mm into the detected port, then hold.

## Scope
This milestone proves controlled approach and partial port entry. It does not seat the connector, release the cable, retreat the robot, or keep vision active during insertion.

## Preconditions
Motion may start only when all of the following are true:

- visual servo reports complete
- final ToolCenter tracking error is at or below 0.3 mm
- cable fixed joint is valid
- the built-in deformable-tail attachment is preserved
- live plug-tip mount error is at or below 0.5 mm
- live plug-axis error is at or below 1 degree

## Frozen insertion frame
At motion start, capture and freeze:

- the current ToolCenter position
- the current ToolCenter orientation
- the insertion axis as ToolCenter local +Z expressed in world coordinates

The axis and orientation remain fixed for the entire motion. No further perception updates alter the target.

## Two-stage motion
The visual-servo completion pose is 50 mm in front of the physical opening. The controller therefore commands 60 mm of total axial travel:

1. **Coarse approach:** 40 mm in eight 5 mm steps, ending 10 mm before the opening.
2. **Fine approach and insertion:** 20 mm in forty 0.5 mm steps, crossing the remaining 10 mm to the opening and continuing 10 mm into the port.

Every target is computed from the frozen start pose, never accumulated from the measured pose. After each command, wait until physical ToolCenter error is at or below 0.3 mm for 6 consecutive simulation frames before issuing the next command. Each step has a 2.0 second timeout. Orientation remains exactly fixed. After 60 mm total travel, hold the final target.

## Abort conditions
Stop issuing new targets and hold the current target when any of these occur:

- lateral drift from the frozen axis exceeds 0.5 mm
- orientation error from the frozen orientation exceeds 1 degree
- live plug-tip mount error exceeds 0.5 mm
- live plug-axis error exceeds 1 degree
- Lula IK rejects a candidate target or raises
- target publication fails
- a single step does not settle within 2.0 seconds
- the fixed joint becomes invalid
- the deformable-tail attachment is no longer preserved

No automatic retreat is included. An abort holds the latest published target and prints the measured failure reason.

## Runtime structure
The post-alignment controller uses these lifecycle states:

1. `WAITING_FOR_ALIGNMENT`
2. `READY`
3. `ADVANCING`
4. `COMPLETE`
5. `ABORTED`

Each command is also labeled as either `COARSE_APPROACH` or `FINE_INSERTION`. The visual-servo controller remains unchanged and relinquishes target ownership only after final physical settle.

## Diagnostics
Print one structured block at start, after every settled command, on abort, and on completion. Include:

- motion stage
- command index and total command count
- total commanded travel from the visual pose
- commanded depth relative to the port opening
- actual axial travel and actual depth relative to the opening
- lateral drift
- ToolCenter tracking error
- orientation error
- plug-tip mount error
- plug-axis error
- settle count and timeout frames

## Testing

### Pure tests
- first command is a 5 mm coarse approach step
- eight settled coarse commands end exactly 40 mm from the frozen start
- the next command changes to 0.5 mm fine motion
- all 48 commands finish exactly 60 mm from the frozen start
- final commanded depth relative to the opening is exactly +10 mm
- lateral drift and orientation violations abort
- timeout, mount, topology, IK, and publication failures enter `ABORTED`
- no target is issued before visual-servo completion or after `COMPLETE`/`ABORTED`

### Isaac Sim qualification
A passing run must show:

- the existing mount and visual-servo qualification still passes
- all eight 5 mm coarse commands settle
- all forty 0.5 mm fine commands settle
- the stage transition occurs at 40.0 mm total travel
- final total travel is 60.0 mm
- final depth relative to the opening is +10.0 mm
- lateral drift remains at or below 0.5 mm
- orientation error remains at or below 1 degree
- plug mount remains within existing limits
- the runtime holds without seating, release, or retreat

## Kill switch
If the first workstation run shows growing lateral error, connector-rim collision, mount displacement, or repeated timeouts, stop and inspect the frozen axis and opening geometry before increasing tolerances, speed, or depth.