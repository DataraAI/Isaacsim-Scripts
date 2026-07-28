# Partial Cable Insertion Design

## Goal
After the existing RGB stereo visual servo completes and the ToolCenter is physically settled, advance the pregrasped RJ45 connector 10 mm into the detected port, then hold.

## Scope
This milestone proves controlled port entry only. It does not seat the connector, release the cable, retreat the robot, or keep vision active during insertion.

## Preconditions
Insertion may start only when all of the following are true:

- visual servo reports complete
- final ToolCenter tracking error is at or below 0.3 mm
- cable fixed joint is valid
- the built-in deformable-tail attachment is preserved
- live plug-tip mount error is at or below 0.5 mm
- live plug-axis error is at or below 1 degree

## Frozen insertion frame
At insertion start, capture and freeze:

- the current ToolCenter position
- the current ToolCenter orientation
- the insertion axis as ToolCenter local +Z expressed in world coordinates

The axis and orientation remain fixed for the entire partial insertion. No further perception updates alter the insertion target.

## Motion

- total insertion depth: 10 mm
- command increment: 0.5 mm
- number of increments: 20
- each new target is computed from the frozen start pose, not accumulated from the measured pose
- after each command, wait until physical ToolCenter error is at or below 0.3 mm for consecutive settled frames before issuing the next increment
- orientation remains exactly the frozen insertion orientation
- after 10 mm, hold the final target

## Abort conditions
Stop issuing new targets and hold the current target when any of these occur:

- lateral drift from the frozen insertion axis exceeds 0.5 mm
- orientation error from the frozen orientation exceeds 1 degree
- live plug-tip mount error exceeds 0.5 mm
- live plug-axis error exceeds 1 degree
- Lula IK returns failure
- a single insertion step does not settle before its timeout
- the fixed joint becomes invalid
- the deformable-tail attachment is no longer preserved

No automatic retreat is included in this milestone. An abort holds the latest safe target and prints the measured failure reason.

## Runtime structure
Add an insertion state machine after visual-servo completion:

1. `WAITING_FOR_ALIGNMENT`
2. `READY`
3. `ADVANCING`
4. `COMPLETE`
5. `ABORTED`

The visual-servo controller remains unchanged. The insertion controller owns target updates only after visual-servo completion.

## Diagnostics
Print one structured insertion block at start, after every settled increment, on abort, and on completion. Include:

- commanded depth
- actual axial depth
- lateral drift
- ToolCenter tracking error
- orientation error
- plug-tip mount error
- plug-axis error
- step settle count and timeout state

## Testing

### Pure tests
- frozen-axis target generation reaches exactly 10 mm in twenty 0.5 mm increments
- lateral drift calculation rejects off-axis motion above 0.5 mm
- orientation error rejects motion above 1 degree
- timeout and structural failures enter `ABORTED`
- no target is issued before visual-servo completion
- no target is issued after `COMPLETE` or `ABORTED`

### Isaac Sim qualification
A passing run must show:

- the existing visual-servo qualification still passes
- all 20 increments settle
- final commanded insertion depth is 10.0 mm
- lateral drift remains at or below 0.5 mm
- orientation error remains at or below 1 degree
- plug mount remains within existing limits
- the runtime holds at 10 mm without seating or release

## Kill switch
If the first workstation run shows growing lateral error, connector-rim collision, mount displacement, or repeated step timeouts, stop insertion work and inspect the frozen axis and port geometry before increasing tolerances or depth.
