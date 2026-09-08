# UR10e 1x Via 82 to Via 95 Slowdown Design

## Goal

Reduce the joint-space speed of only the `port-via-0.82` to
`port-via-0.95` transition in `ur10e_1x_cable_insertion`. The transition
will use 480 interpolation steps instead of the generic 120 steps, making
its nominal speed approximately one quarter of the current speed.

## Design

Add a transition-specific step-count mapping to the one-arm configuration.
The mapping key is the ordered pair of source and destination approach
fractions, and its initial entry is `(0.82, 0.95): 480`.

While queuing phase-two port approach waypoints, track the previous approach
fraction and look up the ordered transition before selecting `joint_steps`.
Use the configured override when present. Otherwise preserve the existing
defaults: 120 steps for intermediate approach segments and 160 steps for the
final approach waypoint.

This change does not alter waypoint geometry, orientation, tolerances,
timeouts, cable-hold monitoring, insertion timing, or the six-arm workflow.

## Verification

Add a host-side unit test for step selection. It will verify that:

- the `0.82` to `0.95` transition selects 480 steps;
- an ordinary intermediate transition selects 120 steps; and
- the final transition selects 160 steps.

Run the one-arm host-side test suite after implementation.
