# Alignment Stress Runtime Isolation Note

The approved behavior and qualification gates are unchanged.

During implementation, passive instrumentation was moved from direct edits to `sim.py` into a stress-only `InstrumentedSimulationRuntime` subclass in `stress_runtime.py`.

Reasons:

- `sim.py` remains byte-for-byte identical to `main` for the normal interactive runtime.
- `main.py` selects the subclass only when the complete stress argument set is present.
- The subclass delegates control decisions to `SimulationRuntime` through `super()`.
- Target-step measurement observes the target immediately before and after the canonical `observe_visual_servo` call.
- No duplicate correction calculation, target command, orientation command, or insertion path was added.

This is an implementation-boundary change only. The 3x3 Y/Z matrix, three repeats, seed, fresh-process isolation, timeout values, result schemas, parent-only ground-truth scoring, safety gates, and 27/27 kill switch remain as approved.
