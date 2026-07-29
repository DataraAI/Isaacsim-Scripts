# Corrected PR Summary

This branch now preserves the exact validated plug-tip world pose and solves a new 30-degree downward hand pose around it. It no longer applies a guessed 180-degree local palm roll.

The corrected solver replaces the fixed startup hand position, fixed startup hand orientation, and hand-to-tool rotation together. Local verification passes 20/20 geometry, insertion-axis, and runtime-wiring tests plus Python compilation.

The PR remains draft and unmergeable until the Isaac Sim workstation run passes the mount, palm-side, plug-horizontal, stereo, and 48-command insertion gates.
