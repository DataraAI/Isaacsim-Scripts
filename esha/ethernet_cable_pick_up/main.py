#!/usr/bin/env python3
"""
Cable-grasp RGB stereo demo entrypoint.

Starts Isaac Sim, then runs synchronized stereo RGB visual servo. Heavy
logic lives in sibling modules:

  config.py       — tunables
  logging_tee.py  — run log capture
  debug.py        — overlays / debug dumps (import after SimulationApp)
  sim.py          — scene, cameras, IK, servo (import after SimulationApp)
  perception.py   — pure NumPy/OpenCV stereo math (no Isaac dependency)
"""

from __future__ import annotations

import traceback

from config import CONFIG
from logging_tee import RunOutputTee
from perception import process_stereo_cable


run_output_path = CONFIG.camera.output_dir / "run_output_latest.txt"
run_output_tee = RunOutputTee(run_output_path)
run_output_tee.start()

simulation_app = None
runtime = None

try:
    print(
        f"[LOG] Saving complete run output to: {run_output_path}",
        flush=True,
    )

    # Isaac Sim must start before importing modules that use omni/pxr APIs.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": CONFIG.app.headless,
            "width": CONFIG.app.width,
            "height": CONFIG.app.height,
        }
    )

    from debug import DebugOutputs
    from sim import SimulationRuntime, warn

    runtime = SimulationRuntime(
        simulation_app=simulation_app,
        cfg=CONFIG,
    )
    debug = DebugOutputs(CONFIG)

    capture_index = 0

    while runtime.is_running():
        runtime.step()

        try:
            runtime.update_ik()
            runtime.update_visual_servo_completion()
        except Exception as exc:
            warn(f"Motion/IK update failed: {exc}")

        if not runtime.capture_due():
            continue

        capture_index += 1

        try:
            frame = runtime.capture()
            debug.save_raw(frame)
            previous_left, previous_right = (
                runtime.visual_servo_references()
            )
            observation = process_stereo_cable(
                frame=frame,
                cfg=CONFIG.perception,
                desired_cable_virtual_camera_usd=(
                    runtime.desired_cable_virtual_camera_usd
                ),
                previous_left=previous_left,
                previous_right=previous_right,
            )
            runtime.observe_visual_servo(observation)
            debug.handle(
                frame,
                observation,
                capture_index,
            )
        except Exception as exc:
            # A rejected stereo pair holds the current target; repeated misses
            # trigger a clean image-space reacquisition.
            runtime.note_perception_failure()
            if "frame" in locals():
                try:
                    debug.save_failure_snapshot(frame, capture_index, str(exc))
                except Exception as snapshot_exc:
                    warn(f"Could not save failure snapshot: {snapshot_exc}")
            warn(
                f"RGB stereo capture {capture_index} skipped: {exc}"
            )

except Exception:
    print(
        "\n[CABLE GRASP RGB STEREO SERVO] FATAL ERROR\n"
        + traceback.format_exc(),
        flush=True,
    )
    raise

finally:
    try:
        if runtime is not None:
            runtime.stop()

        if simulation_app is not None:
            simulation_app.close()
    finally:
        print(
            f"[LOG] Run output saved to: {run_output_path}",
            flush=True,
        )
        run_output_tee.stop()
