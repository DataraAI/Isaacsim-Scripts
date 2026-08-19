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
from pathlib import Path

from config import CONFIG
from head_detector import YOLOEHeadDetector
from logging_tee import RunOutputTee
from perception import process_stereo_cable
from run_logger import RunLogger


def run_pickup_phase(simulation_app) -> RunLogger:
    run_output_path = CONFIG.camera.output_dir / "run_output_latest.txt"
    run_output_tee = RunOutputTee(run_output_path)
    run_output_tee.start()

    runtime = None
    logger = None
    run_failed = False

    try:
        print(
            f"[LOG] Saving complete run output to: {run_output_path}",
            flush=True,
        )

        logger = RunLogger(
            output_dir=Path("run_logs"),
            pipeline="esha",
            task="grasp_and_carry",
        )

        from debug import DebugOutputs
        from sim import SimulationRuntime, warn

        runtime = SimulationRuntime(
            simulation_app=simulation_app,
            cfg=CONFIG,
            run_logger=logger,
        )
        debug = DebugOutputs(CONFIG)

        head_detector = YOLOEHeadDetector(CONFIG.cable_head_yoloe)
        head_detector.initialize()

        capture_index = 0

        while runtime.is_running():
            runtime.step()

            try:
                runtime.update_ik()
                runtime.update_visual_servo_completion()
            except Exception as exc:
                warn(f"Motion/IK update failed: {exc}")

            try:
                runtime.update_pre_grasp()
            except Exception as exc:
                warn(f"Pre-grasp update failed: {exc}")

            if not runtime.capture_due():
                continue

            capture_index += 1

            try:
                frame = runtime.capture()
                debug.save_raw(frame)
                if capture_index % 10 == 0:  # tune this — one every 10 captures
                    debug.save_reference_candidate(frame, capture_index)
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
                    head_detector=head_detector,
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
        run_failed = True
        print(
            "\n[CABLE GRASP RGB STEREO SERVO] FATAL ERROR\n"
            + traceback.format_exc(),
            flush=True,
        )
        raise

    finally:
        if logger is not None:
            try:
                logger.finalize("failure" if run_failed else "success")
            except Exception:
                pass
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

    return logger


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    run_pickup_phase(simulation_app)
