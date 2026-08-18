#!/usr/bin/env python3
"""Run the canonical 1280x960 stereo front-opening visual servo."""

from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

from config import CONFIG
from run_logger import RunLogger
from vision.front_lip_calibration import (
    VISIBLE_FRONT_LIP_HEIGHT_M,
    VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
    VISIBLE_FRONT_LIP_WIDTH_M,
)


class RunOutputTee:
    """Mirror process stdout/stderr to the terminal and one run log."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._log_fd: int | None = None
        self._pipe_read_fd: int | None = None
        self._pipe_write_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError("Console tee write returned no progress.")
            view = view[written:]

    def _copy_output(self) -> None:
        if (
            self._pipe_read_fd is None
            or self._saved_stdout_fd is None
            or self._log_fd is None
        ):
            return
        try:
            while True:
                chunk = os.read(self._pipe_read_fd, 65536)
                if not chunk:
                    break
                self._write_all(self._saved_stdout_fd, chunk)
                self._write_all(self._log_fd, chunk)
        except OSError:
            pass

    def start(self) -> None:
        if self._started:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        self._log_fd = os.open(
            self.output_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
        )
        self._pipe_read_fd, self._pipe_write_fd = os.pipe()
        os.dup2(self._pipe_write_fd, 1)
        os.dup2(self._pipe_write_fd, 2)
        os.close(self._pipe_write_fd)
        self._pipe_write_fd = None
        self._thread = threading.Thread(
            target=self._copy_output,
            name="run-output-tee",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for fd in (
            self._pipe_read_fd,
            self._log_fd,
            self._saved_stdout_fd,
            self._saved_stderr_fd,
        ):
            if fd is None:
                continue
            try:
                os.close(fd)
            except OSError:
                pass
        self._pipe_read_fd = None
        self._log_fd = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None
        self._thread = None
        self._started = False


run_output_path = CONFIG.camera.output_dir / "run_output_latest.txt"
run_output_tee = RunOutputTee(run_output_path)
run_output_tee.start()

simulation_app = None
runtime = None
logger = None
run_failed = False

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

    logger = RunLogger(
        output_dir=Path("run_logs"),
        pipeline="single_rack_cv",
        task="port_insertion",
    )

    from runtime.full_insertion_runtime import (
        AngledHandStereoHandoffRuntime as CableMountedSimulationRuntime,
    )
    from debug import DebugOutputs
    from vision.live_control_projective import refine_live_observation
    from vision.perception import YOLOEPortDetector, process_stereo_port
    from sim import warn

    runtime = CableMountedSimulationRuntime(
        simulation_app=simulation_app,
        cfg=CONFIG,
    )
    runtime.run_logger = logger
    runtime.prepare_for_perception()

    if (
        runtime.cable_mount is not None
        and getattr(runtime.cable_mount, "tcp_probe_only", False)
    ):
        print(
            "[CONNECTOR TCP PROBE] MOTION LOCKED\n"
            "  red marker: legacy full-bounds tip\n"
            "  cyan marker: mesh-derived insertion TCP\n"
            "  inspect the markers on the RJ45 nose, then close Isaac Sim\n"
            "  YOLOE, visual servo, handoff, and insertion are disabled",
            flush=True,
        )
        while runtime.is_running():
            runtime.step()
            runtime.update_ik()
        raise SystemExit(0)

    debug = DebugOutputs(CONFIG)
    detector = YOLOEPortDetector(CONFIG.yoloe)
    detector.initialize()
    print(
        "[YOLOE] Visual prompt initialized once; "
        "full-frame stereo inference is active.",
        flush=True,
    )
    logger.log_event(0, "yoloe_initialized")
    if CONFIG.front_plane.enabled:
        print(
            "[LIVE FRONT PLANE] automatic refined local SGBM control enabled; "
            "no manual depth offset and no RTX/USD ground truth in runtime.\n"
            f"  visible front-lip validation: "
            f"{VISIBLE_FRONT_LIP_WIDTH_M * 1000.0:.3f} x "
            f"{VISIBLE_FRONT_LIP_HEIGHT_M * 1000.0:.3f} mm\n"
            f"  side-edge localization width: "
            f"{VISIBLE_FRONT_LIP_SEARCH_WIDTH_M * 1000.0:.3f} mm",
            flush=True,
        )

    capture_index = 0
    while runtime.is_running():
        runtime.step()
        try:
            runtime.update_ik()
            runtime.update_visual_servo_completion()
            runtime.update_partial_insertion()
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
            observation = process_stereo_port(
                frame=frame,
                cfg=CONFIG.perception,
                desired_port_virtual_camera_usd=(
                    runtime.desired_port_virtual_camera_usd
                ),
                previous_left=previous_left,
                previous_right=previous_right,
                detector=detector,
            )
            if CONFIG.front_plane.enabled:
                observation, front_plane = refine_live_observation(
                    frame=frame,
                    observation=observation,
                    desired_port_virtual_camera_usd=(
                        runtime.desired_port_virtual_camera_usd
                    ),
                    aperture_width_m=VISIBLE_FRONT_LIP_WIDTH_M,
                    aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
                    search_width_m=VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
                )
                print(
                    "[LIVE FRONT PLANE] "
                    f"capture={capture_index} "
                    f"cavity_range={front_plane.cavity_range_m * 1000.0:.2f}mm "
                    f"opening_range={front_plane.opening_range_m * 1000.0:.2f}mm "
                    f"recess={front_plane.recess_depth_m * 1000.0:+.2f}mm "
                    f"center={list(np.round(front_plane.aperture_center_world_m, 6))} "
                    f"center_pair={front_plane.aperture_center_disagreement_m * 1000.0:.3f}mm "
                    f"plane_residual={front_plane.plane_residual_m * 1000.0:.3f}mm "
                    f"ray_gap={front_plane.max_ray_gap_m * 1000.0:.3f}mm "
                    f"dense={front_plane.consistent_disparity_count}/"
                    f"{front_plane.valid_disparity_count} "
                    f"ring={front_plane.ring_candidate_count} "
                    f"triangulated={front_plane.triangulated_count} "
                    f"cluster={front_plane.cluster_count} "
                    f"sides={front_plane.side_support_counts}",
                    flush=True,
                )
                logger.log_frame(
                    capture_index,
                    cavity_range_mm=front_plane.cavity_range_m * 1000.0,
                    opening_range_mm=front_plane.opening_range_m * 1000.0,
                    recess_depth_mm=front_plane.recess_depth_m * 1000.0,
                    plane_residual_mm=front_plane.plane_residual_m * 1000.0,
                    yoloe_conf_left=observation.left.detection.shape_score,
                    yoloe_conf_right=observation.right.detection.shape_score,
                    phase=runtime.current_phase(),
                )
            runtime.observe_visual_servo(observation)
            frozen_port_point = runtime.frozen_port_point_world_m
            if frozen_port_point is not None:
                debug.update_frozen_port_point(frozen_port_point)
            debug.handle(frame, observation, capture_index)
        except Exception as exc:
            runtime.note_perception_failure()
            warn(f"RGB stereo capture {capture_index} skipped: {exc}")

except Exception:
    run_failed = True
    print(
        "\n[SINGLE RACK RGB STEREO SERVO] FATAL ERROR\n"
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
