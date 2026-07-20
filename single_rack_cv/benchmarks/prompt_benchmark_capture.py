#!/usr/bin/env python3
"""Capture one frozen stationary stereo frame set for prompt comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
CAPTURE_TIMEOUT_SECONDS = 180.0
BENCHMARK_DIR_NAME = "prompt_ab_benchmark_v1"


def _camera_to_dict(camera) -> dict[str, object]:
    return {
        "image_height_px": int(camera.image_height_px),
        "image_width_px": int(camera.image_width_px),
        "focal_length_mm": float(camera.focal_length_mm),
        "horizontal_aperture_mm": float(camera.horizontal_aperture_mm),
        "vertical_aperture_mm": float(camera.vertical_aperture_mm),
        "world_from_camera": camera.world_from_camera.tolist(),
    }


def main() -> int:
    simulation_app = None
    runtime = None

    try:
        # Keep the native Isaac/RTX process clean. Project simulation modules
        # are imported only after the app has finished initializing.
        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": CONFIG.app.headless,
                "width": CONFIG.app.width,
                "height": CONFIG.app.height,
            }
        )

        from PIL import Image
        from prompt_benchmark_core import (
            BENCHMARK_FRAME_COUNT,
            MANIFEST_SCHEMA_VERSION,
        )
        from sim import SimulationRuntime

        output_root = CONFIG.camera.output_dir / BENCHMARK_DIR_NAME
        frames_dir = output_root / "frames"
        if output_root.exists():
            shutil.rmtree(output_root)
        frames_dir.mkdir(parents=True, exist_ok=True)

        runtime = SimulationRuntime(
            simulation_app=simulation_app,
            cfg=CONFIG,
        )

        print(
            "[PROMPT BENCHMARK CAPTURE]\n"
            f"  target frame pairs: {BENCHMARK_FRAME_COUNT}\n"
            "  motion policy: hold the existing fixed startup target\n"
            f"  output: {output_root}",
            flush=True,
        )

        started = time.monotonic()
        entries: list[dict[str, object]] = []

        while runtime.is_running() and len(entries) < BENCHMARK_FRAME_COUNT:
            if time.monotonic() - started > CAPTURE_TIMEOUT_SECONDS:
                raise TimeoutError(
                    "Timed out waiting for the fixed ToolCenter pose and "
                    f"{BENCHMARK_FRAME_COUNT} synchronized captures."
                )

            runtime.step()
            runtime.update_ik()

            if not runtime.capture_due():
                continue

            frame = runtime.capture()
            frame_index = len(entries) + 1
            left_name = f"left_{frame_index:04d}.png"
            right_name = f"right_{frame_index:04d}.png"

            Image.fromarray(frame.left.rgb, mode="RGB").save(
                frames_dir / left_name
            )
            Image.fromarray(frame.right.rgb, mode="RGB").save(
                frames_dir / right_name
            )

            entries.append(
                {
                    "frame_index": frame_index,
                    "sim_frame_index": int(runtime.frame_index),
                    "left_image": f"frames/{left_name}",
                    "right_image": f"frames/{right_name}",
                    "left_camera": _camera_to_dict(frame.left.camera),
                    "right_camera": _camera_to_dict(frame.right.camera),
                    "virtual_camera": _camera_to_dict(frame.virtual_camera),
                }
            )

            print(
                f"[CAPTURE] {frame_index:02d}/{BENCHMARK_FRAME_COUNT} "
                f"sim_frame={runtime.frame_index}",
                flush=True,
            )

        if len(entries) != BENCHMARK_FRAME_COUNT:
            raise RuntimeError(
                "Isaac Sim stopped before the complete frame set was captured: "
                f"{len(entries)}/{BENCHMARK_FRAME_COUNT}."
            )

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "frame_count": BENCHMARK_FRAME_COUNT,
            "resolution_height_width": list(CONFIG.camera.resolution),
            "preinsert_standoff_m": float(
                CONFIG.visual_servo.preinsert_standoff_m
            ),
            "desired_port_virtual_camera_usd": (
                runtime.desired_port_virtual_camera_usd.tolist()
            ),
            "frames": entries,
        }

        temporary_manifest = output_root / "manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        temporary_manifest.replace(output_root / "manifest.json")

        print(
            "\n[PASS] FROZEN STEREO FRAME SET CAPTURED\n"
            f"  frame pairs: {BENCHMARK_FRAME_COUNT}\n"
            f"  manifest: {output_root / 'manifest.json'}",
            flush=True,
        )
        return 0

    except Exception:
        print(
            "\n[FAIL] PROMPT BENCHMARK CAPTURE\n"
            + traceback.format_exc(),
            flush=True,
        )
        return 1

    finally:
        try:
            if runtime is not None:
                runtime.stop()
        finally:
            if simulation_app is not None:
                simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
