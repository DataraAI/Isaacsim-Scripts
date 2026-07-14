#!/usr/bin/env python3
"""Run the single-rack RGB-D port-perception demo."""

from __future__ import annotations

import traceback

from config import CONFIG

# Isaac Sim must start before importing modules that use omni/pxr APIs.
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": CONFIG.app.headless,
        "width": CONFIG.app.width,
        "height": CONFIG.app.height,
    }
)

runtime = None

try:
    from debug import DebugOutputs
    from perception import process_port
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
            runtime.update_auto_preinsert_motion()
            runtime.update_ik()
        except Exception as exc:
            warn(f"Motion/IK update failed: {exc}")

        if not runtime.capture_due():
            continue

        capture_index += 1

        try:
            frame = runtime.capture()
            estimate = process_port(
                frame,
                CONFIG.perception,
            )
            debug.handle(
                frame,
                estimate,
                capture_index,
            )
            runtime.observe_preinsert_estimate(estimate)
        except Exception as exc:
            # One bad perception frame should not terminate Isaac Sim.
            runtime.note_perception_failure()
            warn(
                f"Perception capture {capture_index} skipped: {exc}"
            )

except Exception:
    print(
        "\n[SINGLE RACK CV] FATAL ERROR\n"
        + traceback.format_exc(),
        flush=True,
    )
    raise

finally:
    if runtime is not None:
        runtime.stop()
    simulation_app.close()
