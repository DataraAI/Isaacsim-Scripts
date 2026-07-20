#!/usr/bin/env python3
"""Interactively author benchmark-only front-opening ground truth in Isaac Sim."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import traceback

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG


OUTPUT_PATH = PROJECT_ROOT / "benchmarks" / "front_rim_ground_truth.json"
GUIDE_PATH = "/World/PortOpeningGroundTruth"


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("Ground-truth normal must be finite and nonzero.")
    return vector / norm


def write_ground_truth(
    center_world_m: np.ndarray,
    normal_world: np.ndarray,
) -> None:
    normal = _unit(normal_world)
    payload = {
        "schema_version": 1,
        "center_world_m": [
            float(value)
            for value in np.asarray(center_world_m, dtype=np.float64).reshape(3)
        ],
        "normal_world": [float(value) for value in normal],
        "width_m": float(CONFIG.perception.port_width_m),
        "height_m": float(CONFIG.perception.port_height_m),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[GROUND TRUTH SAVED] {OUTPUT_PATH}", flush=True)


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": CONFIG.app.width,
            "height": CONFIG.app.height,
        }
    )
    runtime = None

    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        from perception import YOLOEPortDetector, process_stereo_port
        from sim import SimulationRuntime, warn

        runtime = SimulationRuntime(simulation_app=app, cfg=CONFIG)
        detector = YOLOEPortDetector(CONFIG.yoloe)
        detector.initialize()

        observation = None
        capture_index = 0
        while runtime.is_running() and observation is None:
            runtime.step()
            runtime.update_ik()
            if not runtime.capture_due():
                continue

            capture_index += 1
            try:
                frame = runtime.capture()
                observation = process_stereo_port(
                    frame=frame,
                    cfg=CONFIG.perception,
                    desired_port_virtual_camera_usd=(
                        runtime.desired_port_virtual_camera_usd
                    ),
                    previous_left=None,
                    previous_right=None,
                    detector=detector,
                )
            except Exception as exc:
                warn(
                    f"Ground-truth initialization capture {capture_index} "
                    f"rejected: {exc}"
                )

        if observation is None:
            raise RuntimeError("Isaac Sim closed before a stereo port was detected.")

        center = np.asarray(
            observation.center_world_xyz_m,
            dtype=np.float64,
        ).reshape(3)
        normal = _unit(observation.normal_world)

        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, GUIDE_PATH)
        root_prim = root.GetPrim()
        UsdGeom.Imageable(root_prim).CreatePurposeAttr().Set(
            UsdGeom.Tokens.guide
        )
        xform = UsdGeom.Xformable(root_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*center.tolist())
        )

        rotation = Gf.Rotation(
            Gf.Vec3d(0.0, 0.0, 1.0),
            Gf.Vec3d(*normal.tolist()),
        )
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            rotation.GetQuat()
        )

        plate = UsdGeom.Cube.Define(stage, GUIDE_PATH + "/OpeningPlane")
        plate.CreateSizeAttr().Set(1.0)
        plate.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.8, 0.0)])
        plate_xform = UsdGeom.Xformable(plate.GetPrim())
        plate_xform.ClearXformOpOrder()
        plate_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(
                0.5 * CONFIG.perception.port_width_m,
                0.5 * CONFIG.perception.port_height_m,
                0.00025,
            )
        )

        axis = UsdGeom.BasisCurves.Define(stage, GUIDE_PATH + "/NormalAxis")
        axis.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        axis.CreateCurveVertexCountsAttr().Set([2])
        axis.CreatePointsAttr().Set(
            [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.03)]
        )
        axis.CreateWidthsAttr().Set([0.0015])
        axis.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        axis.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.1, 0.1)])

        omni.usd.get_context().get_selection().set_selected_prim_paths(
            [GUIDE_PATH],
            True,
        )

        print(
            "\nFRONT-RIM GROUND-TRUTH CALIBRATION\n"
            f"  selected guide: {GUIDE_PATH}\n"
            "  yellow plate: physical front-opening plane\n"
            "  red axis: local +Z / camera-facing port normal\n\n"
            "In the Isaac Transform panel:\n"
            "  1. Translate the selected guide outward from the dark cavity until "
            "the yellow plate is exactly flush with the front opening.\n"
            "  2. Rotate only if needed so the plate matches the opening plane and "
            "the red +Z axis points toward the stereo cameras.\n"
            "  3. Do not move the rack, robot, or camera.\n"
            "  4. Return to this terminal and press Enter to save.\n",
            flush=True,
        )

        save_requested = threading.Event()

        def wait_for_enter() -> None:
            input()
            save_requested.set()

        input_thread = threading.Thread(
            target=wait_for_enter,
            name="front-rim-ground-truth-input",
            daemon=True,
        )
        input_thread.start()

        while runtime.is_running() and not save_requested.is_set():
            runtime.step()
            runtime.update_ik()

        if not save_requested.is_set():
            raise RuntimeError("Isaac Sim closed before calibration was saved.")

        transform = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        calibrated_center = np.asarray(
            transform.ExtractTranslation(),
            dtype=np.float64,
        )
        calibrated_normal = np.asarray(
            transform.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        write_ground_truth(calibrated_center, calibrated_normal)
        return 0

    finally:
        if runtime is not None:
            runtime.stop()
        app.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
