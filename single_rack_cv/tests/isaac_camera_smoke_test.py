#!/usr/bin/env python3
"""Live Isaac Sim smoke test for the mounted stereo camera geometry."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_STATUS_ENV = "GEOMETRY_SMOKE_STATUS_FILE"

BASELINE_EXPECTED_M = 0.040
BASELINE_TOLERANCE_M = 0.0001
LOCAL_POSITION_TOLERANCE_M = 1.0e-5
DIRECTION_DOT_MIN = 0.9999
EPIPOLAR_TOLERANCE_PX = 1.0e-3
TRIANGULATION_TOLERANCE_M = 1.0e-6
RAY_GAP_TOLERANCE_M = 1.0e-7
REPROJECTION_TOLERANCE_PX = 1.0e-6


def _same_resolved_path(value: str, expected: Path) -> bool:
    try:
        return Path(value or ".").resolve() == expected.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _prioritize_project_root(
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Force local modules ahead of Isaac/OpenCV's pip-prebundle paths."""
    project_root = project_root.resolve()
    root_text = str(project_root)

    sys.path[:] = [
        entry
        for entry in sys.path
        if not _same_resolved_path(entry, project_root)
    ]
    sys.path.insert(0, root_text)

    # Isaac Sim/OpenCV can expose a top-level module named `config`.
    # Remove any already-cached non-project module before importing ours.
    for module_name in ("config", "perception", "sim"):
        existing = sys.modules.get(module_name)
        if existing is None:
            continue

        existing_file = getattr(existing, "__file__", None)
        expected_file = project_root / f"{module_name}.py"
        if (
            existing_file is None
            or not _same_resolved_path(existing_file, expected_file)
        ):
            del sys.modules[module_name]


def _write_status(state: str, details: str = "") -> None:
    """Write a result before SimulationApp shutdown can alter exit behavior."""
    status_name = os.environ.get(SMOKE_STATUS_ENV)
    if not status_name:
        return

    body = state.strip().upper()
    if details:
        body += "\n" + details.rstrip()
    Path(status_name).write_text(body + "\n", encoding="utf-8")


def _pass(name: str, details: str = "") -> None:
    suffix = f" | {details}" if details else ""
    print(f"[PASS] {name}{suffix}", flush=True)


def _require(condition: bool, name: str, details: str) -> None:
    if not condition:
        raise AssertionError(f"{name}: {details}")
    _pass(name, details)


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise AssertionError(f"Cannot normalize vector {value.tolist()}")
    return value / norm


def _point_in_frame(
    uv: np.ndarray,
    width_px: int,
    height_px: int,
) -> bool:
    u, v = map(float, np.asarray(uv).reshape(2))
    return 0.0 <= u < width_px and 0.0 <= v < height_px


def run_smoke_test() -> bool:
    """Build the existing scene and validate live stereo camera transforms."""
    simulation_app = None
    runtime = None
    success = False

    try:
        # Match the working main.py startup order: load only the project's
        # lightweight config module before SimulationApp. Do not import
        # perception/OpenCV/YOLOE until after Isaac Sim has started.
        _prioritize_project_root()
        from config import CONFIG

        imported_config_path = Path(
            sys.modules["config"].__file__
        ).resolve()
        expected_config_path = (PROJECT_ROOT / "config.py").resolve()
        _require(
            imported_config_path == expected_config_path,
            "local project config import",
            f"path={imported_config_path}",
        )

        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": True,
                "width": 640,
                "height": 480,
            }
        )

        # Isaac startup prepends its own pip-prebundle paths. Put the project
        # root back first while preserving the already-loaded local config.
        _prioritize_project_root()

        global np
        import numpy as np
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        from perception import (
            build_virtual_camera_model,
            transform_point_to_world,
            triangulate_pixel_pair,
        )
        from sim import (
            SimulationRuntime,
            quaternion_wxyz_to_matrix,
        )

        cfg = replace(
            CONFIG,
            app=replace(
                CONFIG.app,
                headless=True,
                width=640,
                height=480,
            ),
        )

        runtime = SimulationRuntime(
            simulation_app=simulation_app,
            cfg=cfg,
        )

        for _ in range(10):
            runtime.step()

        height_px, width_px = cfg.camera.resolution
        dummy_rgb = np.zeros(
            (height_px, width_px, 3),
            dtype=np.uint8,
        )
        left = runtime._camera_model(
            runtime.left_camera_path,
            dummy_rgb,
        )
        right = runtime._camera_model(
            runtime.right_camera_path,
            dummy_rgb,
        )
        virtual = build_virtual_camera_model(left, right)

        print("\n[LIVE CAMERA TRANSFORMS]", flush=True)
        print(
            "left world_from_camera:\n"
            + np.array2string(
                left.world_from_camera,
                precision=8,
                suppress_small=True,
            ),
            flush=True,
        )
        print(
            "right world_from_camera:\n"
            + np.array2string(
                right.world_from_camera,
                precision=8,
                suppress_small=True,
            ),
            flush=True,
        )
        print(
            "left center world:  "
            f"{np.round(left.camera_center_world_m, 8).tolist()}",
            flush=True,
        )
        print(
            "right center world: "
            f"{np.round(right.camera_center_world_m, 8).tolist()}",
            flush=True,
        )

        baseline_world = (
            right.camera_center_world_m
            - left.camera_center_world_m
        )
        baseline_m = float(np.linalg.norm(baseline_world))
        _require(
            abs(baseline_m - BASELINE_EXPECTED_M)
            <= BASELINE_TOLERANCE_M,
            "live stereo baseline",
            f"actual={baseline_m * 1000.0:.4f} mm, "
            f"expected={BASELINE_EXPECTED_M * 1000.0:.1f} mm",
        )

        if runtime.ik is None:
            raise AssertionError("SimulationRuntime did not create IK runtime.")

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise AssertionError("USD stage is unavailable.")

        hand_prim = stage.GetPrimAtPath(runtime.ik.hand_path)
        if not hand_prim.IsValid():
            raise AssertionError(
                f"Invalid panda_hand prim: {runtime.ik.hand_path}"
            )

        hand_world = np.asarray(
            UsdGeom.Xformable(hand_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            ),
            dtype=np.float64,
        )
        hand_from_world = np.linalg.inv(hand_world)

        def center_in_hand(camera_center_world: np.ndarray) -> np.ndarray:
            homogeneous = np.append(camera_center_world, 1.0)
            local = homogeneous @ hand_from_world
            return local[:3] / local[3]

        left_center_hand = center_in_hand(
            left.camera_center_world_m
        )
        right_center_hand = center_in_hand(
            right.camera_center_world_m
        )

        np.testing.assert_allclose(
            left_center_hand,
            np.asarray(cfg.camera.left_local_position),
            atol=LOCAL_POSITION_TOLERANCE_M,
            rtol=0.0,
        )
        _pass(
            "left camera local position",
            f"actual={np.round(left_center_hand, 7).tolist()}",
        )

        np.testing.assert_allclose(
            right_center_hand,
            np.asarray(cfg.camera.right_local_position),
            atol=LOCAL_POSITION_TOLERANCE_M,
            rtol=0.0,
        )
        _pass(
            "right camera local position",
            f"actual={np.round(right_center_hand, 7).tolist()}",
        )

        left_forward_world = _normalize(
            (
                np.array([0.0, 0.0, -1.0, 0.0])
                @ left.world_from_camera
            )[:3]
        )
        right_forward_world = _normalize(
            (
                np.array([0.0, 0.0, -1.0, 0.0])
                @ right.world_from_camera
            )[:3]
        )
        parallel_dot = float(
            np.dot(left_forward_world, right_forward_world)
        )
        _require(
            parallel_dot >= DIRECTION_DOT_MIN,
            "parallel optical axes",
            f"dot={parallel_dot:.9f}",
        )

        left_forward_hand = _normalize(
            (
                np.append(left_forward_world, 0.0)
                @ hand_from_world
            )[:3]
        )
        right_forward_hand = _normalize(
            (
                np.append(right_forward_world, 0.0)
                @ hand_from_world
            )[:3]
        )

        y_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 1.0, 0.0),
            cfg.camera.local_y_rotation_deg,
        ).GetQuat()
        roll_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 0.0, 1.0),
            cfg.camera.local_roll_deg,
        ).GetQuat()
        local_quat = y_quat * roll_quat
        imag = local_quat.GetImaginary()
        local_q_wxyz = np.array(
            [
                local_quat.GetReal(),
                imag[0],
                imag[1],
                imag[2],
            ],
            dtype=np.float64,
        )
        expected_forward_hand = _normalize(
            quaternion_wxyz_to_matrix(local_q_wxyz)
            @ np.array([0.0, 0.0, -1.0])
        )

        left_orientation_dot = float(
            np.dot(left_forward_hand, expected_forward_hand)
        )
        right_orientation_dot = float(
            np.dot(right_forward_hand, expected_forward_hand)
        )
        _require(
            left_orientation_dot >= DIRECTION_DOT_MIN,
            "left camera configured orientation",
            f"dot={left_orientation_dot:.9f}",
        )
        _require(
            right_orientation_dot >= DIRECTION_DOT_MIN,
            "right camera configured orientation",
            f"dot={right_orientation_dot:.9f}",
        )

        point_virtual = np.array(
            [0.005, -0.003, -0.200],
            dtype=np.float64,
        )
        point_world = transform_point_to_world(
            point_virtual,
            virtual.world_from_camera,
        )
        left_uv = left.project_world(point_world)
        right_uv = right.project_world(point_world)

        _require(
            _point_in_frame(left_uv, width_px, height_px),
            "known point visible in left eye",
            f"uv={np.round(left_uv, 6).tolist()}",
        )
        _require(
            _point_in_frame(right_uv, width_px, height_px),
            "known point visible in right eye",
            f"uv={np.round(right_uv, 6).tolist()}",
        )

        left_depth = -float(left.camera_point_from_world(point_world)[2])
        right_depth = -float(right.camera_point_from_world(point_world)[2])
        _require(
            left_depth > 0.0 and right_depth > 0.0,
            "known point has positive stereo depth",
            f"left={left_depth:.6f} m, right={right_depth:.6f} m",
        )

        epipolar_error_px = abs(
            float(left_uv[1]) - float(right_uv[1])
        )
        _require(
            epipolar_error_px <= EPIPOLAR_TOLERANCE_PX,
            "live epipolar alignment",
            f"vertical error={epipolar_error_px:.9f} px",
        )

        disparity_px = float(left_uv[0] - right_uv[0])
        left_x_axis_world = _normalize(
            (
                np.array([1.0, 0.0, 0.0, 0.0])
                @ left.world_from_camera
            )[:3]
        )
        baseline_camera_x_m = float(
            np.dot(baseline_world, left_x_axis_world)
        )
        _require(
            disparity_px * baseline_camera_x_m > 0.0,
            "live disparity sign",
            f"disparity={disparity_px:.6f} px, "
            f"baseline camera-x={baseline_camera_x_m:.6f} m",
        )

        reconstructed_world, ray_gap_m = triangulate_pixel_pair(
            left_uv,
            right_uv,
            left,
            right,
        )
        triangulation_error_m = float(
            np.linalg.norm(reconstructed_world - point_world)
        )
        _require(
            triangulation_error_m <= TRIANGULATION_TOLERANCE_M,
            "live triangulation recovers known point",
            f"error={triangulation_error_m * 1000.0:.6f} mm",
        )
        _require(
            ray_gap_m <= RAY_GAP_TOLERANCE_M,
            "live stereo ray gap",
            f"gap={ray_gap_m * 1000.0:.6f} mm",
        )

        left_reprojection_px = float(
            np.linalg.norm(
                left.project_world(reconstructed_world) - left_uv
            )
        )
        right_reprojection_px = float(
            np.linalg.norm(
                right.project_world(reconstructed_world) - right_uv
            )
        )
        _require(
            max(left_reprojection_px, right_reprojection_px)
            <= REPROJECTION_TOLERANCE_PX,
            "live stereo reprojection",
            f"left={left_reprojection_px:.9f} px, "
            f"right={right_reprojection_px:.9f} px",
        )

        success = True
        _write_status("PASS")
        print(
            "\n[PASS] ISAAC SIM CAMERA SMOKE TEST COMPLETE",
            flush=True,
        )

    except Exception:
        details = traceback.format_exc()
        _write_status("FAIL", details)
        print(
            "\n[FAIL] ISAAC SIM CAMERA SMOKE TEST\n" + details,
            flush=True,
        )

    finally:
        try:
            if runtime is not None:
                runtime.stop()
        finally:
            if simulation_app is not None:
                simulation_app.close()

    return success


if __name__ == "__main__":
    raise SystemExit(0 if run_smoke_test() else 1)
