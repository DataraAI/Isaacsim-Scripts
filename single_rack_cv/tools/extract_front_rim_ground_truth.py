#!/usr/bin/env python3
"""Automatically derive the physical port-opening plane from Isaac raycasts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automatic_port_ground_truth import (
    RaycastGroundTruthConfig,
    RaycastHit,
    build_automatic_ground_truth,
    offset_rim_samples_outward,
)
from config import CONFIG
from front_rim import extract_front_rim

OUTPUT_PATH = PROJECT_ROOT / "benchmarks" / "front_rim_ground_truth.json"


def _float3(value: np.ndarray):
    import carb

    x, y, z = np.asarray(value, dtype=np.float64).reshape(3)
    return carb.Float3(float(x), float(y), float(z))


def _hit_prim_path(hit: dict[str, object], rack_prefix: str) -> str:
    candidates = (
        str(hit.get("collision", "")),
        str(hit.get("rigidBody", "")),
    )
    for candidate in candidates:
        if candidate.startswith(rack_prefix):
            return candidate
    return next((candidate for candidate in candidates if candidate), "")


def _cast_rays(
    camera,
    pixels_uv: np.ndarray,
    cfg: RaycastGroundTruthConfig,
) -> tuple[list[RaycastHit], list[str]]:
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    hits: list[RaycastHit] = []
    misses: list[str] = []
    for index, pixel_uv in enumerate(
        np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    ):
        origin, direction = camera.pixel_to_world_ray(pixel_uv)
        hit = query.raycast_closest(
            _float3(origin),
            _float3(direction),
            float(cfg.max_raycast_distance_m),
        )
        if not bool(hit.get("hit", False)):
            misses.append(f"#{index}: miss at {np.round(pixel_uv, 2).tolist()}")
            continue
        prim_path = _hit_prim_path(hit, cfg.rack_path_prefix)
        position = np.asarray(hit["position"], dtype=np.float64)
        normal = np.asarray(hit["normal"], dtype=np.float64)
        distance = float(hit["distance"])
        try:
            hits.append(
                RaycastHit(
                    position_world_m=position,
                    normal_world=normal,
                    prim_path=prim_path,
                    distance_m=distance,
                )
            )
        except ValueError as exc:
            misses.append(f"#{index}: invalid hit: {exc}")
    return hits, misses


def _write_result(result, ring_pixels_uv: np.ndarray) -> None:
    payload = {
        "schema_version": 2,
        "source": "automatic_physx_raycast_front_bezel_plane",
        "control_usage": "forbidden; benchmark scoring only",
        "center_world_m": [float(value) for value in result.center_world_m],
        "normal_world": [float(value) for value in result.normal_world],
        "width_m": float(CONFIG.perception.port_width_m),
        "height_m": float(CONFIG.perception.port_height_m),
        "plane_residual_m": float(result.plane_residual_m),
        "valid_hit_count": int(result.valid_hit_count),
        "used_hit_count": int(result.used_hit_count),
        "used_prim_paths": list(result.used_prim_paths),
        "ring_pixel_count": int(np.prod(ring_pixels_uv.shape[:2])),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[AUTOMATIC GROUND TRUTH SAVED] {OUTPUT_PATH}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": CONFIG.app.width,
            "height": CONFIG.app.height,
        }
    )
    runtime = None
    try:
        from perception import YOLOEPortDetector, process_stereo_port
        from sim import SimulationRuntime, warn

        runtime = SimulationRuntime(simulation_app=app, cfg=CONFIG)
        detector = YOLOEPortDetector(CONFIG.yoloe)
        detector.initialize()
        ray_cfg = RaycastGroundTruthConfig(
            rack_path_prefix=CONFIG.scene.rack_path,
        )

        capture_index = 0
        while runtime.is_running():
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
                left_rim = extract_front_rim(
                    frame.left.rgb,
                    observation.left.detection.bbox_xywh,
                    CONFIG.front_rim,
                )
                right_rim = extract_front_rim(
                    frame.right.rgb,
                    observation.right.detection.bbox_xywh,
                    CONFIG.front_rim,
                )

                left_outer = offset_rim_samples_outward(
                    left_rim.side_samples_uv,
                    left_rim.center_uv,
                    ray_cfg.rim_outward_offset_px,
                )
                right_outer = offset_rim_samples_outward(
                    right_rim.side_samples_uv,
                    right_rim.center_uv,
                    ray_cfg.rim_outward_offset_px,
                )
                virtual_ring_uv = 0.5 * (left_outer + right_outer)
                virtual_center_uv = 0.5 * (
                    np.asarray(left_rim.center_uv, dtype=np.float64)
                    + np.asarray(right_rim.center_uv, dtype=np.float64)
                )

                runtime.step()
                hits, misses = _cast_rays(
                    camera=frame.virtual_camera,
                    pixels_uv=virtual_ring_uv,
                    cfg=ray_cfg,
                )
                center_origin, center_direction = (
                    frame.virtual_camera.pixel_to_world_ray(virtual_center_uv)
                )
                result = build_automatic_ground_truth(
                    hits=hits,
                    camera_center_world_m=center_origin,
                    center_ray_direction_world=center_direction,
                    cfg=ray_cfg,
                )
                print(
                    "AUTOMATIC FRONT-RIM GROUND TRUTH\n"
                    f"  capture: {capture_index}\n"
                    f"  cast rays: {virtual_ring_uv.reshape(-1, 2).shape[0]}\n"
                    f"  valid rack hits: {result.valid_hit_count}\n"
                    f"  plane inliers: {result.used_hit_count}\n"
                    f"  misses/invalid hits: {len(misses)}\n"
                    f"  plane residual: {result.plane_residual_m * 1000.0:.4f} mm\n"
                    f"  opening center: {np.round(result.center_world_m, 6).tolist()}\n"
                    f"  camera-facing normal: {np.round(result.normal_world, 6).tolist()}",
                    flush=True,
                )
                if misses:
                    print(
                        "  first ray diagnostics: " + "; ".join(misses[:8]),
                        flush=True,
                    )
                _write_result(result, virtual_ring_uv)
                return 0
            except Exception as exc:
                warn(
                    f"Automatic ground-truth capture {capture_index} rejected: "
                    f"{exc}"
                )
                if capture_index >= 20:
                    raise RuntimeError(
                        "Automatic front-plane extraction failed for 20 captures."
                    ) from exc

        raise RuntimeError("Isaac Sim closed before automatic extraction completed.")
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
