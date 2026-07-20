#!/usr/bin/env python3
"""Automatically derive the physical port-opening plane from Isaac raycasts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw

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

OUTPUT_PATH = PROJECT_ROOT / "benchmarks" / "front_rim_ground_truth.json"
DEBUG_DIR = CONFIG.camera.output_dir / "front_rim_ground_truth_debug"


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


def _bbox_side_samples(
    bbox_xywh: tuple[int, int, int, int],
    *,
    samples_per_side: int,
    corner_trim_fraction: float,
) -> np.ndarray:
    """Return top/right/bottom/left samples on a refined cavity box."""
    x, y, width, height = map(float, bbox_xywh)
    if width <= 1.0 or height <= 1.0:
        raise RuntimeError(f"Cavity box is too small for a ray ring: {bbox_xywh}.")
    if samples_per_side < 2:
        raise ValueError("samples_per_side must be at least 2.")
    trim = float(corner_trim_fraction)
    if not 0.0 <= trim < 0.5:
        raise ValueError("corner_trim_fraction must be in [0, 0.5).")

    left = x
    top = y
    right = x + width - 1.0
    bottom = y + height - 1.0
    values = np.linspace(trim, 1.0 - trim, samples_per_side)

    top_side = np.column_stack(
        (left + values * (right - left), np.full_like(values, top))
    )
    right_side = np.column_stack(
        (np.full_like(values, right), top + values * (bottom - top))
    )
    bottom_side = np.column_stack(
        (left + values * (right - left), np.full_like(values, bottom))
    )
    left_side = np.column_stack(
        (np.full_like(values, left), top + values * (bottom - top))
    )
    return np.stack((top_side, right_side, bottom_side, left_side), axis=0)


def _cavity_outer_ring(
    bbox_xywh: tuple[int, int, int, int],
    center_uv: tuple[float, float],
    cfg: RaycastGroundTruthConfig,
) -> np.ndarray:
    samples = _bbox_side_samples(
        bbox_xywh,
        samples_per_side=CONFIG.front_rim.samples_per_side,
        corner_trim_fraction=CONFIG.front_rim.sample_corner_trim_fraction,
    )
    return offset_rim_samples_outward(
        side_samples_uv=samples,
        center_uv=center_uv,
        offset_px=cfg.rim_outward_offset_px,
    )


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


def _draw_eye_debug(
    rgb: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    center_uv: tuple[float, float],
    ring_uv: np.ndarray,
    label: str,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    x, y, width, height = bbox_xywh
    draw.rectangle(
        [x, y, x + width - 1, y + height - 1],
        outline=(0, 255, 0),
        width=2,
    )
    center_u, center_v = center_uv
    draw.line(
        [center_u - 5, center_v, center_u + 5, center_v],
        fill=(0, 255, 255),
        width=1,
    )
    draw.line(
        [center_u, center_v - 5, center_u, center_v + 5],
        fill=(0, 255, 255),
        width=1,
    )
    for u, v in np.asarray(ring_uv).reshape(-1, 2):
        draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 0, 255))
    bbox_aspect = width / max(1.0, float(height))
    draw.text(
        (8, 8),
        f"{label} bbox={bbox_xywh} aspect={bbox_aspect:.3f}",
        fill=(255, 255, 255),
    )
    return np.asarray(image, dtype=np.uint8)


def _save_debug_capture(
    capture_index: int,
    frame,
    observation,
    left_ring_uv: np.ndarray,
    right_ring_uv: np.ndarray,
) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    left = observation.left.detection
    right = observation.right.detection
    Image.fromarray(
        _draw_eye_debug(
            frame.left.rgb,
            left.bbox_xywh,
            left.center_uv,
            left_ring_uv,
            "left",
        ),
        mode="RGB",
    ).save(DEBUG_DIR / f"left_{capture_index:02d}.png")
    Image.fromarray(
        _draw_eye_debug(
            frame.right.rgb,
            right.bbox_xywh,
            right.center_uv,
            right_ring_uv,
            "right",
        ),
        mode="RGB",
    ).save(DEBUG_DIR / f"right_{capture_index:02d}.png")
    Image.fromarray(left.mask, mode="L").save(
        DEBUG_DIR / f"left_mask_{capture_index:02d}.png"
    )
    Image.fromarray(right.mask, mode="L").save(
        DEBUG_DIR / f"right_mask_{capture_index:02d}.png"
    )


def _print_detection_diagnostics(capture_index: int, observation) -> None:
    left = observation.left.detection
    right = observation.right.detection
    left_aspect = left.bbox_xywh[2] / max(1.0, float(left.bbox_xywh[3]))
    right_aspect = right.bbox_xywh[2] / max(1.0, float(right.bbox_xywh[3]))
    print(
        "[GROUND TRUTH DETECTION]\n"
        f"  capture: {capture_index}\n"
        f"  left bbox: {left.bbox_xywh}; aspect={left_aspect:.3f}; "
        f"center={np.round(left.center_uv, 3).tolist()}\n"
        f"  right bbox: {right.bbox_xywh}; aspect={right_aspect:.3f}; "
        f"center={np.round(right.center_uv, 3).tolist()}",
        flush=True,
    )


def _print_raycast_diagnostics(
    hits: list[RaycastHit],
    misses: list[str],
    cfg: RaycastGroundTruthConfig,
) -> None:
    path_counts = Counter(hit.prim_path or "<empty>" for hit in hits)
    rack_hits = [
        hit for hit in hits if hit.prim_path.startswith(cfg.rack_path_prefix)
    ]
    print(
        "[GROUND TRUTH RAYCAST]\n"
        f"  returned hits: {len(hits)}\n"
        f"  rack-prefix hits: {len(rack_hits)}\n"
        f"  misses/invalid: {len(misses)}\n"
        f"  path counts: {dict(path_counts)}",
        flush=True,
    )
    if hits:
        distances = np.asarray([hit.distance_m for hit in hits])
        print(
            "  hit distance range: "
            f"{float(np.min(distances)):.6f} .. "
            f"{float(np.max(distances)):.6f} m",
            flush=True,
        )
    if misses:
        print("  first misses: " + "; ".join(misses[:8]), flush=True)


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
                _print_detection_diagnostics(capture_index, observation)

                left_detection = observation.left.detection
                right_detection = observation.right.detection
                left_outer = _cavity_outer_ring(
                    left_detection.bbox_xywh,
                    left_detection.center_uv,
                    ray_cfg,
                )
                right_outer = _cavity_outer_ring(
                    right_detection.bbox_xywh,
                    right_detection.center_uv,
                    ray_cfg,
                )
                _save_debug_capture(
                    capture_index,
                    frame,
                    observation,
                    left_outer,
                    right_outer,
                )

                virtual_ring_uv = 0.5 * (left_outer + right_outer)
                virtual_center_uv = 0.5 * (
                    np.asarray(left_detection.center_uv, dtype=np.float64)
                    + np.asarray(right_detection.center_uv, dtype=np.float64)
                )

                runtime.step()
                hits, misses = _cast_rays(
                    camera=frame.virtual_camera,
                    pixels_uv=virtual_ring_uv,
                    cfg=ray_cfg,
                )
                _print_raycast_diagnostics(hits, misses, ray_cfg)
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
                _write_result(result, virtual_ring_uv)
                return 0
            except Exception as exc:
                warn(
                    f"Automatic ground-truth capture {capture_index} rejected: "
                    f"{exc}"
                )
                if capture_index >= 20:
                    raise RuntimeError(
                        "Automatic front-plane extraction failed for 20 captures. "
                        f"Inspect {DEBUG_DIR}."
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
