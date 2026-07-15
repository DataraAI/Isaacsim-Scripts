#!/usr/bin/env python3
"""RGB visual-servo markers, overlays, output files, and concise logs."""

from __future__ import annotations

import numpy as np
import omni.usd
from PIL import Image, ImageDraw
from pxr import Gf, UsdGeom

from config import Config
from perception import CameraFrame, PortObservation


class DebugOutputs:
    """Own all visualization and file-output side effects."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.output_dir = cfg.camera.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def handle(
        self,
        frame: CameraFrame,
        observation: PortObservation,
        capture_index: int,
    ) -> None:
        self.update_stage(observation)
        self.save_files(frame, observation)
        self.print_summary(observation, capture_index)

    # ------------------------------------------------------------------
    # Stage marker
    # ------------------------------------------------------------------

    def update_stage(self, observation: PortObservation) -> None:
        cfg = self.cfg.debug
        self._update_sphere(
            cfg.estimated_port_marker_path,
            observation.port_world_xyz_m,
            cfg.estimated_port_marker_radius_m,
            cfg.estimated_port_marker_color,
        )

    @staticmethod
    def _update_sphere(
        path: str,
        position: np.ndarray,
        radius: float,
        color: tuple[float, float, float],
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)

        if not prim.IsValid():
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.CreateRadiusAttr().Set(float(radius))
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            prim = sphere.GetPrim()
            UsdGeom.Imageable(prim).CreatePurposeAttr().Set(
                UsdGeom.Tokens.guide
            )

            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(*np.asarray(position).tolist()))
            return

        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*np.asarray(position).tolist()))
                return

        raise RuntimeError(f"Missing translate op on {path}")

    # ------------------------------------------------------------------
    # Files and overlay
    # ------------------------------------------------------------------

    def save_files(
        self,
        frame: CameraFrame,
        observation: PortObservation,
    ) -> None:
        Image.fromarray(frame.rgb, mode="RGB").save(
            self.output_dir / "rgb_latest.png"
        )
        Image.fromarray(
            self._draw_overlay(frame.rgb, observation),
            mode="RGB",
        ).save(self.output_dir / "rgb_port_detected.png")
        Image.fromarray(
            observation.detection.mask,
            mode="L",
        ).save(self.output_dir / "port_detection_mask.png")

    def _draw_overlay(
        self,
        rgb: np.ndarray,
        observation: PortObservation,
    ) -> np.ndarray:
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        detection = observation.detection

        u0, v0, u1, v1 = detection.roi_uv
        x, y, width, height = detection.bbox_xywh
        detected_u, detected_v = detection.center_uv
        desired_u, desired_v = observation.desired_center_uv
        desired_width, desired_height = observation.desired_size_wh_px

        draw.rectangle(
            [u0, v0, u1 - 1, v1 - 1],
            outline=(0, 128, 255),
            width=1,
        )
        draw.rectangle(
            [x, y, x + width - 1, y + height - 1],
            outline=(0, 255, 0),
            width=2,
        )

        desired_box = [
            desired_u - desired_width / 2.0,
            desired_v - desired_height / 2.0,
            desired_u + desired_width / 2.0,
            desired_v + desired_height / 2.0,
        ]
        draw.rectangle(
            desired_box,
            outline=(0, 255, 255),
            width=2,
        )

        self._draw_crosshair(
            draw,
            detected_u,
            detected_v,
            color=(0, 255, 0),
        )
        self._draw_crosshair(
            draw,
            desired_u,
            desired_v,
            color=(0, 255, 255),
        )

        center_error = float(
            np.linalg.norm(observation.center_error_px)
        )
        label = (
            f"RGB SERVO  center_err={center_error:.1f}px  "
            f"range={observation.estimated_range_m * 1000.0:.1f}mm  "
            f"range_err={observation.range_error_m * 1000.0:+.1f}mm"
        )
        box = draw.textbbox((10, 10), label)
        draw.rectangle(
            [box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3],
            fill=(0, 0, 0),
        )
        draw.text((10, 10), label, fill=(255, 255, 255))

        legend = "green=detected  cyan=desired"
        legend_box = draw.textbbox((10, 30), legend)
        draw.rectangle(
            [
                legend_box[0] - 4,
                legend_box[1] - 3,
                legend_box[2] + 4,
                legend_box[3] + 3,
            ],
            fill=(0, 0, 0),
        )
        draw.text((10, 30), legend, fill=(255, 255, 255))

        return np.asarray(image, dtype=np.uint8).copy()

    def _draw_crosshair(
        self,
        draw: ImageDraw.ImageDraw,
        u: float,
        v: float,
        color: tuple[int, int, int],
    ) -> None:
        half = self.cfg.debug.crosshair_half_length_px
        width = self.cfg.debug.crosshair_width_px
        draw.line(
            [u - half, v, u + half, v],
            fill=color,
            width=width,
        )
        draw.line(
            [u, v - half, u, v + half],
            fill=color,
            width=width,
        )

    # ------------------------------------------------------------------
    # Concise normal-operation log
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(
        observation: PortObservation,
        capture_index: int,
    ) -> None:
        detection = observation.detection
        center_error = float(
            np.linalg.norm(observation.center_error_px)
        )
        correction = float(
            np.linalg.norm(observation.correction_world_m)
        )

        print(
            "[RGB SERVO] "
            f"capture={capture_index} "
            f"pixel=({detection.center_uv[0]:.1f}, "
            f"{detection.center_uv[1]:.1f}) "
            f"box={detection.bbox_xywh[2]}x{detection.bbox_xywh[3]} "
            f"range={observation.estimated_range_m * 1000.0:.1f}mm "
            f"center_error={center_error:.1f}px "
            f"range_error={observation.range_error_m * 1000.0:+.1f}mm "
            f"raw_correction={correction * 1000.0:.1f}mm "
            f"shape={detection.shape_score:.3f}",
            flush=True,
        )
