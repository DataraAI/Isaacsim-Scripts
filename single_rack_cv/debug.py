#!/usr/bin/env python3
"""Stereo RGB markers, overlays, output files, and concise logs."""

from __future__ import annotations

import numpy as np
import omni.usd
from PIL import Image, ImageDraw
from pxr import Gf, UsdGeom

from config import Config
from perception import PortCorners, StereoFrame, StereoPortObservation


class DebugOutputs:
    """Own all stereo visualization and file-output side effects."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.output_dir = cfg.camera.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(self, frame: StereoFrame) -> None:
        """Overwrite the latest raw eye images even when stereo is rejected."""
        Image.fromarray(frame.left.rgb, mode="RGB").save(
            self.output_dir / "rgb_left_latest.png"
        )
        Image.fromarray(frame.right.rgb, mode="RGB").save(
            self.output_dir / "rgb_right_latest.png"
        )

    def handle(
        self,
        frame: StereoFrame,
        observation: StereoPortObservation,
        capture_index: int,
    ) -> None:
        self.update_stage(observation)
        self.save_files(frame, observation)
        self.print_summary(observation, capture_index)

    def update_stage(self, observation: StereoPortObservation) -> None:
        cfg = self.cfg.debug
        self._update_sphere(
            cfg.estimated_port_marker_path,
            observation.center_world_xyz_m,
            cfg.estimated_port_marker_radius_m,
            cfg.estimated_port_marker_color,
        )

    def update_frozen_port_point(self, position: np.ndarray) -> None:
        """Show the immutable qualified mouth point separately from live vision."""

        self._update_sphere(
            "/World/FrozenPortPoint",
            np.asarray(position, dtype=np.float64),
            0.0015,
            (1.0, 0.0, 1.0),
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

    def save_files(
        self,
        frame: StereoFrame,
        observation: StereoPortObservation,
    ) -> None:
        left_overlay = self._draw_eye_overlay(
            frame.left.rgb,
            observation.left,
            observation.desired_left_center_uv,
            observation.desired_size_wh_px,
            "YOLOE LEFT EYE",
        )
        right_overlay = self._draw_eye_overlay(
            frame.right.rgb,
            observation.right,
            observation.desired_right_center_uv,
            observation.desired_size_wh_px,
            "YOLOE RIGHT EYE",
        )

        Image.fromarray(frame.left.rgb, mode="RGB").save(
            self.output_dir / "rgb_left_latest.png"
        )
        Image.fromarray(frame.right.rgb, mode="RGB").save(
            self.output_dir / "rgb_right_latest.png"
        )
        Image.fromarray(left_overlay, mode="RGB").save(
            self.output_dir / "rgb_left_port_detected.png"
        )
        Image.fromarray(right_overlay, mode="RGB").save(
            self.output_dir / "rgb_right_port_detected.png"
        )
        Image.fromarray(observation.left.detection.mask, mode="L").save(
            self.output_dir / "port_detection_mask_left.png"
        )
        Image.fromarray(observation.right.detection.mask, mode="L").save(
            self.output_dir / "port_detection_mask_right.png"
        )
        Image.fromarray(
            self._draw_stereo_summary(left_overlay, right_overlay, observation),
            mode="RGB",
        ).save(self.output_dir / "stereo_port_detected.png")

    def _draw_eye_overlay(
        self,
        rgb: np.ndarray,
        port: PortCorners,
        desired_center_uv: tuple[float, float],
        desired_size_wh_px: tuple[float, float],
        title: str,
    ) -> np.ndarray:
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        detection = port.detection
        u0, v0, u1, v1 = detection.roi_uv
        x, y, width, height = detection.bbox_xywh
        draw.rectangle([u0, v0, u1 - 1, v1 - 1], outline=(0, 128, 255), width=1)
        draw.rectangle(
            [x, y, x + width - 1, y + height - 1],
            outline=(0, 255, 0),
            width=2,
        )
        corners = port.corners_uv
        polygon = [tuple(point) for point in corners] + [tuple(corners[0])]
        draw.line(polygon, fill=(255, 128, 0), width=2)
        for index, (u, v) in enumerate(corners):
            self._draw_crosshair(draw, float(u), float(v), (255, 128, 0), half=5)
            draw.text((float(u) + 4, float(v) + 2), str(index), fill=(255, 255, 255))

        desired_u, desired_v = desired_center_uv
        desired_width, desired_height = desired_size_wh_px
        draw.rectangle(
            [
                desired_u - desired_width / 2.0,
                desired_v - desired_height / 2.0,
                desired_u + desired_width / 2.0,
                desired_v + desired_height / 2.0,
            ],
            outline=(0, 255, 255),
            width=2,
        )
        self._draw_crosshair(draw, desired_u, desired_v, (0, 255, 255))
        self._label(draw, (10, 10), title)
        return np.asarray(image, dtype=np.uint8).copy()

    def _draw_stereo_summary(
        self,
        left_overlay: np.ndarray,
        right_overlay: np.ndarray,
        observation: StereoPortObservation,
    ) -> np.ndarray:
        combined = np.concatenate((left_overlay, right_overlay), axis=1)
        image = Image.fromarray(combined, mode="RGB")
        draw = ImageDraw.Draw(image)
        height, width = left_overlay.shape[:2]
        for corner_index in range(4):
            left_point = observation.left.corners_uv[corner_index]
            right_point = observation.right.corners_uv[corner_index]
            draw.line(
                [
                    (float(left_point[0]), float(left_point[1])),
                    (float(right_point[0]) + width, float(right_point[1])),
                ],
                fill=(255, 0, 255),
                width=1,
            )
        center_error = float(np.linalg.norm(observation.center_error_px))
        label = (
            f"STEREO  range={observation.estimated_range_m * 1000.0:.2f}mm  "
            f"disp={observation.mean_disparity_px:.2f}px  "
            f"reproj={observation.reprojection_rms_px:.3f}px  "
            f"size={observation.width_m * 1000.0:.2f}x"
            f"{observation.height_m * 1000.0:.2f}mm  "
            f"center_err={center_error:.2f}px  "
            f"conf={observation.left.detection.shape_score:.2f}/"
            f"{observation.right.detection.shape_score:.2f}"
        )
        self._label(draw, (10, height - 24), label)
        return np.asarray(image, dtype=np.uint8).copy()

    @staticmethod
    def _label(
        draw: ImageDraw.ImageDraw,
        position: tuple[int, int],
        text: str,
    ) -> None:
        box = draw.textbbox(position, text)
        draw.rectangle(
            [box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3],
            fill=(0, 0, 0),
        )
        draw.text(position, text, fill=(255, 255, 255))

    def _draw_crosshair(
        self,
        draw: ImageDraw.ImageDraw,
        u: float,
        v: float,
        color: tuple[int, int, int],
        half: int | None = None,
    ) -> None:
        half_length = (
            self.cfg.debug.crosshair_half_length_px
            if half is None
            else half
        )
        width = self.cfg.debug.crosshair_width_px
        draw.line([u - half_length, v, u + half_length, v], fill=color, width=width)
        draw.line([u, v - half_length, u, v + half_length], fill=color, width=width)

    @staticmethod
    def print_summary(
        observation: StereoPortObservation,
        capture_index: int,
    ) -> None:
        center_error = float(np.linalg.norm(observation.center_error_px))
        correction = float(np.linalg.norm(observation.correction_world_m))
        left_center = observation.left.detection.center_uv
        right_center = observation.right.detection.center_uv
        print(
            "[RGB STEREO SERVO] "
            f"capture={capture_index} "
            f"left=({left_center[0]:.1f},{left_center[1]:.1f}) "
            f"right=({right_center[0]:.1f},{right_center[1]:.1f}) "
            f"disparity={observation.mean_disparity_px:.2f}px "
            f"range={observation.estimated_range_m * 1000.0:.2f}mm "
            f"center_error={center_error:.2f}px "
            f"range_error={observation.range_error_m * 1000.0:+.2f}mm "
            f"size={observation.width_m * 1000.0:.2f}x"
            f"{observation.height_m * 1000.0:.2f}mm "
            f"reproj={observation.reprojection_rms_px:.3f}px "
            f"ray_gap={observation.max_ray_gap_m * 1000.0:.3f}mm "
            f"raw_correction={correction * 1000.0:.2f}mm "
            f"yoloe_conf={observation.left.detection.shape_score:.3f}/"
            f"{observation.right.detection.shape_score:.3f}",
            flush=True,
        )
