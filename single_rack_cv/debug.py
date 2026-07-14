#!/usr/bin/env python3
"""Debug markers, annotated images, output files, and concise summaries."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import omni.usd
from PIL import Image, ImageDraw
from pxr import Gf, UsdGeom

from config import Config
from perception import CameraFrame, PortEstimate


class DebugOutputs:
    """Own all visualization and file-output side effects."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.output_dir = cfg.camera.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def handle(
        self,
        frame: CameraFrame,
        estimate: PortEstimate,
        capture_index: int,
    ) -> None:
        self.update_stage(estimate)
        self.save_files(frame, estimate)
        self.print_summary(estimate, capture_index)

    # ------------------------------------------------------------------
    # Stage markers
    # ------------------------------------------------------------------

    def update_stage(self, estimate: PortEstimate) -> None:
        debug = self.cfg.debug

        self._update_sphere(
            debug.cavity_marker_path,
            estimate.cavity_world_xyz_m,
            debug.cavity_marker_radius_m,
            debug.cavity_marker_color,
        )
        self._update_sphere(
            debug.opening_marker_path,
            estimate.opening_world_xyz_m,
            debug.opening_marker_radius_m,
            debug.opening_marker_color,
        )
        self._update_sphere(
            debug.preinsert_marker_path,
            estimate.preinsert_world_xyz_m,
            debug.preinsert_marker_radius_m,
            debug.preinsert_marker_color,
        )
        self._update_normal_arrow(
            estimate.opening_world_xyz_m,
            estimate.outward_world_normal,
            self.cfg.perception.preinsert_standoff_m,
        )

    def _update_sphere(
        self,
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

        self._set_translate(prim, position)

    def _update_normal_arrow(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        length: float,
    ) -> None:
        cfg = self.cfg.debug
        stage = omni.usd.get_context().get_stage()

        root_path = cfg.normal_root_path
        shaft_path = f"{root_path}/Shaft"
        tip_path = f"{root_path}/Tip"

        tip_length = cfg.normal_tip_length_m
        shaft_length = length - tip_length

        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            root = stage.DefinePrim(root_path, "Xform")
            root_xform = UsdGeom.Xformable(root)
            root_xform.ClearXformOpOrder()
            root_xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(0.0, 0.0, 0.0))
            root_xform.AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))

            UsdGeom.Imageable(root).CreatePurposeAttr().Set(
                UsdGeom.Tokens.guide
            )

            shaft = UsdGeom.Cylinder.Define(stage, shaft_path)
            shaft.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            shaft.CreateRadiusAttr().Set(cfg.normal_shaft_radius_m)
            shaft.CreateHeightAttr().Set(shaft_length)
            shaft.CreateDisplayColorAttr().Set(
                [Gf.Vec3f(*cfg.normal_color)]
            )
            shaft_xform = UsdGeom.Xformable(shaft.GetPrim())
            shaft_xform.ClearXformOpOrder()
            shaft_xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(0.0, 0.0, shaft_length / 2.0))

            tip = UsdGeom.Cone.Define(stage, tip_path)
            tip.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            tip.CreateRadiusAttr().Set(cfg.normal_tip_radius_m)
            tip.CreateHeightAttr().Set(tip_length)
            tip.CreateDisplayColorAttr().Set(
                [Gf.Vec3f(*cfg.normal_color)]
            )
            tip_xform = UsdGeom.Xformable(tip.GetPrim())
            tip_xform.ClearXformOpOrder()
            tip_xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(
                Gf.Vec3d(
                    0.0,
                    0.0,
                    shaft_length + tip_length / 2.0,
                )
            )

        quaternion = self._quaternion_from_positive_z(direction)
        self._set_pose(root, origin, quaternion)

    @staticmethod
    def _quaternion_from_positive_z(
        direction: np.ndarray,
    ) -> np.ndarray:
        target = np.asarray(direction, dtype=np.float64)
        target /= np.linalg.norm(target)

        source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

        if dot > 1.0 - 1.0e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        if dot < -1.0 + 1.0e-12:
            return np.array([0.0, 1.0, 0.0, 0.0])

        cross = np.cross(source, target)
        quaternion = np.array(
            [1.0 + dot, cross[0], cross[1], cross[2]],
            dtype=np.float64,
        )
        return quaternion / np.linalg.norm(quaternion)

    @staticmethod
    def _set_translate(prim, position: np.ndarray) -> None:
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*np.asarray(position).tolist()))
                return
        raise RuntimeError(f"Missing translate op on {prim.GetPath()}")

    @staticmethod
    def _set_pose(
        prim,
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
    ) -> None:
        xform = UsdGeom.Xformable(prim)
        translate = None
        orient = None

        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient = op

        if translate is None or orient is None:
            raise RuntimeError(f"Missing pose ops on {prim.GetPath()}")

        position = np.asarray(position, dtype=np.float64)
        q = np.asarray(orientation_wxyz, dtype=np.float64)
        q /= np.linalg.norm(q)

        translate.Set(Gf.Vec3d(*position.tolist()))
        orient.Set(
            Gf.Quatd(
                float(q[0]),
                Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])),
            )
        )

    # ------------------------------------------------------------------
    # Files and overlays
    # ------------------------------------------------------------------

    def save_files(
        self,
        frame: CameraFrame,
        estimate: PortEstimate,
    ) -> None:
        Image.fromarray(frame.rgb, mode="RGB").save(
            self.output_dir / "rgb_latest.png"
        )
        Image.fromarray(
            self._depth_preview(frame.depth_m),
            mode="L",
        ).save(self.output_dir / "depth_preview_latest.png")

        np.save(
            self.output_dir / "depth_latest_meters.npy",
            frame.depth_m,
        )

        Image.fromarray(
            self._draw_overlay(frame.rgb, estimate),
            mode="RGB",
        ).save(self.output_dir / "rgb_port_detected.png")

        Image.fromarray(
            estimate.detection.mask,
            mode="L",
        ).save(self.output_dir / "port_detection_mask.png")

    def _draw_overlay(
        self,
        rgb: np.ndarray,
        estimate: PortEstimate,
    ) -> np.ndarray:
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)

        detection = estimate.detection
        u0, v0, u1, v1 = detection.roi_uv
        x, y, w, h = detection.bbox_xywh
        u, v = detection.center_uv
        ring_x0, ring_y0, ring_x1, ring_y1 = (
            estimate.opening.ring_bounds_xyxy
        )
        patch_u0, patch_v0, patch_u1, patch_v1 = (
            estimate.cavity.patch_bounds_uv
        )

        draw.rectangle(
            [u0, v0, u1 - 1, v1 - 1],
            outline=(0, 128, 255),
            width=1,
        )
        draw.rectangle(
            [ring_x0, ring_y0, ring_x1 - 1, ring_y1 - 1],
            outline=(0, 255, 255),
            width=2,
        )
        draw.rectangle(
            [x, y, x + w - 1, y + h - 1],
            outline=(0, 255, 0),
            width=2,
        )
        draw.rectangle(
            [patch_u0, patch_v0, patch_u1 - 1, patch_v1 - 1],
            outline=(255, 255, 0),
            width=1,
        )

        half = self.cfg.debug.crosshair_half_length_px
        width_px = self.cfg.debug.crosshair_width_px
        draw.line(
            [u - half, v, u + half, v],
            fill=(0, 255, 0),
            width=width_px,
        )
        draw.line(
            [u, v - half, u, v + half],
            fill=(0, 255, 0),
            width=width_px,
        )

        label = (
            f"PORT ({u}, {v})  "
            f"opening={estimate.opening.depth_m:.4f} m  "
            f"recess={estimate.opening.recess_depth_m * 1000.0:.1f} mm"
        )
        box = draw.textbbox((10, 10), label)
        draw.rectangle(
            [box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3],
            fill=(0, 0, 0),
        )
        draw.text((10, 10), label, fill=(255, 255, 255))

        return np.asarray(image, dtype=np.uint8).copy()

    @staticmethod
    def _depth_preview(depth_m: np.ndarray) -> np.ndarray:
        valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
        values = depth_m[valid_mask]

        if values.size == 0:
            raise RuntimeError("Depth frame contains no valid values.")

        near = float(np.percentile(values, 2.0))
        far = float(np.percentile(values, 98.0))
        if far <= near:
            far = near + 1.0e-6

        normalized = np.clip((depth_m - near) / (far - near), 0.0, 1.0)
        preview = ((1.0 - normalized) * 255.0).astype(np.uint8)
        preview[~valid_mask] = 0
        return preview

    # ------------------------------------------------------------------
    # Compact normal-operation log
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(
        estimate: PortEstimate,
        capture_index: int,
    ) -> None:
        point = np.round(estimate.preinsert_world_xyz_m, 4).tolist()

        print(
            "[PORT] "
            f"capture={capture_index} "
            f"pixel={estimate.detection.center_uv} "
            f"opening={estimate.opening.depth_m:.4f}m "
            f"recess={estimate.opening.recess_depth_m * 1000.0:.1f}mm "
            f"normal_error={estimate.plane.camera_angle_deg:.1f}deg "
            f"plane_rms={estimate.plane.rms_residual_m * 1000.0:.3f}mm "
            f"preinsert={point}",
            flush=True,
        )
