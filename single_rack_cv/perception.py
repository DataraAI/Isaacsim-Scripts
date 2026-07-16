#!/usr/bin/env python3
"""Pure NumPy/OpenCV stereo RGB port perception and servo geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from config import PerceptionConfig, YOLOEConfig


@dataclass(frozen=True)
class CameraModel:
    """Pinhole calibration plus the current USD camera-to-world matrix."""

    image_height_px: int
    image_width_px: int
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    world_from_camera: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.world_from_camera, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"world_from_camera must be 4x4, got {matrix.shape}."
            )
        object.__setattr__(self, "world_from_camera", matrix.copy())

    @property
    def fx_px(self) -> float:
        return (
            self.focal_length_mm
            * self.image_width_px
            / self.horizontal_aperture_mm
        )

    @property
    def fy_px(self) -> float:
        return (
            self.focal_length_mm
            * self.image_height_px
            / self.vertical_aperture_mm
        )

    @property
    def cx_px(self) -> float:
        return (self.image_width_px - 1) / 2.0

    @property
    def cy_px(self) -> float:
        return (self.image_height_px - 1) / 2.0

    @property
    def camera_from_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_from_camera)

    @property
    def camera_center_world_m(self) -> np.ndarray:
        return transform_point_to_world(
            np.zeros(3, dtype=np.float64),
            self.world_from_camera,
        )

    def camera_point_from_world(self, point_world_m: np.ndarray) -> np.ndarray:
        point = np.append(
            np.asarray(point_world_m, dtype=np.float64).reshape(3),
            1.0,
        )
        local = point @ self.camera_from_world
        if abs(local[3]) > 1.0e-12:
            local = local / local[3]
        return local[:3]

    def project_world(self, point_world_m: np.ndarray) -> np.ndarray:
        point = self.camera_point_from_world(point_world_m)
        range_m = -float(point[2])
        if range_m <= 0.0:
            raise RuntimeError("World point is behind the camera.")
        u = self.cx_px + self.fx_px * float(point[0]) / range_m
        v = self.cy_px + self.fy_px * (-float(point[1])) / range_m
        return np.array([u, v], dtype=np.float64)

    def pixel_to_world_ray(
        self,
        pixel_uv: np.ndarray | tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
        x = (u - self.cx_px) / self.fx_px
        y = -(v - self.cy_px) / self.fy_px
        local_direction = np.array([x, y, -1.0], dtype=np.float64)
        local_direction /= np.linalg.norm(local_direction)
        world_direction = (
            np.append(local_direction, 0.0) @ self.world_from_camera
        )[:3]
        world_direction /= np.linalg.norm(world_direction)
        return self.camera_center_world_m, world_direction


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    camera: CameraModel


@dataclass(frozen=True)
class StereoFrame:
    left: CameraFrame
    right: CameraFrame
    virtual_camera: CameraModel


@dataclass(frozen=True)
class PortDetection:
    bbox_xywh: tuple[int, int, int, int]
    center_uv: tuple[float, float]
    shape_score: float
    roi_uv: tuple[int, int, int, int]
    mask: np.ndarray

    @property
    def scale_px(self) -> float:
        _, _, width, height = self.bbox_xywh
        return math.sqrt(float(width * height))


@dataclass(frozen=True)
class PortCorners:
    detection: PortDetection
    corners_uv: np.ndarray

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_uv, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError(f"corners_uv must be (4,2), got {corners.shape}.")
        object.__setattr__(self, "corners_uv", corners.copy())


@dataclass(frozen=True)
class StereoTriangulation:
    corners_world_m: np.ndarray
    center_world_m: np.ndarray
    normal_world: np.ndarray
    width_m: float
    height_m: float
    reprojection_rms_px: float
    max_reprojection_px: float
    max_ray_gap_m: float
    plane_residual_m: float
    opposite_edge_ratio: float


@dataclass(frozen=True)
class StereoPortObservation:
    left: PortCorners
    right: PortCorners
    corners_world_m: np.ndarray
    center_world_xyz_m: np.ndarray
    center_virtual_camera_usd_m: np.ndarray
    normal_world: np.ndarray
    projected_virtual_center_uv: tuple[float, float]
    desired_center_uv: tuple[float, float]
    desired_size_wh_px: tuple[float, float]
    desired_left_center_uv: tuple[float, float]
    desired_right_center_uv: tuple[float, float]
    center_error_px: np.ndarray
    estimated_range_m: float
    range_error_m: float
    correction_world_m: np.ndarray
    width_m: float
    height_m: float
    mean_disparity_px: float
    reprojection_rms_px: float
    max_reprojection_px: float
    max_ray_gap_m: float
    plane_residual_m: float


# ---------------------------------------------------------------------------
# Image normalization and YOLOE full-frame visual-prompt detection
# ---------------------------------------------------------------------------


def normalize_rgb(
    rgb: np.ndarray,
    resolution_hw: tuple[int, int],
) -> np.ndarray:
    """Return contiguous HxWx3 uint8 RGB data."""
    height, width = resolution_hw
    rgb = np.asarray(rgb)

    if rgb.ndim == 1:
        pixels = height * width
        channels = 4 if rgb.size == pixels * 4 else 3
        if rgb.size != pixels * channels:
            raise ValueError(
                f"Cannot reshape flat RGB array of size {rgb.size}."
            )
        rgb = rgb.reshape(height, width, channels)

    if rgb.ndim != 3 or rgb.shape[:2] != (height, width):
        raise ValueError(
            f"RGB shape {rgb.shape} does not match {(height, width)}."
        )

    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    elif rgb.shape[2] != 3:
        raise ValueError(f"RGB must have 3 or 4 channels, got {rgb.shape}.")

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32, copy=False)
        if np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(rgb)


def _tensor_to_numpy(value) -> np.ndarray:
    """Convert an Ultralytics/Torch result tensor or array to NumPy."""
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)



def _component_touches_border(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> bool:
    return (
        x <= 0
        or y <= 0
        or x + width >= image_width
        or y + height >= image_height
    )


def refine_detection_to_dark_cavity(
    rgb: np.ndarray,
    coarse: PortDetection,
    cfg: YOLOEConfig,
) -> PortDetection:
    """Refine one semantic YOLOE proposal to the physical dark RJ45 opening.

    YOLOE remains responsible for the full-frame object search. This function
    only measures the dark cavity inside a returned proposal so stereo width,
    height, and center refer to the opening rather than the white bezel.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {rgb.shape}.")

    image_height, image_width = rgb.shape[:2]
    x, y, width, height = map(int, coarse.bbox_xywh)
    if width <= 0 or height <= 0:
        raise RuntimeError("YOLOE proposal has zero area.")

    margin = max(
        int(cfg.refine_min_margin_px),
        int(round(cfg.refine_expand_ratio * max(width, height))),
    )
    crop_x0 = max(0, x - margin)
    crop_y0 = max(0, y - margin)
    crop_x1 = min(image_width, x + width + margin)
    crop_y1 = min(image_height, y + height + margin)
    if crop_x1 - crop_x0 < 3 or crop_y1 - crop_y0 < 3:
        raise RuntimeError("YOLOE proposal is too close to the image boundary.")

    crop_rgb = np.ascontiguousarray(
        rgb[crop_y0:crop_y1, crop_x0:crop_x1]
    )
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)

    thresholds = []
    for percentile in cfg.refine_percentiles:
        value = int(round(float(np.percentile(gray, percentile))))
        value = int(np.clip(
            value,
            cfg.refine_min_gray,
            cfg.refine_max_gray,
        ))
        if value not in thresholds:
            thresholds.append(value)

    kernel_size = max(1, int(cfg.refine_morph_kernel_px))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    coarse_center = np.asarray(coarse.center_uv, dtype=np.float64)
    coarse_diagonal = max(
        1.0,
        math.hypot(float(width), float(height)),
    )
    coarse_area = max(1.0, float(width * height))

    candidates: list[
        tuple[float, tuple[int, int, int, int], np.ndarray]
    ] = []
    rejection_counts = {
        "border": 0,
        "size": 0,
        "aspect": 0,
        "fill": 0,
        "center": 0,
    }

    for threshold in thresholds:
        binary = np.where(gray <= threshold, 255, 0).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        for label_index in range(1, count):
            local_x, local_y, candidate_width, candidate_height, area = map(
                int,
                stats[label_index],
            )
            if _component_touches_border(
                local_x,
                local_y,
                candidate_width,
                candidate_height,
                binary.shape[1],
                binary.shape[0],
            ):
                rejection_counts["border"] += 1
                continue
            if not (
                cfg.refine_min_width_px
                <= candidate_width
                <= cfg.refine_max_width_px
                and cfg.refine_min_height_px
                <= candidate_height
                <= cfg.refine_max_height_px
            ):
                rejection_counts["size"] += 1
                continue

            aspect = candidate_width / max(1.0, float(candidate_height))
            if not (
                cfg.refine_min_aspect_ratio
                <= aspect
                <= cfg.refine_max_aspect_ratio
            ):
                rejection_counts["aspect"] += 1
                continue

            fill_ratio = area / max(
                1.0,
                float(candidate_width * candidate_height),
            )
            if not (
                cfg.refine_min_fill_ratio
                <= fill_ratio
                <= cfg.refine_max_fill_ratio
            ):
                rejection_counts["fill"] += 1
                continue

            absolute_x = crop_x0 + local_x
            absolute_y = crop_y0 + local_y
            center = np.asarray(
                [
                    absolute_x + 0.5 * candidate_width,
                    absolute_y + 0.5 * candidate_height,
                ],
                dtype=np.float64,
            )
            center_distance_ratio = (
                float(np.linalg.norm(center - coarse_center))
                / coarse_diagonal
            )
            if (
                center_distance_ratio
                > cfg.refine_max_center_distance_ratio
            ):
                rejection_counts["center"] += 1
                continue

            component_pixels = gray[labels == label_index]
            mean_gray = (
                float(np.mean(component_pixels))
                if component_pixels.size
                else 255.0
            )
            darkness_score = 1.0 - min(1.0, mean_gray / 255.0)
            center_score = max(
                0.0,
                1.0
                - center_distance_ratio
                / max(1.0e-6, cfg.refine_max_center_distance_ratio),
            )
            aspect_score = math.exp(
                -abs(
                    math.log(
                        max(1.0e-6, aspect)
                        / cfg.refine_target_aspect_ratio
                    )
                )
            )
            fill_score = max(
                0.0,
                1.0 - abs(fill_ratio - 0.65) / 0.65,
            )
            area_ratio = (
                candidate_width * candidate_height / coarse_area
            )
            size_score = math.exp(
                -abs(math.log(max(1.0e-6, area_ratio) / 0.40))
            )
            score = (
                0.33 * center_score
                + 0.25 * aspect_score
                + 0.17 * darkness_score
                + 0.15 * fill_score
                + 0.10 * size_score
                + 0.10 * float(coarse.shape_score)
            )

            component_mask = np.zeros(
                (image_height, image_width),
                dtype=np.uint8,
            )
            local_component = labels == label_index
            component_mask[
                crop_y0:crop_y1,
                crop_x0:crop_x1,
            ][local_component] = 255
            candidates.append(
                (
                    score,
                    (
                        absolute_x,
                        absolute_y,
                        candidate_width,
                        candidate_height,
                    ),
                    component_mask,
                )
            )

    if not candidates:
        raise RuntimeError(
            "no dark RJ45 cavity was found inside YOLOE proposal "
            f"{coarse.bbox_xywh}; thresholds={thresholds}; "
            f"rejections={rejection_counts}"
        )

    score, bbox, mask = max(candidates, key=lambda item: item[0])
    refined_x, refined_y, refined_width, refined_height = bbox
    return PortDetection(
        bbox_xywh=bbox,
        center_uv=(
            refined_x + 0.5 * refined_width,
            refined_y + 0.5 * refined_height,
        ),
        shape_score=float(
            np.clip(
                0.75 * coarse.shape_score + 0.25 * score,
                0.0,
                1.0,
            )
        ),
        roi_uv=(0, 0, image_width, image_height),
        mask=mask,
    )


class YOLOEPortDetector:
    """Long-lived full-frame YOLOE detector with local cavity measurement."""

    def __init__(self, cfg: YOLOEConfig):
        self.cfg = cfg
        self._model = None
        self._initialized = False
        self._last_diagnostics: dict[str, str] = {}

    def validate_reference_prompt(self) -> None:
        path = self.cfg.reference_image_path
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLOE reference image not found: {path}"
            )
        with Image.open(path) as image:
            width, height = image.size

        boxes = np.asarray(
            self.cfg.reference_boxes_xyxy,
            dtype=np.float64,
        )
        classes = np.asarray(
            self.cfg.reference_class_ids,
            dtype=np.int64,
        )
        if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] < 1:
            raise ValueError(
                "YOLOE reference_boxes_xyxy must contain one or more XYXY boxes."
            )
        if classes.shape != (boxes.shape[0],):
            raise ValueError(
                "YOLOE reference boxes and class IDs must have equal length."
            )
        if set(map(int, classes.tolist())) != {0}:
            raise ValueError(
                "All multiscale examples must use the same sequential class ID 0."
            )
        if not np.all(np.isfinite(boxes)):
            raise ValueError("YOLOE reference boxes must be finite.")
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(
            boxes[:, 3] <= boxes[:, 1]
        ):
            raise ValueError("YOLOE reference boxes must have positive area.")
        if (
            np.any(boxes[:, 0] < 0.0)
            or np.any(boxes[:, 1] < 0.0)
            or np.any(boxes[:, 2] > width)
            or np.any(boxes[:, 3] > height)
        ):
            raise ValueError(
                "A YOLOE reference box is outside reference image "
                f"{width}x{height}: {boxes.tolist()}"
            )

    def initialize(self) -> None:
        """Load weights and cache one multiscale visual class embedding."""
        if self._initialized:
            return
        if not self.cfg.enabled:
            raise RuntimeError("YOLOE detector is disabled in config.")
        self.validate_reference_prompt()
        try:
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics with YOLOE support is not installed in the "
                "Isaac Sim Python environment. Install/upgrade ultralytics "
                "before running this script."
            ) from exc

        self._model = YOLOE(self.cfg.model_name)
        visual_prompts = {
            "bboxes": np.asarray(
                self.cfg.reference_boxes_xyxy,
                dtype=np.float32,
            ),
            "cls": np.asarray(
                self.cfg.reference_class_ids,
                dtype=np.int64,
            ),
        }
        reference = str(self.cfg.reference_image_path)
        self._model.predict(
            source=reference,
            refer_image=reference,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            **self._predict_kwargs(),
        )
        self._initialized = True

    def _predict_kwargs(self) -> dict[str, object]:
        return {
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.confidence,
            "iou": self.cfg.iou,
            "device": self.cfg.device,
            "quantize": self.cfg.quantize,
            "max_det": self.cfg.max_detections,
            "retina_masks": self.cfg.retina_masks,
            "verbose": self.cfg.verbose,
        }

    def _predict_batch(self, rgb_images: list[np.ndarray]):
        if self._model is None:
            raise RuntimeError("YOLOE model has not been initialized.")
        bgr_images = [
            np.ascontiguousarray(image[:, :, ::-1])
            for image in rgb_images
        ]
        return self._model.predict(
            source=bgr_images,
            **self._predict_kwargs(),
        )

    def diagnostic(self, eye_name: str) -> str:
        return self._last_diagnostics.get(
            eye_name,
            "no YOLOE diagnostics were recorded",
        )

    def detections_from_result(
        self,
        result,
        rgb: np.ndarray,
        eye_name: str,
    ) -> list[PortDetection]:
        """Convert full-frame YOLOE proposals to refined dark-port openings."""
        image_height, image_width = map(int, rgb.shape[:2])
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            self._last_diagnostics[eye_name] = "raw_boxes=0"
            return []

        xyxy = _tensor_to_numpy(getattr(boxes, "xyxy", None))
        confidences = _tensor_to_numpy(
            getattr(boxes, "conf", None)
        ).reshape(-1)
        if xyxy.size == 0 or confidences.size == 0:
            self._last_diagnostics[eye_name] = "raw_boxes=0"
            return []
        xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)

        count = min(xyxy.shape[0], confidences.shape[0])
        detections: list[PortDetection] = []
        rejections: list[str] = []
        raw_summaries: list[str] = []

        for index in range(count):
            box = np.asarray(xyxy[index], dtype=np.float64)
            confidence = float(confidences[index])
            if not np.all(np.isfinite(box)):
                rejections.append(f"#{index}: non-finite box")
                continue

            x1f = float(np.clip(box[0], 0.0, float(image_width)))
            y1f = float(np.clip(box[1], 0.0, float(image_height)))
            x2f = float(np.clip(box[2], 0.0, float(image_width)))
            y2f = float(np.clip(box[3], 0.0, float(image_height)))
            if x2f <= x1f or y2f <= y1f:
                rejections.append(f"#{index}: zero-area box")
                continue

            x0 = max(0, min(image_width - 1, int(math.floor(x1f))))
            y0 = max(0, min(image_height - 1, int(math.floor(y1f))))
            x1 = max(x0 + 1, min(image_width, int(math.ceil(x2f))))
            y1 = max(y0 + 1, min(image_height, int(math.ceil(y2f))))
            width = x1 - x0
            height = y1 - y0
            raw_summaries.append(
                f"#{index} conf={confidence:.4f} "
                f"box=({x0},{y0},{width},{height})"
            )

            if width * height < self.cfg.min_proposal_area_px:
                rejections.append(f"#{index}: proposal area too small")
                continue
            if not (
                self.cfg.min_proposal_width_px
                <= width
                <= self.cfg.max_proposal_width_px
                and self.cfg.min_proposal_height_px
                <= height
                <= self.cfg.max_proposal_height_px
            ):
                rejections.append(f"#{index}: proposal dimensions rejected")
                continue

            proposal_mask = np.zeros(
                (image_height, image_width),
                dtype=np.uint8,
            )
            proposal_mask[y0:y1, x0:x1] = 255
            coarse = PortDetection(
                bbox_xywh=(x0, y0, width, height),
                center_uv=(
                    0.5 * (x1f + x2f),
                    0.5 * (y1f + y2f),
                ),
                shape_score=confidence,
                roi_uv=(0, 0, image_width, image_height),
                mask=proposal_mask,
            )
            try:
                detections.append(
                    refine_detection_to_dark_cavity(
                        rgb,
                        coarse,
                        self.cfg,
                    )
                )
            except RuntimeError as exc:
                rejections.append(f"#{index}: {exc}")

        detections.sort(key=lambda item: item.shape_score, reverse=True)
        raw_text = "; ".join(raw_summaries[:8]) or "none"
        rejection_text = "; ".join(rejections[:8]) or "none"
        self._last_diagnostics[eye_name] = (
            f"raw_boxes={count}; refined={len(detections)}; "
            f"raw=[{raw_text}]; rejected=[{rejection_text}]"
        )
        return detections

    def detect_stereo(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
    ) -> tuple[list[PortDetection], list[PortDetection]]:
        """Run one full-frame batch while preserving left/right result order."""
        if not self._initialized:
            self.initialize()
        if left_rgb.shape[:2] != right_rgb.shape[:2]:
            raise ValueError("Stereo eye images must have identical dimensions.")
        results = list(self._predict_batch([left_rgb, right_rgb]))
        if len(results) != 2:
            raise RuntimeError(
                f"YOLOE stereo batch returned {len(results)} results, expected 2."
            )
        return (
            self.detections_from_result(
                results[0],
                left_rgb,
                eye_name="left",
            ),
            self.detections_from_result(
                results[1],
                right_rgb,
                eye_name="right",
            ),
        )


# ---------------------------------------------------------------------------
# Camera and servo geometry
# ---------------------------------------------------------------------------


def compute_desired_port_camera_usd(
    camera_position_hand_m: np.ndarray,
    hand_from_camera: np.ndarray,
    tool_center_position_hand_m: np.ndarray,
    hand_from_tool: np.ndarray,
    preinsert_standoff_m: float,
) -> np.ndarray:
    camera_position = np.asarray(
        camera_position_hand_m,
        dtype=np.float64,
    ).reshape(3)
    tool_position = np.asarray(
        tool_center_position_hand_m,
        dtype=np.float64,
    ).reshape(3)
    hand_from_camera = np.asarray(
        hand_from_camera,
        dtype=np.float64,
    ).reshape(3, 3)
    hand_from_tool = np.asarray(
        hand_from_tool,
        dtype=np.float64,
    ).reshape(3, 3)
    if not math.isfinite(preinsert_standoff_m) or preinsert_standoff_m <= 0.0:
        raise ValueError("preinsert_standoff_m must be finite and positive.")
    port_in_tool = np.array([0.0, 0.0, preinsert_standoff_m])
    port_in_hand = tool_position + hand_from_tool @ port_in_tool
    port_in_camera = hand_from_camera.T @ (port_in_hand - camera_position)
    if port_in_camera[2] >= 0.0:
        raise RuntimeError(
            "Configured pre-insert point is not in front of the camera: "
            f"{np.round(port_in_camera, 6).tolist()}"
        )
    return port_in_camera


def project_port_feature(
    point_camera_usd_m: np.ndarray,
    camera: CameraModel,
    cfg: PerceptionConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    point = np.asarray(point_camera_usd_m, dtype=np.float64).reshape(3)
    range_m = -float(point[2])
    if range_m <= 0.0:
        raise ValueError("Desired port point must be in front of the camera.")
    u = camera.cx_px + camera.fx_px * float(point[0]) / range_m
    v = camera.cy_px + camera.fy_px * (-float(point[1])) / range_m
    width_px = camera.fx_px * cfg.port_width_m / range_m
    height_px = camera.fy_px * cfg.port_height_m / range_m
    return (float(u), float(v)), (float(width_px), float(height_px))


def transform_point_to_world(
    point_usd_local: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera matrix, got {matrix.shape}.")
    homogeneous = np.append(
        np.asarray(point_usd_local, dtype=np.float64),
        1.0,
    )
    world = homogeneous @ matrix
    if abs(world[3]) > 1.0e-12:
        world = world / world[3]
    return world[:3]


def camera_point_error_to_world(
    current_point_usd: np.ndarray,
    desired_point_usd: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    current = np.asarray(current_point_usd, dtype=np.float64).reshape(3)
    desired = np.asarray(desired_point_usd, dtype=np.float64).reshape(3)
    matrix = np.asarray(world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera matrix, got {matrix.shape}.")
    local_motion = current - desired
    return np.asarray((np.append(local_motion, 0.0) @ matrix)[:3])


def compute_bounded_step(
    correction_world_m: np.ndarray,
    gain: float,
    max_step_m: float,
) -> np.ndarray:
    correction = np.asarray(correction_world_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(correction)):
        raise ValueError("Visual correction must be finite.")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("Control gain must be finite and positive.")
    if not math.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError("Maximum target step must be finite and positive.")
    step = gain * correction
    norm = float(np.linalg.norm(step))
    if norm > max_step_m:
        step *= max_step_m / norm
    return step


def build_virtual_camera_model(
    left: CameraModel,
    right: CameraModel,
) -> CameraModel:
    """Return a mathematical midpoint eye; no rendered sensor is created."""
    scalar_pairs = (
        (left.image_height_px, right.image_height_px),
        (left.image_width_px, right.image_width_px),
        (left.focal_length_mm, right.focal_length_mm),
        (left.horizontal_aperture_mm, right.horizontal_aperture_mm),
        (left.vertical_aperture_mm, right.vertical_aperture_mm),
    )
    for a, b in scalar_pairs:
        if not np.isclose(a, b, atol=1.0e-9):
            raise ValueError("Stereo cameras must have identical intrinsics.")
    left_rotation = left.world_from_camera[:3, :3]
    right_rotation = right.world_from_camera[:3, :3]
    if not np.allclose(left_rotation, right_rotation, atol=1.0e-8):
        raise ValueError("Stereo cameras must have parallel optical axes.")
    matrix = left.world_from_camera.copy()
    matrix[3, :3] = (
        left.camera_center_world_m + right.camera_center_world_m
    ) / 2.0
    return CameraModel(
        image_height_px=left.image_height_px,
        image_width_px=left.image_width_px,
        focal_length_mm=left.focal_length_mm,
        horizontal_aperture_mm=left.horizontal_aperture_mm,
        vertical_aperture_mm=left.vertical_aperture_mm,
        world_from_camera=matrix,
    )


def triangulate_pixel_pair(
    left_uv: np.ndarray | tuple[float, float],
    right_uv: np.ndarray | tuple[float, float],
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> tuple[np.ndarray, float]:
    left_origin, left_direction = left_camera.pixel_to_world_ray(left_uv)
    right_origin, right_direction = right_camera.pixel_to_world_ray(right_uv)
    system = np.column_stack((left_direction, -right_direction))
    values, _, rank, _ = np.linalg.lstsq(
        system,
        right_origin - left_origin,
        rcond=None,
    )
    if rank < 2:
        raise RuntimeError("Stereo rays are parallel or numerically singular.")
    left_distance, right_distance = map(float, values)
    if left_distance <= 0.0 or right_distance <= 0.0:
        raise RuntimeError("Triangulated point lies behind a camera.")
    left_point = left_origin + left_distance * left_direction
    right_point = right_origin + right_distance * right_direction
    gap = float(np.linalg.norm(left_point - right_point))
    return (left_point + right_point) / 2.0, gap


def triangulate_detection_centers(
    left_detection: PortDetection,
    right_detection: PortDetection,
    left_camera: CameraModel,
    right_camera: CameraModel,
    virtual_camera: CameraModel,
) -> StereoTriangulation:
    """Triangulate the matched dark-port center and build a diagnostic plane."""
    center_world, ray_gap = triangulate_pixel_pair(
        left_detection.center_uv,
        right_detection.center_uv,
        left_camera,
        right_camera,
    )
    reprojection_errors = np.asarray(
        [
            np.linalg.norm(
                left_camera.project_world(center_world)
                - np.asarray(left_detection.center_uv, dtype=np.float64)
            ),
            np.linalg.norm(
                right_camera.project_world(center_world)
                - np.asarray(right_detection.center_uv, dtype=np.float64)
            ),
        ],
        dtype=np.float64,
    )
    center_virtual = virtual_camera.camera_point_from_world(center_world)
    range_m = -float(center_virtual[2])
    if range_m <= 0.0:
        raise RuntimeError("Triangulated center lies behind the virtual camera.")

    _, _, left_width_px, left_height_px = left_detection.bbox_xywh
    _, _, right_width_px, right_height_px = right_detection.bbox_xywh
    width_m = range_m * 0.5 * (
        float(left_width_px) / left_camera.fx_px
        + float(right_width_px) / right_camera.fx_px
    )
    height_m = range_m * 0.5 * (
        float(left_height_px) / left_camera.fy_px
        + float(right_height_px) / right_camera.fy_px
    )

    half_width = width_m / 2.0
    half_height = height_m / 2.0
    local_corners = np.asarray(
        [
            center_virtual + [-half_width, +half_height, 0.0],
            center_virtual + [+half_width, +half_height, 0.0],
            center_virtual + [+half_width, -half_height, 0.0],
            center_virtual + [-half_width, -half_height, 0.0],
        ],
        dtype=np.float64,
    )
    corners_world = np.vstack(
        [
            transform_point_to_world(point, virtual_camera.world_from_camera)
            for point in local_corners
        ]
    )
    normal_world = (
        np.asarray([0.0, 0.0, 1.0, 0.0])
        @ virtual_camera.world_from_camera
    )[:3]
    normal_world /= np.linalg.norm(normal_world)

    rms = float(np.sqrt(np.mean(reprojection_errors * reprojection_errors)))
    maximum = float(np.max(reprojection_errors))
    return StereoTriangulation(
        corners_world_m=corners_world,
        center_world_m=center_world,
        normal_world=normal_world,
        width_m=float(width_m),
        height_m=float(height_m),
        reprojection_rms_px=rms,
        max_reprojection_px=maximum,
        max_ray_gap_m=float(ray_gap),
        plane_residual_m=0.0,
        opposite_edge_ratio=1.0,
    )


def _edge_ratio(a: float, b: float) -> float:
    minimum = min(a, b)
    if minimum <= 1.0e-12:
        return math.inf
    return max(a, b) / minimum


def triangulate_port_corners(
    left_corners_uv: np.ndarray,
    right_corners_uv: np.ndarray,
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> StereoTriangulation:
    left_uv = np.asarray(left_corners_uv, dtype=np.float64)
    right_uv = np.asarray(right_corners_uv, dtype=np.float64)
    if left_uv.shape != (4, 2) or right_uv.shape != (4, 2):
        raise ValueError("Stereo corner arrays must both have shape (4,2).")
    points: list[np.ndarray] = []
    ray_gaps: list[float] = []
    reprojection_errors: list[float] = []
    for left_pixel, right_pixel in zip(left_uv, right_uv, strict=True):
        point, gap = triangulate_pixel_pair(
            left_pixel,
            right_pixel,
            left_camera,
            right_camera,
        )
        points.append(point)
        ray_gaps.append(gap)
        reprojection_errors.extend(
            [
                float(np.linalg.norm(left_camera.project_world(point) - left_pixel)),
                float(np.linalg.norm(right_camera.project_world(point) - right_pixel)),
            ]
        )
    corners = np.vstack(points)
    center = np.mean(corners, axis=0)
    _, _, vh = np.linalg.svd(corners - center)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    camera_midpoint = (
        left_camera.camera_center_world_m
        + right_camera.camera_center_world_m
    ) / 2.0
    if float(np.dot(normal, camera_midpoint - center)) < 0.0:
        normal *= -1.0
    residuals = np.abs((corners - center) @ normal)
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left_edge = float(np.linalg.norm(corners[3] - corners[0]))
    right_edge = float(np.linalg.norm(corners[2] - corners[1]))
    width = (top + bottom) / 2.0
    height = (left_edge + right_edge) / 2.0
    opposite_ratio = max(
        _edge_ratio(top, bottom),
        _edge_ratio(left_edge, right_edge),
    )
    reprojection = np.asarray(reprojection_errors, dtype=np.float64)
    rms = float(np.sqrt(np.mean(reprojection * reprojection)))
    maximum = float(np.max(reprojection))
    max_gap = float(np.max(ray_gaps))
    plane_residual = float(np.max(residuals))

    # Broad structural gates belong here so obviously wrong corner ordering
    # never reaches application-specific dimension checks.
    if not np.all(np.isfinite(corners)):
        raise RuntimeError("Triangulation produced non-finite corners.")
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("Triangulated rectangle has zero-sized edges.")
    if max_gap > 0.010 or maximum > 10.0:
        raise RuntimeError("Corner correspondence has excessive ray/reprojection error.")
    if opposite_ratio > 3.0:
        raise RuntimeError("Corner correspondence does not form a rectangle.")

    return StereoTriangulation(
        corners_world_m=corners,
        center_world_m=center,
        normal_world=normal,
        width_m=width,
        height_m=height,
        reprojection_rms_px=rms,
        max_reprojection_px=maximum,
        max_ray_gap_m=max_gap,
        plane_residual_m=plane_residual,
        opposite_edge_ratio=opposite_ratio,
    )


# ---------------------------------------------------------------------------
# YOLOE stereo processing
# ---------------------------------------------------------------------------


def _candidate_continuity_ok(
    candidate: PortDetection,
    previous: PortDetection | None,
    cfg: PerceptionConfig,
) -> bool:
    if previous is None:
        return True
    center_distance = float(
        np.linalg.norm(
            np.asarray(candidate.center_uv) - np.asarray(previous.center_uv)
        )
    )
    scale_ratio = max(
        candidate.scale_px / previous.scale_px,
        previous.scale_px / candidate.scale_px,
    )
    return (
        center_distance <= cfg.tracking_max_center_jump_px
        and scale_ratio <= cfg.tracking_max_scale_ratio
    )


def _validate_stereo_result(
    result: StereoTriangulation,
    virtual_camera: CameraModel,
    cfg: PerceptionConfig,
) -> None:
    center_virtual = virtual_camera.camera_point_from_world(
        result.center_world_m
    )
    range_m = -float(center_virtual[2])
    checks = (
        (
            cfg.min_estimated_range_m <= range_m <= cfg.max_estimated_range_m,
            "stereo range is outside the configured working distance",
        ),
        (
            result.reprojection_rms_px <= cfg.stereo_max_reprojection_rms_px,
            "stereo reprojection RMS is too high",
        ),
        (
            result.max_reprojection_px <= cfg.stereo_max_reprojection_px,
            "stereo center reprojection error is too high",
        ),
        (
            result.max_ray_gap_m <= cfg.stereo_max_ray_gap_m,
            "stereo rays do not intersect closely enough",
        ),
        (
            cfg.stereo_min_width_m <= result.width_m <= cfg.stereo_max_width_m,
            "triangulated port width is implausible",
        ),
        (
            cfg.stereo_min_height_m <= result.height_m <= cfg.stereo_max_height_m,
            "triangulated port height is implausible",
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            raise RuntimeError(reason)


def _box_port_corners(detection: PortDetection) -> PortCorners:
    """Provide deterministic box corners for overlays; depth uses mask centers."""
    x, y, width, height = detection.bbox_xywh
    corners = np.asarray(
        [
            [x, y],
            [x + width - 1, y],
            [x + width - 1, y + height - 1],
            [x, y + height - 1],
        ],
        dtype=np.float64,
    )
    return PortCorners(detection=detection, corners_uv=corners)


def process_stereo_port(
    frame: StereoFrame,
    cfg: PerceptionConfig,
    desired_port_virtual_camera_usd: np.ndarray,
    previous_left: PortDetection | None,
    previous_right: PortDetection | None,
    detector: YOLOEPortDetector,
) -> StereoPortObservation:
    """Require YOLOE in both full eye views and compute one stereo correction."""
    left_candidates, right_candidates = detector.detect_stereo(
        frame.left.rgb,
        frame.right.rgb,
    )
    left_candidates = [
        candidate
        for candidate in left_candidates
        if _candidate_continuity_ok(candidate, previous_left, cfg)
    ]
    right_candidates = [
        candidate
        for candidate in right_candidates
        if _candidate_continuity_ok(candidate, previous_right, cfg)
    ]
    if not left_candidates:
        raise RuntimeError(
            "left eye did not return a refined YOLOE port: "
            + detector.diagnostic("left")
        )
    if not right_candidates:
        raise RuntimeError(
            "right eye did not return a refined YOLOE port: "
            + detector.diagnostic("right")
        )

    pair_results: list[
        tuple[float, PortCorners, PortCorners, StereoTriangulation]
    ] = []
    pair_rejections: list[str] = []
    for left_detection in left_candidates:
        for right_detection in right_candidates:
            vertical_error = abs(
                left_detection.center_uv[1] - right_detection.center_uv[1]
            )
            if vertical_error > cfg.stereo_max_epipolar_error_px:
                continue
            scale_ratio = max(
                left_detection.scale_px / right_detection.scale_px,
                right_detection.scale_px / left_detection.scale_px,
            )
            if scale_ratio > cfg.stereo_max_scale_ratio:
                continue
            disparity = abs(
                left_detection.center_uv[0] - right_detection.center_uv[0]
            )
            if disparity < cfg.stereo_min_abs_disparity_px:
                continue
            try:
                result = triangulate_detection_centers(
                    left_detection,
                    right_detection,
                    frame.left.camera,
                    frame.right.camera,
                    frame.virtual_camera,
                )
                _validate_stereo_result(result, frame.virtual_camera, cfg)
            except RuntimeError as exc:
                pair_rejections.append(str(exc))
                continue
            dimension_error = (
                abs(result.width_m - cfg.port_width_m) / cfg.port_width_m
                + abs(result.height_m - cfg.port_height_m) / cfg.port_height_m
            )
            score = (
                left_detection.shape_score
                + right_detection.shape_score
                - 0.05 * vertical_error
                - 0.75 * dimension_error
                - result.reprojection_rms_px
            )
            pair_results.append(
                (
                    score,
                    _box_port_corners(left_detection),
                    _box_port_corners(right_detection),
                    result,
                )
            )

    if not pair_results:
        detail = pair_rejections[-1] if pair_rejections else (
            "epipolar, scale, or disparity gate rejected every pair"
        )
        raise RuntimeError(
            "no left/right YOLOE port pair passed stereo correspondence and "
            f"geometry checks: {detail}"
        )
    _, left, right, result = max(pair_results, key=lambda item: item[0])

    virtual_camera = frame.virtual_camera
    center_virtual = virtual_camera.camera_point_from_world(
        result.center_world_m
    )
    desired = np.asarray(
        desired_port_virtual_camera_usd,
        dtype=np.float64,
    ).reshape(3)
    desired_center, desired_size = project_port_feature(
        desired,
        virtual_camera,
        cfg,
    )
    projected_center = virtual_camera.project_world(result.center_world_m)
    center_error = projected_center - np.asarray(desired_center)
    estimated_range_m = -float(center_virtual[2])
    range_error_m = estimated_range_m - (-float(desired[2]))
    correction_world = camera_point_error_to_world(
        center_virtual,
        desired,
        virtual_camera.world_from_camera,
    )
    desired_world = transform_point_to_world(
        desired,
        virtual_camera.world_from_camera,
    )
    desired_left = frame.left.camera.project_world(desired_world)
    desired_right = frame.right.camera.project_world(desired_world)
    center_disparity = abs(
        left.detection.center_uv[0] - right.detection.center_uv[0]
    )

    return StereoPortObservation(
        left=left,
        right=right,
        corners_world_m=result.corners_world_m,
        center_world_xyz_m=result.center_world_m,
        center_virtual_camera_usd_m=center_virtual,
        normal_world=result.normal_world,
        projected_virtual_center_uv=(
            float(projected_center[0]),
            float(projected_center[1]),
        ),
        desired_center_uv=desired_center,
        desired_size_wh_px=desired_size,
        desired_left_center_uv=(float(desired_left[0]), float(desired_left[1])),
        desired_right_center_uv=(float(desired_right[0]), float(desired_right[1])),
        center_error_px=center_error,
        estimated_range_m=estimated_range_m,
        range_error_m=float(range_error_m),
        correction_world_m=correction_world,
        width_m=result.width_m,
        height_m=result.height_m,
        mean_disparity_px=float(center_disparity),
        reprojection_rms_px=result.reprojection_rms_px,
        max_reprojection_px=result.max_reprojection_px,
        max_ray_gap_m=result.max_ray_gap_m,
        plane_residual_m=result.plane_residual_m,
    )
