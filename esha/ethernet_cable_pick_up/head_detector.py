"""YOLOE-based coarse localization for the ethernet connector head.
Trimmed version of single_rack_cv/perception.py's YOLOEPortDetector:
same model-loading/predict machinery, no dark-cavity refinement (that
step is replaced by esha's own detect_cable_candidates()/roi_uv, which
is already tuned for this object's actual appearance)."""

from __future__ import annotations

import numpy as np

from config import CableHeadYOLOEConfig


class YOLOEHeadDetector:
    def __init__(self, cfg: CableHeadYOLOEConfig):
        self.cfg = cfg
        self._model = None

    def initialize(self) -> None:
        try:
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics with YOLOE support is not installed. "
                "Install/upgrade ultralytics before running this script."
            ) from exc

        self._model = YOLOE(self.cfg.model_name)
        visual_prompts = {
            "bboxes": np.asarray(self.cfg.reference_boxes_xyxy, dtype=np.float32),
            "cls": np.asarray(self.cfg.reference_class_ids, dtype=np.int64),
        }
        reference = str(self.cfg.reference_image_path)
        self._model.predict(
            source=reference,
            refer_image=reference,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            device=self.cfg.device,
            quantize=self.cfg.quantize,
        )

    def coarse_roi_stereo(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        margin_px: int = 40,
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        if self._model is None:
            raise RuntimeError("call initialize() first")

        results = self._model.predict(
            source=[left_rgb[:, :, ::-1], right_rgb[:, :, ::-1]],
            imgsz=self.cfg.imgsz,
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            device=self.cfg.device,
            verbose=False,
        )

        rois: list[tuple[int, int, int, int] | None] = []
        for result, rgb in zip(results, (left_rgb, right_rgb)):
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy.shape[0] == 0:
                rois.append(None)
                continue
            widths = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]).cpu().numpy()
            heights = (boxes.xyxy[:, 3] - boxes.xyxy[:, 1]).cpu().numpy()
            valid = (
                (widths >= self.cfg.min_proposal_width_px)
                & (widths <= self.cfg.max_proposal_width_px)
                & (heights >= self.cfg.min_proposal_height_px)
                & (heights <= self.cfg.max_proposal_height_px)
            )
            if not valid.any():
                rois.append(None)
                continue
            confs = boxes.conf.cpu().numpy()
            confs[~valid] = -1.0
            best = boxes.xyxy[confs.argmax()].cpu().numpy()
            h, w = rgb.shape[:2]
            x0 = max(0, int(best[0]) - margin_px)
            y0 = max(0, int(best[1]) - margin_px)
            x1 = min(w, int(best[2]) + margin_px)
            y1 = min(h, int(best[3]) + margin_px)
            rois.append((x0, y0, x1, y1))

        return tuple(rois)
