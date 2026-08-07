#!/usr/bin/env python3
"""Pure frozen-axis state machine for guarded cable approach and insertion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


_EPS = 1.0e-12


def orientation_abort_is_due(
    *,
    target_error_m: float,
    settle_tolerance_m: float,
    actual_port_depth_m: float,
) -> bool:
    """
    Decide when the strict insertion orientation limit must be enforced.

    A non-contact translation can produce a brief orientation transient while
    the ToolCenter is still converging. The limit becomes active as soon as the
    ToolCenter settles or the plug reaches the opening plane.
    """

    return bool(
        target_error_m <= settle_tolerance_m
        or actual_port_depth_m >= -_EPS
    )


def _vector3(value, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{label} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values")
    return vector.copy()


def _quaternion_wxyz(value, *, label: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,):
        raise ValueError(f"{label} must have shape (4,), got {quaternion.shape}")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{label} must contain only finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= _EPS:
        raise ValueError(f"{label} cannot have zero length")
    return quaternion / norm


def _normalized_axis(value) -> np.ndarray:
    axis = _vector3(value, label="axis_world")
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        raise ValueError("axis_world cannot have zero length")
    return axis / norm


def _quaternion_to_matrix_wxyz(value) -> np.ndarray:
    w, x, y, z = _quaternion_wxyz(value, label="orientation_wxyz")
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def decompose_axis_motion(
    *,
    start_position_m,
    actual_position_m,
    axis_world,
) -> tuple[float, float]:
    """Return signed axial depth and unsigned lateral drift from a frozen axis."""

    start = _vector3(start_position_m, label="start_position_m")
    actual = _vector3(actual_position_m, label="actual_position_m")
    axis = _normalized_axis(axis_world)
    displacement = actual - start
    axial_depth_m = float(np.dot(displacement, axis))
    lateral_vector = displacement - axial_depth_m * axis
    lateral_drift_m = float(np.linalg.norm(lateral_vector))
    return axial_depth_m, lateral_drift_m


def quaternion_angular_error_deg(reference_wxyz, actual_wxyz) -> float:
    """Return the shortest full-orientation angular distance in degrees."""

    reference = _quaternion_wxyz(
        reference_wxyz,
        label="reference_orientation_wxyz",
    )
    actual = _quaternion_wxyz(
        actual_wxyz,
        label="actual_orientation_wxyz",
    )
    dot = float(np.clip(abs(np.dot(reference, actual)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


class InsertionPhase(str, Enum):
    WAITING_FOR_ALIGNMENT = "waiting_for_alignment"
    READY = "ready"
    ADVANCING = "advancing"
    COMPLETE = "complete"
    ABORTED = "aborted"


class InsertionStage(str, Enum):
    COARSE_APPROACH = "coarse_approach"
    FINE_INSERTION = "fine_insertion"


@dataclass(frozen=True)
class InsertionLimits:
    total_depth_m: float
    step_size_m: float
    settle_tolerance_m: float
    required_settled_frames: int
    step_timeout_frames: int
    max_lateral_drift_m: float
    max_orientation_error_deg: float
    max_mount_tip_error_m: float
    max_mount_axis_error_deg: float
    coarse_approach_depth_m: float = 0.0
    coarse_step_size_m: float = 0.0
    opening_depth_m: float = 0.0

    def __post_init__(self) -> None:
        positive_float_fields = (
            "total_depth_m",
            "step_size_m",
            "settle_tolerance_m",
            "max_lateral_drift_m",
            "max_orientation_error_deg",
            "max_mount_tip_error_m",
            "max_mount_axis_error_deg",
        )
        for name in positive_float_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        coarse_depth = float(self.coarse_approach_depth_m)
        coarse_step = float(self.coarse_step_size_m)
        opening_depth = float(self.opening_depth_m)
        for name, value in (
            ("coarse_approach_depth_m", coarse_depth),
            ("coarse_step_size_m", coarse_step),
            ("opening_depth_m", opening_depth),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

        if (coarse_depth <= _EPS) != (coarse_step <= _EPS):
            raise ValueError(
                "coarse_approach_depth_m and coarse_step_size_m must both be zero or positive"
            )
        if coarse_depth > self.total_depth_m + _EPS:
            raise ValueError("coarse_approach_depth_m cannot exceed total_depth_m")
        if coarse_step > coarse_depth + _EPS and coarse_depth > _EPS:
            raise ValueError("coarse_step_size_m cannot exceed coarse_approach_depth_m")
        if self.step_size_m > self.total_depth_m:
            raise ValueError("step_size_m cannot exceed total_depth_m")
        if opening_depth > self.total_depth_m + _EPS:
            raise ValueError("opening_depth_m cannot exceed total_depth_m")
        if self.required_settled_frames <= 0:
            raise ValueError("required_settled_frames must be positive")
        if self.step_timeout_frames <= 0:
            raise ValueError("step_timeout_frames must be positive")

    @property
    def total_step_count(self) -> int:
        coarse_depth = float(self.coarse_approach_depth_m)
        coarse_steps = 0
        if coarse_depth > _EPS:
            coarse_steps = int(
                math.ceil(coarse_depth / self.coarse_step_size_m - _EPS)
            )
        fine_distance = max(0.0, float(self.total_depth_m) - coarse_depth)
        fine_steps = int(math.ceil(fine_distance / self.step_size_m - _EPS))
        return coarse_steps + fine_steps


@dataclass(frozen=True)
class InsertionSample:
    frame_index: int
    alignment_complete: bool
    actual_position_m: np.ndarray
    actual_orientation_wxyz: np.ndarray
    target_error_m: float
    mount_tip_error_m: float
    mount_axis_error_deg: float
    fixed_joint_valid: bool
    attachment_preserved: bool

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        object.__setattr__(
            self,
            "actual_position_m",
            _vector3(self.actual_position_m, label="actual_position_m"),
        )
        object.__setattr__(
            self,
            "actual_orientation_wxyz",
            _quaternion_wxyz(
                self.actual_orientation_wxyz,
                label="actual_orientation_wxyz",
            ),
        )
        for name in (
            "target_error_m",
            "mount_tip_error_m",
            "mount_axis_error_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class InsertionCommand:
    step_index: int
    stage: InsertionStage
    commanded_depth_m: float
    commanded_port_depth_m: float
    target_position_m: np.ndarray
    target_orientation_wxyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_position_m",
            _vector3(self.target_position_m, label="target_position_m"),
        )
        object.__setattr__(
            self,
            "target_orientation_wxyz",
            _quaternion_wxyz(
                self.target_orientation_wxyz,
                label="target_orientation_wxyz",
            ),
        )


@dataclass(frozen=True)
class InsertionMetrics:
    stage: InsertionStage | None
    commanded_depth_m: float
    commanded_port_depth_m: float
    actual_axial_depth_m: float
    actual_port_depth_m: float
    lateral_drift_m: float
    target_error_m: float
    orientation_error_deg: float
    mount_tip_error_m: float
    mount_axis_error_deg: float
    settled_frame_count: int
    elapsed_step_frames: int


@dataclass(frozen=True)
class InsertionEvent:
    kind: str
    phase: InsertionPhase
    command: InsertionCommand | None
    metrics: InsertionMetrics | None
    reason: str | None = None
    settled_step_index: int | None = None


class PartialInsertionController:
    """Own exact frozen-axis targets and all insertion terminal transitions."""

    def __init__(self, limits: InsertionLimits):
        self.limits = limits
        self.phase = InsertionPhase.WAITING_FOR_ALIGNMENT
        self.frozen_start_position_m: np.ndarray | None = None
        self.frozen_orientation_wxyz: np.ndarray | None = None
        self.axis_world: np.ndarray | None = None
        self.commanded_step_index = 0
        self.commanded_depth_m = 0.0
        self.settled_frame_count = 0
        self.step_start_frame = 0
        self.last_command: InsertionCommand | None = None
        self.abort_reason: str | None = None

    def _freeze_from(self, sample: InsertionSample) -> None:
        self.frozen_start_position_m = sample.actual_position_m.copy()
        self.frozen_orientation_wxyz = sample.actual_orientation_wxyz.copy()
        self.axis_world = _normalized_axis(
            _quaternion_to_matrix_wxyz(
                self.frozen_orientation_wxyz
            )[:, 2]
        )
        self.phase = InsertionPhase.READY

    def _metrics(self, sample: InsertionSample) -> InsertionMetrics | None:
        if (
            self.frozen_start_position_m is None
            or self.frozen_orientation_wxyz is None
            or self.axis_world is None
        ):
            return None
        axial_depth_m, lateral_drift_m = decompose_axis_motion(
            start_position_m=self.frozen_start_position_m,
            actual_position_m=sample.actual_position_m,
            axis_world=self.axis_world,
        )
        stage = self.last_command.stage if self.last_command is not None else None
        return InsertionMetrics(
            stage=stage,
            commanded_depth_m=self.commanded_depth_m,
            commanded_port_depth_m=(
                self.commanded_depth_m - self.limits.opening_depth_m
            ),
            actual_axial_depth_m=axial_depth_m,
            actual_port_depth_m=(
                axial_depth_m - self.limits.opening_depth_m
            ),
            lateral_drift_m=lateral_drift_m,
            target_error_m=float(sample.target_error_m),
            orientation_error_deg=quaternion_angular_error_deg(
                self.frozen_orientation_wxyz,
                sample.actual_orientation_wxyz,
            ),
            mount_tip_error_m=float(sample.mount_tip_error_m),
            mount_axis_error_deg=float(sample.mount_axis_error_deg),
            settled_frame_count=self.settled_frame_count,
            elapsed_step_frames=max(
                0,
                int(sample.frame_index - self.step_start_frame),
            ),
        )

    def _next_depth_and_stage(self) -> tuple[float, InsertionStage]:
        coarse_depth = float(self.limits.coarse_approach_depth_m)
        if (
            coarse_depth > _EPS
            and self.commanded_depth_m < coarse_depth - _EPS
        ):
            return (
                min(
                    coarse_depth,
                    self.commanded_depth_m
                    + float(self.limits.coarse_step_size_m),
                ),
                InsertionStage.COARSE_APPROACH,
            )
        return (
            min(
                float(self.limits.total_depth_m),
                self.commanded_depth_m + float(self.limits.step_size_m),
            ),
            InsertionStage.FINE_INSERTION,
        )

    def _issue_next_command(self, frame_index: int) -> InsertionCommand:
        if (
            self.frozen_start_position_m is None
            or self.frozen_orientation_wxyz is None
            or self.axis_world is None
        ):
            raise RuntimeError("Insertion frame has not been frozen")
        next_step_index = self.commanded_step_index + 1
        depth_m, stage = self._next_depth_and_stage()
        if depth_m <= self.commanded_depth_m + _EPS:
            raise RuntimeError("Insertion command sequence made no progress")
        command = InsertionCommand(
            step_index=next_step_index,
            stage=stage,
            commanded_depth_m=float(depth_m),
            commanded_port_depth_m=float(
                depth_m - self.limits.opening_depth_m
            ),
            target_position_m=(
                self.frozen_start_position_m
                + self.axis_world * depth_m
            ),
            target_orientation_wxyz=self.frozen_orientation_wxyz,
        )
        self.commanded_step_index = next_step_index
        self.commanded_depth_m = float(depth_m)
        self.settled_frame_count = 0
        self.step_start_frame = int(frame_index)
        self.last_command = command
        self.phase = InsertionPhase.ADVANCING
        return command

    def _abort_reason(
        self,
        sample: InsertionSample,
        metrics: InsertionMetrics | None,
        *,
        include_start_tracking_error: bool,
    ) -> str | None:
        if not sample.fixed_joint_valid:
            return "cable fixed joint became invalid"
        if not sample.attachment_preserved:
            return "built-in deformable attachment was not preserved"
        if sample.mount_tip_error_m > self.limits.max_mount_tip_error_m:
            return (
                "plug-tip mount error exceeded limit: "
                f"{sample.mount_tip_error_m * 1000.0:.6f} mm"
            )
        if sample.mount_axis_error_deg > self.limits.max_mount_axis_error_deg:
            return (
                "plug-axis error exceeded limit: "
                f"{sample.mount_axis_error_deg:.6f} deg"
            )
        if include_start_tracking_error and (
            sample.target_error_m > self.limits.settle_tolerance_m
        ):
            return (
                "start ToolCenter tracking error exceeded limit: "
                f"{sample.target_error_m * 1000.0:.6f} mm"
            )
        if metrics is not None:
            if metrics.lateral_drift_m > self.limits.max_lateral_drift_m:
                return (
                    "lateral drift exceeded limit: "
                    f"{metrics.lateral_drift_m * 1000.0:.6f} mm"
                )
            if (
                metrics.orientation_error_deg
                > self.limits.max_orientation_error_deg
                and orientation_abort_is_due(
                    target_error_m=sample.target_error_m,
                    settle_tolerance_m=self.limits.settle_tolerance_m,
                    actual_port_depth_m=metrics.actual_port_depth_m,
                )
            ):
                return (
                    "orientation error exceeded limit: "
                    f"{metrics.orientation_error_deg:.6f} deg"
                )
        return None

    def abort(
        self,
        reason: str,
        sample: InsertionSample | None = None,
    ) -> InsertionEvent:
        if self.phase in (
            InsertionPhase.COMPLETE,
            InsertionPhase.ABORTED,
        ):
            return InsertionEvent(
                kind="holding",
                phase=self.phase,
                command=None,
                metrics=self._metrics(sample) if sample is not None else None,
                reason=self.abort_reason,
            )
        self.phase = InsertionPhase.ABORTED
        self.abort_reason = str(reason)
        return InsertionEvent(
            kind="aborted",
            phase=self.phase,
            command=None,
            metrics=self._metrics(sample) if sample is not None else None,
            reason=self.abort_reason,
        )

    def update(self, sample: InsertionSample) -> InsertionEvent:
        if self.phase in (
            InsertionPhase.COMPLETE,
            InsertionPhase.ABORTED,
        ):
            return InsertionEvent(
                kind="holding",
                phase=self.phase,
                command=None,
                metrics=self._metrics(sample),
                reason=self.abort_reason,
            )

        if self.phase is InsertionPhase.WAITING_FOR_ALIGNMENT:
            if not sample.alignment_complete:
                return InsertionEvent(
                    kind="waiting",
                    phase=self.phase,
                    command=None,
                    metrics=None,
                )
            reason = self._abort_reason(
                sample,
                metrics=None,
                include_start_tracking_error=True,
            )
            if reason is not None:
                return self.abort(reason, sample)
            self._freeze_from(sample)
            command = self._issue_next_command(sample.frame_index)
            return InsertionEvent(
                kind="started",
                phase=self.phase,
                command=command,
                metrics=self._metrics(sample),
            )

        metrics = self._metrics(sample)
        reason = self._abort_reason(
            sample,
            metrics=metrics,
            include_start_tracking_error=False,
        )
        if reason is not None:
            return self.abort(reason, sample)

        if sample.target_error_m <= self.limits.settle_tolerance_m:
            self.settled_frame_count += 1
        else:
            self.settled_frame_count = 0

        metrics = self._metrics(sample)
        if self.settled_frame_count < self.limits.required_settled_frames:
            if (
                sample.frame_index - self.step_start_frame
                >= self.limits.step_timeout_frames
            ):
                return self.abort(
                    "insertion step timeout before physical settle",
                    sample,
                )
            return InsertionEvent(
                kind="waiting_for_settle",
                phase=self.phase,
                command=None,
                metrics=metrics,
            )

        settled_step_index = self.commanded_step_index
        if self.commanded_depth_m >= self.limits.total_depth_m - _EPS:
            self.phase = InsertionPhase.COMPLETE
            return InsertionEvent(
                kind="complete",
                phase=self.phase,
                command=None,
                metrics=metrics,
                settled_step_index=settled_step_index,
            )

        command = self._issue_next_command(sample.frame_index)
        return InsertionEvent(
            kind="step_settled",
            phase=self.phase,
            command=command,
            metrics=metrics,
            settled_step_index=settled_step_index,
        )
