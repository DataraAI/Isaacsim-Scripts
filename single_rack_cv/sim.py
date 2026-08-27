#!/usr/bin/env python3
"""Isaac Sim scene, synchronized stereo RGB servo, and Lula IK."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import carb
import numpy as np
import omni.usd

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DomeLight, GroundPlane
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from isaacsim.storage.native import get_assets_root_path

from config import Config
from vision.perception import (
    CameraFrame,
    CameraModel,
    PortDetection,
    StereoFrame,
    StereoPortObservation,
    build_virtual_camera_model,
    compute_bounded_step,
    compute_desired_port_camera_usd,
    normalize_rgb,
)


def log(message: str) -> None:
    print(f"[SIM] {message}", flush=True)


def warn(message: str) -> None:
    full = f"[SIM] WARNING: {message}"
    print(full, flush=True)
    carb.log_warn(full)


def _normalize_quaternion_wxyz(
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Return one normalized scalar-first quaternion."""
    quaternion = np.asarray(
        quaternion_wxyz,
        dtype=np.float64,
    )

    if quaternion.shape != (4,):
        raise ValueError(
            f"Quaternion must have shape (4,), got {quaternion.shape}."
        )

    norm = float(np.linalg.norm(quaternion))

    if norm <= 1.0e-12:
        raise ValueError("Quaternion cannot have zero length.")

    return quaternion / norm


def _rotate_z(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a 3D vector about the world Z axis."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = vec
    return np.array([cos_a * x - sin_a * y, sin_a * x + cos_a * y, z])


def quaternion_wxyz_to_matrix(
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Convert a normalized WXYZ quaternion to a 3x3 rotation matrix."""
    w, x, y, z = _normalize_quaternion_wxyz(
        quaternion_wxyz
    )

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


def matrix_to_quaternion_wxyz(
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized WXYZ quaternion."""
    matrix = np.asarray(
        rotation_matrix,
        dtype=np.float64,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            f"Rotation matrix must have shape (3, 3), got {matrix.shape}."
        )

    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(
            1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
        ) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(
            1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
        ) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(
            1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
        ) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = _normalize_quaternion_wxyz(
        np.array(
            [w, x, y, z],
            dtype=np.float64,
        )
    )

    # q and -q represent the same rotation. Keep W nonnegative so logs and
    # round-trip tests remain stable.
    if quaternion[0] < 0.0:
        quaternion *= -1.0

    return quaternion


def hand_pose_to_tool_pose(
    hand_position_m: np.ndarray,
    hand_orientation_wxyz: np.ndarray,
    tool_local_position_m: np.ndarray,
    tool_local_orientation_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose world_T_hand with hand_T_tool."""
    hand_position = np.asarray(
        hand_position_m,
        dtype=np.float64,
    )
    tool_local_position = np.asarray(
        tool_local_position_m,
        dtype=np.float64,
    )

    if hand_position.shape != (3,) or tool_local_position.shape != (3,):
        raise ValueError("Hand and tool positions must both have shape (3,).")

    world_from_hand = quaternion_wxyz_to_matrix(
        hand_orientation_wxyz
    )
    hand_from_tool = quaternion_wxyz_to_matrix(
        tool_local_orientation_wxyz
    )

    tool_position = (
        hand_position
        + world_from_hand @ tool_local_position
    )
    world_from_tool = (
        world_from_hand
        @ hand_from_tool
    )
    tool_orientation = matrix_to_quaternion_wxyz(
        world_from_tool
    )

    return tool_position, tool_orientation


def tool_pose_to_hand_pose(
    tool_position_m: np.ndarray,
    tool_orientation_wxyz: np.ndarray,
    tool_local_position_m: np.ndarray,
    tool_local_orientation_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert hand_T_tool so Lula can solve the required panda_hand pose."""
    tool_position = np.asarray(
        tool_position_m,
        dtype=np.float64,
    )
    tool_local_position = np.asarray(
        tool_local_position_m,
        dtype=np.float64,
    )

    if tool_position.shape != (3,) or tool_local_position.shape != (3,):
        raise ValueError("Tool positions must both have shape (3,).")

    world_from_tool = quaternion_wxyz_to_matrix(
        tool_orientation_wxyz
    )
    hand_from_tool = quaternion_wxyz_to_matrix(
        tool_local_orientation_wxyz
    )

    world_from_hand = (
        world_from_tool
        @ hand_from_tool.T
    )
    hand_position = (
        tool_position
        - world_from_hand @ tool_local_position
    )
    hand_orientation = matrix_to_quaternion_wxyz(
        world_from_hand
    )

    return hand_position, hand_orientation


def update_convergence_counter(
    position_error_m: float,
    tolerance_m: float,
    current_count: int,
) -> int:
    """Count consecutive in-tolerance frames; reset immediately on a miss."""
    error = float(position_error_m)
    tolerance = float(tolerance_m)

    if not math.isfinite(error) or error < 0.0:
        raise ValueError("position_error_m must be finite and nonnegative.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance_m must be finite and positive.")
    if current_count < 0:
        raise ValueError("current_count must be nonnegative.")

    return current_count + 1 if error <= tolerance else 0


def target_is_settled(
    position_error_m: float,
    tolerance_m: float,
) -> bool:
    """Return whether a physical ToolCenter target is ready for the next step."""
    error = float(position_error_m)
    tolerance = float(tolerance_m)

    if not math.isfinite(error) or error < 0.0:
        raise ValueError("position_error_m must be finite and nonnegative.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance_m must be finite and positive.")

    return error <= tolerance


@dataclass
class VisualServoState:
    """Minimal state for RGB acquisition, tracking, and final settling."""

    startup_ready: bool = False
    startup_settled_frame_count: int = 0

    left_reference: PortDetection | None = None
    right_reference: PortDetection | None = None
    acquisition_features: list[np.ndarray] = field(default_factory=list)
    acquired: bool = False
    consecutive_misses: int = 0

    aligned_capture_count: int = 0
    visual_aligned: bool = False
    complete: bool = False

    settled_frame_count: int = 0
    settle_start_frame: int = 0
    settle_timeout_reported: bool = False


@dataclass
class IKRuntime:
    articulation: Articulation
    target: XFormPrim
    actual_tool: XFormPrim
    hand_path: str
    kinematics_solver: LulaKinematicsSolver
    articulation_solver: ArticulationKinematicsSolver

    last_warning_frame: int = -1_000_000


class SimulationRuntime:
    """Small public interface around the Isaac Sim-specific implementation."""

    def __init__(self, simulation_app, cfg: Config):
        self.app = simulation_app
        self.cfg = cfg

        # Structured run logger; injected by main.py after construction.
        # Optional so alternate entry points can run without one.
        self.run_logger = None

        self.frame_index = 0
        self.left_camera_path = ""
        self.right_camera_path = ""
        self.left_camera_sensor: CameraSensor | None = None
        self.right_camera_sensor: CameraSensor | None = None
        self.ik: IKRuntime | None = None

        self.visual_servo = VisualServoState()
        self.desired_port_virtual_camera_usd = (
            self._compute_desired_port_camera_usd()
        )

        self._build_scene()

    # ------------------------------------------------------------------
    # Public runtime API used by main.py
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self.app.is_running()

    def step(self) -> None:
        self.app.update()
        self.frame_index += 1

    def current_phase(self) -> str:
        """Coarse pipeline-stage label for structured frame logging.

        Combines the visual-servo progression (base class) with the
        two-stage insertion phase/stage (present on cable-mount subclasses)
        into a single string such as "port_detect", "coarse_approach",
        "fine_insertion", or "done". InsertionPhase/InsertionStage are str
        enums, so plain string comparisons are used to avoid importing them.
        """
        insertion = getattr(self, "partial_insertion", None)
        if insertion is not None and insertion.phase != "waiting_for_alignment":
            phase = insertion.phase
            if phase == "complete":
                return "done"
            if phase == "aborted":
                return "aborted"
            command = getattr(insertion, "last_command", None)
            if command is not None:
                return str(command.stage.value)
            return "coarse_approach"

        state = self.visual_servo
        if not state.startup_ready:
            return "startup"
        if not state.acquired:
            return "port_detect"
        if not state.visual_aligned:
            return "servoing"
        if not state.complete:
            return "aligning"
        return "insertion_pending"

    def capture_due(self) -> bool:
        """
        Capture only when the camera is physically stationary enough to use.

        This creates a simple stop-and-look controller: one RGB correction is
        issued, the arm settles to that target, then the next control image is
        captured. It prevents stale images from stacking unfinished commands.
        """
        cfg = self.cfg.visual_servo
        state = self.visual_servo

        if cfg.freeze_after_complete and state.complete:
            return False

        # Once the image target is locked, freeze perception and let the arm
        # settle onto the final fixed target without further visual jitter.
        if state.visual_aligned:
            return False

        if not state.startup_ready:
            self._update_startup_settle()
            return False

        if state.acquired:
            position_error_m = self._tool_target_position_error_m()

            if not target_is_settled(
                position_error_m,
                cfg.target_settle_tolerance_m,
            ):
                return False

        interval = self.cfg.camera.capture_every_sim_frames
        return self.frame_index > 0 and self.frame_index % interval == 0

    def _tool_target_position_error_m(self) -> float:
        """Measure actual ToolCenter-to-current-target position error."""
        if self.ik is None:
            return math.inf

        self._update_actual_tool_frame(self.ik)
        target_position, _ = self.ik.target.get_world_pose()
        actual_position, _ = self.ik.actual_tool.get_world_pose()

        target = np.asarray(target_position, dtype=np.float64)
        actual = np.asarray(actual_position, dtype=np.float64)

        return float(np.linalg.norm(actual - target))

    def _update_startup_settle(self) -> None:
        """Wait for a stationary eye-in-hand camera before RGB acquisition."""
        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if state.startup_ready or self.ik is None:
            return

        position_error_m = self._tool_target_position_error_m()

        state.startup_settled_frame_count = update_convergence_counter(
            position_error_m=position_error_m,
            tolerance_m=cfg.startup_settle_tolerance_m,
            current_count=state.startup_settled_frame_count,
        )

        if (
            state.startup_settled_frame_count
            < cfg.required_startup_settled_frames
        ):
            return

        state.startup_ready = True

        log(
            "RGB STEREO VISUAL SERVO STARTUP SETTLED\n"
            f"  ToolCenter error: {position_error_m * 1000.0:.3f} mm\n"
            f"  stable frames: {state.startup_settled_frame_count}/"
            f"{cfg.required_startup_settled_frames}\n"
            "  next action: begin synchronized stereo acquisition"
        )

    def visual_servo_references(
        self,
    ) -> tuple[PortDetection | None, PortDetection | None]:
        """Return both previous eye detections for continuity tracking."""
        state = self.visual_servo
        return state.left_reference, state.right_reference

    def note_perception_failure(self) -> None:
        """Hold position on a missed frame and reacquire after repeated misses."""
        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if state.complete:
            return

        state.consecutive_misses += 1
        state.aligned_capture_count = 0
        state.visual_aligned = False
        state.settled_frame_count = 0
        state.settle_timeout_reported = False

        if state.consecutive_misses < cfg.max_consecutive_misses:
            return

        had_track = (
            state.left_reference is not None
            or state.right_reference is not None
            or state.acquired
        )
        state.left_reference = None
        state.right_reference = None
        state.acquisition_features.clear()
        state.acquired = False
        state.consecutive_misses = 0

        if had_track:
            log("RGB stereo track lost; holding position and reacquiring both eyes.")

    def observe_visual_servo(
        self,
        observation: StereoPortObservation,
    ) -> None:
        """Apply one bounded translation from a valid two-eye observation."""
        cfg = self.cfg.visual_servo
        state = self.visual_servo

        if not cfg.enabled or state.complete:
            return
        if self.ik is None:
            raise RuntimeError("Visual servo requires initialized IK.")

        state.left_reference = observation.left.detection
        state.right_reference = observation.right.detection
        state.consecutive_misses = 0

        if not state.acquired:
            self._update_visual_acquisition(observation)
            if not state.acquired:
                return

        center_error_px = float(
            np.linalg.norm(observation.center_error_px)
        )
        range_error_m = abs(float(observation.range_error_m))

        visually_aligned = (
            center_error_px <= cfg.center_tolerance_px
            and range_error_m <= cfg.range_tolerance_m
        )

        if visually_aligned:
            state.aligned_capture_count += 1

            if (
                state.aligned_capture_count
                >= cfg.required_aligned_captures
                and not state.visual_aligned
            ):
                state.visual_aligned = True
                state.settled_frame_count = 0
                state.settle_start_frame = self.frame_index
                state.settle_timeout_reported = False

                log(
                    "RGB STEREO VISUAL ALIGNMENT LOCKED\n"
                    f"  center error: {center_error_px:.3f} px\n"
                    f"  range error: {range_error_m * 1000.0:.3f} mm\n"
                    f"  aligned captures: "
                    f"{state.aligned_capture_count}/"
                    f"{cfg.required_aligned_captures}"
                )
            return

        state.aligned_capture_count = 0
        state.visual_aligned = False
        state.settled_frame_count = 0
        state.settle_timeout_reported = False

        self._update_actual_tool_frame(self.ik)
        target_position, target_orientation = self.ik.target.get_world_pose()
        actual_position, _ = self.ik.actual_tool.get_world_pose()

        target_position = np.asarray(target_position, dtype=np.float64)
        actual_position = np.asarray(actual_position, dtype=np.float64)
        target_lead_m = float(
            np.linalg.norm(target_position - actual_position)
        )

        if not target_is_settled(
            target_lead_m,
            cfg.target_settle_tolerance_m,
        ):
            return

        step_world_m = compute_bounded_step(
            correction_world_m=observation.correction_world_m,
            gain=cfg.control_gain,
            max_step_m=cfg.max_target_step_m,
        )

        self.ik.target.set_world_pose(
            position=target_position + step_world_m,
            orientation=target_orientation,
        )

    def _update_visual_acquisition(
        self,
        observation: StereoPortObservation,
    ) -> None:
        """Require stable virtual-center pixels and stereo depth before motion."""
        state = self.visual_servo
        cfg = self.cfg.visual_servo

        feature = np.array(
            [
                observation.projected_virtual_center_uv[0],
                observation.projected_virtual_center_uv[1],
                observation.estimated_range_m,
            ],
            dtype=np.float64,
        )
        state.acquisition_features.append(feature)

        if len(state.acquisition_features) > cfg.required_acquisition_samples:
            state.acquisition_features.pop(0)

        count = len(state.acquisition_features)
        if count < cfg.required_acquisition_samples:
            log(
                "RGB stereo acquisition "
                f"{count}/{cfg.required_acquisition_samples}: "
                f"virtual_pixel=({feature[0]:.1f}, {feature[1]:.1f}), "
                f"range={feature[2] * 1000.0:.1f} mm"
            )
            return

        samples = np.vstack(state.acquisition_features)
        center_median = np.median(samples[:, :2], axis=0)
        center_spread_px = float(
            np.max(
                np.linalg.norm(
                    samples[:, :2] - center_median,
                    axis=1,
                )
            )
        )
        range_median = float(np.median(samples[:, 2]))
        range_spread_m = float(
            np.max(np.abs(samples[:, 2] - range_median))
        )

        stable = (
            center_spread_px <= cfg.max_acquisition_center_spread_px
            and range_spread_m <= cfg.range_tolerance_m
        )

        if not stable:
            log(
                "RGB stereo acquisition not stable: "
                f"center spread={center_spread_px:.2f} px, "
                f"range spread={range_spread_m * 1000.0:.2f} mm"
            )
            return

        state.acquired = True
        state.acquisition_features.clear()

        log(
            "RGB STEREO TRACK ACQUIRED\n"
            f"  virtual-center spread: {center_spread_px:.3f} px\n"
            f"  stereo range spread: {range_spread_m * 1000.0:.3f} mm\n"
            "  controller: virtual-center translation-only feedback"
        )

        if self.run_logger is not None:
            self.run_logger.log_event(
                t=self.frame_index,
                event="RGB_STEREO_TRACK_ACQUIRED",
                center_spread_px=center_spread_px,
                range_spread_mm=range_spread_m * 1000.0,
            )

    def update_visual_servo_completion(self) -> None:
        """Verify that the actual ToolCenter reaches the final visual target."""
        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if not state.visual_aligned or state.complete:
            return
        if self.ik is None:
            return

        self._update_actual_tool_frame(self.ik)
        target_position, _ = self.ik.target.get_world_pose()
        actual_position, _ = self.ik.actual_tool.get_world_pose()

        target_position = np.asarray(target_position, dtype=np.float64)
        actual_position = np.asarray(actual_position, dtype=np.float64)
        position_error_m = self._tool_target_position_error_m()

        state.settled_frame_count = update_convergence_counter(
            position_error_m=position_error_m,
            tolerance_m=cfg.settle_position_tolerance_m,
            current_count=state.settled_frame_count,
        )

        if state.settled_frame_count >= cfg.required_settled_frames:
            state.complete = True

            log(
                "RGB STEREO VISUAL SERVO COMPLETE\n"
                f"  final ToolCenter target: "
                f"{np.round(target_position, 6).tolist()}\n"
                f"  actual ToolCenter: "
                f"{np.round(actual_position, 6).tolist()}\n"
                f"  physical tracking error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  settled frames: "
                f"{state.settled_frame_count}/"
                f"{cfg.required_settled_frames}\n"
                "  next action: hold; no insertion is commanded."
            )
            return

        timeout_frames = max(
            1,
            int(
                round(
                    cfg.settle_warning_timeout_s
                    / self.cfg.scene.physics_dt
                )
            ),
        )

        if (
            self.frame_index - state.settle_start_frame
            >= timeout_frames
            and not state.settle_timeout_reported
        ):
            state.settle_timeout_reported = True
            warn(
                "RGB stereo alignment is stable, but ToolCenter has not settled "
                "within the warning timeout.\n"
                f"  current physical error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  required: "
                f"{cfg.settle_position_tolerance_m * 1000.0:.3f} mm"
            )

    def _compute_desired_port_camera_usd(self) -> np.ndarray:
        """Return the port point seen by the camera at the desired standoff."""
        camera_cfg = self.cfg.camera
        ik_cfg = self.cfg.ik
        servo_cfg = self.cfg.visual_servo

        y_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 1.0, 0.0),
            camera_cfg.local_y_rotation_deg,
        ).GetQuat()
        roll_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 0.0, 1.0),
            camera_cfg.local_roll_deg,
        ).GetQuat()
        camera_quat = y_quat * roll_quat
        camera_imag = camera_quat.GetImaginary()
        camera_orientation_wxyz = np.array(
            [
                camera_quat.GetReal(),
                camera_imag[0],
                camera_imag[1],
                camera_imag[2],
            ],
            dtype=np.float64,
        )

        hand_from_camera = quaternion_wxyz_to_matrix(
            camera_orientation_wxyz
        )
        hand_from_tool = quaternion_wxyz_to_matrix(
            np.asarray(
                ik_cfg.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            )
        )

        return compute_desired_port_camera_usd(
            camera_position_hand_m=np.asarray(
                camera_cfg.virtual_local_position,
                dtype=np.float64,
            ),
            hand_from_camera=hand_from_camera,
            tool_center_position_hand_m=np.asarray(
                ik_cfg.tool_center_local_position_m,
                dtype=np.float64,
            ),
            hand_from_tool=hand_from_tool,
            preinsert_standoff_m=servo_cfg.preinsert_standoff_m,
        )

    def update_ik(self) -> None:
        """
        Track a virtual tool-center target while Lula solves panda_hand.

        /World/IK_Target is the desired center-between-fingers pose.
        /World/ToolCenter is the actual center-between-fingers pose.
        """
        cfg = self.cfg.ik
        runtime = self.ik

        if runtime is None:
            return

        self._update_actual_tool_frame(runtime)

        if not cfg.tracking_enabled:
            return

        if self.frame_index % cfg.update_every_sim_frames != 0:
            return

        desired_tool_position, desired_tool_orientation = (
            runtime.target.get_world_pose()
        )

        hand_target_position, hand_target_orientation = (
            tool_pose_to_hand_pose(
                tool_position_m=desired_tool_position,
                tool_orientation_wxyz=desired_tool_orientation,
                tool_local_position_m=np.asarray(
                    cfg.tool_center_local_position_m,
                    dtype=np.float64,
                ),
                tool_local_orientation_wxyz=np.asarray(
                    cfg.tool_center_local_orientation_wxyz,
                    dtype=np.float64,
                ),
            )
        )

        base_position, base_orientation = (
            runtime.articulation.get_world_pose()
        )

        runtime.kinematics_solver.set_robot_base_pose(
            base_position,
            base_orientation,
        )

        action, success = (
            runtime.articulation_solver.compute_inverse_kinematics(
                target_position=hand_target_position,
                target_orientation=hand_target_orientation,
                position_tolerance=cfg.position_tolerance_m,
                orientation_tolerance=cfg.orientation_tolerance_rad,
            )
        )

        if success:
            runtime.articulation.apply_action(action)
            return

        if (
            self.frame_index - runtime.last_warning_frame
            >= cfg.warn_every_sim_frames
        ):
            runtime.last_warning_frame = self.frame_index

            warn(
                "Tool-center IK did not converge; no command was applied.\n"
                f"  desired tool: "
                f"{np.round(desired_tool_position, 4).tolist()}\n"
                f"  required hand: "
                f"{np.round(hand_target_position, 4).tolist()}"
            )

    def _update_actual_tool_frame(
        self,
        runtime: IKRuntime,
    ) -> None:
        """Move /World/ToolCenter to the tool pose achieved by the robot."""
        cfg = self.cfg.ik

        hand_position, hand_orientation = self._get_world_pose(
            runtime.hand_path
        )

        tool_position, tool_orientation = hand_pose_to_tool_pose(
            hand_position_m=hand_position,
            hand_orientation_wxyz=hand_orientation,
            tool_local_position_m=np.asarray(
                cfg.tool_center_local_position_m,
                dtype=np.float64,
            ),
            tool_local_orientation_wxyz=np.asarray(
                cfg.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            ),
        )

        runtime.actual_tool.set_world_pose(
            position=tool_position,
            orientation=tool_orientation,
        )

    def capture(self) -> StereoFrame:
        """Capture both physical eyes on the current simulation frame."""
        if self.left_camera_sensor is None:
            raise RuntimeError("Left CameraSensor is not initialized.")
        if self.right_camera_sensor is None:
            raise RuntimeError("Right CameraSensor is not initialized.")

        left_data, _ = self.left_camera_sensor.get_data("rgb")
        right_data, _ = self.right_camera_sensor.get_data("rgb")
        left_rgb = normalize_rgb(
            self._sensor_to_numpy(left_data, "left rgb"),
            self.cfg.camera.resolution,
        )
        right_rgb = normalize_rgb(
            self._sensor_to_numpy(right_data, "right rgb"),
            self.cfg.camera.resolution,
        )

        left_frame = CameraFrame(
            rgb=left_rgb,
            camera=self._camera_model(self.left_camera_path, left_rgb),
        )
        right_frame = CameraFrame(
            rgb=right_rgb,
            camera=self._camera_model(self.right_camera_path, right_rgb),
        )
        virtual_camera = build_virtual_camera_model(
            left_frame.camera,
            right_frame.camera,
        )
        return StereoFrame(
            left=left_frame,
            right=right_frame,
            virtual_camera=virtual_camera,
        )

    def _camera_model(
        self,
        camera_path: str,
        rgb: np.ndarray,
    ) -> CameraModel:
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        camera = UsdGeom.Camera(camera_prim)
        camera_world = UsdGeom.Xformable(
            camera_prim
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return CameraModel(
            image_height_px=rgb.shape[0],
            image_width_px=rgb.shape[1],
            focal_length_mm=float(camera.GetFocalLengthAttr().Get()),
            horizontal_aperture_mm=float(
                camera.GetHorizontalApertureAttr().Get()
            ),
            vertical_aperture_mm=float(
                camera.GetVerticalApertureAttr().Get()
            ),
            world_from_camera=np.asarray(camera_world, dtype=np.float64),
        )

    def stop(self) -> None:
        try:
            app_utils.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        scene = self.cfg.scene

        if not os.path.isfile(scene.rack_usd_path):
            raise FileNotFoundError(
                f"Rack USD not found: {scene.rack_usd_path}"
            )

        log("Creating stage")
        if not self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
            omni.usd.get_context().new_stage()
        self._update_app(5)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not create a valid stage.")

        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        GroundPlane("/World/GroundPlane")
        light = DomeLight("/World/DomeLight")
        light.set_intensities(scene.light_intensity)

        self._define_xform(
            scene.rack_path,
            position=(0.0, 0.0, 0.0),
            yaw_deg=scene.rack_yaw_deg,
            scale=(scene.rack_scale,) * 3,
        )
        self._add_reference(scene.rack_usd_path, scene.rack_asset_path)
        self._center_rack()

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Could not resolve Isaac Sim assets root.")

        franka_usd = (
            assets_root
            + "/Isaac/Robots/FrankaRobotics/"
            "FrankaPanda/franka.usd"
        )

        self._define_xform(
            scene.franka_path,
            position=scene.franka_position,
            yaw_deg=scene.franka_yaw_deg,
            scale=(1.0, 1.0, 1.0),
        )
        self._add_reference(franka_usd, scene.franka_asset_path)
        self._configure_franka_gravity()
        self._configure_franka_arm_drives()

        (
            self.left_camera_path,
            left_rtx_camera,
        ) = self._create_hand_camera(
            self.cfg.camera.left_camera_name,
            self.cfg.camera.left_local_position,
            "left",
        )
        (
            self.right_camera_path,
            right_rtx_camera,
        ) = self._create_hand_camera(
            self.cfg.camera.right_camera_name,
            self.cfg.camera.right_local_position,
            "right",
        )
        self.left_camera_sensor = CameraSensor(
            left_rtx_camera,
            resolution=self.cfg.camera.resolution,
            annotators=["rgb"],
        )
        self.right_camera_sensor = CameraSensor(
            right_rtx_camera,
            resolution=self.cfg.camera.resolution,
            annotators=["rgb"],
        )
        self.cfg.camera.output_dir.mkdir(parents=True, exist_ok=True)

        SimulationManager.setup_simulation(
            dt=scene.physics_dt,
            device=scene.device,
        )
        physics_scenes = SimulationManager.get_physics_scenes()
        if not physics_scenes:
            raise RuntimeError("No physics scene was created.")
        physics_scenes[0].set_enabled_gpu_dynamics(False)

        app_utils.play()
        app_utils.update_app(steps=30)

        self.ik = self._create_ik(assets_root)
        self._set_external_view()

        log(
            "READY\n"
            f"  rack:       {scene.rack_usd_path}\n"
            f"  Franka:     pos={scene.franka_position}, "
            f"yaw={scene.franka_yaw_deg}°\n"
            f"  left eye:   {self.left_camera_path}\n"
            f"  right eye:  {self.right_camera_path}\n"
            f"  sensors:    synchronized RGB pair at "
            f"{self.cfg.camera.tick_rate_hz:.1f} Hz\n"
            "  baseline:   40.0 mm; no physical center camera\n"
            f"  tool target:{self.cfg.ik.target_path}\n"
            f"  actual tool:{self.cfg.ik.actual_tool_path}\n"
            f"  visual servo: "
            f"{self.cfg.visual_servo.max_target_step_m * 1000.0:.1f} "
            "mm max step, 50 mm pre-insert standoff\n"
            f"  desired port in virtual center eye: "
            f"{np.round(self.desired_port_virtual_camera_usd, 5).tolist()}"
        )

    def _create_hand_camera(
        self,
        camera_name: str,
        local_position: tuple[float, float, float],
        eye_label: str,
    ) -> tuple[str, RtxCamera]:
        camera_cfg = self.cfg.camera
        hand_path = self._find_unique_descendant(
            self.cfg.scene.franka_asset_path,
            camera_cfg.hand_link_name,
        )
        camera_path = f"{hand_path}/{camera_name}"

        y_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 1.0, 0.0),
            camera_cfg.local_y_rotation_deg,
        ).GetQuat()
        roll_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 0.0, 1.0),
            camera_cfg.local_roll_deg,
        ).GetQuat()
        local_quat = y_quat * roll_quat
        imag = local_quat.GetImaginary()

        rtx_camera = RtxCamera(
            path=camera_path,
            translations=np.asarray(
                local_position,
                dtype=np.float64,
            ),
            orientations=np.asarray(
                [
                    local_quat.GetReal(),
                    imag[0],
                    imag[1],
                    imag[2],
                ],
                dtype=np.float64,
            ),
            tick_rate=camera_cfg.tick_rate_hz,
        )
        self._update_app(5)

        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)

        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Invalid RTX camera prim: {camera_path}")

        schemas = camera_prim.GetAppliedSchemas()
        if not any("OmniSensorAPI" in schema for schema in schemas):
            raise RuntimeError("RtxCamera did not apply OmniSensorAPI.")

        camera = UsdGeom.Camera(camera_prim)
        camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
        camera.CreateFocalLengthAttr().Set(camera_cfg.focal_length_mm)
        camera.CreateHorizontalApertureAttr().Set(
            camera_cfg.horizontal_aperture_mm
        )
        camera.CreateVerticalApertureAttr().Set(
            camera_cfg.vertical_aperture_mm
        )
        camera.CreateClippingRangeAttr().Set(
            Gf.Vec2f(*camera_cfg.clipping_range_m)
        )
        camera.CreateFocusDistanceAttr().Set(
            camera_cfg.focus_distance_m
        )
        camera.CreateFStopAttr().Set(0.0)

        log(
            f"{eye_label.capitalize()} RGB eye created: "
            f"offset={local_position}, "
            f"Y={camera_cfg.local_y_rotation_deg}°, "
            f"roll={camera_cfg.local_roll_deg}°"
        )
        return camera_path, rtx_camera

    def _configure_franka_gravity(self) -> None:
        """
        Disable gravity only on the Franka's rigid links.

        This is the smallest simulation-only way to remove the measured
        gravity-induced steady joint bias. The IK target stays unchanged,
        there is no hidden Cartesian offset, and joints are not teleported.
        """
        cfg = self.cfg.drive_tuning

        if not cfg.disable_gravity_on_franka:
            log("Franka link gravity left enabled.")
            return

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            raise RuntimeError(
                "Cannot configure Franka gravity without a valid USD stage."
            )

        root = stage.GetPrimAtPath(
            self.cfg.scene.franka_asset_path
        )

        if not root.IsValid():
            raise RuntimeError(
                "Franka asset root is invalid: "
                f"{self.cfg.scene.franka_asset_path}"
            )

        changed_paths: list[str] = []

        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue

            physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_body.CreateDisableGravityAttr().Set(True)
            changed_paths.append(str(prim.GetPath()))

        if not changed_paths:
            raise RuntimeError(
                "No Franka rigid bodies were found below "
                f"{self.cfg.scene.franka_asset_path}"
            )

        log(
            "FRANKA SIM ACCURACY MODE\n"
            f"  gravity disabled on: {len(changed_paths)} rigid links\n"
            "  scene/object gravity: unchanged\n"
            "  hidden target offset: none\n"
            "  joint teleporting: none"
        )

    def _configure_franka_arm_drives(self) -> None:
        """
        Scale the existing Franka arm drive gains at their source.

        This does not alter IK targets, joint commands, tool transforms, or
        add Cartesian compensation. It only makes the seven existing arm
        position drives track their commanded joint angles more tightly.
        """
        cfg = self.cfg.drive_tuning

        if not cfg.enabled:
            log("Franka arm drive tuning disabled.")
            return

        if (
            not math.isfinite(cfg.stiffness_multiplier)
            or cfg.stiffness_multiplier <= 0.0
        ):
            raise ValueError(
                "stiffness_multiplier must be finite and positive."
            )

        if (
            not math.isfinite(cfg.damping_multiplier)
            or cfg.damping_multiplier <= 0.0
        ):
            raise ValueError(
                "damping_multiplier must be finite and positive."
            )

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            raise RuntimeError(
                "Cannot tune Franka drives without a valid USD stage."
            )

        root = stage.GetPrimAtPath(
            self.cfg.scene.franka_asset_path
        )

        if not root.IsValid():
            raise RuntimeError(
                "Franka asset root is invalid: "
                f"{self.cfg.scene.franka_asset_path}"
            )

        requested_names = set(cfg.arm_joint_names)
        found: dict[str, Usd.Prim] = {}

        for prim in Usd.PrimRange(root):
            name = prim.GetName()

            if name in requested_names:
                found[name] = prim

        missing = [
            name
            for name in cfg.arm_joint_names
            if name not in found
        ]

        if missing:
            discovered = sorted(
                prim.GetName()
                for prim in Usd.PrimRange(root)
                if "joint" in prim.GetName().lower()
            )

            raise RuntimeError(
                "Could not find all Franka arm joint prims. "
                f"Missing={missing}; discovered={discovered}"
            )

        report_lines = [
            "FRANKA ARM DRIVE TUNING",
            f"  stiffness multiplier: "
            f"{cfg.stiffness_multiplier:.3f}x",
            f"  damping multiplier: "
            f"{cfg.damping_multiplier:.3f}x",
        ]

        for joint_name in cfg.arm_joint_names:
            joint_prim = found[joint_name]

            if not joint_prim.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(
                    f"{joint_prim.GetPath()} is not a revolute joint."
                )

            # Read the existing angular-drive attributes directly from
            # the composed USD prim. This avoids creating a new drive or
            # depending on a controller-side gain override.
            stiffness_attr = joint_prim.GetAttribute(
                "drive:angular:physics:stiffness"
            )
            damping_attr = joint_prim.GetAttribute(
                "drive:angular:physics:damping"
            )
            max_force_attr = joint_prim.GetAttribute(
                "drive:angular:physics:maxForce"
            )

            if (
                not stiffness_attr.IsValid()
                or not damping_attr.IsValid()
            ):
                raise RuntimeError(
                    "Missing existing angular-drive gain attributes on "
                    f"{joint_prim.GetPath()}"
                )

            old_stiffness = stiffness_attr.Get()
            old_damping = damping_attr.Get()
            max_force = (
                max_force_attr.Get()
                if max_force_attr.IsValid()
                else "not authored"
            )

            if old_stiffness is None or old_damping is None:
                raise RuntimeError(
                    "Drive gains are missing on "
                    f"{joint_prim.GetPath()}"
                )

            old_stiffness = float(old_stiffness)
            old_damping = float(old_damping)

            if (
                not math.isfinite(old_stiffness)
                or old_stiffness <= 0.0
            ):
                raise RuntimeError(
                    f"Invalid existing stiffness on {joint_name}: "
                    f"{old_stiffness}"
                )

            if (
                not math.isfinite(old_damping)
                or old_damping < 0.0
            ):
                raise RuntimeError(
                    f"Invalid existing damping on {joint_name}: "
                    f"{old_damping}"
                )

            new_stiffness = (
                old_stiffness
                * cfg.stiffness_multiplier
            )
            new_damping = (
                old_damping
                * cfg.damping_multiplier
            )

            stiffness_attr.Set(new_stiffness)
            damping_attr.Set(new_damping)

            report_lines.append(
                f"  {joint_name}: "
                f"Kp {old_stiffness:.3f} -> "
                f"{new_stiffness:.3f}, "
                f"Kd {old_damping:.3f} -> "
                f"{new_damping:.3f}, "
                f"maxForce unchanged={max_force}"
            )

        log("\n".join(report_lines))

    def _create_ik(self, assets_root: str) -> IKRuntime:
        """
        Create a tool-center target backed by a panda_hand Lula solver.

        The fixed startup values remain a panda_hand pose. They are converted
        to a tool-center pose before creating /World/IK_Target, so changing
        target semantics does not move the robot at startup.
        """
        cfg = self.cfg.ik

        articulation = Articulation(
            self.cfg.scene.franka_asset_path,
            name="franka_ik_articulation",
        )
        articulation.initialize()

        if not articulation.handles_initialized:
            raise RuntimeError(
                "Franka articulation handles did not initialize."
            )

        lula_config = (
            interface_config_loader
            .load_supported_lula_kinematics_solver_config(
                "Franka"
            )
        )
        kinematics = LulaKinematicsSolver(
            **lula_config
        )

        valid_frames = (
            kinematics.get_all_frame_names()
        )

        if cfg.end_effector_frame not in valid_frames:
            raise RuntimeError(
                f"Lula frame not found: {cfg.end_effector_frame}"
            )

        related_frames = [
            frame
            for frame in valid_frames
            if any(
                token in frame.lower()
                for token in (
                    "hand",
                    "finger",
                    "grasp",
                    "tool",
                )
            )
        ]

        solver = ArticulationKinematicsSolver(
            articulation,
            kinematics,
            cfg.end_effector_frame,
        )

        hand_path = self._find_unique_descendant(
            self.cfg.scene.franka_asset_path,
            cfg.end_effector_frame,
        )

        if cfg.use_fixed_start_pose:
            hand_position = np.asarray(
                cfg.initial_position,
                dtype=np.float64,
            )
            hand_orientation = _normalize_quaternion_wxyz(
                np.asarray(
                    cfg.initial_orientation_wxyz,
                    dtype=np.float64,
                )
            )
        else:
            hand_position, hand_orientation = (
                self._get_world_pose(
                    hand_path
                )
            )

        tool_position, tool_orientation = hand_pose_to_tool_pose(
            hand_position_m=hand_position,
            hand_orientation_wxyz=hand_orientation,
            tool_local_position_m=np.asarray(
                cfg.tool_center_local_position_m,
                dtype=np.float64,
            ),
            tool_local_orientation_wxyz=np.asarray(
                cfg.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            ),
        )

        frame_asset_path = (
            assets_root
            + "/Isaac/Props/UIElements/frame_prim.usd"
        )

        stage_utils.add_reference_to_stage(
            usd_path=frame_asset_path,
            path=cfg.target_path,
        )

        stage_utils.add_reference_to_stage(
            usd_path=frame_asset_path,
            path=cfg.actual_tool_path,
        )

        self._update_app(5)

        target = XFormPrim(
            prim_path=cfg.target_path,
            name=cfg.target_name,
            position=tool_position,
            orientation=tool_orientation,
            scale=np.full(
                3,
                cfg.target_scale,
                dtype=np.float64,
            ),
            visible=cfg.target_visible,
        )
        target.initialize()

        actual_tool = XFormPrim(
            prim_path=cfg.actual_tool_path,
            name=cfg.actual_tool_name,
            position=tool_position,
            orientation=tool_orientation,
            scale=np.full(
                3,
                cfg.actual_tool_scale,
                dtype=np.float64,
            ),
            visible=cfg.actual_tool_visible,
        )
        actual_tool.initialize()

        if cfg.select_target_on_start:
            try:
                (
                    omni.usd.get_context()
                    .get_selection()
                    .set_selected_prim_paths(
                        [cfg.target_path],
                        True,
                    )
                )
            except Exception:
                pass

        base_position, base_orientation = (
            articulation.get_world_pose()
        )

        kinematics.set_robot_base_pose(
            base_position,
            base_orientation,
        )

        log(
            "TOOL-CENTER IK READY\n"
            f"  Lula solver frame:   {cfg.end_effector_frame}\n"
            f"  commanded frame:     tool_center\n"
            f"  hand -> tool offset: "
            f"{cfg.tool_center_local_position_m} m\n"
            f"  desired tool target: {cfg.target_path}\n"
            f"  actual tool frame:   {cfg.actual_tool_path}\n"
            f"  initial hand pose:   "
            f"{np.round(hand_position, 4).tolist()}\n"
            f"  initial tool pose:   "
            f"{np.round(tool_position, 4).tolist()}\n"
            f"  related Lula frames: {related_frames}"
        )

        return IKRuntime(
            articulation=articulation,
            target=target,
            actual_tool=actual_tool,
            hand_path=hand_path,
            kinematics_solver=kinematics,
            articulation_solver=solver,
        )

    # ------------------------------------------------------------------
    # Small USD helpers
    # ------------------------------------------------------------------

    def _define_xform(
        self,
        path: str,
        position: tuple[float, float, float],
        yaw_deg: float,
        scale: tuple[float, float, float],
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim(path, "Xform")
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(Gf.Vec3d(*position))

        yaw = math.radians(yaw_deg)
        xform.AddOrientOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(
            Gf.Quatd(
                math.cos(yaw / 2.0),
                Gf.Vec3d(0.0, 0.0, math.sin(yaw / 2.0)),
            )
        )
        xform.AddScaleOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(Gf.Vec3d(*scale))

    def _add_reference(self, usd_path: str, prim_path: str) -> None:
        stage_utils.add_reference_to_stage(
            usd_path=usd_path,
            path=prim_path,
        )
        self._update_app(15)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            raise RuntimeError(f"Reference failed at {prim_path}: {usd_path}")

        descendants = sum(1 for _ in Usd.PrimRange(prim)) - 1
        if descendants <= 0:
            raise RuntimeError(f"Reference has no descendants: {prim_path}")

    @staticmethod
    def _set_xform_translate_orient(
        xform: UsdGeom.Xformable,
        *,
        translation: np.ndarray,
        yaw_rad: float,
    ) -> None:
        """Set translate+orient ops on an xformable (Quatf or Quatd)."""

        quat_d = Gf.Quatd(
            math.cos(yaw_rad / 2.0),
            Gf.Vec3d(0.0, 0.0, math.sin(yaw_rad / 2.0)),
        )
        quat_f = Gf.Quatf(
            float(quat_d.GetReal()),
            Gf.Vec3f(
                float(quat_d.GetImaginary()[0]),
                float(quat_d.GetImaginary()[1]),
                float(quat_d.GetImaginary()[2]),
            ),
        )
        wrote_orient = False
        wrote_translate = False
        for op in xform.GetOrderedXformOps():
            if (
                not wrote_orient
                and op.GetOpType() == UsdGeom.XformOp.TypeOrient
            ):
                current = op.Get()
                if isinstance(current, Gf.Quatf):
                    op.Set(quat_f)
                else:
                    op.Set(quat_d)
                wrote_orient = True
            elif (
                not wrote_translate
                and op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ):
                current = op.Get()
                if isinstance(current, Gf.Vec3f):
                    op.Set(
                        Gf.Vec3f(
                            float(translation[0]),
                            float(translation[1]),
                            float(translation[2]),
                        )
                    )
                else:
                    op.Set(Gf.Vec3d(*translation.tolist()))
                wrote_translate = True
        if not wrote_translate:
            raise RuntimeError("Xformable has no translation op.")

    def _center_rack(self) -> None:
        scene = self.cfg.scene

        # Let the freshly added reference finish loading/resolving its
        # transform before we read it; without this the pose comes back at
        # exactly (0, 0, 0) because the reference has not populated yet.
        self._update_app(45)

        # Read the Asset prim's actual current world position (its local
        # origin in world space) and cancel it so that origin lands at
        # (0, 0, 0), matching where the rack sat on main with the old asset.
        # ALWAYS read Rack_42U_01 / rack_asset_path — never DataHall — so
        # correction amounts stay identical to the pre-DataHall-apply path.
        current_world_position, _ = self._get_world_pose(
            scene.rack_asset_path
        )

        # Re-derive the rack correction relative to the robot's actual base
        # pose. rack_position_correction_m was tuned against a fixed
        # reference Franka pose; if the robot is placed elsewhere (e.g. the
        # merged pickup+insertion demo), rotate and translate the correction
        # so the rack keeps the same pose relative to the robot.
        robot_position = np.array(scene.franka_position, dtype=np.float64)
        robot_yaw_rad = math.radians(scene.franka_yaw_deg)

        reference_position = np.array(
            scene.reference_franka_position, dtype=np.float64
        )
        reference_yaw_rad = math.radians(scene.reference_franka_yaw_deg)

        correction_world = np.array(
            scene.rack_position_correction_m, dtype=np.float64
        )
        delta_yaw_rad = robot_yaw_rad - reference_yaw_rad
        relative_offset = correction_world - reference_position
        rotated_offset = _rotate_z(relative_offset, delta_yaw_rad)
        correction = robot_position + rotated_offset

        translation = -current_world_position + correction

        new_rack_yaw_deg = scene.rack_yaw_deg + math.degrees(delta_yaw_rad)
        new_rack_yaw_rad = math.radians(new_rack_yaw_deg)

        stage = omni.usd.get_context().get_stage()
        rack_prim = stage.GetPrimAtPath(scene.rack_path)
        rack_xform = UsdGeom.Xformable(rack_prim)

        datahall_path = "/World/DataHall"
        datahall_prim = stage.GetPrimAtPath(datahall_path)
        apply_on_datahall = bool(
            self.cfg.cable_mount.already_grasped_by_pickup_pipeline
            and datahall_prim.IsValid()
            and datahall_prim.IsA(UsdGeom.Xformable)
        )

        if not apply_on_datahall:
            # Standalone: author translate+orient on the rack container.
            self._set_xform_translate_orient(
                rack_xform,
                translation=translation,
                yaw_rad=new_rack_yaw_rad,
            )
            self._update_app(10)
            log(
                "Rack aligned: "
                f"current_world_position="
                f"{np.round(current_world_position, 4).tolist()} "
                f"correction={np.round(correction, 4).tolist()} "
                f"=> translation={np.round(translation, 4).tolist()} "
                f"robot_position={np.round(robot_position, 4).tolist()} "
                f"robot_yaw_deg={math.degrees(robot_yaw_rad):.3f} "
                f"delta_yaw_deg={math.degrees(delta_yaw_rad):.3f} "
                f"new_rack_yaw_deg={new_rack_yaw_deg:.3f}"
            )
            return

        # Merged DataHall asset: equipment siblings are NOT under Rack_42U_01.
        # Compute the same rack-local authoring as standalone to obtain the
        # desired rack WORLD pose, restore the rack, then apply the rigid
        # delta to /World/DataHall so rack+equipment move as one unit.
        saved_ops = [
            (op, op.Get()) for op in rack_xform.GetOrderedXformOps()
        ]
        W_rack_before = rack_xform.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rack_pos_before = np.asarray(
            W_rack_before.ExtractTranslation(), dtype=np.float64
        )

        self._set_xform_translate_orient(
            rack_xform,
            translation=translation,
            yaw_rad=new_rack_yaw_rad,
        )
        W_rack_desired = rack_xform.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rack_pos_desired = np.asarray(
            W_rack_desired.ExtractTranslation(), dtype=np.float64
        )

        for op, value in saved_ops:
            if value is not None:
                op.Set(value)

        equipment_paths = (
            "/World/DataHall/DGX_Servers/DGX_A100_01",
            "/World/DataHall/Blank_Panels/Blank_1U_BlackGold",
            "/World/DataHall/Network_Switches/AS4610_01",
            "/World/DataHall/Network_Switches/AS4610_Ethernet_Row_Top",
            "/World/DataHall/CPU_Servers/Server_1U_A_01",
            "/World/DataHall/Power/rPDU_A_01",
            "/World/DataHall/Patch/Fiber_Patch_Panel_1U_A_01",
        )
        equip_before = {}
        for path in equipment_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
                equip_before[path] = np.asarray(
                    UsdGeom.Xformable(prim)
                    .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    .ExtractTranslation(),
                    dtype=np.float64,
                )

        D = W_rack_desired * W_rack_before.GetInverse()
        datahall_xform = UsdGeom.Xformable(datahall_prim)
        W_datahall = datahall_xform.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        W_datahall_new = D * W_datahall

        parent = datahall_prim.GetParent()
        if parent.IsValid() and parent.IsA(UsdGeom.Xformable):
            W_parent = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
        else:
            W_parent = Gf.Matrix4d(1.0)
        local_datahall = W_parent.GetInverse() * W_datahall_new

        local_t = local_datahall.ExtractTranslation()
        local_q = local_datahall.ExtractRotationQuat()
        imag = local_q.GetImaginary()
        wrote_t = False
        wrote_o = False
        for op in datahall_xform.GetOrderedXformOps():
            if (
                not wrote_t
                and op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ):
                current = op.Get()
                if isinstance(current, Gf.Vec3f):
                    op.Set(
                        Gf.Vec3f(
                            float(local_t[0]),
                            float(local_t[1]),
                            float(local_t[2]),
                        )
                    )
                else:
                    op.Set(
                        Gf.Vec3d(
                            float(local_t[0]),
                            float(local_t[1]),
                            float(local_t[2]),
                        )
                    )
                wrote_t = True
            elif (
                not wrote_o
                and op.GetOpType() == UsdGeom.XformOp.TypeOrient
            ):
                current = op.Get()
                quat_d = Gf.Quatd(
                    float(local_q.GetReal()),
                    Gf.Vec3d(
                        float(imag[0]),
                        float(imag[1]),
                        float(imag[2]),
                    ),
                )
                if isinstance(current, Gf.Quatf):
                    op.Set(
                        Gf.Quatf(
                            float(quat_d.GetReal()),
                            Gf.Vec3f(
                                float(quat_d.GetImaginary()[0]),
                                float(quat_d.GetImaginary()[1]),
                                float(quat_d.GetImaginary()[2]),
                            ),
                        )
                    )
                else:
                    op.Set(quat_d)
                wrote_o = True
        if not wrote_t or not wrote_o:
            raise RuntimeError(
                "/World/DataHall is missing translate/orient xformOps "
                "required to apply the merged rack correction"
            )

        self._update_app(10)
        rack_pos_after = np.asarray(
            rack_xform.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            ).ExtractTranslation(),
            dtype=np.float64,
        )
        rack_delta = rack_pos_after - rack_pos_before
        pose_err = float(np.linalg.norm(rack_pos_after - rack_pos_desired))
        print(
            "[DEBUG] _center_rack DataHall-apply verification:\n"
            f"  apply_root={datahall_path}\n"
            f"  correction_translation(op amounts)="
            f"{np.round(translation, 4).tolist()}\n"
            f"  rack_world_before={np.round(rack_pos_before, 6).tolist()}\n"
            f"  rack_world_desired(old-authoring)="
            f"{np.round(rack_pos_desired, 6).tolist()}\n"
            f"  rack_world_after={np.round(rack_pos_after, 6).tolist()}\n"
            f"  rack_world_match_err={pose_err:.9e}\n"
            f"  rack_world_delta={np.round(rack_delta, 4).tolist()} "
            f"|delta|={np.linalg.norm(rack_delta):.4f}",
            flush=True,
        )
        if pose_err > 1.0e-6:
            raise RuntimeError(
                "DataHall-apply broke fix #6: Rack_42U_01 world position "
                f"differs from old rack-authoring target by {pose_err:.9e} m"
            )

        for path, before in equip_before.items():
            after = np.asarray(
                UsdGeom.Xformable(stage.GetPrimAtPath(path))
                .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                .ExtractTranslation(),
                dtype=np.float64,
            )
            delta = after - before
            print(
                f"[DEBUG] equipment move {path}: "
                f"delta={np.round(delta, 4).tolist()} "
                f"|delta|={np.linalg.norm(delta):.4f} "
                f"matches_rack={np.allclose(delta, rack_delta, atol=1e-4)}",
                flush=True,
            )

        log(
            "Rack aligned (via /World/DataHall): "
            f"current_world_position="
            f"{np.round(current_world_position, 4).tolist()} "
            f"correction={np.round(correction, 4).tolist()} "
            f"=> translation={np.round(translation, 4).tolist()} "
            f"robot_position={np.round(robot_position, 4).tolist()} "
            f"robot_yaw_deg={math.degrees(robot_yaw_rad):.3f} "
            f"delta_yaw_deg={math.degrees(delta_yaw_rad):.3f} "
            f"new_rack_yaw_deg={new_rack_yaw_deg:.3f} "
            f"rack_world_after={np.round(rack_pos_after, 4).tolist()}"
        )

    def _world_bounds(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
            useExtentsHint=True,
        )
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
        maximum = np.asarray(aligned.GetMax(), dtype=np.float64)

        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise RuntimeError(f"Invalid bounds for {path}")

        return minimum, maximum

    def _find_unique_descendant(self, root_path: str, name: str) -> str:
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_path)

        matches = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.GetName() == name
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one '{name}' below {root_path}, found {matches}"
            )

        return matches[0]

    def _get_world_pose(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        transform = UsdGeom.Xformable(
            prim
        ).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        position = np.asarray(
            transform.ExtractTranslation(),
            dtype=np.float64,
        )
        quat = transform.ExtractRotationQuat()
        imag = quat.GetImaginary()
        orientation = np.asarray(
            [quat.GetReal(), imag[0], imag[1], imag[2]],
            dtype=np.float64,
        )
        orientation /= np.linalg.norm(orientation)
        return position, orientation

    def _sensor_to_numpy(self, value, label: str) -> np.ndarray:
        if value is None:
            raise RuntimeError(f"{label} annotator returned None.")

        if hasattr(value, "numpy"):
            array = np.array(value.numpy(), copy=True)
        else:
            array = np.array(value, copy=True)

        if array.size == 0:
            raise RuntimeError(f"{label} annotator returned an empty array.")

        return array

    def _set_external_view(self) -> None:
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(
                eye=np.asarray(
                    self.cfg.scene.viewport_eye,
                    dtype=np.float64,
                ),
                target=np.asarray(
                    self.cfg.scene.viewport_target,
                    dtype=np.float64,
                ),
                camera_prim_path="/OmniverseKit_Persp",
            )
        except Exception as exc:
            warn(f"Could not set external viewport camera: {exc}")

    def _update_app(self, steps: int) -> None:
        for _ in range(steps):
            self.app.update()
