#!/usr/bin/env python3
"""Isaac Sim scene, synchronized stereo RGB servo, and Lula IK.

Import only after SimulationApp has started.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import carb
import numpy as np
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import DomeLight
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from isaacsim.storage.native import get_assets_root_path

from config import Config
from perception import (
    CameraFrame,
    CameraModel,
    CableDetection,
    StereoFrame,
    StereoCableObservation,
    build_virtual_camera_model,
    compute_bounded_step,
    compute_desired_cable_camera_usd,
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
class PreGraspState:
    """One-shot tip approach + finger open-around after servo hold."""

    phase: str = "idle"  # idle | descend | open | done
    cable_half_width_m: float | None = None
    cable_point_world_m: np.ndarray | None = None
    descend_settled_frames: int = 0
    open_frames: int = 0
    finger_half_gap_m: float = 0.0
    started: bool = False


@dataclass
class VisualServoState:
    """Minimal state for RGB acquisition, tracking, and final settling."""

    startup_ready: bool = False
    startup_settled_frame_count: int = 0
    last_startup_warn_frame: int = -1_000_000

    left_reference: CableDetection | None = None
    right_reference: CableDetection | None = None
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


class _LulaCpuArticulation:
    """
    Adapt SingleArticulation for Lula when PhysX runs on CUDA.

    Soft-cable GPU dynamics forces SimulationManager device=cuda, so
    get_joint_positions() returns cuda tensors. Lula's CCD IK only
    accepts list[np.ndarray]; without this bridge every IK update fails
    and the arm never moves to the start pose.
    """

    def __init__(self, articulation: Articulation):
        self._articulation = articulation

    def get_joint_positions(self, *args, **kwargs):
        positions = self._articulation.get_joint_positions(*args, **kwargs)
        return SimulationRuntime._as_cpu_numpy(positions).reshape(-1)

    def __getattr__(self, name):
        return getattr(self._articulation, name)


class SimulationRuntime:
    """Small public interface around the Isaac Sim-specific implementation."""

    def __init__(self, simulation_app, cfg: Config):
        self.app = simulation_app
        self.cfg = cfg

        self.frame_index = 0
        self.left_camera_path = ""
        self.right_camera_path = ""
        self.left_camera_sensor: CameraSensor | None = None
        self.right_camera_sensor: CameraSensor | None = None
        self.ik: IKRuntime | None = None

        self.visual_servo = VisualServoState()
        self.pre_grasp = PreGraspState()
        self._finger_dof_indices: list[int] = []
        self.desired_cable_virtual_camera_usd = (
            self._compute_desired_cable_camera_usd()
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
                self._target_settle_tolerance_m(),
            ):
                return False

        interval = self.cfg.camera.capture_every_sim_frames
        return self.frame_index > 0 and self.frame_index % interval == 0

    def _startup_settle_tolerance_m(self) -> float:
        cfg = self.cfg.visual_servo
        if self.cfg.scene.enable_gpu_dynamics:
            return float(cfg.gpu_startup_settle_tolerance_m)
        return float(cfg.startup_settle_tolerance_m)

    def _target_settle_tolerance_m(self) -> float:
        cfg = self.cfg.visual_servo
        if self.cfg.scene.enable_gpu_dynamics:
            return float(cfg.gpu_target_settle_tolerance_m)
        return float(cfg.target_settle_tolerance_m)

    def _tool_target_position_error_m(self) -> float:
        """Measure actual ToolCenter-to-current-target position error."""
        if self.ik is None:
            return math.inf

        self._update_actual_tool_frame(self.ik)
        target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
        actual_position, _ = self._tool_pose_from_hand()
        return float(np.linalg.norm(actual_position - target_position))

    def _tool_pose_from_hand(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Tool-center pose from articulation joints + Lula FK.

        With CUDA PhysX / GPU dynamics, UsdGeom hand xforms often stay at the
        Franka rest pose while the simulated arm moves. Joint-state FK matches
        the visual robot.
        """
        if self.ik is None:
            raise RuntimeError("IK runtime is not initialized.")
        cfg = self.cfg.ik
        hand_position, hand_orientation = self._hand_pose_from_articulation()
        return hand_pose_to_tool_pose(
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

    def _hand_pose_from_articulation(self) -> tuple[np.ndarray, np.ndarray]:
        """panda_hand pose from Lula FK on current articulation joints."""
        if self.ik is None:
            raise RuntimeError("IK runtime is not initialized.")

        base_position, base_orientation = self._world_pose_numpy(
            self.ik.articulation
        )
        self.ik.kinematics_solver.set_robot_base_pose(
            base_position,
            base_orientation,
        )

        translation, rotation = (
            self.ik.articulation_solver.compute_end_effector_pose()
        )
        if translation is None or rotation is None:
            raise RuntimeError(
                "Lula forward kinematics failed for panda_hand."
            )

        hand_position = np.asarray(
            translation,
            dtype=np.float64,
        ).reshape(3)
        hand_orientation = matrix_to_quaternion_wxyz(
            np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        )
        return hand_position, hand_orientation

    def _hand_camera_local_matrix(
        self,
        local_position: tuple[float, float, float],
    ) -> np.ndarray:
        """Hand←camera transform matching _create_hand_camera (row-vector 4x4)."""
        camera_cfg = self.cfg.camera
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
        orientation = _normalize_quaternion_wxyz(
            np.array(
                [
                    local_quat.GetReal(),
                    imag[0],
                    imag[1],
                    imag[2],
                ],
                dtype=np.float64,
            )
        )
        # Column rotation R maps hand←camera offsets; CameraModel / USD Gf
        # matrices use row vectors, so store R.T in the upper 3x3.
        rotation = quaternion_wxyz_to_matrix(orientation)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation.T
        matrix[3, :3] = np.asarray(local_position, dtype=np.float64)
        return matrix

    def _world_from_hand_matrix(self) -> np.ndarray:
        """Row-vector 4x4 world←hand from articulation FK."""
        position, orientation = self._hand_pose_from_articulation()
        rotation = quaternion_wxyz_to_matrix(orientation)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation.T
        matrix[3, :3] = position
        return matrix

    def _update_startup_settle(self) -> None:
        """Wait for a stationary eye-in-hand camera before RGB acquisition."""
        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if state.startup_ready or self.ik is None:
            return

        position_error_m = self._tool_target_position_error_m()
        settle_tol_m = self._startup_settle_tolerance_m()

        state.startup_settled_frame_count = update_convergence_counter(
            position_error_m=position_error_m,
            tolerance_m=settle_tol_m,
            current_count=state.startup_settled_frame_count,
        )

        if (
            state.startup_settled_frame_count
            < cfg.required_startup_settled_frames
        ):
            warn_every = max(
                1,
                int(
                    round(
                        cfg.startup_settle_warn_s
                        / self.cfg.scene.physics_dt
                    )
                ),
            )
            if (
                self.frame_index - state.last_startup_warn_frame
                >= warn_every
            ):
                state.last_startup_warn_frame = self.frame_index
                warn(
                    "Waiting for startup settle before stereo capture:\n"
                    f"  ToolCenter error: "
                    f"{position_error_m * 1000.0:.3f} mm "
                    f"(need <= {settle_tol_m * 1000.0:.3f} mm)\n"
                    f"  target={np.round(self._get_world_pose(self.cfg.ik.target_path)[0], 4).tolist()}\n"
                    f"  actual={np.round(self._tool_pose_from_hand()[0], 4).tolist()}\n"
                    f"  stable frames: "
                    f"{state.startup_settled_frame_count}/"
                    f"{cfg.required_startup_settled_frames}\n"
                    f"  gpu_dynamics={self.cfg.scene.enable_gpu_dynamics}"
                )
            return

        state.startup_ready = True

        log(
            "RGB STEREO VISUAL SERVO STARTUP SETTLED\n"
            f"  ToolCenter error: {position_error_m * 1000.0:.3f} mm\n"
            f"  settle tolerance: {settle_tol_m * 1000.0:.3f} mm\n"
            f"  stable frames: {state.startup_settled_frame_count}/"
            f"{cfg.required_startup_settled_frames}\n"
            "  next action: begin synchronized stereo acquisition"
        )

    def visual_servo_references(
        self,
    ) -> tuple[CableDetection | None, CableDetection | None]:
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
        observation: StereoCableObservation,
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

        self.pre_grasp.cable_half_width_m = 0.5 * float(observation.width_m)
        self.pre_grasp.cable_point_world_m = np.asarray(
            observation.center_world_xyz_m,
            dtype=np.float64,
        ).reshape(3)

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
        target_position, target_orientation = self._get_world_pose(
            self.cfg.ik.target_path
        )
        actual_position, _ = self._tool_pose_from_hand()

        target_lead_m = float(
            np.linalg.norm(target_position - actual_position)
        )

        if not target_is_settled(
            target_lead_m,
            self._target_settle_tolerance_m(),
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
        observation: StereoCableObservation,
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

    def update_visual_servo_completion(self) -> None:
        """Verify that the actual ToolCenter reaches the final visual target."""
        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if not state.visual_aligned or state.complete:
            return
        if self.ik is None:
            return

        self._update_actual_tool_frame(self.ik)
        target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
        actual_position, _ = self._tool_pose_from_hand()

        position_error_m = float(
            np.linalg.norm(actual_position - target_position)
        )

        state.settled_frame_count = update_convergence_counter(
            position_error_m=position_error_m,
            tolerance_m=cfg.settle_position_tolerance_m,
            current_count=state.settled_frame_count,
        )

        if state.settled_frame_count >= cfg.required_settled_frames:
            state.complete = True

            next_action = (
                "begin pre-grasp tip approach"
                if self.cfg.pre_grasp.enabled
                else "hold; no grasping is commanded"
            )
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
                f"  next action: {next_action}."
            )
            if self.cfg.pre_grasp.enabled:
                self._begin_pre_grasp_descend()
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

    def update_pre_grasp(self) -> None:
        """Descend tip toward cable while opening fingers around it (no close)."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp

        if not cfg.enabled or self.ik is None:
            return
        if state.phase in ("idle", "done"):
            if state.phase == "done":
                self._apply_finger_open(state.finger_half_gap_m)
            return

        # Open during descend so the fingers clear the cable as the tip drops.
        if state.phase in ("descend", "open"):
            self._apply_finger_open(state.finger_half_gap_m)

        if state.phase == "descend":
            position_error_m = self._tool_target_position_error_m()
            state.descend_settled_frames = update_convergence_counter(
                position_error_m=position_error_m,
                tolerance_m=self._target_settle_tolerance_m(),
                current_count=state.descend_settled_frames,
            )
            if state.descend_settled_frames < (
                self.cfg.visual_servo.required_settled_frames
            ):
                return

            state.phase = "open"
            state.open_frames = 0
            log(
                "PRE-GRASP OPEN HOLD (no close)\n"
                f"  tip settled; keeping fingers open at "
                f"{state.finger_half_gap_m * 1000.0:.1f} mm per side"
            )
            return

        if state.phase == "open":
            state.open_frames += 1
            if state.open_frames < cfg.open_hold_frames:
                return
            state.phase = "done"
            log(
                "PRE-GRASP DONE\n"
                f"  tip_clearance_m={cfg.tip_clearance_m:.4f}\n"
                f"  fingers held open at "
                f"{state.finger_half_gap_m * 1000.0:.1f} mm per side\n"
                "  next action: hold open; no close commanded."
            )

    def _compute_finger_half_gap_m(self) -> float:
        cfg = self.cfg.pre_grasp
        half_width = self.pre_grasp.cable_half_width_m
        if half_width is None or not math.isfinite(float(half_width)):
            half_width = cfg.fallback_cable_half_width_m
        return min(
            float(half_width) + cfg.side_allowance_m,
            cfg.finger_max_open_m,
        )

    def _begin_pre_grasp_descend(self) -> None:
        """Move IK_Target down the tool axis to the configured tip clearance."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if state.started or self.ik is None:
            return
        state.started = True
        state.finger_half_gap_m = self._compute_finger_half_gap_m()

        cable_point = state.cable_point_world_m
        if cable_point is None:
            plug_min, plug_max = self._world_bounds(
                self.cfg.scene.tracked_connector_path
            )
            cable_point = 0.5 * (plug_min + plug_max)
            state.cable_point_world_m = cable_point

        target_position, target_orientation = self._get_world_pose(
            self.cfg.ik.target_path
        )
        rotation = quaternion_wxyz_to_matrix(target_orientation)
        tool_z = rotation[:, 2]
        tool_z_norm = float(np.linalg.norm(tool_z))
        if tool_z_norm <= 1.0e-9:
            warn("Pre-grasp descend aborted: invalid tool Z axis.")
            state.phase = "done"
            return
        approach = tool_z / tool_z_norm

        # Cable lies along +tool Z from the tool center (grasp_standoff).
        desired_tool = np.asarray(cable_point, dtype=np.float64) - (
            cfg.tip_clearance_m * approach
        )
        desired_tool[0] = float(target_position[0])
        desired_tool[1] = float(target_position[1])

        block_top = self._cable_support_top_z_m()
        # Keep tool center above block top + tip clearance as a floor.
        min_z = block_top + cfg.tip_clearance_m + cfg.block_safety_margin_m
        if desired_tool[2] < min_z:
            desired_tool[2] = min_z

        self.ik.target.set_world_pose(
            position=desired_tool,
            orientation=target_orientation,
        )
        state.phase = "descend"
        state.descend_settled_frames = 0
        log(
            "PRE-GRASP DESCEND + OPEN\n"
            f"  cable_point={np.round(cable_point, 4).tolist()}\n"
            f"  tip_clearance_m={cfg.tip_clearance_m:.4f}\n"
            f"  new IK_Target={np.round(desired_tool, 4).tolist()}\n"
            f"  finger_half_gap_m={state.finger_half_gap_m:.4f}\n"
            f"  side_allowance_m={cfg.side_allowance_m:.4f}"
        )
        self._apply_finger_open(state.finger_half_gap_m)

    def _apply_finger_open(self, half_gap_m: float) -> None:
        """Command both finger joints to an open half-gap (no close)."""
        if self.ik is None or not self._finger_dof_indices:
            return
        gap = float(
            min(
                max(half_gap_m, 0.0),
                self.cfg.pre_grasp.finger_max_open_m,
            )
        )
        # CUDA PhysX backend: ArticulationAction joint_indices must be a
        # torch tensor (or list). A NumPy array hits resolve_indices().to()
        # and raises AttributeError.
        try:
            import torch

            device = getattr(self.ik.articulation, "_device", None)
            if device is None:
                device = (
                    "cuda:0"
                    if self.cfg.scene.enable_gpu_dynamics
                    else "cpu"
                )
            joint_positions = torch.tensor(
                [gap, gap],
                dtype=torch.float32,
                device=device,
            )
            joint_indices = torch.tensor(
                self._finger_dof_indices,
                dtype=torch.long,
                device=device,
            )
        except Exception:
            joint_positions = np.array([gap, gap], dtype=np.float64)
            joint_indices = list(self._finger_dof_indices)

        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_indices=joint_indices,
        )
        self.ik.articulation.apply_action(action)

    def _resolve_finger_dof_indices(self, articulation: Articulation) -> None:
        """Map configured finger joint names to articulation DOF indices."""
        names = list(articulation.dof_names)
        indices: list[int] = []
        for joint_name in self.cfg.pre_grasp.finger_joint_names:
            if joint_name not in names:
                warn(
                    f"Finger joint '{joint_name}' not found in articulation "
                    f"DOFs: {names}"
                )
                continue
            indices.append(int(names.index(joint_name)))
        self._finger_dof_indices = indices
        if len(indices) == 2:
            log(
                "Pre-grasp finger DOFs: "
                f"{list(self.cfg.pre_grasp.finger_joint_names)} -> {indices}"
            )

    def _compute_desired_cable_camera_usd(self) -> np.ndarray:
        """Return the cable point seen by the camera at the desired standoff."""
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

        return compute_desired_cable_camera_usd(
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
            grasp_standoff_m=servo_cfg.grasp_standoff_m,
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
            self._get_world_pose(cfg.target_path)
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

        base_position, base_orientation = self._world_pose_numpy(
            runtime.articulation
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
        tool_position, tool_orientation = self._tool_pose_from_hand()
        # Keep the debug marker in sync. Settle / IK math uses USD hand pose
        # directly and does not depend on this round-trip.
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
            camera=self._camera_model(
                self.left_camera_path,
                left_rgb,
                self.cfg.camera.left_local_position,
            ),
        )
        right_frame = CameraFrame(
            rgb=right_rgb,
            camera=self._camera_model(
                self.right_camera_path,
                right_rgb,
                self.cfg.camera.right_local_position,
            ),
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
        local_position: tuple[float, float, float],
    ) -> CameraModel:
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        camera = UsdGeom.Camera(camera_prim)

        # CUDA PhysX does not keep child USD xforms in sync with the moving
        # hand. Compose world←camera from articulation FK + the known local
        # eye offset so stereo geometry matches the rendered images.
        if (
            self.ik is not None
            and self.cfg.scene.enable_gpu_dynamics
        ):
            world_from_camera = (
                self._hand_camera_local_matrix(local_position)
                @ self._world_from_hand_matrix()
            )
        else:
            camera_world = UsdGeom.Xformable(
                camera_prim
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world_from_camera = np.asarray(
                camera_world,
                dtype=np.float64,
            )

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
            world_from_camera=world_from_camera,
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

        if not os.path.isfile(scene.cable_usd_path):
            raise FileNotFoundError(
                f"Cable USD not found: {scene.cable_usd_path}"
            )

        log("Creating stage")
        omni.usd.get_context().new_stage()
        self._update_app(5)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not create a valid stage.")

        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        self._create_plain_ground_plane(
            "/World/GroundPlane",
            scene.ground_plane_color,
        )
        light = DomeLight("/World/DomeLight")
        light.set_intensities(scene.light_intensity)

        if scene.cable_support_enabled:
            self._create_cable_support_block()
        self._load_cable()

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
        self._select_franka_variants()
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

        physics_device = scene.device
        if scene.enable_gpu_dynamics and physics_device == "cpu":
            physics_device = "cuda"
            log(
                "enable_gpu_dynamics=True with device=cpu; "
                "using cuda for soft-cable PhysX"
            )

        SimulationManager.setup_simulation(
            dt=scene.physics_dt,
            device=physics_device,
        )
        physics_scenes = SimulationManager.get_physics_scenes()
        if not physics_scenes:
            raise RuntimeError("No physics scene was created.")
        self._configure_gpu_dynamics(scene.enable_gpu_dynamics)

        # Reload after GPU dynamics is configured so deformable bodies /
        # attachments initialize against a valid GPU PhysX scene (same
        # pattern as detailedInsertion/cable/network_connector_pickup.py).
        if scene.enable_gpu_dynamics:
            self._load_cable()

        app_utils.play()
        settle_frames = (
            scene.deformable_settle_frames
            if scene.enable_gpu_dynamics
            else 30
        )
        app_utils.update_app(steps=settle_frames)

        self.ik = self._create_ik(assets_root)
        self._set_external_view()

        log(
            "READY\n"
            f"  cable:      {scene.cable_usd_path}\n"
            f"  connector:  {scene.tracked_connector_path}\n"
            f"  Franka:     pos={scene.franka_position}, "
            f"yaw={scene.franka_yaw_deg}°\n"
            f"  left eye:   {self.left_camera_path}\n"
            f"  right eye:  {self.right_camera_path}\n"
            f"  sensors:    synchronized RGB pair at "
            f"{self.cfg.camera.tick_rate_hz:.1f} Hz\n"
            "  baseline:   40.0 mm; no physical center camera\n"
            f"  physics:    device={physics_device}, "
            f"gpu_dynamics={scene.enable_gpu_dynamics}\n"
            f"  tool target:{self.cfg.ik.target_path}\n"
            f"  actual tool:{self.cfg.ik.actual_tool_path}\n"
            f"  visual servo: "
            f"{self.cfg.visual_servo.max_target_step_m * 1000.0:.1f} "
            "mm max step, "
            f"{self.cfg.visual_servo.grasp_standoff_m * 1000.0:.0f} "
            "mm pre-insert standoff\n"
            f"  desired cable in virtual center eye: "
            f"{np.round(self.desired_cable_virtual_camera_usd, 5).tolist()}"
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

    def _select_franka_variants(self) -> None:
        """
        Pick the gripper-finger and PhysX variants the pickup task expects.

        The stock franka.usd asset ships multiple variant sets (finger
        geometry, physics backend). The original pickup script explicitly
        selects "AlternateFinger" and a PhysX-flavored physics variant;
        without this, grasping later may use the wrong finger geometry.
        """
        scene = self.cfg.scene
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(scene.franka_asset_path)

        if not prim.IsValid():
            raise RuntimeError(
                f"Franka asset root is invalid: {scene.franka_asset_path}"
            )

        gripper_variant = prim.GetVariantSet("Gripper")
        if gripper_variant.IsValid():
            names = list(gripper_variant.GetVariantNames())
            if scene.franka_gripper_variant in names:
                gripper_variant.SetVariantSelection(
                    scene.franka_gripper_variant
                )
            else:
                warn(
                    "Franka gripper variant "
                    f"'{scene.franka_gripper_variant}' not found "
                    f"(available: {names}); leaving default."
                )

        physics_variant = prim.GetVariantSet("Physics")
        if physics_variant.IsValid():
            names = list(physics_variant.GetVariantNames())
            physx_name = next(
                (name for name in names if name.lower() == "physx"),
                None,
            )
            if physx_name is not None:
                physics_variant.SetVariantSelection(physx_name)
            elif names:
                physics_variant.SetVariantSelection(names[0])

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

        self._resolve_finger_dof_indices(articulation)

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
            _LulaCpuArticulation(articulation),
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

        base_position, base_orientation = self._world_pose_numpy(
            articulation
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

    def _create_plain_ground_plane(
        self,
        prim_path: str,
        color: tuple[float, float, float],
        half_extent_m: float = 25.0,
        thickness_m: float = 0.01,
    ) -> None:
        """
        Build a plain flat static collider instead of using the
        GroundPlane helper, whose default grid-textured material
        resisted being overridden (its binding apparently wins deeper
        in its own prim hierarchy than we could reach). Building our
        own from scratch sidesteps that entirely: the material is
        bound at creation time, not fighting an existing one.

        A thin flat Cube, scaled way out in X/Y, with its top face at
        world z=0 (matching what the rest of the code assumes "ground"
        means) and plain PhysX collision (no RigidBodyAPI, so PhysX
        treats it as static/immovable, same as an ordinary ground).
        """
        stage = omni.usd.get_context().get_stage()
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
        cube.CreateSizeAttr(1.0)

        prim = cube.GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(half_extent_m, half_extent_m, thickness_m / 2.0)
        )
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(0.0, 0.0, -thickness_m / 2.0)
        )

        UsdPhysics.CollisionAPI.Apply(prim)

        material = UsdShade.Material.Define(
            stage, Sdf.Path(f"{prim_path}/PlainMaterial")
        )
        shader = UsdShade.Shader.Define(
            stage, Sdf.Path(f"{prim_path}/PlainMaterial/Shader")
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(*color))
        shader.CreateInput(
            "roughness", Sdf.ValueTypeNames.Float
        ).Set(0.8)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

        log(
            f"Plain ground plane created at {prim_path}: "
            f"color={color}, half_extent={half_extent_m} m"
        )

    def _configure_gpu_dynamics(self, enabled: bool) -> None:
        """
        Soft-cable deformables need GPU PhysX. Mirrors
        detailedInsertion/cable/network_connector_pickup.enable_gpu_dynamics.
        """
        stage = omni.usd.get_context().get_stage()
        scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        if not scenes:
            scenes = [
                UsdPhysics.Scene.Define(
                    stage, Sdf.Path("/physicsScene")
                ).GetPrim()
            ]

        for prim in scenes:
            api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            api.CreateEnableGPUDynamicsAttr(enabled).Set(enabled)
            if enabled:
                api.CreateBroadphaseTypeAttr("GPU").Set("GPU")
                api.CreateSolverTypeAttr("TGS").Set("TGS")

        managed = SimulationManager.get_physics_scenes()
        for scene in managed:
            scene.set_enabled_gpu_dynamics(enabled)

        log(
            "PhysX GPU dynamics "
            f"{'ENABLED (soft cable)' if enabled else 'disabled'}"
        )

    def _load_cable(self) -> None:
        """Reference the network cable and place the tracked plug."""
        scene = self.cfg.scene
        stage = omni.usd.get_context().get_stage()

        if stage.GetPrimAtPath(scene.cable_root_path).IsValid():
            stage.RemovePrim(scene.cable_root_path)

        self._add_reference(scene.cable_usd_path, scene.cable_root_path)
        if scene.cable_support_enabled:
            self._place_tracked_plug_on_support()
        else:
            self._place_cable_on_ground()

    def _create_cable_support_block(self) -> None:
        """
        Visible static pedestal at cable_spawn_xy.

        Uses the same XY as the cable spawn so a +40 mm X shift moves both,
        but the block is not re-parented or snapped after cable placement.
        """
        scene = self.cfg.scene
        path = scene.cable_support_path
        size_x, size_y, size_z = (float(v) for v in scene.cable_support_size_m)
        if min(size_x, size_y, size_z) <= 0.0:
            raise ValueError(
                "cable_support_size_m must be positive in every axis."
            )

        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)

        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr(
            [Gf.Vec3f(*scene.cable_support_color)]
        )

        # Translate then scale — same order as network_cable_on_block_spawn.py.
        # Scale-then-translate scaled the spawn XY toward the origin and put
        # the orange box near the far connector instead of under the plug.
        center = Gf.Vec3d(
            scene.cable_spawn_xy[0],
            scene.cable_spawn_xy[1],
            size_z / 2.0,
        )
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(center)
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(size_x, size_y, size_z)
        )

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        top_z = float(center[2] + 0.5 * size_z)
        log(
            f"Cable support block created at {path}: "
            f"xy=({float(center[0]):.4f}, {float(center[1]):.4f}), "
            f"size_mm=({size_x * 1000.0:.1f}, {size_y * 1000.0:.1f}, "
            f"{size_z * 1000.0:.1f}), top_z={top_z:.4f} m"
        )

    def _cable_support_top_z_m(self) -> float:
        scene = self.cfg.scene
        if not scene.cable_support_enabled:
            return 0.0
        return float(scene.cable_support_size_m[2])

    def _set_prim_world_translate(
        self,
        prim_path: str,
        translation: np.ndarray,
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing prim: {prim_path}")

        value = Gf.Vec3d(*np.asarray(translation, dtype=np.float64).tolist())
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(value)
                return
        xform.AddTranslateOp().Set(value)

    def _place_tracked_plug_on_support(self) -> None:
        """
        Translate only /World/NetworkCable so the tracked plug rests on the
        block. Do not move individual heads — that detaches them from the
        wire when deformables are off.
        """
        scene = self.cfg.scene
        block_top_z = self._cable_support_top_z_m()
        plug_min, plug_max = self._world_bounds(scene.tracked_connector_path)
        plug_center = 0.5 * (plug_min + plug_max)
        root_position, _ = self._get_world_pose(scene.cable_root_path)

        desired_plug_min_z = (
            block_top_z + scene.cable_support_plug_clearance_m
        )
        delta = np.array(
            [
                scene.cable_spawn_xy[0] - plug_center[0],
                scene.cable_spawn_xy[1] - plug_center[1],
                desired_plug_min_z - plug_min[2],
            ],
            dtype=np.float64,
        )
        translation = root_position + delta
        self._set_prim_world_translate(scene.cable_root_path, translation)
        self._update_app(10)

        plug_min, plug_max = self._world_bounds(scene.tracked_connector_path)
        plug_center = 0.5 * (plug_min + plug_max)
        root_min, _ = self._world_bounds(scene.cable_root_path)
        log(
            "Cable plug placed on support:\n"
            f"  plug_center={np.round(plug_center, 4).tolist()}\n"
            f"  plug_min_z={plug_min[2]:.4f} "
            f"(block_top={block_top_z:.4f})\n"
            f"  root_min_z={root_min[2]:.4f}\n"
            f"  translation={np.round(translation, 4).tolist()}"
        )
        if root_min[2] < -0.002:
            warn(
                "Part of the cable root bbox is below ground; the far "
                "end of the wire may clip initially."
            )

    def _place_cable_on_ground(self) -> None:
        """
        Align the connector's bbox center over cable_spawn_xy, and drop the
        whole cable so its lowest point sits at ground_clearance above the
        ground plane. Root-only translate keeps both heads attached.
        """
        scene = self.cfg.scene
        root_min, _ = self._world_bounds(scene.cable_root_path)
        connector_min, connector_max = self._world_bounds(
            scene.tracked_connector_path
        )
        connector_center = (connector_min + connector_max) / 2.0
        root_position, _ = self._get_world_pose(scene.cable_root_path)

        delta = np.array(
            [
                scene.cable_spawn_xy[0] - connector_center[0],
                scene.cable_spawn_xy[1] - connector_center[1],
                scene.ground_clearance - root_min[2],
            ],
            dtype=np.float64,
        )
        translation = root_position + delta
        self._set_prim_world_translate(scene.cable_root_path, translation)
        self._update_app(10)
        log(
            "Cable placed on ground: "
            f"connector_center={np.round(connector_center, 4).tolist()}, "
            f"translation={np.round(translation, 4).tolist()}"
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

    @staticmethod
    def _as_cpu_numpy(value, dtype=np.float64) -> np.ndarray:
        """Copy Isaac pose/tensors to host NumPy (needed when device=cuda)."""
        if value is None:
            raise RuntimeError("Expected a pose value, got None.")
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            array = np.asarray(value.numpy(), dtype=dtype)
        else:
            array = np.asarray(value, dtype=dtype)
        return np.array(array, dtype=dtype, copy=True)

    def _world_pose_numpy(self, prim) -> tuple[np.ndarray, np.ndarray]:
        """get_world_pose() as host float64 arrays for Lula / NumPy math."""
        position, orientation = prim.get_world_pose()
        return (
            self._as_cpu_numpy(position).reshape(3),
            self._as_cpu_numpy(orientation).reshape(4),
        )

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

