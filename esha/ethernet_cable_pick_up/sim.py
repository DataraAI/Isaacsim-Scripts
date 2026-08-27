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
from grasp_control import (
    apply_grasp_x_offset,
    bounded_linear_step,
    clearance_target_position,
    finger_target_reached,
    fingers_moved_toward_closed,
    grasp_orientation_active,
    resolve_tool_orientation,
    select_open_half_gap,
)
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


def _find_unique_descendant(root: Usd.Prim, name: str) -> str:
    matches = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.GetName() == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one '{name}' below {root.GetPath()}, found {matches}"
        )
    return matches[0]


def _matrix_to_gf_quatf(rotation: np.ndarray) -> Gf.Quatf:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Rotation matrix must be finite with shape (3, 3).")
    quaternion = matrix_to_quaternion_wxyz(matrix)
    return Gf.Quatf(
        float(quaternion[0]),
        Gf.Vec3f(
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ),
    )


def _world_transform(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(
            f"Cannot read transform for invalid prim: {path}"
        )
    gf_matrix = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(prim)
    return np.asarray(gf_matrix, dtype=np.float64).T


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
    """30° standoff → open → grasp → close → lift → pullback → reorient → carry."""

    # idle | angle | open | grasp_descend | close | lift | pullback | reorient | carry | done
    phase: str = "idle"
    # Half of connector short-axis thickness (from stereo height_m).
    cable_half_width_m: float | None = None
    cable_point_world_m: np.ndarray | None = None
    grasp_target_world_m: np.ndarray | None = None
    move_settled_frames: int = 0
    open_frames: int = 0
    close_frames: int = 0
    reorient_frames: int = 0
    finger_settle_frames: int = 0
    finger_half_gap_m: float = 0.0
    finger_close_half_gap_m: float = 0.0
    last_finger_positions_m: np.ndarray | None = None
    close_start_finger_positions_m: np.ndarray | None = None
    open_timeout_reported: bool = False
    close_timeout_reported: bool = False
    grasp_orientation_wxyz: np.ndarray | None = None
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

    def __init__(self, simulation_app, cfg: Config, run_logger=None):
        self.app = simulation_app
        self.cfg = cfg
        # Structured run logger (run_logger.RunLogger); optional so the
        # runtime can still be constructed without one.
        self.run_logger = run_logger

        self.frame_index = 0
        self._post_grasp_fixed_joint_path = "/World/CableGraspFixedJoint"
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
        """Tool-center pose from the live panda_hand USD Xform."""
        if self.ik is None:
            raise RuntimeError("IK runtime is not initialized.")
        cfg = self.cfg.ik
        hand_position, hand_orientation = self._get_world_pose(
            self.ik.hand_path
        )
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

        # Fingers close on the short axis (thickness), not the long axis.
        self.pre_grasp.cable_half_width_m = 0.5 * float(
            observation.height_m
        )
        self.pre_grasp.cable_point_world_m = np.asarray(
            observation.grasp_world_xyz_m,
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
                "30° standoff → open → grasp → close → lift → carry"
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
        """Angled approach first, then grasp/close/lift without reorienting."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp

        if not cfg.enabled or self.ik is None:
            return
        if state.phase in ("idle", "done"):
            if state.phase == "done":
                self._apply_finger_gap(state.finger_close_half_gap_m)
            return

        open_phases = ("open", "grasp_descend")
        close_phases = ("close", "lift", "pullback", "reorient", "carry")
        if state.phase in open_phases:
            self._apply_finger_gap(state.finger_half_gap_m)
        elif state.phase in close_phases:
            self._apply_finger_gap(state.finger_close_half_gap_m)

        required_settled = self.cfg.visual_servo.required_settled_frames

        if state.phase == "angle":
            if not self._pre_grasp_pose_settled(required_settled):
                return
            state.phase = "open"
            state.open_frames = 0
            state.finger_settle_frames = 0
            state.last_finger_positions_m = None
            state.open_timeout_reported = False
            self._apply_finger_gap(state.finger_half_gap_m)
            log(
                "GRASP ANGLE SETTLED - OPEN FINGERS\n"
                f"  fingers open at "
                f"{state.finger_half_gap_m * 1000.0:.1f} mm per side"
            )
            return

        if state.phase == "open":
            state.open_frames += 1
            fingers_open = self._pre_grasp_fingers_open_settled()
            if (
                state.open_frames >= cfg.open_timeout_frames
                and not fingers_open
                and not state.open_timeout_reported
            ):
                state.open_timeout_reported = True
                open_positions = self._finger_joint_positions_m()
                warn(
                    "GRASP OPEN has not reached the commanded gap; "
                    "holding at standoff.\n"
                    f"  target_positions_m="
                    f"{state.finger_half_gap_m:.4f} per side\n"
                    f"  actual_positions_m="
                    f"{None if open_positions is None else np.round(open_positions, 4).tolist()}"
                )
            if (
                state.open_frames < cfg.open_hold_frames
                or not fingers_open
            ):
                return
            finger_pos = self._finger_joint_positions_m()
            log(
                "GRASP OPEN SETTLED\n"
                f"  finger_positions_m="
                f"{None if finger_pos is None else np.round(finger_pos, 4).tolist()}\n"
                "  moving to cable with fingers held open"
            )
            self._begin_grasp_descend()
            return

        if state.phase == "grasp_descend":
            if not self._advance_linear_ik_target(
                state.grasp_target_world_m,
                max_step_m=cfg.grasp_approach_step_m,
            ):
                return
            if not self._pre_grasp_pose_settled(required_settled):
                return
            state.phase = "close"
            state.close_frames = 0
            state.finger_settle_frames = 0
            state.last_finger_positions_m = None
            state.close_start_finger_positions_m = (
                self._finger_joint_positions_m()
            )
            state.close_timeout_reported = False
            state.finger_close_half_gap_m = (
                self._compute_finger_close_half_gap_m()
            )
            self._apply_finger_gap(state.finger_close_half_gap_m)
            log(
                "GRASP CLOSE\n"
                f"  commanding "
                f"{state.finger_close_half_gap_m * 1000.0:.1f} mm "
                f"per side (full close drive)\n"
                "  waiting for finger joints to stop moving before lift"
            )
            return

        if state.phase == "close":
            state.close_frames += 1
            fingers_settled = self._pre_grasp_fingers_settled()
            timed_out = state.close_frames >= cfg.close_timeout_frames
            if (
                timed_out
                and not fingers_settled
                and not state.close_timeout_reported
            ):
                state.close_timeout_reported = True
                close_positions = self._finger_joint_positions_m()
                warn(
                    "GRASP CLOSE timed out before fingers fully settled; "
                    "holding closed without lifting.\n"
                    f"  close_frames={state.close_frames}\n"
                    f"  finger_positions_m="
                    f"{None if close_positions is None else np.round(close_positions, 4).tolist()}"
                )
            if not fingers_settled:
                return
            if state.finger_settle_frames < (
                cfg.finger_settle_frames + cfg.close_hold_frames
            ):
                return
            finger_pos = self._finger_joint_positions_m()
            log(
                "GRASP CLOSE SETTLED\n"
                f"  close_frames={state.close_frames}\n"
                f"  finger_positions_m="
                f"{None if finger_pos is None else np.round(finger_pos, 4).tolist()}"
            )
            if self.run_logger is not None:
                self.run_logger.log_event(
                    t=self.frame_index,
                    event="GRASP_CLOSE_SETTLED",
                    close_frames=state.close_frames,
                    finger_close_half_gap_mm=(
                        state.finger_close_half_gap_m * 1000.0
                    ),
                )
            self._weld_grasped_cable_to_hand()
            self._begin_grasp_lift()
            return

        if state.phase == "lift":
            if not self._advance_linear_ik_target(
                state.grasp_target_world_m,
                max_step_m=cfg.lift_step_m,
            ):
                return
            if not self._pre_grasp_pose_settled(required_settled):
                return
            target_position, target_orientation = self._get_world_pose(
                self.cfg.ik.target_path
            )
            commanded_orientation = resolve_tool_orientation(
                target_orientation,
                state.grasp_orientation_wxyz,
                grasp_active=True,
            )
            log(
                "GRASP LIFT DONE\n"
                f"  lift_z_m={cfg.lift_z_m:.4f}\n"
                f"  orientation held (no post-grasp rotate)\n"
                f"  IK_Target={np.round(target_position, 4).tolist()}\n"
                f"  commanded_grasp_ori_wxyz="
                f"{np.round(commanded_orientation, 4).tolist()}\n"
                f"  fingers held closed at "
                f"{state.finger_close_half_gap_m * 1000.0:.1f} mm per side"
            )
            if self.run_logger is not None:
                self.run_logger.log_event(
                    t=self.frame_index,
                    event="GRASP_LIFT_DONE",
                    lift_z_m=cfg.lift_z_m,
                    ik_target=np.round(target_position, 4).tolist(),
                )
            if cfg.carry_orientation_wxyz is not None:
                self._begin_grasp_pullback()
            else:
                self._begin_grasp_carry()
            return

        if state.phase == "pullback":
            if not self._advance_linear_ik_target(
                state.grasp_target_world_m,
                max_step_m=cfg.pullback_step_m,
            ):
                return
            if not self._pre_grasp_pose_settled(required_settled):
                return
            target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
            log(
                "GRASP PULLBACK DONE\n"
                f"  reorient_pullback_x_m={cfg.reorient_pullback_x_m:.4f}\n"
                f"  IK_Target={np.round(target_position, 4).tolist()}"
            )
            self._begin_grasp_reorient()
            return

        if state.phase == "reorient":
            state.reorient_frames += 1
            if not self._pre_grasp_pose_settled(required_settled):
                if state.reorient_frames > cfg.reorient_timeout_frames:
                    warn(
                        "GRASP REORIENT timed out; target orientation may be "
                        "unreachable after pullback.\n"
                        f"  reorient_frames={state.reorient_frames}\n"
                        f"  timeout_frames={cfg.reorient_timeout_frames}"
                    )
                    self._begin_grasp_carry()
                return
            target_position, target_orientation = self._get_world_pose(
                self.cfg.ik.target_path
            )
            commanded_orientation = resolve_tool_orientation(
                target_orientation,
                state.grasp_orientation_wxyz,
                grasp_active=True,
            )
            log(
                "GRASP REORIENT DONE\n"
                f"  reorient_frames={state.reorient_frames}\n"
                f"  IK_Target={np.round(target_position, 4).tolist()}\n"
                f"  commanded_grasp_ori_wxyz="
                f"{np.round(commanded_orientation, 4).tolist()}"
            )
            self._begin_grasp_carry()
            return

        if state.phase == "carry":
            if not self._advance_linear_ik_target(
                state.grasp_target_world_m,
                max_step_m=cfg.carry_step_m,
            ):
                return
            if not self._pre_grasp_pose_settled(required_settled):
                return
            target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
            log(
                "GRASP CARRY DONE\n"
                f"  carry_offset_m={np.round(cfg.carry_offset_m, 4).tolist()}\n"
                f"  IK_Target={np.round(target_position, 4).tolist()}\n"
                f"  fingers held closed at "
                f"{state.finger_close_half_gap_m * 1000.0:.1f} mm per side"
            )
            if self.run_logger is not None:
                self.run_logger.log_event(
                    t=self.frame_index,
                    event="GRASP_CARRY_DONE",
                    carry_offset_m=np.round(cfg.carry_offset_m, 4).tolist(),
                    ik_target=np.round(target_position, 4).tolist(),
                )
            # Temporary: skip handoff reorient/move so phase 1 ends at the
            # normal GRASP CARRY DONE pose while we verify single_rack_cv's
            # corrected target. Keep _begin_handoff_* helpers/config intact.
            state.phase = "done"
            return

        if state.phase == "handoff_reorient":
            state.reorient_frames += 1
            if not self._pre_grasp_pose_settled(required_settled):
                if state.reorient_frames > cfg.reorient_timeout_frames:
                    warn(
                        "HANDOFF REORIENT timed out; target orientation may "
                        "be unreachable.\n"
                        f"  reorient_frames={state.reorient_frames}\n"
                        f"  timeout_frames={cfg.reorient_timeout_frames}"
                    )
                    self._begin_handoff_move()
                return
            target_position, target_orientation = self._get_world_pose(
                self.cfg.ik.target_path
            )
            log(
                "HANDOFF REORIENT DONE\n"
                f"  reorient_frames={state.reorient_frames}\n"
                f"  IK_Target={np.round(target_position, 4).tolist()}\n"
                f"  orientation_wxyz="
                f"{np.round(target_orientation, 4).tolist()}"
            )
            self._begin_handoff_move()
            return

        if state.phase == "handoff_move":
            if not self._advance_linear_ik_target(
                state.grasp_target_world_m,
                max_step_m=cfg.handoff_step_m,
            ):
                return
            if not self._pre_grasp_pose_settled(required_settled):
                return
            state.phase = "done"
            target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
            log(
                "HANDOFF MOVE DONE\n"
                f"  IK_Target={np.round(target_position, 4).tolist()}\n"
                f"  fingers held closed at "
                f"{state.finger_close_half_gap_m * 1000.0:.1f} mm per side"
            )
            return

    def _tool_target_orientation_error_rad(self) -> float:
        """Angle between actual ToolCenter and IK_Target orientations."""
        if self.ik is None:
            return math.inf
        self._update_actual_tool_frame(self.ik)
        _, target_orientation = self._get_world_pose(self.cfg.ik.target_path)
        _, actual_orientation = self._tool_pose_from_hand()
        qa = _normalize_quaternion_wxyz(actual_orientation)
        qt = resolve_tool_orientation(
            target_orientation,
            self.pre_grasp.grasp_orientation_wxyz,
            grasp_active=grasp_orientation_active(
                self.pre_grasp.phase
            ),
        )
        dot = abs(float(np.dot(qa, qt)))
        dot = min(1.0, max(0.0, dot))
        return float(2.0 * math.acos(dot))

    def _pre_grasp_pose_settled(self, required_settled: int) -> bool:
        """Require both position and orientation to settle before the next phase."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        position_ok = (
            self._tool_target_position_error_m()
            <= self._target_settle_tolerance_m()
        )
        orientation_ok = (
            self._tool_target_orientation_error_rad()
            <= float(cfg.orientation_settle_tolerance_rad)
        )
        if position_ok and orientation_ok:
            state.move_settled_frames += 1
        else:
            state.move_settled_frames = 0
        return state.move_settled_frames >= required_settled

    def _pre_grasp_move_settled(self, required_settled: int) -> bool:
        # Kept for any callers; pose settle is the grasp-path check.
        return self._pre_grasp_pose_settled(required_settled)

    def _finger_joint_positions_m(self) -> np.ndarray | None:
        if self.ik is None or not self._finger_dof_indices:
            return None
        positions = self._as_cpu_numpy(
            self.ik.articulation.get_joint_positions()
        ).reshape(-1)
        return positions[np.asarray(self._finger_dof_indices, dtype=int)]

    def _pre_grasp_fingers_open_settled(self) -> bool:
        """Require both fingers to reach and settle at the open command."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        positions = self._finger_joint_positions_m()
        if positions is None:
            state.finger_settle_frames = 0
            return False

        reached = finger_target_reached(
            positions,
            target_position_m=state.finger_half_gap_m,
            tolerance_m=cfg.finger_open_target_tolerance_m,
        )
        if state.last_finger_positions_m is None:
            state.last_finger_positions_m = positions.copy()
            state.finger_settle_frames = 0
            return False

        delta_m = float(
            np.max(np.abs(positions - state.last_finger_positions_m))
        )
        state.last_finger_positions_m = positions.copy()
        if reached and delta_m <= cfg.finger_settle_tolerance_m:
            state.finger_settle_frames += 1
        else:
            state.finger_settle_frames = 0
        return state.finger_settle_frames >= cfg.finger_settle_frames

    def _pre_grasp_fingers_settled(self) -> bool:
        """Require real inward travel followed by stable finger positions."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        positions = self._finger_joint_positions_m()
        if positions is None:
            state.finger_settle_frames = 0
            return False

        if state.last_finger_positions_m is None:
            state.last_finger_positions_m = positions.copy()
            state.finger_settle_frames = 0
            return False

        moved_inward = (
            state.close_start_finger_positions_m is not None
            and fingers_moved_toward_closed(
                positions,
                state.close_start_finger_positions_m,
                minimum_travel_m=cfg.close_min_travel_m,
            )
        )
        delta_m = float(
            np.max(np.abs(positions - state.last_finger_positions_m))
        )
        state.last_finger_positions_m = positions.copy()
        if moved_inward and delta_m <= cfg.finger_settle_tolerance_m:
            state.finger_settle_frames += 1
        else:
            state.finger_settle_frames = 0
        return state.finger_settle_frames >= cfg.finger_settle_frames

    def _cable_half_width_m(self) -> float:
        cfg = self.cfg.pre_grasp
        half_width = self.pre_grasp.cable_half_width_m
        if half_width is None or not math.isfinite(float(half_width)):
            return float(cfg.fallback_cable_half_width_m)
        return float(half_width)

    def _compute_finger_half_gap_m(self) -> float:
        cfg = self.cfg.pre_grasp
        return select_open_half_gap(
            cable_half_width_m=self._cable_half_width_m(),
            side_allowance_m=cfg.side_allowance_m,
            minimum_half_gap_m=cfg.minimum_open_half_gap_m,
            maximum_half_gap_m=cfg.finger_max_open_m,
        )

    def _compute_finger_close_half_gap_m(self) -> float:
        cfg = self.cfg.pre_grasp
        return max(0.0, float(cfg.close_target_half_gap_m))

    def _ensure_cable_point_world_m(self) -> np.ndarray:
        state = self.pre_grasp
        if state.cable_point_world_m is None:
            pos, _ = self._get_world_pose(
                self.cfg.scene.tracked_connector_path
            )
            state.cable_point_world_m = np.asarray(
                pos,
                dtype=np.float64,
            ).reshape(3)
        return apply_grasp_x_offset(
            state.cable_point_world_m,
            x_offset_m=self.cfg.pre_grasp.grasp_point_x_offset_m,
        )

    def _grasp_approach_direction(self) -> np.ndarray:
        """Unit tool-Z: points from tool toward cable at configured elevation."""
        cfg = self.cfg.pre_grasp
        elev = math.radians(float(cfg.grasp_elevation_deg))
        azim = math.radians(float(cfg.grasp_azimuth_deg))
        cos_e = math.cos(elev)
        sin_e = math.sin(elev)
        # Horizontal component toward azimuth; -Z for downward pitch.
        approach = np.array(
            [
                cos_e * math.cos(azim),
                cos_e * math.sin(azim),
                -sin_e,
            ],
            dtype=np.float64,
        )
        return approach / float(np.linalg.norm(approach))

    def _grasp_orientation_wxyz(self) -> np.ndarray:
        """
        Tool frame with +Z along the angled approach and fingers along ~world Y.

        Visual servo stays top-down; this orientation is applied at pre-grasp.
        """
        tool_z = self._grasp_approach_direction()
        # Prefer world +Y as the finger-opening axis (grip short sides of plug).
        finger_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(finger_hint, tool_z))) > 0.95:
            finger_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        tool_x = np.cross(finger_hint, tool_z)
        tool_x_norm = float(np.linalg.norm(tool_x))
        if tool_x_norm <= 1.0e-9:
            raise RuntimeError("Failed to build grasp tool X axis.")
        tool_x /= tool_x_norm
        tool_y = np.cross(tool_z, tool_x)
        tool_y /= float(np.linalg.norm(tool_y))
        tool_x = np.cross(tool_y, tool_z)
        tool_x /= float(np.linalg.norm(tool_x))

        rotation = np.column_stack((tool_x, tool_y, tool_z))
        return matrix_to_quaternion_wxyz(rotation)

    def _set_ik_target_clearance_from_cable(
        self,
        clearance_m: float,
        *,
        log_title: str,
    ) -> bool:
        """Place IK_Target at cable_point - clearance * grasp approach."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if self.ik is None:
            return False

        if state.grasp_orientation_wxyz is None:
            state.grasp_orientation_wxyz = self._grasp_orientation_wxyz()
        orientation = _normalize_quaternion_wxyz(state.grasp_orientation_wxyz)
        cable_point = self._ensure_cable_point_world_m()
        approach = self._grasp_approach_direction()
        desired_tool = self._ik_target_position_at_cable_clearance(clearance_m)

        self.ik.target.set_world_pose(
            position=desired_tool,
            orientation=orientation,
        )
        state.move_settled_frames = 0
        log(
            f"{log_title}\n"
            f"  cable_point={np.round(cable_point, 4).tolist()}\n"
            f"  clearance_m={float(clearance_m):.4f}\n"
            f"  elevation_deg={cfg.grasp_elevation_deg:.1f}\n"
            f"  azimuth_deg={cfg.grasp_azimuth_deg:.1f}\n"
            f"  approach={np.round(approach, 4).tolist()}\n"
            f"  new IK_Target={np.round(desired_tool, 4).tolist()}"
        )
        return True

    def _ik_target_position_at_cable_clearance(
        self,
        clearance_m: float,
    ) -> np.ndarray:
        """Calculate a collision-clamped ToolCenter waypoint."""
        cfg = self.cfg.pre_grasp
        cable_point = self._ensure_cable_point_world_m()
        approach = self._grasp_approach_direction()
        return clearance_target_position(
            cable_point,
            approach,
            clearance_m=float(clearance_m),
            minimum_z_m=(
                self._cable_support_top_z_m()
                + cfg.block_safety_margin_m
            ),
        )

    def _begin_pre_grasp_descend(self) -> None:
        """Move once to a clear 30° standoff before opening the fingers."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if state.started or self.ik is None:
            return
        state.started = True
        state.finger_half_gap_m = self._compute_finger_half_gap_m()
        state.finger_close_half_gap_m = self._compute_finger_close_half_gap_m()
        state.grasp_orientation_wxyz = self._grasp_orientation_wxyz()

        if not self._set_ik_target_clearance_from_cable(
            cfg.approach_standoff_m,
            log_title="30 DEGREE GRASP STANDOFF",
        ):
            return
        state.phase = "angle"
        log(
            f"  finger_half_gap_m={state.finger_half_gap_m:.4f}\n"
            f"  side_allowance_m={cfg.side_allowance_m:.4f}\n"
            "  fingers stay unchanged until the 30 degree pose settles"
        )

    def _begin_grasp_descend(self) -> None:
        """Start a bounded straight-line approach with fingers held open."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if self.ik is None:
            return
        state.grasp_target_world_m = (
            self._ik_target_position_at_cable_clearance(
                cfg.grasp_clearance_m
            )
        )
        current_target, _ = self._get_world_pose(
            self.cfg.ik.target_path
        )
        state.phase = "grasp_descend"
        state.move_settled_frames = 0
        self._apply_finger_gap(state.finger_half_gap_m)
        log(
            "STRAIGHT GRASP APPROACH\n"
            f"  start IK_Target={np.round(current_target, 4).tolist()}\n"
            f"  final IK_Target="
            f"{np.round(state.grasp_target_world_m, 4).tolist()}\n"
            f"  max_step_m={cfg.grasp_approach_step_m:.4f}\n"
            f"  fingers held open at "
            f"{state.finger_half_gap_m * 1000.0:.1f} mm per side"
        )

    def _advance_linear_ik_target(
        self,
        final_target_world_m: np.ndarray | None,
        *,
        max_step_m: float,
    ) -> bool:
        """Advance IK_Target only after the arm catches up to each line step."""
        state = self.pre_grasp
        if self.ik is None or final_target_world_m is None:
            return False
        current_target, _ = self._get_world_pose(
            self.cfg.ik.target_path
        )
        final_target = np.asarray(
            final_target_world_m,
            dtype=np.float64,
        ).reshape(3)
        remaining_m = float(np.linalg.norm(final_target - current_target))
        if remaining_m <= 1.0e-6:
            return True
        if (
            self._tool_target_position_error_m()
            > self._target_settle_tolerance_m()
        ):
            return False
        next_target = bounded_linear_step(
            current_target,
            final_target,
            max_step_m=max_step_m,
        )
        orientation = _normalize_quaternion_wxyz(
            state.grasp_orientation_wxyz
        )
        self.ik.target.set_world_pose(
            position=next_target,
            orientation=orientation,
        )
        state.move_settled_frames = 0
        return False

    def _begin_grasp_lift(self) -> None:
        """Lift +world Z in small steps while holding the angled grasp orientation."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if self.ik is None:
            return

        target_position, target_orientation = self._get_world_pose(
            self.cfg.ik.target_path
        )
        if state.grasp_orientation_wxyz is not None:
            target_orientation = _normalize_quaternion_wxyz(
                state.grasp_orientation_wxyz
            )
        lift_position = np.asarray(
            target_position,
            dtype=np.float64,
        ).reshape(3).copy()
        lift_position[2] += float(cfg.lift_z_m)
        state.grasp_target_world_m = lift_position
        self.ik.target.set_world_pose(
            position=target_position,
            orientation=target_orientation,
        )
        state.phase = "lift"
        state.move_settled_frames = 0
        log(
            "GRASP LIFT\n"
            f"  lift_z_m={cfg.lift_z_m:.4f}\n"
            f"  max_step_m={cfg.lift_step_m:.4f}\n"
            "  keeping angled orientation (no rotate after grasp)\n"
            f"  start IK_Target={np.round(target_position, 4).tolist()}\n"
            f"  final IK_Target={np.round(lift_position, 4).tolist()}"
        )
        self._apply_finger_gap(state.finger_close_half_gap_m)

    def fixed_joint_is_valid(self) -> bool:
        """Return whether the post-grasp panda_hand-to-plug weld is intact."""
        if self.ik is None:
            return False
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        joint_prim = stage.GetPrimAtPath(self._post_grasp_fixed_joint_path)
        if (
            not joint_prim.IsValid()
            or not joint_prim.IsA(UsdPhysics.FixedJoint)
        ):
            return False
        joint = UsdPhysics.FixedJoint(joint_prim)
        body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        return (
            body0 == [self.ik.hand_path]
            and body1 == [self.cfg.scene.tracked_connector_path]
        )

    def _author_fixed_joint(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("A valid USD stage is required to weld the cable.")
        if self.ik is None:
            raise RuntimeError("IK runtime is not initialized.")

        world_from_hand = _world_transform(stage, self.ik.hand_path)
        world_from_plug = _world_transform(
            stage,
            self.cfg.scene.tracked_connector_path,
        )
        hand_from_plug = np.linalg.inv(world_from_hand) @ world_from_plug
        if hand_from_plug.shape != (4, 4) or not np.all(
            np.isfinite(hand_from_plug)
        ):
            raise RuntimeError("Computed hand-to-plug transform is invalid.")

        joint = UsdPhysics.FixedJoint.Define(
            stage,
            Sdf.Path(self._post_grasp_fixed_joint_path),
        )
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.ik.hand_path)])
        joint.CreateBody1Rel().SetTargets(
            [Sdf.Path(self.cfg.scene.tracked_connector_path)]
        )
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*[float(value) for value in hand_from_plug[:3, 3]])
        )
        joint.CreateLocalRot0Attr().Set(
            _matrix_to_gf_quatf(hand_from_plug[:3, :3])
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        )

    def _filter_hand_and_finger_collisions(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "A valid USD stage is required to set collision filters."
            )
        root = stage.GetPrimAtPath(self.cfg.scene.franka_asset_path)
        if not root.IsValid():
            raise RuntimeError(
                "Franka asset root is invalid: "
                f"{self.cfg.scene.franka_asset_path}"
            )

        names = (
            self.cfg.camera.hand_link_name,
            *self.cfg.pre_grasp.finger_link_names,
        )
        filtered_paths: list[str] = []
        for name in names:
            path = _find_unique_descendant(root, name)
            prim = stage.GetPrimAtPath(path)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(
                    f"Collision-filter target is not a rigid body: {path}"
                )
            filtered_paths.append(path)

        plug = stage.GetPrimAtPath(self.cfg.scene.tracked_connector_path)
        if not plug.IsValid():
            raise RuntimeError(
                "Tracked plug is invalid: "
                f"{self.cfg.scene.tracked_connector_path}"
            )
        api = UsdPhysics.FilteredPairsAPI.Apply(plug)
        relationship = api.CreateFilteredPairsRel()
        existing = {str(path) for path in relationship.GetTargets()}
        combined = sorted(existing.union(filtered_paths))
        relationship.SetTargets([Sdf.Path(path) for path in combined])

    def _weld_grasped_cable_to_hand(self) -> None:
        """Create the fixed joint once, immediately after the grasp settles."""
        if self.fixed_joint_is_valid():
            return
        self._author_fixed_joint()
        self._filter_hand_and_finger_collisions()
        if not self.fixed_joint_is_valid():
            raise RuntimeError(
                "Direct panda_hand-to-plug fixed joint is invalid."
            )
        log(
            "GRASP WELD AUTHORED\n"
            f"  fixed_joint={self._post_grasp_fixed_joint_path}\n"
            f"  body0={self.ik.hand_path if self.ik is not None else 'N/A'}\n"
            f"  body1={self.cfg.scene.tracked_connector_path}"
        )
        if self.run_logger is not None:
            self.run_logger.log_event(
                t=self.frame_index,
                event="GRASP_WELD_AUTHORED",
                fixed_joint=self._post_grasp_fixed_joint_path,
                body0=(
                    self.ik.hand_path if self.ik is not None else "N/A"
                ),
                body1=self.cfg.scene.tracked_connector_path,
            )

    def _begin_grasp_pullback(self) -> None:
        """Pull the tool closer to the base before rotating.

        The lift position is too near full reach to also rotate there
        (Franka max reach ~0.85 m).
        """
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if self.ik is None:
            return

        target_position, target_orientation = self._get_world_pose(
            self.cfg.ik.target_path
        )
        if state.grasp_orientation_wxyz is not None:
            target_orientation = _normalize_quaternion_wxyz(
                state.grasp_orientation_wxyz
            )
        pullback_position = np.asarray(
            target_position,
            dtype=np.float64,
        ).reshape(3).copy()
        pullback_position[0] = float(cfg.reorient_pullback_x_m)
        state.grasp_target_world_m = pullback_position
        # Seed the current pose so _advance_linear_ik_target can step in.
        self.ik.target.set_world_pose(
            position=target_position,
            orientation=target_orientation,
        )
        state.phase = "pullback"
        state.move_settled_frames = 0
        self._apply_finger_gap(state.finger_close_half_gap_m)
        log(
            "GRASP PULLBACK\n"
            f"  reorient_pullback_x_m={cfg.reorient_pullback_x_m:.4f}\n"
            f"  start IK_Target={np.round(target_position, 4).tolist()}\n"
            f"  final IK_Target={np.round(pullback_position, 4).tolist()}\n"
            f"  max_step_m={cfg.pullback_step_m:.4f}\n"
            "  keeping angled orientation until reorient"
        )

    def _begin_grasp_reorient(self) -> None:
        """Rotate in place to carry_orientation_wxyz at the pulled-in position."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if self.ik is None:
            return

        target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
        orientation = _normalize_quaternion_wxyz(
            np.asarray(cfg.carry_orientation_wxyz, dtype=np.float64)
        )
        state.grasp_orientation_wxyz = orientation
        self.ik.target.set_world_pose(
            position=target_position,
            orientation=orientation,
        )
        state.phase = "reorient"
        state.move_settled_frames = 0
        state.reorient_frames = 0
        log(
            "GRASP REORIENT\n"
            f"  held IK_Target={np.round(target_position, 4).tolist()}\n"
            f"  new orientation_wxyz={np.round(orientation, 4).tolist()}"
        )
        self._apply_finger_gap(state.finger_close_half_gap_m)

    def _begin_grasp_carry(self) -> None:
        """Step current IK_Target by carry_offset_m; keep gripper closed."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if self.ik is None:
            return

        target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
        offset = np.asarray(cfg.carry_offset_m, dtype=np.float64).reshape(3)
        state.grasp_target_world_m = (
            np.asarray(target_position, dtype=np.float64).reshape(3) + offset
        )
        state.phase = "carry"
        state.move_settled_frames = 0
        self._apply_finger_gap(state.finger_close_half_gap_m)
        log(
            "GRASP CARRY\n"
            f"  carry_offset_m={np.round(offset, 4).tolist()}\n"
            f"  start IK_Target={np.round(target_position, 4).tolist()}\n"
            f"  final IK_Target="
            f"{np.round(state.grasp_target_world_m, 4).tolist()}\n"
            f"  max_step_m={cfg.carry_step_m:.4f}\n"
            "  fingers stay closed"
        )

    def _begin_handoff_reorient(self) -> None:
        """Rotate in place to the exact hand orientation single_rack_cv's
        insertion IK expects at startup, before moving to the handoff
        position."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if self.ik is None:
            return

        target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
        hand_orientation = _normalize_quaternion_wxyz(
            np.asarray(
                cfg.handoff_hand_target_orientation_wxyz,
                dtype=np.float64,
            )
        )
        _, tool_orientation = hand_pose_to_tool_pose(
            hand_position_m=np.asarray(
                cfg.handoff_hand_target_position_m, dtype=np.float64
            ),
            hand_orientation_wxyz=hand_orientation,
            tool_local_position_m=np.asarray(
                self.cfg.ik.tool_center_local_position_m, dtype=np.float64
            ),
            tool_local_orientation_wxyz=np.asarray(
                self.cfg.ik.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            ),
        )
        state.grasp_orientation_wxyz = tool_orientation
        self.ik.target.set_world_pose(
            position=target_position,
            orientation=tool_orientation,
        )
        state.phase = "handoff_reorient"
        state.move_settled_frames = 0
        state.reorient_frames = 0
        log(
            "HANDOFF REORIENT\n"
            f"  held IK_Target={np.round(target_position, 4).tolist()}\n"
            f"  new orientation_wxyz="
            f"{np.round(tool_orientation, 4).tolist()}"
        )
        self._apply_finger_gap(state.finger_close_half_gap_m)

    def _begin_handoff_move(self) -> None:
        """Translate to the exact tool-center position matching
        single_rack_cv's expected startup hand pose."""
        cfg = self.cfg.pre_grasp
        state = self.pre_grasp
        if self.ik is None:
            return

        target_position, _ = self._get_world_pose(self.cfg.ik.target_path)
        hand_orientation = _normalize_quaternion_wxyz(
            np.asarray(
                cfg.handoff_hand_target_orientation_wxyz,
                dtype=np.float64,
            )
        )
        tool_position, _ = hand_pose_to_tool_pose(
            hand_position_m=np.asarray(
                cfg.handoff_hand_target_position_m, dtype=np.float64
            ),
            hand_orientation_wxyz=hand_orientation,
            tool_local_position_m=np.asarray(
                self.cfg.ik.tool_center_local_position_m, dtype=np.float64
            ),
            tool_local_orientation_wxyz=np.asarray(
                self.cfg.ik.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            ),
        )
        state.grasp_target_world_m = tool_position
        state.phase = "handoff_move"
        state.move_settled_frames = 0
        self._apply_finger_gap(state.finger_close_half_gap_m)
        log(
            "HANDOFF MOVE\n"
            f"  start IK_Target={np.round(target_position, 4).tolist()}\n"
            f"  final IK_Target={np.round(tool_position, 4).tolist()}\n"
            f"  max_step_m={cfg.handoff_step_m:.4f}"
        )

    def _apply_finger_gap(self, half_gap_m: float) -> None:
        """Command both finger joints to the given half-gap."""
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

        if self.run_logger is not None:
            self.run_logger.log_frame(
                t=self.frame_index,
                tool_center_error_mm=(
                    self._tool_target_position_error_m() * 1000.0
                ),
                tool_center_orientation_error_deg=math.degrees(
                    self._tool_target_orientation_error_rad()
                ),
                phase=self.pre_grasp.phase,
            )

        if not cfg.tracking_enabled:
            return

        if self.frame_index % cfg.update_every_sim_frames != 0:
            return

        desired_tool_position, desired_tool_orientation = (
            self._get_world_pose(cfg.target_path)
        )
        desired_tool_orientation = resolve_tool_orientation(
            desired_tool_orientation,
            self.pre_grasp.grasp_orientation_wxyz,
            grasp_active=grasp_orientation_active(
                self.pre_grasp.phase
            ),
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
            ),
        )
        right_frame = CameraFrame(
            rgb=right_rgb,
            camera=self._camera_model(
                self.right_camera_path,
                right_rgb,
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
    ) -> CameraModel:
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        camera = UsdGeom.Camera(camera_prim)

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
        if scene.datahall_enabled:
            self._load_datahall()

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

        # Always run SimulationManager on CPU. Soft-cable GPU dynamics are
        # enabled separately via PhysxSceneAPI.EnableGPUDynamicsAttr in
        # _configure_gpu_dynamics(); that must stay on for deformables.
        physics_device = "cpu"

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

        self._apply_grasp_physics_materials()
        self.ik = self._create_ik(assets_root)
        self._set_external_view()

        if scene.datahall_enabled:
            datahall_line = (
                f"  datahall:   {scene.datahall_prim_path} at "
                f"({scene.cable_spawn_xy[0] + scene.datahall_offset_from_cable_xy[0]:.2f}, "
                f"{scene.cable_spawn_xy[1] + scene.datahall_offset_from_cable_xy[1]:.2f}, 0)\n"
            )
        else:
            datahall_line = "  datahall:   disabled\n"
        log(
            "READY\n"
            f"  cable:      {scene.cable_usd_path}\n"
            f"  connector:  {scene.tracked_connector_path}\n"
            f"{datahall_line}"
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

    @staticmethod
    def _set_schema_attr(api, create_attr_name: str, value) -> bool:
        """Best-effort setter for generated USD/PhysX schema attributes."""
        create_attr = getattr(api, create_attr_name, None)
        if create_attr is None:
            return False
        try:
            attr = create_attr()
        except TypeError:
            attr = create_attr(value)
        attr.Set(value)
        return True

    def _define_physics_material(
        self,
        path: str,
        *,
        static_friction: float,
        dynamic_friction: float,
        restitution: float,
    ) -> UsdShade.Material:
        cfg = self.cfg.pre_grasp
        stage = omni.usd.get_context().get_stage()
        material = UsdShade.Material.Define(stage, Sdf.Path(path))
        prim = material.GetPrim()

        usd_mat = UsdPhysics.MaterialAPI.Apply(prim)
        self._set_schema_attr(
            usd_mat, "CreateStaticFrictionAttr", float(static_friction)
        )
        self._set_schema_attr(
            usd_mat, "CreateDynamicFrictionAttr", float(dynamic_friction)
        )
        self._set_schema_attr(
            usd_mat, "CreateRestitutionAttr", float(restitution)
        )

        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        self._set_schema_attr(
            physx_mat,
            "CreateFrictionCombineModeAttr",
            cfg.friction_combine_mode,
        )
        self._set_schema_attr(
            physx_mat,
            "CreateRestitutionCombineModeAttr",
            cfg.restitution_combine_mode,
        )
        return material

    def _bind_physics_material(
        self,
        prim: Usd.Prim,
        material: UsdShade.Material,
    ) -> None:
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        try:
            binding.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        except TypeError:
            try:
                binding.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )
            except TypeError:
                binding.Bind(material)

    def _tune_collision_contact(self, prim: Usd.Prim) -> bool:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            return False
        cfg = self.cfg.pre_grasp
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        touched = False
        touched |= self._set_schema_attr(
            physx_collision,
            "CreateContactOffsetAttr",
            float(cfg.contact_offset_m),
        )
        touched |= self._set_schema_attr(
            physx_collision,
            "CreateRestOffsetAttr",
            float(cfg.rest_offset_m),
        )
        return touched

    def _bind_material_tree(
        self,
        root_path: str,
        material: UsdShade.Material,
    ) -> tuple[int, int]:
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            warn(f"Grasp material bind skipped; missing prim: {root_path}")
            return 0, 0

        bound = 0
        contact_tuned = 0
        for prim in Usd.PrimRange(root):
            if not (
                prim.IsA(UsdGeom.Gprim)
                or prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                continue
            self._bind_physics_material(prim, material)
            bound += 1
            if self._tune_collision_contact(prim):
                contact_tuned += 1

        if bound == 0:
            self._bind_physics_material(root, material)
            bound = 1
        return bound, contact_tuned

    def _find_franka_finger_paths(self) -> list[str]:
        cfg = self.cfg.pre_grasp
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self.cfg.scene.franka_asset_path)
        if not root or not root.IsValid():
            return []

        wanted = set(cfg.finger_link_names)
        found: dict[str, str] = {}
        for prim in Usd.PrimRange(root):
            name = prim.GetName()
            if name in wanted and name not in found:
                found[name] = str(prim.GetPath())
        return [found[name] for name in cfg.finger_link_names if name in found]

    def _apply_grasp_physics_materials(self) -> None:
        """
        Bind realistic rubber-pad / hard-plastic PhysX materials for grasp.

        Values approximate rubber finger pads (μ≈0.85/0.65) on ABS/PVC plug
        plastic (μ≈0.45/0.35) with average friction combine (~0.6–0.7 μ_s).
        """
        cfg = self.cfg.pre_grasp
        finger_material = self._define_physics_material(
            cfg.finger_material_path,
            static_friction=cfg.finger_static_friction,
            dynamic_friction=cfg.finger_dynamic_friction,
            restitution=cfg.contact_restitution,
        )
        plug_material = self._define_physics_material(
            cfg.plug_material_path,
            static_friction=cfg.plug_static_friction,
            dynamic_friction=cfg.plug_dynamic_friction,
            restitution=cfg.contact_restitution,
        )

        finger_paths = self._find_franka_finger_paths()
        total_bound = 0
        total_contact = 0
        for path in finger_paths:
            bound, contact = self._bind_material_tree(path, finger_material)
            total_bound += bound
            total_contact += contact

        plug_bound, plug_contact = self._bind_material_tree(
            self.cfg.scene.tracked_connector_path,
            plug_material,
        )
        total_bound += plug_bound
        total_contact += plug_contact

        log(
            "GRASP PHYSICS MATERIALS\n"
            f"  fingers: μ_s={cfg.finger_static_friction:.2f}, "
            f"μ_d={cfg.finger_dynamic_friction:.2f} "
            f"(rubber pad)\n"
            f"  plug:    μ_s={cfg.plug_static_friction:.2f}, "
            f"μ_d={cfg.plug_dynamic_friction:.2f} "
            f"(hard plastic)\n"
            f"  combine: friction={cfg.friction_combine_mode}, "
            f"restitution={cfg.restitution_combine_mode}\n"
            f"  finger links: {finger_paths or '(none found)'}\n"
            f"  bound prims={total_bound}, "
            f"contact-tuned={total_contact}"
        )

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
        self._set_prim_local_scale(scene.cable_root_path, scene.cable_scale)
        if scene.cable_support_enabled:
            self._place_tracked_plug_on_support()
        else:
            self._place_cable_on_ground()

    def _load_datahall(self) -> None:
        """Place DataHall_Full_01 behind the robot/cable workspace in -X."""
        scene = self.cfg.scene
        if not scene.datahall_enabled:
            log("Data Hall load skipped (datahall_enabled=False)")
            return
        if not os.path.isfile(scene.datahall_usd_path):
            raise FileNotFoundError(
                f"Data Hall USD not found: {scene.datahall_usd_path}"
            )

        position = (
            scene.cable_spawn_xy[0] + scene.datahall_offset_from_cable_xy[0],
            scene.cable_spawn_xy[1] + scene.datahall_offset_from_cable_xy[1],
            0.0,
        )
        self._define_xform(
            scene.datahall_prim_path,
            position=position,
            yaw_deg=0.0,
            scale=(1.0, 1.0, 1.0),
        )
        self._add_reference(scene.datahall_usd_path, scene.datahall_prim_path)
        log(f"Data Hall loaded at {position}")

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

    def _set_prim_local_scale(
        self,
        prim_path: str,
        scale_factor: float,
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing prim: {prim_path}")

        value = Gf.Vec3d(scale_factor, scale_factor, scale_factor)
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()

        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(value)
                return

        # Keep scale as the innermost op (last in the order) so the
        # world translate applied during placement stays outermost and
        # isn't itself scaled. Ensure a translate op precedes it.
        has_translate = any(
            op.GetOpType() == UsdGeom.XformOp.TypeTranslate for op in ops
        )
        if not has_translate:
            xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(value)

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

