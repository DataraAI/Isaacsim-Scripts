#!/usr/bin/env python3
"""
Cable-grasp RGB stereo demo: config, scene, perception glue, IK/servo
control, debug output, and the run loop, all in one file.

perception.py stays separate on purpose: it is pure NumPy/OpenCV math with
no Isaac Sim dependency, so it can be tested against saved images without
ever launching the simulator. Everything else lives here because none of
it has that same independent-testability benefit.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from perception import (
    CameraFrame,
    CameraModel,
    PerceptionConfig,
    CableCorners,
    CableDetection,
    StereoFrame,
    StereoCableObservation,
    build_virtual_camera_model,
    compute_bounded_step,
    compute_desired_cable_camera_usd,
    normalize_rgb,
    process_stereo_cable,
)


# =============================================================================
# Configuration (formerly config.py)
# =============================================================================

@dataclass(frozen=True)
class AppConfig:
    headless: bool = False
    width: int = 1600
    height: int = 900


@dataclass(frozen=True)
class SceneConfig:
    # Same network-cable asset and connector sub-prim used in the
    # detailedInsertion/cable pickup scripts, so ground-truth comparisons
    # (get_bbox on the connector) stay meaningful during development.
    cable_usd_path: str = (
        "/home/advaith/Isaacsim-assets/Network cable 001/"
        "model_Networkcable1_69323.usd"
    )
    cable_root_path: str = "/World/NetworkCable"
    tracked_connector_path: str = "/World/NetworkCable/E_crystal_head1_45"

    # Where the connector's bbox center should land on the ground plane.
    cable_spawn_xy: tuple[float, float] = (0.5, 0.0)
    ground_clearance: float = 0.002

    franka_path: str = "/World/Franka"
    franka_asset_path: str = "/World/Franka/Robot"
    franka_gripper_variant: str = "AlternateFinger"
    franka_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    franka_yaw_deg: float = 0.0

    physics_dt: float = 1.0 / 60.0
    device: str = "cpu"
    light_intensity: float = 1000.0
    # Plain, untextured ground color - the default grid texture confuses
    # the brightness-threshold detector, which expects a simple background.
    ground_plane_color: tuple[float, float, float] = (0.6, 0.6, 0.6)

    # Tuned for a tabletop-height pickup scene rather than a tall rack.
    viewport_eye: tuple[float, float, float] = (1.3, 1.1, 0.9)
    viewport_target: tuple[float, float, float] = (0.5, 0.0, 0.05)


@dataclass(frozen=True)
class CameraConfig:
    hand_link_name: str = "panda_hand"
    left_camera_name: str = "left_eye_camera"
    right_camera_name: str = "right_eye_camera"

    # REVERTED: zeroing this X offset (in an earlier attempt to "center"
    # the desired target) caused a regression - detection failed on every
    # frame instead of the ~20 that worked before. Restored to the
    # original working value. See the comment on grasp_standoff_m / the
    # servo logic for the corrected understanding of what "desired
    # off-center" actually means physically.
    left_local_position: tuple[float, float, float] = (
        0.04,
        -0.020,
        0.025,
    )
    right_local_position: tuple[float, float, float] = (
        0.04,
        +0.020,
        0.025,
    )
    # Mathematical midpoint only. No camera sensor is created here.
    virtual_local_position: tuple[float, float, float] = (
        0.04,
        0.0,
        0.025,
    )
    # y_rotation=180 points the view straight down (unchanged). roll=90 is
    # restored from the original value - it's what makes the physical
    # left/right baseline (along the hand's local Y) project onto the
    # camera's *horizontal* image axis instead of vertical. Zeroing it
    # earlier was a mistake: it left the baseline aligned with each
    # camera's vertical axis, producing ~0px horizontal disparity and a
    # huge (~150px) vertical mismatch between the two eyes - confirmed by
    # measuring the actual left/right detections against each other.
    local_y_rotation_deg: float = 180.0
    local_roll_deg: float = 90.0

    focal_length_mm: float = 18.0
    horizontal_aperture_mm: float = 20.955
    vertical_aperture_mm: float = 20.955 * 9.0 / 16.0
    clipping_range_m: tuple[float, float] = (0.01, 10.0)
    focus_distance_m: float = 1.0

    # CameraSensor uses (height, width). At 60 Hz physics, every 4 frames
    # matches the camera's 15 Hz tick rate.
    resolution: tuple[int, int] = (480, 640)
    tick_rate_hz: float = 15.0
    capture_every_sim_frames: int = 4

    output_dir: Path = Path(__file__).resolve().parent / "camera_output"


@dataclass(frozen=True)
class IKConfig:
    end_effector_frame: str = "panda_hand"

    target_path: str = "/World/IK_Target"
    target_name: str = "ik_target"
    target_scale: float = 0.08
    target_visible: bool = False
    select_target_on_start: bool = False

    actual_tool_path: str = "/World/ToolCenter"
    actual_tool_name: str = "tool_center"
    actual_tool_scale: float = 0.05
    actual_tool_visible: bool = False

    # Center between the fingers, 103.4 mm along panda_hand local +Z.
    tool_center_local_position_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.1034,
    )
    tool_center_local_orientation_wxyz: tuple[
        float,
        float,
        float,
        float,
    ] = (1.0, 0.0, 0.0, 0.0)

    use_fixed_start_pose: bool = True
    # Keep a detection-safe height with no lateral offset. Even 20 mm in Y
    # broke stereo epipolar matching (~12 px vertical mismatch vs a 3 px
    # gate). Visibility comes from the larger max_target_step_m instead.
    initial_position: tuple[float, float, float] = (
        0.5000,
        0.0000,
        0.2200,
    )
    initial_orientation_wxyz: tuple[float, float, float, float] = (
        0.0,
        1.0,
        0.0,
        0.0,
    )

    tracking_enabled: bool = True
    update_every_sim_frames: int = 1
    position_tolerance_m: float = 0.0001
    orientation_tolerance_rad: float = 0.01
    warn_every_sim_frames: int = 120


@dataclass(frozen=True)
class DriveTuningConfig:
    """Simulation accuracy settings for the seven Franka arm joints."""

    enabled: bool = True
    arm_joint_names: tuple[str, ...] = (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )
    stiffness_multiplier: float = 4.0
    damping_multiplier: float = 2.0
    disable_gravity_on_franka: bool = True


@dataclass(frozen=True)
class VisualServoConfig:
    """Continuous RGB feedback for translation-only pre-grasp alignment."""

    enabled: bool = True
    # Increased from 0.050: at the tighter standoff, the gripper geometry
    # (a fixed distance from the camera regardless of range-to-table) grew
    # to occupy enough of the frame to clip/merge with the detected blob
    # near the bottom edge - not a detector bug, just not enough clearance.
    # This gives the alignment phase more room before anything descends.
    grasp_standoff_m: float = 0.120

    # Do not start perception while the arm/camera is still moving to its
    # fixed startup pose.
    startup_settle_tolerance_m: float = 0.0005
    # This originally only waited for the *arm* to stop moving (a quarter
    # second), never for the RTX render itself to converge to a clean,
    # low-noise image. On a long static hold, early frames can still be
    # noisy enough to break the detector even though the scene looks
    # visually fine by the time you look at it. Bumped way up (~5s at 60Hz)
    # to test whether waiting longer before the first capture fixes it.
    required_startup_settled_frames: int = 300

    # Require a short stable image track before the robot is allowed to move.
    required_acquisition_samples: int = 3
    max_acquisition_center_spread_px: float = 8.0
    max_acquisition_scale_spread_ratio: float = 0.12
    max_consecutive_misses: int = 3

    # Stop-and-look visual servo: issue one small correction, wait until the
    # ToolCenter reaches that target, then capture the next control image.
    # 1 mm/step was too subtle to see in the viewport; 3 mm still keeps
    # tracking stable while making the approach obvious.
    control_gain: float = 0.35
    max_target_step_m: float = 0.003
    target_settle_tolerance_m: float = 0.0005

    # Visual alignment is intentionally looser than physical articulation
    # tracking because the detected box size is quantized in whole pixels.
    center_tolerance_px: float = 2.0
    range_tolerance_m: float = 0.003
    required_aligned_captures: int = 5

    # The measured physical tracking floor at this pose is about 0.22 mm.
    # A 0.30 mm gate remains far smaller than the RGB range uncertainty.
    settle_position_tolerance_m: float = 0.0003
    required_settled_frames: int = 30
    settle_warning_timeout_s: float = 8.0

    freeze_after_complete: bool = True


@dataclass(frozen=True)
class DebugConfig:
    estimated_cable_marker_path: str = "/World/EstimatedCablePoint"
    estimated_cable_marker_radius_m: float = 0.006
    estimated_cable_marker_color: tuple[float, float, float] = (
        0.0,
        1.0,
        0.0,
    )

    crosshair_half_length_px: int = 14
    crosshair_width_px: int = 2


@dataclass(frozen=True)
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    drive_tuning: DriveTuningConfig = field(
        default_factory=DriveTuningConfig
    )
    visual_servo: VisualServoConfig = field(
        default_factory=VisualServoConfig
    )
    debug: DebugConfig = field(default_factory=DebugConfig)



CONFIG = Config()


# =============================================================================
# Run-output tee: mirror stdout/stderr to the terminal AND a log file, at the
# OS file-descriptor level, so it also captures native Isaac/RTX output that
# never passes through Python's own sys.stdout object.
# =============================================================================


class RunOutputTee:
    """
    Mirror process stdout/stderr to the terminal and one overwrite-on-run file.

    This operates at the OS file-descriptor level, so it captures ordinary
    Python prints plus native Isaac/RTX output written directly to stdout or
    stderr.
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._log_fd: int | None = None
        self._pipe_read_fd: int | None = None
        self._pipe_write_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._started = False

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)

        while view:
            written = os.write(fd, view)

            if written <= 0:
                raise RuntimeError("Console tee write returned no progress.")

            view = view[written:]

    def _copy_output(self) -> None:
        if (
            self._pipe_read_fd is None
            or self._saved_stdout_fd is None
            or self._log_fd is None
        ):
            return

        try:
            while True:
                chunk = os.read(self._pipe_read_fd, 65536)

                if not chunk:
                    break

                self._write_all(self._saved_stdout_fd, chunk)
                self._write_all(self._log_fd, chunk)
        except OSError:
            # Shutdown may close descriptors while the reader is exiting.
            pass

    def start(self) -> None:
        if self._started:
            return

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sys.stdout.flush()
        sys.stderr.flush()

        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        self._log_fd = os.open(
            self.output_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
        )

        (
            self._pipe_read_fd,
            self._pipe_write_fd,
        ) = os.pipe()

        os.dup2(self._pipe_write_fd, 1)
        os.dup2(self._pipe_write_fd, 2)
        os.close(self._pipe_write_fd)
        self._pipe_write_fd = None

        self._thread = threading.Thread(
            target=self._copy_output,
            name="run-output-tee",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        sys.stdout.flush()
        sys.stderr.flush()

        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)

        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        descriptors = (
            self._pipe_read_fd,
            self._log_fd,
            self._saved_stdout_fd,
            self._saved_stderr_fd,
        )

        for fd in descriptors:
            if fd is None:
                continue

            try:
                os.close(fd)
            except OSError:
                pass

        self._pipe_read_fd = None
        self._log_fd = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None
        self._thread = None
        self._started = False


run_output_path = (
    CONFIG.camera.output_dir
    / "run_output_latest.txt"
)
run_output_tee = RunOutputTee(run_output_path)
run_output_tee.start()

simulation_app = None
runtime = None

try:
    print(
        f"[LOG] Saving complete run output to: {run_output_path}",
        flush=True,
    )

    # Isaac Sim must start before importing modules that use omni/pxr APIs.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": CONFIG.app.headless,
            "width": CONFIG.app.width,
            "height": CONFIG.app.height,
        }
    )

    import carb
    import numpy as np
    import omni.usd
    from PIL import Image, ImageDraw
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import DomeLight
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

    # =========================================================================
    # Debug outputs (formerly debug.py)
    # =========================================================================

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

        def save_failure_snapshot(
            self,
            frame: StereoFrame,
            capture_index: int,
            reason: str,
        ) -> None:
            """
            Keep one stable copy of the last cleanly-completed failure.

            "*_latest.png" is overwritten every capture, so a Ctrl+C right
            as the process dies can truncate it mid-write, leaving no
            trace of what the last real failure actually looked like.
            These files only get overwritten by this call (once per
            failure), never mid-interrupt, so they stay a reliable record.
            """
            Image.fromarray(frame.left.rgb, mode="RGB").save(
                self.output_dir / "rgb_left_last_failure.png"
            )
            Image.fromarray(frame.right.rgb, mode="RGB").save(
                self.output_dir / "rgb_right_last_failure.png"
            )
            (self.output_dir / "last_failure_reason.txt").write_text(
                f"capture={capture_index}\nreason={reason}\n"
            )

        def handle(
            self,
            frame: StereoFrame,
            observation: StereoCableObservation,
            capture_index: int,
        ) -> None:
            self.update_stage(observation)
            self.save_files(frame, observation)
            self.print_summary(observation, capture_index)

        def update_stage(self, observation: StereoCableObservation) -> None:
            cfg = self.cfg.debug
            self._update_sphere(
                cfg.estimated_cable_marker_path,
                observation.center_world_xyz_m,
                cfg.estimated_cable_marker_radius_m,
                cfg.estimated_cable_marker_color,
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
            observation: StereoCableObservation,
        ) -> None:
            left_overlay = self._draw_eye_overlay(
                frame.left.rgb,
                observation.left,
                observation.desired_left_center_uv,
                observation.desired_size_wh_px,
                "LEFT EYE",
            )
            right_overlay = self._draw_eye_overlay(
                frame.right.rgb,
                observation.right,
                observation.desired_right_center_uv,
                observation.desired_size_wh_px,
                "RIGHT EYE",
            )

            Image.fromarray(frame.left.rgb, mode="RGB").save(
                self.output_dir / "rgb_left_latest.png"
            )
            Image.fromarray(frame.right.rgb, mode="RGB").save(
                self.output_dir / "rgb_right_latest.png"
            )
            Image.fromarray(left_overlay, mode="RGB").save(
                self.output_dir / "rgb_left_cable_detected.png"
            )
            Image.fromarray(right_overlay, mode="RGB").save(
                self.output_dir / "rgb_right_cable_detected.png"
            )
            Image.fromarray(observation.left.detection.mask, mode="L").save(
                self.output_dir / "cable_detection_mask_left.png"
            )
            Image.fromarray(observation.right.detection.mask, mode="L").save(
                self.output_dir / "cable_detection_mask_right.png"
            )
            Image.fromarray(
                self._draw_stereo_summary(left_overlay, right_overlay, observation),
                mode="RGB",
            ).save(self.output_dir / "stereo_cable_detected.png")

        def _draw_eye_overlay(
            self,
            rgb: np.ndarray,
            cable: CableCorners,
            desired_center_uv: tuple[float, float],
            desired_size_wh_px: tuple[float, float],
            title: str,
        ) -> np.ndarray:
            image = Image.fromarray(rgb, mode="RGB")
            draw = ImageDraw.Draw(image)
            detection = cable.detection
            u0, v0, u1, v1 = detection.roi_uv
            x, y, width, height = detection.bbox_xywh
            draw.rectangle([u0, v0, u1 - 1, v1 - 1], outline=(0, 128, 255), width=1)
            draw.rectangle(
                [x, y, x + width - 1, y + height - 1],
                outline=(0, 255, 0),
                width=2,
            )
            corners = cable.corners_uv
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
            observation: StereoCableObservation,
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
                f"center_err={center_error:.2f}px"
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
            observation: StereoCableObservation,
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
                f"raw_correction={correction * 1000.0:.2f}mm",
                flush=True,
            )

    # =========================================================================
    # Isaac Sim scene, synchronized stereo RGB servo, and Lula IK
    # (formerly sim.py)
    # =========================================================================

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
    class VisualServoState:
        """Minimal state for RGB acquisition, tracking, and final settling."""

        startup_ready: bool = False
        startup_settled_frame_count: int = 0

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
                f"  cable:      {scene.cable_usd_path}\n"
                f"  connector:  {scene.tracked_connector_path}\n"
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

        def _load_cable(self) -> None:
            """Reference the network cable asset and drop it flat on the ground."""
            scene = self.cfg.scene
            stage = omni.usd.get_context().get_stage()

            if stage.GetPrimAtPath(scene.cable_root_path).IsValid():
                stage.RemovePrim(scene.cable_root_path)

            self._add_reference(scene.cable_usd_path, scene.cable_root_path)
            self._place_cable_on_ground()

        def _place_cable_on_ground(self) -> None:
            """
            Align the connector's bbox center over cable_spawn_xy, and drop the
            whole cable so its lowest point sits at ground_clearance above the
            ground plane. Mirrors place_cable_on_ground() from
            network_connector_pickup.py, adapted to this file's helper style.
            """
            scene = self.cfg.scene
            root_min, _ = self._world_bounds(scene.cable_root_path)
            connector_min, connector_max = self._world_bounds(
                scene.tracked_connector_path
            )
            connector_center = (connector_min + connector_max) / 2.0

            stage = omni.usd.get_context().get_stage()
            root_prim = stage.GetPrimAtPath(scene.cable_root_path)
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

            xform = UsdGeom.Xformable(root_prim)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*translation.tolist()))
                    self._update_app(10)
                    log(
                        "Cable placed: "
                        f"connector_center={np.round(connector_center, 4).tolist()}, "
                        f"translation={np.round(translation, 4).tolist()}"
                    )
                    return

            xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(*translation.tolist()))
            self._update_app(10)
            log(
                "Cable placed (new translate op): "
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

    # =========================================================================
    # Run loop (formerly main.py)
    # =========================================================================

    runtime = SimulationRuntime(
        simulation_app=simulation_app,
        cfg=CONFIG,
    )
    debug = DebugOutputs(CONFIG)

    capture_index = 0

    while runtime.is_running():
        runtime.step()

        try:
            runtime.update_ik()
            runtime.update_visual_servo_completion()
        except Exception as exc:
            warn(f"Motion/IK update failed: {exc}")

        if not runtime.capture_due():
            continue

        capture_index += 1

        try:
            frame = runtime.capture()
            debug.save_raw(frame)
            previous_left, previous_right = (
                runtime.visual_servo_references()
            )
            observation = process_stereo_cable(
                frame=frame,
                cfg=CONFIG.perception,
                desired_cable_virtual_camera_usd=(
                    runtime.desired_cable_virtual_camera_usd
                ),
                previous_left=previous_left,
                previous_right=previous_right,
            )
            runtime.observe_visual_servo(observation)
            debug.handle(
                frame,
                observation,
                capture_index,
            )
        except Exception as exc:
            # A rejected stereo pair holds the current target; repeated misses
            # trigger a clean image-space reacquisition.
            runtime.note_perception_failure()
            if "frame" in locals():
                try:
                    debug.save_failure_snapshot(frame, capture_index, str(exc))
                except Exception as snapshot_exc:
                    warn(f"Could not save failure snapshot: {snapshot_exc}")
            warn(
                f"RGB stereo capture {capture_index} skipped: {exc}"
            )

except Exception:
    print(
        "\n[CABLE GRASP RGB STEREO SERVO] FATAL ERROR\n"
        + traceback.format_exc(),
        flush=True,
    )
    raise

finally:
    try:
        if runtime is not None:
            runtime.stop()

        if simulation_app is not None:
            simulation_app.close()
    finally:
        print(
            f"[LOG] Run output saved to: {run_output_path}",
            flush=True,
        )
        run_output_tee.stop()