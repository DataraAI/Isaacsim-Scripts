#!/usr/bin/env python3
"""Cable-grasp demo configuration (no Isaac Sim dependency)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from perception import PerceptionConfig


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
    # Soft-cable deformables require GPU dynamics. When enable_gpu_dynamics
    # is True and device is still "cpu", sim.py upgrades to "cuda".
    device: str = "cuda"
    enable_gpu_dynamics: bool = True
    # Extra settle frames after play so the deformable can rest on the
    # ground before the first stereo capture.
    deformable_settle_frames: int = 60
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

    # can incrementally change arm position to get a better view of the cable
    initial_position: tuple[float, float, float] = (
        0.4600,
        0.0000, #need to loosen/rework the epipolar matching
        0.3000,
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

