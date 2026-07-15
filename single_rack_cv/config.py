#!/usr/bin/env python3
"""All user-tunable settings for the single-rack RGB visual-servo demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    headless: bool = False
    width: int = 1600
    height: int = 900


@dataclass(frozen=True)
class SceneConfig:
    rack_usd_path: str = (
        "/home/aayush/isaacsim_assets/datacenter/"
        "single_server_rack_fixed.usda"
    )
    rack_path: str = "/World/ServerRack"
    rack_asset_path: str = "/World/ServerRack/Asset"
    rack_scale: float = 1.0
    rack_yaw_deg: float = 0.0

    franka_path: str = "/World/Franka"
    franka_asset_path: str = "/World/Franka/Robot"
    franka_position: tuple[float, float, float] = (1.35, 0.0, 1.0)
    franka_yaw_deg: float = 180.0

    physics_dt: float = 1.0 / 60.0
    device: str = "cpu"
    light_intensity: float = 1000.0

    viewport_eye: tuple[float, float, float] = (3.4, 3.2, 2.7)
    viewport_target: tuple[float, float, float] = (0.25, 0.0, 1.0)


@dataclass(frozen=True)
class CameraConfig:
    hand_link_name: str = "panda_hand"
    camera_name: str = "hand_camera"

    local_position: tuple[float, float, float] = (0.04, 0.0, 0.025)
    local_y_rotation_deg: float = 177.5
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
class PerceptionConfig:
    """RGB-only RJ45 cavity detection and known-size ranging."""

    roi_uv: tuple[int, int, int, int] | None = None
    max_gray: int = 60
    edge_margin_px: int = 4

    # The box grows as the eye-in-hand camera approaches the port.
    min_width_px: int = 12
    max_width_px: int = 80
    min_height_px: int = 10
    max_height_px: int = 70
    min_aspect_ratio: float = 0.65
    max_aspect_ratio: float = 1.80
    min_area_px: int = 100
    max_area_px: int = 5000
    min_fill_ratio: float = 0.35

    # RGB-only replacement for the old depth validation: the dark cavity
    # must sit inside a noticeably brighter bezel/surround.
    surround_ring_px: int = 6
    min_surround_mean_gray: float = 90.0
    min_surround_contrast_gray: float = 25.0

    target_aspect_ratio: float = 1.22
    target_fill_ratio: float = 0.63
    aspect_score_tolerance: float = 0.55
    fill_score_tolerance: float = 0.35
    aspect_score_weight: float = 0.70
    fill_score_weight: float = 0.30
    min_shape_score: float = 0.25

    # Calibrated dark-cavity dimensions for the rack asset. These are the
    # only metric assumptions used by the RGB-only range estimate.
    port_width_m: float = 0.0114
    port_height_m: float = 0.0070
    min_estimated_range_m: float = 0.08
    max_estimated_range_m: float = 0.35

    # Once acquired, prefer image continuity over an unrelated blob with a
    # slightly better single-frame shape score.
    tracking_max_center_jump_px: float = 45.0
    tracking_max_scale_ratio: float = 1.35
    tracking_center_penalty: float = 0.35
    tracking_scale_penalty: float = 0.25


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
    initial_position: tuple[float, float, float] = (
        0.9000,
        -0.1375,
        1.3000,
    )
    initial_orientation_wxyz: tuple[float, float, float, float] = (
        0.7071067811865476,
        0.0,
        -0.7071067811865475,
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
    """Continuous RGB feedback for translation-only pre-insert alignment."""

    enabled: bool = True
    preinsert_standoff_m: float = 0.050

    # Do not start perception while the arm/camera is still moving to its
    # fixed startup pose.
    startup_settle_tolerance_m: float = 0.0005
    required_startup_settled_frames: int = 15

    # Require a short stable image track before the robot is allowed to move.
    required_acquisition_samples: int = 3
    max_acquisition_center_spread_px: float = 8.0
    max_acquisition_scale_spread_ratio: float = 0.12
    max_consecutive_misses: int = 3

    # Stop-and-look visual servo: issue one small correction, wait until the
    # ToolCenter reaches that target, then capture the next control image.
    control_gain: float = 0.35
    max_target_step_m: float = 0.001
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
    estimated_port_marker_path: str = "/World/EstimatedPortPoint"
    estimated_port_marker_radius_m: float = 0.006
    estimated_port_marker_color: tuple[float, float, float] = (
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
