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
    device: str = "cuda:0"
    light_intensity: float = 1000.0

    viewport_eye: tuple[float, float, float] = (3.4, 3.2, 2.7)
    viewport_target: tuple[float, float, float] = (0.25, 0.0, 1.0)


@dataclass(frozen=True)
class CameraConfig:
    hand_link_name: str = "panda_hand"
    left_camera_name: str = "left_eye_camera"
    right_camera_name: str = "right_eye_camera"

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
    virtual_local_position: tuple[float, float, float] = (
        0.04,
        0.0,
        0.025,
    )
    local_y_rotation_deg: float = 177.5
    local_roll_deg: float = 90.0

    focal_length_mm: float = 18.0
    horizontal_aperture_mm: float = 20.955
    vertical_aperture_mm: float = 20.955 * 9.0 / 16.0
    clipping_range_m: tuple[float, float] = (0.01, 10.0)
    focus_distance_m: float = 1.0

    # CameraSensor uses (height, width).
    resolution: tuple[int, int] = (960, 1280)
    tick_rate_hz: float = 15.0
    capture_every_sim_frames: int = 4

    output_dir: Path = Path(__file__).resolve().parent / "camera_output"


@dataclass(frozen=True)
class YOLOEConfig:
    """One-time visual prompt plus full-frame stereo inference."""

    enabled: bool = True
    model_name: str = str(
        Path(__file__).resolve().parent
        / "assets"
        / "models"
        / "yoloe-26l-seg.pt"
    )
    reference_image_path: Path = (
        Path(__file__).resolve().parent
        / "assets"
        / "prompts"
        / "yoloe_reference_port_atlas.png"
    )

    # Production prompt selected by the qualified 60-frame benchmark.
    reference_boxes_xyxy: tuple[
        tuple[float, float, float, float],
        ...,
    ] = (
        (659.5, 107.25, 688.75, 131.0),
    )
    reference_class_ids: tuple[int, ...] = (0,)

    imgsz: int = 1280
    confidence: float = 0.005
    iou: float = 0.80
    device: int | str = 0
    quantize: int | str | None = 32
    max_detections: int = 100
    retina_masks: bool = False
    verbose: bool = False

    min_proposal_area_px: int = 36
    min_proposal_width_px: int = 6
    min_proposal_height_px: int = 6
    max_proposal_width_px: int = 180
    max_proposal_height_px: int = 150

    refine_expand_ratio: float = 0.35
    refine_min_margin_px: int = 6
    refine_percentiles: tuple[float, ...] = (
        15.0,
        20.0,
        25.0,
        30.0,
    )
    refine_min_gray: int = 30
    refine_max_gray: int = 95
    refine_min_width_px: int = 8
    refine_min_height_px: int = 8
    refine_max_width_px: int = 140
    refine_max_height_px: int = 110
    refine_min_aspect_ratio: float = 0.85
    refine_max_aspect_ratio: float = 2.10
    refine_target_aspect_ratio: float = 1.50
    refine_min_fill_ratio: float = 0.22
    refine_max_fill_ratio: float = 1.00
    refine_max_center_distance_ratio: float = 0.80
    refine_morph_kernel_px: int = 3


@dataclass(frozen=True)
class PerceptionConfig:
    """YOLOE instance pairing plus calibrated stereo geometry."""

    port_width_m: float = 0.0114
    port_height_m: float = 0.0070
    min_estimated_range_m: float = 0.08
    max_estimated_range_m: float = 0.35

    stereo_max_epipolar_error_px: float = 3.0
    stereo_max_scale_ratio: float = 1.30
    stereo_min_abs_disparity_px: float = 4.0
    stereo_max_ray_gap_m: float = 0.0020
    stereo_max_reprojection_rms_px: float = 1.0
    stereo_max_reprojection_px: float = 2.0
    stereo_min_width_m: float = 0.008
    stereo_max_width_m: float = 0.015
    stereo_min_height_m: float = 0.005
    stereo_max_height_m: float = 0.010

    tracking_max_center_jump_px: float = 45.0
    tracking_max_scale_ratio: float = 1.35


@dataclass(frozen=True)
class FrontPlaneRuntimeConfig:
    """Automatic front-opening geometry is mandatory in production."""

    enabled: bool = True


@dataclass(frozen=True)
class CableMountConfig:
    """Permanent direct mount for the cable's existing rigid RJ45 plug."""

    enabled: bool = True
    usd_path: str = (
        "/home/aayush/isaacsim_assets/Network cable 001/"
        "model_Networkcable1_69323.usd"
    )
    root_path: str = "/World/NetworkCable"
    tracked_plug_path: str = "/World/NetworkCable/E_crystal_head1_45"
    fixed_joint_path: str = "/World/CableMountFixedJoint"
    hand_link_name: str = "panda_hand"
    finger_link_names: tuple[str, str] = (
        "panda_leftfinger",
        "panda_rightfinger",
    )
    finger_joint_names: tuple[str, str] = (
        "panda_finger_joint1",
        "panda_finger_joint2",
    )
    axis_ratio_min: float = 1.5
    cable_projection_min_m: float = 0.002
    finger_total_clearance_m: float = 0.001
    initial_settle_frames: int = 60
    validation_frames: int = 30
    max_tip_error_m: float = 0.0005
    max_axis_error_deg: float = 1.0


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

    tool_center_local_position_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.1334,
    )
    tool_center_local_orientation_wxyz: tuple[
        float,
        float,
        float,
        float,
    ] = (
        0.7071067811865476,
        0.0,
        0.0,
        -0.7071067811865475,
    )

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
    """Translation-only stop-and-look pre-insert alignment."""

    enabled: bool = True
    preinsert_standoff_m: float = 0.050

    startup_settle_tolerance_m: float = 0.0005
    required_startup_settled_frames: int = 15

    required_acquisition_samples: int = 3
    max_acquisition_center_spread_px: float = 8.0
    max_acquisition_scale_spread_ratio: float = 0.12
    max_consecutive_misses: int = 3

    control_gain: float = 0.35
    max_target_step_m: float = 0.001
    target_settle_tolerance_m: float = 0.0005

    center_tolerance_px: float = 2.0
    range_tolerance_m: float = 0.003
    required_aligned_captures: int = 5

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
    yoloe: YOLOEConfig = field(default_factory=YOLOEConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    front_plane: FrontPlaneRuntimeConfig = field(
        default_factory=FrontPlaneRuntimeConfig
    )
    cable_mount: CableMountConfig = field(
        default_factory=CableMountConfig
    )
    ik: IKConfig = field(default_factory=IKConfig)
    drive_tuning: DriveTuningConfig = field(
        default_factory=DriveTuningConfig
    )
    visual_servo: VisualServoConfig = field(
        default_factory=VisualServoConfig
    )
    debug: DebugConfig = field(default_factory=DebugConfig)


CONFIG = Config()
