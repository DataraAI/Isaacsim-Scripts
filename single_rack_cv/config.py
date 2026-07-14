#!/usr/bin/env python3
"""All user-tunable settings for the single-rack RGB-D demo."""

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

    # CameraSensor uses (height, width).
    resolution: tuple[int, int] = (480, 640)
    tick_rate_hz: float = 15.0
    capture_every_sim_frames: int = 60

    output_dir: Path = Path(
        "/home/aayush/Isaacsim-Scripts/single_rack_cv/camera_output"
    )


@dataclass(frozen=True)
class PerceptionConfig:
    # (u_min, v_min, u_max_exclusive, v_max_exclusive)
    roi_uv: tuple[int, int, int, int] = (145, 220, 225, 290)
    max_gray: int = 60

    min_width_px: int = 12
    max_width_px: int = 45
    min_height_px: int = 12
    max_height_px: int = 40
    min_aspect_ratio: float = 0.65
    max_aspect_ratio: float = 1.80
    min_area_px: int = 100
    max_area_px: int = 1200
    min_fill_ratio: float = 0.35

    depth_patch_size_px: int = 11
    opening_ring_width_px: int = 4
    min_valid_ring_pixels: int = 80
    min_recess_depth_m: float = 0.002
    max_recess_depth_m: float = 0.050

    plane_mad_scale: float = 3.0
    plane_min_depth_tolerance_m: float = 0.0005
    plane_min_inlier_points: int = 100
    plane_max_rms_residual_m: float = 0.0015
    plane_max_camera_angle_deg: float = 35.0

    preinsert_standoff_m: float = 0.050


@dataclass(frozen=True)
class IKConfig:
    # Lula still solves this stable kinematic frame internally.
    end_effector_frame: str = "panda_hand"

    # The draggable target now represents the virtual center between the
    # fingers, not panda_hand itself.
    target_path: str = "/World/IK_Target"
    target_name: str = "ik_target"
    target_scale: float = 0.08

    # Visible frame showing the tool center actually achieved by the robot.
    actual_tool_path: str = "/World/ToolCenter"
    actual_tool_name: str = "tool_center"
    actual_tool_scale: float = 0.05

    # Nominal Franka hand TCP: 103.4 mm along panda_hand local +Z.
    # This is the grasp/insertion center between the two fingers.
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
    ] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )

    # These values remain the approved panda_hand observation pose.
    # The startup IK target is derived from this hand pose so the robot does
    # not jump when target semantics change from hand to tool center.
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
    position_tolerance_m: float = 0.001
    orientation_tolerance_rad: float = 0.01
    warn_every_sim_frames: int = 120


@dataclass(frozen=True)
class AutoPreinsertConfig:
    """Quality gates and timing for the one-shot automatic move."""

    enabled: bool = True
    required_stable_samples: int = 5
    max_sample_spread_m: float = 0.002

    min_recess_depth_m: float = 0.010
    max_recess_depth_m: float = 0.016
    max_plane_rms_m: float = 0.00075
    max_normal_angle_deg: float = 8.0

    move_duration_s: float = 3.0
    freeze_perception_after_latch: bool = True


@dataclass(frozen=True)
class DebugConfig:
    cavity_marker_path: str = "/World/DetectedPortPoint"
    cavity_marker_radius_m: float = 0.006
    cavity_marker_color: tuple[float, float, float] = (0.0, 1.0, 0.0)

    opening_marker_path: str = "/World/PortOpeningCenter"
    opening_marker_radius_m: float = 0.007
    opening_marker_color: tuple[float, float, float] = (0.0, 1.0, 1.0)

    normal_root_path: str = "/World/PortApproachNormal"
    normal_shaft_radius_m: float = 0.0025
    normal_tip_radius_m: float = 0.006
    normal_tip_length_m: float = 0.012
    normal_color: tuple[float, float, float] = (1.0, 0.5, 0.0)

    preinsert_marker_path: str = "/World/PreInsertPoint"
    preinsert_marker_radius_m: float = 0.008
    preinsert_marker_color: tuple[float, float, float] = (1.0, 0.0, 1.0)

    crosshair_half_length_px: int = 14
    crosshair_width_px: int = 2


@dataclass(frozen=True)
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    auto_preinsert: AutoPreinsertConfig = field(
        default_factory=AutoPreinsertConfig
    )
    debug: DebugConfig = field(default_factory=DebugConfig)


CONFIG = Config()
