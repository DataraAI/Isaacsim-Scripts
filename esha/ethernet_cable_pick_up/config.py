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

    # Where the tracked connector should sit in XY.
    # +240 mm in X vs the original (0.5, 0).
    cable_spawn_xy: tuple[float, float] = (0.74, 0.0)
    # Gap between support surface and the cable/connector lowest point.
    ground_clearance: float = 0.002

    # Raise the tracked connector on a visible static block for grasp
    # clearance (pattern from network_cable_on_block_spawn.py). Soft
    # cable stays off; only the root xform is translated so the plug
    # bottom sits on the block top.
    cable_support_enabled: bool = True
    cable_support_path: str = "/World/CableSupportBlock"
    # Full edge lengths in meters (X, Y, Z). Y is the cross-cable width —
    # kept slightly under the ~22 mm connector/cable thickness so the plug
    # overhangs the pedestal a bit for finger access. Z is stand height.
    cable_support_size_m: tuple[float, float, float] = (
        0.080,
        0.018,
        0.040,
    )
    # Warm orange — clear against the gray ground, not blue.
    cable_support_color: tuple[float, float, float] = (
        0.85,
        0.45,
        0.15,
    )
    # Extra gap between plug bottom and block top.
    cable_support_plug_clearance_m: float = 0.004

    franka_path: str = "/World/Franka"
    franka_asset_path: str = "/World/Franka/Robot"
    franka_gripper_variant: str = "AlternateFinger"
    franka_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    franka_yaw_deg: float = 0.0

    physics_dt: float = 1.0 / 60.0
    # Soft-cable deformables require GPU PhysX. When True, sim upgrades
    # device=cpu → cuda and reloads the cable after GPU dynamics is on.
    # Expect more wobble during stereo servo than with rigid placement.
    device: str = "cpu"
    enable_gpu_dynamics: bool = True
    # Settle frames after play() so the soft wire can rest on the block.
    deformable_settle_frames: int = 60
    light_intensity: float = 1000.0
    # Plain, untextured ground color - the default grid texture confuses
    # the brightness-threshold detector, which expects a simple background.
    ground_plane_color: tuple[float, float, float] = (0.6, 0.6, 0.6)

    # Tuned for a tabletop-height pickup scene rather than a tall rack.
    viewport_eye: tuple[float, float, float] = (1.54, 1.1, 0.95)
    viewport_target: tuple[float, float, float] = (0.74, 0.0, 0.08)


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
    # Z includes cable_support_size_m height (0.04). Hand X matches
    # cable_spawn_xy (0.54) so the raised plug stays in the stereo FOV.
    initial_position: tuple[float, float, float] = (
        0.7200,
        0.1000,
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
    # Pre-grasp hold range along tool Z. Reduced from 0.120 for a slightly
    # closer final hold after visual alignment.
    grasp_standoff_m: float = 0.100

    # Do not start perception while the arm/camera is still moving to its
    # fixed startup pose. 0.5 mm works on CPU PhysX; GPU soft-cable dynamics
    # add enough Franka tracking noise that the counter never reaches
    # required_startup_settled_frames (stereo never starts).
    startup_settle_tolerance_m: float = 0.0005
    gpu_startup_settle_tolerance_m: float = 0.002
    # This originally only waited for the *arm* to stop moving (a quarter
    # second), never for the RTX render itself to converge to a clean,
    # low-noise image. On a long static hold, early frames can still be
    # noisy enough to break the detector even though the scene looks
    # visually fine by the time you look at it. Bumped way up (~5s at 60Hz)
    # to test whether waiting longer before the first capture fixes it.
    required_startup_settled_frames: int = 300
    # Log if startup settle stays incomplete this long.
    startup_settle_warn_s: float = 4.0

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
    gpu_target_settle_tolerance_m: float = 0.002

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
class PreGraspConfig:
    """After visual servo: angle standoff, open, grasp, close, then lift."""

    enabled: bool = True
    # Approach elevation above the horizontal plane (0=sideways, 90=top-down).
    grasp_elevation_deg: float = 30.0
    # Approach azimuth in world XY: 0=+X (from robot toward cable_spawn_xy).
    grasp_azimuth_deg: float = 0.0
    # One clear angled waypoint before opening and moving directly to grasp.
    approach_standoff_m: float = 0.120
    # Orientation settle tolerance while moving into the angled approach.
    orientation_settle_tolerance_rad: float = 0.05
    # Shift the stereo grasp point backward in world X without changing Y/Z.
    grasp_point_x_offset_m: float = -0.016 # change x offset to -0.016 to move the grasp point backwards
    # Tool-center clearance at grasp (before close).
    grasp_clearance_m: float = 0.0025
    # Commanded close half-gap. 0 = drive fully closed; contact stops the fingers.
    close_target_half_gap_m: float = 0.0
    # Post-grasp lift along +world Z.
    lift_z_m: float = 0.020
    # Extra open gap per side beyond measured cable half-thickness.
    side_allowance_m: float = 0.002
    # Never approach with less than this opening per finger.
    minimum_open_half_gap_m: float = 0.018
    # Fallback half of short-axis thickness if stereo height is unavailable.
    fallback_cable_half_width_m: float = 0.011
    finger_joint_names: tuple[str, str] = (
        "panda_finger_joint1",
        "panda_finger_joint2",
    )
    finger_link_names: tuple[str, str] = (
        "panda_leftfinger",
        "panda_rightfinger",
    )
    finger_max_open_m: float = 0.04
    # Increment IK_Target along a straight line during final approach.
    grasp_approach_step_m: float = 0.003
    block_safety_margin_m: float = 0.002
    # Minimum open hold; actual joint feedback must also confirm both fingers.
    open_hold_frames: int = 60
    open_timeout_frames: int = 180
    finger_open_target_tolerance_m: float = 0.00075
    # Finger joint motion must stay within this for finger_settle_frames.
    finger_settle_tolerance_m: float = 0.0005
    finger_settle_frames: int = 30
    # Reject a false "settled" close unless both fingers moved inward.
    close_min_travel_m: float = 0.001
    # Extra hold after fingers stop moving before lift.
    close_hold_frames: int = 30
    # Warn after this many frames, but never lift before close settles.
    close_timeout_frames: int = 180

    # Realistic rubber pad (fingers) / hard plastic (plug) contact.
    # PhysX combine=average → ~0.6–0.7 effective static friction.
    finger_static_friction: float = 0.85
    finger_dynamic_friction: float = 0.65
    plug_static_friction: float = 0.45
    plug_dynamic_friction: float = 0.35
    contact_restitution: float = 0.05
    friction_combine_mode: str = "average"
    restitution_combine_mode: str = "min"
    contact_offset_m: float = 0.002
    rest_offset_m: float = 0.0
    finger_material_path: str = "/World/Looks/FingerPadPhysicsMaterial"
    plug_material_path: str = "/World/Looks/PlugPlasticPhysicsMaterial"


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
    pre_grasp: PreGraspConfig = field(default_factory=PreGraspConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)



CONFIG = Config()

