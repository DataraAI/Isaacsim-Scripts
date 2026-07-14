#!/usr/bin/env python3
"""
Single server rack + Franka scene for Isaac Sim 6.0.

This version uses the repaired wrapper asset:
    /home/aayush/isaacsim_assets/datacenter/single_server_rack_fixed.usda

Important:
- The repaired wrapper preserves the source asset's unit metadata.
- Isaac Sim already composes it at the correct real-world size.
- Therefore the rack container scale must remain 1.0.
- The rack is automatically centered in X/Y and placed on the ground.
- An RTX-authored USD camera is rigidly parented to the Franka panda_hand link.
- The hand camera uses an over-the-hand 18-degree oblique view toward the rack.
- Its POV remains editable while the script is running; no per-frame restore is applied.
- An RTX CameraSensor produces RGB and metric depth arrays from that camera.
- Latest RGB/depth outputs are saved under test/camera_output.
- A visible /World/IK_Target lets the user drag the Franka hand with Lula IK.
"""

from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------

RACK_USD_PATH = (
    "/home/aayush/isaacsim_assets/datacenter/"
    "single_server_rack_fixed.usda"
)

RACK_CONTAINER_PATH = "/World/ServerRack"
RACK_ASSET_PATH = f"{RACK_CONTAINER_PATH}/Asset"

# The repaired wrapper already preserves the source stage units.
# Isaac Sim composes it at the correct real-world size, so do NOT scale again.
RACK_SCALE = 1.0
RACK_YAW_DEG = 0.0

FRANKA_CONTAINER_PATH = "/World/Franka"
FRANKA_ASSET_PATH = f"{FRANKA_CONTAINER_PATH}/Robot"

# Initial placement after the rack is centered around the world origin.
# The rack depth is approximately 1.42 m, so this puts the Franka in
# front of the +X side with useful working clearance.
FRANKA_POSITION = (1.35, 0.0, 1)
FRANKA_YAW_DEG = 180.0

# ---------------------------------------------------------------------
# Eye-in-hand camera configuration
# ---------------------------------------------------------------------

HAND_LINK_NAME = "panda_hand"
HAND_CAMERA_NAME = "hand_camera"

# Over-the-hand eye-in-hand mount.
#
# Franka hand local axes used here:
#   +Z = forward through the gripper toward the work area
#   +X = above the gripper for this mount
#
# The camera is moved out of the palm/finger silhouette, but kept close enough
# to the hand to preserve a practical eye-in-hand viewpoint.
# This keeps the future cable connector near the lower part of the image while
# preserving a clear view of the rack port ahead.
HAND_CAMERA_LOCAL_POSITION = (0.04, 0.0, 0.025)

# USD cameras look along local -Z with +Y as image-up.
# 180 degrees about local Y would look exactly along hand +Z.
# Using 162 degrees adds an 18-degree downward tilt toward hand -X.
HAND_CAMERA_LOCAL_Y_ROTATION_DEG = 177.5
HAND_CAMERA_DOWNWARD_TILT_DEG = (
    180.0 - HAND_CAMERA_LOCAL_Y_ROTATION_DEG
)

# Roll the camera image 90 degrees clockwise so what currently appears on the
# left side of the image moves to the top.
HAND_CAMERA_LOCAL_ROLL_DEG = 90.0

HAND_CAMERA_FOCAL_LENGTH_MM = 18.0
HAND_CAMERA_HORIZONTAL_APERTURE_MM = 20.955
HAND_CAMERA_VERTICAL_APERTURE_MM = (
    HAND_CAMERA_HORIZONTAL_APERTURE_MM * 9.0 / 16.0
)
HAND_CAMERA_CLIPPING_RANGE_M = (0.01, 10.0)
HAND_CAMERA_FOCUS_DISTANCE_M = 1.0

# ---------------------------------------------------------------------
# RTX camera data configuration
# ---------------------------------------------------------------------

# CameraSensor resolution follows the OpenCV/NumPy convention:
# (height, width).
HAND_CAMERA_RESOLUTION = (480, 640)

# Render the hand camera at 15 Hz while physics continues at 60 Hz.
HAND_CAMERA_TICK_RATE_HZ = 15.0

# Read and save the latest RGB/depth frame about once per simulated second.
CAPTURE_EVERY_SIM_FRAMES = 60

# These files are overwritten each capture so this first test does not fill
# the disk with thousands of images.
CAMERA_OUTPUT_DIR = Path(
    "/home/aayush/Isaacsim-Scripts/test/camera_output"
)

# ---------------------------------------------------------------------
# First automatic one-port detector
# ---------------------------------------------------------------------

# Deliberately constrained search region around the single exposed Ethernet
# port in the approved observation view.
#
# Pixel convention:
#   U = image X coordinate / column
#   V = image Y coordinate / row
#
# Stored as (u_min, v_min, u_max_exclusive, v_max_exclusive).
PORT_DETECTION_ROI_UV = (
    145,
    220,
    225,
    290,
)

# The exposed RJ45 opening is one of the darkest compact components inside
# the ROI. This first detector thresholds grayscale and filters connected
# components by size, aspect ratio, and fill ratio.
PORT_DETECTION_MAX_GRAY = 60

PORT_DETECTION_MIN_WIDTH_PX = 12
PORT_DETECTION_MAX_WIDTH_PX = 45

PORT_DETECTION_MIN_HEIGHT_PX = 12
PORT_DETECTION_MAX_HEIGHT_PX = 40

PORT_DETECTION_MIN_ASPECT_RATIO = 0.65
PORT_DETECTION_MAX_ASPECT_RATIO = 1.80

PORT_DETECTION_MIN_AREA_PX = 100
PORT_DETECTION_MAX_AREA_PX = 1200

PORT_DETECTION_MIN_FILL_RATIO = 0.35

# Use an odd square patch so the detected center remains the exact center.
PORT_DEPTH_PATCH_SIZE_PX = 11

# Detection overlay appearance.
PORT_DETECTION_CROSSHAIR_HALF_LENGTH_PX = 14
PORT_DETECTION_CROSSHAIR_WIDTH_PX = 2

# ---------------------------------------------------------------------
# Reconstructed 3D port-point marker
# ---------------------------------------------------------------------

DETECTED_PORT_MARKER_PATH = "/World/DetectedPortPoint"

# Small enough not to hide the target, but large enough to see in the
# external viewport. The marker is authored with USD purpose=guide so it is
# intended as a debugging aid rather than scene geometry.
DETECTED_PORT_MARKER_RADIUS_M = 0.006

DETECTED_PORT_MARKER_COLOR = (
    0.0,
    1.0,
    0.0,
)

# ---------------------------------------------------------------------
# Front opening-plane estimate
# ---------------------------------------------------------------------

# Sample a rectangular ring immediately outside the detected dark opening.
# Those pixels belong to the visible front face/rim rather than the recessed
# socket interior.
PORT_OPENING_RING_WIDTH_PX = 4

# Reject obviously bad estimates instead of quietly creating a misleading
# insertion target.
PORT_OPENING_MIN_VALID_RING_PIXELS = 80
PORT_OPENING_MIN_RECESS_DEPTH_M = 0.002
PORT_OPENING_MAX_RECESS_DEPTH_M = 0.050

# The existing green marker remains the recessed socket-interior point.
# This cyan marker represents the center of the opening on the front plane.
PORT_OPENING_CENTER_MARKER_PATH = "/World/PortOpeningCenter"
PORT_OPENING_CENTER_MARKER_RADIUS_M = 0.007
PORT_OPENING_CENTER_MARKER_COLOR = (
    0.0,
    1.0,
    1.0,
)

# ---------------------------------------------------------------------
# Front-face plane normal and pre-insertion target
# ---------------------------------------------------------------------

# Robustly fit a plane to the same front-face depth ring used above.
PORT_PLANE_MAD_SCALE = 3.0
PORT_PLANE_MIN_DEPTH_TOLERANCE_M = 0.0005
PORT_PLANE_MIN_INLIER_POINTS = 100

# Fail loudly rather than trusting a visibly bad plane.
PORT_PLANE_MAX_RMS_RESIDUAL_M = 0.0015
PORT_PLANE_MAX_CAMERA_ANGLE_DEG = 35.0

# The normal is oriented outward, toward the camera. The pre-insertion point
# is placed exactly 50 mm in front of the opening along that outward normal.
PREINSERT_STANDOFF_M = 0.050

PORT_APPROACH_NORMAL_PATH = "/World/PortApproachNormal"
PORT_APPROACH_NORMAL_SHAFT_PATH = (
    f"{PORT_APPROACH_NORMAL_PATH}/Shaft"
)
PORT_APPROACH_NORMAL_TIP_PATH = (
    f"{PORT_APPROACH_NORMAL_PATH}/Tip"
)

PORT_APPROACH_NORMAL_SHAFT_RADIUS_M = 0.0025
PORT_APPROACH_NORMAL_TIP_RADIUS_M = 0.006
PORT_APPROACH_NORMAL_TIP_LENGTH_M = 0.012
PORT_APPROACH_NORMAL_COLOR = (
    1.0,
    0.5,
    0.0,
)

PREINSERT_POINT_PATH = "/World/PreInsertPoint"
PREINSERT_POINT_RADIUS_M = 0.008
PREINSERT_POINT_COLOR = (
    1.0,
    0.0,
    1.0,
)

# ---------------------------------------------------------------------
# Draggable Lula IK target configuration
# ---------------------------------------------------------------------

IK_TARGET_PATH = "/World/IK_Target"
IK_TARGET_NAME = "ik_target"
IK_END_EFFECTOR_FRAME = "panda_hand"

# Start the IK target at the manually approved pre-insertion observation pose.
# These values match the Transform panel shown in the screenshot:
#   Translate: X=0.9000, Y=-0.1375, Z=1.3000
#   Orient:    X=0 deg, Y=-90 deg, Z=0 deg
IK_TARGET_USE_FIXED_START_POSE = True
IK_TARGET_INITIAL_POSITION = (0.9000, -0.1375, 1.3000)
IK_TARGET_INITIAL_ORIENTATION_WXYZ = (
    0.7071067811865476,
    0.0,
    -0.7071067811865475,
    0.0,
)

# Size of the visible XYZ frame used as the draggable target.
IK_TARGET_SCALE = 0.08

# Solve every simulation frame while the target is being dragged.
IK_TRACKING_ENABLED = True
IK_UPDATE_EVERY_SIM_FRAMES = 1

# Tight enough for manual camera-pose setup without asking Lula for
# unnecessary sub-millimeter precision during interactive dragging.
IK_POSITION_TOLERANCE_M = 0.001
IK_ORIENTATION_TOLERANCE_RAD = 0.01

# Avoid flooding the console if the target is dragged outside the workspace.
IK_WARN_EVERY_SIM_FRAMES = 120

PHYSICS_DT = 1.0 / 60.0
DEVICE = "cpu"


if not os.path.isfile(RACK_USD_PATH):
    raise FileNotFoundError(
        "\n"
        "Repaired rack USD was not found:\n"
        f"  {RACK_USD_PATH}\n\n"
        "Run analyze_and_fix_rack_usd.py first."
    )


# ---------------------------------------------------------------------
# Start Isaac Sim before importing Omniverse APIs.
# ---------------------------------------------------------------------

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1600,
        "height": 900,
    }
)


try:
    import carb
    import cv2
    import numpy as np
    import omni.usd

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

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


    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    def log(message: str) -> None:
        print(f"[SINGLE RACK SCENE] {message}", flush=True)


    def warn(message: str) -> None:
        full = f"[SINGLE RACK SCENE] WARNING: {message}"
        print(full, flush=True)
        carb.log_warn(full)


    @dataclass
    class IKRuntime:
        """Objects and small bits of state needed for interactive IK."""

        articulation: Articulation
        target: XFormPrim
        kinematics_solver: LulaKinematicsSolver
        articulation_solver: ArticulationKinematicsSolver

        consecutive_failures: int = 0
        last_warning_frame: int = -1_000_000
        successful_solves: int = 0


    # -----------------------------------------------------------------
    # Transform helpers
    # -----------------------------------------------------------------

    def define_xform(
        prim_path: str,
        position: tuple[float, float, float],
        yaw_deg: float,
        scale: tuple[float, float, float],
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim(prim_path, "Xform")

        if not prim.IsValid():
            raise RuntimeError(f"Could not define Xform: {prim_path}")

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()

        xform.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(Gf.Vec3d(*position))

        yaw_rad = math.radians(yaw_deg)
        orientation = Gf.Quatd(
            math.cos(yaw_rad / 2.0),
            Gf.Vec3d(
                0.0,
                0.0,
                math.sin(yaw_rad / 2.0),
            ),
        )

        xform.AddOrientOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(orientation)

        xform.AddScaleOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(Gf.Vec3d(*scale))


    def set_translation(
        prim_path: str,
        position: tuple[float, float, float],
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            raise RuntimeError(f"Cannot move missing prim: {prim_path}")

        xform = UsdGeom.Xformable(prim)

        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*position))
                return

        raise RuntimeError(
            f"No translation op exists on {prim_path}"
        )


    # -----------------------------------------------------------------
    # Reference and geometry validation
    # -----------------------------------------------------------------

    def add_reference_checked(
        usd_path: str,
        prim_path: str,
    ) -> None:
        log(
            "Referencing:\n"
            f"  USD:  {usd_path}\n"
            f"  Prim: {prim_path}"
        )

        stage_utils.add_reference_to_stage(
            usd_path=usd_path,
            path=prim_path,
        )

        for _ in range(15):
            simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            raise RuntimeError(
                f"No valid prim composed at {prim_path}\n"
                f"Source: {usd_path}"
            )

        descendants = sum(
            1 for _ in Usd.PrimRange(prim)
        ) - 1

        log(
            f"Reference composed at {prim_path}\n"
            f"  descendants: {descendants}"
        )

        if descendants <= 0:
            raise RuntimeError(
                f"The reference at {prim_path} has no descendants."
            )


    def get_world_bounds(
        prim_path: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            return None

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
            useExtentsHint=True,
        )

        aligned = cache.ComputeWorldBound(
            prim
        ).ComputeAlignedRange()

        minimum = np.asarray(
            aligned.GetMin(),
            dtype=np.float64,
        )
        maximum = np.asarray(
            aligned.GetMax(),
            dtype=np.float64,
        )
        size = maximum - minimum

        if not np.all(np.isfinite(minimum)):
            return None
        if not np.all(np.isfinite(maximum)):
            return None
        if not np.all(np.isfinite(size)):
            return None
        if np.any(size < 0.0):
            return None
        if np.max(np.abs(minimum)) > 1.0e20:
            return None
        if np.max(np.abs(maximum)) > 1.0e20:
            return None

        return minimum, maximum, size


    def report_bounds(
        label: str,
        prim_path: str,
        required: bool = True,
    ):
        bounds = get_world_bounds(prim_path)

        if bounds is None:
            message = (
                f"No valid world bounds for {label}: "
                f"{prim_path}"
            )

            if required:
                raise RuntimeError(message)

            warn(message)
            return None

        minimum, maximum, size = bounds

        log(
            f"{label} world bounds:\n"
            f"  min  = {np.round(minimum, 4).tolist()}\n"
            f"  max  = {np.round(maximum, 4).tolist()}\n"
            f"  size = {np.round(size, 4).tolist()} meters"
        )

        return bounds


    # -----------------------------------------------------------------
    # Rack placement
    # -----------------------------------------------------------------

    def center_rack_on_ground() -> None:
        """
        Center the scaled rack at X=0, Y=0 and put its lowest point at Z=0.
        """
        initial_bounds = report_bounds(
            "Rack before centering",
            RACK_CONTAINER_PATH,
        )

        minimum, maximum, _ = initial_bounds
        center = (minimum + maximum) / 2.0

        translation = (
            float(-center[0]),
            float(-center[1]),
            float(-minimum[2]),
        )

        set_translation(
            RACK_CONTAINER_PATH,
            translation,
        )

        for _ in range(10):
            simulation_app.update()

        log(
            "Rack auto-centering translation:\n"
            f"  {np.round(np.array(translation), 4).tolist()}"
        )

        report_bounds(
            "Rack after centering",
            RACK_CONTAINER_PATH,
        )


    # -----------------------------------------------------------------
    # Robot and camera helpers
    # -----------------------------------------------------------------

    def find_articulation_roots(
        root_path: str,
    ) -> list[str]:
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_path)

        if not root.IsValid():
            return []

        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(
                UsdPhysics.ArticulationRootAPI
            )
        ]


    def find_unique_descendant_by_name(
        root_path: str,
        prim_name: str,
    ) -> str:
        """
        Find exactly one descendant by prim name.

        This avoids assuming the Franka asset hierarchy will never gain an
        extra wrapper Xform while still ensuring we attach to panda_hand.
        """
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_path)

        if not root.IsValid():
            raise RuntimeError(
                f"Cannot search below missing prim: {root_path}"
            )

        matches = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.GetName() == prim_name
        ]

        if not matches:
            available = [
                str(prim.GetPath())
                for prim in Usd.PrimRange(root)
                if "hand" in prim.GetName().lower()
            ]

            raise RuntimeError(
                f"Could not find a prim named '{prim_name}' below "
                f"{root_path}.\n"
                f"Hand-like prims found: {available}"
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"Expected one prim named '{prim_name}' below "
                f"{root_path}, but found:\n  "
                + "\n  ".join(matches)
            )

        return matches[0]


    def create_hand_camera() -> tuple[str, RtxCamera]:
        """
        Create the eye-in-hand camera through RtxCamera.

        Important:
        Do not create a plain UsdGeom.Camera first. On Isaac Sim 6.0,
        RtxCamera expects an existing prim to already carry OmniSensorAPI.
        Letting RtxCamera create the prim applies the required schema.
        """
        stage = omni.usd.get_context().get_stage()

        hand_path = find_unique_descendant_by_name(
            FRANKA_ASSET_PATH,
            HAND_LINK_NAME,
        )

        hand_prim = stage.GetPrimAtPath(hand_path)

        if hand_prim.IsInstanceProxy():
            raise RuntimeError(
                "The Franka hand is an instance proxy, so a child camera "
                "cannot be authored below it without de-instancing."
            )

        camera_path = f"{hand_path}/{HAND_CAMERA_NAME}"

        # RtxCamera expects quaternion order (w, x, y, z).
        # Compose the existing local Y rotation with a local optical-axis roll.
        y_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 1.0, 0.0),
            HAND_CAMERA_LOCAL_Y_ROTATION_DEG,
        ).GetQuat()

        roll_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 0.0, 1.0),
            HAND_CAMERA_LOCAL_ROLL_DEG,
        ).GetQuat()

        local_quat = y_quat * roll_quat
        imag = local_quat.GetImaginary()

        local_orientation_wxyz = np.array(
            [
                local_quat.GetReal(),
                imag[0],
                imag[1],
                imag[2],
            ],
            dtype=np.float64,
        )

        # Let RtxCamera create the Camera prim so OmniSensorAPI is applied
        # before CameraSensor tries to use the camera.
        rtx_camera = RtxCamera(
            path=camera_path,
            translations=np.array(
                HAND_CAMERA_LOCAL_POSITION,
                dtype=np.float64,
            ),
            orientations=local_orientation_wxyz,
            tick_rate=HAND_CAMERA_TICK_RATE_HZ,
        )

        for _ in range(5):
            simulation_app.update()

        camera_prim = stage.GetPrimAtPath(
            camera_path
        )

        if not camera_prim.IsValid():
            raise RuntimeError(
                f"RtxCamera did not create a valid prim at {camera_path}"
            )

        if not camera_prim.IsA(
            UsdGeom.Camera
        ):
            raise RuntimeError(
                f"Prim created at {camera_path} is not a USD Camera."
            )

        camera = UsdGeom.Camera(
            camera_prim
        )

        camera.CreateProjectionAttr().Set(
            UsdGeom.Tokens.perspective
        )

        camera.CreateFocalLengthAttr().Set(
            HAND_CAMERA_FOCAL_LENGTH_MM
        )

        camera.CreateHorizontalApertureAttr().Set(
            HAND_CAMERA_HORIZONTAL_APERTURE_MM
        )

        camera.CreateVerticalApertureAttr().Set(
            HAND_CAMERA_VERTICAL_APERTURE_MM
        )

        camera.CreateClippingRangeAttr().Set(
            Gf.Vec2f(*HAND_CAMERA_CLIPPING_RANGE_M)
        )

        camera.CreateFocusDistanceAttr().Set(
            HAND_CAMERA_FOCUS_DISTANCE_M
        )

        camera.CreateFStopAttr().Set(0.0)

        applied_schemas = list(
            camera_prim.GetAppliedSchemas()
        )

        log(
            "Hand RTX camera created:\n"
            f"  hand link:      {hand_path}\n"
            f"  camera prim:    {camera_path}\n"
            f"  local position: {HAND_CAMERA_LOCAL_POSITION}\n"
            f"  local Y rot:    "
            f"{HAND_CAMERA_LOCAL_Y_ROTATION_DEG} deg\n"
            f"  local Z roll:   "
            f"{HAND_CAMERA_LOCAL_ROLL_DEG} deg\n"
            f"  downward tilt:  "
            f"{HAND_CAMERA_DOWNWARD_TILT_DEG} deg\n"
            f"  focal length:   "
            f"{HAND_CAMERA_FOCAL_LENGTH_MM} mm\n"
            f"  clipping range: "
            f"{HAND_CAMERA_CLIPPING_RANGE_M} m\n"
            f"  applied schemas:{applied_schemas}\n"
            f"  POV editable:   True"
        )

        if not any(
            "OmniSensorAPI" in schema
            for schema in applied_schemas
        ):
            raise RuntimeError(
                "RtxCamera created the camera, but OmniSensorAPI was "
                "not found in the prim's applied schemas."
            )

        return camera_path, rtx_camera



    def verify_hand_camera_mount(
        camera_path: str,
    ) -> None:
        """
        Verify the configured over-the-hand oblique camera mount.

        The camera must:
        - remain parented directly to panda_hand,
        - match the configured local Y rotation,
        - point mostly along hand +Z,
        - tilt toward hand -X by the configured amount.
        """
        stage = omni.usd.get_context().get_stage()

        camera_prim = stage.GetPrimAtPath(camera_path)

        if not camera_prim.IsValid():
            raise RuntimeError(
                f"Missing camera prim during verification: {camera_path}"
            )

        hand_prim = camera_prim.GetParent()

        if not hand_prim.IsValid():
            raise RuntimeError(
                f"Hand camera has no valid parent: {camera_path}"
            )

        if hand_prim.GetName() != HAND_LINK_NAME:
            raise RuntimeError(
                "Hand camera is not parented directly to panda_hand. "
                f"Actual parent: {hand_prim.GetPath()}"
            )

        camera_world = UsdGeom.Xformable(
            camera_prim
        ).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        hand_world = UsdGeom.Xformable(
            hand_prim
        ).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        camera_position = camera_world.ExtractTranslation()

        camera_forward = camera_world.TransformDir(
            Gf.Vec3d(0.0, 0.0, -1.0)
        ).GetNormalized()

        hand_forward = hand_world.TransformDir(
            Gf.Vec3d(0.0, 0.0, 1.0)
        ).GetNormalized()

        hand_down = hand_world.TransformDir(
            Gf.Vec3d(-1.0, 0.0, 0.0)
        ).GetNormalized()

        theta_rad = math.radians(
            HAND_CAMERA_LOCAL_Y_ROTATION_DEG
        )

        expected_local_forward = Gf.Vec3d(
            -math.sin(theta_rad),
            0.0,
            -math.cos(theta_rad),
        )

        expected_world_forward = hand_world.TransformDir(
            expected_local_forward
        ).GetNormalized()

        expected_alignment = float(
            Gf.Dot(
                camera_forward,
                expected_world_forward,
            )
        )

        forward_alignment = float(
            Gf.Dot(
                camera_forward,
                hand_forward,
            )
        )

        downward_alignment = float(
            Gf.Dot(
                camera_forward,
                hand_down,
            )
        )

        clamped_forward_alignment = max(
            -1.0,
            min(1.0, forward_alignment),
        )

        measured_tilt_deg = math.degrees(
            math.acos(clamped_forward_alignment)
        )

        log(
            "Hand camera mount verification:\n"
            f"  world position:    "
            f"{np.round(np.array(camera_position), 4).tolist()}\n"
            f"  camera forward:    "
            f"{np.round(np.array(camera_forward), 4).tolist()}\n"
            f"  expected forward:  "
            f"{np.round(np.array(expected_world_forward), 4).tolist()}\n"
            f"  hand +Z forward:   "
            f"{np.round(np.array(hand_forward), 4).tolist()}\n"
            f"  hand -X downward:  "
            f"{np.round(np.array(hand_down), 4).tolist()}\n"
            f"  expected dot:      {expected_alignment:.6f}\n"
            f"  forward dot:       {forward_alignment:.6f}\n"
            f"  downward dot:      {downward_alignment:.6f}\n"
            f"  measured tilt:     {measured_tilt_deg:.3f} deg\n"
            f"  configured tilt:   "
            f"{HAND_CAMERA_DOWNWARD_TILT_DEG:.3f} deg"
        )

        if expected_alignment < 0.999:
            raise RuntimeError(
                "Hand camera does not match the configured oblique mount. "
                f"Expected-direction dot product: "
                f"{expected_alignment:.6f}"
            )

        if abs(
            measured_tilt_deg
            - HAND_CAMERA_DOWNWARD_TILT_DEG
        ) > 0.25:
            raise RuntimeError(
                "Hand camera tilt differs from its configuration. "
                f"Measured={measured_tilt_deg:.3f} deg, "
                f"configured={HAND_CAMERA_DOWNWARD_TILT_DEG:.3f} deg"
            )

        if downward_alignment <= 0.0:
            raise RuntimeError(
                "Hand camera is tilting upward instead of downward "
                "toward hand -X."
            )


    def create_hand_camera_sensor(
        camera_path: str,
        rtx_camera: RtxCamera,
    ) -> CameraSensor:
        """
        Create the runtime render product and RGB/depth annotators.

        The RtxCamera object is created earlier by create_hand_camera().
        Reusing the same object avoids re-wrapping a plain USD Camera prim.
        """
        log(
            "Creating RTX hand-camera sensor:\n"
            f"  camera prim: {camera_path}\n"
            f"  resolution:  {HAND_CAMERA_RESOLUTION}\n"
            f"  tick rate:   {HAND_CAMERA_TICK_RATE_HZ} Hz\n"
            "  annotators:  rgb, distance_to_image_plane"
        )

        sensor = CameraSensor(
            rtx_camera,
            resolution=HAND_CAMERA_RESOLUTION,
            annotators=[
                "rgb",
                "distance_to_image_plane",
            ],
        )

        CAMERA_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        log(
            "RTX hand-camera sensor created.\n"
            f"  output dir: {CAMERA_OUTPUT_DIR}"
        )

        return sensor



    def sensor_array_to_numpy(
        value,
        label: str,
    ) -> np.ndarray:
        """
        Convert a CameraSensor annotator result into a detached NumPy array.

        Isaac Sim 6.0 returns Warp arrays from CameraSensor.get_data().
        Warp arrays expose .numpy(); the copy keeps the frame valid after the
        renderer updates its internal buffers.
        """
        if value is None:
            raise RuntimeError(
                f"{label} annotator returned None."
            )

        if hasattr(value, "numpy"):
            array = np.array(
                value.numpy(),
                copy=True,
            )
        else:
            array = np.array(
                value,
                copy=True,
            )

        if array.size == 0:
            raise RuntimeError(
                f"{label} annotator returned an empty array."
            )

        return array


    def reshape_rgb(
        rgb: np.ndarray,
    ) -> np.ndarray:
        """Normalize RGB annotator output to H x W x 3 uint8."""
        height, width = HAND_CAMERA_RESOLUTION

        if rgb.ndim == 1:
            pixels = width * height

            if rgb.size == pixels * 4:
                rgb = rgb.reshape(
                    height,
                    width,
                    4,
                )
            elif rgb.size == pixels * 3:
                rgb = rgb.reshape(
                    height,
                    width,
                    3,
                )
            else:
                raise RuntimeError(
                    "Cannot reshape flat RGB data. "
                    f"size={rgb.size}, expected "
                    f"{pixels * 3} or {pixels * 4}."
                )

        if rgb.ndim != 3:
            raise RuntimeError(
                f"RGB data must be 3D, got shape {rgb.shape}."
            )

        if rgb.shape[0] != height or rgb.shape[1] != width:
            raise RuntimeError(
                "RGB resolution mismatch. "
                f"got={rgb.shape}, expected height={height}, width={width}."
            )

        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        elif rgb.shape[2] != 3:
            raise RuntimeError(
                f"RGB channel count must be 3 or 4, got {rgb.shape[2]}."
            )

        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                max_value = float(np.nanmax(rgb))
                if max_value <= 1.0:
                    rgb = rgb * 255.0

            rgb = np.clip(
                rgb,
                0,
                255,
            ).astype(np.uint8)

        return np.ascontiguousarray(rgb)


    def reshape_depth(
        depth: np.ndarray,
    ) -> np.ndarray:
        """Normalize depth annotator output to H x W float32 meters."""
        height, width = HAND_CAMERA_RESOLUTION

        depth = np.squeeze(depth)

        if depth.ndim == 1:
            expected = width * height

            if depth.size != expected:
                raise RuntimeError(
                    "Cannot reshape flat depth data. "
                    f"size={depth.size}, expected={expected}."
                )

            depth = depth.reshape(
                height,
                width,
            )

        if depth.ndim != 2:
            raise RuntimeError(
                f"Depth data must be 2D, got shape {depth.shape}."
            )

        if depth.shape != (height, width):
            raise RuntimeError(
                "Depth resolution mismatch. "
                f"got={depth.shape}, expected={(height, width)}."
            )

        return np.ascontiguousarray(
            depth.astype(
                np.float32,
                copy=False,
            )
        )


    def make_depth_preview(
        depth_m: np.ndarray,
    ) -> tuple[np.ndarray, float, float, int]:
        """
        Convert metric depth into an 8-bit preview image.

        Raw metric depth is saved separately as .npy. The preview uses the
        2nd-98th percentile range so a few extreme pixels do not flatten the
        contrast.
        """
        valid_mask = (
            np.isfinite(depth_m)
            & (depth_m > 0.0)
        )

        valid_count = int(
            np.count_nonzero(valid_mask)
        )

        if valid_count == 0:
            raise RuntimeError(
                "Depth frame contains no finite positive values."
            )

        valid_values = depth_m[valid_mask]

        near_m = float(
            np.percentile(
                valid_values,
                2.0,
            )
        )

        far_m = float(
            np.percentile(
                valid_values,
                98.0,
            )
        )

        if far_m <= near_m:
            far_m = near_m + 1.0e-6

        normalized = (
            depth_m - near_m
        ) / (
            far_m - near_m
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        # Near objects are bright; far objects are dark.
        preview = (
            (1.0 - normalized)
            * 255.0
        ).astype(np.uint8)

        preview[~valid_mask] = 0

        return (
            preview,
            near_m,
            far_m,
            valid_count,
        )


    def detect_single_ethernet_port(
        rgb: np.ndarray,
    ) -> dict[str, object]:
        """
        Detect the single exposed Ethernet port inside a constrained ROI.

        This is intentionally a first-stage detector, not a general port
        detector. It assumes the approved observation pose and one visible
        target port.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(
                "Port detector expects H x W x 3 RGB data, "
                f"got shape {rgb.shape}."
            )

        image_height, image_width = rgb.shape[:2]

        roi_u0, roi_v0, roi_u1, roi_v1 = (
            int(value)
            for value in PORT_DETECTION_ROI_UV
        )

        if not (
            0 <= roi_u0 < roi_u1 <= image_width
            and 0 <= roi_v0 < roi_v1 <= image_height
        ):
            raise RuntimeError(
                "PORT_DETECTION_ROI_UV is outside the RGB image. "
                f"roi={PORT_DETECTION_ROI_UV}, "
                f"image=(width={image_width}, height={image_height})."
            )

        roi_rgb = rgb[
            roi_v0:roi_v1,
            roi_u0:roi_u1,
        ]

        roi_gray = cv2.cvtColor(
            roi_rgb,
            cv2.COLOR_RGB2GRAY,
        )

        binary_mask = cv2.inRange(
            roi_gray,
            0,
            int(PORT_DETECTION_MAX_GRAY),
        )

        (
            component_count,
            _labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )

        candidates: list[
            dict[str, object]
        ] = []

        for component_index in range(
            1,
            component_count,
        ):
            (
                local_x,
                local_y,
                width,
                height,
                area,
            ) = (
                int(value)
                for value in stats[
                    component_index
                ]
            )

            if width <= 0 or height <= 0:
                continue

            aspect_ratio = (
                float(width)
                / float(height)
            )

            fill_ratio = (
                float(area)
                / float(width * height)
            )

            if not (
                PORT_DETECTION_MIN_WIDTH_PX
                <= width
                <= PORT_DETECTION_MAX_WIDTH_PX
            ):
                continue

            if not (
                PORT_DETECTION_MIN_HEIGHT_PX
                <= height
                <= PORT_DETECTION_MAX_HEIGHT_PX
            ):
                continue

            if not (
                PORT_DETECTION_MIN_ASPECT_RATIO
                <= aspect_ratio
                <= PORT_DETECTION_MAX_ASPECT_RATIO
            ):
                continue

            if not (
                PORT_DETECTION_MIN_AREA_PX
                <= area
                <= PORT_DETECTION_MAX_AREA_PX
            ):
                continue

            if (
                fill_ratio
                < PORT_DETECTION_MIN_FILL_RATIO
            ):
                continue

            global_x = (
                roi_u0
                + local_x
            )

            global_y = (
                roi_v0
                + local_y
            )

            center_u = (
                global_x
                + width // 2
            )

            center_v = (
                global_y
                + height // 2
            )

            # Prefer a component with both substantial dark area and a
            # substantial fraction of its bounding box filled.
            score = (
                float(area)
                * fill_ratio
            )

            candidates.append(
                {
                    "score": score,
                    "component_index": component_index,
                    "bbox_xywh": (
                        global_x,
                        global_y,
                        width,
                        height,
                    ),
                    "center_uv": (
                        center_u,
                        center_v,
                    ),
                    "area_px": area,
                    "aspect_ratio": aspect_ratio,
                    "fill_ratio": fill_ratio,
                    "local_centroid_xy": (
                        float(
                            centroids[
                                component_index
                            ][0]
                        ),
                        float(
                            centroids[
                                component_index
                            ][1]
                        ),
                    ),
                }
            )

        if not candidates:
            raise RuntimeError(
                "No Ethernet-port candidate passed the configured filters.\n"
                f"  ROI: {PORT_DETECTION_ROI_UV}\n"
                f"  grayscale threshold: <= "
                f"{PORT_DETECTION_MAX_GRAY}\n"
                "Inspect port_detection_mask.png and then adjust one "
                "detector setting at a time."
            )

        candidates.sort(
            key=lambda candidate: float(
                candidate["score"]
            ),
            reverse=True,
        )

        best = dict(
            candidates[0]
        )

        full_mask = np.zeros(
            (
                image_height,
                image_width,
            ),
            dtype=np.uint8,
        )

        full_mask[
            roi_v0:roi_v1,
            roi_u0:roi_u1,
        ] = binary_mask

        best.update(
            {
                "roi_uv": (
                    roi_u0,
                    roi_v0,
                    roi_u1,
                    roi_v1,
                ),
                "candidate_count": len(
                    candidates
                ),
                "mask": full_mask,
            }
        )

        return best


    def compute_detection_depth(
        depth_m: np.ndarray,
        center_uv: tuple[int, int],
    ) -> dict[str, object]:
        """
        Read robust metric depth around the automatically detected center.
        """
        if depth_m.ndim != 2:
            raise RuntimeError(
                "Port depth lookup expects a 2D depth image, "
                f"got shape {depth_m.shape}."
            )

        height, width = depth_m.shape

        u = int(
            center_uv[0]
        )

        v = int(
            center_uv[1]
        )

        if not (
            0 <= u < width
            and 0 <= v < height
        ):
            raise RuntimeError(
                "Detected port center is outside the depth image. "
                f"center=(u={u}, v={v}), "
                f"image=(width={width}, height={height})."
            )

        patch_size = int(
            PORT_DEPTH_PATCH_SIZE_PX
        )

        if (
            patch_size <= 0
            or patch_size % 2 == 0
        ):
            raise RuntimeError(
                "PORT_DEPTH_PATCH_SIZE_PX must be a positive odd integer, "
                f"got {patch_size}."
            )

        half = patch_size // 2

        u0 = max(
            0,
            u - half,
        )

        u1 = min(
            width,
            u + half + 1,
        )

        v0 = max(
            0,
            v - half,
        )

        v1 = min(
            height,
            v + half + 1,
        )

        patch = depth_m[
            v0:v1,
            u0:u1,
        ]

        valid_mask = (
            np.isfinite(
                patch
            )
            & (
                patch > 0.0
            )
        )

        valid_values = patch[
            valid_mask
        ]

        if valid_values.size == 0:
            raise RuntimeError(
                "The detected port depth patch contains no finite "
                "positive values."
            )

        return {
            "center_uv": (
                u,
                v,
            ),
            "patch_bounds_uv": (
                u0,
                v0,
                u1,
                v1,
            ),
            "patch_shape": tuple(
                int(value)
                for value in patch.shape
            ),
            "valid_count": int(
                valid_values.size
            ),
            "center_depth_m": float(
                depth_m[
                    v,
                    u,
                ]
            ),
            "median_depth_m": float(
                np.median(
                    valid_values
                )
            ),
            "minimum_depth_m": float(
                np.min(
                    valid_values
                )
            ),
            "maximum_depth_m": float(
                np.max(
                    valid_values
                )
            ),
        }


    def estimate_port_opening_plane_depth(
        depth_m: np.ndarray,
        bbox_xywh: tuple[int, int, int, int],
        cavity_depth_m: float,
    ) -> dict[str, object]:
        """
        Estimate the port's front opening plane from a depth ring around it.

        The detector box encloses the dark recessed opening. A narrow ring just
        outside that box lands on the visible front face/rim. The median ring
        depth is therefore a robust first estimate of the insertion plane.
        """
        if depth_m.ndim != 2:
            raise RuntimeError(
                "Opening-plane estimation expects a 2D depth image, "
                f"got shape {depth_m.shape}."
            )

        image_height, image_width = depth_m.shape

        x, y, width, height = (
            int(value)
            for value in bbox_xywh
        )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"Invalid detected port box: {bbox_xywh}"
            )

        if not (
            0 <= x < image_width
            and 0 <= y < image_height
            and x + width <= image_width
            and y + height <= image_height
        ):
            raise RuntimeError(
                "Detected port box is outside the depth image. "
                f"bbox={bbox_xywh}, "
                f"image=(width={image_width}, height={image_height})."
            )

        ring_width = int(
            PORT_OPENING_RING_WIDTH_PX
        )

        if ring_width <= 0:
            raise RuntimeError(
                "PORT_OPENING_RING_WIDTH_PX must be positive, "
                f"got {ring_width}."
            )

        outer_x0 = max(0, x - ring_width)
        outer_y0 = max(0, y - ring_width)
        outer_x1 = min(image_width, x + width + ring_width)
        outer_y1 = min(image_height, y + height + ring_width)

        outer_patch = depth_m[
            outer_y0:outer_y1,
            outer_x0:outer_x1,
        ]

        ring_mask = np.ones(
            outer_patch.shape,
            dtype=bool,
        )

        inner_x0 = x - outer_x0
        inner_y0 = y - outer_y0
        inner_x1 = inner_x0 + width
        inner_y1 = inner_y0 + height

        ring_mask[
            inner_y0:inner_y1,
            inner_x0:inner_x1,
        ] = False

        valid_mask = (
            ring_mask
            & np.isfinite(outer_patch)
            & (outer_patch > 0.0)
        )

        ring_values = outer_patch[valid_mask]
        valid_count = int(ring_values.size)

        if valid_count < PORT_OPENING_MIN_VALID_RING_PIXELS:
            raise RuntimeError(
                "Too few valid depth pixels around the port opening. "
                f"valid={valid_count}, "
                f"required={PORT_OPENING_MIN_VALID_RING_PIXELS}."
            )

        opening_plane_depth_m = float(
            np.median(ring_values)
        )

        cavity_depth_m = float(cavity_depth_m)

        if not np.isfinite(cavity_depth_m) or cavity_depth_m <= 0.0:
            raise RuntimeError(
                f"Invalid cavity depth: {cavity_depth_m}"
            )

        recess_depth_m = cavity_depth_m - opening_plane_depth_m

        if not (
            PORT_OPENING_MIN_RECESS_DEPTH_M
            <= recess_depth_m
            <= PORT_OPENING_MAX_RECESS_DEPTH_M
        ):
            raise RuntimeError(
                "Opening-plane estimate produced an implausible recess depth.\n"
                f"  opening plane: {opening_plane_depth_m:.6f} m\n"
                f"  cavity depth:  {cavity_depth_m:.6f} m\n"
                f"  recess depth:  {recess_depth_m:.6f} m\n"
                f"  expected range: "
                f"{PORT_OPENING_MIN_RECESS_DEPTH_M:.6f} to "
                f"{PORT_OPENING_MAX_RECESS_DEPTH_M:.6f} m"
            )

        return {
            "outer_bounds_xyxy": (
                outer_x0,
                outer_y0,
                outer_x1,
                outer_y1,
            ),
            "inner_bounds_xyxy": (
                x,
                y,
                x + width,
                y + height,
            ),
            "ring_width_px": ring_width,
            "valid_count": valid_count,
            "opening_plane_depth_m": opening_plane_depth_m,
            "cavity_depth_m": cavity_depth_m,
            "recess_depth_m": float(recess_depth_m),
            "minimum_ring_depth_m": float(np.min(ring_values)),
            "maximum_ring_depth_m": float(np.max(ring_values)),
        }


    def fit_port_opening_plane_normal(
        depth_m: np.ndarray,
        bbox_xywh: tuple[int, int, int, int],
        intrinsics: dict[str, float],
    ) -> dict[str, object]:
        """
        Fit a 3D plane to the front-face ring around the detected port.

        The returned normal is oriented outward, toward the camera.

        Coordinate conventions:
            OpenCV camera:
                +X = image right
                +Y = image down
                +Z = forward into the scene

            USD camera-local:
                +X = image right
                +Y = image up
                -Z = forward into the scene

        Therefore an outward normal generally has negative Z in OpenCV and
        positive Z in USD camera-local coordinates.
        """
        if depth_m.ndim != 2:
            raise RuntimeError(
                "Plane fitting expects a 2D depth image, "
                f"got shape {depth_m.shape}."
            )

        image_height, image_width = depth_m.shape

        x, y, width, height = (
            int(value)
            for value in bbox_xywh
        )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"Invalid detected port box for plane fit: {bbox_xywh}"
            )

        if not (
            0 <= x < image_width
            and 0 <= y < image_height
            and x + width <= image_width
            and y + height <= image_height
        ):
            raise RuntimeError(
                "Detected port box is outside the depth image during "
                f"plane fitting: bbox={bbox_xywh}, "
                f"image=(width={image_width}, height={image_height})."
            )

        ring_width = int(
            PORT_OPENING_RING_WIDTH_PX
        )

        outer_x0 = max(
            0,
            x - ring_width,
        )

        outer_y0 = max(
            0,
            y - ring_width,
        )

        outer_x1 = min(
            image_width,
            x + width + ring_width,
        )

        outer_y1 = min(
            image_height,
            y + height + ring_width,
        )

        patch = depth_m[
            outer_y0:outer_y1,
            outer_x0:outer_x1,
        ]

        ring_mask = np.ones(
            patch.shape,
            dtype=bool,
        )

        inner_x0 = x - outer_x0
        inner_y0 = y - outer_y0
        inner_x1 = inner_x0 + width
        inner_y1 = inner_y0 + height

        ring_mask[
            inner_y0:inner_y1,
            inner_x0:inner_x1,
        ] = False

        local_v, local_u = np.indices(
            patch.shape
        )

        u_all = (
            local_u
            + outer_x0
        )[ring_mask]

        v_all = (
            local_v
            + outer_y0
        )[ring_mask]

        z_all = patch[
            ring_mask
        ].astype(
            np.float64
        )

        valid = (
            np.isfinite(
                z_all
            )
            & (
                z_all > 0.0
            )
        )

        u_valid = u_all[
            valid
        ].astype(
            np.float64
        )

        v_valid = v_all[
            valid
        ].astype(
            np.float64
        )

        z_valid = z_all[
            valid
        ]

        if (
            z_valid.size
            < PORT_PLANE_MIN_INLIER_POINTS
        ):
            raise RuntimeError(
                "Too few valid front-face ring samples for plane fit. "
                f"valid={z_valid.size}, "
                f"required={PORT_PLANE_MIN_INLIER_POINTS}."
            )

        depth_median = float(
            np.median(
                z_valid
            )
        )

        depth_mad = float(
            np.median(
                np.abs(
                    z_valid
                    - depth_median
                )
            )
        )

        robust_sigma = (
            1.4826
            * depth_mad
        )

        depth_tolerance = max(
            float(
                PORT_PLANE_MIN_DEPTH_TOLERANCE_M
            ),
            float(
                PORT_PLANE_MAD_SCALE
            )
            * robust_sigma,
        )

        inlier_mask = (
            np.abs(
                z_valid
                - depth_median
            )
            <= depth_tolerance
        )

        u = u_valid[
            inlier_mask
        ]

        v = v_valid[
            inlier_mask
        ]

        z = z_valid[
            inlier_mask
        ]

        inlier_count = int(
            z.size
        )

        if (
            inlier_count
            < PORT_PLANE_MIN_INLIER_POINTS
        ):
            raise RuntimeError(
                "Too few depth inliers remain after robust filtering. "
                f"inliers={inlier_count}, "
                f"required={PORT_PLANE_MIN_INLIER_POINTS}, "
                f"median={depth_median:.6f} m, "
                f"MAD={depth_mad:.6f} m, "
                f"tolerance={depth_tolerance:.6f} m."
            )

        fx_px = float(
            intrinsics["fx_px"]
        )

        fy_px = float(
            intrinsics["fy_px"]
        )

        cx_px = float(
            intrinsics["cx_px"]
        )

        cy_px = float(
            intrinsics["cy_px"]
        )

        x_cv = (
            (u - cx_px)
            * z
            / fx_px
        )

        y_cv = (
            (v - cy_px)
            * z
            / fy_px
        )

        points_cv = np.column_stack(
            (
                x_cv,
                y_cv,
                z,
            )
        ).astype(
            np.float64
        )

        centroid_cv = np.mean(
            points_cv,
            axis=0,
        )

        centered = (
            points_cv
            - centroid_cv
        )

        _u_svd, singular_values, vh = (
            np.linalg.svd(
                centered,
                full_matrices=False,
            )
        )

        normal_cv = vh[
            -1
        ].astype(
            np.float64
        )

        normal_norm = float(
            np.linalg.norm(
                normal_cv
            )
        )

        if normal_norm <= 1.0e-12:
            raise RuntimeError(
                "Plane fit returned a zero-length normal."
            )

        normal_cv /= normal_norm

        # SVD plane normals have an arbitrary sign. Choose the direction
        # toward the camera, which is negative Z in OpenCV camera space.
        if normal_cv[2] > 0.0:
            normal_cv *= -1.0

        signed_residuals = (
            centered
            @ normal_cv
        )

        rms_residual_m = float(
            np.sqrt(
                np.mean(
                    signed_residuals
                    * signed_residuals
                )
            )
        )

        if (
            rms_residual_m
            > PORT_PLANE_MAX_RMS_RESIDUAL_M
        ):
            raise RuntimeError(
                "Front-face plane fit residual is too large.\n"
                f"  RMS residual: "
                f"{rms_residual_m * 1000.0:.3f} mm\n"
                f"  allowed: "
                f"{PORT_PLANE_MAX_RMS_RESIDUAL_M * 1000.0:.3f} mm"
            )

        toward_camera_cv = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float64,
        )

        camera_alignment = float(
            np.clip(
                np.dot(
                    normal_cv,
                    toward_camera_cv,
                ),
                -1.0,
                1.0,
            )
        )

        camera_angle_deg = float(
            math.degrees(
                math.acos(
                    camera_alignment
                )
            )
        )

        if (
            camera_angle_deg
            > PORT_PLANE_MAX_CAMERA_ANGLE_DEG
        ):
            raise RuntimeError(
                "Front-face normal points too far sideways.\n"
                f"  angle from camera-facing direction: "
                f"{camera_angle_deg:.3f} deg\n"
                f"  allowed: "
                f"{PORT_PLANE_MAX_CAMERA_ANGLE_DEG:.3f} deg"
            )

        normal_usd_local = np.array(
            [
                normal_cv[0],
                -normal_cv[1],
                -normal_cv[2],
            ],
            dtype=np.float64,
        )

        return {
            "normal_cv": normal_cv,
            "normal_usd_local": (
                normal_usd_local
            ),
            "centroid_cv_m": (
                centroid_cv
            ),
            "input_valid_count": int(
                z_valid.size
            ),
            "inlier_count": inlier_count,
            "depth_median_m": (
                depth_median
            ),
            "depth_mad_m": (
                depth_mad
            ),
            "depth_tolerance_m": (
                float(
                    depth_tolerance
                )
            ),
            "rms_residual_m": (
                rms_residual_m
            ),
            "camera_angle_deg": (
                camera_angle_deg
            ),
            "singular_values": (
                singular_values.astype(
                    np.float64
                )
            ),
            "ring_bounds_xyxy": (
                outer_x0,
                outer_y0,
                outer_x1,
                outer_y1,
            ),
        }


    def compute_preinsert_point(
        opening_world_point_m: np.ndarray,
        outward_world_normal: np.ndarray,
        standoff_m: float,
    ) -> np.ndarray:
        """
        Place a point a fixed distance outward from the port opening.
        """
        opening_point = np.asarray(
            opening_world_point_m,
            dtype=np.float64,
        )

        normal = np.asarray(
            outward_world_normal,
            dtype=np.float64,
        )

        if opening_point.shape != (3,):
            raise RuntimeError(
                "Opening world point must have shape (3,), "
                f"got {opening_point.shape}."
            )

        if normal.shape != (3,):
            raise RuntimeError(
                "World normal must have shape (3,), "
                f"got {normal.shape}."
            )

        normal_norm = float(
            np.linalg.norm(
                normal
            )
        )

        if normal_norm <= 1.0e-12:
            raise RuntimeError(
                "Cannot build pre-insertion point from zero normal."
            )

        normal = (
            normal
            / normal_norm
        )

        standoff = float(
            standoff_m
        )

        if (
            not math.isfinite(
                standoff
            )
            or standoff <= 0.0
        ):
            raise RuntimeError(
                f"Pre-insertion standoff must be positive, got {standoff}."
            )

        return (
            opening_point
            + normal
            * standoff
        )


    def draw_port_detection_overlay(
        rgb: np.ndarray,
        detection: dict[str, object],
        depth_debug: dict[str, object],
        opening_plane: dict[str, object],
    ) -> np.ndarray:
        """
        Draw the ROI, detected opening, depth patch, and front-plane ring.
        """
        from PIL import Image, ImageDraw

        image = Image.fromarray(
            rgb,
            mode="RGB",
        )

        draw = ImageDraw.Draw(
            image
        )

        roi_u0, roi_v0, roi_u1, roi_v1 = (
            int(value)
            for value in detection[
                "roi_uv"
            ]
        )

        x, y, width, height = (
            int(value)
            for value in detection[
                "bbox_xywh"
            ]
        )

        u, v = (
            int(value)
            for value in detection[
                "center_uv"
            ]
        )

        patch_u0, patch_v0, patch_u1, patch_v1 = (
            int(value)
            for value in depth_debug[
                "patch_bounds_uv"
            ]
        )

        ring_x0, ring_y0, ring_x1, ring_y1 = (
            int(value)
            for value in opening_plane[
                "outer_bounds_xyxy"
            ]
        )

        half_length = int(
            PORT_DETECTION_CROSSHAIR_HALF_LENGTH_PX
        )

        line_width = int(
            PORT_DETECTION_CROSSHAIR_WIDTH_PX
        )

        roi_color = (
            0,
            128,
            255,
        )

        detection_color = (
            0,
            255,
            0,
        )

        patch_color = (
            255,
            255,
            0,
        )

        opening_ring_color = (
            0,
            255,
            255,
        )

        draw.rectangle(
            [
                roi_u0,
                roi_v0,
                roi_u1 - 1,
                roi_v1 - 1,
            ],
            outline=roi_color,
            width=1,
        )

        draw.rectangle(
            [
                ring_x0,
                ring_y0,
                ring_x1 - 1,
                ring_y1 - 1,
            ],
            outline=opening_ring_color,
            width=2,
        )

        draw.rectangle(
            [
                x,
                y,
                x + width - 1,
                y + height - 1,
            ],
            outline=detection_color,
            width=2,
        )

        draw.rectangle(
            [
                patch_u0,
                patch_v0,
                patch_u1 - 1,
                patch_v1 - 1,
            ],
            outline=patch_color,
            width=1,
        )

        draw.line(
            [
                u - half_length,
                v,
                u + half_length,
                v,
            ],
            fill=detection_color,
            width=line_width,
        )

        draw.line(
            [
                u,
                v - half_length,
                u,
                v + half_length,
            ],
            fill=detection_color,
            width=line_width,
        )

        label = (
            f"PORT center=({u}, {v})  "
            f"cavity="
            f"{float(depth_debug['median_depth_m']):.4f} m  "
            f"opening="
            f"{float(opening_plane['opening_plane_depth_m']):.4f} m  "
            f"recess="
            f"{float(opening_plane['recess_depth_m']) * 1000.0:.1f} mm"
        )

        text_x = 10
        text_y = 10

        text_bbox = draw.textbbox(
            (
                text_x,
                text_y,
            ),
            label,
        )

        draw.rectangle(
            [
                text_bbox[0] - 4,
                text_bbox[1] - 3,
                text_bbox[2] + 4,
                text_bbox[3] + 3,
            ],
            fill=(
                0,
                0,
                0,
            ),
        )

        draw.text(
            (
                text_x,
                text_y,
            ),
            label,
            fill=(
                255,
                255,
                255,
            ),
        )

        return np.asarray(
            image,
            dtype=np.uint8,
        ).copy()


    def compute_camera_intrinsics(
        camera_path: str,
        image_shape_hw: tuple[int, int],
    ) -> dict[str, float]:
        """
        Compute pinhole intrinsics from the actual USD camera properties.

        The image shape follows NumPy/OpenCV convention: (height, width).
        """
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(
            camera_path
        )

        if not camera_prim.IsValid():
            raise RuntimeError(
                f"Cannot compute intrinsics for missing camera: {camera_path}"
            )

        if not camera_prim.IsA(
            UsdGeom.Camera
        ):
            raise RuntimeError(
                f"Prim is not a USD Camera: {camera_path}"
            )

        camera = UsdGeom.Camera(
            camera_prim
        )

        image_height = int(
            image_shape_hw[0]
        )

        image_width = int(
            image_shape_hw[1]
        )

        if image_height <= 0 or image_width <= 0:
            raise RuntimeError(
                "Image dimensions must be positive, "
                f"got {(image_height, image_width)}."
            )

        focal_length_mm = float(
            camera.GetFocalLengthAttr().Get()
        )

        horizontal_aperture_mm = float(
            camera.GetHorizontalApertureAttr().Get()
        )

        vertical_aperture_mm = float(
            camera.GetVerticalApertureAttr().Get()
        )

        if focal_length_mm <= 0.0:
            raise RuntimeError(
                f"Invalid focal length: {focal_length_mm} mm"
            )

        if horizontal_aperture_mm <= 0.0:
            raise RuntimeError(
                "Invalid horizontal aperture: "
                f"{horizontal_aperture_mm} mm"
            )

        if vertical_aperture_mm <= 0.0:
            raise RuntimeError(
                "Invalid vertical aperture: "
                f"{vertical_aperture_mm} mm"
            )

        fx_px = (
            focal_length_mm
            * float(image_width)
            / horizontal_aperture_mm
        )

        fy_px = (
            focal_length_mm
            * float(image_height)
            / vertical_aperture_mm
        )

        # Use the center of the pixel grid as the principal point because this
        # camera has no authored aperture offsets or lens distortion.
        cx_px = (
            float(image_width - 1)
            / 2.0
        )

        cy_px = (
            float(image_height - 1)
            / 2.0
        )

        return {
            "image_width_px": float(
                image_width
            ),
            "image_height_px": float(
                image_height
            ),
            "focal_length_mm": focal_length_mm,
            "horizontal_aperture_mm": horizontal_aperture_mm,
            "vertical_aperture_mm": vertical_aperture_mm,
            "fx_px": float(
                fx_px
            ),
            "fy_px": float(
                fy_px
            ),
            "cx_px": float(
                cx_px
            ),
            "cy_px": float(
                cy_px
            ),
        }


    def deproject_pixel_to_camera(
        center_uv: tuple[int, int],
        depth_to_image_plane_m: float,
        intrinsics: dict[str, float],
    ) -> dict[str, np.ndarray]:
        """
        Convert one RGB pixel plus axial depth into a 3D camera point.

        OpenCV camera convention:
            +X = image right
            +Y = image down
            +Z = forward

        USD camera-local convention:
            +X = image right
            +Y = image up
            -Z = forward
        """
        u = float(
            center_uv[0]
        )

        v = float(
            center_uv[1]
        )

        depth_m = float(
            depth_to_image_plane_m
        )

        if not math.isfinite(
            depth_m
        ) or depth_m <= 0.0:
            raise RuntimeError(
                f"Depth must be finite and positive, got {depth_m}."
            )

        fx_px = float(
            intrinsics["fx_px"]
        )

        fy_px = float(
            intrinsics["fy_px"]
        )

        cx_px = float(
            intrinsics["cx_px"]
        )

        cy_px = float(
            intrinsics["cy_px"]
        )

        if fx_px <= 0.0 or fy_px <= 0.0:
            raise RuntimeError(
                "Camera focal lengths in pixels must be positive."
            )

        x_cv_m = (
            (u - cx_px)
            * depth_m
            / fx_px
        )

        y_cv_m = (
            (v - cy_px)
            * depth_m
            / fy_px
        )

        z_cv_m = depth_m

        point_cv_m = np.array(
            [
                x_cv_m,
                y_cv_m,
                z_cv_m,
            ],
            dtype=np.float64,
        )

        point_usd_local_m = np.array(
            [
                x_cv_m,
                -y_cv_m,
                -z_cv_m,
            ],
            dtype=np.float64,
        )

        return {
            "point_cv_m": point_cv_m,
            "point_usd_local_m": point_usd_local_m,
        }


    def transform_camera_point_to_world(
        camera_path: str,
        point_usd_local_m: np.ndarray,
    ) -> np.ndarray:
        """
        Transform one USD camera-local point into world coordinates.
        """
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(
            camera_path
        )

        if not camera_prim.IsValid():
            raise RuntimeError(
                f"Cannot transform through missing camera: {camera_path}"
            )

        local_point = np.asarray(
            point_usd_local_m,
            dtype=np.float64,
        )

        if local_point.shape != (3,):
            raise RuntimeError(
                "Camera-local point must have shape (3,), "
                f"got {local_point.shape}."
            )

        camera_world = UsdGeom.Xformable(
            camera_prim
        ).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        world_point = camera_world.Transform(
            Gf.Vec3d(
                float(local_point[0]),
                float(local_point[1]),
                float(local_point[2]),
            )
        )

        return np.asarray(
            world_point,
            dtype=np.float64,
        )


    def transform_camera_direction_to_world(
        camera_path: str,
        direction_usd_local: np.ndarray,
    ) -> np.ndarray:
        """
        Transform and normalize one USD camera-local direction into world space.
        """
        stage = omni.usd.get_context().get_stage()

        camera_prim = stage.GetPrimAtPath(
            camera_path
        )

        if not camera_prim.IsValid():
            raise RuntimeError(
                f"Cannot transform through missing camera: {camera_path}"
            )

        direction = np.asarray(
            direction_usd_local,
            dtype=np.float64,
        )

        if direction.shape != (3,):
            raise RuntimeError(
                "Camera-local direction must have shape (3,), "
                f"got {direction.shape}."
            )

        direction_norm = float(
            np.linalg.norm(
                direction
            )
        )

        if direction_norm <= 1.0e-12:
            raise RuntimeError(
                "Cannot transform a zero-length camera direction."
            )

        direction = (
            direction
            / direction_norm
        )

        camera_world = UsdGeom.Xformable(
            camera_prim
        ).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        world_direction = camera_world.TransformDir(
            Gf.Vec3d(
                float(
                    direction[0]
                ),
                float(
                    direction[1]
                ),
                float(
                    direction[2]
                ),
            )
        )

        world_direction_np = np.asarray(
            world_direction,
            dtype=np.float64,
        )

        world_norm = float(
            np.linalg.norm(
                world_direction_np
            )
        )

        if world_norm <= 1.0e-12:
            raise RuntimeError(
                "Camera-to-world direction transform returned zero length."
            )

        return (
            world_direction_np
            / world_norm
        )


    def quaternion_wxyz_from_positive_z(
        target_direction: np.ndarray,
    ) -> np.ndarray:
        """
        Return a scalar-first quaternion rotating local +Z onto target_direction.
        """
        target = np.asarray(
            target_direction,
            dtype=np.float64,
        )

        if target.shape != (3,):
            raise RuntimeError(
                "Target direction must have shape (3,), "
                f"got {target.shape}."
            )

        target_norm = float(
            np.linalg.norm(
                target
            )
        )

        if target_norm <= 1.0e-12:
            raise RuntimeError(
                "Cannot orient an arrow toward a zero-length vector."
            )

        target = (
            target
            / target_norm
        )

        source = np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        dot = float(
            np.clip(
                np.dot(
                    source,
                    target,
                ),
                -1.0,
                1.0,
            )
        )

        if dot > 1.0 - 1.0e-12:
            return np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        if dot < -1.0 + 1.0e-12:
            # Any axis perpendicular to +Z is valid for the 180-degree case.
            return np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        cross = np.cross(
            source,
            target,
        )

        quat = np.array(
            [
                1.0 + dot,
                cross[0],
                cross[1],
                cross[2],
            ],
            dtype=np.float64,
        )

        quat_norm = float(
            np.linalg.norm(
                quat
            )
        )

        if quat_norm <= 1.0e-12:
            raise RuntimeError(
                "Failed to build arrow orientation quaternion."
            )

        return (
            quat
            / quat_norm
        )


    def set_world_xform_pose(
        prim_path: str,
        position_m: np.ndarray,
        orientation_wxyz: np.ndarray,
    ) -> None:
        """
        Set one world-authored prim's translate and orient ops.
        """
        stage = omni.usd.get_context().get_stage()

        prim = stage.GetPrimAtPath(
            prim_path
        )

        if not prim.IsValid():
            raise RuntimeError(
                f"Cannot pose missing prim: {prim_path}"
            )

        position = np.asarray(
            position_m,
            dtype=np.float64,
        )

        orientation = np.asarray(
            orientation_wxyz,
            dtype=np.float64,
        )

        if position.shape != (3,):
            raise RuntimeError(
                f"Position must have shape (3,), got {position.shape}."
            )

        if orientation.shape != (4,):
            raise RuntimeError(
                "Orientation must have shape (4,) in WXYZ order, "
                f"got {orientation.shape}."
            )

        orientation_norm = float(
            np.linalg.norm(
                orientation
            )
        )

        if orientation_norm <= 1.0e-12:
            raise RuntimeError(
                "Cannot apply a zero quaternion."
            )

        orientation = (
            orientation
            / orientation_norm
        )

        xform = UsdGeom.Xformable(
            prim
        )

        translate_ops = [
            op
            for op in xform.GetOrderedXformOps()
            if (
                op.GetOpType()
                == UsdGeom.XformOp.TypeTranslate
            )
        ]

        orient_ops = [
            op
            for op in xform.GetOrderedXformOps()
            if (
                op.GetOpType()
                == UsdGeom.XformOp.TypeOrient
            )
        ]

        if not translate_ops or not orient_ops:
            raise RuntimeError(
                "World Xform is missing translate/orient ops: "
                f"{prim_path}"
            )

        translate_ops[0].Set(
            Gf.Vec3d(
                float(
                    position[0]
                ),
                float(
                    position[1]
                ),
                float(
                    position[2]
                ),
            )
        )

        orient_ops[0].Set(
            Gf.Quatd(
                float(
                    orientation[0]
                ),
                Gf.Vec3d(
                    float(
                        orientation[1]
                    ),
                    float(
                        orientation[2]
                    ),
                    float(
                        orientation[3]
                    ),
                ),
            )
        )


    def create_or_update_port_normal_arrow(
        opening_world_point_m: np.ndarray,
        outward_world_normal: np.ndarray,
        arrow_length_m: float,
    ) -> None:
        """
        Draw an orange guide arrow from the opening toward the pre-insert point.

        The arrow root uses local +Z as its forward axis. A cylinder forms the
        shaft and a cone forms the tip.
        """
        stage = omni.usd.get_context().get_stage()

        opening_point = np.asarray(
            opening_world_point_m,
            dtype=np.float64,
        )

        normal = np.asarray(
            outward_world_normal,
            dtype=np.float64,
        )

        length = float(
            arrow_length_m
        )

        if length <= PORT_APPROACH_NORMAL_TIP_LENGTH_M:
            raise RuntimeError(
                "Arrow length must exceed the configured tip length."
            )

        normal_norm = float(
            np.linalg.norm(
                normal
            )
        )

        if normal_norm <= 1.0e-12:
            raise RuntimeError(
                "Cannot draw port normal from zero-length direction."
            )

        normal = (
            normal
            / normal_norm
        )

        root_prim = stage.GetPrimAtPath(
            PORT_APPROACH_NORMAL_PATH
        )

        shaft_length = (
            length
            - PORT_APPROACH_NORMAL_TIP_LENGTH_M
        )

        if not root_prim.IsValid():
            root_prim = stage.DefinePrim(
                PORT_APPROACH_NORMAL_PATH,
                "Xform",
            )

            if not root_prim.IsValid():
                raise RuntimeError(
                    "Failed to create port-normal arrow root."
                )

            root_xform = UsdGeom.Xformable(
                root_prim
            )

            root_xform.ClearXformOpOrder()

            root_xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(
                Gf.Vec3d(
                    0.0,
                    0.0,
                    0.0,
                )
            )

            root_xform.AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(
                Gf.Quatd(
                    1.0,
                    Gf.Vec3d(
                        0.0,
                        0.0,
                        0.0,
                    ),
                )
            )

            UsdGeom.Imageable(
                root_prim
            ).CreatePurposeAttr().Set(
                UsdGeom.Tokens.guide
            )

            shaft = UsdGeom.Cylinder.Define(
                stage,
                PORT_APPROACH_NORMAL_SHAFT_PATH,
            )

            shaft.CreateAxisAttr().Set(
                UsdGeom.Tokens.z
            )

            shaft.CreateRadiusAttr().Set(
                float(
                    PORT_APPROACH_NORMAL_SHAFT_RADIUS_M
                )
            )

            shaft.CreateHeightAttr().Set(
                float(
                    shaft_length
                )
            )

            shaft.CreateDisplayColorAttr().Set(
                [
                    Gf.Vec3f(
                        *PORT_APPROACH_NORMAL_COLOR
                    )
                ]
            )

            shaft_xform = UsdGeom.Xformable(
                shaft.GetPrim()
            )

            shaft_xform.ClearXformOpOrder()

            shaft_xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(
                Gf.Vec3d(
                    0.0,
                    0.0,
                    shaft_length / 2.0,
                )
            )

            tip = UsdGeom.Cone.Define(
                stage,
                PORT_APPROACH_NORMAL_TIP_PATH,
            )

            tip.CreateAxisAttr().Set(
                UsdGeom.Tokens.z
            )

            tip.CreateRadiusAttr().Set(
                float(
                    PORT_APPROACH_NORMAL_TIP_RADIUS_M
                )
            )

            tip.CreateHeightAttr().Set(
                float(
                    PORT_APPROACH_NORMAL_TIP_LENGTH_M
                )
            )

            tip.CreateDisplayColorAttr().Set(
                [
                    Gf.Vec3f(
                        *PORT_APPROACH_NORMAL_COLOR
                    )
                ]
            )

            tip_xform = UsdGeom.Xformable(
                tip.GetPrim()
            )

            tip_xform.ClearXformOpOrder()

            tip_xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(
                Gf.Vec3d(
                    0.0,
                    0.0,
                    shaft_length
                    + PORT_APPROACH_NORMAL_TIP_LENGTH_M / 2.0,
                )
            )

            log(
                "Created port approach-normal arrow:\n"
                f"  root:   {PORT_APPROACH_NORMAL_PATH}\n"
                f"  length: {length:.4f} m"
            )

        else:
            shaft_prim = stage.GetPrimAtPath(
                PORT_APPROACH_NORMAL_SHAFT_PATH
            )

            tip_prim = stage.GetPrimAtPath(
                PORT_APPROACH_NORMAL_TIP_PATH
            )

            if (
                not shaft_prim.IsValid()
                or not tip_prim.IsValid()
            ):
                raise RuntimeError(
                    "Port-normal arrow root exists but child geometry "
                    "is missing."
                )

            UsdGeom.Cylinder(
                shaft_prim
            ).GetHeightAttr().Set(
                float(
                    shaft_length
                )
            )

            shaft_xform = UsdGeom.Xformable(
                shaft_prim
            )

            shaft_translate_ops = [
                op
                for op in shaft_xform.GetOrderedXformOps()
                if (
                    op.GetOpType()
                    == UsdGeom.XformOp.TypeTranslate
                )
            ]

            if not shaft_translate_ops:
                raise RuntimeError(
                    "Port-normal arrow shaft is missing its translate op."
                )

            shaft_translate_ops[0].Set(
                Gf.Vec3d(
                    0.0,
                    0.0,
                    shaft_length / 2.0,
                )
            )

            tip_xform = UsdGeom.Xformable(
                tip_prim
            )

            tip_translate_ops = [
                op
                for op in tip_xform.GetOrderedXformOps()
                if (
                    op.GetOpType()
                    == UsdGeom.XformOp.TypeTranslate
                )
            ]

            if not tip_translate_ops:
                raise RuntimeError(
                    "Port-normal arrow tip is missing its translate op."
                )

            tip_translate_ops[0].Set(
                Gf.Vec3d(
                    0.0,
                    0.0,
                    shaft_length
                    + PORT_APPROACH_NORMAL_TIP_LENGTH_M / 2.0,
                )
            )

        orientation_wxyz = (
            quaternion_wxyz_from_positive_z(
                normal
            )
        )

        set_world_xform_pose(
            prim_path=PORT_APPROACH_NORMAL_PATH,
            position_m=opening_point,
            orientation_wxyz=orientation_wxyz,
        )


    def update_preinsert_point_marker(
        world_point_m: np.ndarray,
    ) -> None:
        """Move the magenta 50 mm pre-insertion marker."""
        update_world_point_marker(
            marker_path=PREINSERT_POINT_PATH,
            world_point_m=world_point_m,
            radius_m=PREINSERT_POINT_RADIUS_M,
            color_rgb=PREINSERT_POINT_COLOR,
            label="pre-insertion-point",
        )


    def update_world_point_marker(
        marker_path: str,
        world_point_m: np.ndarray,
        radius_m: float,
        color_rgb: tuple[float, float, float],
        label: str,
    ) -> None:
        """Create or move one colored guide sphere at a world point."""
        stage = omni.usd.get_context().get_stage()
        world_point = np.asarray(world_point_m, dtype=np.float64)

        if world_point.shape != (3,):
            raise RuntimeError(
                f"{label} world point must have shape (3,), "
                f"got {world_point.shape}."
            )

        if not np.all(np.isfinite(world_point)):
            raise RuntimeError(
                f"{label} world point contains non-finite values: "
                f"{world_point.tolist()}"
            )

        marker_prim = stage.GetPrimAtPath(marker_path)

        if not marker_prim.IsValid():
            marker = UsdGeom.Sphere.Define(stage, marker_path)
            marker.CreateRadiusAttr().Set(float(radius_m))
            marker.CreateDisplayColorAttr().Set([
                Gf.Vec3f(*color_rgb)
            ])
            marker_prim = marker.GetPrim()
            UsdGeom.Imageable(marker_prim).CreatePurposeAttr().Set(
                UsdGeom.Tokens.guide
            )
            marker_xform = UsdGeom.Xformable(marker_prim)
            marker_xform.ClearXformOpOrder()
            marker_xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(*world_point.tolist()))
            log(
                f"Created {label} marker:\n"
                f"  prim:   {marker_path}\n"
                f"  radius: {radius_m:.4f} m"
            )
            return

        marker_xform = UsdGeom.Xformable(marker_prim)
        translate_ops = [
            op
            for op in marker_xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]

        if not translate_ops:
            raise RuntimeError(
                f"{label} marker exists but has no translate op: "
                f"{marker_path}"
            )

        translate_ops[0].Set(Gf.Vec3d(*world_point.tolist()))


    def update_detected_port_marker(
        world_point_m: np.ndarray,
    ) -> None:
        """Move the green recessed socket-interior marker."""
        update_world_point_marker(
            marker_path=DETECTED_PORT_MARKER_PATH,
            world_point_m=world_point_m,
            radius_m=DETECTED_PORT_MARKER_RADIUS_M,
            color_rgb=DETECTED_PORT_MARKER_COLOR,
            label="socket-interior",
        )


    def update_port_opening_center_marker(
        world_point_m: np.ndarray,
    ) -> None:
        """Move the cyan front opening-plane center marker."""
        update_world_point_marker(
            marker_path=PORT_OPENING_CENTER_MARKER_PATH,
            world_point_m=world_point_m,
            radius_m=PORT_OPENING_CENTER_MARKER_RADIUS_M,
            color_rgb=PORT_OPENING_CENTER_MARKER_COLOR,
            label="port-opening-center",
        )


    def save_rgb_and_depth(
        rgb: np.ndarray,
        depth_m: np.ndarray,
        capture_index: int,
        camera_path: str,
    ) -> None:
        """
        Save the latest RGB image, a viewable depth preview, and raw depth.

        Raw depth is stored in meters as float32 in depth_latest_meters.npy.
        """
        from PIL import Image

        depth_preview, near_m, far_m, valid_count = (
            make_depth_preview(
                depth_m
            )
        )

        port_detection = detect_single_ethernet_port(
            rgb
        )

        port_depth = compute_detection_depth(
            depth_m=depth_m,
            center_uv=port_detection[
                "center_uv"
            ],
        )

        opening_plane = estimate_port_opening_plane_depth(
            depth_m=depth_m,
            bbox_xywh=port_detection[
                "bbox_xywh"
            ],
            cavity_depth_m=port_depth[
                "median_depth_m"
            ],
        )

        camera_intrinsics = compute_camera_intrinsics(
            camera_path=camera_path,
            image_shape_hw=(
                rgb.shape[0],
                rgb.shape[1],
            ),
        )

        opening_plane_fit = fit_port_opening_plane_normal(
            depth_m=depth_m,
            bbox_xywh=port_detection[
                "bbox_xywh"
            ],
            intrinsics=camera_intrinsics,
        )

        port_camera_points = deproject_pixel_to_camera(
            center_uv=port_detection[
                "center_uv"
            ],
            depth_to_image_plane_m=port_depth[
                "median_depth_m"
            ],
            intrinsics=camera_intrinsics,
        )

        port_world_point_m = transform_camera_point_to_world(
            camera_path=camera_path,
            point_usd_local_m=port_camera_points[
                "point_usd_local_m"
            ],
        )

        update_detected_port_marker(
            port_world_point_m
        )

        opening_camera_points = deproject_pixel_to_camera(
            center_uv=port_detection[
                "center_uv"
            ],
            depth_to_image_plane_m=opening_plane[
                "opening_plane_depth_m"
            ],
            intrinsics=camera_intrinsics,
        )

        opening_world_point_m = transform_camera_point_to_world(
            camera_path=camera_path,
            point_usd_local_m=opening_camera_points[
                "point_usd_local_m"
            ],
        )

        update_port_opening_center_marker(
            opening_world_point_m
        )

        outward_world_normal = transform_camera_direction_to_world(
            camera_path=camera_path,
            direction_usd_local=opening_plane_fit[
                "normal_usd_local"
            ],
        )

        preinsert_world_point_m = compute_preinsert_point(
            opening_world_point_m=opening_world_point_m,
            outward_world_normal=outward_world_normal,
            standoff_m=PREINSERT_STANDOFF_M,
        )

        create_or_update_port_normal_arrow(
            opening_world_point_m=opening_world_point_m,
            outward_world_normal=outward_world_normal,
            arrow_length_m=PREINSERT_STANDOFF_M,
        )

        update_preinsert_point_marker(
            preinsert_world_point_m
        )

        port_detection_rgb = draw_port_detection_overlay(
            rgb=rgb,
            detection=port_detection,
            depth_debug=port_depth,
            opening_plane=opening_plane,
        )

        rgb_path = (
            CAMERA_OUTPUT_DIR
            / "rgb_latest.png"
        )

        depth_preview_path = (
            CAMERA_OUTPUT_DIR
            / "depth_preview_latest.png"
        )

        depth_raw_path = (
            CAMERA_OUTPUT_DIR
            / "depth_latest_meters.npy"
        )

        port_detection_path = (
            CAMERA_OUTPUT_DIR
            / "rgb_port_detected.png"
        )

        port_mask_path = (
            CAMERA_OUTPUT_DIR
            / "port_detection_mask.png"
        )

        Image.fromarray(
            rgb,
            mode="RGB",
        ).save(rgb_path)

        Image.fromarray(
            depth_preview,
            mode="L",
        ).save(depth_preview_path)

        np.save(
            depth_raw_path,
            depth_m,
        )

        Image.fromarray(
            port_detection_rgb,
            mode="RGB",
        ).save(
            port_detection_path
        )

        Image.fromarray(
            port_detection[
                "mask"
            ],
            mode="L",
        ).save(
            port_mask_path
        )

        log(
            "AUTOMATIC ETHERNET PORT DETECTION\n"
            f"  search ROI:        "
            f"{port_detection['roi_uv']}\n"
            f"  candidates:        "
            f"{port_detection['candidate_count']}\n"
            f"  bounding box XYWH: "
            f"{port_detection['bbox_xywh']}\n"
            f"  center (u, v):     "
            f"{port_detection['center_uv']}\n"
            f"  component area:    "
            f"{port_detection['area_px']} px\n"
            f"  aspect ratio:      "
            f"{port_detection['aspect_ratio']:.4f}\n"
            f"  fill ratio:        "
            f"{port_detection['fill_ratio']:.4f}\n"
            f"  depth patch:       "
            f"{port_depth['patch_bounds_uv']}\n"
            f"  valid depth pixels:"
            f"{port_depth['valid_count']}\n"
            f"  center depth:      "
            f"{port_depth['center_depth_m']:.6f} m\n"
            f"  median depth:      "
            f"{port_depth['median_depth_m']:.6f} m\n"
            f"  patch min/max:     "
            f"{port_depth['minimum_depth_m']:.6f} / "
            f"{port_depth['maximum_depth_m']:.6f} m\n"
            f"  opening ring:      "
            f"{opening_plane['outer_bounds_xyxy']}\n"
            f"  valid ring pixels: "
            f"{opening_plane['valid_count']}\n"
            f"  opening depth:     "
            f"{opening_plane['opening_plane_depth_m']:.6f} m\n"
            f"  cavity depth:      "
            f"{opening_plane['cavity_depth_m']:.6f} m\n"
            f"  recess depth:      "
            f"{opening_plane['recess_depth_m'] * 1000.0:.3f} mm\n"
            f"  ring min/max:      "
            f"{opening_plane['minimum_ring_depth_m']:.6f} / "
            f"{opening_plane['maximum_ring_depth_m']:.6f} m\n"
            f"  intrinsics fx/fy:  "
            f"{camera_intrinsics['fx_px']:.3f} / "
            f"{camera_intrinsics['fy_px']:.3f} px\n"
            f"  principal cx/cy:   "
            f"{camera_intrinsics['cx_px']:.3f} / "
            f"{camera_intrinsics['cy_px']:.3f} px\n"
            f"  camera XYZ (CV):   "
            f"{np.round(port_camera_points['point_cv_m'], 6).tolist()} m\n"
            f"  camera XYZ (USD):  "
            f"{np.round(port_camera_points['point_usd_local_m'], 6).tolist()} m\n"
            f"  cavity world XYZ:  "
            f"{np.round(port_world_point_m, 6).tolist()} m\n"
            f"  opening camera XYZ:"
            f"{np.round(opening_camera_points['point_cv_m'], 6).tolist()} m\n"
            f"  opening world XYZ: "
            f"{np.round(opening_world_point_m, 6).tolist()} m\n"
            f"  plane inliers:     "
            f"{opening_plane_fit['inlier_count']} / "
            f"{opening_plane_fit['input_valid_count']}\n"
            f"  plane RMS:         "
            f"{opening_plane_fit['rms_residual_m'] * 1000.0:.4f} mm\n"
            f"  normal angle:      "
            f"{opening_plane_fit['camera_angle_deg']:.3f} deg\n"
            f"  normal CV XYZ:     "
            f"{np.round(opening_plane_fit['normal_cv'], 6).tolist()}\n"
            f"  normal USD XYZ:    "
            f"{np.round(opening_plane_fit['normal_usd_local'], 6).tolist()}\n"
            f"  normal world XYZ:  "
            f"{np.round(outward_world_normal, 6).tolist()}\n"
            f"  preinsert distance:"
            f"{PREINSERT_STANDOFF_M * 1000.0:.1f} mm\n"
            f"  preinsert world:   "
            f"{np.round(preinsert_world_point_m, 6).tolist()} m\n"
            f"  cavity marker:     "
            f"{DETECTED_PORT_MARKER_PATH}\n"
            f"  opening marker:    "
            f"{PORT_OPENING_CENTER_MARKER_PATH}\n"
            f"  normal arrow:      "
            f"{PORT_APPROACH_NORMAL_PATH}\n"
            f"  preinsert marker:  "
            f"{PREINSERT_POINT_PATH}\n"
            f"  annotated RGB:     "
            f"{port_detection_path}\n"
            f"  binary mask:       "
            f"{port_mask_path}"
        )

        log(
            "RGB + DEPTH CAPTURE\n"
            f"  capture index: {capture_index}\n"
            f"  RGB shape:     {rgb.shape}\n"
            f"  RGB dtype:     {rgb.dtype}\n"
            f"  RGB min/max:   {int(rgb.min())} / {int(rgb.max())}\n"
            f"  Depth shape:   {depth_m.shape}\n"
            f"  Depth dtype:   {depth_m.dtype}\n"
            f"  Valid depth:   {valid_count} pixels\n"
            f"  Preview range: {near_m:.4f} to {far_m:.4f} m\n"
            f"  RGB PNG:       {rgb_path}\n"
            f"  Depth PNG:     {depth_preview_path}\n"
            f"  Raw depth NPY: {depth_raw_path}\n"
            f"  Detected PNG:  {port_detection_path}\n"
            f"  Detection mask:{port_mask_path}"
        )


    def capture_rgb_and_depth(
        sensor: CameraSensor,
        capture_index: int,
        camera_path: str,
    ) -> None:
        """Read the latest RGB and metric depth frames from CameraSensor."""
        rgb_data, rgb_info = sensor.get_data(
            "rgb"
        )

        depth_data, depth_info = sensor.get_data(
            "distance_to_image_plane"
        )

        rgb = reshape_rgb(
            sensor_array_to_numpy(
                rgb_data,
                "rgb",
            )
        )

        depth_m = reshape_depth(
            sensor_array_to_numpy(
                depth_data,
                "distance_to_image_plane",
            )
        )

        save_rgb_and_depth(
            rgb=rgb,
            depth_m=depth_m,
            capture_index=capture_index,
            camera_path=camera_path,
        )

        if capture_index == 1:
            log(
                "Annotator info keys:\n"
                f"  rgb:   {sorted(rgb_info.keys()) if isinstance(rgb_info, dict) else type(rgb_info)}\n"
                f"  depth: {sorted(depth_info.keys()) if isinstance(depth_info, dict) else type(depth_info)}"
            )


    def get_prim_world_pose_wxyz(
        prim_path: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read a USD prim's world pose as position + scalar-first quaternion.
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            raise RuntimeError(
                f"Cannot read world pose of missing prim: {prim_path}"
            )

        world_transform = UsdGeom.Xformable(
            prim
        ).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        position = np.asarray(
            world_transform.ExtractTranslation(),
            dtype=np.float64,
        )

        rotation = world_transform.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()

        orientation_wxyz = np.array(
            [
                rotation.GetReal(),
                imaginary[0],
                imaginary[1],
                imaginary[2],
            ],
            dtype=np.float64,
        )

        norm = float(
            np.linalg.norm(
                orientation_wxyz
            )
        )

        if norm <= 1.0e-12:
            raise RuntimeError(
                f"Invalid zero quaternion at {prim_path}"
            )

        orientation_wxyz /= norm

        return position, orientation_wxyz


    def create_ik_runtime(
        assets_root: str,
    ) -> IKRuntime:
        """
        Initialize the Franka articulation, Lula IK, and a visible target.

        The target starts exactly at panda_hand's current world pose. Moving
        /World/IK_Target with the normal viewport transform gizmo changes the
        pose that Lula tries to reach.
        """
        log(
            "Initializing draggable Lula IK target..."
        )

        articulation = Articulation(
            FRANKA_ASSET_PATH,
            name="franka_ik_articulation",
        )

        articulation.initialize()

        if not articulation.handles_initialized:
            raise RuntimeError(
                "Franka articulation physics handles did not initialize."
            )

        kinematics_config = (
            interface_config_loader
            .load_supported_lula_kinematics_solver_config(
                "Franka"
            )
        )

        kinematics_solver = LulaKinematicsSolver(
            **kinematics_config
        )

        valid_frames = (
            kinematics_solver
            .get_all_frame_names()
        )

        if IK_END_EFFECTOR_FRAME not in valid_frames:
            raise RuntimeError(
                f"Lula does not recognize end-effector frame "
                f"'{IK_END_EFFECTOR_FRAME}'.\n"
                f"Available frames: {valid_frames}"
            )

        articulation_solver = ArticulationKinematicsSolver(
            articulation,
            kinematics_solver,
            IK_END_EFFECTOR_FRAME,
        )

        hand_path = find_unique_descendant_by_name(
            FRANKA_ASSET_PATH,
            IK_END_EFFECTOR_FRAME,
        )

        if IK_TARGET_USE_FIXED_START_POSE:
            target_position = np.array(
                IK_TARGET_INITIAL_POSITION,
                dtype=np.float64,
            )
            target_orientation = np.array(
                IK_TARGET_INITIAL_ORIENTATION_WXYZ,
                dtype=np.float64,
            )

            target_orientation /= np.linalg.norm(
                target_orientation
            )
        else:
            target_position, target_orientation = (
                get_prim_world_pose_wxyz(
                    hand_path
                )
            )

        target_asset_path = (
            assets_root
            + "/Isaac/Props/UIElements/frame_prim.usd"
        )

        stage_utils.add_reference_to_stage(
            usd_path=target_asset_path,
            path=IK_TARGET_PATH,
        )

        for _ in range(5):
            simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(
            IK_TARGET_PATH
        )

        if not target_prim.IsValid():
            raise RuntimeError(
                "Visible IK target frame did not compose at "
                f"{IK_TARGET_PATH}"
            )

        target = XFormPrim(
            prim_path=IK_TARGET_PATH,
            name=IK_TARGET_NAME,
            position=target_position,
            orientation=target_orientation,
            scale=np.array(
                [
                    IK_TARGET_SCALE,
                    IK_TARGET_SCALE,
                    IK_TARGET_SCALE,
                ],
                dtype=np.float64,
            ),
            visible=True,
        )

        target.initialize()

        # Make the target immediately easy to find and manipulate.
        try:
            omni.usd.get_context().get_selection().set_selected_prim_paths(
                [IK_TARGET_PATH],
                True,
            )
        except Exception as exc:
            warn(
                "Could not auto-select IK target in the Stage panel: "
                f"{exc}"
            )

        runtime = IKRuntime(
            articulation=articulation,
            target=target,
            kinematics_solver=kinematics_solver,
            articulation_solver=articulation_solver,
        )

        base_position, base_orientation = (
            articulation.get_world_pose()
        )

        kinematics_solver.set_robot_base_pose(
            base_position,
            base_orientation,
        )

        log(
            "DRAGGABLE IK READY\n"
            f"  target prim:      {IK_TARGET_PATH}\n"
            f"  end effector:     {IK_END_EFFECTOR_FRAME}\n"
            f"  fixed start pose:  "
            f"{IK_TARGET_USE_FIXED_START_POSE}\n"
            f"  initial position: "
            f"{np.round(target_position, 4).tolist()}\n"
            f"  initial quat WXYZ:"
            f"{np.round(target_orientation, 5).tolist()}\n"
            f"  robot base world: "
            f"{np.round(base_position, 4).tolist()}\n"
            "  Usage: select /World/IK_Target, press W for the move "
            "gizmo or E for the rotate gizmo, then drag the target."
        )

        return runtime


    def update_ik_target_tracking(
        runtime: IKRuntime,
        sim_frame_index: int,
    ) -> bool:
        """
        Solve Lula IK from the current target pose and command the Franka.

        The robot base pose is refreshed every solve because this scene moves
        the entire Franka container to Z=1.0 m.
        """
        if not IK_TRACKING_ENABLED:
            return False

        if (
            sim_frame_index
            % IK_UPDATE_EVERY_SIM_FRAMES
            != 0
        ):
            return False

        target_position, target_orientation = (
            runtime.target.get_world_pose()
        )

        base_position, base_orientation = (
            runtime.articulation.get_world_pose()
        )

        runtime.kinematics_solver.set_robot_base_pose(
            base_position,
            base_orientation,
        )

        action, success = (
            runtime.articulation_solver
            .compute_inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                position_tolerance=IK_POSITION_TOLERANCE_M,
                orientation_tolerance=IK_ORIENTATION_TOLERANCE_RAD,
            )
        )

        if success:
            runtime.articulation.apply_action(
                action
            )

            runtime.successful_solves += 1

            if runtime.consecutive_failures > 0:
                log(
                    "IK recovered after "
                    f"{runtime.consecutive_failures} failed solve(s)."
                )

            runtime.consecutive_failures = 0
            return True

        runtime.consecutive_failures += 1

        if (
            sim_frame_index
            - runtime.last_warning_frame
            >= IK_WARN_EVERY_SIM_FRAMES
        ):
            runtime.last_warning_frame = (
                sim_frame_index
            )

            warn(
                "IK did not converge. No robot command was applied.\n"
                f"  target position: "
                f"{np.round(target_position, 4).tolist()}\n"
                "  Move /World/IK_Target closer to the current hand "
                "or reduce the requested rotation."
            )

        return False


    def set_external_viewport_camera() -> None:
        try:
            from isaacsim.core.utils.viewports import (
                set_camera_view,
            )

            eye = np.array(
                [3.4, 3.2, 2.7],
                dtype=np.float64,
            )

            target = np.array(
                [0.25, 0.0, 1.0],
                dtype=np.float64,
            )

            set_camera_view(
                eye=eye,
                target=target,
                camera_prim_path="/OmniverseKit_Persp",
            )

            log(
                "Camera set:\n"
                f"  eye:    {eye.tolist()}\n"
                f"  target: {target.tolist()}"
            )

        except Exception as exc:
            warn(f"Could not set viewport camera: {exc}")


    # -----------------------------------------------------------------
    # Scene construction
    # -----------------------------------------------------------------

    def build_scene() -> tuple[CameraSensor, IKRuntime, str]:
        log("Creating a new stage...")

        omni.usd.get_context().new_stage()

        for _ in range(5):
            simulation_app.update()

        stage = omni.usd.get_context().get_stage()

        if stage is None:
            raise RuntimeError(
                "Isaac Sim did not create a valid stage."
            )

        # Main scene uses meters and Z-up.
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(
            stage,
            UsdGeom.Tokens.z,
        )

        GroundPlane("/World/GroundPlane")

        light = DomeLight("/World/DomeLight")
        light.set_intensities(1000)

        # -------------------------------------------------------------
        # Rack
        # -------------------------------------------------------------

        define_xform(
            prim_path=RACK_CONTAINER_PATH,
            position=(0.0, 0.0, 0.0),
            yaw_deg=RACK_YAW_DEG,
            scale=(
                RACK_SCALE,
                RACK_SCALE,
                RACK_SCALE,
            ),
        )

        add_reference_checked(
            usd_path=RACK_USD_PATH,
            prim_path=RACK_ASSET_PATH,
        )

        center_rack_on_ground()

        # -------------------------------------------------------------
        # Franka
        # -------------------------------------------------------------

        assets_root = get_assets_root_path()

        if assets_root is None:
            raise RuntimeError(
                "Could not resolve the Isaac Sim assets root."
            )

        franka_usd = (
            assets_root
            + "/Isaac/Robots/FrankaRobotics/"
            "FrankaPanda/franka.usd"
        )

        define_xform(
            prim_path=FRANKA_CONTAINER_PATH,
            position=FRANKA_POSITION,
            yaw_deg=FRANKA_YAW_DEG,
            scale=(1.0, 1.0, 1.0),
        )

        add_reference_checked(
            usd_path=franka_usd,
            prim_path=FRANKA_ASSET_PATH,
        )

        # -------------------------------------------------------------
        # Eye-in-hand camera
        # -------------------------------------------------------------

        (
            hand_camera_path,
            hand_rtx_camera,
        ) = create_hand_camera()

        hand_camera_sensor = create_hand_camera_sensor(
            camera_path=hand_camera_path,
            rtx_camera=hand_rtx_camera,
        )

        # -------------------------------------------------------------
        # Physics
        # -------------------------------------------------------------

        log("Initializing physics...")

        SimulationManager.setup_simulation(
            dt=PHYSICS_DT,
            device=DEVICE,
        )

        scenes = SimulationManager.get_physics_scenes()

        if not scenes:
            raise RuntimeError(
                "No physics scene was created."
            )

        scenes[0].set_enabled_gpu_dynamics(False)

        app_utils.play()
        app_utils.update_app(steps=30)

        # -------------------------------------------------------------
        # Interactive Lula IK target
        # -------------------------------------------------------------

        ik_runtime = create_ik_runtime(
            assets_root
        )

        # -------------------------------------------------------------
        # Final verification
        # -------------------------------------------------------------

        rack_bounds = report_bounds(
            "Server rack final",
            RACK_CONTAINER_PATH,
        )

        franka_bounds = report_bounds(
            "Franka final",
            FRANKA_CONTAINER_PATH,
        )

        articulation_roots = find_articulation_roots(
            FRANKA_ASSET_PATH
        )

        if not articulation_roots:
            raise RuntimeError(
                "No Franka articulation root was found."
            )

        log(
            "Franka articulation root(s):\n  "
            + "\n  ".join(articulation_roots)
        )

        verify_hand_camera_mount(
            hand_camera_path
        )

        rack_size = rack_bounds[2]

        log(
            "SCENE READY\n"
            f"  Rack USD:       {RACK_USD_PATH}\n"
            f"  Rack size:      "
            f"{np.round(rack_size, 4).tolist()} m\n"
            f"  Rack scale:     {RACK_SCALE}\n"
            f"  Franka pose:    "
            f"position={FRANKA_POSITION}, "
            f"yaw={FRANKA_YAW_DEG} deg\n"
            f"  Hand camera:    {hand_camera_path}\n"
            f"  Camera mount:   over-hand, "
            f"{HAND_CAMERA_DOWNWARD_TILT_DEG:.1f} deg downward\n"
            f"  Camera offset:  {HAND_CAMERA_LOCAL_POSITION}\n"
            f"  Camera POV:     editable\n"
            f"  IK target:      {IK_TARGET_PATH}\n"
            f"  IK tracking:    {IK_TRACKING_ENABLED}\n"
            f"  IK start pos:   {IK_TARGET_INITIAL_POSITION}"
        )

        set_external_viewport_camera()

        return (
            hand_camera_sensor,
            ik_runtime,
            hand_camera_path,
        )


    (
        hand_camera_sensor,
        ik_runtime,
        hand_camera_path,
    ) = build_scene()

    sim_frame_index = 0
    capture_index = 0

    while simulation_app.is_running():
        # Render/update first so CameraSensor.get_data() sees the newest frame.
        simulation_app.update()

        sim_frame_index += 1

        try:
            update_ik_target_tracking(
                runtime=ik_runtime,
                sim_frame_index=sim_frame_index,
            )
        except Exception as exc:
            warn(
                "IK tracking update failed: "
                f"{exc}"
            )

        if (
            sim_frame_index
            % CAPTURE_EVERY_SIM_FRAMES
            == 0
        ):
            capture_index += 1

            try:
                capture_rgb_and_depth(
                    sensor=hand_camera_sensor,
                    capture_index=capture_index,
                    camera_path=hand_camera_path,
                )
            except Exception as exc:
                warn(
                    "RGB/depth capture failed on "
                    f"capture {capture_index}: {exc}"
                )


except Exception:
    print(
        "\n[SINGLE RACK SCENE] FATAL ERROR\n"
        + traceback.format_exc(),
        flush=True,
    )
    raise


finally:
    try:
        app_utils.stop()
    except Exception:
        pass

    simulation_app.close()
