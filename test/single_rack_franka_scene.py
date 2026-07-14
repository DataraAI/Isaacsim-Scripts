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
- A USD camera is rigidly parented to the Franka panda_hand link.
- The hand camera uses an over-the-hand 18-degree oblique view toward the rack.
- Its POV remains editable while the script is running; no per-frame restore is applied.
"""

from __future__ import annotations

import math
import os
import traceback

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
FRANKA_POSITION = (1.35, 0.0, 0.0)
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

HAND_CAMERA_FOCAL_LENGTH_MM = 18.0
HAND_CAMERA_HORIZONTAL_APERTURE_MM = 20.955
HAND_CAMERA_VERTICAL_APERTURE_MM = (
    HAND_CAMERA_HORIZONTAL_APERTURE_MM * 9.0 / 16.0
)
HAND_CAMERA_CLIPPING_RANGE_M = (0.01, 10.0)
HAND_CAMERA_FOCUS_DISTANCE_M = 1.0

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
    import numpy as np
    import omni.usd

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import DomeLight, GroundPlane
    from isaacsim.core.simulation_manager import SimulationManager
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


    def create_hand_camera() -> str:
        """
        Create one eye-in-hand USD Camera under the Franka panda_hand link.

        The camera is authored directly as a child of panda_hand, so its
        transform is inherited from the robot hand automatically.
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

        camera = UsdGeom.Camera.Define(
            stage,
            camera_path,
        )

        camera_prim = camera.GetPrim()

        if not camera_prim.IsValid():
            raise RuntimeError(
                f"Failed to define hand camera at {camera_path}"
            )

        camera_xform = UsdGeom.Xformable(camera_prim)
        camera_xform.ClearXformOpOrder()

        camera_xform.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(
            Gf.Vec3d(*HAND_CAMERA_LOCAL_POSITION)
        )

        local_rotation = Gf.Rotation(
            Gf.Vec3d(0.0, 1.0, 0.0),
            HAND_CAMERA_LOCAL_Y_ROTATION_DEG,
        ).GetQuat()

        camera_xform.AddOrientOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(local_rotation)

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

        # fStop = 0 disables depth-of-field blur, which is preferable for
        # the later computer-vision port-detection pipeline.
        camera.CreateFStopAttr().Set(0.0)

        for _ in range(5):
            simulation_app.update()

        log(
            "Hand camera created:\n"
            f"  hand link:      {hand_path}\n"
            f"  camera prim:    {camera_path}\n"
            f"  local position: {HAND_CAMERA_LOCAL_POSITION}\n"
            f"  local Y rot:    "
            f"{HAND_CAMERA_LOCAL_Y_ROTATION_DEG} deg\n"
            f"  downward tilt:  "
            f"{HAND_CAMERA_DOWNWARD_TILT_DEG} deg\n"
            f"  focal length:   "
            f"{HAND_CAMERA_FOCAL_LENGTH_MM} mm\n"
            f"  clipping range: "
            f"{HAND_CAMERA_CLIPPING_RANGE_M} m\n"
            f"  POV editable:   True"
        )

        return camera_path



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

    def build_scene() -> None:
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

        hand_camera_path = create_hand_camera()

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
            f"  Camera POV:     editable"
        )

        set_external_viewport_camera()


    build_scene()

    while simulation_app.is_running():
        # No camera-pose restore is applied here. The hand camera remains
        # parented to panda_hand, but its local POV can be edited freely.
        simulation_app.update()


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
