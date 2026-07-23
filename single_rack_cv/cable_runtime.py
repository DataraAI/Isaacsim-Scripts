#!/usr/bin/env python3
"""Cable-mounted startup wrapper around the canonical visual-servo runtime."""

from __future__ import annotations
import os

import numpy as np
import omni.usd
from pxr import UsdGeom

import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import DomeLight, GroundPlane
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.rtx import CameraSensor
from isaacsim.storage.native import get_assets_root_path

import sim as sim_module
from cable_geometry import validate_mount_window
from host_array_bridge import HostSafePoseObject, to_numpy_cpu
from scale_aware_cable_mount import ScaleAwareCableMount
from sim import (
    SimulationRuntime,
    hand_pose_to_tool_pose,
    log,
    quaternion_wxyz_to_matrix,
)


class CableMountedSimulationRuntime(SimulationRuntime):
    """Canonical controller with cable-specific pre-play scene authoring."""

    def __init__(self, simulation_app, cfg):
        self.cable_mount: ScaleAwareCableMount | None = None
        self.physics_scene = None
        super().__init__(simulation_app=simulation_app, cfg=cfg)

    def _create_ik(self, assets_root: str):
        """Bridge every CUDA pose source used by the NumPy/Lula controller."""

        original_set_robot_base_pose = (
            sim_module.LulaKinematicsSolver.set_robot_base_pose
        )

        def host_safe_set_robot_base_pose(
            solver,
            robot_position,
            robot_orientation,
        ):
            return original_set_robot_base_pose(
                solver,
                to_numpy_cpu(
                    robot_position,
                    shape=(3,),
                    label="Lula robot base position",
                ),
                to_numpy_cpu(
                    robot_orientation,
                    shape=(4,),
                    label="Lula robot base orientation",
                ),
            )

        sim_module.LulaKinematicsSolver.set_robot_base_pose = (
            host_safe_set_robot_base_pose
        )
        try:
            runtime = super()._create_ik(assets_root)
        finally:
            sim_module.LulaKinematicsSolver.set_robot_base_pose = (
                original_set_robot_base_pose
            )

        runtime.articulation = HostSafePoseObject(
            runtime.articulation,
            label="Franka articulation",
        )
        runtime.target = HostSafePoseObject(
            runtime.target,
            label="IK target",
        )
        runtime.actual_tool = HostSafePoseObject(
            runtime.actual_tool,
            label="actual ToolCenter",
        )
        return runtime

    def _build_scene(self) -> None:
        scene = self.cfg.scene

        if not os.path.isfile(scene.rack_usd_path):
            raise FileNotFoundError(
                f"Rack USD not found: {scene.rack_usd_path}"
            )

        log("Creating cable-mounted stage")
        omni.usd.get_context().new_stage()
        self._update_app(5)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not create a valid stage.")

        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        GroundPlane("/World/GroundPlane")
        light = DomeLight("/World/DomeLight")
        light.set_intensities(scene.light_intensity)

        self._define_xform(
            scene.rack_path,
            position=(0.0, 0.0, 0.0),
            yaw_deg=scene.rack_yaw_deg,
            scale=(scene.rack_scale,) * 3,
        )
        self._add_reference(scene.rack_usd_path, scene.rack_asset_path)
        self._center_rack()

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
        self.physics_scene = physics_scenes[0]
        self.physics_scene.set_enabled_gpu_dynamics(True)
        self.physics_scene.set_broadphase_type("GPU")
        self.physics_scene.set_solver_type("TGS")
        if not self.physics_scene.get_enabled_gpu_dynamics():
            raise RuntimeError("Cable mount requires GPU dynamics")
        if self.physics_scene.get_broadphase_type() != "GPU":
            raise RuntimeError("Cable mount requires GPU broadphase")
        if self.physics_scene.get_solver_type() != "TGS":
            raise RuntimeError("Cable mount requires TGS")

        hand_path = self._find_unique_descendant(
            self.cfg.scene.franka_asset_path,
            self.cfg.cable_mount.hand_link_name,
        )
        hand_position, hand_orientation = self._get_world_pose(hand_path)
        tool_position, tool_orientation = hand_pose_to_tool_pose(
            hand_position_m=hand_position,
            hand_orientation_wxyz=hand_orientation,
            tool_local_position_m=np.asarray(
                self.cfg.ik.tool_center_local_position_m,
                dtype=np.float64,
            ),
            tool_local_orientation_wxyz=np.asarray(
                self.cfg.ik.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            ),
        )
        world_from_toolcenter = np.eye(4, dtype=np.float64)
        world_from_toolcenter[:3, :3] = quaternion_wxyz_to_matrix(
            tool_orientation
        )
        world_from_toolcenter[:3, 3] = tool_position

        if self.cfg.cable_mount.enabled:
            self.cable_mount = ScaleAwareCableMount(self.cfg)
            self.cable_mount.author_before_play(
                stage=stage,
                hand_path=hand_path,
                world_from_toolcenter=world_from_toolcenter,
            )

        app_utils.play()
        app_utils.update_app(steps=30)

        self.ik = self._create_ik(assets_root)
        if self.cable_mount is not None:
            self.cable_mount.configure_fingers(self.ik.articulation)
        self._set_external_view()

        log(
            "READY\n"
            f"  rack:       {scene.rack_usd_path}\n"
            f"  Franka:     pos={scene.franka_position}, "
            f"yaw={scene.franka_yaw_deg}°\n"
            f"  cable:      {self.cfg.cable_mount.usd_path}\n"
            f"  tracked plug:{self.cfg.cable_mount.tracked_plug_path}\n"
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
            f"  desired port in virtual center eye: "
            f"{np.round(self.desired_port_virtual_camera_usd, 5).tolist()}"
        )

    def prepare_for_perception(self) -> None:
        """Settle and validate the mount before any detector is initialized."""

        if self.cable_mount is None:
            return
        cfg = self.cfg.cable_mount
        samples: list[tuple[float, float]] = []
        max_prepare_frames = (
            cfg.initial_settle_frames
            + cfg.validation_frames
            + 600
        )
        for frame_count in range(max_prepare_frames):
            self.step()
            self.update_ik()
            self._update_startup_settle()
            if frame_count < cfg.initial_settle_frames:
                continue
            if not self.visual_servo.startup_ready:
                continue
            samples.append(self.cable_mount.sample_validation(self))
            if len(samples) == cfg.validation_frames:
                break
        else:
            raise RuntimeError(
                "Cable mount did not settle and validate within the startup "
                "frame cap"
            )

        validation = validate_mount_window(
            samples,
            cfg.validation_frames,
            cfg.max_tip_error_m,
            cfg.max_axis_error_deg,
        )
        self.cable_mount.log_success(validation)
