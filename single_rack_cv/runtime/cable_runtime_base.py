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
from cable.cable_geometry import validate_mount_window
from robot.host_array_bridge import (
    HostSafePoseObject,
    install_host_safe_ik_warm_start,
    to_numpy_cpu,
)
from cable.scale_aware_cable_mount import ScaleAwareCableMount
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

        runtime.articulation_solver = install_host_safe_ik_warm_start(
            runtime.articulation_solver
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
        print("[DEBUG] about to call new_stage() (guarded)", flush=True)
        if not self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
            omni.usd.get_context().new_stage()
        print("[DEBUG] new_stage() (guarded) returned", flush=True)
        print("[DEBUG] about to call _update_app(5)", flush=True)
        self._update_app(5)
        print("[DEBUG] _update_app(5) returned", flush=True)

        print("[DEBUG] about to call get_stage()", flush=True)
        stage = omni.usd.get_context().get_stage()
        print("[DEBUG] get_stage() returned", flush=True)
        if stage is None:
            raise RuntimeError("Isaac Sim did not create a valid stage.")

        print("[DEBUG] about to call SetStageMetersPerUnit()", flush=True)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        print("[DEBUG] SetStageMetersPerUnit() returned", flush=True)
        print("[DEBUG] about to call SetStageUpAxis()", flush=True)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        print("[DEBUG] SetStageUpAxis() returned", flush=True)

        print("[DEBUG] about to call GroundPlane() (guarded)", flush=True)
        if not self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
            GroundPlane("/World/GroundPlane")
        print("[DEBUG] GroundPlane() (guarded) returned", flush=True)
        print("[DEBUG] about to call DomeLight()", flush=True)
        light = DomeLight("/World/DomeLight")
        print("[DEBUG] DomeLight() returned", flush=True)
        print("[DEBUG] about to call light.set_intensities()", flush=True)
        light.set_intensities(scene.light_intensity)
        print("[DEBUG] light.set_intensities() returned", flush=True)

        print("[DEBUG] about to enter rack spawn block (guarded)", flush=True)
        if self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
            print(
                "[DEBUG] rack already present from esha's datahall load, "
                "skipping define_xform/add_reference",
                flush=True,
            )
            # SceneConfig is frozen; mutate in place so _center_rack() and
            # later readers see the datahall-embedded rack prim paths.
            object.__setattr__(
                scene,
                "rack_path",
                "/World/DataHall/DataHall_01/DataHall_01/DataHall_Racks/Rack_42U_01",
            )
            object.__setattr__(scene, "rack_asset_path", scene.rack_path)
        else:
            self._define_xform(
                scene.rack_path,
                position=(0.0, 0.0, 0.0),
                yaw_deg=scene.rack_yaw_deg,
                scale=(scene.rack_scale,) * 3,
            )
            self._add_reference(scene.rack_usd_path, scene.rack_asset_path)
        print("[DEBUG] about to call _center_rack()", flush=True)
        self._center_rack()
        print("[DEBUG] _center_rack() returned", flush=True)

        print("[DEBUG] about to call get_assets_root_path()", flush=True)
        assets_root = get_assets_root_path()
        print("[DEBUG] get_assets_root_path() returned", flush=True)
        if assets_root is None:
            raise RuntimeError("Could not resolve Isaac Sim assets root.")

        print("[DEBUG] about to enter Franka spawn block (guarded)", flush=True)
        if not self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
            franka_usd = (
                assets_root
                + "/Isaac/Robots/FrankaRobotics/"
                "FrankaPanda/franka.usd"
            )
            print("[DEBUG] about to call _define_xform(franka)", flush=True)
            self._define_xform(
                scene.franka_path,
                position=scene.franka_position,
                yaw_deg=scene.franka_yaw_deg,
                scale=(1.0, 1.0, 1.0),
            )
            print("[DEBUG] _define_xform(franka) returned", flush=True)
            print("[DEBUG] about to call _add_reference(franka)", flush=True)
            self._add_reference(franka_usd, scene.franka_asset_path)
            print("[DEBUG] _add_reference(franka) returned", flush=True)
            print("[DEBUG] about to call _configure_franka_gravity()", flush=True)
            self._configure_franka_gravity()
            print("[DEBUG] _configure_franka_gravity() returned", flush=True)
            print(
                "[DEBUG] about to call _configure_franka_arm_drives()",
                flush=True,
            )
            self._configure_franka_arm_drives()
            print(
                "[DEBUG] _configure_franka_arm_drives() returned",
                flush=True,
            )
        print("[DEBUG] exited Franka spawn block (guarded)", flush=True)

        left_camera_name = (
            self.cfg.camera.left_camera_name + "_insertion"
            if self.cfg.cable_mount.already_grasped_by_pickup_pipeline
            else self.cfg.camera.left_camera_name
        )
        right_camera_name = (
            self.cfg.camera.right_camera_name + "_insertion"
            if self.cfg.cable_mount.already_grasped_by_pickup_pipeline
            else self.cfg.camera.right_camera_name
        )

        print("[DEBUG] about to call _create_hand_camera(left)", flush=True)
        (
            self.left_camera_path,
            left_rtx_camera,
        ) = self._create_hand_camera(
            left_camera_name,
            self.cfg.camera.left_local_position,
            "left",
        )
        print("[DEBUG] _create_hand_camera(left) returned", flush=True)
        print("[DEBUG] about to call _create_hand_camera(right)", flush=True)
        (
            self.right_camera_path,
            right_rtx_camera,
        ) = self._create_hand_camera(
            right_camera_name,
            self.cfg.camera.right_local_position,
            "right",
        )
        print("[DEBUG] _create_hand_camera(right) returned", flush=True)
        print("[DEBUG] about to call CameraSensor(left)", flush=True)
        self.left_camera_sensor = CameraSensor(
            left_rtx_camera,
            resolution=self.cfg.camera.resolution,
            annotators=["rgb"],
        )
        print("[DEBUG] CameraSensor(left) returned", flush=True)
        print("[DEBUG] about to call CameraSensor(right)", flush=True)
        self.right_camera_sensor = CameraSensor(
            right_rtx_camera,
            resolution=self.cfg.camera.resolution,
            annotators=["rgb"],
        )
        print("[DEBUG] CameraSensor(right) returned", flush=True)
        print("[DEBUG] about to call output_dir.mkdir()", flush=True)
        self.cfg.camera.output_dir.mkdir(parents=True, exist_ok=True)
        print("[DEBUG] output_dir.mkdir() returned", flush=True)

        print("[DEBUG] about to call setup_simulation()", flush=True)
        SimulationManager.setup_simulation(
            dt=scene.physics_dt,
            device=scene.device,
        )
        print("[DEBUG] setup_simulation() returned", flush=True)
        print("[DEBUG] about to call get_physics_scenes()", flush=True)
        physics_scenes = SimulationManager.get_physics_scenes()
        print("[DEBUG] get_physics_scenes() returned", flush=True)
        if not physics_scenes:
            raise RuntimeError("No physics scene was created.")
        self.physics_scene = physics_scenes[0]
        print("[DEBUG] about to call set_enabled_gpu_dynamics()", flush=True)
        self.physics_scene.set_enabled_gpu_dynamics(True)
        print("[DEBUG] set_enabled_gpu_dynamics() returned", flush=True)
        print("[DEBUG] about to call set_broadphase_type()", flush=True)
        self.physics_scene.set_broadphase_type("GPU")
        print("[DEBUG] set_broadphase_type() returned", flush=True)
        print("[DEBUG] about to call set_solver_type()", flush=True)
        self.physics_scene.set_solver_type("TGS")
        print("[DEBUG] set_solver_type() returned", flush=True)
        if not self.physics_scene.get_enabled_gpu_dynamics():
            raise RuntimeError("Cable mount requires GPU dynamics")
        if self.physics_scene.get_broadphase_type() != "GPU":
            raise RuntimeError("Cable mount requires GPU broadphase")
        if self.physics_scene.get_solver_type() != "TGS":
            raise RuntimeError("Cable mount requires TGS")

        print("[DEBUG] about to call _find_unique_descendant()", flush=True)
        hand_path = self._find_unique_descendant(
            self.cfg.scene.franka_asset_path,
            self.cfg.cable_mount.hand_link_name,
        )
        print("[DEBUG] _find_unique_descendant() returned", flush=True)
        print("[DEBUG] about to call _get_world_pose(hand_path)", flush=True)
        hand_position, hand_orientation = self._get_world_pose(hand_path)
        print("[DEBUG] _get_world_pose(hand_path) returned", flush=True)
        print("[DEBUG] about to call hand_pose_to_tool_pose()", flush=True)
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
        print("[DEBUG] hand_pose_to_tool_pose() returned", flush=True)
        world_from_toolcenter = np.eye(4, dtype=np.float64)
        print("[DEBUG] about to call quaternion_wxyz_to_matrix()", flush=True)
        world_from_toolcenter[:3, :3] = quaternion_wxyz_to_matrix(
            tool_orientation
        )
        print("[DEBUG] quaternion_wxyz_to_matrix() returned", flush=True)
        world_from_toolcenter[:3, 3] = tool_position

        print("[DEBUG] about to enter cable_mount block", flush=True)
        if self.cfg.cable_mount.enabled:
            print("[DEBUG] about to call ScaleAwareCableMount()", flush=True)
            self.cable_mount = ScaleAwareCableMount(self.cfg)
            print("[DEBUG] ScaleAwareCableMount() returned", flush=True)
            if self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
                print(
                    "[DEBUG] about to call author_from_existing_grasp()",
                    flush=True,
                )
                self.cable_mount.author_from_existing_grasp(
                    stage=stage,
                    hand_path=hand_path,
                )
                print(
                    "[DEBUG] author_from_existing_grasp() returned",
                    flush=True,
                )
            else:
                print("[DEBUG] about to call author_before_play()", flush=True)
                self.cable_mount.author_before_play(
                    stage=stage,
                    hand_path=hand_path,
                    world_from_toolcenter=world_from_toolcenter,
                )
                print("[DEBUG] author_before_play() returned", flush=True)
        print("[DEBUG] exited cable_mount block", flush=True)

        print("[DEBUG] about to call app_utils.play()", flush=True)
        app_utils.play()
        print("[DEBUG] app_utils.play() returned", flush=True)
        print("[DEBUG] about to call app_utils.update_app(steps=30)", flush=True)
        app_utils.update_app(steps=30)
        print("[DEBUG] app_utils.update_app(steps=30) returned", flush=True)

        print("[DEBUG] about to call _create_ik()", flush=True)
        self.ik = self._create_ik(assets_root)
        print("[DEBUG] _create_ik() returned", flush=True)
        if self.cable_mount is not None:
            print("[DEBUG] about to call configure_fingers()", flush=True)
            self.cable_mount.configure_fingers(self.ik.articulation)
            print("[DEBUG] configure_fingers() returned", flush=True)
        print("[DEBUG] about to call _set_external_view()", flush=True)
        self._set_external_view()
        print("[DEBUG] _set_external_view() returned", flush=True)

        print("[DEBUG] about to call log(READY)", flush=True)
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
        print("[DEBUG] log(READY) returned", flush=True)
        print("[DEBUG] _build_scene() complete", flush=True)

    def _log_startup_diagnostics(
        self,
        *,
        frame_count: int,
        minimum_tool_error_m: float,
        maximum_tool_error_m: float,
        validation_sample_count: int,
    ) -> float:
        if self.ik is None or self.cable_mount is None:
            raise RuntimeError("Startup diagnostics require IK and cable mount")

        self._update_actual_tool_frame(self.ik)
        target_position, _ = self.ik.target.get_world_pose()
        actual_position, _ = self.ik.actual_tool.get_world_pose()
        hand_position, _ = self._get_world_pose(self.ik.hand_path)
        target = np.asarray(target_position, dtype=np.float64)
        actual = np.asarray(actual_position, dtype=np.float64)
        hand = np.asarray(hand_position, dtype=np.float64)
        current_tool_error_m = float(np.linalg.norm(actual - target))

        try:
            mount_tip_error_m, mount_axis_error_deg = (
                self.cable_mount.sample_validation(self)
            )
            mount_status = (
                f"  plug-tip to ToolCenter error: "
                f"{mount_tip_error_m * 1000.0:.6f} mm\n"
                f"  plug-axis error: {mount_axis_error_deg:.6f} deg"
            )
        except Exception as error:
            mount_status = (
                "  mount sample failed: "
                f"{type(error).__name__}: {error}"
            )

        log(
            "CABLE STARTUP DIAGNOSTICS\n"
            f"  prepare frame: {frame_count}\n"
            f"  startup ready: {self.visual_servo.startup_ready}\n"
            f"  settled frames: "
            f"{self.visual_servo.startup_settled_frame_count}/"
            f"{self.cfg.visual_servo.required_startup_settled_frames}\n"
            f"  ToolCenter error current/min/max: "
            f"{current_tool_error_m * 1000.0:.6f} / "
            f"{minimum_tool_error_m * 1000.0:.6f} / "
            f"{maximum_tool_error_m * 1000.0:.6f} mm\n"
            f"  target ToolCenter: {np.round(target, 7).tolist()}\n"
            f"  actual ToolCenter: {np.round(actual, 7).tolist()}\n"
            f"  panda_hand: {np.round(hand, 7).tolist()}\n"
            f"{mount_status}\n"
            f"  fixed joint valid: "
            f"{self.cable_mount.fixed_joint_is_valid()}\n"
            f"  built-in attachment preserved: "
            f"{self.cable_mount.built_in_attachment_is_preserved()}\n"
            f"  validation samples collected: {validation_sample_count}/"
            f"{self.cfg.cable_mount.validation_frames}"
        )
        return current_tool_error_m

    def prepare_for_perception(self) -> None:
        if self.cable_mount is None:
            return
        cfg = self.cfg.cable_mount
        samples: list[tuple[float, float]] = []
        max_prepare_frames = (
            cfg.initial_settle_frames
            + cfg.validation_frames
            + 600
        )
        minimum_tool_error_m = float("inf")
        maximum_tool_error_m = 0.0
        current_tool_error_m = float("inf")

        for frame_count in range(max_prepare_frames):
            self.step()
            self.update_ik()
            self._update_startup_settle()
            current_tool_error_m = self._tool_target_position_error_m()
            minimum_tool_error_m = min(
                minimum_tool_error_m,
                current_tool_error_m,
            )
            maximum_tool_error_m = max(
                maximum_tool_error_m,
                current_tool_error_m,
            )

            if frame_count == 0 or (frame_count + 1) % 120 == 0:
                self._log_startup_diagnostics(
                    frame_count=frame_count + 1,
                    minimum_tool_error_m=minimum_tool_error_m,
                    maximum_tool_error_m=maximum_tool_error_m,
                    validation_sample_count=len(samples),
                )

            if frame_count < cfg.initial_settle_frames:
                continue
            if not self.visual_servo.startup_ready:
                continue
            samples.append(self.cable_mount.sample_validation(self))
            if len(samples) == cfg.validation_frames:
                break
        else:
            self._log_startup_diagnostics(
                frame_count=max_prepare_frames,
                minimum_tool_error_m=minimum_tool_error_m,
                maximum_tool_error_m=maximum_tool_error_m,
                validation_sample_count=len(samples),
            )
            raise RuntimeError(
                "Cable mount startup gate timed out.\n"
                f"  startup ready: {self.visual_servo.startup_ready}\n"
                f"  settled frames: "
                f"{self.visual_servo.startup_settled_frame_count}/"
                f"{self.cfg.visual_servo.required_startup_settled_frames}\n"
                f"  ToolCenter error current/min/max mm: "
                f"{current_tool_error_m * 1000.0:.6f} / "
                f"{minimum_tool_error_m * 1000.0:.6f} / "
                f"{maximum_tool_error_m * 1000.0:.6f}\n"
                f"  validation samples: {len(samples)}/"
                f"{cfg.validation_frames}"
            )

        validation = validate_mount_window(
            samples,
            cfg.validation_frames,
            cfg.max_tip_error_m,
            cfg.max_axis_error_deg,
        )
        self.cable_mount.log_success(validation)
