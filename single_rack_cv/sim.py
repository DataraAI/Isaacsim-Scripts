#!/usr/bin/env python3
"""Isaac Sim scene, hand-camera, RGB-D capture, and draggable Lula IK."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import carb
import numpy as np
import omni.usd

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

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

from config import Config
from perception import (
    CameraFrame,
    CameraModel,
    PortEstimate,
    normalize_depth,
    normalize_rgb,
)


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


def smoothstep01(progress: float) -> float:
    """Clamp progress to [0, 1] and apply cubic smoothstep."""
    t = min(1.0, max(0.0, float(progress)))
    return t * t * (3.0 - 2.0 * t)


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

    if error <= tolerance:
        return current_count + 1

    return 0


@dataclass
class StablePreinsertLatch:
    """Latch the median of one stable window of 3D target samples."""

    required_samples: int
    max_spread_m: float
    samples: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.required_samples <= 0:
            raise ValueError("required_samples must be positive.")
        if self.max_spread_m <= 0.0:
            raise ValueError("max_spread_m must be positive.")

    def reset(self) -> None:
        self.samples.clear()

    def add(
        self,
        point_m: np.ndarray,
    ) -> tuple[bool, np.ndarray | None, float]:
        """
        Add one point and return (latched, median_target, max_spread).

        The window contains only the most recent required_samples points.
        """
        point = np.asarray(point_m, dtype=np.float64)

        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("Pre-insert point must be a finite 3-vector.")

        self.samples.append(point.copy())

        if len(self.samples) > self.required_samples:
            self.samples.pop(0)

        if len(self.samples) < self.required_samples:
            return False, None, math.inf

        stacked = np.vstack(self.samples)
        median = np.median(stacked, axis=0)
        spread = float(
            np.max(
                np.linalg.norm(
                    stacked - median,
                    axis=1,
                )
            )
        )

        if spread > self.max_spread_m:
            return False, None, spread

        return True, median.astype(np.float64), spread


@dataclass
class AutoPreinsertState:
    latch: StablePreinsertLatch
    target_latched: bool = False
    perception_frozen: bool = False
    moving: bool = False
    settling: bool = False
    complete: bool = False

    start_frame: int = 0
    duration_frames: int = 1
    settle_start_frame: int = 0
    settled_frame_count: int = 0
    settle_timeout_reported: bool = False

    start_position_m: np.ndarray | None = None
    target_position_m: np.ndarray | None = None
    held_orientation_wxyz: np.ndarray | None = None


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
        self.camera_path = ""
        self.camera_sensor: CameraSensor | None = None
        self.ik: IKRuntime | None = None

        auto_cfg = cfg.auto_preinsert
        self.auto_preinsert = AutoPreinsertState(
            latch=StablePreinsertLatch(
                required_samples=auto_cfg.required_stable_samples,
                max_spread_m=auto_cfg.max_sample_spread_m,
            )
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
        auto_cfg = self.cfg.auto_preinsert

        if (
            auto_cfg.freeze_perception_after_latch
            and self.auto_preinsert.perception_frozen
        ):
            return False

        interval = self.cfg.camera.capture_every_sim_frames
        return self.frame_index > 0 and self.frame_index % interval == 0

    def note_perception_failure(self) -> None:
        """Require stable samples to be consecutive by clearing on failure."""
        state = self.auto_preinsert

        if state.target_latched or not state.latch.samples:
            return

        state.latch.reset()
        log("Auto pre-insert sample window reset after perception failure.")

    def observe_preinsert_estimate(
        self,
        estimate: PortEstimate,
    ) -> bool:
        """
        Validate and accumulate one estimate; latch and start one move.

        Returns True only on the frame where the automatic target is latched.
        """
        cfg = self.cfg.auto_preinsert
        state = self.auto_preinsert

        if not cfg.enabled or state.target_latched:
            return False

        latched, target_position, spread = state.latch.add(
            estimate.preinsert_world_xyz_m
        )

        sample_count = len(state.latch.samples)

        log(
            "Auto pre-insert sample "
            f"{sample_count}/{cfg.required_stable_samples}: "
            f"point={np.round(estimate.preinsert_world_xyz_m, 5).tolist()}"
        )

        if not latched or target_position is None:
            if math.isfinite(spread):
                log(
                    "Auto pre-insert window spread: "
                    f"{spread * 1000.0:.3f} mm "
                    f"(limit {cfg.max_sample_spread_m * 1000.0:.3f} mm)"
                )
            return False

        if self.ik is None:
            raise RuntimeError("Cannot start automatic motion before IK exists.")

        current_position, current_orientation = (
            self.ik.target.get_world_pose()
        )

        duration_frames = max(
            1,
            int(round(cfg.move_duration_s / self.cfg.scene.physics_dt)),
        )

        state.target_latched = True
        state.perception_frozen = cfg.freeze_perception_after_latch
        state.moving = True
        state.settling = False
        state.complete = False
        state.start_frame = self.frame_index
        state.duration_frames = duration_frames
        state.settle_start_frame = 0
        state.settled_frame_count = 0
        state.settle_timeout_reported = False
        state.start_position_m = np.asarray(
            current_position,
            dtype=np.float64,
        ).copy()
        state.target_position_m = np.asarray(
            target_position,
            dtype=np.float64,
        ).copy()
        state.held_orientation_wxyz = _normalize_quaternion_wxyz(
            np.asarray(
                current_orientation,
                dtype=np.float64,
            )
        )

        log(
            "AUTO PRE-INSERT TARGET LATCHED\n"
            f"  samples:      {cfg.required_stable_samples}\n"
            f"  max spread:   {spread * 1000.0:.3f} mm\n"
            f"  start:        "
            f"{np.round(state.start_position_m, 5).tolist()}\n"
            f"  target:       "
            f"{np.round(state.target_position_m, 5).tolist()}\n"
            f"  move duration:{cfg.move_duration_s:.2f} s\n"
            f"  perception:   "
            f"{'frozen' if state.perception_frozen else 'running'}"
        )

        return True

    def update_auto_preinsert_motion(self) -> None:
        """
        Move the desired tool target, then wait for the actual ToolCenter.

        The target endpoint never moves backward to meet the robot. Lula keeps
        solving the fixed endpoint until the achieved ToolCenter is inside the
        configured tolerance for the required number of consecutive frames.
        """
        state = self.auto_preinsert
        cfg = self.cfg.auto_preinsert

        if not state.moving and not state.settling:
            return

        if self.ik is None:
            raise RuntimeError("Automatic motion requires initialized IK.")

        if (
            state.start_position_m is None
            or state.target_position_m is None
            or state.held_orientation_wxyz is None
        ):
            raise RuntimeError("Automatic motion state is incomplete.")

        if state.moving:
            elapsed = max(0, self.frame_index - state.start_frame)
            progress = min(1.0, elapsed / state.duration_frames)
            blend = smoothstep01(progress)

            target_position = (
                state.start_position_m
                + blend
                * (
                    state.target_position_m
                    - state.start_position_m
                )
            )

            self.ik.target.set_world_pose(
                position=target_position,
                orientation=state.held_orientation_wxyz,
            )

            if progress < 1.0:
                return

            # Hold the exact detected endpoint throughout convergence.
            self.ik.target.set_world_pose(
                position=state.target_position_m,
                orientation=state.held_orientation_wxyz,
            )

            state.moving = False
            state.settling = True
            state.complete = False
            state.settle_start_frame = self.frame_index
            state.settled_frame_count = 0
            state.settle_timeout_reported = False

            log(
                "AUTO PRE-INSERT TARGET MOTION FINISHED\n"
                f"  fixed tool target: "
                f"{np.round(state.target_position_m, 5).tolist()}\n"
                f"  convergence tolerance: "
                f"{cfg.settle_position_tolerance_m * 1000.0:.3f} mm\n"
                f"  required settled frames: "
                f"{cfg.required_settled_frames}"
            )

            return

        # Settling phase: never move the desired target away from its exact
        # endpoint. Only the robot is allowed to converge toward it.
        self.ik.target.set_world_pose(
            position=state.target_position_m,
            orientation=state.held_orientation_wxyz,
        )

        actual_tool_position, _ = self.ik.actual_tool.get_world_pose()
        actual_tool_position = np.asarray(
            actual_tool_position,
            dtype=np.float64,
        )

        position_error_m = float(
            np.linalg.norm(
                actual_tool_position
                - state.target_position_m
            )
        )

        state.settled_frame_count = update_convergence_counter(
            position_error_m=position_error_m,
            tolerance_m=cfg.settle_position_tolerance_m,
            current_count=state.settled_frame_count,
        )

        if state.settled_frame_count >= cfg.required_settled_frames:
            state.settling = False
            state.complete = True

            log(
                "AUTO PRE-INSERT CONVERGED\n"
                f"  target tool center: "
                f"{np.round(state.target_position_m, 6).tolist()}\n"
                f"  actual tool center: "
                f"{np.round(actual_tool_position, 6).tolist()}\n"
                f"  final position error: "
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

        settle_elapsed_frames = (
            self.frame_index
            - state.settle_start_frame
        )

        if (
            settle_elapsed_frames >= timeout_frames
            and not state.settle_timeout_reported
        ):
            state.settle_timeout_reported = True

            warn(
                "ToolCenter has not converged within the warning timeout. "
                "The target remains fixed and IK will keep trying.\n"
                f"  current position error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  required: "
                f"{cfg.settle_position_tolerance_m * 1000.0:.3f} mm"
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

    def capture(self) -> CameraFrame:
        if self.camera_sensor is None:
            raise RuntimeError("CameraSensor is not initialized.")

        rgb_data, _ = self.camera_sensor.get_data("rgb")
        depth_data, _ = self.camera_sensor.get_data(
            "distance_to_image_plane"
        )

        rgb = normalize_rgb(
            self._sensor_to_numpy(rgb_data, "rgb"),
            self.cfg.camera.resolution,
        )
        depth_m = normalize_depth(
            self._sensor_to_numpy(depth_data, "depth"),
            self.cfg.camera.resolution,
        )

        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(self.camera_path)
        camera = UsdGeom.Camera(camera_prim)

        camera_world = UsdGeom.Xformable(
            camera_prim
        ).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        model = CameraModel(
            image_height_px=rgb.shape[0],
            image_width_px=rgb.shape[1],
            focal_length_mm=float(camera.GetFocalLengthAttr().Get()),
            horizontal_aperture_mm=float(
                camera.GetHorizontalApertureAttr().Get()
            ),
            vertical_aperture_mm=float(
                camera.GetVerticalApertureAttr().Get()
            ),
            world_from_camera=np.asarray(
                camera_world,
                dtype=np.float64,
            ),
        )

        return CameraFrame(
            rgb=rgb,
            depth_m=depth_m,
            camera=model,
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

        if not os.path.isfile(scene.rack_usd_path):
            raise FileNotFoundError(
                f"Rack USD not found: {scene.rack_usd_path}"
            )

        log("Creating stage")
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

        self.camera_path, rtx_camera = self._create_hand_camera()
        self.camera_sensor = CameraSensor(
            rtx_camera,
            resolution=self.cfg.camera.resolution,
            annotators=[
                "rgb",
                "distance_to_image_plane",
            ],
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
            f"  rack:       {scene.rack_usd_path}\n"
            f"  Franka:     pos={scene.franka_position}, "
            f"yaw={scene.franka_yaw_deg}°\n"
            f"  hand camera:{self.camera_path}\n"
            f"  tool target:{self.cfg.ik.target_path}\n"
            f"  actual tool:{self.cfg.ik.actual_tool_path}\n"
            f"  auto move:  "
            f"{self.cfg.auto_preinsert.required_stable_samples} stable "
            f"samples, {self.cfg.auto_preinsert.move_duration_s:.1f} s"
        )

    def _create_hand_camera(self) -> tuple[str, RtxCamera]:
        camera_cfg = self.cfg.camera
        hand_path = self._find_unique_descendant(
            self.cfg.scene.franka_asset_path,
            camera_cfg.hand_link_name,
        )
        camera_path = f"{hand_path}/{camera_cfg.camera_name}"

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
                camera_cfg.local_position,
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
            "Hand camera created: "
            f"offset={camera_cfg.local_position}, "
            f"Y={camera_cfg.local_y_rotation_deg}°, "
            f"roll={camera_cfg.local_roll_deg}°"
        )
        return camera_path, rtx_camera

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

    def _center_rack(self) -> None:
        scene = self.cfg.scene
        minimum, maximum = self._world_bounds(scene.rack_path)
        center = (minimum + maximum) / 2.0

        translation = np.array(
            [-center[0], -center[1], -minimum[2]],
            dtype=np.float64,
        )

        stage = omni.usd.get_context().get_stage()
        xform = UsdGeom.Xformable(
            stage.GetPrimAtPath(scene.rack_path)
        )

        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*translation.tolist()))
                self._update_app(10)
                log(
                    "Rack centered: "
                    f"translation={np.round(translation, 4).tolist()}"
                )
                return

        raise RuntimeError("Rack container has no translation op.")

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
