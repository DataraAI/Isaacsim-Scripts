#!/usr/bin/env python3
"""Cable-mounted startup wrapper around the canonical visual-servo runtime."""

from __future__ import annotations
import os

import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics

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
from cable.cable_mount import _world_transform
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

    def _maybe_recompute_ik_target(self) -> None:
        """No-op by default. Overridden for the already_grasped case to
        correct cfg.ik.initial_position/initial_orientation_wxyz using
        the real measured grasp, instead of the fixed calibration
        constant. Runs after author_from_existing_grasp() and before
        _create_ik()."""

    def _sample_tracked_plug_tip_and_axis_from_stage(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """USD-stage sample of tracked plug tip/nose (same math as live PhysX).

        Returns None if mount/plug_frame/stage are not ready yet.
        """

        if (
            self.cable_mount is None
            or self.cable_mount.stage is None
            or self.cable_mount.plug_frame is None
        ):
            return None
        world_from_plug = _world_transform(
            self.cable_mount.stage,
            self.cfg.cable_mount.tracked_plug_path,
        )
        tip_world = (
            world_from_plug
            @ np.r_[self.cable_mount.plug_frame.tip_local_m, 1.0]
        )[:3]
        nose_world = (
            world_from_plug[:3, :3]
            @ self.cable_mount.plug_frame.nose_axis_local
        )
        nose_norm = float(np.linalg.norm(nose_world))
        if nose_norm <= 1.0e-12:
            return tip_world, np.zeros(3, dtype=np.float64)
        return tip_world, nose_world / nose_norm

    def _dump_fixed_joint_break_and_compliance(self, joint_prim) -> None:
        """Report break thresholds, drives, and PhysxJointAPI attrs (no fix)."""

        from pxr import PhysxSchema

        joint = UsdPhysics.FixedJoint(joint_prim)
        bf_attr = joint.GetBreakForceAttr()
        bt_attr = joint.GetBreakTorqueAttr()
        bf = bf_attr.Get() if bf_attr.IsValid() else None
        bt = bt_attr.Get() if bt_attr.IsValid() else None
        bf_authored = (
            bf_attr.HasAuthoredValueOpinion() if bf_attr.IsValid() else False
        )
        bt_authored = (
            bt_attr.HasAuthoredValueOpinion() if bt_attr.IsValid() else False
        )

        def _finite_report(value) -> str:
            if value is None:
                return "None"
            try:
                f = float(value)
            except (TypeError, ValueError):
                return repr(value)
            if not np.isfinite(f):
                return f"{f} (non-finite / never-breaks)"
            # PhysX often uses FLT_MAX (~3.4e38) as "infinite"
            if f >= 1.0e30:
                return f"{f:.6e} (effectively infinite)"
            return f"{f:.6e} (FINITE break threshold)"

        schemas = list(joint_prim.GetAppliedSchemas())
        drive_schemas = [
            s for s in schemas if "Drive" in s or "drive" in s.lower()
        ]
        physx_joint = None
        physx_joint_attrs = {}
        if joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
            physx_joint = PhysxSchema.PhysxJointAPI(joint_prim)
            for name in (
                "physxJoint:armature",
                "physxJoint:jointFriction",
                "physxJoint:maxJointVelocity",
            ):
                attr = joint_prim.GetAttribute(name)
                if attr.IsValid():
                    physx_joint_attrs[name] = {
                        "value": attr.Get(),
                        "authored": attr.HasAuthoredValueOpinion(),
                    }

        # Any LimitAPI / DriveAPI properties authored on this prim.
        limit_or_drive = []
        for attr in joint_prim.GetAttributes():
            name = str(attr.GetName())
            if (
                "drive" in name.lower()
                or "stiffness" in name.lower()
                or "damping" in name.lower()
                or "limit" in name.lower()
            ) and attr.HasAuthoredValueOpinion():
                limit_or_drive.append(f"{name}={attr.Get()!r}")

        print(
            "[DEBUG] FixedJoint break/compliance dump:\n"
            f"  path={joint_prim.GetPath()}\n"
            f"  typeName={joint_prim.GetTypeName()}\n"
            f"  schemas={schemas}\n"
            f"  physics:breakForce={_finite_report(bf)} "
            f"authored={bf_authored}\n"
            f"  physics:breakTorque={_finite_report(bt)} "
            f"authored={bt_authored}\n"
            f"  drive-like schemas={drive_schemas or 'none'}\n"
            f"  PhysxJointAPI applied="
            f"{joint_prim.HasAPI(PhysxSchema.PhysxJointAPI)}\n"
            f"  PhysxJointAPI attrs={physx_joint_attrs or 'none/default'}\n"
            f"  authored drive/stiffness/damping/limit attrs="
            f"{limit_or_drive or 'none'}",
            flush=True,
        )

    def _subscribe_joint_break_events(self, expected_joint_path: str) -> None:
        """Listen for PhysX JOINT_BREAK events for the grasp FixedJoint."""

        try:
            from omni.physx import get_physx_interface
            from omni.physx.bindings._physx import SimulationEvent
            from pxr import PhysicsSchemaTools
        except Exception as error:
            print(
                "[DEBUG] joint-break subscribe unavailable: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return

        self._joint_break_events = []
        expected = str(expected_joint_path)

        def _on_simulation_event(event) -> None:
            if event.type != int(SimulationEvent.JOINT_BREAK):
                return
            try:
                joint_path = str(
                    PhysicsSchemaTools.decodeSdfPath(
                        event.payload["jointPath"][0],
                        event.payload["jointPath"][1],
                    )
                )
            except Exception as error:
                joint_path = f"<decode-failed:{type(error).__name__}>"
            payload_keys = list(event.payload.keys()) if event.payload else []
            entry = {
                "joint_path": joint_path,
                "matches_expected": joint_path == expected,
                "payload_keys": payload_keys,
                "orient_cmp_n": getattr(self, "_plug_orient_cmp_count", None),
            }
            # Capture any force/torque fields if present in this PhysX version.
            for key in payload_keys:
                try:
                    entry[f"payload:{key}"] = event.payload[key]
                except Exception:
                    pass
            self._joint_break_events.append(entry)
            print(
                "[DEBUG] JOINT_BREAK event received:\n"
                f"  joint_path={joint_path}\n"
                f"  matches_CableGraspFixedJoint={joint_path == expected}\n"
                f"  at_orient_cmp_n={entry['orient_cmp_n']}\n"
                f"  payload_keys={payload_keys}",
                flush=True,
            )

        events = get_physx_interface().get_simulation_event_stream_v2()
        self._joint_break_sub = events.create_subscription_to_pop(
            _on_simulation_event
        )
        print(
            "[DEBUG] subscribed to PhysX SimulationEvent.JOINT_BREAK "
            f"for expected joint={expected}",
            flush=True,
        )

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
        print(
            f"[DEBUG] SimulationManager backend={SimulationManager.get_backend()!r} "
            f"device={SimulationManager.get_device()!r}",
            flush=True,
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
        self._joint_break_events: list[dict] = []
        self._joint_break_sub = None
        if (
            self.cable_mount is not None
            and self.cfg.cable_mount.already_grasped_by_pickup_pipeline
        ):
            stage = omni.usd.get_context().get_stage()
            plug_path = self.cfg.cable_mount.tracked_plug_path
            joint_path = self.cfg.cable_mount.fixed_joint_path
            plug = stage.GetPrimAtPath(plug_path) if stage else None
            joint_prim = stage.GetPrimAtPath(joint_path) if stage else None
            kin_report = "no-stage"
            attach_report = "n/a"
            if plug is not None and plug.IsValid():
                has_rb = plug.HasAPI(UsdPhysics.RigidBodyAPI)
                if has_rb:
                    rb = UsdPhysics.RigidBodyAPI(plug)
                    kattr = rb.GetKinematicEnabledAttr()
                    kin_val = kattr.Get() if kattr.IsValid() else None
                    kin_authored = (
                        kattr.HasAuthoredValueOpinion()
                        if kattr.IsValid()
                        else False
                    )
                    kin_report = (
                        f"HasRigidBodyAPI=True kinematicEnabled={kin_val!r} "
                        f"authored={kin_authored} "
                        f"schemas={list(plug.GetAppliedSchemas())}"
                    )
                else:
                    kin_report = (
                        f"HasRigidBodyAPI=False schemas="
                        f"{list(plug.GetAppliedSchemas())}"
                    )
            if joint_prim is not None and joint_prim.IsValid():
                joint = UsdPhysics.FixedJoint(joint_prim)
                body0_paths = [
                    str(p) for p in joint.GetBody0Rel().GetTargets()
                ]
                body1_paths = [
                    str(p) for p in joint.GetBody1Rel().GetTargets()
                ]
                raw_b0 = joint_prim.GetRelationship("physics:body0")
                raw_b1 = joint_prim.GetRelationship("physics:body1")
                raw_b0_t = (
                    [str(p) for p in raw_b0.GetTargets()]
                    if raw_b0.IsValid()
                    else None
                )
                raw_b1_t = (
                    [str(p) for p in raw_b1.GetTargets()]
                    if raw_b1.IsValid()
                    else None
                )
                je = joint_prim.GetAttribute("physics:jointEnabled")
                joint_enabled = je.Get() if je.IsValid() else "attr-missing"
                print(
                    "[DEBUG] FixedJoint body targets (exact path check):\n"
                    f"  joint={joint_path}\n"
                    f"  api_body0={body0_paths}\n"
                    f"  api_body1={body1_paths}\n"
                    f"  raw physics:body0={raw_b0_t}\n"
                    f"  raw physics:body1={raw_b1_t}\n"
                    f"  tracked_plug_path={plug_path}\n"
                    f"  body1==tracked_plug? {body1_paths == [plug_path]}\n"
                    f"  raw_body1==tracked_plug? {raw_b1_t == [plug_path]}\n"
                    f"  physics:jointEnabled={joint_enabled!r}",
                    flush=True,
                )
                self._dump_fixed_joint_break_and_compliance(joint_prim)
                self._subscribe_joint_break_events(joint_path)
            if (
                self.cable_mount.topology is not None
                and stage is not None
            ):
                ap = stage.GetPrimAtPath(
                    self.cable_mount.topology.existing_attachment_path
                )
                if ap.IsValid():
                    rels = {
                        str(attr.GetName()): [
                            str(t) for t in attr.GetTargets()
                        ]
                        for attr in ap.GetRelationships()
                    }
                    attrs = {
                        str(a.GetName()): a.Get()
                        for a in ap.GetAttributes()
                        if a.HasAuthoredValueOpinion()
                    }
                    attach_report = (
                        f"path={ap.GetPath()} schemas={list(ap.GetAppliedSchemas())} "
                        f"rels={rels} authored_attrs_keys={sorted(attrs)}"
                    )
            print(
                "[DEBUG] tracked plug rigid/kinematic state BEFORE play():\n"
                f"  {kin_report}\n"
                f"  attachment: {attach_report}",
                flush=True,
            )
            print(
                "[DEBUG] cfg.scene.franka_position (MERGED env → config)="
                f"{self.cfg.scene.franka_position} "
                f"env MERGED_FRANKA_POSITION_Z="
                f"{os.environ.get('MERGED_FRANKA_POSITION_Z', '<unset>')!r}",
                flush=True,
            )

        print("[DEBUG] about to call app_utils.play()", flush=True)
        app_utils.play()
        print("[DEBUG] app_utils.play() returned", flush=True)
        if (
            self.cable_mount is not None
            and self.cable_mount.topology is not None
        ):
            print(
                "[DEBUG] cable topology after play():\n"
                f"  tracked_plug={self.cfg.cable_mount.tracked_plug_path}\n"
                f"  fixed_joint={self.cfg.cable_mount.fixed_joint_path}\n"
                f"  deformable_body="
                f"{self.cable_mount.topology.deformable_body_path}\n"
                f"  attachment="
                f"{self.cable_mount.topology.existing_attachment_path}\n"
                f"  attachment_targets=("
                f"{self.cable_mount.topology.attachment_target0}, "
                f"{self.cable_mount.topology.attachment_target1})",
                flush=True,
            )
            # Re-read kinematic AFTER play — PhysX may change effective state.
            stage = omni.usd.get_context().get_stage()
            plug = stage.GetPrimAtPath(self.cfg.cable_mount.tracked_plug_path)
            if plug.IsValid() and plug.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = UsdPhysics.RigidBodyAPI(plug)
                kattr = rb.GetKinematicEnabledAttr()
                print(
                    "[DEBUG] tracked plug kinematic AFTER play(): "
                    f"kinematicEnabled={kattr.Get() if kattr.IsValid() else None!r}",
                    flush=True,
                )
            # Physics scene solver iterations (next place if joint doesn't break).
            try:
                if self.physics_scene is not None:
                    print(
                        "[DEBUG] physics scene solver settings AFTER play():\n"
                        f"  solver_type={self.physics_scene.get_solver_type()!r}\n"
                        f"  gpu_dynamics={self.physics_scene.get_enabled_gpu_dynamics()!r}\n"
                        f"  broadphase={self.physics_scene.get_broadphase_type()!r}",
                        flush=True,
                    )
                stage = omni.usd.get_context().get_stage()
                # Common PhysX scene iteration attrs
                for scene_path in (
                    "/physicsScene",
                    "/World/physicsScene",
                    "/PhysicsScene",
                ):
                    sp = stage.GetPrimAtPath(scene_path) if stage else None
                    if sp is None or not sp.IsValid():
                        continue
                    iter_attrs = {}
                    for name in (
                        "physxScene:maxPositionIterationCount",
                        "physxScene:maxVelocityIterationCount",
                        "physxScene:minPositionIterationCount",
                        "physxScene:minVelocityIterationCount",
                        "physxScene:solverType",
                    ):
                        attr = sp.GetAttribute(name)
                        if attr.IsValid():
                            iter_attrs[name] = {
                                "value": attr.Get(),
                                "authored": attr.HasAuthoredValueOpinion(),
                            }
                    print(
                        f"[DEBUG] USD physics scene {scene_path} "
                        f"iteration attrs={iter_attrs or 'none found'}",
                        flush=True,
                    )
            except Exception as error:
                print(
                    f"[DEBUG] solver settings dump failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            # Live Franka root Z vs cfg / MERGED env.
            try:
                franka_root = stage.GetPrimAtPath(self.cfg.scene.franka_path)
                if franka_root.IsValid():
                    from cable.cable_mount import _world_transform as _wt

                    T = _wt(stage, self.cfg.scene.franka_path)
                    print(
                        "[DEBUG] live Franka root world pose AFTER play():\n"
                        f"  path={self.cfg.scene.franka_path}\n"
                        f"  world_pos={np.round(T[:3, 3], 6).tolist()}\n"
                        f"  cfg.franka_position={self.cfg.scene.franka_position}\n"
                        f"  MERGED_FRANKA_POSITION_XYZ=("
                        f"{os.environ.get('MERGED_FRANKA_POSITION_X')}, "
                        f"{os.environ.get('MERGED_FRANKA_POSITION_Y')}, "
                        f"{os.environ.get('MERGED_FRANKA_POSITION_Z')})",
                        flush=True,
                    )
                asset = stage.GetPrimAtPath(self.cfg.scene.franka_asset_path)
                if asset.IsValid():
                    from cable.cable_mount import _world_transform as _wt2

                    T2 = _wt2(stage, self.cfg.scene.franka_asset_path)
                    print(
                        "[DEBUG] live Franka asset world pose AFTER play():\n"
                        f"  path={self.cfg.scene.franka_asset_path}\n"
                        f"  world_pos={np.round(T2[:3, 3], 6).tolist()}",
                        flush=True,
                    )
            except Exception as error:
                print(
                    f"[DEBUG] Franka root pose read failed: {type(error).__name__}: {error}",
                    flush=True,
                )
        print(
            "[DEBUG] about to call app_utils.update_app(steps=30) "
            "with plug_axis sampling every 5 steps",
            flush=True,
        )
        for step in range(30):
            app_utils.update_app(steps=1)
            if step % 5 == 0 or step == 29:
                sample = self._sample_tracked_plug_tip_and_axis_from_stage()
                if sample is not None:
                    tip_world, plug_axis = sample
                    print(
                        f"[DEBUG] post-play settle step={step}: "
                        f"tip={np.round(tip_world, 6).tolist()} "
                        f"plug_axis={np.round(plug_axis, 6).tolist()} "
                        f"plug_axis[2]={plug_axis[2]:.6f}",
                        flush=True,
                    )
        print("[DEBUG] app_utils.update_app(steps=30) returned", flush=True)

        self._maybe_recompute_ik_target()
        if self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
            print(
                "[DEBUG] cfg.ik.initial_position before _create_ik(): "
                f"{list(self.cfg.ik.initial_position)} "
                f"initial_orientation_wxyz="
                f"{list(self.cfg.ik.initial_orientation_wxyz)} "
                f"backend={SimulationManager.get_backend()!r}",
                flush=True,
            )

        print("[DEBUG] about to call _create_ik()", flush=True)
        self.ik = self._create_ik(assets_root)
        print("[DEBUG] _create_ik() returned", flush=True)
        print(
            "[DEBUG] sampling plug_axis for 10 steps after _create_ik()",
            flush=True,
        )
        for step in range(10):
            app_utils.update_app(steps=1)
            if self.ik is not None:
                try:
                    hand_position, hand_orientation = (
                        self.ik.articulation.get_world_pose()
                    )
                    hand_position = np.asarray(hand_position, dtype=np.float64)
                    hand_rotation = quaternion_wxyz_to_matrix(
                        np.asarray(hand_orientation, dtype=np.float64)
                    )
                    hand_forward = hand_rotation[:, 2]
                except Exception as error:
                    hand_forward = f"unavailable:{type(error).__name__}"
            else:
                hand_forward = "no-ik"
            sample = self._sample_tracked_plug_tip_and_axis_from_stage()
            if sample is not None:
                tip_world, plug_axis = sample
                print(
                    f"[DEBUG] post-_create_ik step={step}: "
                    f"tip={np.round(tip_world, 6).tolist()} "
                    f"plug_axis={np.round(plug_axis, 6).tolist()} "
                    f"plug_axis[2]={plug_axis[2]:.6f} "
                    f"hand_forward="
                    f"{np.round(hand_forward, 6).tolist() if not isinstance(hand_forward, str) else hand_forward}",
                    flush=True,
                )
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
