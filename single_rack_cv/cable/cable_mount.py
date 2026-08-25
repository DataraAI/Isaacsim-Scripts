#!/usr/bin/env python3
"""One-time placement and topology checks for the pregrasped network cable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from cable.cable_geometry import (
    CableMountValidation,
    PlugFrame,
    angular_error_deg,
    compute_world_from_root_for_tip,
    detect_plug_frame,
    validate_transform,
)
from config import Config


_ATTACHABLE0_REL = "physxAutoDeformableAttachment:attachable0"
_ATTACHABLE1_REL = "physxAutoDeformableAttachment:attachable1"
_DEFORMABLE_API = "OmniPhysicsDeformableBodyAPI"


@dataclass(frozen=True)
class CableTopology:
    deformable_body_path: str
    existing_attachment_path: str
    attachment_target0: str
    attachment_target1: str
    attachment_relationships: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass
class CableMountDiagnostics:
    plug_dimensions_m: tuple[float, float, float]
    tip_local_m: tuple[float, float, float]
    deformable_body_path: str
    existing_attachment_path: str
    finger_total_gap_m: float = 0.0


class CableMount:
    """Author and verify a permanent mount for the asset's existing rigid plug."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.mount_cfg = cfg.cable_mount
        self.stage: Usd.Stage | None = None
        self.hand_path = ""
        self.topology: CableTopology | None = None
        self.plug_frame: PlugFrame | None = None
        self.diagnostics: CableMountDiagnostics | None = None

    def author_before_play(
        self,
        stage: Usd.Stage,
        hand_path: str,
        world_from_toolcenter: np.ndarray,
    ) -> None:
        """Load, validate, and place the connected cable before physics starts."""

        if not self.mount_cfg.enabled:
            return
        if not Path(self.mount_cfg.usd_path).is_file():
            raise FileNotFoundError(
                f"Cable USD not found: {self.mount_cfg.usd_path}"
            )
        if not stage:
            raise RuntimeError("A valid USD stage is required for cable mounting")

        self.stage = stage
        self.hand_path = hand_path
        hand = stage.GetPrimAtPath(hand_path)
        if not hand.IsValid() or not hand.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Franka hand is not a rigid body: {hand_path}")

        root = stage.DefinePrim(self.mount_cfg.root_path, "Xform")
        if not root.GetReferences().AddReference(self.mount_cfg.usd_path):
            raise RuntimeError(
                f"Could not add cable reference: {self.mount_cfg.usd_path}"
            )
        stage.Load(Sdf.Path(self.mount_cfg.root_path))

        self.topology = self._discover_topology(stage)
        local_min, local_max = _local_bounds(
            stage,
            self.mount_cfg.tracked_plug_path,
        )
        world_from_root = _world_transform(stage, self.mount_cfg.root_path)
        world_from_plug = _world_transform(
            stage,
            self.mount_cfg.tracked_plug_path,
        )
        cable_center_world = _world_bounds_center(
            stage,
            self.mount_cfg.root_path,
        )
        self.plug_frame = detect_plug_frame(
            local_min,
            local_max,
            world_from_plug,
            cable_center_world,
            axis_ratio_min=self.mount_cfg.axis_ratio_min,
            cable_projection_min_m=self.mount_cfg.cable_projection_min_m,
        )
        desired_world_from_tip = validate_transform(
            world_from_toolcenter,
            "world_from_toolcenter",
        )
        mounted_world_from_root = compute_world_from_root_for_tip(
            world_from_root,
            world_from_plug,
            self.plug_frame,
            desired_world_from_tip,
        )
        _set_root_transform(
            stage,
            self.mount_cfg.root_path,
            mounted_world_from_root,
        )

        actual_world_from_plug = _world_transform(
            stage,
            self.mount_cfg.tracked_plug_path,
        )
        actual_world_from_tip = (
            actual_world_from_plug @ self.plug_frame.plug_from_tip
        )
        tip_error_m = float(
            np.linalg.norm(
                actual_world_from_tip[:3, 3]
                - desired_world_from_tip[:3, 3]
            )
        )
        axis_error_deg = angular_error_deg(
            actual_world_from_tip[:3, 2],
            desired_world_from_tip[:3, 2],
        )
        if tip_error_m >= 1.0e-6:
            raise RuntimeError(
                "Pre-play RJ45 tip placement error is too large: "
                f"{tip_error_m * 1000.0:.6f} mm"
            )
        if axis_error_deg >= 1.0e-6:
            raise RuntimeError(
                "Pre-play RJ45 axis placement error is too large: "
                f"{axis_error_deg:.9f} deg"
            )
        if not self.built_in_attachment_is_preserved():
            raise RuntimeError(
                "The asset-authored plug-to-tail attachment changed during placement"
            )

        self._author_fixed_joint()
        self._filter_hand_and_finger_collisions()
        if not self.fixed_joint_is_valid():
            raise RuntimeError("Direct hand-to-plug fixed joint is invalid")
        if not self.built_in_attachment_is_preserved():
            raise RuntimeError(
                "The asset-authored plug-to-tail attachment changed during joint authoring"
            )

        self.diagnostics = CableMountDiagnostics(
            plug_dimensions_m=tuple(
                float(value) for value in self.plug_frame.dimensions_m
            ),
            tip_local_m=tuple(
                float(value) for value in self.plug_frame.tip_local_m
            ),
            deformable_body_path=self.topology.deformable_body_path,
            existing_attachment_path=self.topology.existing_attachment_path,
        )

    def author_from_existing_grasp(
        self,
        stage: Usd.Stage,
        hand_path: str,
    ) -> None:
        """Compute cable topology and plug_frame from a cable that's already
        loaded and physically grasped (real FixedJoint from esha's pickup
        pipeline), instead of teleporting a synthetic pregrasp mount.

        Mirrors author_before_play()'s geometry-analysis steps exactly, but
        never loads a cable reference and never repositions anything — the
        cable's current pose is the ground truth to analyze, not something
        to override.
        """
        if not self.mount_cfg.enabled:
            return
        if not stage:
            raise RuntimeError("A valid USD stage is required for cable mounting")

        self.stage = stage
        self.hand_path = hand_path
        hand = stage.GetPrimAtPath(hand_path)
        if not hand.IsValid() or not hand.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Franka hand is not a rigid body: {hand_path}")

        cable_root = stage.GetPrimAtPath(self.mount_cfg.root_path)
        if not cable_root.IsValid():
            raise RuntimeError(
                f"Expected cable already loaded at {self.mount_cfg.root_path}, "
                "but no valid prim was found. author_from_existing_grasp() "
                "requires the pickup pipeline to have already loaded and "
                "grasped the cable."
            )

        self.topology = self._discover_topology(stage)
        local_min, local_max = _local_bounds(
            stage,
            self.mount_cfg.tracked_plug_path,
        )
        world_from_plug = _world_transform(
            stage,
            self.mount_cfg.tracked_plug_path,
        )
        cable_center_world = _world_bounds_center(
            stage,
            self.mount_cfg.root_path,
        )
        self.plug_frame = detect_plug_frame(
            local_min,
            local_max,
            world_from_plug,
            cable_center_world,
            axis_ratio_min=self.mount_cfg.axis_ratio_min,
            cable_projection_min_m=self.mount_cfg.cable_projection_min_m,
        )

        self.diagnostics = CableMountDiagnostics(
            plug_dimensions_m=tuple(
                float(value) for value in self.plug_frame.dimensions_m
            ),
            tip_local_m=tuple(
                float(value) for value in self.plug_frame.tip_local_m
            ),
            deformable_body_path=self.topology.deformable_body_path,
            existing_attachment_path=self.topology.existing_attachment_path,
        )

    def configure_fingers(self, articulation) -> None:
        """Set a symmetric cosmetic gap around the rigidly mounted plug."""

        if self.plug_frame is None or self.diagnostics is None:
            raise RuntimeError(
                "Cable geometry must be authored before finger setup"
            )
        indices = np.asarray(
            [
                articulation.get_dof_index(name)
                for name in self.mount_cfg.finger_joint_names
            ],
            dtype=np.int32,
        )
        if indices.shape != (2,) or np.any(indices < 0):
            raise RuntimeError(
                "Could not resolve both Franka finger joints: "
                f"{self.mount_cfg.finger_joint_names}"
            )

        properties = articulation.dof_properties
        try:
            lower = np.asarray(properties["lower"], dtype=np.float64)[indices]
            upper = np.asarray(properties["upper"], dtype=np.float64)[indices]
        except (IndexError, KeyError, TypeError, ValueError):
            array = np.asarray(properties)
            if array.ndim != 2 or array.shape[1] < 4:
                raise RuntimeError(
                    "Unsupported articulation DOF property layout"
                )
            lower = np.asarray(array[indices, 2], dtype=np.float64)
            upper = np.asarray(array[indices, 3], dtype=np.float64)

        if (
            lower.shape != (2,)
            or upper.shape != (2,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(upper < lower)
        ):
            raise RuntimeError(
                f"Invalid Franka finger limits: lower={lower}, upper={upper}"
            )

        plug_width_m = float(
            self.plug_frame.dimensions_m[
                self.plug_frame.wide_transverse_axis_index
            ]
        )
        total_gap_m = plug_width_m + self.mount_cfg.finger_total_clearance_m
        desired = np.full(2, 0.5 * total_gap_m, dtype=np.float64)
        commanded = np.clip(desired, lower, upper)
        articulation.set_joint_positions(
            commanded,
            joint_indices=indices,
        )
        articulation.set_joint_position_targets(
            commanded,
            joint_indices=indices,
        )
        self.diagnostics.finger_total_gap_m = float(np.sum(commanded))

    def sample_validation(self, runtime) -> tuple[float, float]:
        """Measure one strict cable-mount validation frame."""

        if (
            self.stage is None
            or self.topology is None
            or self.plug_frame is None
        ):
            raise RuntimeError("Cable mount is not initialized")
        plug = self.stage.GetPrimAtPath(self.mount_cfg.tracked_plug_path)
        deformable = self.stage.GetPrimAtPath(
            self.topology.deformable_body_path
        )
        if not plug.IsValid() or not plug.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("Tracked RJ45 plug lost rigid-body validity")
        if not deformable.IsValid() or not deformable.HasAPI(_DEFORMABLE_API):
            raise RuntimeError(
                "Cable tail lost Omni Physics deformable validity"
            )
        if not self.fixed_joint_is_valid():
            raise RuntimeError("Direct hand-to-plug fixed joint is invalid")
        if not self.built_in_attachment_is_preserved():
            raise RuntimeError("Built-in plug-to-tail attachment changed")

        physics_scene = getattr(runtime, "physics_scene", None)
        if (
            physics_scene is None
            or not physics_scene.get_enabled_gpu_dynamics()
        ):
            raise RuntimeError("Cable mount requires active GPU dynamics")

        world_from_plug = _world_transform(
            self.stage,
            self.mount_cfg.tracked_plug_path,
        )
        tip_world = (
            world_from_plug @ np.r_[self.plug_frame.tip_local_m, 1.0]
        )[:3]
        nose_world = (
            world_from_plug[:3, :3] @ self.plug_frame.nose_axis_local
        )

        hand_position, hand_orientation = runtime._get_world_pose(
            self.hand_path
        )
        world_from_hand = _quaternion_wxyz_to_matrix(hand_orientation)
        hand_from_tool = _quaternion_wxyz_to_matrix(
            np.asarray(
                self.cfg.ik.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            )
        )
        tool_position = (
            np.asarray(hand_position, dtype=np.float64)
            + world_from_hand
            @ np.asarray(
                self.cfg.ik.tool_center_local_position_m,
                dtype=np.float64,
            )
        )
        tool_axis = (world_from_hand @ hand_from_tool)[:, 2]
        tip_error_m = float(np.linalg.norm(tip_world - tool_position))
        axis_error_deg = angular_error_deg(nose_world, tool_axis)
        return tip_error_m, axis_error_deg

    def log_success(self, validation: CableMountValidation) -> None:
        """Print the complete validated mount state."""

        if self.diagnostics is None or self.topology is None:
            raise RuntimeError("Cable mount diagnostics are unavailable")
        dimensions_mm = [
            round(value * 1000.0, 3)
            for value in self.diagnostics.plug_dimensions_m
        ]
        tip_local = [
            round(value, 6) for value in self.diagnostics.tip_local_m
        ]
        print(
            "[CABLE MOUNT]\n"
            f"  cable USD: {self.mount_cfg.usd_path}\n"
            f"  tracked plug: {self.mount_cfg.tracked_plug_path}\n"
            f"  deformable body: {self.topology.deformable_body_path}\n"
            f"  preserved attachment: "
            f"{self.topology.existing_attachment_path}\n"
            f"  plug dimensions mm: {dimensions_mm}\n"
            f"  insertion-tip local position m: {tip_local}\n"
            f"  finger total gap mm: "
            f"{self.diagnostics.finger_total_gap_m * 1000.0:.3f}\n"
            f"  validation frames: {validation.frame_count}/"
            f"{self.mount_cfg.validation_frames}\n"
            f"  maximum tip error mm: "
            f"{validation.maximum_tip_error_m * 1000.0:.6f}\n"
            f"  maximum axis error deg: "
            f"{validation.maximum_axis_error_deg:.6f}\n"
            "  fixed joint: valid\n"
            "  built-in attachment: preserved\n"
            "  cable tail: deformable\n"
            "  GPU dynamics: enabled",
            flush=True,
        )

    def fixed_joint_is_valid(self) -> bool:
        """Return whether the direct panda_hand-to-plug joint is intact."""

        if self.stage is None or not self.hand_path:
            return False
        joint_prim = self.stage.GetPrimAtPath(
            self.mount_cfg.fixed_joint_path
        )
        if (
            not joint_prim.IsValid()
            or not joint_prim.IsA(UsdPhysics.FixedJoint)
        ):
            return False
        joint = UsdPhysics.FixedJoint(joint_prim)
        body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        return body0 == [self.hand_path] and body1 == [
            self.mount_cfg.tracked_plug_path
        ]

    def _author_fixed_joint(self) -> None:
        if self.stage is None:
            raise RuntimeError("Cable mount stage is not initialized")
        world_from_hand = _world_transform(self.stage, self.hand_path)
        world_from_plug = _world_transform(
            self.stage,
            self.mount_cfg.tracked_plug_path,
        )
        hand_from_plug = np.linalg.inv(world_from_hand) @ world_from_plug
        hand_from_plug = validate_transform(
            hand_from_plug,
            "hand_from_plug",
        )

        joint = UsdPhysics.FixedJoint.Define(
            self.stage,
            Sdf.Path(self.mount_cfg.fixed_joint_path),
        )
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.hand_path)])
        joint.CreateBody1Rel().SetTargets(
            [Sdf.Path(self.mount_cfg.tracked_plug_path)]
        )
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(
                *[float(value) for value in hand_from_plug[:3, 3]]
            )
        )
        joint.CreateLocalRot0Attr().Set(
            _matrix_to_gf_quatf(hand_from_plug[:3, :3])
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        )

    def _filter_hand_and_finger_collisions(self) -> None:
        if self.stage is None:
            raise RuntimeError("Cable mount stage is not initialized")
        root = self.stage.GetPrimAtPath(
            self.cfg.scene.franka_asset_path
        )
        if not root.IsValid():
            raise RuntimeError(
                "Franka asset root is invalid: "
                f"{self.cfg.scene.franka_asset_path}"
            )

        names = (
            self.mount_cfg.hand_link_name,
            *self.mount_cfg.finger_link_names,
        )
        filtered_paths: list[str] = []
        for name in names:
            path = _find_unique_descendant(root, name)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(
                    f"Collision-filter target is not a rigid body: {path}"
                )
            filtered_paths.append(path)

        plug = self.stage.GetPrimAtPath(
            self.mount_cfg.tracked_plug_path
        )
        api = UsdPhysics.FilteredPairsAPI.Apply(plug)
        relationship = api.CreateFilteredPairsRel()
        existing = {str(path) for path in relationship.GetTargets()}
        combined = sorted(existing.union(filtered_paths))
        relationship.SetTargets([Sdf.Path(path) for path in combined])

    def built_in_attachment_is_preserved(self) -> bool:
        """Return whether the asset-authored plug-to-tail link is unchanged."""

        if self.stage is None or self.topology is None:
            return False
        attachment = self.stage.GetPrimAtPath(
            self.topology.existing_attachment_path
        )
        if (
            not attachment.IsValid()
            or not attachment.HasAPI(
                "PhysxAutoDeformableAttachmentAPI"
            )
        ):
            return False
        return (
            _relationship_snapshot(attachment)
            == self.topology.attachment_relationships
        )

    def _discover_topology(self, stage: Usd.Stage) -> CableTopology:
        root = stage.GetPrimAtPath(self.mount_cfg.root_path)
        plug = stage.GetPrimAtPath(self.mount_cfg.tracked_plug_path)
        if not root.IsValid():
            raise RuntimeError(
                f"Cable root is invalid: {self.mount_cfg.root_path}"
            )
        if not plug.IsValid():
            raise RuntimeError(
                f"Tracked plug is invalid: {self.mount_cfg.tracked_plug_path}"
            )
        if not plug.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                "Tracked plug must retain PhysicsRigidBodyAPI: "
                f"{self.mount_cfg.tracked_plug_path}"
            )

        deformable_paths = sorted(
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(_DEFORMABLE_API)
        )
        if len(deformable_paths) != 1:
            raise RuntimeError(
                "Expected exactly one Omni Physics deformable body below "
                f"{self.mount_cfg.root_path}, found {deformable_paths}"
            )
        deformable_path = deformable_paths[0]

        matching: list[tuple[Usd.Prim, str, str]] = []
        expected_targets = {
            deformable_path,
            self.mount_cfg.tracked_plug_path,
        }
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI("PhysxAutoDeformableAttachmentAPI"):
                continue
            target0 = _single_relationship_target(
                prim,
                _ATTACHABLE0_REL,
            )
            target1 = _single_relationship_target(
                prim,
                _ATTACHABLE1_REL,
            )
            if {target0, target1} == expected_targets:
                matching.append((prim, target0, target1))

        if len(matching) != 1:
            paths = [str(item[0].GetPath()) for item in matching]
            raise RuntimeError(
                "Expected exactly one existing attachment connecting the "
                f"deformable tail to the tracked plug, found {paths}"
            )
        attachment, target0, target1 = matching[0]
        return CableTopology(
            deformable_body_path=deformable_path,
            existing_attachment_path=str(attachment.GetPath()),
            attachment_target0=target0,
            attachment_target1=target1,
            attachment_relationships=_relationship_snapshot(attachment),
        )


def _quaternion_wxyz_to_matrix(quaternion) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("Quaternion must be finite with shape (4,)")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion cannot have zero length")
    w, x, y, z = value / norm
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


def _find_unique_descendant(root: Usd.Prim, name: str) -> str:
    matches = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.GetName() == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one '{name}' below {root.GetPath()}, found {matches}"
        )
    return matches[0]


def _matrix_to_gf_quatf(rotation: np.ndarray) -> Gf.Quatf:
    rotation = np.asarray(rotation, dtype=np.float64)
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, :3] = rotation
    validate_transform(candidate, "rotation")
    row_rotation = rotation.T
    gf_rotation = Gf.Matrix3d(
        *[float(value) for value in row_rotation.reshape(-1)]
    )
    quat = Gf.Rotation(gf_rotation).GetQuat()
    imaginary = quat.GetImaginary()
    return Gf.Quatf(
        float(quat.GetReal()),
        Gf.Vec3f(
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _single_relationship_target(prim: Usd.Prim, name: str) -> str:
    relationship = prim.GetRelationship(name)
    if not relationship.IsValid():
        raise RuntimeError(
            f"Missing relationship {name} on {prim.GetPath()}"
        )
    targets = [str(path) for path in relationship.GetTargets()]
    if len(targets) != 1:
        raise RuntimeError(
            f"Relationship {name} on {prim.GetPath()} must have one target, "
            f"found {targets}"
        )
    return targets[0]


def _relationship_snapshot(
    prim: Usd.Prim,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        sorted(
            (
                relationship.GetName(),
                tuple(str(path) for path in relationship.GetTargets()),
            )
            for relationship in prim.GetRelationships()
        )
    )


def _bbox_cache() -> UsdGeom.BBoxCache:
    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )


def _local_bounds(
    stage: Usd.Stage,
    path: str,
) -> tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(
            f"Cannot compute bounds for invalid prim: {path}"
        )
    aligned = (
        _bbox_cache()
        .ComputeUntransformedBound(prim)
        .ComputeAlignedRange()
    )
    minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
    maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(maximum <= minimum)
    ):
        raise RuntimeError(
            "Tracked plug has invalid local bounds: "
            f"min={minimum}, max={maximum}"
        )
    return minimum, maximum


def _world_bounds_center(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(
            f"Cannot compute bounds for invalid prim: {path}"
        )
    aligned = _bbox_cache().ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
    maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
    center = 0.5 * (minimum + maximum)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise RuntimeError(
            f"Cable root has invalid world bounds: {path}"
        )
    return center


def _world_transform(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(
            f"Cannot read transform for invalid prim: {path}"
        )
    gf_matrix = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(prim)
    return np.asarray(gf_matrix, dtype=np.float64).T


def _numpy_to_gf_matrix(matrix: np.ndarray) -> Gf.Matrix4d:
    matrix = validate_transform(matrix, "matrix")
    row_vector_matrix = matrix.T
    return Gf.Matrix4d(
        *[float(value) for value in row_vector_matrix.reshape(-1)]
    )


def _set_root_transform(
    stage: Usd.Stage,
    path: str,
    world_from_root: np.ndarray,
) -> None:
    if path != "/World/NetworkCable":
        raise RuntimeError(
            "Only /World/NetworkCable may receive placement transform"
        )
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Cable root is invalid: {path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(
        _numpy_to_gf_matrix(world_from_root)
    )
