#!/usr/bin/env python3
"""One-time placement and topology checks for the pregrasped network cable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from cable_geometry import (
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

    def built_in_attachment_is_preserved(self) -> bool:
        """Return whether the asset-authored plug-to-tail relationship is unchanged."""

        if self.stage is None or self.topology is None:
            return False
        attachment = self.stage.GetPrimAtPath(
            self.topology.existing_attachment_path
        )
        if not attachment.IsValid() or not attachment.HasAPI(
            "PhysxAutoDeformableAttachmentAPI"
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
            target0 = _single_relationship_target(prim, _ATTACHABLE0_REL)
            target1 = _single_relationship_target(prim, _ATTACHABLE1_REL)
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


def _single_relationship_target(prim: Usd.Prim, name: str) -> str:
    relationship = prim.GetRelationship(name)
    if not relationship.IsValid():
        raise RuntimeError(f"Missing relationship {name} on {prim.GetPath()}")
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
        raise RuntimeError(f"Cannot compute bounds for invalid prim: {path}")
    aligned = _bbox_cache().ComputeUntransformedBound(prim).ComputeAlignedRange()
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
            f"Tracked plug has invalid local bounds: min={minimum}, max={maximum}"
        )
    return minimum, maximum


def _world_bounds_center(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot compute bounds for invalid prim: {path}")
    aligned = _bbox_cache().ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = np.asarray(aligned.GetMin(), dtype=np.float64)
    maximum = np.asarray(aligned.GetMax(), dtype=np.float64)
    center = 0.5 * (minimum + maximum)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise RuntimeError(f"Cable root has invalid world bounds: {path}")
    return center


def _world_transform(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Cannot read transform for invalid prim: {path}")
    gf_matrix = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(prim)
    # Gf uses row-vector transforms with translation in the last row. The
    # pure geometry module uses column vectors with translation in the last
    # column, so transpose at this boundary.
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
