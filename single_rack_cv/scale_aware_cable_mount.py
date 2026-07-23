#!/usr/bin/env python3
"""Scale-aware fixed-joint authoring for the cable mount."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pxr import Gf, Sdf, UsdPhysics

from cable_geometry import rigid_pose_from_affine
from cable_mount import (
    CableMount,
    _matrix_to_gf_quatf,
    _world_transform,
)


class ScaleAwareCableMount(CableMount):
    """CableMount variant that removes authored scale from rigid-only values."""

    def _author_fixed_joint(self) -> None:
        if self.stage is None:
            raise RuntimeError("Cable mount stage is not initialized")

        world_from_hand_pose = rigid_pose_from_affine(
            _world_transform(self.stage, self.hand_path),
            "world_from_hand",
        )
        world_from_plug_pose = rigid_pose_from_affine(
            _world_transform(
                self.stage,
                self.mount_cfg.tracked_plug_path,
            ),
            "world_from_plug",
        )
        hand_from_plug = (
            np.linalg.inv(world_from_hand_pose)
            @ world_from_plug_pose
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

    def configure_fingers(self, articulation) -> None:
        """Convert asset-local plug bounds to physical meters before gap setup."""

        if self.stage is None or self.plug_frame is None:
            raise RuntimeError("Cable geometry must exist before finger setup")

        world_from_plug = _world_transform(
            self.stage,
            self.mount_cfg.tracked_plug_path,
        )
        axis_scale_m_per_local_unit = np.linalg.norm(
            world_from_plug[:3, :3],
            axis=0,
        )
        if (
            axis_scale_m_per_local_unit.shape != (3,)
            or not np.all(np.isfinite(axis_scale_m_per_local_unit))
            or np.any(axis_scale_m_per_local_unit <= 1.0e-12)
        ):
            raise RuntimeError(
                "Tracked plug has invalid physical axis scale: "
                f"{axis_scale_m_per_local_unit}"
            )

        local_dimensions = (
            np.asarray(self.plug_frame.local_max_m, dtype=np.float64)
            - np.asarray(self.plug_frame.local_min_m, dtype=np.float64)
        )
        physical_dimensions_m = (
            local_dimensions * axis_scale_m_per_local_unit
        )
        physical_longitudinal = int(np.argmax(physical_dimensions_m))
        if physical_longitudinal != self.plug_frame.longitudinal_axis_index:
            raise RuntimeError(
                "Authored non-uniform scale changes the detected plug axis: "
                f"local={self.plug_frame.longitudinal_axis_index}, "
                f"physical={physical_longitudinal}, "
                f"dimensions_m={physical_dimensions_m.tolist()}"
            )

        self.plug_frame = replace(
            self.plug_frame,
            dimensions_m=physical_dimensions_m,
        )
        if self.diagnostics is not None:
            self.diagnostics.plug_dimensions_m = tuple(
                float(value) for value in physical_dimensions_m
            )
        super().configure_fingers(articulation)
