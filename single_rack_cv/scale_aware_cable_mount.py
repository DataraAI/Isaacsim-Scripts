#!/usr/bin/env python3
"""Scale-aware fixed-joint authoring for the cable mount."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pxr import Gf, Sdf, UsdPhysics

import cable_mount as cable_mount_module
from affine_root_geometry import (
    compute_world_from_root_for_tip_preserving_affine,
)
from cable_geometry import (
    matrix_to_quaternion_wxyz,
    rigid_pose_from_affine,
    validate_affine_transform,
)
from cable_mount import (
    CableMount,
    _world_transform,
)


def _numpy_to_gf_matrix_affine(matrix: np.ndarray) -> Gf.Matrix4d:
    """Convert a column-vector affine transform to OpenUSD row-vector form."""

    affine = validate_affine_transform(matrix, "matrix")
    row_vector_matrix = affine.T
    return Gf.Matrix4d(
        *[float(value) for value in row_vector_matrix.reshape(-1)]
    )


def _matrix_to_gf_quatf_compatible(rotation: np.ndarray) -> Gf.Quatf:
    """Convert a proper matrix without relying on unsupported GfRotation overloads."""

    wxyz = matrix_to_quaternion_wxyz(rotation)
    return Gf.Quatf(
        float(wxyz[0]),
        float(wxyz[1]),
        float(wxyz[2]),
        float(wxyz[3]),
    )


class ScaleAwareCableMount(CableMount):
    """CableMount variant that removes scale only from rigid-only values."""

    def author_before_play(
        self,
        stage,
        hand_path: str,
        world_from_toolcenter: np.ndarray,
    ) -> None:
        """Use affine-safe placement helpers only during pre-play authoring."""

        original_compute = (
            cable_mount_module.compute_world_from_root_for_tip
        )
        original_converter = cable_mount_module._numpy_to_gf_matrix
        cable_mount_module.compute_world_from_root_for_tip = (
            compute_world_from_root_for_tip_preserving_affine
        )
        cable_mount_module._numpy_to_gf_matrix = (
            _numpy_to_gf_matrix_affine
        )
        try:
            super().author_before_play(
                stage=stage,
                hand_path=hand_path,
                world_from_toolcenter=world_from_toolcenter,
            )
        finally:
            cable_mount_module.compute_world_from_root_for_tip = (
                original_compute
            )
            cable_mount_module._numpy_to_gf_matrix = original_converter

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
            _matrix_to_gf_quatf_compatible(hand_from_plug[:3, :3])
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(1.0, 0.0, 0.0, 0.0)
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
