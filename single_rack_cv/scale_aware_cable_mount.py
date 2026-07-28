#!/usr/bin/env python3
"""Scale-aware fixed-joint authoring for the cable mount."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

import cable_mount as cable_mount_module
from affine_root_geometry import (
    compute_world_from_root_for_tip_preserving_affine,
)
from articulation_host_bridge import HostSafeDofPropertiesArticulation
from cable_geometry import (
    matrix_to_quaternion_wxyz,
    rigid_pose_from_affine,
    validate_affine_transform,
)
from cable_mount import (
    CableMount,
    _world_transform,
)
from tail_preshape import preshape_free_hanging_tail


_TAIL_PRESHAPE_ANCHOR_LENGTH_M = 0.015
_TAIL_PRESHAPE_BEND_LENGTH_M = 0.100
_TAIL_PRESHAPE_FAR_ANCHOR_LENGTH_M = 0.015
_TAIL_PRESHAPE_DROP_M = 0.100


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack(
        [values, np.ones(values.shape[0], dtype=np.float64)]
    )
    return (matrix @ homogeneous.T).T[:, :3]


def _vec3f_array(points: np.ndarray) -> Vt.Vec3fArray:
    values = np.asarray(points, dtype=np.float64)
    return Vt.Vec3fArray(
        [
            Gf.Vec3f(float(row[0]), float(row[1]), float(row[2]))
            for row in values
        ]
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
            self._preshape_deformable_tail()
        finally:
            cable_mount_module.compute_world_from_root_for_tip = (
                original_compute
            )
            cable_mount_module._numpy_to_gf_matrix = original_converter

    def _preshape_deformable_tail(self) -> None:
        """Curve the cable away from the palm while keeping both ends fixed."""

        if self.stage is None or self.topology is None:
            raise RuntimeError("Cable mount topology is not initialized")
        deformable = self.stage.GetPrimAtPath(
            self.topology.deformable_body_path
        )
        if not deformable.IsValid():
            raise RuntimeError("Cable deformable body is invalid")

        world_from_plug = _world_transform(
            self.stage,
            self.mount_cfg.tracked_plug_path,
        )
        plug_world = world_from_plug[:3, 3]
        attribute_names = (
            "points",
            "omniphysics:restShapePoints",
            "physxDeformable:simulationRestPoints",
            "physxDeformable:collisionRestPoints",
        )
        changed_arrays = 0
        maximum_drop_m = 0.0
        inspected: list[str] = []

        for prim in Usd.PrimRange(deformable):
            world_from_prim = _world_transform(
                self.stage,
                str(prim.GetPath()),
            )
            prim_from_world = np.linalg.inv(world_from_prim)
            inspected.append(
                f"{prim.GetPath()} schemas={list(prim.GetAppliedSchemas())}"
            )

            for attribute_name in attribute_names:
                attribute = prim.GetAttribute(attribute_name)
                if not attribute.IsValid():
                    continue
                value = attribute.Get()
                if value is None:
                    continue
                local_points = np.asarray(value, dtype=np.float64)
                if (
                    local_points.ndim != 2
                    or local_points.shape[1] != 3
                    or local_points.shape[0] < 4
                    or not np.all(np.isfinite(local_points))
                ):
                    continue

                world_points = _transform_points(
                    world_from_prim,
                    local_points,
                )
                try:
                    bent_world = preshape_free_hanging_tail(
                        world_points,
                        plug_world_m=plug_world,
                        down_world_axis=np.array(
                            [0.0, 0.0, -1.0],
                            dtype=np.float64,
                        ),
                        anchor_length_m=(
                            _TAIL_PRESHAPE_ANCHOR_LENGTH_M
                        ),
                        bend_length_m=_TAIL_PRESHAPE_BEND_LENGTH_M,
                        far_anchor_length_m=(
                            _TAIL_PRESHAPE_FAR_ANCHOR_LENGTH_M
                        ),
                        drop_m=_TAIL_PRESHAPE_DROP_M,
                    )
                except ValueError as error:
                    if "too short" in str(error):
                        continue
                    raise RuntimeError(
                        f"Could not pre-shape {prim.GetPath()} "
                        f"attribute {attribute_name}: {error}"
                    ) from error

                bent_local = _transform_points(
                    prim_from_world,
                    bent_world,
                )
                if not attribute.Set(_vec3f_array(bent_local)):
                    raise RuntimeError(
                        f"Could not author {attribute_name} on "
                        f"{prim.GetPath()}"
                    )
                changed_arrays += 1
                maximum_drop_m = max(
                    maximum_drop_m,
                    float(
                        np.max(
                            np.linalg.norm(
                                bent_world - world_points,
                                axis=1,
                            )
                        )
                    ),
                )

                if attribute_name == "points":
                    if not prim.IsA(UsdGeom.PointBased):
                        raise RuntimeError(
                            f"Points attribute is not PointBased: {prim.GetPath()}"
                        )
                    point_based = UsdGeom.PointBased(prim)
                    minimum = np.min(bent_local, axis=0)
                    maximum = np.max(bent_local, axis=0)
                    point_based.GetExtentAttr().Set(
                        _vec3f_array(np.vstack([minimum, maximum]))
                    )
                    normals = point_based.GetNormalsAttr()
                    if (
                        normals.IsValid()
                        and normals.HasAuthoredValueOpinion()
                    ):
                        normals.Clear()
                    primvar_normals = prim.GetAttribute("primvars:normals")
                    if (
                        primvar_normals.IsValid()
                        and primvar_normals.HasAuthoredValueOpinion()
                    ):
                        primvar_normals.Clear()

        if changed_arrays == 0:
            raise RuntimeError(
                "No deformable tail point arrays were eligible for pre-shape. "
                + " | ".join(inspected)
            )
        if not self.built_in_attachment_is_preserved():
            raise RuntimeError(
                "The built-in plug-to-tail attachment changed during pre-shape"
            )

        print(
            "[CABLE MOUNT] TAIL PRE-SHAPED\n"
            f"  point arrays changed: {changed_arrays}\n"
            f"  maximum downward displacement mm: "
            f"{maximum_drop_m * 1000.0:.3f}\n"
            "  tracked-plug end: fixed\n"
            "  far-cable end: fixed\n"
            "  additional attachment: none",
            flush=True,
        )

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
        super().configure_fingers(
            HostSafeDofPropertiesArticulation(articulation)
        )
