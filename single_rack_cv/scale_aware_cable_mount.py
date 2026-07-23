#!/usr/bin/env python3
"""Scale-aware fixed-joint authoring for the cable mount."""

from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, UsdPhysics

from cable_geometry import rigid_pose_from_affine
from cable_mount import (
    CableMount,
    _matrix_to_gf_quatf,
    _world_transform,
)


class ScaleAwareCableMount(CableMount):
    """CableMount variant that removes authored scale from joint frames only."""

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
