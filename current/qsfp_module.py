"""Approximate QSFP-DD transceiver rigid body for insertion simulation."""

from __future__ import annotations

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from pxr import PhysxSchema, Usd, UsdPhysics

# Slim proxy body (meters). The proxy's local +Z is the insertion axis so it
# matches FrankaLulaController.tool_offset and panda_hand grasping.
QSFP_WIDTH_M = 0.003
QSFP_HEIGHT_M = 0.006
QSFP_LENGTH_M = 0.090
QSFP_MASS_KG = 0.005
# Shift the vertical-pick IK target toward the module top (+local Z) so the grip
# sits on the upper portion of the body and more length hangs below the fingers.
QSFP_GRASP_OFFSET_TO_TOP_M = 0.0275

# Visual proxy scale for DynamicCuboid (size * scale = edge length).
_PROXY_SIZE = 1.0


def _half_extent(edge: float) -> float:
    return edge / 2.0


def configure_qsfp_physics(prim: Usd.Prim) -> None:
    rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb_api.CreateEnableCCDAttr(True)
    rb_api.CreateSolverPositionIterationCountAttr(16)
    rb_api.CreateSolverVelocityIterationCountAttr(4)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(QSFP_MASS_KG)
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateContactOffsetAttr(0.001)
    collision_api.CreateRestOffsetAttr(0.0)


def create_qsfp_module(
    world,
    prim_path: str = "/World/QSFP_Module",
    position: np.ndarray | None = None,
    name: str = "qsfp_module",
):
    """Spawn a rigid QSFP-DD proxy as a scaled cuboid."""
    if position is None:
        position = np.array([0.3, 0.3, _half_extent(QSFP_LENGTH_M)], dtype=np.float64)

    scale = np.array([QSFP_WIDTH_M, QSFP_HEIGHT_M, QSFP_LENGTH_M], dtype=np.float64)
    module = world.scene.add(
        DynamicCuboid(
            name=name,
            position=position,
            prim_path=prim_path,
            scale=scale,
            size=_PROXY_SIZE,
            color=np.array([0.1, 0.35, 0.85]),
        )
    )
    configure_qsfp_physics(module.prim)
    return module


def grasp_tool_offset() -> float:
    """Controller distance from panda_hand to the grasped module target (m)."""
    return 0.1


def pick_grasp_block_z(module_center_z: float) -> float:
    """Block-center Z for vertical pick, shifted up toward the module top (+local Z)."""
    half_len = QSFP_LENGTH_M * 0.5
    offset = min(QSFP_GRASP_OFFSET_TO_TOP_M, half_len * 0.8)
    return float(module_center_z + offset)


def gripper_closed_positions() -> np.ndarray:
    """Close past the tiny module width so physics contact grips it."""
    return np.array([0.001, 0.001], dtype=np.float64)
