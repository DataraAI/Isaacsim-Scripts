"""Minimal USD geometry helpers needed by the UR5e scene builder."""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom

from ur5e_6x_cable_insertions import config as cfg


def meters_per_unit(stage) -> float:
    """Return the USD stage scale in meters per authored unit."""

    try:
        return float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        return 1.0


def prim_bbox(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a prim's world AABB in meters."""

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    scale = meters_per_unit(stage)
    minimum = np.asarray(box.GetMin(), dtype=np.float64) * scale
    maximum = np.asarray(box.GetMax(), dtype=np.float64) * scale
    return minimum, maximum, 0.5 * (minimum + maximum)


def _find_named_descendant(stage, root_path: str, name: str) -> str | None:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    wanted = name.lower()
    for prim in Usd.PrimRange(root):
        if prim.GetName().lower() == wanted:
            return str(prim.GetPath())
    return None


def compute_port_targets(
    stage, contacts_path: str | None = None
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Resolve temporary pin-based insert and +X approach targets."""

    if not contacts_path or not stage.GetPrimAtPath(contacts_path).IsValid():
        raise RuntimeError(f"Port contacts missing: {contacts_path}")
    path_a = _find_named_descendant(stage, contacts_path, cfg.PORT_PIN_A_NAME)
    path_b = _find_named_descendant(stage, contacts_path, cfg.PORT_PIN_B_NAME)
    if not path_a or not path_b:
        raise RuntimeError(f"Missing target pins under {contacts_path}")
    _amin, _amax, center_a = prim_bbox(stage, path_a)
    _bmin, _bmax, center_b = prim_bbox(stage, path_b)
    insert = 0.5 * (center_a + center_b)
    approach = insert.copy()
    approach[0] += float(cfg.PORT_APPROACH_X_OFFSET_M)
    return insert, approach, path_a, path_b
