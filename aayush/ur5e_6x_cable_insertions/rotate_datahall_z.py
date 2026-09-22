#!/usr/bin/env python3
"""One-shot: rotate Robots +180° about +Z; CableBlocks + Cables +90° about +Z."""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

USD_PATH = Path.home() / "Desktop/Aayush_ws/DataHall_6r_ur5e.usd"


def _quat_z(deg: float) -> Gf.Quatd:
    half = math.radians(float(deg)) * 0.5
    return Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))


def _compose(q_delta: Gf.Quatd, q_old: Gf.Quatd) -> Gf.Quatd:
    # World-frame left-multiply: new = delta * old
    return Gf.Quatd(q_delta * q_old)


def _set_orient(prim: Usd.Prim, q: Gf.Quatd) -> None:
    xformable = UsdGeom.Xformable(prim)
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op
            break
    if orient_op is None:
        orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    # Match authored precision if possible
    try:
        orient_op.Set(q)
    except Exception:
        orient_op.Set(Gf.Quatf(q))


def _get_orient(prim: Usd.Prim) -> Gf.Quatd:
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            val = op.Get()
            if val is None:
                return Gf.Quatd(1.0)
            if isinstance(val, Gf.Quatd):
                return Gf.Quatd(val)
            return Gf.Quatd(val)
    return Gf.Quatd(1.0)


def _rotate_children(stage: Usd.Stage, root: str, deg: float) -> list[str]:
    parent = stage.GetPrimAtPath(root)
    if not parent or not parent.IsValid():
        raise RuntimeError(f"Missing {root}")
    q_delta = _quat_z(deg)
    changed: list[str] = []
    for child in parent.GetChildren():
        q_old = _get_orient(child)
        q_new = _compose(q_delta, q_old)
        _set_orient(child, q_new)
        changed.append(str(child.GetPath()))
        print(f"  {child.GetPath()}  +{deg:.0f}° Z  {q_old} -> {q_new}")
    return changed


def main() -> int:
    if not USD_PATH.is_file():
        raise SystemExit(f"USD not found: {USD_PATH}")
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise SystemExit(f"Failed to open {USD_PATH}")

    print(f"[USD ROTATE] {USD_PATH}")
    print("[USD ROTATE] Robots +180° about +Z")
    robots = _rotate_children(stage, "/World/Robots", 180.0)
    print("[USD ROTATE] CableBlocks +90° about +Z")
    blocks = _rotate_children(stage, "/World/CableBlocks", 90.0)
    print("[USD ROTATE] NetworkCables +90° about +Z")
    cables = _rotate_children(stage, "/World/NetworkCables", 90.0)

    stage.GetRootLayer().Save()
    print(
        f"[USD ROTATE] Saved. robots={len(robots)} blocks={len(blocks)} "
        f"cables={len(cables)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
