#!/usr/bin/env python3
"""Inspect the configured cable for Isaac Sim 6 deformable and connection topology."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaacsim import SimulationApp

from config import CONFIG


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _has_omniphysics_deformable(prim) -> bool:
    return bool(prim.HasAPI("OmniPhysicsDeformableBodyAPI"))


def _relationship_targets(prim, relationship_name: str) -> list[str]:
    relationship = prim.GetRelationship(relationship_name)
    if not relationship.IsValid():
        return []
    return [str(path) for path in relationship.GetTargets()]


def _path_is_below(path: str, root_path: str, Sdf) -> bool:
    return Sdf.Path(path).HasPrefix(Sdf.Path(root_path))


def _candidate_paths(stage, root_path: str, plug_path: str) -> list[str]:
    plug = stage.GetPrimAtPath(plug_path)
    root = stage.GetPrimAtPath(root_path)
    if not plug.IsValid() or not root.IsValid():
        return []

    current = plug
    while current.IsValid():
        if _has_omniphysics_deformable(current):
            return [str(current.GetPath())]
        if current.GetPath() == root.GetPath():
            break
        current = current.GetParent()

    candidates = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(root.GetPath())
        and _has_omniphysics_deformable(prim)
    ]
    return sorted(set(candidates))


def _scan_asset_connections(stage, root_path: str):
    from pxr import Sdf, UsdPhysics

    joint_connections: list[dict[str, object]] = []
    auto_attachments: list[dict[str, object]] = []
    attachment_like_prims: list[dict[str, object]] = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        applied = list(prim.GetAppliedSchemas())
        type_name = prim.GetTypeName()

        if prim.IsA(UsdPhysics.Joint):
            body0 = _relationship_targets(prim, "physics:body0")
            body1 = _relationship_targets(prim, "physics:body1")
            targets = body0 + body1
            if any(_path_is_below(target, root_path, Sdf) for target in targets):
                joint_connections.append(
                    {
                        "path": path,
                        "type_name": type_name,
                        "body0": body0,
                        "body1": body1,
                        "applied_schemas": applied,
                    }
                )

        if prim.HasAPI("PhysxAutoDeformableAttachmentAPI"):
            attachable0 = _relationship_targets(
                prim,
                "physxAutoDeformableAttachment:attachable0",
            )
            attachable1 = _relationship_targets(
                prim,
                "physxAutoDeformableAttachment:attachable1",
            )
            mask_shapes = _relationship_targets(
                prim,
                "physxAutoDeformableAttachment:maskShapes",
            )
            targets = attachable0 + attachable1
            if any(_path_is_below(target, root_path, Sdf) for target in targets):
                auto_attachments.append(
                    {
                        "path": path,
                        "attachable0": attachable0,
                        "attachable1": attachable1,
                        "mask_shapes": mask_shapes,
                        "applied_schemas": applied,
                    }
                )

        if (
            "Attachment" in type_name
            or any("Attachment" in schema for schema in applied)
        ):
            attachment_like_prims.append(
                {
                    "path": path,
                    "type_name": type_name,
                    "applied_schemas": applied,
                }
            )

    return (
        sorted(joint_connections, key=lambda item: str(item["path"])),
        sorted(auto_attachments, key=lambda item: str(item["path"])),
        sorted(attachment_like_prims, key=lambda item: str(item["path"])),
    )


def main() -> int:
    cfg = CONFIG.cable_mount
    output_path = CONFIG.camera.output_dir / "cable_asset_schema.json"
    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_path": cfg.usd_path,
        "root_path": cfg.root_path,
        "tracked_plug_path": cfg.tracked_plug_path,
        "asset_exists": False,
        "root_valid": False,
        "tracked_plug_valid": False,
        "tracked_plug_parent_path": "",
        "tracked_plug_is_rigid_body": False,
        "tracked_plug_applied_schemas": [],
        "deformable_candidates": [],
        "deformable_candidate_schemas": {},
        "joint_connections": [],
        "auto_deformable_attachments": [],
        "attachment_like_prims": [],
        "schema_family": "unsupported",
        "supported": False,
        "fatal_error": "",
    }

    simulation_app = None
    try:
        asset_exists = Path(cfg.usd_path).is_file()
        report["asset_exists"] = asset_exists
        if not asset_exists:
            raise FileNotFoundError(f"Cable USD not found: {cfg.usd_path}")

        simulation_app = SimulationApp(
            {
                "headless": True,
                "width": CONFIG.app.width,
                "height": CONFIG.app.height,
            }
        )

        import omni.usd
        from isaacsim.core.utils import stage as stage_utils
        from pxr import UsdPhysics

        omni.usd.get_context().new_stage()
        for _ in range(5):
            simulation_app.update()
        stage_utils.add_reference_to_stage(
            usd_path=cfg.usd_path,
            prim_path=cfg.root_path,
        )
        for _ in range(30):
            simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not create a valid stage")

        root = stage.GetPrimAtPath(cfg.root_path)
        plug = stage.GetPrimAtPath(cfg.tracked_plug_path)
        report["root_valid"] = bool(root.IsValid())
        report["tracked_plug_valid"] = bool(plug.IsValid())
        if plug.IsValid():
            report["tracked_plug_parent_path"] = str(plug.GetParent().GetPath())
            report["tracked_plug_is_rigid_body"] = bool(
                plug.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            report["tracked_plug_applied_schemas"] = list(
                plug.GetAppliedSchemas()
            )

        candidates = _candidate_paths(
            stage,
            cfg.root_path,
            cfg.tracked_plug_path,
        )
        report["deformable_candidates"] = candidates
        report["deformable_candidate_schemas"] = {
            candidate: list(
                stage.GetPrimAtPath(candidate).GetAppliedSchemas()
            )
            for candidate in candidates
        }

        (
            report["joint_connections"],
            report["auto_deformable_attachments"],
            report["attachment_like_prims"],
        ) = _scan_asset_connections(stage, cfg.root_path)

        supported = (
            root.IsValid()
            and plug.IsValid()
            and len(candidates) == 1
        )
        report["schema_family"] = "omniphysics" if supported else "unsupported"
        report["supported"] = bool(supported)
        _write_json_atomic(output_path, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0 if supported else 2
    except FileNotFoundError as exc:
        report["fatal_error"] = str(exc)
        _write_json_atomic(output_path, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1
    except Exception:
        report["fatal_error"] = traceback.format_exc()
        _write_json_atomic(output_path, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 1
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
