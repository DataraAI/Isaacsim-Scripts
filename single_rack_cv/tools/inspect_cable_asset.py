#!/usr/bin/env python3
"""Inspect the configured cable for Isaac Sim 6 Omni Physics deformable schemas."""

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
    # Isaac Sim 6 codeless schemas are queried by API-schema token.
    return bool(prim.HasAPI("OmniPhysicsDeformableBodyAPI"))


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
        "tracked_plug_applied_schemas": [],
        "deformable_candidates": [],
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

        omni.usd.get_context().new_stage()
        for _ in range(5):
            simulation_app.update()
        stage_utils.add_reference_to_stage(
            usd_path=cfg.usd_path,
            path=cfg.root_path,
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
        report["tracked_plug_applied_schemas"] = (
            list(plug.GetAppliedSchemas()) if plug.IsValid() else []
        )
        candidates = _candidate_paths(
            stage,
            cfg.root_path,
            cfg.tracked_plug_path,
        )
        report["deformable_candidates"] = candidates

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
