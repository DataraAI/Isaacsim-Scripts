"""Spawn DataHall and visualize extracted RJ45 jack insertion features.

Opens the DataHall USD read-only. Debug markers exist only on the live stage;
the original asset is never saved.

By default Isaac Sim stays open with debug markers visible. Run from
Isaacsim-Scripts:

    /home/aayush/isaacsim/python.sh aayush/insertion_features/port_main.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AAYUSH_DIR = Path(__file__).resolve().parents[1]
if str(AAYUSH_DIR) not in sys.path:
    sys.path.insert(0, str(AAYUSH_DIR))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usd",
        type=Path,
        default=None,
        help="DataHall USD to open (does not modify the file)",
    )
    parser.add_argument(
        "--pack",
        type=str,
        default=None,
        help="Prim path of the 12-pack RJ45 group to extract",
    )
    parser.add_argument("--headless", action="store_true", help="Extract and print, then exit")
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Skip debug-marker spawn (print extracted features only)",
    )
    parser.add_argument(
        "--write-cache",
        action="store_true",
        help="Rewrite aayush/insertion_features/port_features_local.json from this extract",
    )
    args, _unknown = parser.parse_known_args()
    return args


ARGS = _parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": bool(ARGS.headless)})

from pxr import Usd, UsdGeom  # noqa: E402

from insertion_features.port_features import (  # noqa: E402
    DATAHALL_USD,
    DEBUG_MARKER_ROOT,
    EXAMPLE_PACK_PATH,
    extract_rj45_group_features,
    format_port_feature_report,
    spawn_port_feature_markers,
    write_local_port_feature_cache,
)


def _wait_for_stage_loading() -> None:
    import time

    import omni.usd

    usd_context = omni.usd.get_context()
    stable_frames = 0
    for _ in range(3600):
        if not simulation_app.is_running():
            return
        simulation_app.update()
        try:
            _message, files_loaded, total_files = usd_context.get_stage_loading_status()
            still_loading = bool(files_loaded or total_files)
        except AttributeError:
            still_loading = False
        if still_loading:
            stable_frames = 0
        else:
            stable_frames += 1
            if stable_frames >= 15:
                return
        time.sleep(0.01)


def _look_at_pack(stage, pack_path: str) -> None:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except Exception as exc:
        print(f"[PORT FEATURES] Could not import set_camera_view: {exc}")
        return
    root = stage.GetPrimAtPath(pack_path)
    if not root or not root.IsValid():
        print(f"[PORT FEATURES] Pack prim missing, skipping camera: {pack_path}")
        return
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(root).ComputeAlignedBox()
    minimum = box.GetMin()
    maximum = box.GetMax()
    center = [
        0.5 * (float(minimum[0]) + float(maximum[0])),
        0.5 * (float(minimum[1]) + float(maximum[1])),
        0.5 * (float(minimum[2]) + float(maximum[2])),
    ]
    size = max(
        float(maximum[0] - minimum[0]),
        float(maximum[1] - minimum[1]),
        float(maximum[2] - minimum[2]),
        5.0,
    )
    if not all(value == value and abs(value) < 1.0e12 for value in center + [size]):
        print("[PORT FEATURES] Pack bbox is not finite, skipping camera")
        return
    try:
        set_camera_view(
            eye=[center[0] - 2.8 * size, center[1], center[2] + 0.35 * size],
            target=center,
        )
    except Exception as exc:
        print(f"[PORT FEATURES] Could not set camera: {exc}")


def main() -> int:
    import omni.usd

    usd_path = Path(ARGS.usd).expanduser() if ARGS.usd is not None else DATAHALL_USD
    pack_path = ARGS.pack if ARGS.pack is not None else EXAMPLE_PACK_PATH
    print(f"[PORT FEATURES] Opening DataHall read-only: {usd_path}")
    usd_context = omni.usd.get_context()
    opened = usd_context.open_stage(str(usd_path))
    if opened is False:
        raise RuntimeError(f"Could not open stage: {usd_path}")
    _wait_for_stage_loading()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {usd_path}")

    jacks = extract_rj45_group_features(stage, pack_path)
    print(format_port_feature_report(jacks))
    if ARGS.write_cache:
        cache_path = write_local_port_feature_cache(stage, jacks, source_usd=usd_path)
        print(f"[PORT FEATURES] Wrote pack-local cache {cache_path}")
    if not ARGS.no_markers:
        spawn_port_feature_markers(stage, jacks)
        print(f"[PORT FEATURES] Debug markers spawned under {DEBUG_MARKER_ROOT}")
        print("[PORT FEATURES] Original DataHall USD is not saved.")

    if ARGS.headless:
        print("[PORT FEATURES] Headless extract finished.")
        return 0

    print("[PORT FEATURES] Isaac Sim will stay open until you quit.")
    _look_at_pack(stage, pack_path)
    while simulation_app.is_running():
        simulation_app.update()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if not simulation_app.is_exiting():
            simulation_app.close()
