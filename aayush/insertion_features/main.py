"""Spawn the network cable and visualize extracted insertion features.

Run from Isaacsim-Scripts:

    /home/aayush/isaacsim/python.sh aayush/insertion_features/main.py
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
        help="Network-cable USD to reference (does not modify the file)",
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
        help="Rewrite aayush/insertion_features/cable_features_local.json from this extract",
    )
    return parser.parse_args()


ARGS = _parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": bool(ARGS.headless)})

from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

from insertion_features.cable_features import (  # noqa: E402
    DEBUG_MARKER_ROOT,
    NETWORK_CABLE_ROOT_PATH,
    NETWORK_CABLE_USD,
    extract_crystal_head_features,
    format_feature_report,
    spawn_cable_feature_markers,
    write_local_feature_cache,
)


def _wait_for_stage_loading() -> None:
    import time

    import omni.usd

    usd_context = omni.usd.get_context()
    stable_frames = 0
    for _ in range(3600):
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
            if stable_frames >= 10:
                return
        time.sleep(0.01)


def _look_at_cable(stage) -> None:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except Exception:
        return
    root = stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH)
    if not root or not root.IsValid():
        return
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(root).ComputeAlignedBox()
    minimum = box.GetMin()
    maximum = box.GetMax()
    center = (
        0.5 * (float(minimum[0]) + float(maximum[0])),
        0.5 * (float(minimum[1]) + float(maximum[1])),
        0.5 * (float(minimum[2]) + float(maximum[2])),
    )
    size = max(
        float(maximum[0] - minimum[0]),
        float(maximum[1] - minimum[1]),
        float(maximum[2] - minimum[2]),
        0.05,
    )
    set_camera_view(
        eye=[center[0], center[1] - 2.4 * size, center[2] + 0.9 * size],
        target=list(center),
    )


def main() -> int:
    import omni.usd

    usd_path = Path(ARGS.usd).expanduser() if ARGS.usd is not None else NETWORK_CABLE_USD
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    light.CreateIntensityAttr(900.0)

    print(f"[CABLE FEATURES] Referencing cable (read-only): {usd_path}")
    add_reference_to_stage(usd_path=str(usd_path), prim_path=NETWORK_CABLE_ROOT_PATH)
    _wait_for_stage_loading()

    heads = extract_crystal_head_features(stage, NETWORK_CABLE_ROOT_PATH)
    print(format_feature_report(heads))
    if ARGS.write_cache:
        cache_path = write_local_feature_cache(stage, heads, source_usd=usd_path)
        print(f"[CABLE FEATURES] Wrote head-local cache {cache_path}")
    if not ARGS.no_markers:
        spawn_cable_feature_markers(stage, heads)
        print(f"[CABLE FEATURES] Debug markers spawned under {DEBUG_MARKER_ROOT}")

    _look_at_cable(stage)
    simulation_app.update()
    if ARGS.headless:
        return 0
    print("[CABLE FEATURES] Isaac Sim will stay open until you quit.")
    while simulation_app.is_running():
        simulation_app.update()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
