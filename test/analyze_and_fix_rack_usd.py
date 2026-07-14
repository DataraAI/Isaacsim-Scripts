#!/usr/bin/env python3
"""
Analyze and repair a USD asset so it can be referenced normally.

Designed for:
    /home/aayush/isaacsim_assets/datacenter/single_server_rack.usd

What this script does
---------------------
1. Opens the source USD.
2. Prints:
   - defaultPrim
   - root prims
   - stage units and up axis
   - composition errors
   - used layers
   - geometry counts and bounds for each root prim
3. Finds the top-level prim(s) that actually contain geometry.
4. Repairs the asset WITHOUT modifying the original:
   - One geometry root:
       copies the original to *_fixed.usd and authors defaultPrim.
   - Multiple geometry roots:
       creates *_fixed.usda as a wrapper asset with one /SingleServerRack
       default prim and explicit references to all geometry roots.
5. Reopens and verifies the repaired asset.

Run:
    ~/isaac-sim-6/python.sh \
        /home/aayush/Isaacsim-Scripts/detailedInsertion/analyze_and_fix_rack_usd.py

Optional:
    ~/isaac-sim-6/python.sh analyze_and_fix_rack_usd.py \
        --source /path/to/another_asset.usd

Analyze only:
    ~/isaac-sim-6/python.sh analyze_and_fix_rack_usd.py \
        --analyze-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pxr import Gf, Kind, Sdf, Usd, UsdGeom


DEFAULT_SOURCE = Path(
    "/home/aayush/isaacsim_assets/datacenter/single_server_rack.usd"
)

ASSET_ROOT_NAME = "SingleServerRack"
EMPTY_SENTINEL_LIMIT = 1.0e20


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class BoundsInfo:
    minimum: list[float]
    maximum: list[float]
    size: list[float]
    center: list[float]


@dataclass
class RootPrimInfo:
    path: str
    name: str
    type_name: str
    active: bool
    loaded: bool
    defined: bool
    abstract: bool
    instance: bool
    instanceable: bool
    kind: str
    descendant_count: int
    boundable_count: int
    mesh_count: int
    point_instancer_count: int
    material_count: int
    camera_count: int
    light_count: int
    has_valid_bounds: bool
    bounds: BoundsInfo | None

    @property
    def contains_geometry(self) -> bool:
        return (
            self.boundable_count > 0
            or self.mesh_count > 0
            or self.point_instancer_count > 0
            or self.has_valid_bounds
        )

    @property
    def largest_dimension(self) -> float:
        if self.bounds is None:
            return -1.0
        return max(self.bounds.size)


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def log(message: str = "") -> None:
    print(f"[RACK USD ANALYZER] {message}", flush=True)


def section(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


def warning(message: str) -> None:
    print(f"[RACK USD ANALYZER] WARNING: {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    print(f"[RACK USD ANALYZER] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(exit_code)


# ---------------------------------------------------------------------
# USD inspection helpers
# ---------------------------------------------------------------------

def vec3_to_list(value: Iterable[float]) -> list[float]:
    return [float(component) for component in value]


def range_to_bounds(aligned_range: Gf.Range3d) -> BoundsInfo | None:
    """
    Convert a USD range into validated numeric bounds.

    USD represents an empty range with enormous positive minimum values and
    enormous negative maximum values. Those values are finite, so merely
    checking math.isfinite() is not enough.
    """
    minimum = vec3_to_list(aligned_range.GetMin())
    maximum = vec3_to_list(aligned_range.GetMax())
    size = [
        maximum[index] - minimum[index]
        for index in range(3)
    ]

    values = minimum + maximum + size

    if not all(math.isfinite(value) for value in values):
        return None

    if any(abs(value) > EMPTY_SENTINEL_LIMIT for value in minimum + maximum):
        return None

    if any(value < 0.0 for value in size):
        return None

    center = [
        (minimum[index] + maximum[index]) / 2.0
        for index in range(3)
    ]

    return BoundsInfo(
        minimum=minimum,
        maximum=maximum,
        size=size,
        center=center,
    )


def compute_bounds(
    stage: Usd.Stage,
    prim: Usd.Prim,
) -> BoundsInfo | None:
    if not prim.IsValid():
        return None

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )

    world_bound = bbox_cache.ComputeWorldBound(prim)
    return range_to_bounds(world_bound.ComputeAlignedRange())


def is_material_prim(prim: Usd.Prim) -> bool:
    return prim.GetTypeName() in {
        "Material",
        "Shader",
        "NodeGraph",
    }


def is_light_prim(prim: Usd.Prim) -> bool:
    return prim.GetTypeName().endswith("Light")


def get_kind(prim: Usd.Prim) -> str:
    try:
        value = Usd.ModelAPI(prim).GetKind()
        return str(value) if value else ""
    except Exception:
        return ""


def inspect_root_prim(
    stage: Usd.Stage,
    root_prim: Usd.Prim,
) -> RootPrimInfo:
    descendant_count = 0
    boundable_count = 0
    mesh_count = 0
    point_instancer_count = 0
    material_count = 0
    camera_count = 0
    light_count = 0

    for prim in Usd.PrimRange(root_prim):
        if prim != root_prim:
            descendant_count += 1

        if prim.IsA(UsdGeom.Boundable):
            boundable_count += 1

        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1

        if prim.IsA(UsdGeom.PointInstancer):
            point_instancer_count += 1

        if is_material_prim(prim):
            material_count += 1

        if prim.IsA(UsdGeom.Camera):
            camera_count += 1

        if is_light_prim(prim):
            light_count += 1

    bounds = compute_bounds(stage, root_prim)

    return RootPrimInfo(
        path=str(root_prim.GetPath()),
        name=root_prim.GetName(),
        type_name=root_prim.GetTypeName() or "<untyped>",
        active=root_prim.IsActive(),
        loaded=root_prim.IsLoaded(),
        defined=root_prim.IsDefined(),
        abstract=root_prim.IsAbstract(),
        instance=root_prim.IsInstance(),
        instanceable=root_prim.IsInstanceable(),
        kind=get_kind(root_prim),
        descendant_count=descendant_count,
        boundable_count=boundable_count,
        mesh_count=mesh_count,
        point_instancer_count=point_instancer_count,
        material_count=material_count,
        camera_count=camera_count,
        light_count=light_count,
        has_valid_bounds=bounds is not None,
        bounds=bounds,
    )


def print_root_info(info: RootPrimInfo) -> None:
    log(f"Root prim: {info.path}")
    print(f"  name:                   {info.name}")
    print(f"  type:                   {info.type_name}")
    print(f"  active:                 {info.active}")
    print(f"  loaded:                 {info.loaded}")
    print(f"  defined:                {info.defined}")
    print(f"  abstract:               {info.abstract}")
    print(f"  instance:               {info.instance}")
    print(f"  instanceable:           {info.instanceable}")
    print(f"  kind:                   {info.kind or '<none>'}")
    print(f"  descendants:            {info.descendant_count}")
    print(f"  boundable prims:        {info.boundable_count}")
    print(f"  meshes:                 {info.mesh_count}")
    print(f"  point instancers:       {info.point_instancer_count}")
    print(f"  material/shader prims:  {info.material_count}")
    print(f"  cameras:                {info.camera_count}")
    print(f"  lights:                 {info.light_count}")
    print(f"  contains geometry:      {info.contains_geometry}")

    if info.bounds is None:
        print("  bounds:                 <NO VALID BOUNDS>")
    else:
        print(
            "  bounds min:             "
            f"{[round(value, 6) for value in info.bounds.minimum]}"
        )
        print(
            "  bounds max:             "
            f"{[round(value, 6) for value in info.bounds.maximum]}"
        )
        print(
            "  bounds size (meters):   "
            f"{[round(value, 6) for value in info.bounds.size]}"
        )
        print(
            "  bounds center:          "
            f"{[round(value, 6) for value in info.bounds.center]}"
        )

    print()


def analyze_stage(
    source_path: Path,
) -> tuple[Usd.Stage, list[RootPrimInfo], dict]:
    section("OPEN SOURCE USD")

    stage = Usd.Stage.Open(str(source_path), Usd.Stage.LoadAll)
    if stage is None:
        fail(f"Usd.Stage.Open() failed:\n  {source_path}")

    default_prim = stage.GetDefaultPrim()
    root_prims = list(stage.GetPseudoRoot().GetChildren())

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))

    composition_errors = [
        str(error)
        for error in stage.GetCompositionErrors()
    ]

    used_layers = []
    for layer in stage.GetUsedLayers():
        used_layers.append(
            {
                "identifier": layer.identifier,
                "real_path": layer.realPath,
                "anonymous": layer.anonymous,
                "dirty": layer.dirty,
            }
        )

    log(f"Source file:      {source_path}")
    log(f"Root layer:       {stage.GetRootLayer().identifier}")
    log(
        "defaultPrim:      "
        + (
            str(default_prim.GetPath())
            if default_prim.IsValid()
            else "<NONE>"
        )
    )
    log(f"metersPerUnit:    {meters_per_unit}")
    log(f"upAxis:           {up_axis}")
    log(f"root prim count:  {len(root_prims)}")
    log(f"used layer count: {len(used_layers)}")

    if stage.HasAuthoredMetadata("defaultPrim"):
        log("defaultPrim metadata is authored.")
    else:
        warning("defaultPrim metadata is NOT authored.")

    if composition_errors:
        warning(
            f"The stage reported {len(composition_errors)} composition error(s):"
        )
        for error in composition_errors:
            print(f"  - {error}")
    else:
        log("Composition errors: none")

    section("USED LAYERS")
    for index, layer_info in enumerate(used_layers):
        print(f"[{index}] identifier: {layer_info['identifier']}")
        print(f"    real path:  {layer_info['real_path'] or '<none>'}")
        print(f"    anonymous:  {layer_info['anonymous']}")
        print(f"    dirty:      {layer_info['dirty']}")

    section("TOP-LEVEL ROOT PRIMS")

    root_infos = [
        inspect_root_prim(stage, prim)
        for prim in root_prims
    ]

    for info in root_infos:
        print_root_info(info)

    geometry_roots = [
        info
        for info in root_infos
        if info.contains_geometry
    ]

    geometry_roots.sort(
        key=lambda info: (
            info.boundable_count,
            info.mesh_count,
            info.point_instancer_count,
            info.largest_dimension,
            info.descendant_count,
        ),
        reverse=True,
    )

    summary = {
        "source_path": str(source_path),
        "root_layer": stage.GetRootLayer().identifier,
        "default_prim": (
            str(default_prim.GetPath())
            if default_prim.IsValid()
            else None
        ),
        "has_authored_default_prim": stage.HasAuthoredMetadata("defaultPrim"),
        "meters_per_unit": meters_per_unit,
        "up_axis": up_axis,
        "composition_errors": composition_errors,
        "used_layers": used_layers,
        "root_prims": [asdict(info) for info in root_infos],
        "geometry_root_paths": [
            info.path
            for info in geometry_roots
        ],
    }

    section("ANALYSIS RESULT")

    if not geometry_roots:
        warning(
            "No top-level root prim contains detectable renderable geometry."
        )
        warning(
            "This file may be an empty assembly, may depend on unresolved "
            "external assets, or may contain unsupported/custom geometry."
        )
    else:
        log(
            "Geometry-bearing root prim(s), ranked strongest first:"
        )
        for info in geometry_roots:
            print(
                f"  - {info.path}"
                f" | boundables={info.boundable_count}"
                f" | meshes={info.mesh_count}"
                f" | size={info.bounds.size if info.bounds else None}"
            )

    return stage, geometry_roots, summary


# ---------------------------------------------------------------------
# Repair helpers
# ---------------------------------------------------------------------

def make_output_path(
    source_path: Path,
    use_wrapper: bool,
) -> Path:
    suffix = ".usda" if use_wrapper else source_path.suffix
    return source_path.with_name(
        f"{source_path.stem}_fixed{suffix}"
    )


def write_analysis_report(
    source_path: Path,
    summary: dict,
) -> Path:
    report_path = source_path.with_name(
        f"{source_path.stem}_analysis.json"
    )

    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return report_path


def create_fixed_copy_with_default_prim(
    source_path: Path,
    selected_root_path: str,
) -> Path:
    """
    Copy the original file and set defaultPrim on the copy.

    This is used only when one top-level root contains all detected geometry.
    Keeping the repaired file in the same directory preserves relative asset
    paths used by references, payloads, textures, or sublayers.
    """
    output_path = make_output_path(
        source_path=source_path,
        use_wrapper=False,
    )

    if output_path.exists():
        output_path.unlink()

    shutil.copy2(source_path, output_path)

    fixed_stage = Usd.Stage.Open(
        str(output_path),
        Usd.Stage.LoadAll,
    )

    if fixed_stage is None:
        fail(f"Could not reopen copied USD:\n  {output_path}")

    selected_prim = fixed_stage.GetPrimAtPath(selected_root_path)

    if not selected_prim.IsValid():
        fail(
            "The selected root prim does not exist in the copied stage:\n"
            f"  {selected_root_path}"
        )

    fixed_stage.SetDefaultPrim(selected_prim)

    if not fixed_stage.GetRootLayer().Save():
        fail(f"Failed to save repaired USD:\n  {output_path}")

    log(
        "Created a corrected copy with an authored defaultPrim:\n"
        f"  output:      {output_path}\n"
        f"  defaultPrim: {selected_root_path}"
    )

    return output_path


def make_valid_identifier(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)

    if not cleaned:
        cleaned = "AssetRoot"

    if cleaned[0].isdigit():
        cleaned = f"Root_{cleaned}"

    return cleaned


def create_wrapper_asset(
    source_path: Path,
    geometry_roots: list[RootPrimInfo],
    source_stage: Usd.Stage,
) -> Path:
    """
    Create a non-destructive wrapper when geometry spans multiple roots.

    The wrapper owns one clean /SingleServerRack default prim and explicitly
    references each geometry-bearing source root below it.
    """
    output_path = make_output_path(
        source_path=source_path,
        use_wrapper=True,
    )

    if output_path.exists():
        output_path.unlink()

    wrapper_stage = Usd.Stage.CreateNew(str(output_path))

    if wrapper_stage is None:
        fail(f"Could not create wrapper stage:\n  {output_path}")

    # Match source-stage units and up-axis.
    UsdGeom.SetStageMetersPerUnit(
        wrapper_stage,
        float(UsdGeom.GetStageMetersPerUnit(source_stage)),
    )
    UsdGeom.SetStageUpAxis(
        wrapper_stage,
        UsdGeom.GetStageUpAxis(source_stage),
    )

    wrapper_root_path = Sdf.Path(f"/{ASSET_ROOT_NAME}")
    wrapper_root = UsdGeom.Xform.Define(
        wrapper_stage,
        wrapper_root_path,
    ).GetPrim()

    Usd.ModelAPI(wrapper_root).SetKind(
        Kind.Tokens.component
    )

    wrapper_stage.SetDefaultPrim(wrapper_root)

    # Wrapper and source are in the same directory, so a filename-only asset
    # path is portable if the whole asset folder is moved.
    relative_source_asset = source_path.name

    used_names: set[str] = set()

    for index, root_info in enumerate(geometry_roots):
        base_name = make_valid_identifier(
            root_info.name or f"Root_{index}"
        )

        child_name = base_name
        counter = 2

        while child_name in used_names:
            child_name = f"{base_name}_{counter}"
            counter += 1

        used_names.add(child_name)

        child_path = wrapper_root_path.AppendChild(child_name)

        child_prim = UsdGeom.Xform.Define(
            wrapper_stage,
            child_path,
        ).GetPrim()

        reference = Sdf.Reference(
            assetPath=relative_source_asset,
            primPath=Sdf.Path(root_info.path),
        )

        if not child_prim.GetReferences().AddReference(reference):
            fail(
                "Failed to add source reference:\n"
                f"  source asset: {relative_source_asset}\n"
                f"  source prim:  {root_info.path}\n"
                f"  wrapper prim: {child_path}"
            )

        log(
            "Wrapper reference:\n"
            f"  {child_path} -> "
            f"{relative_source_asset}{root_info.path}"
        )

    wrapper_stage.GetRootLayer().documentation = (
        "Auto-generated wrapper for single_server_rack.usd. "
        "Provides one valid default prim while preserving the original file."
    )

    if not wrapper_stage.GetRootLayer().Save():
        fail(f"Failed to save wrapper USD:\n  {output_path}")

    log(
        "Created wrapper asset:\n"
        f"  output:      {output_path}\n"
        f"  defaultPrim: {wrapper_root_path}"
    )

    return output_path


# ---------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------

def verify_repaired_asset(
    repaired_path: Path,
) -> dict:
    section("VERIFY REPAIRED ASSET")

    stage = Usd.Stage.Open(
        str(repaired_path),
        Usd.Stage.LoadAll,
    )

    if stage is None:
        fail(f"Could not open repaired asset:\n  {repaired_path}")

    default_prim = stage.GetDefaultPrim()

    if not default_prim.IsValid():
        fail(
            "The repaired asset still does not have a valid defaultPrim:\n"
            f"  {repaired_path}"
        )

    bounds = compute_bounds(stage, default_prim)

    boundable_count = 0
    mesh_count = 0
    descendant_count = 0

    for prim in Usd.PrimRange(default_prim):
        if prim != default_prim:
            descendant_count += 1

        if prim.IsA(UsdGeom.Boundable):
            boundable_count += 1

        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1

    composition_errors = [
        str(error)
        for error in stage.GetCompositionErrors()
    ]

    log(f"Repaired asset:        {repaired_path}")
    log(f"defaultPrim:           {default_prim.GetPath()}")
    log(f"descendants:           {descendant_count}")
    log(f"boundable prims:       {boundable_count}")
    log(f"meshes:                {mesh_count}")
    log(f"valid default bounds:  {bounds is not None}")
    log(f"composition errors:    {len(composition_errors)}")

    if bounds is not None:
        log(
            "defaultPrim size:      "
            f"{[round(value, 6) for value in bounds.size]} meters"
        )

    if composition_errors:
        for error in composition_errors:
            warning(error)

    if descendant_count <= 0:
        fail(
            "Verification failed: the repaired defaultPrim has no descendants."
        )

    if boundable_count <= 0 and bounds is None:
        fail(
            "Verification failed: the repaired defaultPrim has no detectable "
            "geometry and no valid bounds."
        )

    log("VERIFICATION PASSED")

    return {
        "repaired_path": str(repaired_path),
        "default_prim": str(default_prim.GetPath()),
        "descendant_count": descendant_count,
        "boundable_count": boundable_count,
        "mesh_count": mesh_count,
        "bounds": asdict(bounds) if bounds else None,
        "composition_errors": composition_errors,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a USD asset and create a safe repaired copy or wrapper "
            "with a valid defaultPrim."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source USD path. Default: {DEFAULT_SOURCE}",
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Print and save the analysis without creating a repaired asset.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source.expanduser().resolve()

    if not source_path.is_file():
        fail(f"Source USD does not exist:\n  {source_path}")

    if source_path.suffix.lower() not in {
        ".usd",
        ".usda",
        ".usdc",
    }:
        warning(
            f"Unexpected extension '{source_path.suffix}'. "
            "Attempting to open it as USD anyway."
        )

    stage, geometry_roots, summary = analyze_stage(source_path)

    report_path = write_analysis_report(
        source_path,
        summary,
    )

    log(f"Analysis report saved:\n  {report_path}")

    if args.analyze_only:
        section("ANALYZE-ONLY COMPLETE")
        log("No USD files were modified or created.")
        return 0

    if not geometry_roots:
        fail(
            "No repair was attempted because no geometry-bearing root prim "
            "could be identified. Review the analysis output and composition "
            "errors first."
        )

    section("CREATE REPAIRED ASSET")

    if len(geometry_roots) == 1:
        selected = geometry_roots[0]

        log(
            "One geometry-bearing top-level root was found. "
            "Creating a corrected copy and authoring defaultPrim."
        )

        repaired_path = create_fixed_copy_with_default_prim(
            source_path=source_path,
            selected_root_path=selected.path,
        )

        repair_mode = "fixed_copy"

    else:
        log(
            f"{len(geometry_roots)} geometry-bearing top-level roots were "
            "found. Creating a wrapper so no geometry root is discarded."
        )

        repaired_path = create_wrapper_asset(
            source_path=source_path,
            geometry_roots=geometry_roots,
            source_stage=stage,
        )

        repair_mode = "wrapper"

    verification = verify_repaired_asset(repaired_path)

    result_path = source_path.with_name(
        f"{source_path.stem}_repair_result.json"
    )

    result_path.write_text(
        json.dumps(
            {
                "source": str(source_path),
                "repair_mode": repair_mode,
                "analysis_report": str(report_path),
                **verification,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    section("DONE")

    log(
        "Use this repaired USD in the Franka scene:\n"
        f"  {repaired_path}"
    )

    log(
        "Repair result saved:\n"
        f"  {result_path}"
    )

    log(
        "Your original source USD was NOT modified:\n"
        f"  {source_path}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(
            "\n[RACK USD ANALYZER] UNHANDLED ERROR\n"
            + traceback.format_exc(),
            file=sys.stderr,
            flush=True,
        )
        raise
