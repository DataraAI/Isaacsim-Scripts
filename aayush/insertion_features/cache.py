"""Cached insertion features, stored in each connector's local frame.

Crystal-head cache (metres, ``crystal_head_local``):

    from insertion_features.cache import (
        E_crystal_head1_45,
        world_features_for_head,
    )

    world = world_features_for_head("E_crystal_head1_45", world_from_head)

RJ45 jack cache (metres, ``rj45_group_local``). DataHall is authored in
centimetres, so pass ``meters_per_unit`` when re-posing from a live USD Gf
transform:

    from insertion_features.cache import jack_upper_c0, world_features_for_jack

    world = world_features_for_jack(
        "jack_upper_c0",
        world_from_pack,
        meters_per_unit=0.01,
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from insertion_features.geometry import (
    ConnectorFeatures,
    Plane,
    transform_connector_features,
)

DEFAULT_CACHE_PATH = Path(__file__).with_name("cable_features_local.json")
DEFAULT_PORT_CACHE_PATH = Path(__file__).with_name("port_features_local.json")
CACHE_FRAME = "crystal_head_local"
PORT_CACHE_FRAME = "rj45_group_local"
CACHE_LENGTH_UNIT = "meters"

CACHED_HEADS: dict[str, ConnectorFeatures] = {}
CACHED_JACKS: dict[str, ConnectorFeatures] = {}


def _vector(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("expected a finite length-3 vector")
    return array


def _points(value, count: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("expected a finite (N, 3) point array")
    if count is not None and array.shape[0] != count:
        raise ValueError(f"expected {count} points, got {array.shape[0]}")
    return array


def _tolist(value: np.ndarray) -> list:
    return np.asarray(value, dtype=np.float64).tolist()


def connector_features_to_dict(features: ConnectorFeatures) -> dict[str, object]:
    return {
        "insertion_axis": _tolist(features.insertion_axis),
        "mating_plane": {
            "point": _tolist(features.mating_plane.point),
            "normal": _tolist(features.mating_plane.normal),
        },
        "mating_center": _tolist(features.mating_center),
        "mating_corners": _tolist(features.mating_corners),
        "latch_plane": {
            "point": _tolist(features.latch_plane.point),
            "normal": _tolist(features.latch_plane.normal),
        },
        "latch_corners": _tolist(features.latch_corners),
        "latch_keypoints": _tolist(features.latch_keypoints),
        "width_axis": _tolist(features.width_axis),
        "up_axis": _tolist(features.up_axis),
    }


def connector_features_from_dict(payload: dict) -> ConnectorFeatures:
    insertion_axis = _vector(payload["insertion_axis"])
    mating_center = _vector(payload["mating_center"])
    latch_point = _vector(payload["latch_plane"]["point"])
    return ConnectorFeatures(
        insertion_axis=insertion_axis,
        mating_plane=Plane(
            point=_vector(payload["mating_plane"]["point"]),
            normal=_vector(payload["mating_plane"]["normal"]),
        ),
        mating_center=mating_center,
        mating_corners=_points(payload["mating_corners"], 4),
        latch_plane=Plane(
            point=latch_point,
            normal=_vector(payload["latch_plane"]["normal"]),
        ),
        latch_corners=_points(payload["latch_corners"], 4),
        latch_keypoints=_points(payload["latch_keypoints"], 2),
        width_axis=_vector(payload["width_axis"]),
        up_axis=_vector(payload["up_axis"]),
    )


def save_local_cable_features(
    heads: dict[str, ConnectorFeatures],
    path: Path | str = DEFAULT_CACHE_PATH,
    *,
    source_usd: str = "",
) -> Path:
    """Write head-local features so later scripts can load them as variables."""

    if not heads:
        raise ValueError("heads must not be empty")
    destination = Path(path)
    payload = {
        "asset": "model_Networkcable1_69323",
        "source_usd": str(source_usd),
        "frame": CACHE_FRAME,
        "length_unit": CACHE_LENGTH_UNIT,
        "heads": {
            name: connector_features_to_dict(features) for name, features in heads.items()
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_local_cable_features(
    path: Path | str = DEFAULT_CACHE_PATH,
) -> dict[str, ConnectorFeatures]:
    """Load cached head-local features. Keys are crystal-head prim names."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cable feature cache is missing: {cache_path}")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("frame") != CACHE_FRAME:
        raise ValueError(
            f"Cache frame must be {CACHE_FRAME!r}, got {payload.get('frame')!r}"
        )
    heads_payload = payload.get("heads")
    if not isinstance(heads_payload, dict) or not heads_payload:
        raise ValueError("Cache is missing crystal-head entries")
    return {
        str(name): connector_features_from_dict(entry)
        for name, entry in heads_payload.items()
    }


def resolve_cached_head_name(
    head_name: str,
    heads: dict[str, ConnectorFeatures] | None = None,
) -> str:
    """Accept a prim name or a full prim path ending in that name."""

    available = CACHED_HEADS if heads is None else heads
    if head_name in available:
        return head_name
    short = str(head_name).rstrip("/").split("/")[-1]
    if short in available:
        return short
    known = ", ".join(sorted(available)) or "<none>"
    raise KeyError(f"Unknown crystal head {head_name!r}. Cached heads: {known}")


def local_features_for_head(
    head_name: str,
    cache_path: Path | str | None = None,
) -> ConnectorFeatures:
    """Return cached features in the named crystal head's local frame."""

    if cache_path is None and CACHED_HEADS:
        heads = CACHED_HEADS
    else:
        heads = load_local_cable_features(
            DEFAULT_CACHE_PATH if cache_path is None else cache_path
        )
    return heads[resolve_cached_head_name(head_name, heads)]


def world_features_for_head(
    head_name: str,
    world_from_head: np.ndarray,
    cache_path: Path | str | None = None,
) -> ConnectorFeatures:
    """Apply a live crystal-head world pose to cached local features.

    ``world_from_head`` is a 4x4 column-vector transform (USD Gf matrix
    transposed), mapping head-local coordinates into the current stage world.
    """

    local = local_features_for_head(head_name, cache_path=cache_path)
    return transform_connector_features(local, world_from_head)


def _bind_imported_variables(heads: dict[str, ConnectorFeatures]) -> None:
    global CACHED_HEADS
    CACHED_HEADS = dict(heads)
    for name, features in heads.items():
        globals()[name] = features


def meters_transform_from_stage(
    transform: np.ndarray,
    meters_per_unit: float,
) -> np.ndarray:
    """Scale a stage-unit Gf.T column-vector transform into meters.

    Rotation/scale is unchanged; translation is multiplied by ``meters_per_unit``.
    """

    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4).copy()
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be finite")
    matrix[:3, 3] *= float(meters_per_unit)
    return matrix


def length_scale_transform(scale: float) -> np.ndarray:
    """Uniform length scale as a 4x4 column-vector transform."""

    value = float(scale)
    if abs(value) <= 1.0e-12:
        raise ValueError("scale must be nonzero")
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = value
    matrix[1, 1] = value
    matrix[2, 2] = value
    return matrix


def save_local_port_features(
    jacks: dict[str, ConnectorFeatures],
    path: Path | str = DEFAULT_PORT_CACHE_PATH,
    *,
    source_usd: str = "",
    pack_prim_path: str = "",
    metadata: dict[str, dict] | None = None,
) -> Path:
    """Write pack-local jack features so later scripts can load them as variables."""

    if not jacks:
        raise ValueError("jacks must not be empty")
    destination = Path(path)
    extra = metadata or {}
    payload_jacks = {}
    for name, features in jacks.items():
        entry = connector_features_to_dict(features)
        entry.update(extra.get(name, {}))
        payload_jacks[name] = entry
    payload = {
        "asset": "DataHall_6r_ur5e",
        "source_usd": str(source_usd),
        "pack_prim_path": str(pack_prim_path),
        "frame": PORT_CACHE_FRAME,
        "length_unit": CACHE_LENGTH_UNIT,
        "jacks": payload_jacks,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_local_port_features(
    path: Path | str = DEFAULT_PORT_CACHE_PATH,
) -> dict[str, ConnectorFeatures]:
    """Load cached pack-local jack features. Keys are jack names."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Port feature cache is missing: {cache_path}")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("frame") != PORT_CACHE_FRAME:
        raise ValueError(
            f"Cache frame must be {PORT_CACHE_FRAME!r}, got {payload.get('frame')!r}"
        )
    jacks_payload = payload.get("jacks")
    if not isinstance(jacks_payload, dict) or not jacks_payload:
        raise ValueError("Cache is missing jack entries")
    return {
        str(name): connector_features_from_dict(entry)
        for name, entry in jacks_payload.items()
    }


def resolve_cached_jack_name(
    jack_name: str,
    jacks: dict[str, ConnectorFeatures] | None = None,
) -> str:
    """Accept a jack name or a copper-group name stored in the cache file."""

    available = CACHED_JACKS if jacks is None else jacks
    if jack_name in available:
        return jack_name
    short = str(jack_name).rstrip("/").split("/")[-1]
    if short in available:
        return short
    known = ", ".join(sorted(available)) or "<none>"
    raise KeyError(f"Unknown RJ45 jack {jack_name!r}. Cached jacks: {known}")


def local_features_for_jack(
    jack_name: str,
    cache_path: Path | str | None = None,
) -> ConnectorFeatures:
    """Return cached features in the parent RJ45 group prim's local frame."""

    if cache_path is None and CACHED_JACKS:
        jacks = CACHED_JACKS
    else:
        jacks = load_local_port_features(
            DEFAULT_PORT_CACHE_PATH if cache_path is None else cache_path
        )
    return jacks[resolve_cached_jack_name(jack_name, jacks)]


def world_features_for_jack(
    jack_name: str,
    world_from_pack: np.ndarray,
    cache_path: Path | str | None = None,
    *,
    meters_per_unit: float = 1.0,
) -> ConnectorFeatures:
    """Apply a live RJ45-group world pose to cached pack-local features.

    ``world_from_pack`` is a 4x4 column-vector transform (USD Gf matrix
    transposed). Pass the stage's ``meters_per_unit`` so a centimetre DataHall
    pose is converted before being applied to the metre cache.
    """

    local = local_features_for_jack(jack_name, cache_path=cache_path)
    world_from_pack_m = meters_transform_from_stage(world_from_pack, meters_per_unit)
    return transform_connector_features(local, world_from_pack_m)


def _bind_imported_jacks(jacks: dict[str, ConnectorFeatures]) -> None:
    global CACHED_JACKS
    CACHED_JACKS = dict(jacks)
    for name, features in jacks.items():
        globals()[name] = features


try:
    _bind_imported_variables(load_local_cable_features())
except FileNotFoundError:
    CACHED_HEADS = {}

try:
    _bind_imported_jacks(load_local_port_features())
except FileNotFoundError:
    CACHED_JACKS = {}
