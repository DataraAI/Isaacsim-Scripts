"""Pure-numpy RJ45 jack-opening feature extraction.

An Ethernet jack opening is a stepped silhouette: a wide entry rectangle, a
narrower latch channel on top of that rectangle, and a still-narrower tip.
The mating plane is the wide entry rectangle, not the full silhouette AABB.
Latch keypoints are the two latch-ward corners of the narrower channel.
"""

from __future__ import annotations

import numpy as np

from insertion_features.geometry import (
    ConnectorFeatures,
    Plane,
    _aabb_corners,
    _collect_facing_faces,
    _finite_points,
    _in_plane_axes,
    _unit,
)

_WIDTH_QUANTIZE_M = 5.0e-5
_EDGE_TOL_M = 1.5e-4
_WIDE_SPAN_RATIO = 0.85
_OPENING_PLANE_GAP_M = 0.00080
_APERTURE_CLUSTER_GAP_M = 0.00150


def _unique_levels(values: np.ndarray, atol: float) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    groups: list[list[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        if float(value) - groups[-1][-1] <= atol:
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return np.array([float(np.mean(group)) for group in groups], dtype=np.float64)


def _on_levels(values: np.ndarray, levels: np.ndarray, atol: float) -> np.ndarray:
    if levels.size == 0:
        return np.zeros(values.shape[0], dtype=bool)
    delta = np.abs(values.reshape(-1, 1) - levels.reshape(1, -1))
    return np.min(delta, axis=1) <= atol


def extract_port_features_from_aperture_points(
    points: np.ndarray,
    insertion_axis: np.ndarray,
) -> ConnectorFeatures:
    """Derive jack mating plane, insertion axis, opening center, and latch keypoints.

    ``points`` are coplanar samples of one jack's stepped opening silhouette.
    ``insertion_axis`` points into the jack (cable travel during insertion).
    """

    aperture = _finite_points(points, "points")
    insertion = _unit(insertion_axis, "insertion_axis")
    axis_a, axis_b = _in_plane_axes(aperture, insertion)
    span_a = float((aperture @ axis_a).max() - (aperture @ axis_a).min())
    span_b = float((aperture @ axis_b).max() - (aperture @ axis_b).min())
    if span_a >= span_b:
        width_guess, up_guess = axis_a, axis_b
    else:
        width_guess, up_guess = axis_b, axis_a

    height_values = aperture @ up_guess
    width_values = aperture @ width_guess
    width_levels = _unique_levels(width_values, _WIDTH_QUANTIZE_M)
    if width_levels.size < 2:
        raise RuntimeError("Port aperture is degenerate in width")
    outer_mask = _on_levels(width_values, width_levels[np.array([0, -1])], _EDGE_TOL_M)
    if not np.any(outer_mask):
        raise RuntimeError("Port aperture has no outer-edge samples")
    big_h_lo = float(height_values[outer_mask].min())
    big_h_hi = float(height_values[outer_mask].max())
    silhouette_lo = float(height_values.min())
    silhouette_hi = float(height_values.max())
    if (silhouette_hi - big_h_hi) + 1.0e-9 < (big_h_lo - silhouette_lo):
        up_guess = -up_guess

    width_axis = _unit(np.cross(up_guess, insertion), "width_axis")
    up_axis = _unit(np.cross(insertion, width_axis), "up_axis")
    width_values = aperture @ width_axis
    height_values = aperture @ up_axis
    width_levels = _unique_levels(width_values, _WIDTH_QUANTIZE_M)
    outer_width = width_levels[np.array([0, -1])]
    outer_mask = _on_levels(width_values, outer_width, _EDGE_TOL_M)
    big_h_lo = float(height_values[outer_mask].min())
    big_h_hi = float(height_values[outer_mask].max())
    silhouette_hi = float(height_values.max())
    wide_span = float(outer_width[-1] - outer_width[0])
    if wide_span <= 1.0e-9:
        raise RuntimeError("Port aperture width is degenerate")

    if width_levels.size >= 4:
        smaller_width = width_levels[np.array([1, -2])]
    else:
        smaller_width = outer_width
    smaller_span = float(np.abs(smaller_width[-1] - smaller_width[0]))
    origin = aperture.mean(axis=0)
    mating_corners, mating_center = _aabb_corners(
        aperture[outer_mask],
        origin,
        width_axis,
        up_axis,
        insertion,
    )

    if smaller_span < _WIDE_SPAN_RATIO * wide_span:
        smaller_mask = _on_levels(width_values, smaller_width, _EDGE_TOL_M)
        latch_height_values = height_values[smaller_mask]
        latch_h_hi = float(latch_height_values.max())
        latch_h_lo = max(big_h_hi, float(latch_height_values.min()))
        latch_source = aperture[smaller_mask]
        latch_band = (latch_source @ up_axis >= latch_h_lo - _EDGE_TOL_M) & (
            latch_source @ up_axis <= latch_h_hi + _EDGE_TOL_M
        )
        if np.count_nonzero(latch_band) >= 2:
            latch_source = latch_source[latch_band]
    else:
        latch_h_lo = big_h_hi
        latch_h_hi = silhouette_hi
        latch_source = aperture[outer_mask]

    latch_corners, latch_center = _aabb_corners(
        latch_source,
        origin,
        width_axis,
        up_axis,
        insertion,
    )
    higher = latch_corners[np.argsort(latch_corners @ up_axis)[-2:]]
    higher = higher[np.argsort(higher @ width_axis)]
    return ConnectorFeatures(
        insertion_axis=insertion,
        mating_plane=Plane(point=mating_center.copy(), normal=insertion.copy()),
        mating_center=mating_center,
        mating_corners=mating_corners,
        latch_plane=Plane(point=latch_center.copy(), normal=insertion.copy()),
        latch_corners=latch_corners,
        latch_keypoints=higher,
        width_axis=width_axis,
        up_axis=up_axis,
    )


def infer_port_insertion_axis(
    *,
    inner_points: np.ndarray,
    outer_points: np.ndarray,
) -> np.ndarray:
    """Point into the jack: from the outer shell's proud face toward the cavity.

    Mesh134 (outer) extends past Mesh133 (inner) on the aisle side. Insertion
    is the opposite direction, into the switch.
    """

    inner = np.asarray(inner_points, dtype=np.float64)
    outer = np.asarray(outer_points, dtype=np.float64)
    if inner.ndim != 2 or inner.shape[1] != 3 or inner.shape[0] < 2:
        raise ValueError("inner_points must have shape (N, 3) with N >= 2")
    if outer.ndim != 2 or outer.shape[1] != 3 or outer.shape[0] < 2:
        raise ValueError("outer_points must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        raise ValueError("inner_points and outer_points must be finite")
    inner_size = inner.max(axis=0) - inner.min(axis=0)
    axis_index = int(np.argmin(inner_size))
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index] = 1.0
    proud_negative = float(inner[:, axis_index].min() - outer[:, axis_index].min())
    proud_positive = float(outer[:, axis_index].max() - inner[:, axis_index].max())
    if proud_negative >= proud_positive:
        return axis
    return -axis


def _unique_points(points: np.ndarray) -> np.ndarray:
    stacked = np.asarray(points, dtype=np.float64)
    rounded = np.round(stacked, 7)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return stacked[np.sort(unique_indices)]


def _in_plane_points(points: np.ndarray, insertion: np.ndarray) -> np.ndarray:
    projected = points - np.outer(points @ insertion, insertion)
    return projected


def _cluster_opening_faces(faces, insertion: np.ndarray, gap: float) -> list[np.ndarray]:
    if not faces:
        raise RuntimeError("No jack-opening faces on the insertion plane")
    plane_vertices = [_in_plane_points(face.vertices, insertion) for face in faces]
    parent = list(range(len(faces)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, verts_i in enumerate(plane_vertices):
        for j in range(i + 1, len(plane_vertices)):
            delta = verts_i[:, None, :] - plane_vertices[j][None, :, :]
            if float(np.linalg.norm(delta, axis=2).min()) <= gap:
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for index in range(len(faces)):
        grouped.setdefault(find(index), []).append(index)
    clusters = []
    for members in grouped.values():
        stacked = np.vstack([faces[index].vertices for index in members])
        clusters.append(_unique_points(stacked))
    return clusters


def extract_port_features_from_pack_mesh(
    *,
    points: np.ndarray,
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
    insertion_axis: np.ndarray,
) -> list[ConnectorFeatures]:
    """Split a 12-pack inner mesh into per-jack opening features."""

    mesh_points = _finite_points(points, "points")
    counts = np.asarray(face_vertex_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(face_vertex_indices, dtype=np.int64).reshape(-1)
    insertion = _unit(insertion_axis, "insertion_axis")
    facing = _collect_facing_faces(mesh_points, counts, indices, insertion)
    min_offset = min(face.offset for face in facing)
    opening = [
        face for face in facing if face.offset <= min_offset + _OPENING_PLANE_GAP_M
    ]
    clusters = _cluster_opening_faces(opening, insertion, _APERTURE_CLUSTER_GAP_M)
    ports = [
        extract_port_features_from_aperture_points(cluster, insertion)
        for cluster in clusters
    ]
    ports.sort(
        key=lambda port: (
            round(float(port.mating_center[2]), 5),
            round(float(port.mating_center[1]), 5),
        )
    )
    return ports
