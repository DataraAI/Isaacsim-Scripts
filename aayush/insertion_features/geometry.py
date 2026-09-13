"""Pure-numpy RJ45 crystal-head feature extraction.

The mating plane is the housing front that enters an Ethernet port, not the
proud latch tab and not the 3D mesh centroid. Latch keypoints are the two
higher corners of the latch face that is parallel to that mating plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FACING_DOT_MIN = 0.85
_CLUSTER_GAP_M = 0.00030
_WIDE_FACE_RATIO = 0.80
_NARROW_FACE_RATIO = 0.75
_MIN_FACE_AREA_M2 = 1.0e-10


@dataclass(frozen=True)
class Plane:
    """A world-space plane. ``normal`` points along the insertion axis."""

    point: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class ConnectorFeatures:
    """Insertion features for one crystal head, in the same units as ``points``."""

    insertion_axis: np.ndarray
    mating_plane: Plane
    mating_center: np.ndarray
    mating_corners: np.ndarray
    latch_plane: Plane
    latch_corners: np.ndarray
    latch_keypoints: np.ndarray
    width_axis: np.ndarray
    up_axis: np.ndarray


@dataclass(frozen=True)
class _Face:
    centroid: np.ndarray
    normal: np.ndarray
    area: float
    vertices: np.ndarray
    offset: float


@dataclass(frozen=True)
class _Cluster:
    faces: tuple[_Face, ...]
    points: np.ndarray
    centroid: np.ndarray
    offset: float
    area: float
    span_major: float
    span_minor: float
    major_axis: np.ndarray
    minor_axis: np.ndarray


def _finite_points(points: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 4:
        raise ValueError(f"{label} must have shape (N, 3) with N >= 4")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be finite")
    return values


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError(f"{label} must be nonzero")
    return value / norm


def _iter_faces(
    points: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
):
    cursor = 0
    for raw_count in counts:
        count = int(raw_count)
        if count < 3:
            raise ValueError("faces must have at least 3 vertices")
        face = indices[cursor : cursor + count]
        cursor += count
        yield points[face]


def _face_normal_and_area(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    if vertices.shape[0] == 3:
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    else:
        centered = vertices - vertices.mean(axis=0)
        normal = np.zeros(3, dtype=np.float64)
        for index in range(vertices.shape[0]):
            previous = centered[index - 1]
            current = centered[index]
            normal += np.cross(previous, current)
    area = 0.5 * float(np.linalg.norm(normal))
    if area <= _MIN_FACE_AREA_M2:
        return np.zeros(3, dtype=np.float64), 0.0
    return normal / (2.0 * area), area


def _in_plane_axes(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    projected = centered - np.outer(centered @ normal, normal)
    _, _, vt = np.linalg.svd(projected, full_matrices=False)
    axes: list[np.ndarray] = []
    for row in vt:
        axis = row - float(np.dot(row, normal)) * normal
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            continue
        axes.append(axis / norm)
        if len(axes) == 2:
            break
    if len(axes) < 2:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, normal))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        major = _unit(np.cross(normal, fallback), "major_axis")
        minor = _unit(np.cross(major, normal), "minor_axis")
        return major, minor
    return axes[0], axes[1]


def _span_on_axis(points: np.ndarray, axis: np.ndarray) -> float:
    projection = points @ axis
    return float(projection.max() - projection.min())


def _collect_facing_faces(
    points: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
    insertion_axis: np.ndarray,
) -> list[_Face]:
    facing: list[_Face] = []
    for vertices in _iter_faces(points, counts, indices):
        normal, area = _face_normal_and_area(vertices)
        if area <= _MIN_FACE_AREA_M2:
            continue
        if float(np.dot(normal, insertion_axis)) < _FACING_DOT_MIN:
            continue
        centroid = vertices.mean(axis=0)
        facing.append(
            _Face(
                centroid=centroid,
                normal=normal,
                area=area,
                vertices=np.asarray(vertices, dtype=np.float64),
                offset=float(np.dot(centroid, insertion_axis)),
            )
        )
    if not facing:
        raise RuntimeError("No connector faces point along the insertion axis")
    return facing


def _cluster_faces(faces: list[_Face], insertion_axis: np.ndarray) -> list[_Cluster]:
    ordered = sorted(faces, key=lambda face: face.offset)
    groups: list[list[_Face]] = [[ordered[0]]]
    for face in ordered[1:]:
        if face.offset - groups[-1][-1].offset <= _CLUSTER_GAP_M:
            groups[-1].append(face)
        else:
            groups.append([face])

    clusters: list[_Cluster] = []
    for group in groups:
        stacked = np.vstack([face.vertices for face in group])
        rounded = np.round(stacked, 7)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        points = stacked[np.sort(unique_indices)]
        major, minor = _in_plane_axes(points, insertion_axis)
        clusters.append(
            _Cluster(
                faces=tuple(group),
                points=points,
                centroid=points.mean(axis=0),
                offset=float(np.mean([face.offset for face in group])),
                area=float(sum(face.area for face in group)),
                span_major=_span_on_axis(points, major),
                span_minor=_span_on_axis(points, minor),
                major_axis=major,
                minor_axis=minor,
            )
        )
    return clusters


def _select_mating_and_latch(clusters: list[_Cluster]) -> tuple[_Cluster, _Cluster]:
    max_width = max(cluster.span_major for cluster in clusters)
    if max_width <= 1.0e-9:
        raise RuntimeError("Connector face clusters are degenerate")
    wide = [cluster for cluster in clusters if cluster.span_major >= _WIDE_FACE_RATIO * max_width]
    narrow = [
        cluster for cluster in clusters if cluster.span_major <= _NARROW_FACE_RATIO * max_width
    ]
    if not wide:
        raise RuntimeError("Could not find a full-width connector mating face")
    if not narrow:
        raise RuntimeError("Could not find a latch face parallel to the mating plane")
    mating = max(wide, key=lambda cluster: (cluster.offset, cluster.area))
    latch = max(narrow, key=lambda cluster: (cluster.offset, cluster.area))
    if latch.offset + 1.0e-9 < mating.offset:
        raise RuntimeError("Latch face is behind the connector mating plane")
    return mating, latch


def _oriented_frame(
    insertion_axis: np.ndarray,
    mating: _Cluster,
    latch: _Cluster,
) -> tuple[np.ndarray, np.ndarray]:
    width_guess, height_guess = mating.major_axis, mating.minor_axis
    latch_delta = latch.centroid - mating.centroid
    if abs(float(np.dot(latch_delta, height_guess))) >= abs(
        float(np.dot(latch_delta, width_guess))
    ):
        up = height_guess
    else:
        up = width_guess
    if float(np.dot(latch_delta, up)) < 0.0:
        up = -up
    up = _unit(up - float(np.dot(up, insertion_axis)) * insertion_axis, "up_axis")
    width = _unit(np.cross(up, insertion_axis), "width_axis")
    up = _unit(np.cross(insertion_axis, width), "up_axis")
    return width, up


def _aabb_corners(
    points: np.ndarray,
    origin: np.ndarray,
    width_axis: np.ndarray,
    up_axis: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    del normal
    width_values = points @ width_axis
    up_values = points @ up_axis
    width_center = float(np.dot(origin, width_axis))
    up_center = float(np.dot(origin, up_axis))
    corners = np.array(
        [
            origin
            + (float(width_values.min()) - width_center) * width_axis
            + (float(up_values.min()) - up_center) * up_axis,
            origin
            + (float(width_values.max()) - width_center) * width_axis
            + (float(up_values.min()) - up_center) * up_axis,
            origin
            + (float(width_values.max()) - width_center) * width_axis
            + (float(up_values.max()) - up_center) * up_axis,
            origin
            + (float(width_values.min()) - width_center) * width_axis
            + (float(up_values.max()) - up_center) * up_axis,
        ],
        dtype=np.float64,
    )
    center = 0.25 * corners.sum(axis=0)
    return corners, center


def _coarse_insertion_axis(points: np.ndarray, cable_center: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    dimensions = maximum - minimum
    aabb_axis = np.zeros(3, dtype=np.float64)
    aabb_axis[int(np.argmax(dimensions))] = 1.0
    head_center = 0.5 * (minimum + maximum)
    from_cable = _unit(head_center - cable_center, "head_from_cable")
    if float(np.dot(from_cable, aabb_axis)) < 0.0:
        aabb_axis = -aabb_axis
    if abs(float(np.dot(from_cable, aabb_axis))) < 0.85:
        return from_cable
    return aabb_axis


def _mean_normal(faces: tuple[_Face, ...], fallback: np.ndarray) -> np.ndarray:
    weighted = np.zeros(3, dtype=np.float64)
    total = 0.0
    for face in faces:
        weighted += face.area * face.normal
        total += face.area
    if total <= _MIN_FACE_AREA_M2:
        return fallback
    return _unit(weighted, "mean_face_normal")


def extract_connector_features_from_mesh(
    *,
    points: np.ndarray,
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
    cable_center_world: np.ndarray,
) -> ConnectorFeatures:
    """Derive mating plane, insertion axis, face center, and latch keypoints."""

    mesh_points = _finite_points(points, "points")
    counts = np.asarray(face_vertex_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(face_vertex_indices, dtype=np.int64).reshape(-1)
    if counts.size == 0 or int(np.sum(counts)) != indices.size:
        raise ValueError("face topology is invalid")
    if np.any(indices < 0) or np.any(indices >= mesh_points.shape[0]):
        raise ValueError("face indices are out of range")

    cable_center = np.asarray(cable_center_world, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(cable_center)):
        raise ValueError("cable_center_world must be finite")

    coarse_axis = _coarse_insertion_axis(mesh_points, cable_center)
    facing = _collect_facing_faces(mesh_points, counts, indices, coarse_axis)
    clusters = _cluster_faces(facing, coarse_axis)
    mating, latch = _select_mating_and_latch(clusters)
    insertion_axis = _mean_normal(mating.faces, coarse_axis)
    if float(np.dot(insertion_axis, coarse_axis)) < 0.0:
        insertion_axis = -insertion_axis
    width_axis, up_axis = _oriented_frame(insertion_axis, mating, latch)
    mating_corners, mating_center = _aabb_corners(
        mating.points,
        mating.centroid,
        width_axis,
        up_axis,
        insertion_axis,
    )
    latch_corners, latch_center = _aabb_corners(
        latch.points,
        latch.centroid,
        width_axis,
        up_axis,
        insertion_axis,
    )
    higher = latch_corners[np.argsort(latch_corners @ up_axis)[-2:]]
    higher = higher[np.argsort(higher @ width_axis)]
    return ConnectorFeatures(
        insertion_axis=insertion_axis,
        mating_plane=Plane(point=mating_center.copy(), normal=insertion_axis.copy()),
        mating_center=mating_center,
        mating_corners=mating_corners,
        latch_plane=Plane(point=latch_center.copy(), normal=insertion_axis.copy()),
        latch_corners=latch_corners,
        latch_keypoints=higher,
        width_axis=width_axis,
        up_axis=up_axis,
    )


def _as_transform(matrix: np.ndarray) -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(transform)):
        raise ValueError("transform must be finite")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9, rtol=0.0):
        raise ValueError("transform must be homogeneous")
    if abs(float(np.linalg.det(transform[:3, :3]))) <= 1.0e-12:
        raise ValueError("transform linear part is singular")
    return transform


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    single = values.ndim == 1
    if single:
        values = values.reshape(1, 3)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (3,) or (N, 3)")
    homogeneous = np.column_stack((values, np.ones(values.shape[0], dtype=np.float64)))
    out = (transform @ homogeneous.T).T[:, :3]
    return out.reshape(3) if single else out


def _transform_directions(linear: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    single = values.ndim == 1
    if single:
        values = values.reshape(1, 3)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vectors must have shape (3,) or (N, 3)")
    out = (linear @ values.T).T
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("direction transform is degenerate")
    out = out / norms
    return out.reshape(3) if single else out


def transform_connector_features(
    features: ConnectorFeatures,
    transform: np.ndarray,
) -> ConnectorFeatures:
    """Map connector features through a 4x4 column-vector affine transform.

    Use ``world_from_head`` to take head-local cached features into world, or
    ``inv(world_from_head)`` to store extracted world features in head-local.
    """

    matrix = _as_transform(transform)
    linear = matrix[:3, :3]
    insertion_axis = _transform_directions(linear, features.insertion_axis)
    width_axis = _transform_directions(linear, features.width_axis)
    up_axis = np.cross(insertion_axis, width_axis)
    up_norm = float(np.linalg.norm(up_axis))
    if up_norm <= 1.0e-12:
        up_axis = _transform_directions(linear, features.up_axis)
    else:
        up_axis = up_axis / up_norm
    width_axis = np.cross(up_axis, insertion_axis)
    width_axis = width_axis / float(np.linalg.norm(width_axis))
    mating_center = _transform_points(matrix, features.mating_center)
    latch_center = _transform_points(matrix, features.latch_plane.point)
    return ConnectorFeatures(
        insertion_axis=insertion_axis,
        mating_plane=Plane(point=mating_center.copy(), normal=insertion_axis.copy()),
        mating_center=mating_center,
        mating_corners=_transform_points(matrix, features.mating_corners),
        latch_plane=Plane(point=latch_center.copy(), normal=insertion_axis.copy()),
        latch_corners=_transform_points(matrix, features.latch_corners),
        latch_keypoints=_transform_points(matrix, features.latch_keypoints),
        width_axis=width_axis,
        up_axis=up_axis,
    )
