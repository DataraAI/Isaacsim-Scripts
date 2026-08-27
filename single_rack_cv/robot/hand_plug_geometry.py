#!/usr/bin/env python3
"""Pure geometry for a pitched Franka hand carrying a horizontal RJ45 plug."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_EPS = 1.0e-12


def _rotation3(value, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-9):
        raise ValueError(f"{label} must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{label} must have determinant +1")
    return matrix.copy()


def _vector3(value, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite length-3 vector")
    return vector.copy()


def _axis3(value, *, label: str) -> np.ndarray:
    axis = _vector3(value, label=label)
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        raise ValueError(f"{label} cannot be zero")
    return axis / norm


def validate_downward_hand_pitch_deg(
    value: float,
    maximum_deg: float = 45.0,
) -> float:
    pitch = float(value)
    maximum = float(maximum_deg)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_deg must be finite and positive")
    if not math.isfinite(pitch) or pitch < 0.0 or pitch > maximum:
        raise ValueError(
            f"downward hand pitch must be finite in [0, {maximum}], got {pitch}"
        )
    return pitch


def horizontal_axis_error_deg(axis_world: np.ndarray) -> float:
    axis = _axis3(axis_world, label="axis_world")
    return math.degrees(
        math.asin(float(np.clip(abs(axis[2]), 0.0, 1.0)))
    )


def _directional_axis_error_deg(
    actual_axis_world: np.ndarray,
    expected_axis_world: np.ndarray,
) -> float:
    actual = _axis3(actual_axis_world, label="actual_axis_world")
    expected = _axis3(expected_axis_world, label="expected_axis_world")
    dot = float(np.clip(np.dot(actual, expected), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def expected_camera_baseline_axis_world(
    plug_axis_world: np.ndarray,
) -> np.ndarray:
    """Return the horizontal world direction for hand-local +Y stereo baseline."""

    plug_axis = _axis3(plug_axis_world, label="plug_axis_world")
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    horizontal_plug = (
        plug_axis - float(np.dot(plug_axis, world_up)) * world_up
    )
    horizontal_norm = float(np.linalg.norm(horizontal_plug))
    if horizontal_norm <= _EPS:
        raise ValueError(
            "plug axis cannot be vertical when checking camera baseline"
        )
    horizontal_plug /= horizontal_norm
    return _axis3(
        np.cross(horizontal_plug, world_up),
        label="expected_camera_baseline_axis_world",
    )


def expected_palm_side_axis_world(plug_axis_world: np.ndarray) -> np.ndarray:
    """Backward-compatible old local-X reference; not used for stereo control."""

    return -expected_camera_baseline_axis_world(plug_axis_world)


def _rotate_axis_about_axis(
    vector: np.ndarray,
    rotation_axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate one unit vector with Rodrigues' formula."""

    vector = _axis3(vector, label="vector")
    rotation_axis = _axis3(rotation_axis, label="rotation_axis")
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    rotated = (
        cosine * vector
        + sine * np.cross(rotation_axis, vector)
        + (1.0 - cosine)
        * float(np.dot(rotation_axis, vector))
        * rotation_axis
    )
    return _axis3(rotated, label="rotated_vector")


@dataclass(frozen=True)
class AngledHandPose:
    """One hand pose and hand-to-tool transform preserving a world tool pose."""

    hand_position_world_m: np.ndarray
    hand_rotation_world: np.ndarray
    hand_from_tool_rotation: np.ndarray
    tool_position_world_m: np.ndarray
    tool_rotation_world: np.ndarray


def compute_angled_hand_pose_preserving_tool(
    *,
    base_hand_position_m: np.ndarray,
    base_hand_rotation_world: np.ndarray,
    base_hand_from_tool_rotation: np.ndarray,
    tool_position_hand_m: np.ndarray,
    downward_pitch_deg: float,
    hand_from_tool_override: np.ndarray | None = None,
) -> AngledHandPose:
    """
    Solve a downward hand pose around the exact existing world tool pose.

    The existing validated plug-tip position and orientation are treated as
    immutable. Hand-local +Y is constrained to the horizontal left/right
    stereo baseline. The hand forward axis is pitched downward around that
    baseline, and hand-local +X is rebuilt to keep a right-handed frame. The
    hand position and hand_T_tool rotation are then solved so composing them
    reconstructs the original tool pose exactly.
    """

    base_hand_position = _vector3(
        base_hand_position_m,
        label="base_hand_position_m",
    )
    base_hand_rotation = _rotation3(
        base_hand_rotation_world,
        label="base_hand_rotation_world",
    )
    if hand_from_tool_override is not None:
        override = np.asarray(hand_from_tool_override, dtype=np.float64)
        if override.shape == (4, 4):
            override_rotation = override[:3, :3]
        else:
            override_rotation = override
        base_hand_from_tool = _rotation3(
            override_rotation,
            label="hand_from_tool_override",
        )
    else:
        base_hand_from_tool = _rotation3(
            base_hand_from_tool_rotation,
            label="base_hand_from_tool_rotation",
        )
    tool_position_hand = _vector3(
        tool_position_hand_m,
        label="tool_position_hand_m",
    )
    pitch_rad = math.radians(
        validate_downward_hand_pitch_deg(downward_pitch_deg)
    )

    tool_position_world = (
        base_hand_position + base_hand_rotation @ tool_position_hand
    )
    tool_rotation_world = _rotation3(
        base_hand_rotation @ base_hand_from_tool,
        label="tool_rotation_world",
    )
    plug_axis_world = _axis3(
        tool_rotation_world[:, 2],
        label="plug_axis_world",
    )
    camera_baseline_world = expected_camera_baseline_axis_world(
        plug_axis_world
    )
    hand_forward_world = _rotate_axis_about_axis(
        plug_axis_world,
        camera_baseline_world,
        -pitch_rad,
    )
    hand_side_world = _axis3(
        np.cross(camera_baseline_world, hand_forward_world),
        label="hand_side_world",
    )
    camera_baseline_world = _axis3(
        np.cross(hand_forward_world, hand_side_world),
        label="camera_baseline_world",
    )
    hand_rotation_world = _rotation3(
        np.column_stack(
            (hand_side_world, camera_baseline_world, hand_forward_world)
        ),
        label="hand_rotation_world",
    )

    hand_from_tool_rotation = _rotation3(
        hand_rotation_world.T @ tool_rotation_world,
        label="hand_from_tool_rotation",
    )
    hand_position_world = (
        tool_position_world - hand_rotation_world @ tool_position_hand
    )

    reconstructed_tool_position = (
        hand_position_world + hand_rotation_world @ tool_position_hand
    )
    reconstructed_tool_rotation = (
        hand_rotation_world @ hand_from_tool_rotation
    )
    if not np.allclose(
        reconstructed_tool_position,
        tool_position_world,
        atol=1.0e-12,
    ):
        raise RuntimeError("angled hand position did not preserve tool position")
    if not np.allclose(
        reconstructed_tool_rotation,
        tool_rotation_world,
        atol=1.0e-12,
    ):
        raise RuntimeError("angled hand rotation did not preserve tool rotation")

    return AngledHandPose(
        hand_position_world_m=hand_position_world,
        hand_rotation_world=hand_rotation_world,
        hand_from_tool_rotation=hand_from_tool_rotation,
        tool_position_world_m=tool_position_world,
        tool_rotation_world=tool_rotation_world,
    )


def _rotation_mapping_pair(
    source_a: np.ndarray,
    source_b: np.ndarray,
    dest_a: np.ndarray,
    dest_b: np.ndarray,
) -> np.ndarray:
    """Return R in SO(3) with R@source_a≈dest_a and R@source_b≈dest_b."""

    a0 = _axis3(source_a, label="source_a")
    a1 = _axis3(source_b, label="source_b")
    b0 = _axis3(dest_a, label="dest_a")
    b1 = _axis3(dest_b, label="dest_b")
    source_angle = math.degrees(
        math.acos(float(np.clip(np.dot(a0, a1), -1.0, 1.0)))
    )
    dest_angle = math.degrees(
        math.acos(float(np.clip(np.dot(b0, b1), -1.0, 1.0)))
    )
    if abs(source_angle - dest_angle) > 1.0e-3:
        raise ValueError(
            "cannot map vector pair: angle mismatch "
            f"{source_angle:.6f} deg vs {dest_angle:.6f} deg"
        )

    f1 = a0
    f2 = _axis3(a1 - float(np.dot(a1, f1)) * f1, label="source_frame_y")
    f3 = np.cross(f1, f2)
    F = np.column_stack((f1, f2, f3))

    g1 = b0
    g2 = _axis3(b1 - float(np.dot(b1, g1)) * g1, label="dest_frame_y")
    g3 = np.cross(g1, g2)
    G = np.column_stack((g1, g2, g3))

    rotation = G @ F.T
    if float(np.linalg.det(rotation)) < 0.0:
        G = np.column_stack((g1, g2, -g3))
        rotation = G @ F.T
    return _rotation3(rotation, label="rotation_mapping_pair")


def compute_angled_hand_pose_from_fixed_grasp(
    *,
    base_hand_position_m: np.ndarray,
    base_hand_rotation_world: np.ndarray,
    hand_from_plug_frozen: np.ndarray,
    plug_from_tip: np.ndarray,
    downward_pitch_deg: float,
    pitch_tolerance_deg: float = 0.5,
) -> AngledHandPose:
    """
    Solve a downward hand pose under a physically fixed hand-to-tip relative.

    Unlike compute_angled_hand_pose_preserving_tool(), this does NOT invent a
    new hand-to-tool relative. The relative
    (hand_from_plug_frozen @ plug_from_tip) is fixed by the real grasp.

    Geometric criterion (same as the original step-3 pitch):
      hand_forward = rotate(plug_axis, about=camera_baseline(plug_axis),
                            by=-pitch)
    with plug_axis = (hand_R @ hand_from_tool_frozen)[:, 2] and
    hand_forward = hand_R[:, 2]. The world plug axis is taken from the current
    base hand composition (base_hand_R @ frozen_nose_in_hand), then hand_R is
    solved by inverting through the frozen relative so that criterion holds
    exactly. Tip world position is preserved; hand position follows.
    """

    base_hand_position = _vector3(
        base_hand_position_m,
        label="base_hand_position_m",
    )
    base_hand_rotation = _rotation3(
        base_hand_rotation_world,
        label="base_hand_rotation_world",
    )
    hand_from_plug = np.asarray(hand_from_plug_frozen, dtype=np.float64)
    tip_from_plug = np.asarray(plug_from_tip, dtype=np.float64)
    if hand_from_plug.shape == (4, 4):
        hand_from_plug_R = _rotation3(
            hand_from_plug[:3, :3],
            label="hand_from_plug_frozen",
        )
        hand_from_plug_t = _vector3(
            hand_from_plug[:3, 3],
            label="hand_from_plug_frozen_translation",
        )
    elif hand_from_plug.shape == (3, 3):
        hand_from_plug_R = _rotation3(
            hand_from_plug,
            label="hand_from_plug_frozen",
        )
        hand_from_plug_t = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError("hand_from_plug_frozen must be 3x3 or 4x4")
    if tip_from_plug.shape != (4, 4):
        raise ValueError("plug_from_tip must be a 4x4 transform")
    plug_from_tip_R = _rotation3(
        tip_from_plug[:3, :3],
        label="plug_from_tip_rotation",
    )
    tip_in_plug = _vector3(
        tip_from_plug[:3, 3],
        label="plug_from_tip_translation",
    )

    configured_pitch_deg = validate_downward_hand_pitch_deg(
        downward_pitch_deg
    )
    hand_from_tool = _rotation3(
        hand_from_plug_R @ plug_from_tip_R,
        label="hand_from_tool_frozen",
    )
    tip_in_hand = hand_from_plug_R @ tip_in_plug + hand_from_plug_t
    nose_in_hand = _axis3(
        hand_from_tool[:, 2],
        label="frozen_nose_in_hand",
    )
    hand_forward_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    inherent_pitch_rad = math.acos(
        float(np.clip(np.dot(nose_in_hand, hand_forward_local), -1.0, 1.0))
    )
    inherent_pitch_deg = math.degrees(inherent_pitch_rad)
    if abs(inherent_pitch_deg - configured_pitch_deg) > float(
        pitch_tolerance_deg
    ):
        raise RuntimeError(
            "fixed-grasp inherent hand-to-plug pitch is incompatible with "
            f"configured pitch: inherent={inherent_pitch_deg:.6f} deg, "
            f"configured={configured_pitch_deg:.6f} deg, "
            f"tolerance={float(pitch_tolerance_deg):.6f} deg"
        )

    # Lock the world plug axis to the current physical composition, then apply
    # the same pitch-about-baseline criterion the preserving-tool solver uses.
    # Use the inherent angle so the frozen relative maps exactly.
    plug_axis_world = _axis3(
        base_hand_rotation @ nose_in_hand,
        label="plug_axis_world",
    )
    camera_baseline_world = expected_camera_baseline_axis_world(
        plug_axis_world
    )
    hand_forward_world = _rotate_axis_about_axis(
        plug_axis_world,
        camera_baseline_world,
        -inherent_pitch_rad,
    )
    hand_rotation_world = _rotation_mapping_pair(
        nose_in_hand,
        hand_forward_local,
        plug_axis_world,
        hand_forward_world,
    )

    tool_position_world = base_hand_position + base_hand_rotation @ tip_in_hand
    hand_position_world = (
        tool_position_world - hand_rotation_world @ tip_in_hand
    )
    tool_rotation_world = _rotation3(
        hand_rotation_world @ hand_from_tool,
        label="tool_rotation_world",
    )

    reconstructed_nose = _axis3(
        hand_rotation_world @ nose_in_hand,
        label="reconstructed_nose",
    )
    if not np.allclose(reconstructed_nose, plug_axis_world, atol=1.0e-9):
        raise RuntimeError(
            "fixed-grasp solve did not preserve plug axis through frozen relative"
        )
    if not np.allclose(
        hand_rotation_world[:, 2],
        hand_forward_world,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "fixed-grasp solve did not achieve pitched hand forward"
        )
    if not np.allclose(
        tool_rotation_world[:, 2],
        reconstructed_nose,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "fixed-grasp tool Z does not match hand @ frozen nose"
        )

    return AngledHandPose(
        hand_position_world_m=hand_position_world,
        hand_rotation_world=hand_rotation_world,
        hand_from_tool_rotation=hand_from_tool,
        tool_position_world_m=tool_position_world,
        tool_rotation_world=tool_rotation_world,
    )


@dataclass(frozen=True)
class HandPlugGeometryMetrics:
    relative_pitch_deg: float
    wrist_above_tip_m: float
    plug_horizontal_error_deg: float
    camera_baseline_error_deg: float
    wrist_higher_fingertips_lower: bool

    @property
    def palm_roll_error_deg(self) -> float:
        """Backward-compatible alias for the retired local-X validation."""
        return self.camera_baseline_error_deg


def measure_hand_plug_geometry(
    *,
    hand_position_m: np.ndarray,
    hand_rotation_world: np.ndarray,
    plug_tip_position_m: np.ndarray,
    plug_axis_world: np.ndarray,
) -> HandPlugGeometryMetrics:
    hand_position = _vector3(hand_position_m, label="hand_position_m")
    tip_position = _vector3(plug_tip_position_m, label="plug_tip_position_m")
    hand_rotation = _rotation3(
        hand_rotation_world,
        label="hand_rotation_world",
    )
    plug_axis = _axis3(plug_axis_world, label="plug_axis_world")
    hand_forward = _axis3(
        hand_rotation[:, 2],
        label="hand_forward_axis",
    )
    camera_baseline = _axis3(
        hand_rotation[:, 1],
        label="camera_baseline_axis",
    )

    dot = float(np.clip(np.dot(hand_forward, plug_axis), -1.0, 1.0))
    relative_pitch_deg = math.degrees(math.acos(dot))
    wrist_above_tip_m = float(hand_position[2] - tip_position[2])
    direction_ok = (
        wrist_above_tip_m > 0.0
        and hand_forward[2] < plug_axis[2]
    )
    print(
        "[DEBUG] measure_hand_plug_geometry direction_ok check:\n"
        f"  wrist_above_tip_m={wrist_above_tip_m:.6f} "
        f"(pass={wrist_above_tip_m > 0.0})\n"
        f"  hand_forward={np.round(hand_forward, 6).tolist()}\n"
        f"  plug_axis={np.round(plug_axis, 6).tolist()}\n"
        f"  compare hand_forward[2] < plug_axis[2]: "
        f"{hand_forward[2]:.6f} < {plug_axis[2]:.6f} "
        f"-> {bool(hand_forward[2] < plug_axis[2])}\n"
        f"  direction_ok={direction_ok}\n"
        f"  relative_pitch_deg={relative_pitch_deg:.6f}",
        flush=True,
    )
    camera_baseline_error_deg = _directional_axis_error_deg(
        camera_baseline,
        expected_camera_baseline_axis_world(plug_axis),
    )

    return HandPlugGeometryMetrics(
        relative_pitch_deg=relative_pitch_deg,
        wrist_above_tip_m=wrist_above_tip_m,
        plug_horizontal_error_deg=horizontal_axis_error_deg(plug_axis),
        camera_baseline_error_deg=camera_baseline_error_deg,
        wrist_higher_fingertips_lower=direction_ok,
    )
