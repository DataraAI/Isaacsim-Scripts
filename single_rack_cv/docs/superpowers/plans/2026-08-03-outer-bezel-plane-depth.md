# Outer-Bezel Plane Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the recessed four-corner depth source with a vision-only dense stereo estimate of the outer white bezel/front-panel plane, then intersect the existing physical opening center with that plane.

**Architecture:** Keep the existing detector and stepped-aperture center logic for image-space Y/Z centering. Estimate depth from a wider outer-bezel stereo support region, choose the nearest qualified planar depth cluster, fit a robust camera-facing plane, and reconstruct the opening center independently in both eyes on that plane. Feed the fused point into the existing stationary three-sample qualification, frozen marker, bounded 5 mm approach, and 48-command insertion controller.

**Tech Stack:** Python 3, NumPy, OpenCV SGBM, Isaac Sim 6.0.0 camera models, `unittest`.

## Global Constraints

- Runtime must not use a rack transform, port prim, RTX ray hit, USD ground truth, or fixed world-space depth offset.
- The dark opening contour determines center geometry only; it must not determine the depth plane.
- The runtime must never fall back to `stereo_front_rim_plane` when outer-bezel support is unavailable.
- Maximum stereo center disagreement remains 0.5 mm.
- Maximum triangulation ray gap remains 0.5 mm.
- Maximum fitted-plane residual remains 0.5 mm.
- Maximum kinematic approach step remains 5 mm.
- Pre-insert standoff remains 50 mm.
- Maximum insertion lateral drift remains 0.5 mm.
- Maximum insertion orientation error remains 1.0 degree.
- The insertion sequence remains 48 commands with approximately +10 mm final depth.

---

## File Structure

- Modify `single_rack_cv/front_plane.py`: add nearest-qualified-cluster selection and partial-visibility support diagnostics while preserving the existing default four-side behavior.
- Create `single_rack_cv/outer_bezel_center.py`: compose outer-plane estimation with plane-rectified aperture-center reconstruction.
- Modify `single_rack_cv/live_control.py`: apply the new result to the control observation and expose outer-bezel diagnostics.
- Modify `single_rack_cv/live_control_projective.py`: route runtime exclusively through `outer_bezel_center.py`.
- Modify `single_rack_cv/main.py`: print diagnostics that distinguish outer-plane support from cavity depth.
- Modify `single_rack_cv/tests/test_front_plane.py`: lock nearest qualified cluster and non-degenerate partial support.
- Create `single_rack_cv/tests/test_outer_bezel_center.py`: test composition, eye disagreement, and no manual offsets.
- Modify `single_rack_cv/tests/test_front_rim_plane_runtime_wiring.py`: forbid the recessed estimator in runtime wiring.
- Modify `single_rack_cv/tests/test_live_control.py`: verify outer-bezel diagnostics and observation replacement.
- Modify `single_rack_cv/README.md`: document the new depth source and fail-closed behavior.

---

### Task 1: Select the nearest spatially qualified bezel plane

**Files:**
- Modify: `single_rack_cv/front_plane.py`
- Modify: `single_rack_cv/tests/test_front_plane.py`

**Interfaces:**
- Consumes: `ranges_m: np.ndarray`, `pixels_uv: np.ndarray`, `side_labels: np.ndarray`, `FrontPlaneConfig`.
- Produces: `BezelSupportDiagnostics`, `support_diagnostics(...)`, and `select_nearest_supported_range_cluster(...)`.
- Preserves: `select_nearest_range_cluster(...)` for existing callers and tests.

- [ ] **Step 1: Write failing tests for partial but two-dimensional support**

Add these imports and tests to `tests/test_front_plane.py`:

```python
from front_plane import (
    FrontPlaneConfig,
    select_nearest_supported_range_cluster,
    support_diagnostics,
)


class OuterBezelSupportTests(unittest.TestCase):
    def test_nearer_supported_plane_beats_larger_recessed_cluster(self):
        near_uv = np.array(
            [
                [20.0 + x, 20.0] for x in range(0, 21, 4)
            ]
            + [
                [20.0, 20.0 + y] for y in range(4, 25, 4)
            ],
            dtype=np.float64,
        )
        near_ranges = np.linspace(0.1800, 0.1804, near_uv.shape[0])
        near_labels = np.array(
            [0] * 6 + [3] * 6,
            dtype=np.int64,
        )

        far_x, far_y = np.meshgrid(
            np.arange(40.0, 80.0, 4.0),
            np.arange(40.0, 72.0, 4.0),
        )
        far_uv = np.column_stack((far_x.reshape(-1), far_y.reshape(-1)))
        far_ranges = np.linspace(0.1900, 0.1908, far_uv.shape[0])
        far_labels = np.full(far_uv.shape[0], 1, dtype=np.int64)

        pixels = np.vstack((near_uv, far_uv))
        ranges = np.concatenate((near_ranges, far_ranges))
        labels = np.concatenate((near_labels, far_labels))
        cfg = FrontPlaneConfig(
            depth_cluster_tolerance_m=0.0010,
            min_cluster_points=10,
            min_points_per_side=1,
            min_supported_regions=2,
            min_support_span_u_px=16.0,
            min_support_span_v_px=16.0,
            min_support_minor_std_px=3.0,
        )

        selected, diagnostics = select_nearest_supported_range_cluster(
            ranges,
            pixels,
            labels,
            cfg,
        )

        self.assertEqual(int(np.count_nonzero(selected)), near_uv.shape[0])
        self.assertEqual(diagnostics.region_count, 2)
        self.assertGreaterEqual(diagnostics.span_u_px, 16.0)
        self.assertGreaterEqual(diagnostics.span_v_px, 16.0)

    def test_single_narrow_edge_is_rejected(self):
        pixels = np.column_stack(
            (
                np.full(24, 30.0),
                np.linspace(10.0, 70.0, 24),
            )
        )
        ranges = np.linspace(0.1800, 0.1803, 24)
        labels = np.zeros(24, dtype=np.int64)
        cfg = FrontPlaneConfig(
            depth_cluster_tolerance_m=0.0010,
            min_cluster_points=12,
            min_points_per_side=1,
            min_supported_regions=2,
            min_support_span_u_px=12.0,
            min_support_span_v_px=12.0,
            min_support_minor_std_px=3.0,
        )

        with self.assertRaisesRegex(RuntimeError, "qualified outer-bezel"):
            select_nearest_supported_range_cluster(
                ranges,
                pixels,
                labels,
                cfg,
            )
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh -m unittest -v tests.test_front_plane
```

Expected: import failure because `select_nearest_supported_range_cluster` and `support_diagnostics` do not exist.

- [ ] **Step 3: Add support configuration and diagnostics**

Extend `FrontPlaneConfig` in `front_plane.py` with defaults that preserve the current four-side behavior:

```python
    min_supported_regions: int = 4
    min_support_span_u_px: float = 6.0
    min_support_span_v_px: float = 6.0
    min_support_minor_std_px: float = 2.0
```

Add validation in `_validate_config`:

```python
    if not 1 <= cfg.min_supported_regions <= len(SIDE_NAMES):
        raise ValueError("min_supported_regions must be between 1 and 4.")
    if cfg.min_support_span_u_px <= 0.0:
        raise ValueError("min_support_span_u_px must be positive.")
    if cfg.min_support_span_v_px <= 0.0:
        raise ValueError("min_support_span_v_px must be positive.")
    if cfg.min_support_minor_std_px <= 0.0:
        raise ValueError("min_support_minor_std_px must be positive.")
```

Add the diagnostics type and pure support calculation:

```python
@dataclass(frozen=True)
class BezelSupportDiagnostics:
    region_count: int
    span_u_px: float
    span_v_px: float
    major_std_px: float
    minor_std_px: float
    side_counts: tuple[int, int, int, int]


def support_diagnostics(
    pixels_uv: np.ndarray,
    side_labels: np.ndarray,
) -> BezelSupportDiagnostics:
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(side_labels, dtype=np.int64).reshape(-1)
    if pixels.shape[0] != labels.shape[0] or pixels.shape[0] < 3:
        raise RuntimeError("Outer-bezel support needs at least three labeled pixels.")
    if not np.all(np.isfinite(pixels)):
        raise RuntimeError("Outer-bezel support pixels must be finite.")

    centered = pixels - np.mean(pixels, axis=0)
    covariance = centered.T @ centered / float(pixels.shape[0])
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    minor_std, major_std = np.sqrt(eigenvalues)
    side_counts = tuple(
        int(np.count_nonzero(labels == side_index))
        for side_index in range(len(SIDE_NAMES))
    )
    return BezelSupportDiagnostics(
        region_count=int(sum(count > 0 for count in side_counts)),
        span_u_px=float(np.ptp(pixels[:, 0])),
        span_v_px=float(np.ptp(pixels[:, 1])),
        major_std_px=float(major_std),
        minor_std_px=float(minor_std),
        side_counts=side_counts,
    )
```

- [ ] **Step 4: Implement nearest qualified cluster selection**

Add this function after `select_nearest_range_cluster`:

```python
def select_nearest_supported_range_cluster(
    ranges_m: np.ndarray,
    pixels_uv: np.ndarray,
    side_labels: np.ndarray,
    cfg: FrontPlaneConfig,
) -> tuple[np.ndarray, BezelSupportDiagnostics]:
    _validate_config(cfg)
    ranges = np.asarray(ranges_m, dtype=np.float64).reshape(-1)
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(side_labels, dtype=np.int64).reshape(-1)
    if ranges.shape[0] != pixels.shape[0] or ranges.shape[0] != labels.shape[0]:
        raise ValueError("Ranges, pixels, and side labels must have equal length.")
    if not np.all(np.isfinite(ranges)):
        raise ValueError("Outer-bezel ranges must be finite.")

    order = np.argsort(ranges)
    sorted_ranges = ranges[order]
    for start in range(sorted_ranges.size):
        end = int(
            np.searchsorted(
                sorted_ranges,
                sorted_ranges[start] + cfg.depth_cluster_tolerance_m,
                side="right",
            )
        )
        candidate_indices = order[start:end]
        if candidate_indices.size < cfg.min_cluster_points:
            continue
        diagnostics = support_diagnostics(
            pixels[candidate_indices],
            labels[candidate_indices],
        )
        qualified = (
            diagnostics.region_count >= cfg.min_supported_regions
            and diagnostics.span_u_px >= cfg.min_support_span_u_px
            and diagnostics.span_v_px >= cfg.min_support_span_v_px
            and diagnostics.minor_std_px >= cfg.min_support_minor_std_px
        )
        if not qualified:
            continue
        selected = np.zeros(ranges.shape[0], dtype=bool)
        selected[candidate_indices] = True
        return selected, diagnostics

    raise RuntimeError(
        "No qualified outer-bezel depth cluster had enough spatially "
        "distributed support."
    )
```

- [ ] **Step 5: Route `estimate_front_plane` through the new selector**

Replace the existing `select_nearest_range_cluster(...)` call with:

```python
    cluster, support = select_nearest_supported_range_cluster(
        depth_array,
        ring_uv[: depth_array.shape[0]],
        label_array,
        cfg,
    )
```

Do not use `ring_uv[: depth_array.shape[0]]` unless `ring_uv` has first been filtered in lockstep with successful triangulations. Add `used_ring_uv` beside `points`, append the exact `left_uv` whenever a point is accepted, then pass `np.vstack(used_ring_uv)` here:

```python
    used_ring_uv: list[np.ndarray] = []
```

Inside the successful triangulation branch:

```python
        used_ring_uv.append(np.asarray(left_uv, dtype=np.float64))
```

After the loop:

```python
    used_ring_uv_array = np.vstack(used_ring_uv)
```

Then call:

```python
    cluster, support = select_nearest_supported_range_cluster(
        depth_array,
        used_ring_uv_array,
        label_array,
        cfg,
    )
```

Replace both hard `min(side_counts)` and `min(final_side_counts)` checks with these rules:

```python
    if support.region_count < cfg.min_supported_regions:
        raise RuntimeError("Outer-bezel cluster lost required visible regions.")
```

After plane inlier trimming, recompute support using the inlier pixels and labels and reject if any configured span, region-count, or minor-axis gate is lost.

- [ ] **Step 6: Run focused tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_front_plane
```

Expected: all front-plane tests pass, including the new nearer-plane and narrow-edge tests.

- [ ] **Step 7: Commit Task 1**

```bash
git add single_rack_cv/front_plane.py single_rack_cv/tests/test_front_plane.py
git commit -m "feat: select nearest qualified outer bezel plane"
```

---

### Task 2: Compose the outer plane with the physical aperture center

**Files:**
- Create: `single_rack_cv/outer_bezel_center.py`
- Create: `single_rack_cv/tests/test_outer_bezel_center.py`

**Interfaces:**
- Consumes: synchronized RGB images, masks, detection boxes and centers, camera models, physical aperture dimensions.
- Produces: `OuterBezelApertureResult` and `estimate_outer_bezel_aperture_center(...)`.
- Calls: `front_plane.estimate_front_plane(...)` and `aperture_center.estimate_planar_aperture_center(...)`.

- [ ] **Step 1: Write failing composition tests**

Create `tests/test_outer_bezel_center.py`:

```python
from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from aperture_center import PlanarApertureCenter
from front_plane import FrontPlaneResult
from outer_bezel_center import estimate_outer_bezel_aperture_center


class DummyCamera:
    image_height_px = 120
    image_width_px = 160

    def project_world(self, point):
        point = np.asarray(point, dtype=np.float64)
        return np.array([80.0 + point[1] * 100.0, 60.0 - point[2] * 100.0])


class OuterBezelCenterTests(unittest.TestCase):
    def test_uses_outer_plane_and_plane_rectified_center(self):
        disparity = object()
        plane = FrontPlaneResult(
            center_world_m=np.array([0.65, -0.19, 1.32]),
            normal_world=np.array([-1.0, 0.0, 0.0]),
            corners_world_m=np.array(
                [
                    [0.65, -0.20, 1.31],
                    [0.65, -0.18, 1.31],
                    [0.65, -0.18, 1.33],
                    [0.65, -0.20, 1.33],
                ]
            ),
            width_m=0.02,
            height_m=0.02,
            max_ray_gap_m=0.0002,
            reprojection_rms_px=0.3,
            max_reprojection_px=0.5,
            plane_residual_m=0.0002,
            valid_disparity_count=300,
            consistent_disparity_count=200,
            ring_candidate_count=80,
            triangulated_count=64,
            cluster_count=28,
            side_support_counts=(12, 0, 8, 8),
            median_disparity_px=220.0,
            disparity=disparity,
        )
        center = PlanarApertureCenter(
            center_world_m=np.array([0.65, -0.192, 1.323]),
            left_center_world_m=np.array([0.65, -0.1921, 1.323]),
            right_center_world_m=np.array([0.65, -0.1919, 1.323]),
            left_right_disagreement_m=0.0002,
        )
        camera = DummyCamera()

        with patch("outer_bezel_center.estimate_front_plane", return_value=plane), patch(
            "outer_bezel_center.estimate_planar_aperture_center",
            return_value=center,
        ):
            result = estimate_outer_bezel_aperture_center(
                left_rgb=np.zeros((120, 160, 3), dtype=np.uint8),
                right_rgb=np.zeros((120, 160, 3), dtype=np.uint8),
                left_mask=np.zeros((120, 160), dtype=np.uint8),
                right_mask=np.zeros((120, 160), dtype=np.uint8),
                left_bbox_xywh=(40, 30, 40, 30),
                right_bbox_xywh=(20, 30, 40, 30),
                left_detection_center_uv=(60.0, 45.0),
                right_detection_center_uv=(40.0, 45.0),
                left_camera=camera,
                right_camera=camera,
                aperture_width_m=0.0114,
                aperture_height_m=0.0070,
            )

        np.testing.assert_allclose(result.center_world_m, center.center_world_m)
        np.testing.assert_allclose(result.plane_origin_world_m, plane.center_world_m)
        self.assertEqual(result.support_region_count, 3)
        self.assertAlmostEqual(result.eye_disagreement_m, 0.0002)

    def test_public_api_has_no_manual_depth_offset(self):
        parameters = inspect.signature(
            estimate_outer_bezel_aperture_center
        ).parameters
        forbidden = {
            "offset",
            "depth_offset",
            "world_offset",
            "bias",
            "port_prim",
            "rack_transform",
        }
        self.assertTrue(forbidden.isdisjoint(parameters))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
~/isaacsim/python.sh -m unittest -v tests.test_outer_bezel_center
```

Expected: import failure because `outer_bezel_center.py` does not exist.

- [ ] **Step 3: Implement `OuterBezelApertureResult`**

Create `outer_bezel_center.py` with these imports and result type:

```python
#!/usr/bin/env python3
"""Physical RJ45 center reconstructed on the dense outer-bezel plane."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from aperture_center import estimate_planar_aperture_center
from front_plane import (
    DEFAULT_FRONT_PLANE_CONFIG,
    FrontPlaneConfig,
    estimate_front_plane,
)


OUTER_BEZEL_CONFIG = replace(
    DEFAULT_FRONT_PLANE_CONFIG,
    ring_inner_offset_px=6,
    ring_outer_offset_px=36,
    depth_cluster_tolerance_m=0.0020,
    min_cluster_points=20,
    min_points_per_side=1,
    min_supported_regions=2,
    min_support_span_u_px=12.0,
    min_support_span_v_px=12.0,
    min_support_minor_std_px=3.0,
)


@dataclass(frozen=True)
class OuterBezelApertureResult:
    center_world_m: np.ndarray
    left_center_world_m: np.ndarray
    right_center_world_m: np.ndarray
    left_center_uv: np.ndarray
    right_center_uv: np.ndarray
    eye_disagreement_m: float
    plane_origin_world_m: np.ndarray
    plane_normal_world: np.ndarray
    plane_residual_m: float
    max_ray_gap_m: float
    reprojection_rms_px: float
    max_reprojection_px: float
    valid_disparity_count: int
    consistent_disparity_count: int
    ring_candidate_count: int
    triangulated_count: int
    cluster_count: int
    side_support_counts: tuple[int, int, int, int]
    support_region_count: int
```

In `__post_init__`, normalize array shapes and reject non-finite values. Do not add any offset field.

- [ ] **Step 4: Implement the estimator composition**

Add:

```python
def estimate_outer_bezel_aperture_center(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_bbox_xywh: tuple[int, int, int, int],
    right_bbox_xywh: tuple[int, int, int, int],
    left_detection_center_uv: tuple[float, float],
    right_detection_center_uv: tuple[float, float],
    left_camera,
    right_camera,
    aperture_width_m: float = 0.0114,
    aperture_height_m: float = 0.0070,
    front_plane_config: FrontPlaneConfig = OUTER_BEZEL_CONFIG,
) -> OuterBezelApertureResult:
    plane = estimate_front_plane(
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        left_bbox_xywh=left_bbox_xywh,
        left_center_uv=left_detection_center_uv,
        right_bbox_xywh=right_bbox_xywh,
        right_center_uv=right_detection_center_uv,
        left_camera=left_camera,
        right_camera=right_camera,
        cfg=front_plane_config,
    )
    center = estimate_planar_aperture_center(
        left_mask=left_mask,
        right_mask=right_mask,
        left_camera=left_camera,
        right_camera=right_camera,
        plane_origin_world_m=plane.center_world_m,
        plane_normal_world=plane.normal_world,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
    )
    left_uv = left_camera.project_world(center.left_center_world_m)
    right_uv = right_camera.project_world(center.right_center_world_m)
    support_regions = int(
        sum(count > 0 for count in plane.side_support_counts)
    )
    return OuterBezelApertureResult(
        center_world_m=center.center_world_m,
        left_center_world_m=center.left_center_world_m,
        right_center_world_m=center.right_center_world_m,
        left_center_uv=left_uv,
        right_center_uv=right_uv,
        eye_disagreement_m=center.left_right_disagreement_m,
        plane_origin_world_m=plane.center_world_m,
        plane_normal_world=plane.normal_world,
        plane_residual_m=plane.plane_residual_m,
        max_ray_gap_m=plane.max_ray_gap_m,
        reprojection_rms_px=plane.reprojection_rms_px,
        max_reprojection_px=plane.max_reprojection_px,
        valid_disparity_count=plane.valid_disparity_count,
        consistent_disparity_count=plane.consistent_disparity_count,
        ring_candidate_count=plane.ring_candidate_count,
        triangulated_count=plane.triangulated_count,
        cluster_count=plane.cluster_count,
        side_support_counts=plane.side_support_counts,
        support_region_count=support_regions,
    )
```

- [ ] **Step 5: Run composition tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_outer_bezel_center \
  tests.test_aperture_center \
  tests.test_front_plane
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  single_rack_cv/outer_bezel_center.py \
  single_rack_cv/tests/test_outer_bezel_center.py
git commit -m "feat: reconstruct port center on outer bezel plane"
```

---

### Task 3: Wire runtime exclusively to the outer-bezel estimator

**Files:**
- Modify: `single_rack_cv/live_control.py`
- Modify: `single_rack_cv/live_control_projective.py`
- Modify: `single_rack_cv/main.py`
- Modify: `single_rack_cv/tests/test_front_rim_plane_runtime_wiring.py`
- Modify: `single_rack_cv/tests/test_live_control.py`

**Interfaces:**
- Consumes: `OuterBezelApertureResult` from Task 2.
- Produces: refined `StereoPortObservation` and `LiveFrontPlaneDiagnostics` populated with outer-bezel support values.
- Runtime entry remains: `refine_live_observation(frame, observation, desired_port_virtual_camera_usd, aperture_width_m, aperture_height_m)`.

- [ ] **Step 1: Update the wiring test first**

Replace the estimator assertions in `tests/test_front_rim_plane_runtime_wiring.py` with:

```python
    def test_runtime_adapter_uses_outer_bezel_plane_estimator(self):
        source = (ROOT / "live_control_projective.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "from outer_bezel_center import "
            "estimate_outer_bezel_aperture_center",
            source,
        )
        self.assertNotIn("from stereo_front_rim_plane import", source)
        self.assertNotIn("estimate_stereo_aperture_center(", source)
```

Add a main-log assertion:

```python
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("[OUTER BEZEL PLANE]", main_source)
        self.assertNotIn(
            "automatic refined local SGBM control enabled",
            main_source,
        )
```

- [ ] **Step 2: Run wiring tests and verify they fail**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_front_rim_plane_runtime_wiring \
  tests.test_live_control
```

Expected: failure because runtime still imports `stereo_front_rim_plane`.

- [ ] **Step 3: Add an explicit outer-bezel result adapter**

Append these fields with defaults to `LiveFrontPlaneDiagnostics` in `live_control.py`:

```python
    support_region_count: int = 0
    support_span_u_px: float = 0.0
    support_span_v_px: float = 0.0
    support_minor_std_px: float = 0.0
    plane_range_m: float = 0.0
```

Add:

```python
def apply_outer_bezel_result(
    frame,
    observation,
    desired_port_virtual_camera_usd,
    outer_result,
):
    disparity_px = float(
        outer_result.left_center_uv[0]
        - outer_result.right_center_uv[0]
    )
    refined = _replace_control_center(
        frame=frame,
        observation=observation,
        desired_port_virtual_camera_usd=desired_port_virtual_camera_usd,
        center_world_m=outer_result.center_world_m,
        mean_disparity_px=disparity_px,
        reprojection_rms_px=outer_result.reprojection_rms_px,
        max_reprojection_px=outer_result.max_reprojection_px,
        max_ray_gap_m=outer_result.max_ray_gap_m,
    )
    refined = replace(
        refined,
        normal_world=np.asarray(
            outer_result.plane_normal_world,
            dtype=np.float64,
        ),
        plane_residual_m=float(outer_result.plane_residual_m),
    )

    virtual_camera = frame.virtual_camera
    plane_virtual = virtual_camera.camera_point_from_world(
        outer_result.plane_origin_world_m
    )
    plane_range_m = -float(plane_virtual[2])
    cavity_range_m = float(observation.estimated_range_m)
    opening_range_m = float(refined.estimated_range_m)
    diagnostics = LiveFrontPlaneDiagnostics(
        cavity_range_m=cavity_range_m,
        opening_range_m=opening_range_m,
        recess_depth_m=cavity_range_m - opening_range_m,
        plane_residual_m=float(outer_result.plane_residual_m),
        max_ray_gap_m=float(outer_result.max_ray_gap_m),
        valid_disparity_count=int(outer_result.valid_disparity_count),
        consistent_disparity_count=int(
            outer_result.consistent_disparity_count
        ),
        ring_candidate_count=int(outer_result.ring_candidate_count),
        triangulated_count=int(outer_result.triangulated_count),
        cluster_count=int(outer_result.cluster_count),
        side_support_counts=tuple(outer_result.side_support_counts),
        aperture_center_world_m=tuple(
            float(value) for value in refined.center_world_xyz_m
        ),
        aperture_center_disagreement_m=float(
            outer_result.eye_disagreement_m
        ),
        support_region_count=int(outer_result.support_region_count),
        plane_range_m=plane_range_m,
    )
    return refined, diagnostics
```

Do not route runtime through `apply_stereo_center_result` after this task.

- [ ] **Step 4: Replace the runtime adapter import and call**

Rewrite `live_control_projective.py` imports as:

```python
from live_control import apply_outer_bezel_result
from outer_bezel_center import estimate_outer_bezel_aperture_center
```

Inside `refine_live_observation`, call:

```python
    outer_result = estimate_outer_bezel_aperture_center(
        left_rgb=frame.left.rgb,
        right_rgb=frame.right.rgb,
        left_mask=observation.left.detection.mask,
        right_mask=observation.right.detection.mask,
        left_bbox_xywh=observation.left.detection.bbox_xywh,
        right_bbox_xywh=observation.right.detection.bbox_xywh,
        left_detection_center_uv=observation.left.detection.center_uv,
        right_detection_center_uv=observation.right.detection.center_uv,
        left_camera=frame.left.camera,
        right_camera=frame.right.camera,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
    )
    return apply_outer_bezel_result(
        frame=frame,
        observation=observation,
        desired_port_virtual_camera_usd=desired_port_virtual_camera_usd,
        outer_result=outer_result,
    )
```

Delete the old `del aperture_width_m, aperture_height_m` statement.

- [ ] **Step 5: Replace misleading runtime logging**

In `main.py`, replace the startup label with:

```python
        print(
            "[OUTER BEZEL PLANE] dense stereo outer-panel depth enabled; "
            "opening mask supplies center geometry only; no manual depth "
            "offset and no RTX/USD ground truth.",
            flush=True,
        )
```

Replace capture diagnostics with a label that reports at minimum:

```python
                    "[OUTER BEZEL PLANE] "
                    f"capture={capture_index} "
                    f"cavity_range={front_plane.cavity_range_m * 1000.0:.2f}mm "
                    f"plane_range={front_plane.plane_range_m * 1000.0:.2f}mm "
                    f"opening_range={front_plane.opening_range_m * 1000.0:.2f}mm "
                    f"center={list(np.round(front_plane.aperture_center_world_m, 6))} "
                    f"eye_pair={front_plane.aperture_center_disagreement_m * 1000.0:.3f}mm "
                    f"plane_residual={front_plane.plane_residual_m * 1000.0:.3f}mm "
                    f"ray_gap={front_plane.max_ray_gap_m * 1000.0:.3f}mm "
                    f"support_regions={front_plane.support_region_count} "
                    f"dense={front_plane.consistent_disparity_count}/"
                    f"{front_plane.valid_disparity_count} "
                    f"ring={front_plane.ring_candidate_count} "
                    f"triangulated={front_plane.triangulated_count} "
                    f"cluster={front_plane.cluster_count} "
                    f"sides={front_plane.side_support_counts}"
```

- [ ] **Step 6: Add live-control regression coverage**

In `tests/test_live_control.py`, add a result fixture with an outer plane at `x=0.650` and a cavity observation at `x=0.645`. Assert that `apply_outer_bezel_result` puts `refined.center_world_xyz_m` at `x=0.650`, reports a positive cavity-to-plane separation, preserves the outer normal, and contains no configurable offset.

- [ ] **Step 7: Run runtime-adapter tests**

Run:

```bash
~/isaacsim/python.sh -m unittest -v \
  tests.test_front_rim_plane_runtime_wiring \
  tests.test_live_control \
  tests.test_outer_bezel_center \
  tests.test_aperture_center \
  tests.test_front_plane
```

Expected: all tests pass and no runtime source import references `stereo_front_rim_plane`.

- [ ] **Step 8: Commit Task 3**

```bash
git add \
  single_rack_cv/live_control.py \
  single_rack_cv/live_control_projective.py \
  single_rack_cv/main.py \
  single_rack_cv/tests/test_front_rim_plane_runtime_wiring.py \
  single_rack_cv/tests/test_live_control.py
git commit -m "feat: use outer bezel plane for runtime port depth"
```

---

### Task 4: Document fail-closed behavior and run the complete non-simulation suite

**Files:**
- Modify: `single_rack_cv/README.md`
- Test: all applicable `single_rack_cv/tests` modules.

**Interfaces:**
- Documents the final runtime contract and workstation kill switch.
- Does not modify perception or control behavior.

- [ ] **Step 1: Update the README depth architecture**

Add a section containing these exact guarantees:

```markdown
## Outer-bezel depth source

Runtime uses the dark stepped opening only for center geometry. Depth comes
from a dense stereo fit of the nearer white outer-bezel/front-panel surface.
The controller accepts partial bezel visibility only when the stereo support
covers at least two separated regions, spans a two-dimensional image patch,
and fits a camera-facing plane within the existing 0.5 mm residual and ray-gap
gates.

If the outer panel cannot be reconstructed, the frame is rejected. Runtime does
not fall back to the recessed four-corner estimator, a fixed depth offset, a port
prim, RTX ray hits, or USD ground truth.
```

Retain the existing 48-command insertion and safety-limit documentation.

- [ ] **Step 2: Run all focused perception, handoff, and insertion tests**

Run:

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh -m unittest -v \
  tests.test_front_plane \
  tests.test_outer_bezel_center \
  tests.test_aperture_center \
  tests.test_front_rim_plane_runtime_wiring \
  tests.test_live_control \
  tests.test_stereo_handoff \
  tests.test_stereo_handoff_runtime_wiring \
  tests.test_consecutive_pose_insertion \
  tests.test_partial_insertion \
  tests.test_two_stage_insertion \
  tests.test_insertion_orientation_guard \
  tests.test_orientation_hold \
  tests.test_runtime_wiring
```

Expected: zero failures.

- [ ] **Step 3: Compile changed Python modules**

Run:

```bash
~/isaacsim/python.sh -m py_compile \
  front_plane.py \
  outer_bezel_center.py \
  aperture_center.py \
  live_control.py \
  live_control_projective.py \
  main.py
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Verify forbidden runtime dependencies are absent**

Run:

```bash
grep -R -nE \
  "stereo_front_rim_plane|depth_offset|world_offset|port_prim|RTX.*ground|USD.*ground" \
  live_control_projective.py outer_bezel_center.py
```

Expected: no matches.

- [ ] **Step 5: Commit Task 4**

```bash
git add single_rack_cv/README.md
git commit -m "docs: explain outer bezel depth and fail closed behavior"
```

---

### Task 5: Workstation Isaac Sim acceptance run

**Files:**
- No source changes unless the run exposes a specific failed gate.
- Evidence: `single_rack_cv/camera_output/run_output_latest.txt` and viewport screenshots.

**Interfaces:**
- Validates the full camera-to-motion pipeline on Isaac Sim 6.0.0.
- This task is the merge gate; unit tests alone do not prove the physical surface was selected.

- [ ] **Step 1: Pull and verify the implementation head**

```bash
cd ~/Isaacsim-Scripts
git switch feature/camera-derived-port-pose
git pull --ff-only
git status --short
git rev-parse --short HEAD
```

Expected: clean working tree and the implementation commit produced by Tasks 1–4.

- [ ] **Step 2: Run Isaac Sim**

```bash
cd ~/Isaacsim-Scripts/single_rack_cv
~/isaacsim/python.sh main.py
```

- [ ] **Step 3: Inspect both markers before motion**

Required visual result:

- `/World/EstimatedPortPoint` lies on the physical front opening plane, not on the cavity floor, rear wall, or inner shelf.
- `/World/FrozenPortPoint` appears only after three accepted stationary estimates and lies on the same physical front opening plane.
- Both markers are centered in Y and Z.

Reject the run immediately if either marker is inside the cavity. Do not compensate with a fixed X offset.

- [ ] **Step 4: Extract the decisive log lines**

```bash
grep -E \
  "OUTER BEZEL PLANE.*capture=|STATIONARY PORT POSE QUALIFIED|PORT-POSE ALIGNMENT COMPLETE|TWO-STAGE PORT ENTRY|PARTIAL INSERTION COMPLETE|PARTIAL INSERTION ABORTED|FATAL ERROR" \
  camera_output/run_output_latest.txt
```

Required perception evidence:

- `support_regions` is at least 2.
- `plane_residual` is at most 0.500 mm.
- `ray_gap` is at most 0.500 mm.
- `eye_pair` is at most 0.500 mm.
- Three stationary centers qualify with at most 1.000 mm spread.

Required motion evidence:

- Qualified kinematic approach reaches the frozen 50 mm pre-insert goal.
- All 48 insertion commands settle.
- Final depth is approximately +10 mm.
- Settled lateral drift is at most 0.500 mm.
- Orientation error is at most 1.000 degree.

- [ ] **Step 5: Apply the kill switch honestly**

Reject the outer-bezel approach if repeated stationary runs cannot reconstruct a non-degenerate outer-panel patch. At that point, choose between changing the camera viewpoint and using USD/prim geometry. Do not add a fixed depth offset and do not restore the recessed four-corner estimator.

- [ ] **Step 6: Keep PR #9 draft until the workstation gate passes**

Do not mark PR #9 ready for review until the viewport and log prove both correct physical depth and 48/48 insertion completion.
