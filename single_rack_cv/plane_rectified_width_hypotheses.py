#!/usr/bin/env python3
"""Select one physical front-lip fit from bounded side-search hypotheses."""

from __future__ import annotations

import math

import numpy as np

from plane_rectified_fit_utils import _mask_lower_mouth_geometry
from plane_rectified_fitting import fit_rectified_front_lip as _fit_single_search
from plane_rectified_types import FrontLipFit, MAX_EDGE_REPROJECTION_PX, RectifiedEye


_SEARCH_HYPOTHESIS_COUNT = 5
_MINIMUM_PRODUCTION_SEARCH_WIDTH_M = 0.0050
_MAXIMUM_WIDTH_PRIOR_ERROR_M = 0.0010


def _search_width_hypotheses(maximum_search_width_m: float) -> tuple[float, ...]:
    maximum = float(maximum_search_width_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Front-lip maximum search width must be positive and finite.")

    minimum = min(maximum, _MINIMUM_PRODUCTION_SEARCH_WIDTH_M)
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1.0e-12):
        return (maximum,)
    values = np.linspace(minimum, maximum, _SEARCH_HYPOTHESIS_COUNT)
    return tuple(float(value) for value in np.unique(np.round(values, 12)))


def fit_rectified_front_lip_width_prior(
    rectified: RectifiedEye,
    *,
    aperture_width_m: float = 0.0129,
    aperture_height_m: float = 0.0070,
    search_width_m: float | None = None,
    max_edge_reprojection_px: float = MAX_EDGE_REPROJECTION_PX,
) -> FrontLipFit:
    """Return the independent fit consistent with the physical mouth width.

    A single search radius is ambiguous in the live images: a narrow radius
    exposes the inner cavity, while a wide radius exposes the outer bezel.
    The unchanged per-search fitter therefore runs at five bounded radii. The
    result closest to the calibrated physical visible-mouth width is selected,
    with semantic-mask center proximity used only as a deterministic tie-break.
    """

    expected_width = float(aperture_width_m)
    if not math.isfinite(expected_width) or expected_width <= 0.0:
        raise ValueError("Front-lip aperture width must be positive and finite.")

    maximum_search = (
        expected_width if search_width_m is None else float(search_width_m)
    )
    widths = _search_width_hypotheses(maximum_search)

    broad_start, broad_end, mask_left, mask_right = _mask_lower_mouth_geometry(
        rectified.mask
    )
    mask_center_px = np.array(
        [
            0.5 * (float(mask_left) + float(mask_right)),
            0.5 * (float(broad_start) + float(broad_end)),
        ],
        dtype=np.float64,
    )
    mask_center_m = rectified.pixel_to_metric(mask_center_px)

    candidates: list[tuple[float, float, float, FrontLipFit]] = []
    failures: list[str] = []
    for width in widths:
        try:
            fit = _fit_single_search(
                rectified,
                aperture_width_m=expected_width,
                aperture_height_m=aperture_height_m,
                search_width_m=width,
                max_edge_reprojection_px=max_edge_reprojection_px,
            )
        except RuntimeError as error:
            failures.append(f"{width * 1000.0:.3f}mm:{error}")
            continue

        width_error = abs(float(fit.width_m) - expected_width)
        center_error = abs(float(fit.center_uv_m[0] - mask_center_m[0]))
        candidates.append((width_error, center_error, width, fit))

    if not candidates:
        details = "; ".join(failures[-3:])
        raise RuntimeError(
            "No front-lip side-search hypothesis produced a qualified fit"
            + (f": {details}" if details else ".")
        )

    width_error, _, selected_search, selected = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    if width_error > _MAXIMUM_WIDTH_PRIOR_ERROR_M:
        summary = ", ".join(
            f"{search * 1000.0:.3f}->{fit.width_m * 1000.0:.3f}mm"
            for _, _, search, fit in candidates
        )
        raise RuntimeError(
            "No RGB front-lip hypothesis matches the physical width prior: "
            f"target={expected_width * 1000.0:.3f}mm "
            f"best={selected.width_m * 1000.0:.3f}mm "
            f"limit={_MAXIMUM_WIDTH_PRIOR_ERROR_M * 1000.0:.3f}mm "
            f"candidates=[{summary}]"
        )

    print(
        "[RGB FRONT LIP WIDTH PRIOR] "
        f"search={selected_search * 1000.0:.3f}mm "
        f"width={selected.width_m * 1000.0:.3f}mm "
        f"target={expected_width * 1000.0:.3f}mm",
        flush=True,
    )
    return selected


__all__ = ["fit_rectified_front_lip_width_prior"]
