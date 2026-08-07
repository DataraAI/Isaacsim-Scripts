#!/usr/bin/env python3
"""Production calibration for the visible RJ45 front-lip rectangle."""

from __future__ import annotations


# Stable right-eye and non-bezel left-eye overlays from the August 6 failure
# measure 254-260 pixels at 0.05 mm/pixel, or 12.7-13.0 mm. Production uses
# the center of that physical-mouth cluster. The earlier 15.3 mm cluster was
# the farther outer bezel and is retained below only as rejected evidence.
VISIBLE_FRONT_LIP_WIDTH_M = 0.0129
VISIBLE_FRONT_LIP_HEIGHT_M = 0.0070

# This is now the maximum side-edge search span, not the selected geometry.
# The width-prior wrapper evaluates five bounded radii from 5.0 to 11.4 mm and
# accepts only a fit within 1.0 mm of the 12.9 mm physical-mouth width.
VISIBLE_FRONT_LIP_SEARCH_WIDTH_M = 0.0114

REJECTED_OUTER_BEZEL_SAMPLE_COUNT = 91
REJECTED_OUTER_BEZEL_MEDIAN_M = 0.015287
REJECTED_OUTER_BEZEL_POPULATION_STD_M = 0.0002513259929648458


__all__ = [
    "VISIBLE_FRONT_LIP_WIDTH_M",
    "VISIBLE_FRONT_LIP_HEIGHT_M",
    "VISIBLE_FRONT_LIP_SEARCH_WIDTH_M",
    "REJECTED_OUTER_BEZEL_SAMPLE_COUNT",
    "REJECTED_OUTER_BEZEL_MEDIAN_M",
    "REJECTED_OUTER_BEZEL_POPULATION_STD_M",
]
