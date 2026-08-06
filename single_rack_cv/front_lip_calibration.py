#!/usr/bin/env python3
"""Production calibration for the visible RJ45 front-lip rectangle."""

from __future__ import annotations


# The live August 5 workstation run produced 91 visible front-lip width
# measurements with a 15.287 mm median and 0.251 mm population standard
# deviation. Production rounds the center of that cluster to 15.3 mm.
VISIBLE_FRONT_LIP_WIDTH_M = 0.0153
VISIBLE_FRONT_LIP_HEIGHT_M = 0.0070

# Keep the narrower localization span that produced the accurate per-eye fits
# before the visible-width calibration was introduced. This controls only how
# far from the semantic mask the RGB fitter searches for side edges. It does
# not redefine the accepted physical width of the visible opening.
VISIBLE_FRONT_LIP_SEARCH_WIDTH_M = 0.0114

LIVE_WIDTH_SAMPLE_COUNT = 91
LIVE_WIDTH_MEDIAN_M = 0.015287
LIVE_WIDTH_POPULATION_STD_M = 0.0002513259929648458


__all__ = [
    "VISIBLE_FRONT_LIP_WIDTH_M",
    "VISIBLE_FRONT_LIP_HEIGHT_M",
    "VISIBLE_FRONT_LIP_SEARCH_WIDTH_M",
    "LIVE_WIDTH_SAMPLE_COUNT",
    "LIVE_WIDTH_MEDIAN_M",
    "LIVE_WIDTH_POPULATION_STD_M",
]
