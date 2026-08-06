#!/usr/bin/env python3
"""Production calibration for the visible RJ45 front-lip rectangle."""

from __future__ import annotations


# The live August 5 workstation run produced 91 visible front-lip width
# measurements with a 15.287 mm median and 0.251 mm population standard
# deviation. Production rounds the center of that cluster to 15.3 mm.
VISIBLE_FRONT_LIP_WIDTH_M = 0.0153
VISIBLE_FRONT_LIP_HEIGHT_M = 0.0070

# Side-edge localization must stay close to the semantic lower-mouth wall.
# The fitter searches 45% of this span outside each mask boundary, so 5.0 mm
# gives a 2.25 mm exterior search radius. That retains the observed mask-to-lip
# under-coverage while excluding the farther outer-bezel edge that the left eye
# selected in the August 6 failed run. This does not change the accepted visible
# width, the independent-eye fit, or the 0.5 mm stereo disagreement gate.
VISIBLE_FRONT_LIP_SEARCH_WIDTH_M = 0.0050

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
