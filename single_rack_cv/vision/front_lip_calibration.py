#!/usr/bin/env python3
"""Physical visible-front-lip calibration used by production RGB fitting."""

from __future__ import annotations

# The exterior mouth boundary measured from the live RGB overlays.  This is
# intentionally distinct from CONFIG.perception.port_width_m, which remains the
# internal cavity model used by the coarse stereo detector.
VISIBLE_FRONT_LIP_WIDTH_M = 0.0129
VISIBLE_FRONT_LIP_HEIGHT_M = 0.0070

# Side-edge localization is deliberately bounded tighter than the physical
# validation width.  Widening this search to the full mouth width exposed the
# outer bezel in the left eye; narrowing it too far exposed the recessed cavity.
VISIBLE_FRONT_LIP_SEARCH_WIDTH_M = 0.0114

# Retained evidence from the rejected outer-bezel hypothesis.  These values are
# diagnostic only and must never become the production mouth-width target.
REJECTED_OUTER_BEZEL_SAMPLE_COUNT = 91
REJECTED_OUTER_BEZEL_MEDIAN_M = 0.015287
REJECTED_OUTER_BEZEL_POPULATION_STD_M = 0.0002513259929648458
