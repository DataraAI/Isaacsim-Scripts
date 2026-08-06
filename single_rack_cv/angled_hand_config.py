#!/usr/bin/env python3
"""Configuration for the angled-hand horizontal-plug runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AngledHandConfig:
    """Robot-right-side convention matching the previous working hand pose."""

    hand_downward_pitch_deg: float = 30.0
    pitch_tolerance_deg: float = 0.5
    palm_side_tolerance_deg: float = 1.0
    maximum_supported_pitch_deg: float = 45.0

    # Measured rigid RJ45 body length from the loaded cable asset. With a
    # horizontal plug and a pitched hand, this derives the local shift needed
    # to place the connector rear/grip section on the finger centerline.
    plug_body_length_m: float = 0.036152


ANGLED_HAND_CONFIG = AngledHandConfig()
