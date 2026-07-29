#!/usr/bin/env python3
"""Configuration for the angled-hand horizontal-plug runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AngledHandConfig:
    """Robot-right-side convention: wrist high, fingertips down toward port."""

    hand_downward_pitch_deg: float = 30.0
    pitch_tolerance_deg: float = 0.5
    maximum_supported_pitch_deg: float = 45.0


ANGLED_HAND_CONFIG = AngledHandConfig()
