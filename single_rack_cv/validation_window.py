#!/usr/bin/env python3
"""Small stateful gate for consecutive validation samples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsecutiveValidityWindow:
    required_frames: int
    valid_frames: int = 0

    def __post_init__(self) -> None:
        self.required_frames = int(self.required_frames)
        self.valid_frames = int(self.valid_frames)
        if self.required_frames <= 0:
            raise ValueError("required_frames must be positive")
        if self.valid_frames < 0:
            raise ValueError("valid_frames must be nonnegative")
        self.valid_frames = min(self.valid_frames, self.required_frames)

    def observe(self, valid: bool) -> bool:
        if bool(valid):
            self.valid_frames = min(
                self.required_frames,
                self.valid_frames + 1,
            )
        else:
            self.valid_frames = 0
        return self.valid_frames >= self.required_frames
