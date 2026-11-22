"""Shared camera resolution/FPS fallback ladder.

The desktop webcam path and the Android phone-camera path use the same
ordered modes.  A camera is first asked for the requested target; if that
mode is rejected, the ladder resumes at the canonical entry closest to the
target and walks down from there.  The final "whatever the camera accepts"
attempt is backend-specific and stays in each capture implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraMode:
    """One capture mode in the shared fallback ladder."""

    width: int
    height: int
    fps: int


DEFAULT_CAMERA_MODES: tuple[CameraMode, ...] = (
    CameraMode(1920, 1080, 120),
    CameraMode(1280, 720, 120),
    CameraMode(1920, 1080, 90),
    CameraMode(1280, 720, 90),
    CameraMode(1920, 1080, 60),
    CameraMode(1280, 720, 60),
    CameraMode(1920, 1080, 30),
    CameraMode(1280, 720, 30),
)
DEFAULT_CAMERA_MODE = DEFAULT_CAMERA_MODES[0]


def _distance(mode: CameraMode, target: CameraMode) -> float:
    """Normalised Manhattan distance, so large FPS/resolution values do not win."""

    return (
        abs(mode.width - target.width) / max(target.width, 1)
        + abs(mode.height - target.height) / max(target.height, 1)
        + abs(mode.fps - target.fps) / max(target.fps, 1)
    )


def camera_mode_attempts(target: CameraMode) -> list[CameraMode]:
    """Return the target followed by the canonical fallback modes.

    The fallback starts at the ladder entry closest to the target among modes
    that do not exceed its resolution or frame rate.  That keeps a manual
    720p60 request from retrying 1080p120 while still giving the requested
    target one chance before falling back.
    """

    if target.width <= 0 or target.height <= 0 or target.fps <= 0:
        return list(DEFAULT_CAMERA_MODES)

    attempts = [target]
    eligible = [
        mode
        for mode in DEFAULT_CAMERA_MODES
        if mode.width <= target.width
        and mode.height <= target.height
        and mode.fps <= target.fps
    ]
    if not eligible:
        return attempts

    closest = min(eligible, key=lambda mode: _distance(mode, target))
    start = eligible.index(closest)
    for mode in eligible[start:]:
        if mode != target and mode not in attempts:
            attempts.append(mode)
    return attempts
