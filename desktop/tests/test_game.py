"""Tests for the built-in stereo demo game source."""

from __future__ import annotations

import math
import time

import numpy as np

from openvrcardboard.config import PhoneConfig
from openvrcardboard.game import GameInput, StereoGameSource
from openvrcardboard.video import create_source


def test_game_source_renders_each_stereo_layout():
    for layout, expected in (
        ("sbs", (320, 180)),
        ("top_bottom", (160, 360)),
        ("mono", (160, 180)),
    ):
        source = StereoGameSource(layout, 160, 180)
        source.open()
        image = source.read()
        assert image.size == expected
        assert image.mode == "RGB"
        source.close()


def test_game_frames_animate():
    source = StereoGameSource("sbs", 160, 180)
    source.open()
    first = source.read()
    time.sleep(0.06)
    second = source.read()
    assert first.tobytes() != second.tobytes()
    source.close()


def test_game_tracks_relative_phone_orientation():
    angle = math.radians(35.0)
    quaternion = np.array(
        [0.0, float(math.sin(angle / 2.0)), 0.0, float(math.cos(angle / 2.0))]
    )
    state = {"quaternion": np.array([0.0, 0.0, 0.0, 1.0]), "reads": 0}

    def provider():
        state["reads"] += 1
        return GameInput(orientation=state["quaternion"])

    source = StereoGameSource("mono", 160, 180, view_provider=provider)
    source.open()
    source.read()
    assert abs(source._yaw) < 1e-6
    state["quaternion"] = quaternion
    for _ in range(40):
        source.read()
    assert abs(source._yaw) > math.radians(5.0)
    source.close()


def test_game_uses_gamepad_steering_and_boost():
    source = StereoGameSource(
        "mono",
        160,
        180,
        view_provider=lambda: GameInput(
            steer_x=1.0, steer_y=1.0, trigger=0.8
        ),
    )
    source.open()
    for _ in range(30):
        source.read()
    assert source._yaw > 0.05
    assert source._pitch > 0.02
    assert source._trigger == 0.8
    source.close()


def test_create_source_selects_the_game():
    config = PhoneConfig()
    config.capture_source = "game"
    config.layout = "sbs"
    config.width = 160
    config.height = 120
    source = create_source(config, lambda: GameInput())
    try:
        assert source.name == "game"
        assert getattr(source, "stereo", False)
        assert source.read().size == (320, 120)
    finally:
        source.close()
