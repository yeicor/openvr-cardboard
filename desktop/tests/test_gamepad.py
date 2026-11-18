import numpy as np
import pytest

from openvrcardboard.config import GamepadConfig
from openvrcardboard.gamepad import map_gamepad, normalize_trigger
from openvrcardboard.protocol import Buttons, DeviceId


def test_trigger_normalisation():
    assert normalize_trigger(0.0) == pytest.approx(0.0)
    assert normalize_trigger(1.0) == pytest.approx(1.0)
    assert normalize_trigger(-1.0) == pytest.approx(0.0)
    assert normalize_trigger(-0.5) == pytest.approx(0.25)
    assert normalize_trigger(2.0) == pytest.approx(1.0)


def test_left_pad_maps_to_left_controller():
    config = GamepadConfig(deadzone=0.1, trigger_threshold=0.5)
    item = map_gamepad(
        int(DeviceId.LEFT),
        config,
        left_stick=(0.5, -0.8),
        right_stick=(-0.4, 0.2),
        trigger=0.7,
        grip=0.1,
        pressed={"a", "dpad_up"},
    )
    assert item.device_id == DeviceId.LEFT
    assert item.trigger == pytest.approx(0.7)
    assert item.grip == pytest.approx(0.1)
    assert item.joystick_x == pytest.approx((0.5 - 0.1) / 0.9)
    # default left Y is inverted so pushing up gives +1
    assert item.joystick_y > 0.0
    assert item.trackpad_x < 0.0
    assert int(item.buttons) & int(Buttons.TRIGGER_CLICK)
    assert not int(item.buttons) & int(Buttons.GRIP_CLICK)
    assert int(item.buttons) & int(Buttons.A)
    assert int(item.buttons) & int(Buttons.DPAD_UP)
    assert int(item.buttons) & int(Buttons.TRIGGER_TOUCH)


def test_right_pad_uses_right_trigger_and_grip():
    config = GamepadConfig(deadzone=0.0, trigger_threshold=0.5)
    item = map_gamepad(
        int(DeviceId.RIGHT),
        config,
        left_stick=(0.0, 0.0),
        right_stick=(0.0, 0.0),
        trigger=0.2,
        grip=0.9,
        pressed={"b", "right_shoulder", "start"},
    )
    assert item.trigger == pytest.approx(0.2)
    assert item.grip == pytest.approx(0.9)
    assert int(item.buttons) & int(Buttons.GRIP_CLICK)
    assert not int(item.buttons) & int(Buttons.TRIGGER_CLICK)
    assert int(item.buttons) & int(Buttons.B)
    assert int(item.buttons) & int(Buttons.START)
    assert int(item.buttons) & int(Buttons.GRIP_TOUCH)


def test_deadzone_zeroes_small_stick_values():
    config = GamepadConfig(deadzone=0.2)
    item = map_gamepad(
        int(DeviceId.LEFT),
        config,
        left_stick=(0.1, -0.15),
        right_stick=(0.0, 0.0),
        trigger=0.0,
        grip=0.0,
        pressed=set(),
    )
    assert item.joystick_x == pytest.approx(0.0)
    assert item.joystick_y == pytest.approx(0.0)
    assert item.buttons == Buttons(0)


def test_synthetic_script_pulses_both_triggers_above_grab_threshold():
    from openvrcardboard.gamepad import synthetic_gamepad_state

    left = [synthetic_gamepad_state(t / 40.0, "left") for t in range(320)]
    right = [synthetic_gamepad_state(t / 40.0, "right") for t in range(320)]
    assert max(item["trigger"] for item in left) > 0.75
    assert max(item["trigger"] for item in right) > 0.75
    assert any("a" in item["pressed"] for item in left)
    assert any("grip" in item and item["grip"] > 0.0 for item in right)
    assert max(abs(item["left_stick"][0]) for item in left) > 0.3
    assert max(abs(item["right_stick"][1]) for item in right) > 0.3


def test_synthetic_gamepad_manager_publishes_inputs_and_records_haptics():
    import time

    from openvrcardboard.gamepad import GamepadManager
    from openvrcardboard.state import SharedState

    state = SharedState()
    manager = GamepadManager(state, GamepadConfig(synthetic=True), poll_hz=100)
    manager.start()
    try:
        deadline = time.monotonic() + 3.0
        snapshot = state.snapshot()
        while time.monotonic() < deadline and len(snapshot.gamepads) < 2:
            time.sleep(0.02)
            snapshot = state.snapshot()
        assert set(snapshot.gamepads) == {int(DeviceId.LEFT), int(DeviceId.RIGHT)}
        assert manager.status()["synthetic"] is True
        assert set(manager.status()["pads"]) == {"left", "right"}

        assert manager.rumble(DeviceId.LEFT, 0.6, 0.0, 0.05)
        deadline = time.monotonic() + 2.0
        status = manager.status()
        while time.monotonic() < deadline and status["haptics"]["count"] < 1:
            time.sleep(0.02)
            status = manager.status()
        haptics = status["haptics"]
        assert haptics["count"] == 1
        assert haptics["left"] == 1
        assert haptics["last"]["side"] == "left"
        assert haptics["last"]["amplitude"] == pytest.approx(0.6)
        assert haptics["last"]["duration_ms"] == 50
    finally:
        manager.stop()
    assert not manager.running
