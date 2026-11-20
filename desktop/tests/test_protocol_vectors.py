"""Cross-language golden protocol vectors.

The same ``protocol/golden_vectors.txt`` file is read by the C++ driver test
and the Kotlin app test. If this test fails after an intentional wire change,
regenerate the file with ``scripts/generate_protocol_vectors.py``.
"""

from pathlib import Path

import pytest

from openvrcardboard.protocol import (
    Buttons,
    ControllerInput,
    DeviceFlags,
    DeviceId,
    DevicePose,
    TimePing,
    pack_haptic,
    pack_imu,
    pack_state,
    pack_status,
    pack_time_ping,
    unpack_imu,
    unpack_state,
    unpack_time_ping,
)

VECTORS_PATH = Path(__file__).resolve().parents[2] / "protocol" / "golden_vectors.txt"
AUTH_TOKEN = "test-token"


def _load() -> dict[str, bytes]:
    vectors: dict[str, bytes] = {}
    for line in VECTORS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, hex_bytes = line.split()
        vectors[name] = bytes.fromhex(hex_bytes)
    return vectors


VECTORS = _load()


def test_state_vector_matches_python_encoder():
    packet = pack_state(
        42,
        12.5,
        [
            DevicePose(
                device_id=int(DeviceId.HMD),
                position=(0.1, 1.7, -0.2),
                rotation=(0.0, 0.0, 0.0, 1.0),
                flags=DeviceFlags.POSE_VALID | DeviceFlags.CONNECTED,
            ),
            DevicePose(
                device_id=int(DeviceId.LEFT),
                position=(-0.3, 1.2, -0.4),
                rotation=(0.0, 0.0, 0.0, 1.0),
                flags=DeviceFlags.POSE_VALID | DeviceFlags.CONNECTED,
            ),
        ],
        [
            ControllerInput(
                device_id=int(DeviceId.LEFT),
                buttons=Buttons.TRIGGER_CLICK | Buttons.A,
                trigger=0.75,
                grip=0.25,
                joystick_x=-0.5,
                joystick_y=0.5,
            )
        ],
        prediction_ms=35.0,
    )
    assert packet == VECTORS["state"]
    state = unpack_state(VECTORS["state"])
    assert state.sequence == 42
    assert state.timestamp == pytest.approx(12.5)
    assert state.prediction_ms == pytest.approx(35.0)
    assert [pose.device_id for pose in state.devices] == [0, 1]
    assert state.devices[0].position == pytest.approx((0.1, 1.7, -0.2))
    assert state.inputs[0].buttons == Buttons.TRIGGER_CLICK | Buttons.A


def test_haptic_and_status_vectors_match_python_encoder():
    assert pack_haptic(7, int(DeviceId.LEFT), 0.8, 120.0, 0.05) == VECTORS["haptic"]
    assert (
        pack_status(8, device_count=5, uptime_ms=1000, state_packets=321, parse_errors=2)
        == VECTORS["status"]
    )


def test_imu_vectors_match_python_encoder_and_authenticator():
    packet = pack_imu(
        0x01020304,
        1234.5,
        (0.1, 0.2, 0.3, 0.9),
        (1.5, -2.5, 3.5),
        (0.0, -9.81, 0.25),
        accuracy=3,
        screen_rotation=2,
    )
    assert packet == VECTORS["imu"]
    parsed = unpack_imu(VECTORS["imu"])
    assert parsed.sequence == 0x01020304
    assert parsed.timestamp == pytest.approx(1234.5)
    assert parsed.quaternion == pytest.approx((0.1, 0.2, 0.3, 0.9))
    assert parsed.accuracy == 3
    assert parsed.screen_rotation == 2
    assert parsed.auth == b""

    authenticated = pack_imu(
        0x01020304,
        1234.5,
        (0.1, 0.2, 0.3, 0.9),
        (1.5, -2.5, 3.5),
        (0.0, -9.81, 0.25),
        accuracy=3,
        screen_rotation=2,
        token=AUTH_TOKEN,
    )
    assert authenticated == VECTORS["imu_auth"]
    assert len(authenticated) == len(VECTORS["imu"]) + 16
    assert unpack_imu(authenticated).auth == authenticated[-16:]


def test_time_ping_vectors_match_python_encoder_and_authenticator():
    assert pack_time_ping(9, 55.125) == VECTORS["time_ping"]
    assert (
        pack_time_ping(9, 55.125, token=AUTH_TOKEN) == VECTORS["time_ping_auth"]
    )
    parsed: TimePing = unpack_time_ping(VECTORS["time_ping_auth"])
    assert parsed.sequence == 9
    assert parsed.client_time == pytest.approx(55.125)
    assert parsed.auth == VECTORS["time_ping_auth"][-16:]
