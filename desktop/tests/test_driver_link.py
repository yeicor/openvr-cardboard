import socket
import threading

import numpy as np
import pytest

from openvrcardboard.driver_link import DriverLink, build_state_packet
from openvrcardboard.fusion import ComposedFrame
from openvrcardboard.mathutil import IDENTITY, from_axis_angle
from openvrcardboard.protocol import (
    Buttons,
    ControllerInput,
    DeviceFlags,
    DeviceId,
    ProtocolError,
    pack_haptic,
    pack_state,
    pack_status,
    parse_packet,
    unpack_state,
)
from openvrcardboard.types import Pose


def make_pose(position=(0.0, 0.0, 0.0), rotation=IDENTITY, **kwargs) -> Pose:
    return Pose(
        position=np.array(position, dtype=np.float64),
        rotation=np.asarray(rotation, dtype=np.float64),
        **kwargs,
    )


def test_build_state_packet_sets_flags_and_inputs():
    frame = ComposedFrame(
        hmd=make_pose(
            position=(0.0, 1.6, 0.0),
            rotation=from_axis_angle((0.0, 1.0, 0.0), 0.5),
            orientation_only=True,
        ),
        left_controller=make_pose(position=(-0.2, 1.2, -0.4)),
        trackers={100: make_pose(position=(0.0, 1.0, 0.0), valid=False)},
        inputs={
            1: ControllerInput(
                device_id=DeviceId.LEFT,
                buttons=Buttons.TRIGGER_CLICK,
                trigger=0.75,
                joystick_y=-0.5,
            )
        },
    )
    data = build_state_packet(frame, sequence=7, timestamp=12.5, prediction_ms=35.0)
    state = unpack_state(data)
    assert state.sequence == 7
    assert state.timestamp == pytest.approx(12.5)
    assert state.prediction_ms == pytest.approx(35.0)
    assert len(state.devices) == 3
    assert len(state.inputs) == 1

    hmd = state.devices[0]
    assert hmd.device_id == DeviceId.HMD
    assert int(hmd.flags) & int(DeviceFlags.POSE_VALID)
    assert int(hmd.flags) & int(DeviceFlags.CONNECTED)
    assert int(hmd.flags) & int(DeviceFlags.ORIENTATION_ONLY)

    tracker = state.devices[2]
    assert tracker.device_id == 100
    assert not (int(tracker.flags) & int(DeviceFlags.POSE_VALID))
    assert int(tracker.flags) & int(DeviceFlags.CONNECTED)

    controller_input = state.inputs[0]
    assert controller_input.trigger == pytest.approx(0.75)
    assert controller_input.joystick_y == pytest.approx(-0.5)
    assert controller_input.buttons == Buttons.TRIGGER_CLICK


def test_driver_link_sends_state_to_target():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(1.0)
    link = DriverLink(port=server.getsockname()[1])
    try:
        link.update_frame(ComposedFrame(hmd=make_pose(position=(0.0, 1.65, 0.0))), now=1.0)
        assert link.send_once(now=1.01)
        data, _ = server.recvfrom(4096)
        state = unpack_state(data)
        assert state.devices[0].device_id == DeviceId.HMD
        assert link.stats(now=1.01).send_packets == 1
    finally:
        link.stop()
        server.close()


def test_driver_link_suppresses_stale_frames():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    link = DriverLink(port=server.getsockname()[1], stale_after_s=0.5)
    try:
        link.update_frame(ComposedFrame(hmd=make_pose()), now=1.0)
        assert not link.send_once(now=2.0)
    finally:
        link.stop()
        server.close()


def test_driver_link_receives_haptics_and_status():
    haptic_event = threading.Event()
    status_event = threading.Event()
    received = {}

    def on_haptic(event):
        received["haptic"] = event
        haptic_event.set()

    def on_status(status):
        received["status"] = status
        status_event.set()

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    link = DriverLink(
        host="127.0.0.1",
        port=server.getsockname()[1],
        on_haptic=on_haptic,
        on_status=on_status,
    )
    link.start()
    try:
        link.update_frame(ComposedFrame(hmd=make_pose()), now=1.0)
        link.send_once(now=1.0)
        target = ("127.0.0.1", link.local_port)
        server.sendto(pack_haptic(3, DeviceId.LEFT, 0.8, 160.0, 0.05), target)
        server.sendto(pack_status(4, device_count=3, uptime_ms=50, state_packets=9, parse_errors=1), target)
        assert haptic_event.wait(1.0)
        assert status_event.wait(1.0)

        haptic = received["haptic"]
        assert haptic.device_id == DeviceId.LEFT
        assert haptic.amplitude == pytest.approx(0.8)
        status = received["status"]
        assert status.device_count == 3
        assert status.state_packets == 9

        stats = link.stats()
        assert stats.haptic_packets == 1
        assert stats.status_packets == 1
        assert stats.connected
    finally:
        link.stop()
        server.close()


def test_unpack_state_rejects_unknown_input_device_id():
    packet = bytearray(
        pack_state(
            1,
            1.0,
            [],
            [ControllerInput(device_id=DeviceId.LEFT, trigger=0.5)],
        )
    )
    # Input count occupies four bytes after the 12-byte header, 16-byte
    # timestamp/count block and the (empty) device array.
    input_offset = 12 + 16 + 4
    packet[input_offset] = 9
    with pytest.raises(ProtocolError, match="invalid controller input"):
        unpack_state(bytes(packet))
    with pytest.raises(ProtocolError):
        parse_packet(bytes(packet))


def test_driver_link_survives_a_malformed_state_input_id():
    status_event = threading.Event()
    received = {}

    def on_status(status):
        received["status"] = status
        status_event.set()

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    link = DriverLink(
        host="127.0.0.1",
        port=server.getsockname()[1],
        on_status=on_status,
    )
    link.start()
    try:
        link.update_frame(ComposedFrame(hmd=make_pose()), now=1.0)
        link.send_once(now=1.0)
        target = ("127.0.0.1", link.local_port)
        malformed = bytearray(
            pack_state(
                2,
                1.0,
                [],
                [ControllerInput(device_id=DeviceId.RIGHT)],
            )
        )
        malformed[12 + 16 + 4] = 42
        server.sendto(bytes(malformed), target)
        server.sendto(
            pack_status(3, device_count=1, uptime_ms=10, state_packets=1, parse_errors=0),
            target,
        )
        assert status_event.wait(1.0)
        assert received["status"].device_count == 1
        assert link.stats().parse_errors == 1
        assert link.stats().status_packets == 1
    finally:
        link.stop()
        server.close()


def test_driver_link_can_restart():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(1.0)
    link = DriverLink(port=server.getsockname()[1])
    try:
        link.start()
        link.stop()
        link.start()
        link.update_frame(ComposedFrame(hmd=make_pose()), now=1.0)
        assert link.send_once(now=1.0)
        data, _ = server.recvfrom(4096)
        assert unpack_state(data).devices
    finally:
        link.stop()
        server.close()
