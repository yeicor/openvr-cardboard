import io
import socket
import struct
import sys
import time

import numpy as np
import pytest
from PIL import Image

from openvrcardboard.fusion import ComposedFrame
from openvrcardboard.openxr import (
    CAPTURE_MAGIC,
    REMOTE_HEADER,
    REMOTE_PACKET_SIZE,
    OpenXRCaptureServer,
    pack_remote_data,
)
from openvrcardboard.protocol import Buttons, ControllerInput, DeviceId
from openvrcardboard.types import Pose


def _pose(position, rotation=(0.0, 0.0, 0.0, 1.0), valid=True):
    return Pose(
        position=np.asarray(position, dtype=np.float64),
        rotation=np.asarray(rotation, dtype=np.float64),
        timestamp=time.monotonic(),
        valid=valid,
        source="markers",
    )


def test_pack_remote_data_layout_and_offsets():
    frame = ComposedFrame(
        hmd=_pose((0.2, 1.8, -0.4)),
        left_controller=_pose((-0.3, 1.2, -0.5)),
        right_controller=_pose((0.3, 1.2, -0.5), valid=False),
        inputs={
            int(DeviceId.LEFT): ControllerInput(
                device_id=DeviceId.LEFT,
                buttons=Buttons.TRIGGER_CLICK | Buttons.A,
                trigger=0.8,
                grip=0.4,
                joystick_x=0.25,
                joystick_y=-0.5,
            )
        },
    )
    packet = pack_remote_data(frame)
    assert len(packet) == REMOTE_PACKET_SIZE == 376
    assert struct.unpack_from("<Q", packet, 0)[0] == REMOTE_HEADER

    # Head centre pose: xrt quaternion (x, y, z, w) then position. Positions
    # stay in tracking (stage) space; Monado maps them into LOCAL itself.
    head = struct.unpack_from("<7f", packet, 104)
    assert head[:4] == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-6)
    assert head[4:] == pytest.approx((0.2, 1.8, -0.4), abs=1e-6)
    assert packet[132] == 0  # per-view data is left for the runtime to derive

    left = struct.unpack_from("<7f", packet, 136)
    assert left[4:] == pytest.approx((-0.3, 1.2, -0.5), abs=1e-6)
    assert struct.unpack_from("<f", packet, 136 + 72)[0] == pytest.approx(0.8)
    assert struct.unpack_from("<f", packet, 136 + 76)[0] == pytest.approx(0.4)
    assert struct.unpack_from("<2f", packet, 136 + 84) == pytest.approx((0.25, -0.5))
    assert packet[136 + 104] == 0  # hand tracking inactive
    assert packet[136 + 105] == 1  # controller active
    assert packet[136 + 108] == 1  # A click
    assert packet[136 + 112] == 1  # trigger click
    assert packet[136 + 113] == 1  # trigger touch

    right = struct.unpack_from("<7f", packet, 256)
    assert right[:4] == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-6)
    assert right[4:] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert packet[256 + 105] == 0  # invalid pose means inactive controller


def test_pack_remote_data_without_devices_is_neutral():
    packet = pack_remote_data(ComposedFrame())
    assert len(packet) == REMOTE_PACKET_SIZE
    head = struct.unpack_from("<7f", packet, 104)
    assert head[:4] == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-6)
    assert head[4:] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def _jpeg(width=64, height=32, color=(30, 120, 200)):
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return width, height, buffer.getvalue()


def _send_message(sock, kind, payload):
    sock.sendall(struct.pack("<4sB3xI", CAPTURE_MAGIC, kind, len(payload)) + payload)


def test_capture_server_receives_frames_and_events():
    haptics = []
    server = OpenXRCaptureServer(port=0, on_haptic=lambda hand, amp, dur: haptics.append((hand, amp, dur)))
    server.start()
    client = None
    try:
        client = socket.create_connection(("127.0.0.1", server.local_port), timeout=2.0)
        client.settimeout(2.0)

        _send_message(client, 2, b'{"type":"ready","width":1280,"height":720,"fps":30}')
        width, height, jpeg = _jpeg()
        payload = struct.pack("<IdII", 7, 1.5, width, height) + jpeg
        _send_message(client, 1, payload)
        _send_message(client, 2, b'{"type":"grab","hand":"right","value":0.75,"active":true}')
        _send_message(
            client,
            2,
            b'{"type":"haptic","hand":"right","amplitude":0.5,"duration_ms":-1,"frequency":0}',
        )

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            stats = server.stats()
            if stats.frames >= 1 and len(server.events()) >= 3 and server.stats().haptic_events >= 1:
                break
            time.sleep(0.02)

        stats = server.stats()
        assert stats.running
        assert stats.connected
        assert stats.frames == 1
        assert (stats.width, stats.height) == (width, height)
        assert stats.grab_right == pytest.approx(0.75)
        assert stats.grab_active_right
        assert stats.haptic_events == 1

        frame = server.latest(timeout=0.1)
        assert frame is not None
        assert frame.sequence == 7
        assert frame.jpeg[:2] == b"\xff\xd8"
        assert Image.open(io.BytesIO(frame.jpeg)).size == (width, height)

        # XR_MIN_HAPTIC_DURATION (-1 ns) becomes a concrete 25 ms pulse for
        # the synthetic gamepad.
        assert haptics == [("right", 0.5, 0.025)]
        assert server.stats().last_haptic["duration_ms"] == -1.0
    finally:
        if client is not None:
            client.close()
        server.stop()
    assert not server.running


def test_capture_server_rejects_bad_magic():
    server = OpenXRCaptureServer(port=0)
    server.start()
    client = None
    try:
        client = socket.create_connection(("127.0.0.1", server.local_port), timeout=2.0)
        client.sendall(struct.pack("<4sB3xI", b"NOPE", 1, 0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and server.stats().last_error == "":
            time.sleep(0.02)
        assert "magic" in server.stats().last_error.lower()
    finally:
        if client is not None:
            client.close()
        server.stop()


def test_unpacked_remote_packet_has_no_padding_surprises():
    packet = pack_remote_data(ComposedFrame())
    assert len(packet) == 376
    assert structure_is_aligned(packet)


def structure_is_aligned(packet):
    # Two optional views (48 bytes each), head centre pose (28 bytes), and a
    # single padded flag byte between the head and the controllers.
    return 8 + 48 * 2 + 28 + 4 + 120 * 2 == len(packet)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="the fake monado-service uses a POSIX shebang",
)
def test_monado_service_starts_and_stops_a_bundled_service(tmp_path):
    from openvrcardboard.openxr import MonadoService

    port = _free_port()
    script = tmp_path / "monado-service"
    script.write_text(
        f"""#!{sys.executable}
import socket
import time

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", {port}))
server.listen(4)
print("Config selected remote", flush=True)
print("null_compositor", flush=True)
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)

    service = MonadoService(
        service=script,
        remote_port=port,
        log_path=tmp_path / "service.log",
        startup_timeout=5.0,
    )
    try:
        service.start()
        stats = service.stats()
        assert stats.running
        assert stats.managed
        assert not stats.external
        assert stats.pid > 0
        assert "Config selected remote" in (tmp_path / "service.log").read_text()
    finally:
        service.stop()
    assert not service.running


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="the Monado service integration is Linux-only",
)
def test_monado_service_uses_an_external_service(tmp_path):
    from openvrcardboard.openxr import MonadoService

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    service = MonadoService(
        service=tmp_path / "missing-service",
        remote_port=port,
        log_path=tmp_path / "service.log",
        startup_timeout=1.0,
    )
    try:
        service.start()
        stats = service.stats()
        assert stats.running
        assert stats.external
        assert not stats.managed
        assert stats.pid == 0
    finally:
        service.stop()
    assert not service.running
    # The external listener must survive our shutdown.
    with socket.create_connection(("127.0.0.1", port), timeout=1.0):
        pass
    listener.close()


def test_monado_service_reports_a_missing_runtime(tmp_path, monkeypatch):
    from openvrcardboard import openxr as openxr_module
    from openvrcardboard.openxr import MonadoService, MonadoServiceError

    monkeypatch.setattr(openxr_module, "find_monado_service", lambda: None)
    service = MonadoService(
        remote_port=_free_port(),
        log_path=tmp_path / "service.log",
        startup_timeout=0.5,
    )
    with pytest.raises(MonadoServiceError):
        service.start()
    assert not service.running
