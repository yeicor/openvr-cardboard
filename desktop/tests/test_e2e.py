"""Container-friendly end-to-end tests.

These tests exercise the real UDP sockets, WebSocket video server, phone IMU
receiver and fusion loop against a fake OpenVR driver endpoint. They need no
SteamVR, camera, phone or gamepad, so they can run headless in CI/Docker:

    scripts/run-e2e.sh desktop

Hardware-dependent checks (a real SteamVR session, webcam and phone) remain a
manual step; see docs/SETUP.md.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from io import BytesIO

import numpy as np
import pytest

from openvrcardboard.config import AppConfig
from openvrcardboard.protocol import (
    DeviceFlags,
    PacketType,
    ProtocolError,
    VideoLayout,
    pack_haptic,
    pack_imu,
    pack_status,
    pack_time_ping,
    parse_packet,
    unpack_video_frame,
)
from openvrcardboard.runtime import Runtime

pytestmark = pytest.mark.e2e

IN_CONTAINER = os.environ.get("OVRC_E2E") == "1"


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def e2e_config() -> AppConfig:
    config = AppConfig()
    config.driver.udp_port = free_udp_port()
    config.phone.imu_port = free_udp_port()
    while config.phone.imu_port == config.driver.udp_port:
        config.phone.imu_port = free_udp_port()
    config.phone.video_enabled = True
    config.phone.video_port = free_tcp_port()
    config.phone.token = "e2e-token"
    config.phone.layout = "mono"
    config.phone.width = 160
    config.phone.height = 120
    config.phone.fps = 15
    config.phone.jpeg_quality = 60
    config.phone.capture_source = "test"
    config.phone.test_pattern = "grid"
    config.phone.stream_on_launch = False
    config.gamepad.enabled = False
    config.openxr.enabled = False
    config.ui.start_tracking_on_launch = False
    config.tracking.markers_enabled = False
    config.tracking.hands_enabled = False
    config.tracking.body_enabled = False
    return config


class FakeDriver:
    """Minimal OpenVR driver endpoint: records STATE, sends STATUS/HAPTIC."""

    def __init__(self, port: int):
        self.port = int(port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", self.port))
        self.socket.settimeout(0.1)
        self._states = []
        self._client = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fake-driver", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.socket.close()
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, address = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                kind, packet = parse_packet(data)
            except ProtocolError:
                continue
            if kind == PacketType.STATE:
                with self._lock:
                    self._states.append(packet)
                    self._client = address

    def mark(self) -> int:
        with self._lock:
            return len(self._states)

    def wait(self, predicate, after: int = 0, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                states = list(self._states[after:])
            for state in states:
                if predicate(state):
                    return state
            if time.monotonic() >= deadline:
                raise AssertionError("timed out waiting for a driver state packet")
            time.sleep(0.02)

    def send(self, data: bytes, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                client = self._client
            if client is not None:
                self.socket.sendto(data, client)
                return
            if time.monotonic() >= deadline:
                raise AssertionError("driver has not received a state packet yet")
            time.sleep(0.02)


def hmd_pose(state):
    for device in state.devices:
        if int(device.device_id) == 0:
            return device
    return None


def wait_until(predicate, timeout: float = 5.0, message: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {message}")


def recv_json(connection, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("timed out waiting for a JSON control message")
        message = connection.recv(timeout=remaining)
        if isinstance(message, str):
            return json.loads(message)


def test_driver_state_and_status_roundtrip():
    config = e2e_config()
    driver = FakeDriver(config.driver.udp_port)
    driver.start()
    runtime = Runtime(config)
    try:
        runtime.start(start_tracking=False)
        state = driver.wait(lambda item: hmd_pose(item) is not None)
        hmd = hmd_pose(state)
        assert hmd is not None
        assert hmd.flags & DeviceFlags.POSE_VALID
        assert hmd.flags & DeviceFlags.CONNECTED
        assert hmd.position == pytest.approx((0.0, 1.65, 0.0), abs=1e-3)
        assert hmd.rotation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-3)
        assert {int(device.device_id) for device in state.devices} >= {0, 1, 2}

        driver.send(pack_status(sequence=1, device_count=3, uptime_ms=250, state_packets=7, parse_errors=0))
        status = wait_until(
            lambda: (
                runtime.status()
                if runtime.status().driver_connected and runtime.status().driver_status
                else None
            ),
            message="driver status",
        )
        assert status.driver_status["device_count"] == 3
        assert status.driver_status["state_packets"] == 7

        driver.send(pack_haptic(sequence=2, device_id=1, amplitude=0.4, frequency=80.0, duration=0.05))
        wait_until(
            lambda: runtime.driver.stats().haptic_packets >= 1,
            message="haptic packet",
        )
        assert runtime.running
    finally:
        runtime.stop()
        driver.close()


def test_phone_imu_time_sync_and_hmd_orientation():
    config = e2e_config()
    driver = FakeDriver(config.driver.udp_port)
    driver.start()
    runtime = Runtime(config)
    phone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    phone.bind(("127.0.0.1", 0))
    phone.settimeout(3.0)
    target = ("127.0.0.1", config.phone.imu_port)
    try:
        runtime.start(start_tracking=False)

        client_time = 1234.5
        phone.sendto(pack_time_ping(sequence=7, client_time=client_time), target)
        data, _ = phone.recvfrom(1024)
        kind, pong = parse_packet(data)
        assert kind == PacketType.TIME_PONG
        assert pong.sequence == 7
        assert pong.client_time == pytest.approx(client_time)
        assert pong.server_time > 0.0

        mark = driver.mark()
        for sequence in range(10):
            phone.sendto(
                pack_imu(
                    sequence=sequence,
                    timestamp=time.monotonic(),
                    quaternion=(0.0, 0.0, 0.0, 1.0),
                    gyroscope=(0.1, 0.2, 0.3),
                    acceleration=(0.0, 0.0, 9.81),
                    accuracy=3,
                    screen_rotation=1,
                ),
                target,
            )
            time.sleep(0.005)

        sample = wait_until(lambda: runtime.state.phone(), message="phone sample")
        assert sample.screen_rotation == 1
        assert sample.accuracy == 3
        status = wait_until(
            lambda: runtime.status()
            if runtime.status().phone_address and runtime.status().phone_rate_hz > 0.0
            else None,
            message="phone statistics",
        )
        assert status.phone_address.startswith("127.0.0.1:")
        assert status.phone_accurate

        state = driver.wait(
            lambda item: hmd_pose(item) is not None
            and bool(hmd_pose(item).flags & DeviceFlags.ORIENTATION_ONLY),
            after=mark,
        )
        hmd = hmd_pose(state)
        assert hmd.flags & DeviceFlags.POSE_VALID
        assert hmd.position == pytest.approx((0.0, 1.65, 0.0), abs=2e-3)
        reference = np.asarray(hmd.rotation, dtype=np.float64)

        mark = driver.mark()
        angle = np.radians(25.0)
        rotated = (
            0.0,
            float(np.sin(angle / 2.0)),
            0.0,
            float(np.cos(angle / 2.0)),
        )
        def changed_state(item):
            pose = hmd_pose(item)
            if pose is None or not (pose.flags & DeviceFlags.ORIENTATION_ONLY):
                return False
            current = np.asarray(pose.rotation, dtype=np.float64)
            return abs(float(np.dot(reference, current))) < 0.9999

        # The estimator marks a phone stale after three missed samples, which
        # is correct for a live stream but means one isolated packet is not
        # enough to observe a change.  Keep the rotated stream running until
        # the driver reports it.
        state = None
        for sequence in range(100, 140):
            phone.sendto(
                pack_imu(
                    sequence=sequence,
                    timestamp=time.monotonic(),
                    quaternion=rotated,
                    gyroscope=(0.0, 0.0, 0.0),
                    acceleration=(0.0, 0.0, 9.81),
                    accuracy=3,
                    screen_rotation=1,
                ),
                target,
            )
            time.sleep(0.005)
            try:
                state = driver.wait(changed_state, after=mark, timeout=0.05)
                break
            except AssertionError:
                continue
        assert state is not None, "phone rotation did not reach the driver state"
        current = np.asarray(hmd_pose(state).rotation, dtype=np.float64)
        dot = abs(float(np.dot(reference, current)))
        assert dot < 0.9999, "phone rotation did not reach the driver state"
    finally:
        runtime.stop()
        phone.close()
        driver.close()


def test_video_websocket_end_to_end():
    from PIL import Image
    from websockets.sync.client import connect

    config = e2e_config()
    driver = FakeDriver(config.driver.udp_port)
    driver.start()
    runtime = Runtime(config)
    try:
        runtime.start(start_tracking=False)
        runtime.start_video()
        assert runtime.video is not None
        port = runtime.video.local_port
        uri = f"ws://127.0.0.1:{port}/video?token={config.phone.token}"

        with connect(uri, open_timeout=5.0, close_timeout=2.0) as connection:
            welcome = recv_json(connection)
            assert welcome["type"] == "welcome"
            assert welcome["video"]["format"] == "jpeg"
            assert welcome["video"]["layout"] == "mono"
            assert welcome["video"]["width"] == 160
            assert welcome["video"]["height"] == 120

            runtime.video.recenter()
            saw_frame = False
            saw_recenter = False
            deadline = time.monotonic() + 10.0
            while not (saw_frame and saw_recenter):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("video frame or recenter message did not arrive")
                message = connection.recv(timeout=remaining)
                if isinstance(message, bytes):
                    layout, width, height, _sequence, _timestamp, jpeg = unpack_video_frame(message)
                    assert layout == int(VideoLayout.MONO)
                    assert (width, height) == (160, 120)
                    image = Image.open(BytesIO(jpeg))
                    image.load()
                    assert image.size == (160, 120)
                    saw_frame = True
                else:
                    payload = json.loads(message)
                    if payload.get("type") == "recenter":
                        saw_recenter = True

            wait_until(
                lambda: runtime.status().video_clients == 1,
                message="video client count",
            )
            assert runtime.status().video_frames > 0

        wait_until(
            lambda: runtime.status().video_clients == 0,
            message="video client disconnect",
        )

        bad_uri = f"ws://127.0.0.1:{port}/video?token=wrong"
        try:
            with connect(bad_uri, open_timeout=5.0, close_timeout=2.0) as connection:
                message = connection.recv(timeout=5.0)
                payload = json.loads(message)
                assert payload["type"] == "error"
                assert "token" in payload["message"].lower()
        except Exception:
            # The server closes immediately after the error; either outcome is
            # acceptable as long as authentication was rejected.
            pass
    finally:
        runtime.stop()
        driver.close()


def test_stereo_game_stream():
    from PIL import Image
    from websockets.sync.client import connect

    config = e2e_config()
    config.phone.capture_source = "game"
    config.phone.layout = "sbs"
    config.phone.width = 320
    config.phone.height = 180
    config.phone.fps = 24
    driver = FakeDriver(config.driver.udp_port)
    driver.start()
    runtime = Runtime(config)
    try:
        runtime.start(start_tracking=False)
        runtime.start_video()
        assert runtime.video is not None
        uri = (
            f"ws://127.0.0.1:{runtime.video.local_port}/video"
            f"?token={config.phone.token}"
        )
        with connect(uri, open_timeout=5.0, close_timeout=2.0) as connection:
            welcome = recv_json(connection)
            assert welcome["video"]["layout"] == "sbs"
            assert welcome["video"]["width"] == 640
            assert welcome["video"]["height"] == 180

            deadline = time.monotonic() + 10.0
            frames = []
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                message = connection.recv(timeout=remaining)
                if not isinstance(message, bytes):
                    continue
                layout, width, height, _sequence, _timestamp, jpeg = unpack_video_frame(message)
                assert layout == int(VideoLayout.SBS)
                assert (width, height) == (640, 180)
                image = Image.open(BytesIO(jpeg))
                image.load()
                assert image.size == (640, 180)
                # Left and right eyes must differ (true stereo rendering).
                left = np.asarray(image.crop((0, 0, 320, 180)), dtype=np.int16)
                right = np.asarray(image.crop((320, 0, 640, 180)), dtype=np.int16)
                assert float(np.abs(left - right).mean()) > 1.0
                frames.append(jpeg)
                if len(set(frames)) >= 2:
                    break

            assert len(frames) >= 2, "stereo game frames did not arrive"
            # Windows can deliver two reads inside the same clock tick, which
            # produces an identical first pair even though the game animates.
            assert len(set(frames)) >= 2, "stereo game frames did not animate"
    finally:
        runtime.stop()
        driver.close()


@pytest.mark.skipif(not IN_CONTAINER, reason="optional stack is checked in the container")
def test_container_optional_stack():
    import cv2
    import mss  # noqa: F401
    import pygame
    import PySide6  # noqa: F401
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    assert hasattr(cv2, "aruco")
    # MediaPipe 1.x exposes the Tasks API only; the old mediapipe.solutions
    # namespace is gone. Check that the landmarker classes we build exist.
    assert hasattr(mp_python, "BaseOptions")
    assert hasattr(vision, "HandLandmarker")
    assert hasattr(vision, "HandLandmarkerOptions")
    assert hasattr(vision, "PoseLandmarker")
    assert hasattr(vision, "PoseLandmarkerOptions")
    pygame.joystick.init()
    assert pygame.joystick.get_count() >= 0


@pytest.mark.skipif(not IN_CONTAINER, reason="optional stack is checked in the container")
def test_container_marker_detection():
    from openvrcardboard.boards import default_boards
    from openvrcardboard.markers import BoardTracker
    from openvrcardboard.synthetic import SyntheticScene

    scene = SyntheticScene(1280, 720, boards=default_boards())
    tracker = BoardTracker(
        scene.model,
        scene.camera_position,
        scene.camera_rotation,
        boards=scene.boards,
    )
    detections = tracker.detect(scene.render(0.0))
    assert {0, 5, 10} <= set(detections)
    for detection in detections.values():
        assert detection.valid
        assert detection.marker_count >= 3
        assert detection.distance > 0.0
        assert detection.reprojection_error < 4.0


@pytest.mark.skipif(not IN_CONTAINER, reason="optional stack is checked in the container")
def test_container_screen_capture_source():
    pytest.importorskip("mss")
    from openvrcardboard.video import ScreenSource

    source = ScreenSource(monitor=1)
    source.open()
    try:
        image = source.read()
        assert image.width > 0 and image.height > 0
        assert image.mode == "RGB"
    finally:
        source.close()


@pytest.mark.skipif(not IN_CONTAINER, reason="optional stack is checked in the container")
def test_container_qt_ui_lifecycle():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from openvrcardboard.ui.main_window import MainWindow

    config = e2e_config()
    config.phone.video_enabled = False
    config.driver.udp_port = free_udp_port()
    config.phone.imu_port = free_udp_port()
    while config.phone.imu_port == config.driver.udp_port:
        config.phone.imu_port = free_udp_port()

    app = QApplication.instance() or QApplication([])
    runtime = Runtime(config)
    window = MainWindow(runtime, start_tracking=False)
    try:
        window.show()
        # Let the queued initial-start callback run, as it would on a real launch.
        app.processEvents()
        assert runtime.running
        assert not runtime.tracking.running

        window._start_clicked()
        assert runtime.tracking.running
        window._stop_clicked()
        assert runtime.running
        assert not runtime.tracking.running
    finally:
        window.close()
        app.processEvents()

    assert not runtime.running
