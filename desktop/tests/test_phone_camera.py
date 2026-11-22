"""Phone-as-webcam protocol, receiver and camera routing tests."""

import json
import time

import numpy as np
import pytest

from openvrcardboard import phone_camera
from openvrcardboard.camera import CameraFrame, CameraRouter, CameraStats
from openvrcardboard.config import AppConfig, PhoneConfig
from openvrcardboard.phone_camera import (
    CAMERA_HEADER_SIZE,
    PhoneCameraReceiver,
    pack_camera_frame,
    parse_camera_frame,
)
from openvrcardboard.pipeline import TrackingWorker
from openvrcardboard.state import SharedState


class FakeDecoder:
    instances: list["FakeDecoder"] = []

    def __init__(
        self,
        width,
        height,
        on_frame,
        on_error=None,
        on_keyframe_request=None,
    ):
        self.width = width
        self.height = height
        self.on_frame = on_frame
        self.on_error = on_error
        self.on_keyframe_request = on_keyframe_request
        self.feeds: list[tuple[bytes, bool]] = []
        self.dropped = 0
        self.frames = 0
        self.closed = False
        self.failed = False
        FakeDecoder.instances.append(self)

    def feed(self, data: bytes, keyframe: bool) -> None:
        self.feeds.append((bytes(data), bool(keyframe)))

    def close(self) -> None:
        self.closed = True


class FakeBaseCamera:
    def __init__(self, frame: CameraFrame):
        self.frame = frame
        self.reports: list[tuple[int, int]] = []
        self.config = type("Config", (), {"source": "webcam"})()
        self.running = False
        self.starts = 0
        self.stops = 0

    def start(self) -> None:
        self.running = True
        self.starts += 1

    def stop(self) -> None:
        self.running = False
        self.stops += 1

    def latest(self):
        return self.frame if self.running else None

    def stats(self) -> CameraStats:
        return CameraStats(
            running=self.running,
            opened=self.running,
            fps=60.0,
            frame_ms=2.0,
            width=self.frame.width,
            height=self.frame.height,
            frames=10,
            source="webcam",
        )

    def report_marker_detection(self, detected: int, visible: int) -> None:
        self.reports.append((detected, visible))


class FakePhoneCamera:
    def __init__(
        self,
        frame: CameraFrame,
        fov: float = 0.0,
        camera_key: str = "",
    ):
        self.frame = frame
        self.connected = True
        self.fov = fov
        self.camera_key_value = camera_key

    def camera_key(self) -> str:
        return self.camera_key_value

    def camera_name(self) -> str:
        return "Fake phone camera"

    def frame_if_active(self, timeout=None):
        return self.frame if self.connected else None

    def horizontal_fov(self) -> float:
        return self.fov

    def stats(self) -> CameraStats:
        return CameraStats(
            running=True,
            opened=True,
            fps=120.0,
            frame_ms=3.0,
            width=self.frame.width,
            height=self.frame.height,
            frames=100,
            source="phone",
            phone_connected=True,
            phone_active=True,
            dropped_frames=4,
        )

    def connected_value(self) -> bool:
        return self.connected


def make_frame(value: int, source: str) -> CameraFrame:
    image = np.full((4, 6, 3), value, dtype=np.uint8)
    return CameraFrame(
        image=image,
        timestamp=time.monotonic(),
        sequence=1,
        width=6,
        height=4,
        source=source,
    )


def test_camera_frame_roundtrip_and_header_size():
    payload = b"\x00\x00\x00\x01\x65payload"
    packet = pack_camera_frame(7, 12.5, 1280, 720, payload, keyframe=True)
    assert len(packet) == CAMERA_HEADER_SIZE + len(payload)
    sequence, timestamp, width, height, body, keyframe = parse_camera_frame(packet)
    assert sequence == 7
    assert timestamp == pytest.approx(12.5)
    assert (width, height) == (1280, 720)
    assert body == payload
    assert keyframe


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data[:10], "shorter"),
        (lambda data: b"XVRC" + data[4:], "magic"),
        (lambda data: data[:4] + b"\x02" + data[5:], "version"),
        (lambda data: data[:5] + b"\x31" + data[6:], "type"),
        (lambda data: data[:20] + b"\x00\x00\x00\x00" + data[24:], "dimensions"),
    ],
)
def test_camera_frame_rejects_malformed_packets(mutate, message):
    packet = pack_camera_frame(1, 0.0, 1280, 720, b"data")
    with pytest.raises(ValueError, match=message):
        parse_camera_frame(mutate(packet))


def test_receiver_authenticates_and_decodes_camera_frames(monkeypatch):
    FakeDecoder.instances.clear()
    monkeypatch.setattr(phone_camera, "_H264Decoder", FakeDecoder)
    config = PhoneConfig(
        camera_enabled=True,
        camera_port=0,
        camera_bitrate_mbps=55.0,
        camera_width=1280,
        camera_height=720,
        camera_fps=120,
        token="shared-token",
    )
    receiver = PhoneCameraReceiver(config, host="127.0.0.1")
    connection = None
    try:
        receiver.start()
        port = receiver.local_port
        assert port > 0
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        with connect(
            f"ws://127.0.0.1:{port}/camera?token=wrong",
            open_timeout=5.0,
            close_timeout=2.0,
        ) as rejected:
            try:
                message = json.loads(rejected.recv(timeout=2.0))
            except ConnectionClosed:
                message = {"type": "error"}
            assert message["type"] == "error"

        connection = connect(
            f"ws://127.0.0.1:{port}/camera?token=shared-token",
            open_timeout=5.0,
            close_timeout=2.0,
            legacy=True,
        )
        welcome = json.loads(connection.recv(timeout=2.0))
        assert welcome["type"] == "camera_welcome"
        assert welcome["bitrate_mbps"] == pytest.approx(55.0)
        assert (
            welcome["prefer_width"],
            welcome["prefer_height"],
            welcome["prefer_fps"],
        ) == (1280, 720, 120)

        # A malformed hello must not tear down the session before the phone
        # retries the handshake.
        connection.send(
            json.dumps(
                {
                    "type": "camera_hello",
                    "width": "wide",
                    "height": None,
                    "fps": "fast",
                    "hfov_deg": "wide",
                }
            )
        )
        connection.send(
            json.dumps(
                {
                    "type": "camera_hello",
                    "width": 1280,
                    "height": 720,
                    "fps": 120,
                    "codec": "h264",
                    "hfov_deg": 66.5,
                    "device": "Test Phone",
                    "camera_id": "0",
                    "camera_name": "Test Phone rear camera",
                }
            )
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            status = receiver.status()
            if status["info"].get("fps") == 120:
                break
            time.sleep(0.02)
        assert status["info"]["fps"] == 120
        assert receiver.horizontal_fov() == pytest.approx(66.5)
        assert receiver.camera_id() == "0"
        assert receiver.camera_key() == "phone:Test Phone:0"
        assert receiver.camera_name() == "Test Phone rear camera"
        assert status["camera_key"] == "phone:Test Phone:0"

        packet = pack_camera_frame(
            1, 12.5, 1280, 720, b"\x00\x00\x00\x01\x65frame", keyframe=True
        )
        connection.send(packet)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not FakeDecoder.instances:
            time.sleep(0.02)
        assert FakeDecoder.instances
        decoder = FakeDecoder.instances[0]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not decoder.feeds:
            time.sleep(0.02)
        assert decoder.feeds == [(b"\x00\x00\x00\x01\x65frame", True)]
        assert receiver.connected()
        assert receiver.status()["connected"]
    finally:
        if connection is not None:
            connection.close()
        receiver.stop()


def _wait_for(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    return predicate()


def test_camera_router_releases_local_camera_and_reopens_it():
    local = make_frame(10, "webcam")
    phone = make_frame(200, "phone")
    base = FakeBaseCamera(local)
    phone_camera = FakePhoneCamera(phone)
    router = CameraRouter(base, phone_camera, timeout_s=0.5)

    try:
        router.start()
        # A phone that is already streaming must not open the local camera.
        frame = router.latest()
        assert frame is phone
        assert router.phone_active()
        assert _wait_for(lambda: not base.running)
        assert base.starts == 0

        phone_camera.connected = False
        frame = _wait_for(router.latest)
        assert frame is local
        assert not router.phone_active()
        assert _wait_for(lambda: base.starts == 1)
        assert base.running

        phone_camera.connected = True
        frame = router.latest()
        assert frame is phone
        assert _wait_for(lambda: not base.running)
        assert base.stops == 1

        copied = router.latest_copy()
        assert copied is not None and copied is not phone
        copied.image[0, 0, 0] = 99
        assert phone.image[0, 0, 0] == 200
    finally:
        router.stop()
    assert not base.running


def test_camera_router_stats_and_marker_reporting():
    local = make_frame(10, "webcam")
    phone = make_frame(200, "phone")
    base = FakeBaseCamera(local)
    phone_camera = FakePhoneCamera(phone)
    router = CameraRouter(base, phone_camera)

    try:
        router.start()
        router.latest()
        stats = router.stats()
        assert stats.source == "phone"
        assert stats.phone_active
        assert stats.dropped_frames == 4
        router.report_marker_detection(5, 10)
        assert base.reports == []

        phone_camera.connected = False
        assert _wait_for(router.latest) is local
        stats = router.stats()
        assert stats.source == "webcam"
        assert not stats.phone_active
        assert not stats.phone_connected
        router.report_marker_detection(3, 10)
        assert base.reports == [(3, 10)]
    finally:
        router.stop()


def test_receiver_replacement_keeps_new_connection_and_decoder(monkeypatch):
    FakeDecoder.instances.clear()
    monkeypatch.setattr(phone_camera, "_H264Decoder", FakeDecoder)
    config = PhoneConfig(camera_enabled=True, camera_port=0, token="")
    receiver = PhoneCameraReceiver(config, host="127.0.0.1")
    first = None
    second = None
    try:
        receiver.start()
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        url = f"ws://127.0.0.1:{receiver.local_port}/camera"
        first = connect(url, open_timeout=5.0, close_timeout=2.0, legacy=True)
        assert json.loads(first.recv(timeout=2.0))["type"] == "camera_welcome"
        first.send(json.dumps({"type": "camera_hello", "width": 1280, "height": 720, "fps": 120}))
        first.send(pack_camera_frame(1, 1.0, 1280, 720, b"\x00\x00\x01\x65one", True))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not FakeDecoder.instances:
            time.sleep(0.02)
        decoder = FakeDecoder.instances[0]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(decoder.feeds) < 1:
            time.sleep(0.02)

        second = connect(url, open_timeout=5.0, close_timeout=2.0, legacy=True)
        assert json.loads(second.recv(timeout=2.0))["type"] == "camera_welcome"
        second.send(json.dumps({"type": "camera_hello", "width": 1280, "height": 720, "fps": 90}))
        second.send(pack_camera_frame(3, 2.0, 1280, 720, b"\x00\x00\x01\x65two", True))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(decoder.feeds) < 2:
            time.sleep(0.02)
        assert [keyframe for _, keyframe in decoder.feeds] == [True, True]

        # The replaced client's handler must not tear down the new session
        # when its socket finally unwinds.
        try:
            first.recv(timeout=2.0)
        except (ConnectionClosed, TimeoutError):
            pass
        first.close()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if receiver.status()["info"].get("fps") == 90:
                break
            time.sleep(0.02)
        assert receiver.connected()
        assert not decoder.closed
        assert receiver.status()["info"]["fps"] == 90

        second.send(pack_camera_frame(4, 3.0, 1280, 720, b"\x00\x00\x01\x65three", True))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(decoder.feeds) < 3:
            time.sleep(0.02)
        assert len(decoder.feeds) == 3
    finally:
        for connection in (first, second):
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
        receiver.stop()


def test_receiver_replaces_a_failed_decoder_on_the_next_frame(monkeypatch):
    FakeDecoder.instances.clear()
    monkeypatch.setattr(phone_camera, "_H264Decoder", FakeDecoder)
    config = PhoneConfig(camera_enabled=True, camera_port=0, token="")
    receiver = PhoneCameraReceiver(config, host="127.0.0.1")
    connection = None
    try:
        receiver.start()
        from websockets.sync.client import connect

        url = f"ws://127.0.0.1:{receiver.local_port}/camera"
        connection = connect(url, open_timeout=5.0, close_timeout=2.0, legacy=True)
        assert json.loads(connection.recv(timeout=2.0))["type"] == "camera_welcome"
        connection.send(json.dumps({"type": "camera_hello", "width": 1280, "height": 720}))
        connection.send(pack_camera_frame(1, 1.0, 1280, 720, b"\x00\x00\x01\x65one", True))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not FakeDecoder.instances:
            time.sleep(0.02)
        assert len(FakeDecoder.instances) == 1
        first = FakeDecoder.instances[0]

        # A decoder thread reports the failure and must not close the decoder
        # from inside its own callbacks.
        first.failed = True
        first.on_error("ffmpeg stopped producing frames")
        # Closing here would join the decoder's own threads and deadlock.
        assert not first.closed

        connection.send(pack_camera_frame(2, 2.0, 1280, 720, b"\x00\x00\x01\x65two", True))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(FakeDecoder.instances) < 2:
            time.sleep(0.02)
        assert first.closed
        assert len(FakeDecoder.instances) == 2
        second = FakeDecoder.instances[1]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not second.feeds:
            time.sleep(0.02)
        assert second.feeds == [(b"\x00\x00\x01\x65two", True)]
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        receiver.stop()


def test_tracking_worker_uses_reported_phone_intrinsics():
    from openvrcardboard.calibration import CameraModel

    local = make_frame(10, "webcam")
    phone_frame = CameraFrame(
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        timestamp=time.monotonic(),
        sequence=1,
        width=1280,
        height=720,
        source="phone",
    )
    local_model = CameraModel.estimate(1920, 1080, 60.0)
    phone_camera = FakePhoneCamera(make_frame(200, "phone"), fov=70.0)
    worker = TrackingWorker(
        SharedState(),
        AppConfig(),
        local_model,
        [0.0, 1.5, -1.5],
        [0.0, 1.0, 0.0, 0.0],
        camera=FakeBaseCamera(local),
        phone_camera=phone_camera,
    )
    try:
        phone_model = worker._camera_model(phone_frame)
        assert (phone_model.width, phone_model.height) == (1280, 720)
        assert phone_model.fx == pytest.approx(
            (1280 * 0.5) / np.tan(np.radians(70.0) * 0.5)
        )
        assert worker._camera_model(local) is local_model
        assert worker._camera_model(phone_frame) is phone_model
    finally:
        worker.stop()


def test_tracking_worker_prefers_stored_phone_calibration(tmp_path):
    from openvrcardboard.calibration import (
        CameraCalibrationStore,
        CameraModel,
    )

    local = make_frame(10, "webcam")
    phone_frame = CameraFrame(
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        timestamp=time.monotonic(),
        sequence=1,
        width=1280,
        height=720,
        source="phone",
    )
    store = CameraCalibrationStore(tmp_path / "phone_calibrations.json")
    store.save(
        "phone:Test Phone:0",
        CameraModel(1280, 720, 1010.0, 1020.0, 650.0, 370.0),
        "Test Phone rear camera",
    )
    phone_camera = FakePhoneCamera(
        make_frame(200, "phone"),
        fov=70.0,
        camera_key="phone:Test Phone:0",
    )
    worker = TrackingWorker(
        SharedState(),
        AppConfig(),
        CameraModel.estimate(1920, 1080, 60.0),
        [0.0, 1.5, -1.5],
        [0.0, 1.0, 0.0, 0.0],
        camera=FakeBaseCamera(local),
        phone_camera=phone_camera,
        phone_calibrations=store,
    )
    try:
        model = worker._camera_model(phone_frame)
        assert model.fx == pytest.approx(1010.0)
        assert model.fy == pytest.approx(1020.0)
        assert model.cx == pytest.approx(650.0)
        assert model.cy == pytest.approx(370.0)
    finally:
        worker.stop()


def test_h264_decoder_decodes_an_annex_b_stream():
    import subprocess

    from openvrcardboard.gpu import find_ffmpeg

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")
    try:
        encoded = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=0.2:size=160x90:rate=10",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-x264-params",
                "keyint=1",
                "-f",
                "h264",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pytest.skip("this ffmpeg build cannot encode a test H.264 stream")
    if not encoded:
        pytest.skip("ffmpeg produced an empty test stream")

    frames: list[np.ndarray] = []
    errors: list[str] = []
    decoder = phone_camera._H264Decoder(
        160,
        90,
        lambda image, _elapsed: frames.append(image.copy()),
        on_error=errors.append,
    )
    try:
        # A bare stream ends after its last NAL, so the H.264 parser cannot
        # know the final access unit is complete until the next start code.
        # Live streams get that from the following frame; the test repeats the
        # short clip so ffmpeg sees a continuous Annex-B stream (and enough
        # data to finish its probe) before stdin is closed.
        decoder.feed((encoded + b"\x00\x00\x00\x01") * 40, keyframe=True)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not frames and not errors:
            time.sleep(0.02)
        assert not errors
        assert frames
        assert frames[0].shape == (90, 160, 3)
        assert decoder.frames >= 1
    finally:
        decoder.close()
