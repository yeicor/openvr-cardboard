import io
import json
import time

import pytest
from PIL import Image

from openvrcardboard.config import PhoneConfig
from openvrcardboard.openxr import CaptureFrame
from openvrcardboard.protocol import (
    VideoLayout,
    pack_video_frame,
    unpack_video_frame,
    unpack_video_frame_details,
)
from openvrcardboard.video import (
    OpenXRCaptureSource,
    TestPatternSource,
    compose_image,
    encode_jpeg,
    frame_dimensions,
)
from openvrcardboard.video_server import VideoStreamServer


class FakeCapture:
    """Minimal stand-in for OpenXRCaptureServer.latest()."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def latest(self, after_sequence=None, timeout=2.0):
        if not self.frames:
            return None
        if after_sequence is None:
            self.calls += 1
            return self.frames[0]
        self.calls += 1
        if self.calls > 200:
            return None
        base = self.frames[self.calls % len(self.frames)]
        return CaptureFrame(
            sequence=(after_sequence or 0) + 1,
            timestamp=time.monotonic(),
            width=base.width,
            height=base.height,
            jpeg=base.jpeg,
        )


def _sbs_frame(eye_width=160, eye_height=90, sequence=1):
    image = TestPatternSource(eye_width, eye_height, "grid").read()
    sbs = compose_image(image, "sbs", eye_width, eye_height)
    return CaptureFrame(
        sequence=sequence,
        timestamp=time.monotonic(),
        width=sbs.width,
        height=sbs.height,
        jpeg=encode_jpeg(sbs, 70),
    )


def test_openxr_source_forwards_matching_jpeg_without_reencoding():
    frame = _sbs_frame()
    source = OpenXRCaptureSource(FakeCapture([frame]))
    source.open()
    encoded = source.read_encoded("sbs", 160, 90, 60)
    assert encoded.raw
    assert encoded.jpeg is frame.jpeg
    assert (encoded.width, encoded.height) == (320, 90)


def test_openxr_source_resizes_when_layout_or_size_changes():
    frame = _sbs_frame()
    source = OpenXRCaptureSource(FakeCapture([frame]))
    source.open()
    encoded = source.read_encoded("sbs", 80, 45, 60)
    assert not encoded.raw
    assert (encoded.width, encoded.height) == (160, 45)
    decoded = Image.open(io.BytesIO(encoded.jpeg))
    assert decoded.size == (160, 45)

    source = OpenXRCaptureSource(FakeCapture([frame]))
    source.open()
    encoded = source.read_encoded("top_bottom", 160, 90, 60)
    assert not encoded.raw
    assert (encoded.width, encoded.height) == (160, 180)


def test_video_server_counts_raw_openxr_passthrough_frames():
    frame = _sbs_frame()
    config = PhoneConfig(
        video_enabled=True,
        video_port=0,
        layout="sbs",
        capture_source="openxr",
        width=160,
        height=90,
        fps=15,
        jpeg_quality=55,
        token="secret",
    )
    server = VideoStreamServer(
        config, host="127.0.0.1", capture_provider=FakeCapture([frame])
    )
    server.start(timeout=5.0)
    connection = None
    try:
        connect = websocket_client()
        connection = connect(
            f"ws://127.0.0.1:{server.local_port}/video?token=secret",
            open_timeout=2.0,
            max_size=1 << 20,
            legacy=True,
        )
        assert json.loads(connection.recv(timeout=2.0))["type"] == "welcome"
        packet = connection.recv(timeout=3.0)
        assert isinstance(packet, bytes)
        layout, width, height, _, _, jpeg = unpack_video_frame(packet)
        assert layout == VideoLayout.SBS
        assert (width, height) == (320, 90)
        assert jpeg == frame.jpeg
        deadline = time.monotonic() + 2.0
        stats = server.stats()
        while time.monotonic() < deadline and stats.raw_frames == 0:
            time.sleep(0.05)
            stats = server.stats()
        assert stats.raw_frames >= 1
        assert stats.composed_frames == 0
    finally:
        if connection is not None:
            connection.close()
        server.stop()



def websocket_client():
    from websockets.sync.client import connect

    return connect


def test_video_frame_flags_carry_codec_keyframe_and_config():
    packet = pack_video_frame(
        7,
        12.5,
        VideoLayout.SBS,
        2400,
        1080,
        b"\x00\x00\x01\x26payload",
        codec="hevc",
        keyframe=True,
        config=False,
    )
    (
        layout,
        width,
        height,
        sequence,
        timestamp,
        payload,
        codec,
        keyframe,
        config,
    ) = unpack_video_frame_details(packet)
    assert layout == VideoLayout.SBS
    assert (width, height) == (2400, 1080)
    assert sequence == 7
    assert timestamp == pytest.approx(12.5)
    assert payload == b"\x00\x00\x01\x26payload"
    assert codec == "hevc"
    assert keyframe
    assert not config
    # The legacy tuple stays compatible with older phone builds.
    assert unpack_video_frame(packet)[:5] == (
        VideoLayout.SBS,
        2400,
        1080,
        7,
        pytest.approx(12.5),
    )


def test_frame_dimensions():
    assert frame_dimensions("mono", 1280, 720) == (1280, 720)
    assert frame_dimensions("sbs", 1280, 720) == (2560, 720)
    assert frame_dimensions("top_bottom", 1280, 720) == (1280, 1440)


def test_compose_image_duplicates_eyes():
    source = TestPatternSource(160, 90, "grid").read()
    sbs = compose_image(source, "sbs", 160, 90)
    assert sbs.size == (320, 90)
    assert sbs.getpixel((10, 10)) == sbs.getpixel((170, 10))

    top_bottom = compose_image(source, "top_bottom", 160, 90)
    assert top_bottom.size == (160, 180)
    assert top_bottom.getpixel((10, 10)) == top_bottom.getpixel((10, 100))

    mono = compose_image(source, "mono", 320, 180)
    assert mono.size == (320, 180)


def test_compose_preserves_aspect_with_letterbox():
    source = Image.new("RGB", (160, 90), (255, 0, 0))
    output = compose_image(source, "mono", 100, 100)
    assert output.size == (100, 100)
    # Top and bottom are padded black, centre is the red source.
    assert output.getpixel((50, 5)) == (0, 0, 0)
    assert output.getpixel((50, 50)) == (255, 0, 0)


def test_encode_jpeg_is_decodable():
    image = TestPatternSource(120, 80, "bars").read()
    data = encode_jpeg(image, quality=60)
    assert data[:2] == b"\xff\xd8"
    decoded = Image.open(io.BytesIO(data))
    assert decoded.size == (120, 80)


def test_video_server_streams_frames_and_answers_ping():
    config = PhoneConfig(
        video_enabled=True,
        video_port=0,
        layout="sbs",
        capture_source="test",
        test_pattern="grid",
        width=160,
        height=90,
        fps=15,
        jpeg_quality=55,
    )
    server = VideoStreamServer(config, host="127.0.0.1")
    server.start(timeout=5.0)
    connection = None
    try:
        connect = websocket_client()
        connection = connect(
            f"ws://127.0.0.1:{server.local_port}/video",
            open_timeout=2.0,
            max_size=1 << 20,
            legacy=True,
        )
        welcome = json.loads(connection.recv(timeout=2.0))
        assert welcome["type"] == "welcome"
        assert welcome["video"]["layout"] == "sbs"
        assert welcome["video"]["width"] == 320
        assert welcome["video"]["height"] == 90

        packet = connection.recv(timeout=3.0)
        assert isinstance(packet, bytes)
        layout, width, height, sequence, timestamp, jpeg = unpack_video_frame(packet)
        assert layout == VideoLayout.SBS
        assert (width, height) == (320, 90)
        assert jpeg[:2] == b"\xff\xd8"
        assert timestamp > 0.0

        connection.send(json.dumps({"type": "ping", "t": 42.5}))
        deadline = time.monotonic() + 2.0
        pong = None
        while time.monotonic() < deadline:
            message = connection.recv(timeout=2.0)
            if isinstance(message, str):
                parsed = json.loads(message)
                if parsed.get("type") == "pong":
                    pong = parsed
                    break
        assert pong is not None
        assert pong["t"] == 42.5

        stats = server.stats()
        assert stats.running
        assert stats.clients == 1
        assert stats.frames >= 1
        assert stats.last_error == ""
        assert stats.width == 320
        assert stats.height == 90
    finally:
        if connection is not None:
            connection.close()
        server.stop()


def test_video_server_rejects_bad_token():
    config = PhoneConfig(
        video_enabled=True,
        video_port=0,
        layout="mono",
        capture_source="test",
        width=80,
        height=60,
        fps=5,
        token="secret",
    )
    server = VideoStreamServer(config, host="127.0.0.1")
    server.start(timeout=5.0)
    connection = None
    try:
        connect = websocket_client()
        connection = connect(
            f"ws://127.0.0.1:{server.local_port}/video",
            open_timeout=2.0,
            legacy=True,
        )
        message = json.loads(connection.recv(timeout=2.0))
        assert message["type"] == "error"
        assert "token" in message["message"]
    finally:
        if connection is not None:
            connection.close()
        server.stop()


def test_video_server_broadcast_recenter():
    config = PhoneConfig(
        video_enabled=True,
        video_port=0,
        layout="mono",
        capture_source="test",
        width=80,
        height=60,
        fps=5,
    )
    server = VideoStreamServer(config, host="127.0.0.1")
    server.start(timeout=5.0)
    connection = None
    try:
        connect = websocket_client()
        connection = connect(
            f"ws://127.0.0.1:{server.local_port}/video",
            open_timeout=2.0,
            legacy=True,
        )
        assert json.loads(connection.recv(timeout=2.0))["type"] == "welcome"
        assert server.recenter() == 1
        deadline = time.monotonic() + 2.0
        found = False
        while time.monotonic() < deadline:
            message = connection.recv(timeout=2.0)
            if isinstance(message, str) and json.loads(message).get("type") == "recenter":
                found = True
                break
        assert found
    finally:
        if connection is not None:
            connection.close()
        server.stop()
