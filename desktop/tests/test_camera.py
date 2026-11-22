import numpy as np

from openvrcardboard.camera import CameraCapture, _fourcc_text, _fourcc_value
from openvrcardboard.config import CameraConfig


class FakeCv2:
    CAP_PROP_FOURCC = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5
    CAP_PROP_AUTOFOCUS = 6
    CAP_PROP_FOCUS = 7
    CAP_PROP_AUTO_EXPOSURE = 8
    CAP_PROP_EXPOSURE = 9
    CAP_PROP_GAIN = 10
    CAP_PROP_AUTO_WB = 11
    CAP_PROP_WB_TEMPERATURE = 12

    def __init__(self, results):
        self.results = list(results)
        self.created = []

    def VideoCapture(self, index, backend=None):
        device = FakeDevice(self, index, self.results.pop(0))
        self.created.append(device)
        return device


class FakeDevice:
    def __init__(self, cv2, index, result):
        self.cv2 = cv2
        self.index = index
        self.result = result
        self.sets = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def get(self, prop):
        width, height, fps, fourcc = self.result
        return {
            self.cv2.CAP_PROP_FRAME_WIDTH: width,
            self.cv2.CAP_PROP_FRAME_HEIGHT: height,
            self.cv2.CAP_PROP_FPS: fps,
            self.cv2.CAP_PROP_FOURCC: _fourcc_value(None, fourcc) if fourcc else 0,
        }.get(prop, 0)

    def read(self):
        return True, np.zeros((self.result[1], self.result[0], 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_fourcc_roundtrip():
    value = _fourcc_value(None, "MJPG")
    assert isinstance(value, int)
    assert _fourcc_text(None, value) == "MJPG"
    assert _fourcc_value(None, "bad") == 0


def test_camera_sets_fourcc_before_frame_geometry():
    cv2 = FakeCv2([(1920, 1080, 60.0, "MJPG")])
    config = CameraConfig(index=2, width=1920, height=1080, fps=60, fourcc="MJPG")
    capture = CameraCapture(config)
    capture._cv2 = cv2
    assert capture._open()
    device = cv2.created[0]
    assert device.index == 2
    properties = [prop for prop, _ in device.sets]
    assert properties.index(cv2.CAP_PROP_FOURCC) < properties.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert properties.index(cv2.CAP_PROP_FRAME_WIDTH) < properties.index(cv2.CAP_PROP_FRAME_HEIGHT)
    assert properties.index(cv2.CAP_PROP_FRAME_HEIGHT) < properties.index(cv2.CAP_PROP_FPS)
    stats = capture.stats()
    assert stats.mode_ok
    assert stats.fourcc == "MJPG"
    assert (stats.requested_width, stats.requested_height, stats.requested_fps) == (
        1920,
        1080,
        60,
    )


def test_camera_retries_a_lower_mode_when_negotiation_fails():
    cv2 = FakeCv2([(1280, 720, 30.0, "MJPG"), (1280, 720, 60.0, "MJPG")])
    config = CameraConfig(width=1920, height=1080, fps=60, fourcc="MJPG")
    capture = CameraCapture(config)
    capture._cv2 = cv2
    assert capture._open()
    assert len(cv2.created) == 2
    assert cv2.created[0].released
    stats = capture.stats()
    assert (stats.width, stats.height) == (1280, 720)
    assert stats.mode_ok is False


def test_camera_falls_back_to_1080p30_before_720p30():
    cv2 = FakeCv2(
        [
            (1920, 1080, 30.0, "MJPG"),
            (1280, 720, 30.0, "MJPG"),
            (1920, 1080, 30.0, "MJPG"),
        ]
    )
    config = CameraConfig(width=1920, height=1080, fps=60, fourcc="MJPG")
    capture = CameraCapture(config)
    capture._cv2 = cv2
    assert capture._open()
    assert len(cv2.created) == 3
    stats = capture.stats()
    assert (stats.width, stats.height, stats.requested_fps) == (1920, 1080, 60)
    assert stats.mode_ok is False
    third_sets = dict(cv2.created[2].sets)
    assert third_sets[cv2.CAP_PROP_FPS] == 30.0


def test_camera_fallback_order_uses_the_shared_ladder():
    attempts = CameraCapture._fallback_attempts(("MJPG", 1920, 1080, 60))
    assert attempts == [
        ("MJPG", 1280, 720, 60),
        ("MJPG", 1920, 1080, 30),
        ("MJPG", 1280, 720, 30),
    ]


def test_camera_default_fallback_starts_at_1080p120():
    attempts = CameraCapture._fallback_attempts(("MJPG", 1920, 1080, 120))
    assert attempts[:4] == [
        ("MJPG", 1280, 720, 120),
        ("MJPG", 1920, 1080, 90),
        ("MJPG", 1280, 720, 90),
        ("MJPG", 1920, 1080, 60),
    ]


def test_camera_fallback_never_upscales_a_lower_request():
    attempts = CameraCapture._fallback_attempts(("MJPG", 1280, 720, 60))
    assert all((width, height) != (1920, 1080) for _, width, height, _ in attempts)
    assert attempts[0] == ("MJPG", 1280, 720, 30)


def test_camera_fallback_defaults_stay_within_the_requested_mode():
    assert CameraCapture._fallback_attempts(("MJPG", 640, 480, 60)) == []
    assert CameraCapture._fallback_attempts(("MJPG", 1920, 1080, 15)) == []


def test_camera_accepts_whatever_the_last_resort_negotiates():
    cv2 = FakeCv2([(640, 480, 15.0, "MJPG")] * 5)
    config = CameraConfig(width=1920, height=1080, fps=60, fourcc="MJPG")
    capture = CameraCapture(config)
    capture._cv2 = cv2
    assert capture._open()
    assert len(cv2.created) == 5
    stats = capture.stats()
    assert (stats.width, stats.height) == (640, 480)
    assert stats.mode_ok is False


def test_camera_controls_only_apply_when_automatic_is_off():
    cv2 = FakeCv2([(1280, 720, 60.0, "MJPG")])
    config = CameraConfig(
        width=1280,
        height=720,
        fps=60,
        fourcc="MJPG",
        autofocus=False,
        focus=10.0,
        auto_exposure=False,
        exposure=5.0,
        gain=2.0,
        auto_white_balance=False,
        white_balance=4500.0,
    )
    capture = CameraCapture(config)
    capture._cv2 = cv2
    assert capture._open()
    sets = dict(cv2.created[0].sets)
    assert sets[cv2.CAP_PROP_AUTOFOCUS] == 0.0
    assert sets[cv2.CAP_PROP_FOCUS] == 10.0
    assert sets[cv2.CAP_PROP_AUTO_EXPOSURE] == 1.0
    assert sets[cv2.CAP_PROP_EXPOSURE] == 5.0
    assert sets[cv2.CAP_PROP_GAIN] == 2.0
    assert sets[cv2.CAP_PROP_AUTO_WB] == 0.0
    assert sets[cv2.CAP_PROP_WB_TEMPERATURE] == 4500.0


def test_camera_mode_matching_accepts_unknown_fps():
    assert CameraCapture._mode_matches((1920, 1080, 0.0, ""), (1920, 1080, 60))
    assert not CameraCapture._mode_matches((1280, 720, 30.0, ""), (1920, 1080, 60))
    assert not CameraCapture._mode_matches((1280, 720, 30.0, ""), (1280, 720, 60))
    assert CameraCapture._mode_matches((1920, 1080, 30.0, ""), (1920, 1080, 30))


def test_marker_auto_control_stops_after_board_leaves_view():
    capture = CameraCapture(
        CameraConfig(auto_marker_control=True, camera_aggressiveness=1.0)
    )
    capture.report_marker_detection(5, 5)
    capture._last_visible_time -= 2.0
    capture.report_marker_detection(0, 0)
    capture._maybe_adapt_camera(FakeCv2([]), np.zeros((8, 8, 3), np.uint8), capture._last_detection_time)
    assert capture._last_auto_action == 0.0
