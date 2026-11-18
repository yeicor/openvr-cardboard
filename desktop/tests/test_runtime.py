from pathlib import Path
import socket

import numpy as np
import pytest

import openvrcardboard.runtime as runtime_module
from openvrcardboard.config import AppConfig, load_config, save_config
from openvrcardboard.markers import MarkerDetection
from openvrcardboard.mathutil import IDENTITY
from openvrcardboard.phone_link import PhoneImuReceiver
from openvrcardboard.protocol import DeviceId
from openvrcardboard.runtime import Runtime, create_runtime
from openvrcardboard.types import PhoneSample


def make_config() -> AppConfig:
    config = AppConfig()
    config.driver.udp_port = 39400
    config.phone.imu_port = 39401
    config.phone.video_enabled = False
    config.gamepad.enabled = False
    config.openxr.enabled = False
    config.ui.start_tracking_on_launch = False
    config.tracking.markers_enabled = False
    config.tracking.hands_enabled = False
    config.tracking.body_enabled = False
    return config


def test_runtime_starts_and_stops_without_optional_dependencies():
    runtime = Runtime(make_config())
    try:
        problems = runtime.start()
        assert isinstance(problems, list)
        assert runtime.fusion.running
        assert runtime.driver.running
        assert runtime.phone.running
        status = runtime.status()
        assert status.driver_packets >= 0
        assert status.camera_resolution == (0, 0)
        runtime.recenter()
        assert runtime.state.recenter_count == 1
    finally:
        runtime.stop()
    assert not runtime.fusion.running
    assert not runtime.driver.running
    assert not runtime.phone.running


def test_runtime_falls_back_to_estimated_camera_model():
    runtime = Runtime(make_config())
    try:
        assert runtime.model.width == 1920
        assert runtime.model.height == 1080
        assert any("calibration" in note for note in runtime.warnings)
    finally:
        runtime.stop()


def test_fusion_step_composes_marker_hmd():
    config = make_config()
    config.tracking.markers_enabled = True
    runtime = Runtime(config)
    try:
        detection = MarkerDetection(
            marker_id=0,
            position=np.array([0.1, 1.7, -0.2]),
            rotation=IDENTITY.copy(),
            optical_position=np.zeros(3),
            optical_rotation=IDENTITY.copy(),
            reprojection_error=0.5,
            distance=1.0,
        )
        runtime.state.set_markers({0: detection}, timestamp=100.0)
        frame = runtime.fusion.step(now=100.1)
        assert frame.hmd is not None
        assert frame.hmd.position == pytest.approx([0.1, 1.7, -0.2])
        assert frame.sources["hmd"] == "markers"
        assert frame.left_controller is not None
        assert frame.right_controller is not None
    finally:
        runtime.stop()


def test_fusion_step_consumes_phone_sample():
    config = make_config()
    runtime = Runtime(config)
    try:
        sample = PhoneSample(
            sequence=1,
            timestamp=time_now(),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            gyroscope=np.zeros(3),
            acceleration=np.zeros(3),
            accuracy=3,
            received_at=time_now(),
        )
        runtime.state.set_phone(sample)
        runtime.fusion.step(now=time_now() + 0.01)
        pose = runtime.state.snapshot().phone_pose
        assert pose is not None
        assert pose.valid
    finally:
        runtime.stop()


def test_save_and_load_config_roundtrip(tmp_path):
    config = make_config()
    config.driver.udp_port = 39123
    config.tracking.hand_swap = True
    path = save_config(config, tmp_path / "config.json")
    loaded, warnings = load_config(path)
    assert loaded.driver.udp_port == 39123
    assert loaded.tracking.hand_swap
    assert warnings == []


def test_create_runtime_reads_config(tmp_path):
    config = make_config()
    config.phone.video_port = 8766
    path = save_config(config, tmp_path / "config.json")
    runtime = create_runtime(str(path))
    try:
        assert runtime.config.phone.video_port == 8766
        assert runtime.config_path == Path(path)
    finally:
        runtime.stop()


@pytest.mark.parametrize("tracking", [True, False])
def test_apply_config_preserves_tracking_state(monkeypatch, tracking):
    runtime = Runtime(make_config())
    try:
        runtime.start(start_tracking=tracking)
        assert runtime.running
        assert runtime.tracking.running == tracking

        calls = []
        monkeypatch.setattr(
            Runtime,
            "start",
            lambda self, start_tracking=None: calls.append(start_tracking) or [],
        )
        runtime.apply_config(make_config())
        assert calls == [tracking]
    finally:
        runtime.stop()


class ExclusivePhoneReceiver(PhoneImuReceiver):
    """Receiver that binds without SO_REUSEADDR, exposing double binds."""

    def _open_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.host, self.port))
            sock.setblocking(False)
            return sock
        except Exception:
            sock.close()
            raise


def test_apply_config_closes_old_imu_socket_before_rebinding(monkeypatch):
    runtime = Runtime(make_config())
    try:
        runtime.start(start_tracking=False)
        monkeypatch.setattr(runtime_module, "PhoneImuReceiver", ExclusivePhoneReceiver)
        warnings = runtime.apply_config(make_config())
        assert warnings == []
        assert runtime.running
        assert isinstance(runtime.phone, ExclusivePhoneReceiver)
        assert runtime.phone.local_port == make_config().phone.imu_port
    finally:
        runtime.stop()


def test_apply_config_failure_restores_previous_runtime(monkeypatch):
    runtime = Runtime(make_config())
    try:
        runtime.start(start_tracking=False)
        attempts = []

        class FlakyReceiver(ExclusivePhoneReceiver):
            def __init__(self, *args, **kwargs):
                attempts.append(1)
                if len(attempts) == 1:
                    raise OSError("simulated bind failure")
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(runtime_module, "PhoneImuReceiver", FlakyReceiver)
        warnings = runtime.apply_config(make_config())
        assert any("could not apply configuration" in item for item in warnings)
        assert runtime.running
        assert isinstance(runtime.phone, FlakyReceiver)
        assert len(attempts) == 2
    finally:
        runtime.stop()


def time_now() -> float:
    import time

    return time.monotonic()


def test_openxr_haptics_are_routed_to_synthetic_gamepad():
    import time

    config = make_config()
    config.gamepad.enabled = True
    config.gamepad.synthetic = True
    runtime = Runtime(config)
    try:
        runtime.start(start_tracking=False)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not runtime.gamepads.status()["pads"]:
            time.sleep(0.02)
        assert runtime.gamepads.status()["pads"]
        runtime._on_openxr_haptic("left", 0.5, 0.025)
        status = runtime.gamepads.status()["haptics"]
        assert status["count"] == 1
        assert status["last"]["side"] == "left"
        assert status["last"]["amplitude"] == pytest.approx(0.5)
        assert status["last"]["duration_ms"] == 25
    finally:
        runtime.stop()
