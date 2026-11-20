"""Full-stack OpenXR test: desktop runtime + real Monado + simple-example.

This is the hardware-free proof that the app feeds a real OpenXR application:
the synthetic webcam scene drives marker and body tracking, synthetic Xbox pads
drive the controllers, the runtime streams poses through Monado's remote
driver, and the capture shim reports the example's stereo eye textures, grab
action and haptics back to the desktop app.

Skipped when ``scripts/build-openxr.sh`` has not produced a runtime.
"""

import io
import os
import socket
import time
from importlib.util import find_spec

import numpy as np
import pytest
from PIL import Image

from openvrcardboard.config import AppConfig, validate_config
from openvrcardboard.openxr_stack import OpenXRStack, find_openxr_prefix
from openvrcardboard.runtime import Runtime

PREFIX = find_openxr_prefix()
_MISSING_OPTIONAL = [
    name for name in ("cv2", "mediapipe") if find_spec(name) is None
]
pytestmark = pytest.mark.skipif(
    PREFIX is None or not os.environ.get("DISPLAY"),
    reason="OpenXR runtime or display server is unavailable",
)
if _MISSING_OPTIONAL:
    pytestmark = pytest.mark.skip(
        reason="tracking stack is unavailable: install " + ", ".join(_MISSING_OPTIONAL)
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _config(remote_port: int, capture_port: int) -> AppConfig:
    config = AppConfig()
    config.driver.udp_port = _free_port()
    config.phone.imu_port = _free_udp_port()
    config.phone.video_enabled = False
    config.phone.capture_source = "openxr"
    config.camera.source = "synthetic"
    config.camera.width = 1280
    config.camera.height = 720
    config.camera.fps = 60
    config.tracking.markers_enabled = True
    config.tracking.hands_enabled = False
    config.tracking.body_enabled = True
    config.tracking.hmd_source = "markers"
    config.tracking.left_source = "markers"
    config.tracking.right_source = "markers"
    config.gamepad.enabled = True
    config.gamepad.synthetic = True
    config.openxr.enabled = True
    config.openxr.manage_service = False
    config.openxr.remote_port = remote_port
    config.openxr.capture_port = capture_port
    config.openxr.send_rate_hz = 100
    validate_config(config)
    return config


def _position(diagnostics, name):
    composed = diagnostics.get("composed") or {}
    value = composed.get(name)
    if not value or not value.get("valid"):
        return None
    return np.asarray(value["position"], dtype=np.float64)


def _eye_stats(jpeg: bytes) -> tuple[float, float, float]:
    image = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.int16)
    width = image.shape[1] // 2
    left = image[:, :width]
    right = image[:, width:]
    return (
        float(left.mean()),
        float(right.mean()),
        float(np.abs(left - right).mean()),
    )


def test_runtime_drives_real_openxr_application(tmp_path):
    remote_port = _free_port()
    capture_port = _free_port()
    config = _config(remote_port, capture_port)
    runtime = Runtime(config, tmp_path / "config.json")
    stack = OpenXRStack(
        prefix=PREFIX,
        work_dir=tmp_path / "openxr",
        remote_port=remote_port,
        capture_port=capture_port,
        display=os.environ.get("DISPLAY"),
    )
    samples = []
    try:
        problems = runtime.start(start_tracking=True)
        assert not [item for item in problems if "OpenXR" in item or "tracking" in item]
        stack.start()
        stack.wait_example_connected(timeout=25.0)

        max_grab = {"left": 0.0, "right": 0.0}
        deadline = time.monotonic() + 22.0
        while time.monotonic() < deadline:
            diagnostics = runtime.diagnostics()
            samples.append(diagnostics)
            capture = diagnostics["openxr"]["capture"]
            for hand in max_grab:
                max_grab[hand] = max(max_grab[hand], float(capture[f"grab_{hand}"]))
            haptics = runtime.gamepads.status()["haptics"]
            if (
                capture["frames"] >= 30
                and max_grab["left"] > 0.75
                and max_grab["right"] > 0.75
                and capture["haptic_events"] > 0
                and haptics["count"] > 0
                and diagnostics["body"]
            ):
                break
            time.sleep(0.2)
        else:
            capture = samples[-1]["openxr"]["capture"] if samples else {}
            haptics = runtime.gamepads.status()["haptics"]
            raise AssertionError(
                "OpenXR chain did not produce the expected evidence; "
                f"capture={capture}, max_grab={max_grab}, haptics={haptics}\n"
                f"example log:\n{stack.example_log.read_text(errors='replace')[-3000:]}"
            )

        last = samples[-1]
        capture = last["openxr"]["capture"]
        assert last["openxr"]["remote"]["connected"]
        assert last["openxr"]["remote"]["packets"] > 0
        assert capture["connected"]
        assert capture["frames"] >= 30
        assert (capture["width"], capture["height"]) == (1280, 720)
        assert capture["grab_active_left"] and capture["grab_active_right"]
        assert max_grab["left"] > 0.75 and max_grab["right"] > 0.75
        assert capture["haptic_events"] > 0
        assert capture["last_haptic"]["amplitude"] > 0.0

        # The runtime's haptics callback reaches the synthetic Xbox pad.
        haptics = runtime.gamepads.status()["haptics"]
        assert haptics["count"] > 0
        assert haptics["last"]["amplitude"] > 0.0
        assert haptics["last"]["duration_ms"] >= 10

        # Markers and body tracking fed the OpenXR scene.  Each five-marker
        # board publishes one 6DoF object under its primary marker ID.
        marker_ids = {
            int(marker_id)
            for sample in samples
            for marker_id in sample["markers"]
        }
        assert {0, 5, 10}.issubset(marker_ids)
        assert any(sample["body"] for sample in samples)
        assert last["tracking"]["markers"]

        # Composed 6DoF poses actually move: HMD sways and both controllers
        # travel through play space while the example renders.
        for name, minimum in (("hmd", 0.02), ("left", 0.02), ("right", 0.02)):
            positions = [pose for pose in (_position(item, name) for item in samples) if pose is not None]
            assert len(positions) >= 10, name
            spread = np.ptp(np.asarray(positions), axis=0)
            assert float(spread.max()) > minimum, f"{name} barely moved: {spread}"

        # Stereo eye textures are both valid and visibly different, so the
        # recording proves per-eye rendering rather than a duplicated frame.
        frame = runtime.openxr.capture.latest(timeout=2.0)
        assert frame is not None
        left_mean, right_mean, difference = _eye_stats(frame.jpeg)
        assert left_mean > 10.0 and right_mean > 10.0, (left_mean, right_mean)
        assert difference > 0.5, difference
    finally:
        stack.stop()
        runtime.stop()


def test_monado_service_supervisor_starts_the_bundled_runtime(tmp_path, monkeypatch):
    """The packaged app starts monado-service itself (no E2E harness)."""
    from openvrcardboard.openxr import MonadoService

    port = _free_port()
    config_home = tmp_path / "config"
    monado_config = config_home / "monado"
    monado_config.mkdir(parents=True)
    (monado_config / "config_v0.json").write_text(
        "{\n"
        '  "active": "remote",\n'
        '  "remote": {\n'
        '    "version": 0,\n'
        f'    "port": {port},\n'
        '    "view_count": 2,\n'
        '    "w_pixels": 2400,\n'
        '    "h_pixels": 1080,\n'
        '    "w_meters": 0.13,\n'
        '    "h_meters": 0.07\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    service = MonadoService(
        service=PREFIX / "bin" / "monado-service",
        remote_port=port,
        log_path=tmp_path / "monado-service.log",
        startup_timeout=20.0,
    )
    try:
        service.start()
        stats = service.stats()
        assert stats.running
        assert stats.managed
        assert not stats.external
        assert stats.pid > 0
        with socket.create_connection(("127.0.0.1", port), timeout=2.0):
            pass
    finally:
        service.stop()
    assert not service.running
