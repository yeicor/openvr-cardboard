"""Settings dialog wiring: every config section round-trips through the UI."""

import copy
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QTabWidget  # noqa: E402

from openvrcardboard.config import AppConfig  # noqa: E402
from openvrcardboard.runtime import Runtime  # noqa: E402
from openvrcardboard.ui.main_window import MainWindow, SettingsDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def _dialog(app):
    dialog = SettingsDialog(AppConfig())
    dialog.show()
    app.processEvents()
    return dialog


def test_settings_exposes_every_configuration_area(app):
    dialog = _dialog(app)
    try:
        tabs = dialog.findChild(QTabWidget)
        names = [tabs.tabText(index) for index in range(tabs.count())]
        assert names == [
            "Camera && markers",
            "Boards",
            "Tracking",
            "Phone video",
            "Lens && display",
            "Gamepad",
            "Driver",
            "OpenXR",
        ]
    finally:
        dialog.close()


def test_settings_round_trips_camera_and_tracking(app):
    dialog = _dialog(app)
    try:
        dialog.camera_source.setCurrentText("synthetic")
        dialog.camera_width.setValue(1280)
        dialog.camera_height.setValue(720)
        dialog.camera_fps.setValue(60)
        dialog.camera_fourcc.setText("YUYV")
        dialog.camera_autofocus.setChecked(False)
        dialog.camera_focus.setValue(12.0)
        dialog.camera_auto_exposure.setChecked(False)
        dialog.camera_exposure.setValue(-4.0)
        dialog.camera_gain.setValue(3.0)
        dialog.camera_auto_wb.setChecked(False)
        dialog.camera_white_balance.setValue(5000.0)
        dialog.camera_auto_marker.setChecked(False)
        dialog.camera_aggressiveness.setValue(0.8)
        dialog.camera_target_brightness.setValue(0.6)
        dialog.camera_max_gain.setValue(32.0)
        dialog.camera_timestamp_source.setCurrentText("pos_msec")
        dialog.camera_timestamp_offset.setValue(-4.5)
        dialog.camera_allow_fallback.setChecked(False)
        dialog.camera_verify_mode.setChecked(False)
        dialog.camera_max_error.setValue(2.5)
        dialog.camera_dictionary.setCurrentText("DICT_4X4_250")
        dialog.camera_detection_width.setValue(1280)
        dialog.camera_refine_pose.setChecked(False)
        dialog.camera_board_cols.setValue(7)
        dialog.camera_board_rows.setValue(5)
        dialog.camera_board_square.setValue(30.0)
        dialog.camera_min_views.setValue(15)
        dialog.track_inference_width.setValue(640)
        dialog.track_inference_height.setValue(360)
        dialog.track_body_model.setCurrentText("lite")
        dialog.track_delegate.setCurrentText("cpu")
        dialog.track_model_dir.setText("/tmp/models")
        dialog.track_smoothing_preset.setCurrentText("Responsive")
        dialog.track_roles.setText("waist, left_foot , ,right_hand")
        dialog.track_marker_process.setValue(0.7)
        dialog.track_marker_angular.setValue(1.4)
        dialog.track_marker_output.setValue(3.5)
        dialog.track_marker_adaptive.setChecked(False)
        dialog.track_marker_static_scale.setValue(0.35)
        dialog.track_marker_min_markers.setValue(4)
        boards = dialog.board_widgets
        boards["hmd"]["width"].setValue(210.0)
        boards["hmd"]["height"].setValue(110.0)
        boards["hmd"]["central"].setValue(82.0)
        boards["hmd"]["peripheral"].setValue(42.0)
        boards["hmd"]["peripheral_x"].setValue(72.0)
        boards["hmd"]["peripheral_y"].setValue(25.0)
        boards["hmd"]["object_position"].setText("0.01, 0.02, 0.03")
        boards["hmd"]["object_rotation"].setText("0, 1, 0, 0")
        cfg = dialog.result_config()
        assert cfg.camera.source == "synthetic"
        assert (cfg.camera.width, cfg.camera.height, cfg.camera.fps) == (1280, 720, 60)
        assert cfg.camera.fourcc == "YUYV"
        assert cfg.camera.autofocus is False
        assert cfg.camera.focus == pytest.approx(12.0)
        assert cfg.camera.auto_exposure is False
        assert cfg.camera.exposure == pytest.approx(-4.0)
        assert cfg.camera.gain == pytest.approx(3.0)
        assert cfg.camera.auto_white_balance is False
        assert cfg.camera.white_balance == pytest.approx(5000.0)
        assert cfg.camera.auto_marker_control is False
        assert cfg.camera.camera_aggressiveness == pytest.approx(0.8)
        assert cfg.camera.camera_target_brightness == pytest.approx(0.6)
        assert cfg.camera.camera_max_gain == pytest.approx(32.0)
        assert cfg.camera.camera_timestamp_source == "pos_msec"
        assert cfg.camera.camera_timestamp_offset_ms == pytest.approx(-4.5)
        assert cfg.camera.allow_mode_fallback is False
        assert cfg.camera.verify_mode is False
        assert cfg.camera.max_reprojection_error == pytest.approx(2.5)
        assert cfg.camera.dictionary == "DICT_4X4_250"
        assert cfg.camera.marker_detection_width == 1280
        assert cfg.camera.marker_refine_pose is False
        assert (cfg.camera.calibration_board_cols, cfg.camera.calibration_board_rows) == (7, 5)
        assert cfg.camera.calibration_square_mm == pytest.approx(30.0)
        assert cfg.camera.calibration_min_views == 15
        assert (cfg.tracking.inference_width, cfg.tracking.inference_height) == (640, 360)
        assert cfg.tracking.body_model == "lite"
        assert cfg.tracking.mediapipe_delegate == "cpu"
        assert cfg.tracking.model_dir == "/tmp/models"
        assert cfg.tracking.smoothing_min_cutoff == pytest.approx(4.0)
        assert cfg.tracking.smoothing_beta == pytest.approx(0.4)
        assert cfg.tracking.orientation_cutoff == pytest.approx(7.0)
        assert cfg.tracking.tracker_roles == ["waist", "left_foot", "right_hand"]
        assert cfg.tracking.marker_process_noise == pytest.approx(0.7)
        assert cfg.tracking.marker_angular_noise == pytest.approx(1.4)
        assert cfg.tracking.marker_output_smoothing_hz == pytest.approx(3.5)
        assert cfg.tracking.marker_adaptive_noise is False
        assert cfg.tracking.marker_static_noise_scale == pytest.approx(0.35)
        assert cfg.tracking.marker_min_markers == 4
        board = cfg.tracking.boards.hmd
        assert board.width_m == pytest.approx(0.210)
        assert board.height_m == pytest.approx(0.110)
        assert board.central_marker_m == pytest.approx(0.082)
        assert board.peripheral_marker_m == pytest.approx(0.042)
        assert board.peripheral_x_m == pytest.approx(0.072)
        assert board.peripheral_y_m == pytest.approx(0.025)
        assert board.object_position == pytest.approx([0.01, 0.02, 0.03])
    finally:
        dialog.close()


def test_smoothing_preset_and_custom_round_trip(app):
    dialog = _dialog(app)
    try:
        dialog.track_smoothing_preset.setCurrentText("Stable")
        assert dialog.track_smoothing.value() == pytest.approx(1.0)
        assert dialog.track_beta.value() == pytest.approx(0.15)
        assert dialog.track_orientation.value() == pytest.approx(2.0)
        dialog.track_smoothing.setValue(1.7)
        assert dialog.track_smoothing_preset.currentText() == "Custom"
    finally:
        dialog.close()


def test_board_tracking_preset_and_custom_round_trip(app):
    dialog = _dialog(app)
    try:
        dialog.track_marker_preset.setCurrentText("Comfort")
        assert dialog.track_marker_process.value() == pytest.approx(0.5)
        assert dialog.track_marker_angular.value() == pytest.approx(1.0)
        assert dialog.track_marker_output.value() == pytest.approx(6.0)
        dialog.track_marker_process.setValue(1.25)
        assert dialog.track_marker_preset.currentText() == "Custom"
    finally:
        dialog.close()


def test_settings_round_trips_stream_lens_and_openxr(app):
    dialog = _dialog(app)
    try:
        dialog.phone_codec.setCurrentText("av1")
        dialog.phone_encoder.setCurrentText("x264")
        dialog.phone_bitrate.setValue(65.0)
        dialog.phone_keyframe.setValue(0.5)
        dialog.phone_camera_width.setValue(1280)
        dialog.phone_camera_height.setValue(720)
        dialog.phone_camera_fps.setValue(90)
        dialog.lens_separation.setValue(64.0)
        dialog.lens_eye_distance.setValue(42.0)
        dialog.lens_fov.setValue(88.0)
        dialog.lens_k1.setValue(-0.12)
        dialog.lens_k2.setValue(0.03)
        dialog.lens_chromatic.setValue(0.01)
        dialog.lens_scale.setValue(0.98)
        dialog.lens_overscan.setValue(0.02)
        dialog.openxr_enabled.setChecked(False)
        dialog.openxr_manage.setChecked(False)
        dialog.openxr_remote_port.setValue(4555)
        dialog.openxr_capture_port.setValue(4556)
        dialog.openxr_view_width.setValue(2560)
        dialog.openxr_view_height.setValue(1440)
        dialog.openxr_ipd.setValue(66.0)
        cfg = dialog.result_config()
        assert cfg.phone.codec == "av1"
        assert cfg.phone.encoder == "x264"
        assert cfg.phone.bitrate_mbps == pytest.approx(65.0)
        assert cfg.phone.keyframe_interval_s == pytest.approx(0.5)
        assert (
            cfg.phone.camera_width,
            cfg.phone.camera_height,
            cfg.phone.camera_fps,
        ) == (1280, 720, 90)
        assert cfg.phone.lens.lens_separation_mm == pytest.approx(64.0)
        assert cfg.phone.lens.eye_screen_distance_mm == pytest.approx(42.0)
        assert cfg.phone.lens.fov_deg == pytest.approx(88.0)
        assert cfg.phone.lens.distortion_k1 == pytest.approx(-0.12)
        assert cfg.phone.lens.distortion_k2 == pytest.approx(0.03)
        assert cfg.phone.lens.chromatic_aberration == pytest.approx(0.01)
        assert cfg.phone.lens.image_scale == pytest.approx(0.98)
        assert cfg.phone.lens.overscan == pytest.approx(0.02)
        assert cfg.openxr.enabled is False
        assert cfg.openxr.manage_service is False
        assert cfg.openxr.remote_port == 4555
        assert cfg.openxr.capture_port == 4556
        assert (cfg.openxr.view_width, cfg.openxr.view_height) == (2560, 1440)
        assert cfg.openxr.ipd_m == pytest.approx(0.066)
    finally:
        dialog.close()


def test_main_window_settings_apply_does_not_block_the_event_loop(app, monkeypatch):
    import threading
    import time
    from pathlib import Path

    from openvrcardboard.ui import main_window as main_window_module

    runtime = Runtime(AppConfig())
    applied = threading.Event()

    def slow_apply(config):
        time.sleep(0.4)
        applied.set()
        return ["calibration loaded"]

    class AcceptedDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def result_config(self):
            return copy.deepcopy(AppConfig())

    monkeypatch.setattr(main_window_module, "SettingsDialog", AcceptedDialog)
    monkeypatch.setattr(runtime, "apply_config", slow_apply)
    monkeypatch.setattr(runtime, "save", lambda: Path("config.json"))

    window = MainWindow(runtime, start_tracking=False)
    window._initial_start = lambda: None
    window.show()
    try:
        begin = time.monotonic()
        window._open_settings()
        assert time.monotonic() - begin < 0.3
        assert window._busy
        assert not applied.is_set()

        deadline = time.monotonic() + 3.0
        while (not applied.is_set() or window._busy) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert applied.is_set()
        assert not window._busy
        assert "calibration loaded" in window.notes.toPlainText()
    finally:
        window.close()
        app.processEvents()
        runtime.stop()


def test_main_window_start_does_not_block_the_event_loop(app, monkeypatch):
    import threading
    import time

    runtime = Runtime(AppConfig())
    started = threading.Event()

    def slow_start(start_tracking=None):
        time.sleep(0.4)
        runtime._running = True
        started.set()
        return []

    monkeypatch.setattr(runtime, "start", slow_start)
    window = MainWindow(runtime, start_tracking=True)
    window.show()
    try:
        begin = time.monotonic()
        until = begin + 0.2
        while time.monotonic() < until:
            app.processEvents()
            time.sleep(0.01)
        # The window must stay responsive while the 400 ms startup runs.
        assert time.monotonic() - begin < 0.35
        assert window._busy
        assert not started.is_set()

        deadline = time.monotonic() + 3.0
        while not started.is_set() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert started.is_set()
        deadline = time.monotonic() + 3.0
        while window._busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not window._busy
    finally:
        window.close()
        app.processEvents()
        runtime.stop()
