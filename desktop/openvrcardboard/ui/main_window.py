"""Main window and settings dialog for the OpenVR Cardboard desktop app."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import ARUCO_DICTIONARIES, AppConfig, validate_config
from ..runtime import Runtime

LOGGER = logging.getLogger(__name__)

STATUS_STYLE = {
    "ok": "color: #7ee787;",
    "warn": "color: #f0b72f;",
    "bad": "color: #ff7b72;",
    "muted": "color: #9aa4b2;",
}

HELP_TEXT = """\
Markers: print the five-marker boards with scripts/generate_markers.py at 100%
scale and measure the black squares. Put board IDs 0-4 on the headset, 5-9 on
the left controller and 10-14 on the right controller. The large marker is the
central one; the four small markers must keep their printed spacing. Extra
body-tracker IDs start at 15. Keep the whole board and its white border
visible. If markers look tilted or distances are wrong, run Tools > Calibrate
camera with the printed chessboard.

Hands/body: MediaPipe tracks bare hands and a sparse body skeleton. Enable them
in Settings and allow a moment for the models to load. Inference runs on a
downscaled copy of the camera frame, so 1080p60 capture stays practical.

Phone: install the Android app, enter the desktop IP shown in the status panel
and use the same IMU/video ports. The phone sends IMU at sensor rate and
receives stereo video over WebSocket. With an NVIDIA GPU the desktop encodes
HEVC/AV1 with NVENC and the phone decodes it on its hardware decoder; without
one the always-available JPEG path is used.

OpenXR: leave "Start Monado service" enabled to run OpenXR applications through
the bundled runtime. SteamVR games keep using the native OpenVR driver.
"""


def _vector_text(values) -> str:
    return ", ".join(f"{float(value):g}" for value in values)


def _parse_vector(text: str, length: int) -> list[float]:
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    values = [float(part) for part in parts]
    if len(values) != length:
        raise ValueError(f"expected {length} comma-separated numbers")
    return values


SMOOTHING_PRESETS = {
    "Responsive": (4.0, 0.4, 7.0),
    "Balanced": (2.0, 0.25, 4.0),
    "Stable": (1.0, 0.15, 2.0),
}

# Board preset: (process noise, rotation noise, optional output smoothing).
# Prediction is deliberately absent: the driver owns the single configurable
# horizon.
BOARD_PRESETS = {
    "Comfort": (0.5, 1.0, 6.0),
    "Balanced": (1.0, 2.0, 0.0),
    "Responsive": (2.0, 4.0, 0.0),
}


def _smoothing_preset(cutoff: float, beta: float, orientation: float) -> str:
    for name, values in SMOOTHING_PRESETS.items():
        if all(abs(a - b) < 1e-6 for a, b in zip(values, (cutoff, beta, orientation))):
            return name
    return "Custom"


def _board_preset(
    process_noise: float,
    angular_noise: float,
    output_smoothing_hz: float,
) -> str:
    values = (process_noise, angular_noise, output_smoothing_hz)
    for name, preset in BOARD_PRESETS.items():
        if all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(preset, values)):
            return name
    return "Custom"


class MainWindow(QMainWindow):
    def __init__(self, runtime: Runtime, start_tracking: bool = True):
        super().__init__()
        self.runtime = runtime
        self._last_preview_sequence = -1
        self._last_notes: list[str] = []
        self._start_tracking = start_tracking

        self.setWindowTitle("OpenVR Cardboard")
        self.resize(1100, 720)
        self._build_ui()
        self._build_menu()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(250)
        QTimer.singleShot(0, self._initial_start)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("OpenVR Cardboard")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        self.driver_badge = QLabel("driver: waiting")
        self.driver_badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.driver_badge)
        outer.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(12)

        preview_group = QGroupBox("Camera preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("No camera frame")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(560, 315)
        self.preview_label.setStyleSheet(
            "background: #11151b; border: 1px solid #30363d; color: #8b949e;"
        )
        preview_layout.addWidget(self.preview_label, 1)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start tracking")
        self.stop_button = QPushButton("Stop tracking")
        self.recenter_button = QPushButton("Recenter")
        self.start_button.clicked.connect(self._start_clicked)
        self.stop_button.clicked.connect(self._stop_clicked)
        self.recenter_button.clicked.connect(self._recenter_clicked)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.recenter_button)
        buttons.addStretch(1)
        preview_layout.addLayout(buttons)
        content.addWidget(preview_group, 3)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        grid = QGridLayout()
        grid.setVerticalSpacing(6)
        self.status_labels: dict[str, QLabel] = {}
        rows = [
            ("camera", "Camera"),
            ("markers", "Markers"),
            ("hands", "Hands"),
            ("body", "Body"),
            ("phone", "Phone IMU"),
            ("gamepads", "Gamepads"),
            ("video", "Phone video"),
            ("driver", "SteamVR driver"),
        ]
        for row, (key, label) in enumerate(rows):
            name = QLabel(label)
            name.setStyleSheet("color: #9aa4b2;")
            value = QLabel("—")
            value.setWordWrap(True)
            grid.addWidget(name, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(value, row, 1)
            self.status_labels[key] = value
        grid.setColumnStretch(1, 1)
        status_layout.addLayout(grid)

        video_buttons = QHBoxLayout()
        self.video_button = QPushButton("Start phone video")
        self.video_button.clicked.connect(self._toggle_video)
        video_buttons.addWidget(self.video_button)
        status_layout.addLayout(video_buttons)

        status_layout.addStretch(1)
        content.addWidget(status_group, 2)
        outer.addLayout(content, 1)

        notes_group = QGroupBox("Log")
        notes_layout = QVBoxLayout(notes_group)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumBlockCount(200)
        self.notes.setPlaceholderText("Startup notes and warnings appear here.")
        notes_layout.addWidget(self.notes)
        outer.addWidget(notes_group, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        save_action = QAction("&Save configuration", self)
        save_action.triggered.connect(self._save_config)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        settings_action = QAction("&Settings…", self)
        settings_action.triggered.connect(self._open_settings)
        tools_menu.addAction(settings_action)
        recenter_action = QAction("&Recenter view", self)
        recenter_action.triggered.connect(self._recenter_clicked)
        tools_menu.addAction(recenter_action)
        calibrate_action = QAction("&Calibrate camera…", self)
        calibrate_action.triggered.connect(self._calibrate_camera)
        tools_menu.addAction(calibrate_action)
        markers_action = QAction("Generate printable &marker boards…", self)
        markers_action.triggered.connect(self._generate_marker_boards)
        tools_menu.addAction(markers_action)
        board_action = QAction("Generate printable &calibration board…", self)
        board_action.triggered.connect(self._generate_calibration_board)
        tools_menu.addAction(board_action)
        video_action = QAction("Toggle phone &video", self)
        video_action.triggered.connect(self._toggle_video)
        tools_menu.addAction(video_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About OpenVR Cardboard", self)
        about_action.triggered.connect(self._about)
        help_menu.addAction(about_action)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _initial_start(self) -> None:
        # The driver, phone links and gamepads always come up with the window.
        # ``--no-start`` / ui.start_tracking_on_launch only leave the camera
        # and tracking worker stopped.
        if self.runtime.running:
            problems = self.runtime.start_tracking() if self._start_tracking else []
        else:
            problems = self.runtime.start(start_tracking=self._start_tracking)
        self._report(problems, "Tracking started" if self._start_tracking else "Runtime started")
        self._refresh()

    def _start_clicked(self) -> None:
        if self.runtime.running:
            problems = self.runtime.start_tracking()
        else:
            problems = self.runtime.start(start_tracking=True)
        self._report(problems, "Tracking started")
        self._refresh()

    def _stop_clicked(self) -> None:
        if self.runtime.running:
            self.runtime.stop_tracking()
        self.statusBar().showMessage("Tracking stopped", 3000)
        self._refresh()

    def _report(self, problems: list[str], success: str) -> None:
        if problems:
            for problem in problems:
                self.statusBar().showMessage(problem, 8000)
        else:
            self.statusBar().showMessage(success, 3000)

    def _recenter_clicked(self) -> None:
        self.runtime.recenter()
        self.statusBar().showMessage("Recentered phone view", 2000)

    def _calibrate_camera(self) -> None:
        from .calibration_dialog import CalibrationDialog

        config = copy.deepcopy(self.runtime.config)
        was_tracking = self.runtime.tracking.running
        if was_tracking:
            self.runtime.stop_tracking()
        try:
            dialog = CalibrationDialog(
                config.camera,
                base_dir=self.runtime.base_dir,
                output=config.camera.calibration_file or None,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            if not dialog.calibration_file:
                return
            config.camera.calibration_file = str(dialog.calibration_file)
            config.camera.calibration_board_cols = dialog.board_cols.value()
            config.camera.calibration_board_rows = dialog.board_rows.value()
            config.camera.calibration_square_mm = dialog.square_mm.value()
            warnings = self.runtime.apply_config(config)
            self.runtime.save()
            if warnings:
                self.notes.appendPlainText("\n".join(warnings))
            self.statusBar().showMessage(
                f"Camera calibrated ({dialog.rms_error:.2f} px RMS)", 6000
            )
        except Exception as exc:
            QMessageBox.critical(self, "Calibrate camera", str(exc))
        finally:
            if was_tracking and not self.runtime.tracking.running:
                self.runtime.start_tracking()
            self._refresh()

    def _generate_marker_boards(self) -> None:
        from ..boards import boards_from_config
        from ..printables import PrintableError, generate_boards

        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the printable marker boards"
        )
        if not directory:
            return
        try:
            boards = boards_from_config(self.runtime.config.tracking)
            sheet = generate_boards(
                Path(directory),
                boards,
                self.runtime.config.camera.dictionary,
            )
        except (PrintableError, OSError) as exc:
            QMessageBox.critical(self, "Marker boards", str(exc))
            return
        QMessageBox.information(
            self,
            "Marker boards",
            f"Wrote {sheet}\n\nOpen it in a browser and print at 100% / actual "
            "size on A4/Letter landscape. Measure the 100 mm ruler before "
            "cutting.",
        )
        self.statusBar().showMessage(f"Marker boards written to {directory}", 5000)

    def _generate_calibration_board(self) -> None:
        from ..printables import PrintableError, generate_calibration_board

        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the printable calibration board"
        )
        if not directory:
            return
        camera = self.runtime.config.camera
        try:
            sheet = generate_calibration_board(
                Path(directory),
                camera.calibration_board_cols,
                camera.calibration_board_rows,
                camera.calibration_square_mm,
            )
        except (PrintableError, OSError) as exc:
            QMessageBox.critical(self, "Calibration board", str(exc))
            return
        QMessageBox.information(
            self,
            "Calibration board",
            f"Wrote {sheet}\n\nPrint at 100% / actual size on A4/Letter "
            "landscape and measure one square before calibrating.",
        )
        self.statusBar().showMessage(
            f"Calibration board written to {directory}", 5000
        )

    def _toggle_video(self) -> None:
        try:
            running = self.runtime.toggle_video()
            self.statusBar().showMessage(
                "Phone video started" if running else "Phone video stopped", 3000
            )
        except Exception as exc:
            QMessageBox.warning(self, "Phone video", str(exc))
        self._refresh()

    def _save_config(self) -> None:
        try:
            target = self.runtime.save()
            self.statusBar().showMessage(f"Saved {target}", 4000)
        except Exception as exc:
            QMessageBox.warning(self, "Save configuration", str(exc))

    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            copy.deepcopy(self.runtime.config),
            self,
            runtime=self.runtime,
            base_dir=self.runtime.base_dir,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            config = dialog.result_config()
            validate_config(config)
        except Exception as exc:
            QMessageBox.warning(self, "Settings", str(exc))
            return
        try:
            warnings = self.runtime.apply_config(config)
            self.runtime.save()
        except Exception as exc:
            QMessageBox.critical(self, "Settings", str(exc))
            return
        if warnings:
            self.notes.appendPlainText("\n".join(warnings))
        self.statusBar().showMessage("Settings applied", 4000)
        self._refresh()

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "About OpenVR Cardboard",
            "Webcam tracking, gamepad controllers and phone HMD streaming for OpenVR.\n\n"
            "Webcam marker/hand/body tracking, gamepad VR controllers, phone "
            "video and IMU streaming.\n\n" + HELP_TEXT,
        )

    # ------------------------------------------------------------------
    # Periodic refresh
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        try:
            status = self.runtime.status()
            tracking = self.runtime.tracking_status()
        except Exception as exc:
            self.statusBar().showMessage(f"Status error: {exc}", 5000)
            return

        connected = status.driver_connected
        self.driver_badge.setText(
            "driver: connected" if connected else "driver: waiting for SteamVR"
        )
        self.driver_badge.setStyleSheet(
            STATUS_STYLE["ok"] if connected else STATUS_STYLE["warn"]
        )

        camera = tracking["camera"]
        if camera["error"]:
            self._set_status("camera", camera["error"], "bad")
        elif camera["opened"]:
            detail = (
                f"{camera['fps']:.1f} fps · {camera['width']}×{camera['height']}"
                f" · {camera['frame_ms']:.1f} ms"
            )
            fourcc = camera.get("fourcc") or ""
            if fourcc:
                detail += f" · {fourcc}"
            if not camera.get("mode_ok", True):
                requested = (
                    f"{camera.get('requested_width', '?')}×"
                    f"{camera.get('requested_height', '?')}@"
                    f"{camera.get('requested_fps', '?')}"
                )
                detail += f" (requested {requested}; camera chose a lower mode)"
            inference = camera.get("inference_ms", 0.0)
            if inference:
                detail += f" · inference {inference:.1f} ms"
            if camera.get("auto_control"):
                exposure = camera.get("exposure", -1.0)
                gain = camera.get("gain", -1.0)
                detected = camera.get("detected_markers", 0)
                visible = camera.get("visible_markers", 0)
                detail += (
                    f" · auto exp {exposure:.1f} gain {gain:.1f}"
                    f" · markers {detected}/{max(visible, detected)}"
                    f" · brightness {camera.get('brightness', 0.0) * 100:.0f}%"
                )
            self._set_status("camera", detail, "warn" if not camera.get("mode_ok", True) else "ok")
        elif self.runtime.tracking.running:
            self._set_status("camera", "waiting for camera", "warn")
        else:
            self._set_status("camera", "tracking stopped", "muted")

        marker_count = status.marker_count
        marker_info = tracking.get("marker_status") or {}
        if marker_count:
            detail = f"{marker_count}/3 boards visible"
            if marker_info.get("detection_ms"):
                detail += f" · {marker_info['detection_ms']:.1f} ms"
            states = [
                f"{board.get('name', '?')}:{board.get('state', '?')}"
                for board in marker_info.get("boards", [])
            ]
            if states:
                detail += " · " + ", ".join(states)
            degraded = any(
                board.get("state") != "TRACKING"
                for board in marker_info.get("boards", [])
            )
            self._set_status("markers", detail, "warn" if degraded else "ok")
        elif not tracking.get("markers"):
            self._set_status("markers", "disabled", "muted")
        elif marker_info.get("error"):
            self._set_status("markers", marker_info["error"], "bad")
        elif not self.runtime.tracking.running:
            self._set_status("markers", "tracking stopped", "muted")
        else:
            self._set_status(
                "markers", f"none visible · {marker_info.get('dictionary', 'ArUco')}", "muted"
            )

        hands_info = tracking.get("hand_status") or {}
        if status.hands_visible:
            detail = "visible"
            if hands_info.get("delegate"):
                detail += f" · {hands_info['delegate'].upper()}"
            if hands_info.get("inference_ms"):
                detail += f" · {hands_info['inference_ms']:.1f} ms"
            self._set_status("hands", detail, "ok")
        elif not tracking.get("hands"):
            self._set_status("hands", "disabled", "muted")
        elif hands_info.get("error"):
            self._set_status("hands", hands_info["error"], "bad")
        else:
            self._set_status("hands", "not visible", "muted")

        body_info = tracking.get("body_status") or {}
        if status.body_visible:
            detail = "visible"
            if body_info.get("model"):
                detail += f" · {body_info['model']}"
            if body_info.get("delegate"):
                detail += f" · {body_info['delegate'].upper()}"
            if body_info.get("inference_ms"):
                detail += f" · {body_info['inference_ms']:.1f} ms"
            self._set_status("body", detail, "ok")
        elif not tracking.get("body"):
            self._set_status("body", "disabled", "muted")
        elif body_info.get("error"):
            self._set_status("body", body_info["error"], "bad")
        else:
            self._set_status("body", "not visible", "muted")

        if status.phone_address:
            accuracy = "accurate" if status.phone_accurate else "low accuracy"
            self._set_status(
                "phone",
                f"{status.phone_address} · {status.phone_rate_hz:.1f} Hz · {accuracy}",
                "ok" if status.phone_rate_hz > 5 else "warn",
            )
        else:
            self._set_status("phone", "waiting for phone", "muted")

        gamepads = status.gamepads
        self._set_status(
            "gamepads",
            " · ".join(gamepads) if gamepads else "none detected",
            "ok" if gamepads else "warn",
        )

        if status.video_clients:
            self._set_status(
                "video",
                f"{status.video_clients} client(s) · {status.video_fps:.1f} fps · "
                f"{status.video_frames} frames",
                "ok",
            )
        elif status.video_last_error:
            self._set_status("video", status.video_last_error, "bad")
        elif self.runtime.video is not None:
            self._set_status("video", "stream running, no phone", "warn")
        else:
            self._set_status("video", "stopped", "muted")

        link_status = status.driver_status or {}
        if connected and link_status:
            self._set_status(
                "driver",
                f"connected · {link_status.get('device_count', '?')} devices · "
                f"{status.driver_packets} state packets",
                "ok",
            )
        elif connected:
            self._set_status("driver", "connected", "ok")
        else:
            self._set_status(
                "driver",
                f"waiting · {status.driver_packets} packets sent",
                "warn",
            )

        self.video_button.setText(
            "Stop phone video" if self.runtime.video is not None else "Start phone video"
        )
        self.start_button.setEnabled(not self.runtime.tracking.running)
        self.stop_button.setEnabled(self.runtime.tracking.running)

        if status.notes != self._last_notes:
            self._last_notes = list(status.notes)
            self.notes.setPlainText("\n".join(status.notes))

        self._refresh_preview()

    def _set_status(self, key: str, text: str, style: str) -> None:
        label = self.status_labels.get(key)
        if label is None:
            return
        label.setText(text)
        label.setStyleSheet(STATUS_STYLE.get(style, ""))

    def _refresh_preview(self) -> None:
        image, _, sequence = self.runtime.state.preview()
        if image is None or sequence == self._last_preview_sequence:
            return
        self._last_preview_sequence = sequence
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 3:
            rgb = np.ascontiguousarray(array[:, :, ::-1])
            height, width = rgb.shape[:2]
            qt_image = QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
        elif array.ndim == 2:
            height, width = array.shape[:2]
            qt_image = QImage(
                np.ascontiguousarray(array).data,
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            ).copy()
        else:
            return
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        try:
            self.runtime.stop()
        finally:
            event.accept()


class SettingsDialog(QDialog):
    """Edits the main configuration sections."""

    def __init__(
        self,
        config: AppConfig,
        parent: Optional[QWidget] = None,
        runtime: Optional[Runtime] = None,
        base_dir: Optional[Path] = None,
    ):
        super().__init__(parent)
        self.config = config
        self.runtime = runtime
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.setWindowTitle("OpenVR Cardboard settings")
        self.resize(620, 560)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._camera_tab(), "Camera && markers")
        tabs.addTab(self._boards_tab(), "Boards")
        tabs.addTab(self._tracking_tab(), "Tracking")
        tabs.addTab(self._phone_tab(), "Phone video")
        tabs.addTab(self._lens_tab(), "Lens && display")
        tabs.addTab(self._gamepad_tab(), "Gamepad")
        tabs.addTab(self._driver_tab(), "Driver")
        tabs.addTab(self._openxr_tab(), "OpenXR")
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- widget helpers -------------------------------------------------
    def _spin(self, value: int, low: int, high: int, suffix: str = "") -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(low, high)
        widget.setValue(int(value))
        if suffix:
            widget.setSuffix(suffix)
        return widget

    def _double(self, value: float, low: float, high: float, suffix: str = "", decimals: int = 3) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(decimals)
        widget.setValue(float(value))
        if suffix:
            widget.setSuffix(suffix)
        return widget

    def _tab(self) -> tuple[QWidget, QFormLayout]:
        widget = QWidget()
        form = QFormLayout(widget)
        return widget, form

    # -- tabs -----------------------------------------------------------
    def _camera_tab(self) -> QWidget:
        cfg = self.config.camera
        widget, form = self._tab()
        self.camera_source = QComboBox()
        self.camera_source.addItems(["webcam", "synthetic"])
        self.camera_source.setCurrentText(cfg.source)
        self.camera_index = self._spin(cfg.index, 0, 32)
        self.camera_width = self._spin(cfg.width, 320, 7680)
        self.camera_height = self._spin(cfg.height, 240, 4320)
        self.camera_fps = self._spin(cfg.fps, 5, 240, " fps")
        self.camera_fov = self._double(cfg.horizontal_fov_deg, 10.0, 150.0, "°", 1)
        self.camera_max_error = self._double(cfg.max_reprojection_error, 0.1, 50.0, " px", 1)
        self.camera_dictionary = QComboBox()
        self.camera_dictionary.addItems(ARUCO_DICTIONARIES)
        self.camera_dictionary.setCurrentText(cfg.dictionary)
        self.camera_dictionary.setToolTip(
            "The dictionary must match the printed boards. DICT_3X3_15 is the "
            "default: its larger modules survive blur and distance better, and "
            "its fifteen codes cover the HMD and both hands exactly."
        )
        self.camera_detection_width = self._spin(
            cfg.marker_detection_width, 0, 3840, " px"
        )
        self.camera_detection_width.setSpecialValueText("full resolution")
        self.camera_detection_width.setToolTip(
            "Marker detection runs on this width and corners are refined at "
            "capture resolution. Full resolution is the most accurate but "
            "slower; 960 px is a good 720p/1080p default."
        )
        self.camera_refine_pose = QCheckBox("Refine board pose with Levenberg–Marquardt")
        self.camera_refine_pose.setChecked(cfg.marker_refine_pose)
        self.camera_calibration = QLineEdit(cfg.calibration_file)
        self.camera_calibrate = QPushButton("Calibrate camera…")
        self.camera_calibrate.setToolTip(
            "Run the guided chessboard calibration to measure lens distortion "
            "and focal length. Recalibrate whenever the camera, resolution or "
            "focus distance changes."
        )
        self.camera_calibrate.clicked.connect(self._calibrate_clicked)
        self.camera_position = QLineEdit(_vector_text(cfg.position))
        self.camera_rotation = QLineEdit(_vector_text(cfg.rotation))
        self.camera_fourcc = QLineEdit(cfg.fourcc)
        self.camera_fourcc.setPlaceholderText("MJPG, YUYV, or blank")
        self.camera_backend = QComboBox()
        self.camera_backend.addItems(["auto", "v4l2", "dshow", "msmf", "any"])
        self.camera_backend.setCurrentText(cfg.backend)
        self.camera_allow_fallback = QCheckBox("Accept a lower camera mode if the requested one is unsupported")
        self.camera_allow_fallback.setChecked(cfg.allow_mode_fallback)
        self.camera_verify_mode = QCheckBox("Read back the negotiated mode after opening")
        self.camera_verify_mode.setChecked(cfg.verify_mode)
        self.camera_autofocus = QCheckBox("Autofocus")
        self.camera_autofocus.setChecked(cfg.autofocus)
        self.camera_focus = self._double(cfg.focus, -1.0, 500.0, "", 1)
        self.camera_auto_exposure = QCheckBox("Camera automatic exposure")
        self.camera_auto_exposure.setChecked(cfg.auto_exposure)
        self.camera_exposure = self._double(cfg.exposure, -13.0, 13.0, "", 2)
        self.camera_gain = self._double(cfg.gain, -1.0, 100.0, "", 1)
        self.camera_auto_marker = QCheckBox("Marker-aware exposure and gain")
        self.camera_auto_marker.setChecked(cfg.auto_marker_control)
        self.camera_auto_marker.setToolTip(
            "Shortens exposure for the least motion blur that still detects "
            "every visible marker, then raises gain only as far as needed. "
            "Disable for full manual control."
        )
        self.camera_aggressiveness = self._double(
            cfg.camera_aggressiveness, 0.0, 1.0, "", 2
        )
        self.camera_aggressiveness.setToolTip(
            "How quickly the marker-aware loop changes exposure and gain. "
            "0 disables adaptation; 1 reacts within a few frames."
        )
        self.camera_target_brightness = self._double(
            cfg.camera_target_brightness, 0.1, 0.9, "", 2
        )
        self.camera_max_gain = self._double(cfg.camera_max_gain, 0.0, 1024.0, "", 1)
        self.camera_timestamp_offset = self._double(
            cfg.camera_timestamp_offset_ms, -1000.0, 1000.0, " ms", 1
        )
        self.camera_timestamp_source = QComboBox()
        self.camera_timestamp_source.addItems(["auto", "pos_msec", "read"])
        self.camera_timestamp_source.setCurrentText(cfg.camera_timestamp_source)
        self.camera_auto_wb = QCheckBox("Automatic white balance")
        self.camera_auto_wb.setChecked(cfg.auto_white_balance)
        self.camera_white_balance = self._double(cfg.white_balance, 1000.0, 10000.0, " K", 0)
        self.camera_mirror = QCheckBox("Mirror camera preview")
        self.camera_mirror.setChecked(cfg.mirror_preview)
        self.camera_board_cols = self._spin(cfg.calibration_board_cols, 3, 20, " cols")
        self.camera_board_rows = self._spin(cfg.calibration_board_rows, 3, 20, " rows")
        self.camera_board_square = self._double(
            cfg.calibration_square_mm, 5.0, 200.0, " mm", 1
        )
        self.camera_min_views = self._spin(cfg.calibration_min_views, 4, 60)
        self.camera_board_cols.setToolTip("Inner corners of the printed chessboard (not squares).")
        self.camera_board_rows.setToolTip("Inner corners of the printed chessboard (not squares).")
        self.camera_board_square.setToolTip("Measured square edge length on the printed board.")
        form.addRow("Camera source", self.camera_source)
        form.addRow("Camera index", self.camera_index)
        form.addRow("Capture size", self._pair(self.camera_width, self.camera_height))
        form.addRow("Capture rate", self.camera_fps)
        form.addRow("Pixel format (FourCC)", self.camera_fourcc)
        form.addRow("Backend", self.camera_backend)
        form.addRow("", self.camera_verify_mode)
        form.addRow("", self.camera_allow_fallback)
        form.addRow("", self.camera_autofocus)
        form.addRow("Focus (manual)", self.camera_focus)
        form.addRow("", self.camera_auto_exposure)
        form.addRow("Exposure (manual)", self.camera_exposure)
        form.addRow("Gain", self.camera_gain)
        form.addRow("", self.camera_auto_marker)
        form.addRow("Adaptation aggressiveness", self.camera_aggressiveness)
        form.addRow("Target brightness", self.camera_target_brightness)
        form.addRow("Maximum auto gain", self.camera_max_gain)
        form.addRow("Timestamp source", self.camera_timestamp_source)
        form.addRow("Timestamp offset", self.camera_timestamp_offset)
        form.addRow("", self.camera_auto_wb)
        form.addRow("White balance (manual)", self.camera_white_balance)
        form.addRow("Horizontal FOV (estimate)", self.camera_fov)
        form.addRow("ArUco dictionary", self.camera_dictionary)
        form.addRow("Marker detection width", self.camera_detection_width)
        form.addRow("", self.camera_refine_pose)
        form.addRow("Max marker reprojection error", self.camera_max_error)
        form.addRow("Calibration file", self.camera_calibration)
        form.addRow("", self.camera_calibrate)
        form.addRow("Calibration board inner corners", self._pair(self.camera_board_cols, self.camera_board_rows))
        form.addRow("Calibration square size", self.camera_board_square)
        form.addRow("Minimum calibration views", self.camera_min_views)
        form.addRow("Camera position (x, y, z)", self.camera_position)
        form.addRow("Camera rotation (x, y, z, w)", self.camera_rotation)
        form.addRow("", self.camera_mirror)
        return widget

    def _pair(self, left: QWidget, right: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(left)
        layout.addWidget(right)
        return container

    def _boards_tab(self) -> QWidget:
        cfg = self.config.tracking
        widget = QWidget()
        outer = QVBoxLayout(widget)
        self.board_widgets: dict[str, dict[str, QWidget]] = {}
        for name, label in (("hmd", "HMD"), ("left", "Left hand"), ("right", "Right hand")):
            board = getattr(cfg.boards, name)
            group = QGroupBox(f"{label} board (IDs {getattr(cfg, f'{name}_marker_id')}-{getattr(cfg, f'{name}_marker_id') + 4})")
            form = QFormLayout(group)
            fields = {
                "width": self._double(board.width_m * 1000.0, 50.0, 3000.0, " mm", 1),
                "height": self._double(board.height_m * 1000.0, 30.0, 2000.0, " mm", 1),
                "central": self._double(board.central_marker_m * 1000.0, 5.0, 500.0, " mm", 1),
                "peripheral": self._double(board.peripheral_marker_m * 1000.0, 4.0, 500.0, " mm", 1),
                "peripheral_x": self._double(board.peripheral_x_m * 1000.0, 5.0, 1500.0, " mm", 1),
                "peripheral_y": self._double(board.peripheral_y_m * 1000.0, 4.0, 1000.0, " mm", 1),
                "object_position": QLineEdit(_vector_text(board.object_position)),
                "object_rotation": QLineEdit(_vector_text(board.object_rotation)),
            }
            fields["object_position"].setToolTip(
                "Board origin in the tracked object frame, in metres."
            )
            fields["object_rotation"].setToolTip(
                "Board orientation in the tracked object frame, quaternion x, y, z, w."
            )
            form.addRow("Plate width", fields["width"])
            form.addRow("Plate height", fields["height"])
            form.addRow("Central marker", fields["central"])
            form.addRow("Peripheral marker", fields["peripheral"])
            form.addRow("Peripheral X", fields["peripheral_x"])
            form.addRow("Peripheral Y", fields["peripheral_y"])
            form.addRow("Object offset (m)", fields["object_position"])
            form.addRow("Object rotation", fields["object_rotation"])
            self.board_widgets[name] = fields
            outer.addWidget(group)
        note = QLabel(
            "Print at 100% scale on A4/Letter landscape. The validator rejects "
            "layouts where markers overlap or leave the plate, and falls back "
            "to the default geometry."
        )
        note.setWordWrap(True)
        outer.addWidget(note)
        outer.addStretch(1)
        return widget

    def _tracking_tab(self) -> QWidget:
        cfg = self.config.tracking
        widget, form = self._tab()
        self.track_markers = QCheckBox("ArUco marker tracking")
        self.track_markers.setChecked(cfg.markers_enabled)
        self.track_markers.setToolTip(
            "Detects square ArUco markers, refines their corners to sub-pixel "
            "accuracy and solves a full 6DoF pose. Print markers with "
            "scripts/generate_markers.py and attach them to the headset and "
            "controllers."
        )
        self.track_hands = QCheckBox("MediaPipe hand tracking")
        self.track_hands.setChecked(cfg.hands_enabled)
        self.track_body = QCheckBox("MediaPipe body tracking")
        self.track_body.setChecked(cfg.body_enabled)
        self.track_hand_inputs = QCheckBox("Hand pinch/grip drives controller inputs")
        self.track_hand_inputs.setChecked(cfg.hand_inputs)
        self.track_hand_swap = QCheckBox("Swap left/right hands")
        self.track_hand_swap.setChecked(cfg.hand_swap)
        self.track_pinch = self._double(cfg.hand_trigger_pinch, 0.05, 0.95, "", 2)
        self.track_mediapipe_fps = self._spin(cfg.mediapipe_max_fps, 1, 120, " fps")
        self.track_inference_width = self._spin(cfg.inference_width, 0, 4096, " px")
        self.track_inference_height = self._spin(cfg.inference_height, 0, 4096, " px")
        self.track_body_model = QComboBox()
        self.track_body_model.addItems(["lite", "full", "heavy"])
        self.track_body_model.setCurrentText(cfg.body_model)
        self.track_body_model.setToolTip(
            "Pose model size: lite is fastest, heavy is most accurate. "
            "The hand model has a single rate."
        )
        self.track_delegate = QComboBox()
        self.track_delegate.addItems(["auto", "gpu", "cpu"])
        self.track_delegate.setCurrentText(cfg.mediapipe_delegate)
        self.track_delegate.setToolTip(
            "auto prefers the GPU delegate and always falls back to CPU."
        )
        self.track_model_dir = QLineEdit(cfg.model_dir)
        self.track_model_dir.setPlaceholderText(
            "empty = per-user cache or models bundled with the app"
        )
        self.track_smoothing_preset = QComboBox()
        self.track_smoothing_preset.addItems(
            ["Balanced", "Responsive", "Stable", "Custom"]
        )
        self.track_smoothing_preset.setToolTip(
            "Presets only change the smoothing fields below; the marker "
            "detector itself is not masked by extra filtering."
        )
        self.track_smoothing_preset.setCurrentText(
            _smoothing_preset(cfg.smoothing_min_cutoff, cfg.smoothing_beta, cfg.orientation_cutoff)
        )
        self._updating_smoothing = False
        self.track_smoothing_preset.currentTextChanged.connect(self._apply_smoothing_preset)
        self.track_roles = QLineEdit(", ".join(cfg.tracker_roles))
        self.track_roles.setPlaceholderText("waist, chest, left_foot, right_foot")
        self.track_hmd_source = QComboBox()
        self.track_hmd_source.addItems(["auto", "markers", "hands", "body", "phone", "manual"])
        self.track_hmd_source.setCurrentText(cfg.hmd_source)
        self.track_left_source = QComboBox()
        self.track_left_source.addItems(["auto", "markers", "hands", "body", "synthetic"])
        self.track_left_source.setCurrentText(cfg.left_source)
        self.track_right_source = QComboBox()
        self.track_right_source.addItems(["auto", "markers", "hands", "body", "synthetic"])
        self.track_right_source.setCurrentText(cfg.right_source)
        self.track_hmd_marker = self._spin(cfg.hmd_marker_id, 0, 999)
        self.track_left_marker = self._spin(cfg.left_marker_id, 0, 999)
        self.track_right_marker = self._spin(cfg.right_marker_id, 0, 999)
        self.track_smoothing = self._double(cfg.smoothing_min_cutoff, 0.0, 100.0, " Hz", 2)
        self.track_beta = self._double(cfg.smoothing_beta, 0.0, 10.0, "", 3)
        self.track_orientation = self._double(cfg.orientation_cutoff, 0.0, 100.0, " Hz", 2)
        self.track_pose_prediction = self._spin(cfg.pose_prediction_ms, 0, 200, " ms")
        self.track_pose_prediction.setToolTip(
            "The single prediction horizon, applied by the driver through "
            "poseTimeOffset. 0 sends poses exactly as measured."
        )
        self.track_marker_preset = QComboBox()
        self.track_marker_preset.addItems(
            ["Balanced", "Comfort", "Responsive", "Custom"]
        )
        self.track_marker_preset.setToolTip(
            "Comfort uses a quieter board filter and a little output smoothing "
            "for motion-sensitive users. Responsive follows fast games with "
            "higher process noise. Prediction is the driver horizon below."
        )
        self.track_marker_preset.setCurrentText(
            _board_preset(
                cfg.marker_process_noise,
                cfg.marker_angular_noise,
                cfg.marker_output_smoothing_hz,
            )
        )
        self._updating_board = False
        self.track_marker_preset.currentTextChanged.connect(
            self._apply_board_preset
        )
        self.track_marker_process = self._double(
            cfg.marker_process_noise, 0.05, 100.0, "", 2
        )
        self.track_marker_angular = self._double(
            cfg.marker_angular_noise, 0.05, 400.0, "", 2
        )
        self.track_marker_output = self._double(
            cfg.marker_output_smoothing_hz, 0.0, 60.0, " Hz", 1
        )
        self.track_marker_adaptive = QCheckBox("Adaptive board noise at rest")
        self.track_marker_adaptive.setChecked(cfg.marker_adaptive_noise)
        self.track_marker_adaptive.setToolTip(
            "Uses the full process noise only while the board moves. "
            "Disable only when comparing manual filter settings."
        )
        self.track_marker_static_scale = self._double(
            cfg.marker_static_noise_scale, 0.0, 1.0, "", 2
        )
        self.track_marker_static_scale.setToolTip(
            "Process-noise fraction used when the board is still. "
            "Lower values give a calmer image but react more slowly to "
            "the first motion. 1.0 disables the reduction."
        )
        self.track_marker_hold = self._double(cfg.marker_hold_s, 0.02, 2.0, " s", 2)
        self.track_marker_lost = self._double(cfg.marker_lost_s, 0.05, 5.0, " s", 2)
        self.track_marker_reacquire = self._spin(cfg.marker_reacquire_frames, 1, 30)
        self.track_marker_min_markers = self._spin(cfg.marker_min_markers, 1, 5)
        self.track_marker_confidence = self._double(cfg.marker_min_confidence, 0.0, 1.0, "", 2)
        self.track_body_visibility = self._double(cfg.body_min_visibility, 0.0, 1.0, "", 2)
        self.track_landmark = self._double(cfg.landmark_cutoff, 0.0, 100.0, " Hz", 2)
        self.track_landmark_beta = self._double(cfg.landmark_beta, 0.0, 10.0, "", 3)
        self.track_body_scale = self._double(cfg.body_scale, 0.2, 3.0, "", 2)
        for field_widget in (
            self.track_smoothing,
            self.track_beta,
            self.track_orientation,
        ):
            field_widget.valueChanged.connect(self._smoothing_edited)
        for field_widget in (
            self.track_marker_process,
            self.track_marker_angular,
            self.track_marker_output,
        ):
            field_widget.valueChanged.connect(self._board_edited)
        form.addRow(self.track_markers)
        form.addRow(self.track_hands)
        form.addRow(self.track_body)
        form.addRow("HMD source", self.track_hmd_source)
        form.addRow("Left controller source", self.track_left_source)
        form.addRow("Right controller source", self.track_right_source)
        form.addRow("HMD marker ID", self.track_hmd_marker)
        form.addRow("Left marker ID", self.track_left_marker)
        form.addRow("Right marker ID", self.track_right_marker)
        form.addRow("Camera smoothing preset", self.track_smoothing_preset)
        form.addRow("Position smoothing", self.track_smoothing)
        form.addRow("Smoothing beta", self.track_beta)
        form.addRow("Orientation smoothing", self.track_orientation)
        form.addRow("Driver pose prediction", self.track_pose_prediction)
        form.addRow("Board tracking preset", self.track_marker_preset)
        form.addRow("Board process noise", self.track_marker_process)
        form.addRow("Board rotation noise", self.track_marker_angular)
        form.addRow("Board output smoothing", self.track_marker_output)
        form.addRow("", self.track_marker_adaptive)
        form.addRow("Idle noise scale", self.track_marker_static_scale)
        form.addRow("Board hold time", self.track_marker_hold)
        form.addRow("Board lost timeout", self.track_marker_lost)
        form.addRow("Reacquisition frames", self.track_marker_reacquire)
        form.addRow("Minimum board markers", self.track_marker_min_markers)
        form.addRow("Minimum board confidence", self.track_marker_confidence)
        form.addRow("Body min visibility", self.track_body_visibility)
        form.addRow("Landmark smoothing", self.track_landmark)
        form.addRow("Landmark beta", self.track_landmark_beta)
        form.addRow("Body scale", self.track_body_scale)
        form.addRow("MediaPipe rate", self.track_mediapipe_fps)
        form.addRow("Inference size (0 = no limit)", self._pair(self.track_inference_width, self.track_inference_height))
        form.addRow("Body tracker roles", self.track_roles)
        form.addRow("Pose model", self.track_body_model)
        form.addRow("MediaPipe delegate", self.track_delegate)
        form.addRow("Model directory", self.track_model_dir)
        form.addRow("Pinch trigger threshold", self.track_pinch)
        form.addRow("", self.track_hand_inputs)
        form.addRow("", self.track_hand_swap)
        return widget

    def _apply_smoothing_preset(self, name: str) -> None:
        values = SMOOTHING_PRESETS.get(name)
        if values is None:
            return
        self._updating_smoothing = True
        try:
            for widget, value in zip(
                (self.track_smoothing, self.track_beta, self.track_orientation),
                values,
            ):
                widget.setValue(value)
        finally:
            self._updating_smoothing = False

    def _smoothing_edited(self, _value=None) -> None:
        if self._updating_smoothing:
            return
        name = _smoothing_preset(
            self.track_smoothing.value(),
            self.track_beta.value(),
            self.track_orientation.value(),
        )
        if self.track_smoothing_preset.currentText() != name:
            self._updating_smoothing = True
            try:
                self.track_smoothing_preset.setCurrentText(name)
            finally:
                self._updating_smoothing = False

    def _apply_board_preset(self, name: str) -> None:
        values = BOARD_PRESETS.get(name)
        if values is None:
            return
        self._updating_board = True
        try:
            for widget, value in zip(
                (
                    self.track_marker_process,
                    self.track_marker_angular,
                    self.track_marker_output,
                ),
                values,
            ):
                widget.setValue(value)
        finally:
            self._updating_board = False

    def _board_edited(self, _value=None) -> None:
        if self._updating_board:
            return
        name = _board_preset(
            self.track_marker_process.value(),
            self.track_marker_angular.value(),
            self.track_marker_output.value(),
        )
        if self.track_marker_preset.currentText() != name:
            self._updating_board = True
            try:
                self.track_marker_preset.setCurrentText(name)
            finally:
                self._updating_board = False

    def _calibrate_clicked(self) -> None:
        if self.runtime is None:
            QMessageBox.information(
                self,
                "Calibrate camera",
                "Open the settings from the main window to run the camera "
                "calibration wizard.",
            )
            return
        from .calibration_dialog import CalibrationDialog

        was_running = self.runtime.tracking.running
        if was_running:
            self.runtime.stop_tracking()
        try:
            dialog = CalibrationDialog(
                copy.deepcopy(self.config.camera),
                base_dir=self.base_dir,
                output=self.camera_calibration.text().strip() or None,
                parent=self,
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if dialog.calibration_file:
                    self.camera_calibration.setText(str(dialog.calibration_file))
                self.camera_board_cols.setValue(dialog.board_cols)
                self.camera_board_rows.setValue(dialog.board_rows)
                self.camera_board_square.setValue(dialog.square_mm)
        except Exception as exc:
            QMessageBox.critical(self, "Calibrate camera", str(exc))
        finally:
            if was_running:
                self.runtime.start_tracking()

    def _phone_tab(self) -> QWidget:
        cfg = self.config.phone
        widget, form = self._tab()
        self.phone_video_enabled = QCheckBox("Enable phone video streaming")
        self.phone_video_enabled.setChecked(cfg.video_enabled)
        self.phone_stream_launch = QCheckBox("Start streaming on launch")
        self.phone_stream_launch.setChecked(cfg.stream_on_launch)
        self.phone_imu_port = self._spin(cfg.imu_port, 1, 65535)
        self.phone_video_port = self._spin(cfg.video_port, 1, 65535)
        self.phone_token = QLineEdit(cfg.token)
        self.phone_token.setPlaceholderText("optional pairing token")
        self.phone_layout = QComboBox()
        self.phone_layout.addItems(["sbs", "mono", "top_bottom"])
        self.phone_layout.setCurrentText(cfg.layout)
        self.phone_width = self._spin(cfg.width, 160, 3840)
        self.phone_height = self._spin(cfg.height, 120, 2160)
        self.phone_fps = self._spin(cfg.fps, 1, 120, " fps")
        self.phone_quality = self._spin(cfg.jpeg_quality, 20, 95)
        self.phone_codec = QComboBox()
        self.phone_codec.addItems(["auto", "jpeg", "hevc", "av1", "h264"])
        self.phone_codec.setCurrentText(cfg.codec)
        self.phone_encoder = QComboBox()
        self.phone_encoder.addItems(["auto", "nvenc", "x264"])
        self.phone_encoder.setCurrentText(cfg.encoder)
        self.phone_bitrate = self._double(cfg.bitrate_mbps, 2.0, 400.0, " Mbps", 1)
        self.phone_keyframe = self._double(cfg.keyframe_interval_s, 0.1, 10.0, " s", 2)
        self.phone_source = QComboBox()
        self.phone_source.addItems(["test", "screen", "camera", "game", "openxr"])
        self.phone_source.setCurrentText(cfg.capture_source)
        self.phone_pattern = QComboBox()
        self.phone_pattern.addItems(["sbs", "grid", "bars"])
        self.phone_pattern.setCurrentText(cfg.test_pattern)
        self.phone_monitor = self._spin(cfg.capture_monitor, 0, 16)
        self.phone_camera_index = self._spin(cfg.camera_index, 0, 32)
        form.addRow(self.phone_video_enabled)
        form.addRow(self.phone_stream_launch)
        form.addRow("IMU UDP port", self.phone_imu_port)
        form.addRow("Video WebSocket port", self.phone_video_port)
        form.addRow("Pairing token", self.phone_token)
        form.addRow("Stereo layout", self.phone_layout)
        form.addRow("Per-eye width / height", self._pair(self.phone_width, self.phone_height))
        form.addRow("Stream rate", self.phone_fps)
        form.addRow("Video codec", self.phone_codec)
        form.addRow("Encoder backend", self.phone_encoder)
        form.addRow("Target bitrate", self.phone_bitrate)
        form.addRow("Keyframe interval", self.phone_keyframe)
        form.addRow("JPEG quality (fallback)", self.phone_quality)
        form.addRow("Capture source", self.phone_source)
        form.addRow("Test pattern", self.phone_pattern)
        form.addRow("Monitor index", self.phone_monitor)
        form.addRow("Camera index (desktop capture)", self.phone_camera_index)
        return widget

    def _lens_tab(self) -> QWidget:
        cfg = self.config.phone.lens
        widget, form = self._tab()
        self.lens_separation = self._double(cfg.lens_separation_mm, 0.0, 150.0, " mm", 1)
        self.lens_eye_distance = self._double(cfg.eye_screen_distance_mm, 10.0, 150.0, " mm", 1)
        self.lens_screen_width = self._double(cfg.screen_width_mm, 0.0, 500.0, " mm", 1)
        self.lens_screen_height = self._double(cfg.screen_height_mm, 0.0, 500.0, " mm", 1)
        self.lens_fov = self._double(cfg.fov_deg, 30.0, 150.0, "°", 1)
        self.lens_k1 = self._double(cfg.distortion_k1, -1.0, 1.0, "", 3)
        self.lens_k2 = self._double(cfg.distortion_k2, -1.0, 1.0, "", 3)
        self.lens_chromatic = self._double(cfg.chromatic_aberration, -0.2, 0.2, "", 3)
        self.lens_scale = self._double(cfg.image_scale, 0.5, 2.0, "", 3)
        self.lens_offset_x = self._double(cfg.eye_offset_x, -0.5, 0.5, "", 3)
        self.lens_offset_y = self._double(cfg.eye_offset_y, -0.5, 0.5, "", 3)
        self.lens_overscan = self._double(cfg.overscan, 0.0, 1.0, "", 3)
        form.addRow("Lens separation", self.lens_separation)
        form.addRow("Eye-to-screen distance", self.lens_eye_distance)
        form.addRow("Panel width (0 = phone reports it)", self.lens_screen_width)
        form.addRow("Panel height (0 = phone reports it)", self.lens_screen_height)
        form.addRow("Nominal field of view", self.lens_fov)
        form.addRow("Barrel distortion k1", self.lens_k1)
        form.addRow("Barrel distortion k2", self.lens_k2)
        form.addRow("Chromatic aberration", self.lens_chromatic)
        form.addRow("Image scale", self.lens_scale)
        form.addRow("Eye offset X", self.lens_offset_x)
        form.addRow("Eye offset Y", self.lens_offset_y)
        form.addRow("Overscan", self.lens_overscan)
        return widget

    def _openxr_tab(self) -> QWidget:
        cfg = self.config.openxr
        widget, form = self._tab()
        self.openxr_enabled = QCheckBox("Run the bundled OpenXR runtime (Monado)")
        self.openxr_enabled.setChecked(cfg.enabled)
        self.openxr_manage = QCheckBox("Start the Monado service with the desktop app")
        self.openxr_manage.setChecked(cfg.manage_service)
        self.openxr_remote_host = QLineEdit(cfg.remote_host)
        self.openxr_remote_port = self._spin(cfg.remote_port, 1, 65535)
        self.openxr_capture_host = QLineEdit(cfg.capture_host)
        self.openxr_capture_port = self._spin(cfg.capture_port, 1, 65535)
        self.openxr_rate = self._spin(cfg.send_rate_hz, 10, 500, " Hz")
        self.openxr_view_width = self._spin(cfg.view_width, 320, 7680)
        self.openxr_view_height = self._spin(cfg.view_height, 240, 4320)
        self.openxr_ipd = self._double(cfg.ipd_m * 1000.0, 30.0, 100.0, " mm", 1)
        form.addRow(self.openxr_enabled)
        form.addRow(self.openxr_manage)
        form.addRow("Remote driver host", self.openxr_remote_host)
        form.addRow("Remote driver port", self.openxr_remote_port)
        form.addRow("Capture shim host", self.openxr_capture_host)
        form.addRow("Capture shim port", self.openxr_capture_port)
        form.addRow("Pose send rate", self.openxr_rate)
        form.addRow("Stereo view size", self._pair(self.openxr_view_width, self.openxr_view_height))
        form.addRow("IPD", self.openxr_ipd)
        return widget

    def _gamepad_tab(self) -> QWidget:
        cfg = self.config.gamepad
        widget, form = self._tab()
        self.pad_enabled = QCheckBox("Emulate controllers from gamepads")
        self.pad_enabled.setChecked(cfg.enabled)
        self.pad_rumble = QCheckBox("Route haptics to gamepad rumble")
        self.pad_rumble.setChecked(cfg.rumble)
        self.pad_left = self._spin(cfg.left_index, 0, 15)
        self.pad_right = self._spin(cfg.right_index, 0, 15)
        self.pad_deadzone = self._double(cfg.deadzone, 0.0, 0.6, "", 2)
        self.pad_trigger = self._double(cfg.trigger_threshold, 0.05, 0.95, "", 2)
        self.pad_poll = self._spin(cfg.poll_hz, 20, 1000, " Hz")
        self.pad_hotplug = self._double(cfg.hotplug_interval_s, 0.25, 10.0, " s", 2)
        self.pad_invert_left = QCheckBox("Invert left stick Y")
        self.pad_invert_left.setChecked(cfg.invert_left_y)
        self.pad_invert_right = QCheckBox("Invert right stick Y")
        self.pad_invert_right.setChecked(cfg.invert_right_y)
        form.addRow(self.pad_enabled)
        form.addRow(self.pad_rumble)
        form.addRow("Left gamepad index", self.pad_left)
        form.addRow("Right gamepad index", self.pad_right)
        form.addRow("Stick deadzone", self.pad_deadzone)
        form.addRow("Trigger/grip click threshold", self.pad_trigger)
        form.addRow("Polling rate", self.pad_poll)
        form.addRow("Hotplug rescan interval", self.pad_hotplug)
        form.addRow("", self.pad_invert_left)
        form.addRow("", self.pad_invert_right)
        return widget

    def _driver_tab(self) -> QWidget:
        cfg = self.config.driver
        widget, form = self._tab()
        self.driver_host = QLineEdit(cfg.udp_host)
        self.driver_port = self._spin(cfg.udp_port, 1, 65535)
        self.driver_rate = self._spin(cfg.state_rate_hz, 30, 500, " Hz")
        self.driver_profile = QComboBox()
        self.driver_profile.addItems(["vive_controller", "knuckles", "oculus_touch"])
        self.driver_profile.setCurrentText(cfg.controller_profile)
        self.driver_hmd = QCheckBox("Create a virtual HMD")
        self.driver_hmd.setChecked(cfg.enable_virtual_hmd)
        self.driver_trackers = self._spin(cfg.tracker_count, 0, 16)
        form.addRow("Driver UDP host", self.driver_host)
        form.addRow("Driver UDP port", self.driver_port)
        form.addRow("State rate", self.driver_rate)
        form.addRow("Controller profile", self.driver_profile)
        form.addRow("", self.driver_hmd)
        form.addRow("Generic tracker count", self.driver_trackers)
        return widget

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    def result_config(self) -> AppConfig:
        cfg = self.config
        cfg.camera.source = self.camera_source.currentText()
        cfg.camera.index = self.camera_index.value()
        cfg.camera.width = self.camera_width.value()
        cfg.camera.height = self.camera_height.value()
        cfg.camera.fps = self.camera_fps.value()
        cfg.camera.horizontal_fov_deg = self.camera_fov.value()
        cfg.camera.dictionary = self.camera_dictionary.currentText()
        cfg.camera.marker_detection_width = self.camera_detection_width.value()
        cfg.camera.marker_refine_pose = self.camera_refine_pose.isChecked()
        cfg.camera.max_reprojection_error = self.camera_max_error.value()
        cfg.camera.calibration_file = self.camera_calibration.text().strip()
        cfg.camera.calibration_board_cols = self.camera_board_cols.value()
        cfg.camera.calibration_board_rows = self.camera_board_rows.value()
        cfg.camera.calibration_square_mm = self.camera_board_square.value()
        cfg.camera.calibration_min_views = self.camera_min_views.value()
        cfg.camera.position = _parse_vector(self.camera_position.text(), 3)
        cfg.camera.rotation = _parse_vector(self.camera_rotation.text(), 4)
        cfg.camera.backend = self.camera_backend.currentText()
        cfg.camera.fourcc = self.camera_fourcc.text().strip()
        cfg.camera.allow_mode_fallback = self.camera_allow_fallback.isChecked()
        cfg.camera.verify_mode = self.camera_verify_mode.isChecked()
        cfg.camera.autofocus = self.camera_autofocus.isChecked()
        cfg.camera.focus = self.camera_focus.value()
        cfg.camera.auto_exposure = self.camera_auto_exposure.isChecked()
        cfg.camera.exposure = self.camera_exposure.value()
        cfg.camera.gain = self.camera_gain.value()
        cfg.camera.auto_marker_control = self.camera_auto_marker.isChecked()
        cfg.camera.camera_aggressiveness = self.camera_aggressiveness.value()
        cfg.camera.camera_target_brightness = self.camera_target_brightness.value()
        cfg.camera.camera_max_gain = self.camera_max_gain.value()
        cfg.camera.camera_timestamp_source = self.camera_timestamp_source.currentText()
        cfg.camera.camera_timestamp_offset_ms = self.camera_timestamp_offset.value()
        cfg.camera.auto_white_balance = self.camera_auto_wb.isChecked()
        cfg.camera.white_balance = self.camera_white_balance.value()
        cfg.camera.mirror_preview = self.camera_mirror.isChecked()

        cfg.tracking.markers_enabled = self.track_markers.isChecked()
        cfg.tracking.hands_enabled = self.track_hands.isChecked()
        cfg.tracking.body_enabled = self.track_body.isChecked()
        cfg.tracking.hmd_source = self.track_hmd_source.currentText()
        cfg.tracking.left_source = self.track_left_source.currentText()
        cfg.tracking.right_source = self.track_right_source.currentText()
        cfg.tracking.hmd_marker_id = self.track_hmd_marker.value()
        cfg.tracking.left_marker_id = self.track_left_marker.value()
        cfg.tracking.right_marker_id = self.track_right_marker.value()
        for name in ("hmd", "left", "right"):
            board = getattr(cfg.tracking.boards, name)
            fields = self.board_widgets[name]
            board.width_m = fields["width"].value() / 1000.0
            board.height_m = fields["height"].value() / 1000.0
            board.central_marker_m = fields["central"].value() / 1000.0
            board.peripheral_marker_m = fields["peripheral"].value() / 1000.0
            board.peripheral_x_m = fields["peripheral_x"].value() / 1000.0
            board.peripheral_y_m = fields["peripheral_y"].value() / 1000.0
            board.object_position = _parse_vector(fields["object_position"].text(), 3)
            board.object_rotation = _parse_vector(fields["object_rotation"].text(), 4)
        cfg.tracking.smoothing_min_cutoff = self.track_smoothing.value()
        cfg.tracking.smoothing_beta = self.track_beta.value()
        cfg.tracking.orientation_cutoff = self.track_orientation.value()
        cfg.tracking.pose_prediction_ms = self.track_pose_prediction.value()
        cfg.tracking.marker_process_noise = self.track_marker_process.value()
        cfg.tracking.marker_angular_noise = self.track_marker_angular.value()
        cfg.tracking.marker_output_smoothing_hz = self.track_marker_output.value()
        cfg.tracking.marker_adaptive_noise = self.track_marker_adaptive.isChecked()
        cfg.tracking.marker_static_noise_scale = (
            self.track_marker_static_scale.value()
        )
        cfg.tracking.marker_hold_s = self.track_marker_hold.value()
        cfg.tracking.marker_lost_s = self.track_marker_lost.value()
        cfg.tracking.marker_reacquire_frames = self.track_marker_reacquire.value()
        cfg.tracking.marker_min_markers = self.track_marker_min_markers.value()
        cfg.tracking.marker_min_confidence = self.track_marker_confidence.value()
        cfg.tracking.body_min_visibility = self.track_body_visibility.value()
        cfg.tracking.landmark_cutoff = self.track_landmark.value()
        cfg.tracking.landmark_beta = self.track_landmark_beta.value()
        cfg.tracking.body_scale = self.track_body_scale.value()
        cfg.tracking.mediapipe_max_fps = self.track_mediapipe_fps.value()
        cfg.tracking.inference_width = self.track_inference_width.value()
        cfg.tracking.inference_height = self.track_inference_height.value()
        cfg.tracking.body_model = self.track_body_model.currentText()
        cfg.tracking.mediapipe_delegate = self.track_delegate.currentText()
        cfg.tracking.model_dir = self.track_model_dir.text().strip()
        cfg.tracking.tracker_roles = [
            role.strip() for role in self.track_roles.text().split(",") if role.strip()
        ]
        cfg.tracking.hand_trigger_pinch = self.track_pinch.value()
        cfg.tracking.hand_inputs = self.track_hand_inputs.isChecked()
        cfg.tracking.hand_swap = self.track_hand_swap.isChecked()

        cfg.phone.video_enabled = self.phone_video_enabled.isChecked()
        cfg.phone.stream_on_launch = self.phone_stream_launch.isChecked()
        cfg.phone.imu_port = self.phone_imu_port.value()
        cfg.phone.video_port = self.phone_video_port.value()
        cfg.phone.token = self.phone_token.text().strip()
        cfg.phone.layout = self.phone_layout.currentText()
        cfg.phone.width = self.phone_width.value()
        cfg.phone.height = self.phone_height.value()
        cfg.phone.fps = self.phone_fps.value()
        cfg.phone.codec = self.phone_codec.currentText()
        cfg.phone.encoder = self.phone_encoder.currentText()
        cfg.phone.bitrate_mbps = self.phone_bitrate.value()
        cfg.phone.keyframe_interval_s = self.phone_keyframe.value()
        cfg.phone.jpeg_quality = self.phone_quality.value()
        cfg.phone.capture_source = self.phone_source.currentText()
        cfg.phone.test_pattern = self.phone_pattern.currentText()
        cfg.phone.capture_monitor = self.phone_monitor.value()
        cfg.phone.camera_index = self.phone_camera_index.value()
        cfg.phone.lens.lens_separation_mm = self.lens_separation.value()
        cfg.phone.lens.eye_screen_distance_mm = self.lens_eye_distance.value()
        cfg.phone.lens.screen_width_mm = self.lens_screen_width.value()
        cfg.phone.lens.screen_height_mm = self.lens_screen_height.value()
        cfg.phone.lens.fov_deg = self.lens_fov.value()
        cfg.phone.lens.distortion_k1 = self.lens_k1.value()
        cfg.phone.lens.distortion_k2 = self.lens_k2.value()
        cfg.phone.lens.chromatic_aberration = self.lens_chromatic.value()
        cfg.phone.lens.image_scale = self.lens_scale.value()
        cfg.phone.lens.eye_offset_x = self.lens_offset_x.value()
        cfg.phone.lens.eye_offset_y = self.lens_offset_y.value()
        cfg.phone.lens.overscan = self.lens_overscan.value()

        cfg.gamepad.enabled = self.pad_enabled.isChecked()
        cfg.gamepad.rumble = self.pad_rumble.isChecked()
        cfg.gamepad.left_index = self.pad_left.value()
        cfg.gamepad.right_index = self.pad_right.value()
        cfg.gamepad.deadzone = self.pad_deadzone.value()
        cfg.gamepad.trigger_threshold = self.pad_trigger.value()
        cfg.gamepad.poll_hz = self.pad_poll.value()
        cfg.gamepad.hotplug_interval_s = self.pad_hotplug.value()
        cfg.gamepad.invert_left_y = self.pad_invert_left.isChecked()
        cfg.gamepad.invert_right_y = self.pad_invert_right.isChecked()

        cfg.driver.udp_host = self.driver_host.text().strip() or "127.0.0.1"
        cfg.driver.udp_port = self.driver_port.value()
        cfg.driver.state_rate_hz = self.driver_rate.value()
        cfg.driver.controller_profile = self.driver_profile.currentText()
        cfg.driver.enable_virtual_hmd = self.driver_hmd.isChecked()
        cfg.driver.tracker_count = self.driver_trackers.value()

        cfg.openxr.enabled = self.openxr_enabled.isChecked()
        cfg.openxr.manage_service = self.openxr_manage.isChecked()
        cfg.openxr.remote_host = self.openxr_remote_host.text().strip() or "127.0.0.1"
        cfg.openxr.remote_port = self.openxr_remote_port.value()
        cfg.openxr.capture_host = self.openxr_capture_host.text().strip() or "127.0.0.1"
        cfg.openxr.capture_port = self.openxr_capture_port.value()
        cfg.openxr.send_rate_hz = self.openxr_rate.value()
        cfg.openxr.view_width = self.openxr_view_width.value()
        cfg.openxr.view_height = self.openxr_view_height.value()
        cfg.openxr.ipd_m = self.openxr_ipd.value() / 1000.0
        return cfg
