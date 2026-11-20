"""Main window for the OpenVR Cardboard desktop app."""

from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import validate_config
from ..runtime import Runtime
from .settings_dialog import SettingsDialog

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
central one; the four small markers must keep their printed spacing. Body
tracking is MediaPipe-only and never claims marker IDs. Keep the whole board
and its white border visible. If markers look tilted or distances are wrong,
run Tools > Calibrate camera with the printed chessboard.

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


class MainWindow(QMainWindow):
    # Emitted from a worker thread when a slow runtime action finishes. Qt
    # delivers it on the GUI thread through a queued connection.
    action_finished = Signal(object, object, object)

    def __init__(self, runtime: Runtime, start_tracking: bool = True):
        super().__init__()
        self.runtime = runtime
        self._last_preview_sequence = -1
        self._last_notes: list[str] = []
        self._start_tracking = start_tracking
        self._busy = False
        self._closing = False

        self.setWindowTitle("OpenVR Cardboard")
        self.resize(1100, 720)
        self._build_ui()
        self._build_menu()

        self.action_finished.connect(self._on_action_finished)
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
            if self._start_tracking:
                self._run_async(
                    self.runtime.start_tracking,
                    "Tracking started",
                    "Starting tracking…",
                )
            else:
                self._refresh()
        else:
            self._run_async(
                lambda: self.runtime.start(start_tracking=self._start_tracking),
                "Tracking started" if self._start_tracking else "Runtime started",
                "Starting tracking…" if self._start_tracking else "Starting services…",
            )

    def _start_clicked(self) -> None:
        if self.runtime.running:
            self._run_async(
                self.runtime.start_tracking, "Tracking started", "Starting tracking…"
            )
        else:
            self._run_async(
                lambda: self.runtime.start(start_tracking=True),
                "Tracking started",
                "Starting tracking…",
            )

    def _stop_clicked(self) -> None:
        self._run_async(
            self.runtime.stop_tracking, "Tracking stopped", "Stopping tracking…"
        )

    def _run_async(self, action, success: str, busy_message: str) -> None:
        """Run a slow runtime action without blocking the Qt event loop.

        Opening the camera, initialising MediaPipe (which may download and
        load models) and starting the OpenXR service can take seconds. Running
        them on the GUI thread leaves the window painted but frozen, so they
        go to a worker and report back through ``action_finished``.
        """
        if self._busy:
            self.statusBar().showMessage("An operation is already running", 3000)
            return
        self._busy = True
        self._set_buttons_enabled(False)
        self.statusBar().showMessage(busy_message)

        def worker() -> None:
            notes: list[str] = []
            try:
                result = action()
                if isinstance(result, tuple) and len(result) == 3:
                    problems, message, notes = result
                elif isinstance(result, tuple) and len(result) == 2:
                    problems, message = result
                else:
                    problems, message = result or [], success
            except Exception as exc:
                LOGGER.exception("runtime action failed")
                problems, message = [f"{type(exc).__name__}: {exc}"], success
            if self._closing:
                return
            try:
                self.action_finished.emit(list(problems), message, list(notes))
            except RuntimeError:
                # The window was destroyed while the worker was running.
                pass

        threading.Thread(
            target=worker, name="openvrcardboard-ui-action", daemon=True
        ).start()

    def _on_action_finished(self, problems, message, notes) -> None:
        self._busy = False
        self._set_buttons_enabled(True)
        if notes:
            self.notes.appendPlainText("\n".join(str(note) for note in notes))
        self._report(list(problems or []), str(message))
        self._refresh()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for button in (self.start_button, self.stop_button, self.video_button):
            button.setEnabled(enabled)

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
        rms: float | None = None
        try:
            dialog = CalibrationDialog(
                config.camera,
                base_dir=self.runtime.base_dir,
                output=config.camera.calibration_file or None,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.calibration_file:
                rms = None
            else:
                config.camera.calibration_file = str(dialog.calibration_file)
                config.camera.calibration_board_cols = dialog.board_cols.value()
                config.camera.calibration_board_rows = dialog.board_rows.value()
                config.camera.calibration_square_mm = dialog.square_mm.value()
                rms = float(dialog.rms_error)
        except Exception as exc:
            QMessageBox.critical(self, "Calibrate camera", str(exc))
            rms = None

        if rms is None:
            if was_tracking:
                self._run_async(
                    self.runtime.start_tracking,
                    "Tracking started",
                    "Restarting tracking…",
                )
            else:
                self._refresh()
            return

        def action():
            notes: list[str] = []
            try:
                notes = self.runtime.apply_config(config)
                self.runtime.save()
            finally:
                if was_tracking and not self.runtime.tracking.running:
                    self.runtime.start_tracking()
            return [], f"Camera calibrated ({rms:.2f} px RMS)", notes

        self._run_async(action, "Camera calibrated", "Applying calibration…")

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
        def action():
            running = self.runtime.toggle_video()
            return [], (
                "Phone video started" if running else "Phone video stopped"
            )

        self._run_async(action, "Phone video updated", "Starting phone video…")

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
        def action():
            # apply_config restarts the runtime and, when tracking was active,
            # reopens the camera and reloads MediaPipe. Keep that off the GUI
            # thread so the window stays responsive while settings are applied.
            notes = self.runtime.apply_config(config)
            self.runtime.save()
            return [], "Settings applied", notes

        self._run_async(action, "Settings applied", "Applying settings…")

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
        self._closing = True
        self._timer.stop()
        # run_gui() stops the runtime after the event loop exits. Do it here
        # as well, off the GUI thread, so links close promptly without making
        # the window wait on a slow camera or startup action.
        threading.Thread(
            target=self.runtime.stop,
            name="openvrcardboard-ui-shutdown",
            daemon=True,
        ).start()
        event.accept()
