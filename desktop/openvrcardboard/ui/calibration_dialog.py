"""Guided chessboard camera-calibration dialog.

The wizard opens the configured webcam, detects a printed chessboard, collects
views from different angles and solves the pinhole intrinsics with OpenCV. It
is deliberately available both from the settings and from *Tools > Calibrate
camera* so recalibrating after moving or refocusing the camera is a two-click
operation.

All calibration maths lives in :mod:`openvrcardboard.calibration`; this module
only adds the Qt workflow and camera loop.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..calibration import (
    CameraCalibrationStore,
    calibrate_camera_views,
    detect_chessboard,
)
from ..camera import CameraCapture
from ..config import CameraConfig

LOGGER = logging.getLogger(__name__)


def _qimage(array: np.ndarray) -> QImage:
    array = np.asarray(array)
    if array.ndim == 3:
        rgb = np.ascontiguousarray(array[:, :, ::-1])
        height, width = rgb.shape[:2]
        return QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
    height, width = array.shape[:2]
    return QImage(
        np.ascontiguousarray(array).data,
        width,
        height,
        width,
        QImage.Format.Format_Grayscale8,
    ).copy()


class CalibrationDialog(QDialog):
    """Collects chessboard views and writes a :class:`CameraModel` JSON file."""

    def __init__(
        self,
        config: CameraConfig,
        base_dir: Optional[Path] = None,
        output: Optional[str] = None,
        parent: Optional[QWidget] = None,
        *,
        source=None,
        store: Optional[CameraCalibrationStore] = None,
        camera_key: str = "",
        camera_name: str = "",
        waiting_message: str = "",
    ):
        super().__init__(parent)
        if source is None and getattr(config, "source", "webcam") == "synthetic":
            raise RuntimeError(
                "the synthetic camera has known intrinsics and cannot be "
                "calibrated; switch the camera source to 'webcam' first"
            )
        self.config = copy.deepcopy(config)
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        default_name = output or "camera-calibration.json"
        path = Path(default_name).expanduser()
        self.output_path = path if path.is_absolute() else self.base_dir / path
        self.views: list[np.ndarray] = []
        self.calibration_file: Optional[Path] = None
        self.calibration_key = ""
        self.rms_error = 0.0
        self._store = store
        self._camera_key = str(camera_key or "").strip()
        self._camera_name = str(camera_name or "").strip()
        self._waiting_message = str(waiting_message or "").strip()
        self._source = source
        self._owns_source = source is None
        self._last_corners: Optional[np.ndarray] = None
        self._stable_frames = 0
        self._last_capture_time = 0.0
        self._capture = None
        self._frame_size = (
            int(self.config.width),
            int(self.config.height),
        )
        self._cv2 = None
        self._shutdown_done = False

        title = "Camera calibration"
        if self._camera_name:
            title = f"{title} — {self._camera_name}"
        self.setWindowTitle(title)
        self.resize(860, 620)
        outer = QVBoxLayout(self)

        instructions = QLabel(
            "Print the chessboard and show every square to the camera. Move it "
            "slowly through the whole view: centre, corners, close up and at an "
            "angle. Keep it flat; views are captured automatically when the "
            "board is still and has moved since the previous view."
        )
        instructions.setWordWrap(True)
        outer.addWidget(instructions)

        self.preview = QLabel("Opening camera…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet(
            "background: #11151b; border: 1px solid #30363d; color: #8b949e;"
        )
        outer.addWidget(self.preview, 1)

        controls = QHBoxLayout()
        self.board_cols = QSpinBox()
        self.board_cols.setRange(3, 20)
        self.board_cols.setValue(int(self.config.calibration_board_cols))
        self.board_cols.setSuffix(" cols")
        self.board_cols.setToolTip("Inner corners, not squares.")
        self.board_rows = QSpinBox()
        self.board_rows.setRange(3, 20)
        self.board_rows.setValue(int(self.config.calibration_board_rows))
        self.board_rows.setSuffix(" rows")
        self.board_rows.setToolTip("Inner corners, not squares.")
        self.square_mm = QDoubleSpinBox()
        self.square_mm.setRange(5.0, 200.0)
        self.square_mm.setDecimals(1)
        self.square_mm.setSuffix(" mm")
        self.square_mm.setValue(float(self.config.calibration_square_mm))
        self.square_mm.setToolTip("Measured square edge length on the printed board.")
        self.min_views = QSpinBox()
        self.min_views.setRange(3, 60)
        self.min_views.setValue(int(self.config.calibration_min_views))
        controls.addWidget(QLabel("Board"))
        controls.addWidget(self.board_cols)
        controls.addWidget(self.board_rows)
        controls.addWidget(QLabel("Square"))
        controls.addWidget(self.square_mm)
        controls.addWidget(QLabel("Minimum views"))
        controls.addWidget(self.min_views)
        outer.addLayout(controls)

        status_row = QHBoxLayout()
        self.status = QLabel(
            self._waiting_message or "Looking for the chessboard…"
        )
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, max(1, int(self.config.calibration_min_views)))
        self.progress.setValue(0)
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.progress)
        outer.addLayout(status_row)

        buttons = QHBoxLayout()
        self.auto_capture = QCheckBox("Capture views automatically")
        self.auto_capture.setChecked(True)
        self.capture_button = QPushButton("Capture view")
        self.capture_button.clicked.connect(self._capture_clicked)
        self.reset_button = QPushButton("Reset views")
        self.reset_button.clicked.connect(self._reset_clicked)
        self.finish_button = QPushButton("Calibrate and save…")
        self.finish_button.setEnabled(False)
        self.finish_button.clicked.connect(self._finish_clicked)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.auto_capture)
        buttons.addWidget(self.capture_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        buttons.addWidget(self.finish_button)
        buttons.addWidget(self.cancel_button)
        outer.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(80)

        try:
            self._cv2 = __import__("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "camera calibration needs the 'opencv-python' package"
            ) from exc
        if self._source is None:
            self._capture = CameraCapture(self.config)
        else:
            self._capture = self._source
        self._capture.start()

    # ------------------------------------------------------------------
    # Camera loop
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        if self._capture is None:
            return
        frame = self._capture.latest_copy()
        if frame is None:
            try:
                stats = self._capture.stats()
            except Exception:
                stats = None
            if stats is not None and stats.last_error:
                self.status.setText(stats.last_error)
            elif self._waiting_message:
                self.status.setText(self._waiting_message)
            return
        image = frame.image
        cv2 = self._cv2
        self._frame_size = (int(frame.width), int(frame.height))
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        corners = detect_chessboard(
            gray, self.board_cols.value(), self.board_rows.value()
        )
        preview = image.copy()
        if corners is not None:
            cv2.drawChessboardCorners(
                preview,
                (self.board_cols.value(), self.board_rows.value()),
                corners,
                True,
            )
            self._consider_capture(corners)
        else:
            self._stable_frames = 0
        self._show(preview)

    def _show(self, image: np.ndarray) -> None:
        pixmap = QPixmap.fromImage(_qimage(image)).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def _consider_capture(self, corners: np.ndarray) -> None:
        import time

        previous = self._last_corners
        if previous is not None and previous.shape == corners.shape:
            movement = float(
                np.mean(np.linalg.norm(previous.reshape(-1, 2) - corners.reshape(-1, 2), axis=1))
            )
        else:
            movement = float("inf")
        self._last_corners = corners.copy()
        self._stable_frames = self._stable_frames + 1 if movement < 3.0 else 0
        if not self.auto_capture.isChecked() or self._stable_frames < 3:
            return
        now = time.monotonic()
        if now - self._last_capture_time < 0.45:
            return
        if self.views:
            last = self.views[-1]
            if last.shape == corners.shape:
                moved = float(
                    np.mean(
                        np.linalg.norm(
                            last.reshape(-1, 2) - corners.reshape(-1, 2), axis=1
                        )
                    )
                )
                if moved < 22.0:
                    return
        self._add_view(corners)

    def _add_view(self, corners: np.ndarray) -> None:
        import time

        self.views.append(corners.copy())
        self._last_capture_time = time.monotonic()
        self._stable_frames = 0
        self._update_progress()

    def _update_progress(self) -> None:
        count = len(self.views)
        target = max(3, self.min_views.value())
        self.progress.setRange(0, target)
        self.progress.setValue(min(count, target))
        if count < target:
            remaining = target - count
            self.status.setText(
                f"{count} view(s) captured · {remaining} more for a solid "
                "calibration. Cover corners and tilt the board."
            )
        else:
            self.status.setText(
                f"{count} views captured. Move the board somewhere new or "
                "press Calibrate and save."
            )
        self.finish_button.setEnabled(count >= 3)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def _capture_clicked(self) -> None:
        if self._last_corners is None:
            self.status.setText("No chessboard detected in the current frame.")
            return
        self._add_view(self._last_corners)

    def _reset_clicked(self) -> None:
        self.views.clear()
        self._last_corners = None
        self._stable_frames = 0
        self._update_progress()

    def _finish_clicked(self) -> None:
        if len(self.views) < 3:
            QMessageBox.warning(
                self, "Camera calibration", "Capture at least three views first."
            )
            return
        if len(self.views) < self.min_views.value():
            answer = QMessageBox.question(
                self,
                "Camera calibration",
                f"Only {len(self.views)} views are captured. A calibration "
                f"normally needs {self.min_views.value()}. Continue anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        width, height = getattr(self, "_frame_size", (self.config.width, self.config.height))
        try:
            model, rms, errors = calibrate_camera_views(
                self.views,
                (width, height),
                self.board_cols.value(),
                self.board_rows.value(),
                self.square_mm.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Camera calibration", str(exc))
            return
        suffix = ""
        if errors.size:
            suffix = f"\nPer-view error: mean {errors.mean():.2f} px, worst {errors.max():.2f} px."
        quality = (
            "excellent"
            if rms <= 0.5
            else "good"
            if rms <= 1.0
            else "acceptable"
            if rms <= 2.0
            else "poor"
        )
        if self._store is not None and self._camera_key:
            try:
                target = self._store.save(
                    self._camera_key, model, self._camera_name
                )
            except (OSError, ValueError) as exc:
                QMessageBox.critical(
                    self, "Camera calibration", f"Could not save: {exc}"
                )
                return
            self.calibration_file = target
            self.calibration_key = self._camera_key
            self.rms_error = float(rms)
            label = self._camera_name or self._camera_key
            QMessageBox.information(
                self,
                "Camera calibration",
                f"Calibration for {label} saved to {target}.\n\n"
                f"RMS reprojection error: {rms:.2f} px ({quality}).{suffix}",
            )
            self.accept()
            return
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Save camera calibration",
            str(self.output_path),
            "JSON (*.json);;All files (*)",
        )
        if not path_text:
            return
        target = Path(path_text).expanduser()
        try:
            model.save(target)
        except OSError as exc:
            QMessageBox.critical(self, "Camera calibration", f"Could not save: {exc}")
            return
        self.calibration_file = target
        self.rms_error = float(rms)
        QMessageBox.information(
            self,
            "Camera calibration",
            f"Calibration saved to {target}.\n\n"
            f"RMS reprojection error: {rms:.2f} px ({quality}).{suffix}",
        )
        self.accept()

    def reject(self) -> None:  # noqa: D401 - Qt override
        self._shutdown()
        super().reject()

    def accept(self) -> None:  # noqa: D401 - Qt override
        self._shutdown()
        super().accept()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._shutdown()
        event.accept()

    def _shutdown(self) -> None:
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        self._timer.stop()
        if self._capture is not None and self._owns_source:
            try:
                self._capture.stop()
            except Exception:
                LOGGER.debug("camera stop failed during calibration", exc_info=True)
        self._capture = None


class PhoneCameraCalibrationSource:
    """Adapter exposing the runtime's phone receiver as a calibration source.

    The receiver is started with the runtime and may keep streaming while
    tracking is stopped for calibration, so the adapter deliberately does not
    own or stop it.
    """

    def __init__(self, receiver):
        self.receiver = receiver

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def latest_copy(self):
        return self.receiver.latest_copy()

    def stats(self):
        return self.receiver.stats()


def _as_bool(value) -> bool:
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return False


def calibrate_phone_camera(runtime, parent: Optional[QWidget] = None) -> Optional[float]:
    """Run the chessboard wizard against the connected phone camera.

    Returns the RMS reprojection error when a calibration was saved, or
    ``None`` when the user cancelled or the phone was not ready. The runtime's
    shared calibration store is updated immediately, so the tracking worker
    picks up the new intrinsics without rebuilding the runtime.
    """
    receiver = getattr(runtime, "phone_camera", None)
    store = getattr(runtime, "phone_calibrations", None)
    if receiver is None or store is None:
        QMessageBox.information(
            parent,
            "Calibrate phone camera",
            "Phone camera support is not available in this build.",
        )
        return None
    if not bool(getattr(runtime.config.phone, "camera_enabled", True)):
        QMessageBox.information(
            parent,
            "Calibrate phone camera",
            "Enable the phone camera in Settings → Phone video, apply the "
            "settings, then reconnect the phone in Camera mode.",
        )
        return None
    if not _as_bool(getattr(receiver, "running", False)):
        QMessageBox.information(
            parent,
            "Calibrate phone camera",
            "The phone camera receiver is not running. Enable phone camera "
            "mode and restart tracking first.",
        )
        return None
    if not _as_bool(getattr(receiver, "connected", False)):
        QMessageBox.information(
            parent,
            "Calibrate phone camera",
            "Connect the Android app in Camera mode before calibrating.",
        )
        return None
    get_key = getattr(receiver, "camera_key", None)
    camera_key = str(get_key() if callable(get_key) else get_key or "").strip()
    if not camera_key:
        QMessageBox.information(
            parent,
            "Calibrate phone camera",
            "The connected app did not report its camera ID. Install the "
            "current Android release and reconnect in Camera mode. Without an "
            "ID the desktop keeps using the reported field of view.",
        )
        return None
    get_active = getattr(receiver, "active", None)
    if callable(get_active):
        try:
            fresh = bool(get_active(2.0))
        except TypeError:
            fresh = _as_bool(get_active)
    else:
        fresh = _as_bool(get_active)
    if not fresh:
        QMessageBox.information(
            parent,
            "Calibrate phone camera",
            "The phone is connected but not sending frames yet. Wait for the "
            "camera preview to update, then try again.",
        )
        return None
    get_name = getattr(receiver, "camera_name", None)
    camera_name = str(get_name() if callable(get_name) else get_name or "").strip()
    if not camera_name:
        try:
            info = receiver.status().get("info", {})
            camera_name = str(
                info.get("camera_name") or info.get("name") or ""
            ).strip()
        except Exception:
            camera_name = ""
    camera_name = camera_name or camera_key

    was_tracking = _as_bool(
        getattr(getattr(runtime, "tracking", None), "running", False)
    )
    if was_tracking:
        runtime.stop_tracking()
    try:
        dialog = CalibrationDialog(
            runtime.config.camera,
            base_dir=getattr(runtime, "base_dir", Path.cwd()),
            parent=parent,
            source=PhoneCameraCalibrationSource(receiver),
            store=store,
            camera_key=camera_key,
            camera_name=camera_name,
            waiting_message="Waiting for the phone camera stream…",
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return float(dialog.rms_error)
    finally:
        if was_tracking and not _as_bool(runtime.tracking.running):
            runtime.start_tracking()
