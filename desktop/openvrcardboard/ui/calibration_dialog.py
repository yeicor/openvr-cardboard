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

from ..calibration import calibrate_camera_views, detect_chessboard
from ..camera import CameraCapture, CameraError
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
    ):
        super().__init__(parent)
        if getattr(config, "source", "webcam") == "synthetic":
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
        self.rms_error = 0.0
        self._last_corners: Optional[np.ndarray] = None
        self._stable_frames = 0
        self._last_capture_time = 0.0
        self._capture: Optional[CameraCapture] = None
        self._cv2 = None
        self._shutdown_done = False

        self.setWindowTitle("Camera calibration")
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
        self.status = QLabel("Looking for the chessboard…")
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
        self._capture = CameraCapture(self.config)
        self._capture.start()

    # ------------------------------------------------------------------
    # Camera loop
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        if self._capture is None:
            return
        frame = self._capture.latest_copy()
        if frame is None:
            stats = self._capture.stats()
            if stats.last_error:
                self.status.setText(stats.last_error)
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
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:
                LOGGER.debug("camera stop failed during calibration", exc_info=True)
            self._capture = None
