"""Camera intrinsics and camera-to-play-space helpers.

Marker tracking needs a pinhole camera model. When no calibration file exists we
fall back to a sensible pinhole estimate based on a horizontal field of view, so
the app still works out of the box. A proper calibration file removes the
biggest source of marker pose error.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import CameraConfig, resolve_relative
from .mathutil import (
    OPTICAL_TO_VIEW,
    camera_optical_to_play,
    multiply,
    multiply_many,
    normalize,
    rotate,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.width = max(1, int(self.width))
        self.height = max(1, int(self.height))
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if self.distortion.size < 5:
            padded = np.zeros(5, dtype=np.float64)
            padded[: self.distortion.size] = self.distortion
            self.distortion = padded

    @classmethod
    def estimate(
        cls, width: int, height: int, horizontal_fov_deg: float = 60.0
    ) -> "CameraModel":
        width = max(1, int(width))
        height = max(1, int(height))
        fov = max(10.0, min(150.0, float(horizontal_fov_deg)))
        fx = (width * 0.5) / math.tan(math.radians(fov) * 0.5)
        return cls(width, height, fx, fx, width * 0.5, height * 0.5)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled(self, width: int, height: int) -> "CameraModel":
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height:
            return self
        sx = width / float(self.width)
        sy = height / float(self.height)
        return CameraModel(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            distortion=self.distortion.copy(),
        )

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """Unproject a pixel at a given optical depth (metres, +Z forward)."""
        depth = max(1e-4, float(depth))
        x = (float(u) - self.cx) * depth / self.fx
        y = (float(v) - self.cy) * depth / self.fy
        return np.array([x, y, depth], dtype=np.float64)

    def project(self, point) -> tuple[float, float]:
        point = np.asarray(point, dtype=np.float64).reshape(3)
        z = float(point[2])
        if abs(z) < 1e-9:
            return (float("nan"), float("nan"))
        u = self.fx * point[0] / z + self.cx
        v = self.fy * point[1] / z + self.cy
        return (float(u), float(v))

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "camera_matrix": self.matrix.tolist(),
            "distortion_coefficients": self.distortion.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict) -> "CameraModel":
        width = int(values.get("width", 0))
        height = int(values.get("height", 0))
        matrix = values.get("camera_matrix")
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError("camera_matrix must be 3x3")
            fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
            cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        else:
            fx = float(values["fx"])
            fy = float(values.get("fy", fx))
            cx = float(values["cx"])
            cy = float(values["cy"])
        distortion = values.get("distortion_coefficients", values.get("distortion", []))
        return cls(width, height, fx, fy, cx, cy, np.asarray(distortion, dtype=np.float64))

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path, width: int, height: int) -> "CameraModel":
        values = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("calibration root is not an object")
        model = cls.from_dict(values)
        if model.width <= 0 or model.height <= 0:
            model = cls(width, height, model.fx, model.fy, model.cx, model.cy, model.distortion)
        return model.scaled(width, height)


def camera_model_for_config(
    config: CameraConfig, base_dir: Path
) -> tuple[CameraModel, list[str]]:
    """Load ``config.calibration_file`` or estimate intrinsics."""
    warnings: list[str] = []
    if getattr(config, "source", "webcam") == "synthetic":
        # The synthetic scene uses exactly the estimated pinhole model, so a
        # real camera calibration would only add error.
        return CameraModel.estimate(
            config.width, config.height, config.horizontal_fov_deg
        ), warnings
    path = resolve_relative(config.calibration_file, base_dir)
    if path.exists():
        try:
            model = CameraModel.load(path, config.width, config.height)
            LOGGER.info("loaded camera calibration from %s", path)
            return model, warnings
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"could not load camera calibration {path}: {exc}")
            LOGGER.warning("could not load camera calibration %s: %s", path, exc)
    else:
        warnings.append(
            f"camera calibration {path} not found; using estimated intrinsics"
        )
    model = CameraModel.estimate(
        config.width, config.height, config.horizontal_fov_deg
    )
    return model, warnings


def camera_pose(config: CameraConfig) -> tuple[np.ndarray, np.ndarray]:
    from .config import _float_list

    position = np.asarray(
        _float_list(config.position, 3, [0.0, 1.5, -1.5]), dtype=np.float64
    )
    rotation = normalize(
        _float_list(config.rotation, 4, [0.0, 1.0, 0.0, 0.0])
    )
    return position, rotation


def checkerboard_object_points(
    cols: int, rows: int, square_mm: float
) -> np.ndarray:
    """3D coordinates of a chessboard's inner corners, in millimetres.

    The board lies on the z=0 plane with the first corner at the origin, so
    calibration is independent of where the board is in the camera view.
    """
    cols = max(2, int(cols))
    rows = max(2, int(rows))
    square_mm = max(0.1, float(square_mm))
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].reshape(2, -1).T
    points[:, :2] = grid * square_mm
    return points


def detect_chessboard(
    gray: np.ndarray, cols: int, rows: int
) -> Optional[np.ndarray]:
    """Find chessboard inner corners, with sub-pixel refinement when possible.

    Returns an ``(N, 1, 2)`` float32 array or ``None``. The symmetric pattern
    detector is preferred because it is much less sensitive to motion blur and
    uneven lighting than the legacy corner detector.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("camera calibration needs the 'opencv-python' package") from exc
    pattern = (max(2, int(cols)), max(2, int(rows)))
    image = np.asarray(gray)
    if image.ndim != 2 or image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    found = False
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            found, corners = cv2.findChessboardCornersSB(
                image,
                pattern,
                flags=(
                    cv2.CALIB_CB_EXHAUSTIVE
                    | cv2.CALIB_CB_ACCURACY
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
                ),
            )
        except Exception:
            found = False
    if not found:
        found, corners = cv2.findChessboardCorners(
            image,
            pattern,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK
            ),
        )
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                40,
                0.001,
            )
            corners = cv2.cornerSubPix(
                image,
                np.asarray(corners, dtype=np.float32),
                (7, 7),
                (-1, -1),
                criteria,
            )
    if not found or corners is None:
        return None
    corners = np.asarray(corners, dtype=np.float32)
    expected = pattern[0] * pattern[1]
    if corners.size != expected * 2:
        return None
    return corners.reshape(-1, 1, 2)


def calibrate_camera_views(
    views: list[np.ndarray],
    image_size: tuple[int, int],
    cols: int,
    rows: int,
    square_mm: float,
) -> tuple[CameraModel, float, np.ndarray]:
    """Calibrate intrinsics from chessboard corner views.

    Returns ``(model, rms_error_px, per_view_errors)``. At least three views
    are required by OpenCV; the wizard normally asks for twelve.
    """
    if len(views) < 3:
        raise ValueError("at least three chessboard views are required")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("camera calibration needs the 'opencv-python' package") from exc
    width, height = (max(1, int(value)) for value in image_size)
    object_points = checkerboard_object_points(cols, rows, square_mm)
    object_views = [object_points.copy() for _ in views]
    image_views = [
        np.asarray(view, dtype=np.float32).reshape(-1, 1, 2) for view in views
    ]
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_views,
        image_views,
        (width, height),
        None,
        None,
        flags=cv2.CALIB_FIX_K3 if hasattr(cv2, "CALIB_FIX_K3") else 0,
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    errors = []
    for index, view in enumerate(image_views):
        projected, _ = cv2.projectPoints(
            object_views[index], rvecs[index], tvecs[index], matrix, distortion
        )
        difference = projected.reshape(-1, 2) - view.reshape(-1, 2)
        errors.append(
            float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))
        )
    model = CameraModel(
        width=width,
        height=height,
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        distortion=distortion,
    )
    return model, float(rms), np.asarray(errors, dtype=np.float64)


def optical_point_to_play(
    camera_position, camera_rotation, optical_point
) -> np.ndarray:
    position, _ = camera_optical_to_play(
        camera_position,
        camera_rotation,
        optical_point,
        (0.0, 0.0, 0.0, 1.0),
    )
    return position


def optical_pose_to_play(
    camera_position, camera_rotation, optical_position, optical_rotation
):
    return camera_optical_to_play(
        camera_position, camera_rotation, optical_position, optical_rotation
    )


def local_point_from_pixel(
    model: CameraModel,
    camera_position,
    camera_rotation,
    u: float,
    v: float,
    depth: float,
) -> np.ndarray:
    optical = model.unproject(u, v, depth)
    return optical_point_to_play(camera_position, camera_rotation, optical)


def direction_from_pixel(
    model: CameraModel, camera_rotation, u: float, v: float
) -> np.ndarray:
    """Unit direction in play space for a pixel ray."""
    optical = np.array(
        [(u - model.cx) / model.fx, (v - model.cy) / model.fy, 1.0],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(optical))
    if length < 1e-12:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    optical /= length
    rotation = multiply_many(normalize(camera_rotation), OPTICAL_TO_VIEW)
    return rotate(rotation, optical)
